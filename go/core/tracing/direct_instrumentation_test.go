// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package tracing

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// withRealtime forces realtime (start + end) export for the test.
func withRealtime(t *testing.T) {
	t.Helper()
	prev := realtimeTelemetryActive
	realtimeTelemetryActive = true
	t.Cleanup(func() { realtimeTelemetryActive = prev })
}

// useChain sets the chain to [d, base] (or [base] when d is nil) for the test.
func useChain(t *testing.T, d *DirectTelemetryInstrumentation, base Instrumentation) {
	t.Helper()
	t.Cleanup(ResetInstrumentation)
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	configured = base
	// Checked explicitly: a nil *DirectTelemetryInstrumentation stored in the
	// interface would be a non-nil Instrumentation.
	direct = nil
	if d != nil {
		direct = d
	}
	rebuildChainLocked()
}

// slowStartClient delays in-progress saves, so without ordering they would
// land after the completed save for the same span.
type slowStartClient struct {
	mu    sync.Mutex
	saves []*SpanData
}

func (c *slowStartClient) Save(_ context.Context, td *Data) error {
	for _, s := range td.Spans {
		if s.EndTime == 0 {
			time.Sleep(20 * time.Millisecond)
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range td.Spans {
		c.saves = append(c.saves, s)
	}
	return nil
}

func TestDirect_EndSaveLandsAfterStartSave(t *testing.T) {
	withRealtime(t)
	client := &slowStartClient{}
	useChain(t, NewDirectTelemetryInstrumentation(client), &fakeInstrumentation{})

	_, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) { return "out", nil })
	if err != nil {
		t.Fatal(err)
	}

	client.mu.Lock()
	defer client.mu.Unlock()
	if len(client.saves) != 2 {
		t.Fatalf("got %d saves, want 2 (start and end)", len(client.saves))
	}
	if client.saves[0].EndTime != 0 || client.saves[1].EndTime == 0 {
		t.Errorf("save order = [end=%v, end=%v], want the in-progress save first",
			client.saves[0].EndTime, client.saves[1].EndTime)
	}
}

func TestTestOnlyTelemetryClient_KeepsCompletedSpan(t *testing.T) {
	c := NewTestOnlyTelemetryClient()
	done := &SpanData{SpanID: "s", EndTime: 2, Attributes: map[string]any{"genkit:state": "success"}}
	stale := &SpanData{SpanID: "s", Attributes: map[string]any{}}
	for _, s := range []*SpanData{done, stale} {
		if err := c.Save(context.Background(), &Data{TraceID: "t", Spans: map[string]*SpanData{"s": s}}); err != nil {
			t.Fatal(err)
		}
	}
	if got := c.Traces["t"].Spans["s"]; got != done {
		t.Errorf("stored span = %+v, want the completed one", got)
	}
}

func TestDirect_NilClientMintsIDsWithoutExporting(t *testing.T) {
	withRealtime(t)
	useChain(t, NewDirectTelemetryInstrumentation(nil), &fakeInstrumentation{})

	var got TraceInfo
	_, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			got = SpanTraceInfo(ctx)
			return "", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.TraceID) != 32 || len(got.SpanID) != 16 {
		t.Errorf("TraceInfo = %+v, want Direct-minted ids", got)
	}
}

// TestDirect_TwoInstancesTrackParentageSeparately covers the dev chain
// [Direct(env), Direct(user)]: each client must receive a self-consistent
// trace whose child points at a root that client actually has.
func TestDirect_TwoInstancesTrackParentageSeparately(t *testing.T) {
	outerClient := NewTestOnlyTelemetryClient()
	innerClient := NewTestOnlyTelemetryClient()
	useChain(t, NewDirectTelemetryInstrumentation(outerClient), NewDirectTelemetryInstrumentation(innerClient))

	_, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			return RunInNewSpan(ctx, &SpanMetadata{Name: "child"}, "in",
				func(ctx context.Context, _ string) (string, error) { return "", nil })
		})
	if err != nil {
		t.Fatal(err)
	}

	for name, c := range map[string]*TestOnlyTelemetryClient{"outer": outerClient, "inner": innerClient} {
		if len(c.Traces) != 1 {
			t.Fatalf("%s: got %d traces, want 1", name, len(c.Traces))
		}
		for _, td := range c.Traces {
			if td.DisplayName != "root" {
				t.Errorf("%s: trace DisplayName = %q, want root", name, td.DisplayName)
			}
			spans := map[string]*SpanData{}
			for _, s := range td.Spans {
				spans[s.DisplayName] = s
			}
			root, child := spans["root"], spans["child"]
			if root == nil || child == nil {
				t.Fatalf("%s: missing spans: root=%v child=%v", name, root, child)
			}
			if root.ParentSpanID != "" {
				t.Errorf("%s: root has parent %q", name, root.ParentSpanID)
			}
			if child.ParentSpanID != root.SpanID {
				t.Errorf("%s: child parent = %q, want root %q", name, child.ParentSpanID, root.SpanID)
			}
		}
	}
}

func TestDirect_RootSetsTraceEnvelope(t *testing.T) {
	client := NewTestOnlyTelemetryClient()
	useChain(t, NewDirectTelemetryInstrumentation(client), &fakeInstrumentation{})

	_, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			return RunInNewSpan(ctx, &SpanMetadata{Name: "child"}, "in",
				func(ctx context.Context, _ string) (string, error) { return "", nil })
		})
	if err != nil {
		t.Fatal(err)
	}
	if len(client.Traces) != 1 {
		t.Fatalf("got %d traces, want 1", len(client.Traces))
	}
	for _, td := range client.Traces {
		var root *SpanData
		for _, s := range td.Spans {
			if s.ParentSpanID == "" {
				root = s
			}
		}
		if root == nil {
			t.Fatal("no root span")
		}
		if td.DisplayName != "root" || td.StartTime != root.StartTime || td.EndTime != root.EndTime || td.EndTime == 0 {
			t.Errorf("trace envelope = {%q %v %v}, want root's {%q %v %v}",
				td.DisplayName, td.StartTime, td.EndTime, root.DisplayName, root.StartTime, root.EndTime)
		}
	}
}

func TestDirect_StartExportCarriesOnlyStartKnownAttributes(t *testing.T) {
	withRealtime(t)
	client := &recordingClient{}
	useChain(t, NewDirectTelemetryInstrumentation(client), &fakeInstrumentation{})

	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "act", Type: "action", Init: map[string]string{"k": "v"}}, "my-input",
		func(ctx context.Context, _ string) (string, error) { return "out", nil })
	if err != nil {
		t.Fatal(err)
	}
	var start *SpanData
	for _, td := range client.all() {
		for _, s := range td.Spans {
			if s.EndTime == 0 {
				start = s
			}
		}
	}
	if start == nil {
		t.Fatal("no in-progress save")
	}
	for _, k := range []string{"genkit:state", "genkit:output"} {
		if _, ok := start.Attributes[k]; ok {
			t.Errorf("in-progress span carries %q", k)
		}
	}
	if got := start.Attributes["genkit:input"]; got != `"my-input"` {
		t.Errorf("genkit:input = %v, want %q", got, `"my-input"`)
	}
	if got := start.Attributes["genkit:init"]; got != `{"k":"v"}` {
		t.Errorf("genkit:init = %v, want %q", got, `{"k":"v"}`)
	}
}

// namedErr is a non-pointer named error type, the case where OTel's
// exception.type differs from %T.
type namedErr struct{}

func (namedErr) Error() string { return "named failure" }

func TestErrorTypeName(t *testing.T) {
	if got, want := errorTypeName(namedErr{}), "github.com/firebase/genkit/go/core/tracing.namedErr"; got != want {
		t.Errorf("named type = %q, want %q", got, want)
	}
	if got, want := errorTypeName(errors.New("x")), "*errors.errorString"; got != want {
		t.Errorf("unnamed type = %q, want %q", got, want)
	}
}

// TestDirect_MatchesOTelExport runs the same spans through the OTel provider
// (exported by telemetryServerExporter's convertSpan) and through Direct, and
// requires the same SpanData apart from ids and timestamps.
func TestDirect_MatchesOTelExport(t *testing.T) {
	cases := []struct {
		name   string
		labels map[string]string
		err    error
	}{
		{name: "success"},
		{name: "label cannot clobber genkit keys", labels: map[string]string{"genkit:state": "bogus", "custom": "v"}},
		{name: "error", err: namedErr{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			run := func() {
				_, _ = RunInNewSpan(context.Background(),
					&SpanMetadata{
						Name: "flow", Type: "action", Subtype: "flow",
						TelemetryLabels: tc.labels, Metadata: map[string]string{"k": "v"},
					},
					"in",
					func(ctx context.Context, _ string) (string, error) {
						SetCustomMetadataAttributes(ctx, map[string]string{"mid": "run"})
						_, err := RunInNewSpan(ctx, &SpanMetadata{Name: "step", Type: "flowStep"}, 1,
							func(ctx context.Context, _ int) (int, error) { return 2, tc.err })
						return "out", err
					})
			}

			prevTP := otel.GetTracerProvider()
			t.Cleanup(func() { otel.SetTracerProvider(prevTP) })
			exp := tracetest.NewInMemoryExporter()
			otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)))
			useChain(t, nil, &OTelInstrumentation{})
			run()
			otelSpans := map[string]*SpanData{}
			for _, s := range exp.GetSpans().Snapshots() {
				sd := convertSpan(s)
				otelSpans[sd.DisplayName] = sd
			}

			client := NewTestOnlyTelemetryClient()
			useChain(t, NewDirectTelemetryInstrumentation(client), &fakeInstrumentation{})
			run()
			directSpans := map[string]*SpanData{}
			for _, s := range client.Spans() {
				directSpans[s.DisplayName] = s
			}

			ignore := cmp.Options{
				cmpopts.IgnoreFields(SpanData{}, "SpanID", "TraceID", "ParentSpanID", "StartTime", "EndTime"),
				cmpopts.IgnoreFields(TimeEvent{}, "Time"),
				cmpopts.EquateEmpty(),
			}
			for _, name := range []string{"flow", "step"} {
				if otelSpans[name] == nil || directSpans[name] == nil {
					t.Fatalf("span %q: otel=%v direct=%v", name, otelSpans[name], directSpans[name])
				}
				if diff := cmp.Diff(otelSpans[name], directSpans[name], ignore); diff != "" {
					t.Errorf("span %q differs (-otel +direct):\n%s", name, diff)
				}
				if got := directSpans[name].Attributes["genkit:state"]; got == "bogus" {
					t.Errorf("span %q: a label clobbered genkit:state", name)
				}
			}
		})
	}
}
