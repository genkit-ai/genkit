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
	"testing"
)

// fakeInstrumentation is a minimal provider that records calls and mints
// deterministic ids. It mirrors the JS instrumentation test's fake.
type fakeInstrumentation struct {
	traceID string
	spanID  string
	spans   []*SpanInfo
	meta    []map[string]string
}

type fakeSpan struct {
	inst *fakeInstrumentation
}

func (s *fakeSpan) TraceInfo() TraceInfo {
	return TraceInfo{TraceID: s.inst.traceID, SpanID: s.inst.spanID}
}

func (s *fakeSpan) SetMetadata(md map[string]string) {
	s.inst.meta = append(s.inst.meta, md)
}

func (f *fakeInstrumentation) RunInNewSpan(ctx context.Context, info *SpanInfo, next NextFunc) (any, error) {
	f.spans = append(f.spans, info)
	return next(ctx, &fakeSpan{inst: f})
}

func TestConfigureInstrumentation_RoutesThroughProvider(t *testing.T) {
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()

	fake := &fakeInstrumentation{traceID: "aaaa", spanID: "bbbb"}
	ConfigureInstrumentation(fake)

	out, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "root", Type: "flow"}, "in",
		func(ctx context.Context, in string) (string, error) {
			return "ran " + in, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if out != "ran in" {
		t.Errorf("output = %q, want %q", out, "ran in")
	}
	if len(fake.spans) != 1 {
		t.Fatalf("provider saw %d spans, want 1", len(fake.spans))
	}
	if got := fake.spans[0].Name(); got != "root" {
		t.Errorf("span name = %q, want root", got)
	}
}

func TestSpanInfoAccessors(t *testing.T) {
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()

	fake := &fakeInstrumentation{traceID: "t", spanID: "s"}
	ConfigureInstrumentation(fake)

	md := map[string]string{"k": "v"}
	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model", Metadata: md},
		"in",
		func(ctx context.Context, _ string) (string, error) { return "", nil })
	if err != nil {
		t.Fatal(err)
	}
	info := fake.spans[0]
	if info.Name() != "googleai/gemini-flash-latest" || info.Type() != "action" || info.Subtype() != "model" {
		t.Errorf("Name/Type/Subtype = %q/%q/%q", info.Name(), info.Type(), info.Subtype())
	}
	if got, want := info.Path(), "/{googleai/gemini-flash-latest,t:action,s:model}"; got != want {
		t.Errorf("Path = %q, want %q", got, want)
	}
	if !info.IsRoot() {
		t.Error("IsRoot = false, want true")
	}
	if info.Input() != "in" {
		t.Errorf("Input = %v, want in", info.Input())
	}
	got := info.Metadata()
	if got["k"] != "v" {
		t.Errorf("Metadata = %v, want {k: v}", got)
	}
	got["k"] = "mutated"
	if md["k"] != "v" {
		t.Error("Metadata returned the caller's map, want a copy")
	}
}

func TestSpanInfoAccessorsNilSafe(t *testing.T) {
	var nilInfo *SpanInfo
	for _, info := range []*SpanInfo{nilInfo, {}} {
		if info.Name() != "" || info.Type() != "" || info.Subtype() != "" || info.Path() != "" ||
			info.IsRoot() || info.Input() != nil || info.Metadata() != nil {
			t.Errorf("accessors on %#v returned non-zero values", info)
		}
	}
}

// TestRunInNewSpan_FallbackIDsWithoutProviderIDs covers a chain in which no
// provider tracks ids: the dispatcher still mints them (continuing the
// parent's trace) and fires the telemetry callback, which the reflection
// server's cancel registry and trace headers depend on.
func TestRunInNewSpan_FallbackIDsWithoutProviderIDs(t *testing.T) {
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()
	ConfigureInstrumentation(&fakeInstrumentation{})

	var cbTrace, cbSpan string
	ctx := WithTelemetryCallback(context.Background(), func(tid, sid string) {
		cbTrace, cbSpan = tid, sid
	})
	var root, child TraceInfo
	_, err := RunInNewSpan(ctx, &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			root = SpanTraceInfo(ctx)
			return RunInNewSpan(ctx, &SpanMetadata{Name: "child"}, "in",
				func(ctx context.Context, _ string) (string, error) {
					child = SpanTraceInfo(ctx)
					return "", nil
				})
		})
	if err != nil {
		t.Fatal(err)
	}
	if len(root.TraceID) != 32 || len(root.SpanID) != 16 {
		t.Errorf("root TraceInfo = %+v, want minted ids", root)
	}
	if cbTrace != root.TraceID || cbSpan == "" {
		t.Errorf("callback got (%q, %q), want root trace %q and a span id", cbTrace, cbSpan, root.TraceID)
	}
	if child.TraceID != root.TraceID || child.SpanID == root.SpanID {
		t.Errorf("child TraceInfo = %+v, want root's trace %q and its own span", child, root.TraceID)
	}
}

// TestResetInstrumentation_IgnoresEnv guards against the registry reading
// GENKIT_TELEMETRY_SERVER on its own: only genkit.Init and the reflection
// server enable Direct, so a reset chain stays clean whatever the shell has.
func TestResetInstrumentation_IgnoresEnv(t *testing.T) {
	t.Setenv("GENKIT_TELEMETRY_SERVER", "http://127.0.0.1:1")
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()
	if chain := activeInstrumentations(); len(chain) != 1 || chain[0] != defaultOTel {
		t.Errorf("chain after reset = %v, want [defaultOTel]", chain)
	}
}

func TestRunInNewSpan_ExposesCompositeTraceInfo(t *testing.T) {
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()

	fake := &fakeInstrumentation{traceID: "trace-123", spanID: "span-456"}
	ConfigureInstrumentation(fake)

	var got TraceInfo
	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			got = SpanTraceInfo(ctx)
			return "", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if got.TraceID != "trace-123" || got.SpanID != "span-456" {
		t.Errorf("SpanTraceInfo = %+v, want {trace-123 span-456}", got)
	}
}

func TestActiveInstrumentations_PrependsDirectWhenServerConfigured(t *testing.T) {
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()

	// No server: chain is just the configured provider.
	fake := &fakeInstrumentation{}
	ConfigureInstrumentation(fake)
	if chain := activeInstrumentations(); len(chain) != 1 {
		t.Fatalf("no-server chain length = %d, want 1", len(chain))
	}

	// Server configured: Direct is prepended as primary.
	EnableDevInstrumentation("http://localhost:4033")
	chain := activeInstrumentations()
	if len(chain) != 2 {
		t.Fatalf("dev chain length = %d, want 2", len(chain))
	}
	if _, ok := chain[0].(*DirectTelemetryInstrumentation); !ok {
		t.Errorf("chain[0] = %T, want *DirectTelemetryInstrumentation", chain[0])
	}
	if chain[1] != fake {
		t.Errorf("chain[1] = %v, want the configured fake", chain[1])
	}
}

func TestCompositeSpan_FirstNonEmptyIDsAndFanOut(t *testing.T) {
	// primary has no ids, secondary supplies them: the composite resolves the
	// first non-empty, and SetMetadata fans out to both.
	primary := &fakeInstrumentation{}
	secondary := &fakeInstrumentation{traceID: "T", spanID: "S"}

	info := &SpanInfo{metadata: &spanMetadata{Name: "n"}}
	_, err := dispatch(context.Background(),
		[]Instrumentation{primary, secondary}, info,
		func(ctx context.Context, span Span) (any, error) {
			if ti := span.TraceInfo(); ti.TraceID != "T" || ti.SpanID != "S" {
				t.Errorf("composite TraceInfo = %+v, want {T S}", ti)
			}
			span.SetMetadata(map[string]string{"k": "v"})
			return nil, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if len(primary.meta) != 1 || len(secondary.meta) != 1 {
		t.Errorf("SetMetadata fan-out: primary=%d secondary=%d, want 1/1",
			len(primary.meta), len(secondary.meta))
	}
}

func TestRunInNewSpan_ProviderSeesErrorState(t *testing.T) {
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()

	fake := &fakeInstrumentation{}
	ConfigureInstrumentation(fake)

	wantErr := errors.New("boom")
	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			return "", wantErr
		})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
	// The dispatcher fills state/error before the provider finalizes. State is
	// internal: an external provider reads the error from next's result.
	if got := fake.spans[0].metadata.State; got != spanStateError {
		t.Errorf("state = %q, want error", got)
	}
	if got := fake.spans[0].metadata.Error; got != "boom" {
		t.Errorf("error = %q, want boom", got)
	}
}

func TestSetCustomMetadataAttributes_FansOutToProvider(t *testing.T) {
	t.Cleanup(ResetInstrumentation)
	ResetInstrumentation()

	fake := &fakeInstrumentation{}
	ConfigureInstrumentation(fake)

	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			SetCustomMetadataAttributes(ctx, map[string]string{"agent:sessionId": "abc"})
			return "", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if len(fake.meta) != 1 || fake.meta[0]["agent:sessionId"] != "abc" {
		t.Errorf("provider metadata = %v, want one entry {agent:sessionId: abc}", fake.meta)
	}
}

func TestSetCustomMetadataAttributes_NoopOutsideSpan(t *testing.T) {
	// Must not panic when there is no active span.
	SetCustomMetadataAttributes(context.Background(), map[string]string{"k": "v"})
}
