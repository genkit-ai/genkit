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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/internal/tracingbridge"
	"github.com/google/go-cmp/cmp"
)

// fakeInstrumentation is a minimal provider that records calls and mints
// deterministic ids. It mirrors the JS instrumentation test's fake.
type fakeInstrumentation struct {
	name    string
	traceID string
	spanID  string
	spans   []*SpanInfo
	meta    []map[string]string
	results []*SpanResult
	// events, when set, is shared across fakes to observe start/end order.
	events *[]string
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

func (s *fakeSpan) End(res *SpanResult) {
	s.inst.results = append(s.inst.results, res)
	if s.inst.events != nil {
		*s.inst.events = append(*s.inst.events, "end "+s.inst.name)
	}
}

func (f *fakeInstrumentation) StartSpan(ctx context.Context, info *SpanInfo) (context.Context, Span) {
	f.spans = append(f.spans, info)
	if f.events != nil {
		*f.events = append(*f.events, "start "+f.name)
	}
	return ctx, &fakeSpan{inst: f}
}

// useInstrumentation sets providers for the test and restores the default on
// cleanup.
func useInstrumentation(t *testing.T, providers ...Instrumentation) {
	t.Helper()
	resetInstrumentation()
	t.Cleanup(resetInstrumentation)
	SetInstrumentation(providers...)
}

func TestSetInstrumentation_RoutesThroughProvider(t *testing.T) {
	fake := &fakeInstrumentation{traceID: "aaaa", spanID: "bbbb"}
	useInstrumentation(t, fake)

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
	fake := &fakeInstrumentation{traceID: "t", spanID: "s"}
	useInstrumentation(t, fake)

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
			info.IsRoot() || info.Input() != nil || info.Init() != nil || info.Metadata() != nil || info.Labels() != nil {
			t.Errorf("accessors on %#v returned non-zero values", info)
		}
	}
}

// TestRunInNewSpan_FallbackIDsWithoutProviderIDs covers a chain in which no
// provider tracks ids: the dispatcher still mints them (continuing the
// parent's trace) and fires the telemetry callback, which the reflection
// server's cancel registry and trace headers depend on.
func TestRunInNewSpan_FallbackIDsWithoutProviderIDs(t *testing.T) {
	useInstrumentation(t, &fakeInstrumentation{})

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
	t.Cleanup(resetInstrumentation)
	resetInstrumentation()
	if chain := activeInstrumentations(); len(chain) != 1 || chain[0] != defaultOTel {
		t.Errorf("chain after reset = %v, want [defaultOTel]", chain)
	}
}

func TestSetInstrumentation_NoArgsRestoresDefault(t *testing.T) {
	useInstrumentation(t, &fakeInstrumentation{})
	SetInstrumentation()
	if chain := activeInstrumentations(); len(chain) != 1 || chain[0] != defaultOTel {
		t.Errorf("chain = %v, want [defaultOTel]", chain)
	}
	// Nil providers are dropped rather than dispatched to.
	SetInstrumentation(nil, nil)
	if chain := activeInstrumentations(); len(chain) != 1 || chain[0] != defaultOTel {
		t.Errorf("chain after nil providers = %v, want [defaultOTel]", chain)
	}
}

func TestSetInstrumentation_MultipleProviders(t *testing.T) {
	var events []string
	a := &fakeInstrumentation{name: "a", events: &events}
	b := &fakeInstrumentation{name: "b", traceID: "T", spanID: "S", events: &events}
	useInstrumentation(t, a, b)

	_, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			events = append(events, "run")
			return "out", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	// Started in order, ended in reverse, the run exactly once in between.
	if diff := cmp.Diff([]string{"start a", "start b", "run", "end b", "end a"}, events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	// Both providers see the same result.
	if len(a.results) != 1 || len(b.results) != 1 || a.results[0] != b.results[0] {
		t.Fatalf("results a=%v b=%v, want one shared result", a.results, b.results)
	}
	if got := a.results[0].Output(); got != "out" {
		t.Errorf("Output = %v, want out", got)
	}
}

// TestWithInstrumentation pins the scoped providers: they see the spans
// started under their context, nested scopes keep the enclosing ones, they
// run after the process-wide providers and end before them, and a span
// started without the context reaches none of them.
func TestWithInstrumentation(t *testing.T) {
	var events []string
	global := &fakeInstrumentation{name: "global", events: &events}
	useInstrumentation(t, global)
	outer := &fakeInstrumentation{name: "outer", events: &events}
	inner := &fakeInstrumentation{name: "inner", events: &events}

	ctx := WithInstrumentation(context.Background(), outer)
	_, err := RunInNewSpan(ctx, &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			return RunInNewSpan(WithInstrumentation(ctx, inner), &SpanMetadata{Name: "child"}, "in",
				func(ctx context.Context, _ string) (string, error) { return "out", nil })
		})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"start global", "start outer",
		"start global", "start outer", "start inner",
		"end inner", "end outer", "end global",
		"end outer", "end global",
	}
	if diff := cmp.Diff(want, events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}

	events = nil
	if _, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "other"}, "in",
		func(ctx context.Context, _ string) (string, error) { return "out", nil }); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff([]string{"start global", "end global"}, events); diff != "" {
		t.Errorf("events outside the scope (-want +got):\n%s", diff)
	}
}

func TestRunInNewSpan_ExposesCompositeTraceInfo(t *testing.T) {
	fake := &fakeInstrumentation{traceID: "trace-123", spanID: "span-456"}
	useInstrumentation(t, fake)

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
	// No server: chain is just the configured provider.
	fake := &fakeInstrumentation{}
	useInstrumentation(t, fake)
	if chain := activeInstrumentations(); len(chain) != 1 {
		t.Fatalf("no-server chain length = %d, want 1", len(chain))
	}

	// Server configured (through the internal bridge, as genkit.Init does):
	// Direct is prepended as primary.
	tracingbridge.SetDevTelemetryServer("http://localhost:4033")
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

	// The dev provider survives SetInstrumentation.
	SetInstrumentation()
	if chain := activeInstrumentations(); len(chain) != 2 || chain[1] != defaultOTel {
		t.Errorf("chain after SetInstrumentation() = %v, want [Direct defaultOTel]", chain)
	}
}

// TestSetDevTelemetryServer_KeepsInstance covers reflection server reconnects:
// the same URL is a no-op and a new URL retargets the existing Direct
// instance, so children of a span open across the reconnect keep their parent.
func TestSetDevTelemetryServer_KeepsInstance(t *testing.T) {
	useInstrumentation(t, &fakeInstrumentation{})
	setDevTelemetryServer("http://127.0.0.1:1")
	d := direct
	before := activeChain.Load()

	setDevTelemetryServer("http://127.0.0.1:1")
	if direct != d || activeChain.Load() != before {
		t.Error("same URL replaced the instance or rebuilt the chain")
	}
	firstClient := d.client.Load()
	setDevTelemetryServer("http://127.0.0.1:1")
	if d.client.Load() != firstClient {
		t.Error("same URL replaced the client")
	}

	// Open a span, retarget mid-run, then open a child: the child must stay
	// in the parent's trace. The client is cleared so nothing is exported.
	d.client.Store(nil)
	var root, child TraceInfo
	_, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			root = SpanTraceInfo(ctx)
			setDevTelemetryServer("http://127.0.0.1:2")
			if got := d.client.Load(); got == nil || got.url != "http://127.0.0.1:2" {
				t.Errorf("client after retarget = %+v, want the new URL", got)
			}
			d.client.Store(nil)
			return RunInNewSpan(ctx, &SpanMetadata{Name: "child"}, "in",
				func(ctx context.Context, _ string) (string, error) {
					child = SpanTraceInfo(ctx)
					return "", nil
				})
		})
	if err != nil {
		t.Fatal(err)
	}
	if direct != d {
		t.Error("new URL replaced the instance, want it retargeted")
	}
	if child.TraceID != root.TraceID {
		t.Errorf("child trace = %q, want parent's %q", child.TraceID, root.TraceID)
	}

	// An empty URL (genkit.Init with GENKIT_TELEMETRY_SERVER unset) must not
	// undo a URL the reflection handshake supplied.
	setDevTelemetryServer("")
	if direct != d || len(activeInstrumentations()) != 2 {
		t.Error("empty URL disturbed the dev provider")
	}
}

func TestCompositeSpan_FirstNonEmptyIDsAndFanOut(t *testing.T) {
	// primary has no ids, secondary supplies them: the composite resolves the
	// first non-empty, and SetSpanMetadata fans out to both.
	primary := &fakeInstrumentation{}
	secondary := &fakeInstrumentation{traceID: "T", spanID: "S"}
	useInstrumentation(t, primary, secondary)

	_, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "n"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			if ti := SpanTraceInfo(ctx); ti.TraceID != "T" || ti.SpanID != "S" {
				t.Errorf("composite TraceInfo = %+v, want {T S}", ti)
			}
			SetSpanMetadata(ctx, map[string]string{"k": "v"})
			return "", nil
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
	fake := &fakeInstrumentation{}
	useInstrumentation(t, fake)

	wantErr := errors.New("boom")
	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			return "", wantErr
		})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
	// The provider sees the error on the result, and the dispatcher has
	// filled the internal state before ending the span.
	if got := fake.results[0].Err(); !errors.Is(got, wantErr) {
		t.Errorf("SpanResult.Err = %v, want %v", got, wantErr)
	}
	if got := fake.spans[0].metadata.State; got != spanStateError {
		t.Errorf("state = %q, want error", got)
	}
	if got := fake.spans[0].metadata.Error; got != "boom" {
		t.Errorf("error = %q, want boom", got)
	}
}

// TestRunInNewSpan_PartialOutputOnError checks a failed run's partial output
// reaches providers alongside the error.
func TestRunInNewSpan_PartialOutputOnError(t *testing.T) {
	fake := &fakeInstrumentation{}
	useInstrumentation(t, fake)

	_, _ = RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			return "partial", errors.New("boom")
		})
	res := fake.results[0]
	if res.Output() != "partial" || res.Err() == nil {
		t.Errorf("result = (%v, %v), want (partial, boom)", res.Output(), res.Err())
	}
}

// TestRunInNewSpan_PanicEndsSpans checks that a panicking run ends every
// started span with an error result and error state, then keeps panicking.
func TestRunInNewSpan_PanicEndsSpans(t *testing.T) {
	a := &fakeInstrumentation{name: "a"}
	b := &fakeInstrumentation{name: "b"}
	useInstrumentation(t, a, b)

	func() {
		defer func() {
			if r := recover(); r != "kaboom" {
				t.Errorf("recovered %v, want the original panic value", r)
			}
		}()
		_, _ = RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
			func(ctx context.Context, _ string) (string, error) { panic("kaboom") })
	}()

	for _, f := range []*fakeInstrumentation{a, b} {
		if len(f.results) != 1 {
			t.Fatalf("%s: ended %d times, want 1", f.name, len(f.results))
		}
		if err := f.results[0].Err(); err == nil || !strings.Contains(err.Error(), "kaboom") {
			t.Errorf("%s: Err = %v, want the panic", f.name, err)
		}
		if got := f.spans[0].metadata.State; got != spanStateError {
			t.Errorf("%s: state = %q, want error", f.name, got)
		}
	}
}

// nilProvider returns neither a context nor a span; the dispatcher must cope.
type nilProvider struct{}

func (nilProvider) StartSpan(context.Context, *SpanInfo) (context.Context, Span) { return nil, nil }

func TestRunInNewSpan_ToleratesNilStart(t *testing.T) {
	fake := &fakeInstrumentation{traceID: "T", spanID: "S"}
	useInstrumentation(t, nilProvider{}, fake)

	out, err := RunInNewSpan(context.Background(), &SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, in string) (string, error) {
			if ti := SpanTraceInfo(ctx); ti.TraceID != "T" {
				t.Errorf("TraceInfo = %+v, want the fake's ids", ti)
			}
			return in, nil
		})
	if err != nil || out != "in" {
		t.Fatalf("got (%q, %v), want (in, nil)", out, err)
	}
	if len(fake.results) != 1 {
		t.Errorf("fake ended %d times, want 1", len(fake.results))
	}
}

func TestSpanInfoLabelsIsCopy(t *testing.T) {
	labels := map[string]string{"k": "v"}
	fake := &fakeInstrumentation{}
	useInstrumentation(t, fake)

	_, _ = RunInNewSpan(context.Background(), &SpanMetadata{Name: "root", TelemetryLabels: labels}, "in",
		func(ctx context.Context, _ string) (string, error) { return "", nil })
	got := fake.spans[0].Labels()
	if got["k"] != "v" {
		t.Fatalf("Labels = %v, want {k: v}", got)
	}
	got["k"] = "mutated"
	if labels["k"] != "v" {
		t.Error("Labels returned the caller's map, want a copy")
	}
}

func TestSetSpanMetadata_FansOutToProvider(t *testing.T) {
	fake := &fakeInstrumentation{}
	useInstrumentation(t, fake)

	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "root"}, "in",
		func(ctx context.Context, _ string) (string, error) {
			SetSpanMetadata(ctx, map[string]string{"agent:sessionId": "abc"})
			return "", nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if len(fake.meta) != 1 || fake.meta[0]["agent:sessionId"] != "abc" {
		t.Errorf("provider metadata = %v, want one entry {agent:sessionId: abc}", fake.meta)
	}
}

func TestSetSpanMetadata_NoopOutsideSpan(t *testing.T) {
	// Must not panic when there is no active span.
	SetSpanMetadata(context.Background(), map[string]string{"k": "v"})
}
