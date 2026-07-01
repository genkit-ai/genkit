// Copyright 2025 Google LLC
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
	"encoding/json"
	"sync"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// recordingClient captures every Save call so a test can inspect what reached
// the telemetry server and in what order. It is safe for the concurrent saves
// the realtime processor performs on span start.
type recordingClient struct {
	mu    sync.Mutex
	saves []*Data
}

func (c *recordingClient) Save(ctx context.Context, td *Data) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.saves = append(c.saves, td)
	return nil
}

func (c *recordingClient) all() []*Data {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*Data(nil), c.saves...)
}

// newRealtimeTestProvider returns a tracer provider wired to a fresh recording
// client through the realtime processor, plus the processor so tests can flush
// its async start-of-span saves before asserting.
func newRealtimeTestProvider(t *testing.T) (*sdktrace.TracerProvider, *realtimeSpanProcessor, *recordingClient) {
	t.Helper()
	client := &recordingClient{}
	rp := newRealtimeSpanProcessor(client)
	tp := sdktrace.NewTracerProvider()
	tp.RegisterSpanProcessor(rp)
	return tp, rp, client
}

// topLevelKeys returns the set of top-level JSON keys td marshals to. Used to
// assert that omitempty drops trace metadata (displayName/startTime/endTime)
// when a save carries no root span.
func topLevelKeys(t *testing.T, td *Data) map[string]bool {
	t.Helper()
	b, err := json.Marshal(td)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	keys := map[string]bool{}
	for k := range m {
		keys[k] = true
	}
	return keys
}

// TestRealtimeSpanProcessorExportsOnStartAndEnd verifies the core behavior: a
// span is sent once as it starts (in progress, no end time) and once as it
// ends (complete), and the start save already carries the root span so the
// telemetry server can list the trace before it finishes.
func TestRealtimeSpanProcessorExportsOnStartAndEnd(t *testing.T) {
	tp, rp, client := newRealtimeTestProvider(t)

	// Start the span with attributes, the way RunInNewSpan seeds its
	// startAttributes, so the test also covers that those reach the start export
	// (a still-running span must render with its type/subtype, not just exist).
	_, span := tp.Tracer("test").Start(context.Background(), "root",
		trace.WithAttributes(
			attribute.String("genkit:type", "action"),
			attribute.String("genkit:metadata:subtype", "agent"),
		))
	spanID := span.SpanContext().SpanID().String()
	span.End()

	if err := rp.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}

	// Exactly two saves: one for OnStart, one for OnEnd. Their order in the
	// recording client is not deterministic, because the start save runs on a
	// goroutine while the end save is synchronous, so identify each by content
	// (the start save's span is in progress, the end save's is complete) rather
	// than by position. The server's span-merge tolerates this same reordering.
	saves := client.all()
	if len(saves) != 2 {
		t.Fatalf("expected exactly 2 saves (start and end), got %d", len(saves))
	}
	var startSave, endSave *Data
	for _, td := range saves {
		s := td.Spans[spanID]
		if s == nil {
			t.Fatalf("save missing span %s", spanID)
		}
		if s.EndTime == 0 {
			startSave = td
		} else {
			endSave = td
		}
	}
	if startSave == nil {
		t.Fatal("no in-progress (start) save found; span was never exported before it ended")
	}
	if endSave == nil {
		t.Fatal("no completed (end) save found")
	}

	// The start save carries the root span, so the trace's startTime is set
	// while its endTime is omitted (the trace is still running).
	keys := topLevelKeys(t, startSave)
	if !keys["startTime"] {
		t.Errorf("start save: want top-level startTime present, got keys %v", keys)
	}
	if keys["endTime"] {
		t.Errorf("start save: want top-level endTime omitted, got keys %v", keys)
	}

	// The span-start attributes must already be on the in-progress span, so the
	// dev UI can render it correctly before it finishes.
	startSpan := startSave.Spans[spanID]
	if got := startSpan.Attributes["genkit:type"]; got != "action" {
		t.Errorf("start save: genkit:type = %v, want \"action\"", got)
	}
	if got := startSpan.Attributes["genkit:metadata:subtype"]; got != "agent" {
		t.Errorf("start save: genkit:metadata:subtype = %v, want \"agent\"", got)
	}

	// The end save completes the trace, so its endTime is present.
	if !topLevelKeys(t, endSave)["endTime"] {
		t.Errorf("end save: want top-level endTime present")
	}
}

// TestRealtimeSpanProcessorChildSaveOmitsTraceMetadata guards the regression
// the omitempty tags exist for: a save carrying only a child span must not
// carry trace-level displayName/startTime/endTime, or it would clobber the
// metadata a prior root-span save established on the server.
func TestRealtimeSpanProcessorChildSaveOmitsTraceMetadata(t *testing.T) {
	tp, rp, client := newRealtimeTestProvider(t)

	ctx, parent := tp.Tracer("test").Start(context.Background(), "root")
	_, child := tp.Tracer("test").Start(ctx, "child")
	childID := child.SpanContext().SpanID().String()
	child.End()
	parent.End()

	if err := rp.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}

	sawChildOnlySave := false
	for _, td := range client.all() {
		// A save is "child-only" when every span it carries has a parent.
		childOnly := len(td.Spans) > 0
		for _, s := range td.Spans {
			if s.ParentSpanID == "" {
				childOnly = false
			}
		}
		if !childOnly {
			continue
		}
		if td.Spans[childID] == nil {
			continue
		}
		sawChildOnlySave = true
		keys := topLevelKeys(t, td)
		for _, k := range []string{"displayName", "startTime", "endTime"} {
			if keys[k] {
				t.Errorf("child-only save: top-level %q should be omitted, got keys %v", k, keys)
			}
		}
	}
	if !sawChildOnlySave {
		t.Fatal("expected at least one child-only save (the child span's start export)")
	}
}

// TestRunInNewSpanSeedsInputAtStartWhenRealtime verifies the end-to-end wiring:
// when live export is active, a span's input and init reach the start export, so
// an in-flight span shows what it was invoked with rather than waiting until it
// ends. Off the live path RunInNewSpan leaves them to the end write, which this
// test does not exercise.
func TestRunInNewSpanSeedsInputAtStartWhenRealtime(t *testing.T) {
	// Force live mode and route the global tracer through a realtime processor
	// for the duration of the test, restoring both afterward.
	prevActive := realtimeTelemetryActive
	realtimeTelemetryActive = true
	defer func() { realtimeTelemetryActive = prevActive }()

	prevTP := otel.GetTracerProvider()
	defer otel.SetTracerProvider(prevTP)

	client := &recordingClient{}
	rp := newRealtimeSpanProcessor(client)
	tp := sdktrace.NewTracerProvider()
	tp.RegisterSpanProcessor(rp)
	otel.SetTracerProvider(tp)

	_, err := RunInNewSpan(context.Background(),
		&SpanMetadata{Name: "act", Type: "action", Init: map[string]string{"k": "v"}},
		"my-input",
		func(ctx context.Context, in string) (any, error) { return "out", nil },
	)
	if err != nil {
		t.Fatalf("RunInNewSpan: %v", err)
	}
	if err := rp.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}

	// Locate the start export: the save whose span is still in progress.
	var startSpan *SpanData
	for _, td := range client.all() {
		for _, s := range td.Spans {
			if s.EndTime == 0 {
				startSpan = s
			}
		}
	}
	if startSpan == nil {
		t.Fatal("no in-progress (start) export found")
	}
	if got := startSpan.Attributes["genkit:input"]; got != `"my-input"` {
		t.Errorf("start export: genkit:input = %v, want %q", got, `"my-input"`)
	}
	if got := startSpan.Attributes["genkit:init"]; got != `{"k":"v"}` {
		t.Errorf("start export: genkit:init = %v, want %q", got, `{"k":"v"}`)
	}
}

// TestRealtimeSpanProcessorDropsStartSavesAfterShutdown verifies the guard that
// keeps Shutdown's drain safe: once Shutdown has begun, a span starting
// afterward must not enqueue an async start save (which would Add to the
// WaitGroup wait is draining). The synchronous end save is unaffected.
func TestRealtimeSpanProcessorDropsStartSavesAfterShutdown(t *testing.T) {
	tp, rp, client := newRealtimeTestProvider(t)

	if err := rp.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	// Start and end a span after shutdown. OnStart's async save must be dropped;
	// OnEnd's synchronous save still lands. Neither may panic.
	_, span := tp.Tracer("test").Start(context.Background(), "root")
	span.End()

	// A second ForceFlush must return immediately (nothing in flight).
	if err := rp.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}

	for _, td := range client.all() {
		for _, s := range td.Spans {
			if s.EndTime == 0 {
				t.Errorf("in-progress (start) save recorded after shutdown; guard did not drop it")
			}
		}
	}
}

// TestRealtimeSpanProcessorNilClient ensures a processor with no client is a
// no-op rather than a panic (matches the exporter's nil-client tolerance).
func TestRealtimeSpanProcessorNilClient(t *testing.T) {
	rp := newRealtimeSpanProcessor(nil)
	tp := sdktrace.NewTracerProvider()
	tp.RegisterSpanProcessor(rp)
	_, span := tp.Tracer("test").Start(context.Background(), "root")
	span.End()
	if err := rp.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}
}
