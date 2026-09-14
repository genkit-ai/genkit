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
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/firebase/genkit/go/internal/base"
)

// DirectTelemetryInstrumentation feeds the Developer UI directly, with no
// OpenTelemetry SDK. It mints its own ids, tracks parentage via context, builds
// the same SpanData/Data shape the OTel-backed telemetryServerExporter produced,
// and POSTs to ${server}/api/traces via the existing TelemetryClient. It reuses
// the realtime (start + end) export behavior so still-running traces appear in
// the Dev UI.
//
// Prepended as the primary provider in dev when a telemetry server is
// configured, replacing the old span-processor auto-registration.
type DirectTelemetryInstrumentation struct {
	client TelemetryClient
}

// NewDirectTelemetryInstrumentation returns a Direct provider that exports to
// client. Useful for wiring a custom or in-memory [TelemetryClient] (e.g. in
// tests via [ConfigureInstrumentation]); the dev wiring uses the URL-based
// constructor.
func NewDirectTelemetryInstrumentation(client TelemetryClient) *DirectTelemetryInstrumentation {
	return &DirectTelemetryInstrumentation{client: client}
}

func newDirectTelemetryInstrumentation(url string) *DirectTelemetryInstrumentation {
	return NewDirectTelemetryInstrumentation(NewHTTPTelemetryClient(url))
}

// directParent carries the enclosing Direct span's ids down the context, so
// child spans share the trace id and point their parentSpanId at it.
type directParent struct {
	traceID string
	spanID  string
}

var directParentKey = base.NewContextKey[*directParent]()

// directSpan is the Span handle over a Direct span.
type directSpan struct {
	traceID string
	spanID  string
	// extra holds mid-run custom metadata; folded into the end write.
	mu    sync.Mutex
	extra map[string]string
}

func (s *directSpan) TraceInfo() TraceInfo {
	return TraceInfo{TraceID: s.traceID, SpanID: s.spanID}
}

func (s *directSpan) SetMetadata(md map[string]string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.extra == nil {
		s.extra = map[string]string{}
	}
	for k, v := range md {
		s.extra[k] = v
	}
}

// RunInNewSpan mints ids, runs next, then builds and exports the span. It
// exports on start too (as in progress) when realtime export is active, so a
// long-lived root span shows up in the Dev UI before it closes.
func (d *DirectTelemetryInstrumentation) RunInNewSpan(ctx context.Context, info *SpanInfo, next NextFunc) (any, error) {
	parent := directParentKey.FromContext(ctx)
	traceID := genID(16)
	if parent != nil {
		traceID = parent.traceID
	}
	spanID := genID(8)
	span := &directSpan{traceID: traceID, spanID: spanID}

	start := time.Now()
	if realtimeTelemetryEnabled() {
		d.exportStart(info, span, parent, start)
	}

	ctx = directParentKey.NewContext(ctx, &directParent{traceID: traceID, spanID: spanID})
	out, err := next(ctx, span)
	d.exportEnd(info, span, parent, start, time.Now(), err)
	return out, err
}

// buildData wraps one span in a Data envelope, setting the trace-level fields
// from the root (parentless) span, exactly as telemetryServerExporter.convertTrace
// did. On a start export end is zero, so EndTime is omitted (in progress).
func (d *DirectTelemetryInstrumentation) buildData(info *SpanInfo, span *directSpan, parent *directParent, start, end time.Time, runErr error) *Data {
	td := &Data{
		TraceID: span.traceID,
		Spans:   map[string]*SpanData{span.spanID: d.buildSpan(info, span, parent, start, end, runErr)},
	}
	if parent == nil {
		td.DisplayName = info.metadata.Name
		td.StartTime = ToMilliseconds(start)
		if !end.IsZero() {
			td.EndTime = ToMilliseconds(end)
		}
	}
	return td
}

// exportStart ships the span as in progress (endTime 0), asynchronously, so
// span creation never blocks on telemetry I/O. A live preview is best-effort:
// a dropped start export is corrected by the synchronous end export.
func (d *DirectTelemetryInstrumentation) exportStart(info *SpanInfo, span *directSpan, parent *directParent, start time.Time) {
	td := d.buildData(info, span, parent, start, time.Time{}, nil)
	go func() {
		if err := d.client.Save(context.Background(), td); err != nil {
			reportTraceSaveError(err)
		}
	}()
}

// exportEnd ships the completed span synchronously, matching the durability of
// the immediate export path. A fresh context is used because the action context
// is often canceled by the time the span ends.
func (d *DirectTelemetryInstrumentation) exportEnd(info *SpanInfo, span *directSpan, parent *directParent, start, end time.Time, runErr error) {
	td := d.buildData(info, span, parent, start, end, runErr)
	if err := d.client.Save(context.Background(), td); err != nil {
		reportTraceSaveError(err)
	}
}

// buildSpan reproduces the SpanData shape telemetryServerExporter.convertSpan
// produced, so before/after Dev UI traces are structurally identical. A zero
// end leaves EndTime at 0, which the telemetry server treats as in progress.
func (d *DirectTelemetryInstrumentation) buildSpan(info *SpanInfo, span *directSpan, parent *directParent, start, end time.Time, runErr error) *SpanData {
	sm := info.metadata
	attrs := attributesToMap(sm.attributes())
	for k, v := range info.Labels {
		attrs[k] = v
	}
	span.mu.Lock()
	for k, v := range span.extra {
		attrs[attrPrefix+":metadata:"+k] = v
	}
	span.mu.Unlock()

	sd := &SpanData{
		SpanID:                  span.spanID,
		TraceID:                 span.traceID,
		StartTime:               ToMilliseconds(start),
		Attributes:              attrs,
		DisplayName:             sm.Name,
		InstrumentationScope:    InstrumentationScope{Name: "genkit-tracer", Version: "v1"},
		SpanKind:                "INTERNAL",
		SameProcessAsParentSpan: BoolValue{Value: true},
	}
	if !end.IsZero() {
		sd.EndTime = ToMilliseconds(end)
	}
	if parent != nil {
		sd.ParentSpanID = parent.spanID
	}
	// On error record the OTel-equivalent status and exception event, matching
	// what span.SetStatus / RecordError emitted through the exporter before. The
	// isErrorAlreadyMarked gate mirrors the OTel provider exactly.
	if runErr != nil && !isErrorAlreadyMarked(runErr) {
		sd.Status = Status{Code: 1, Description: runErr.Error()}
		if !end.IsZero() {
			sd.TimeEvents.TimeEvent = []TimeEvent{{
				Time: ToMilliseconds(end),
				Annotation: Annotation{
					Description: "exception",
					Attributes: map[string]any{
						"exception.message": runErr.Error(),
						"exception.type":    fmt.Sprintf("%T", runErr),
					},
				},
			}}
		}
	}
	return sd
}

// genID returns a hex-encoded random id of n bytes (16 for a trace id, 8 for a
// span id).
func genID(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand.Read never fails on supported platforms; if it somehow
		// does, a debug note beats crashing the caller's flow.
		slog.Debug("direct telemetry: failed to generate id", "error", err)
	}
	return hex.EncodeToString(b)
}
