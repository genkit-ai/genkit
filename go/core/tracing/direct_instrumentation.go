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
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"reflect"
	"sync"
	"time"

	"github.com/firebase/genkit/go/internal/base"
	"go.opentelemetry.io/otel/attribute"
)

// maxInFlightStartExports bounds concurrent start-of-span saves per provider.
// A start save is only a live preview, so when a high fan-out turn saturates
// the slots the preview is skipped rather than queued; the end save still
// delivers the span.
const maxInFlightStartExports = 16

// DirectTelemetryInstrumentation feeds the Developer UI directly, with no
// OpenTelemetry SDK. It mints its own ids, tracks parentage via context, and
// POSTs each span to ${server}/api/traces through a [TelemetryClient], in the
// same SpanData/Data shape the telemetry server ingests from the OTel exporter.
// When realtime export is active (GENKIT_ENABLE_REALTIME_TELEMETRY) it also
// saves each span as it starts, so still-running traces appear in the Dev UI.
//
// In dev it is prepended to the chain when a telemetry server is configured.
// One built with a nil client mints ids but exports nothing.
type DirectTelemetryInstrumentation struct {
	client TelemetryClient
	// parentKey is per instance: two Direct providers in one chain (the dev
	// one plus a user-configured one) each track their own parentage and must
	// not read each other's span ids.
	parentKey base.ContextKey[*directParent]
	// startSlots is a semaphore over in-flight start saves.
	startSlots chan struct{}
}

// NewDirectTelemetryInstrumentation returns a Direct provider that exports to
// client. Useful for wiring a custom or in-memory [TelemetryClient] (e.g. in
// tests via [ConfigureInstrumentation]); the dev wiring uses the URL-based
// constructor.
func NewDirectTelemetryInstrumentation(client TelemetryClient) *DirectTelemetryInstrumentation {
	return &DirectTelemetryInstrumentation{
		client:     client,
		parentKey:  base.NewContextKey[*directParent](),
		startSlots: make(chan struct{}, maxInFlightStartExports),
	}
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

// directSpan is the Span handle over a Direct span.
type directSpan struct {
	traceID string
	spanID  string
	// startDone is closed when the start save finishes; nil when none was
	// sent. Written and read only on the span's own goroutine.
	startDone chan struct{}
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
func (d *DirectTelemetryInstrumentation) RunInNewSpan(ctx context.Context, info *SpanInfo, next NextFunc) (out any, err error) {
	parent := d.parentKey.FromContext(ctx)
	traceID := genID(16)
	if parent != nil {
		traceID = parent.traceID
	}
	spanID := genID(8)
	span := &directSpan{traceID: traceID, spanID: spanID}

	start := time.Now()
	if d.client != nil && realtimeTelemetryEnabled() {
		d.exportStart(info, span, parent, start)
	}
	// Deferred so a panic in next still finalizes the span, matching the OTel
	// provider's defer span.End(); otherwise the Dev UI shows it stuck "in
	// progress".
	defer func() {
		if d.client != nil {
			d.exportEnd(info, span, parent, start, time.Now(), err)
		}
	}()

	ctx = d.parentKey.NewContext(ctx, &directParent{traceID: traceID, spanID: spanID})
	out, err = next(ctx, span)
	return out, err
}

// buildData wraps one span in a Data envelope. Only the root (parentless) span
// sets the trace-level fields, and the trace EndTime stays unset until the root
// ends.
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
// span creation never blocks on telemetry I/O. The preview is best-effort: it
// is skipped when too many start saves are already in flight.
func (d *DirectTelemetryInstrumentation) exportStart(info *SpanInfo, span *directSpan, parent *directParent, start time.Time) {
	select {
	case d.startSlots <- struct{}{}:
	default:
		return
	}
	// Built synchronously: the goroutine only gets a plain value.
	td := d.buildData(info, span, parent, start, time.Time{}, nil)
	done := make(chan struct{})
	span.startDone = done
	go func() {
		defer func() {
			<-d.startSlots
			close(done)
		}()
		if err := d.client.Save(context.Background(), td); err != nil {
			reportTraceSaveError(err)
		}
	}()
}

// exportEnd ships the completed span synchronously, so it has been handed to
// the client once the span returns. It first waits for the span's own start
// save: otherwise a slow start save could land last and overwrite the
// completed span in a client that does not merge (the telemetry server does,
// but TelemetryClient implementations need not). A fresh context is used
// because the action context is often canceled by the time the span ends.
func (d *DirectTelemetryInstrumentation) exportEnd(info *SpanInfo, span *directSpan, parent *directParent, start, end time.Time, runErr error) {
	if span.startDone != nil {
		<-span.startDone
	}
	td := d.buildData(info, span, parent, start, end, runErr)
	if err := d.client.Save(context.Background(), td); err != nil {
		reportTraceSaveError(err)
	}
}

// buildSpan encodes the span in the shape telemetryServerExporter.convertSpan
// produces for an OTel span, so the Dev UI renders both the same. A zero end
// leaves EndTime at 0, which the telemetry server treats as in progress.
func (d *DirectTelemetryInstrumentation) buildSpan(info *SpanInfo, span *directSpan, parent *directParent, start, end time.Time, runErr error) *SpanData {
	sm := info.metadata
	final := !end.IsZero()

	// Layered in the order an OTel span accumulates them, later writes
	// winning: labels at start, mid-run custom metadata, then the genkit
	// attributes. So a label can never clobber a genkit key such as
	// genkit:state. An in-progress span carries only what is known at start.
	attrs := make(map[string]any, len(info.Labels)+10)
	for k, v := range info.Labels {
		attrs[k] = v
	}
	if final {
		span.mu.Lock()
		for k, v := range span.extra {
			attrs[attrPrefix+":metadata:"+k] = v
		}
		span.mu.Unlock()
		putAttributes(attrs, sm.attributes())
	} else {
		putAttributes(attrs, sm.startAttributes())
		putAttributes(attrs, sm.inputAttributes())
	}

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
	if final {
		sd.EndTime = ToMilliseconds(end)
	}
	if parent != nil {
		sd.ParentSpanID = parent.spanID
	}
	// On error, the status and exception event OTel's span.SetStatus and
	// span.RecordError produce. runErr is only set on the end export.
	if runErr != nil {
		sd.Status = Status{Code: 1, Description: runErr.Error()}
		sd.TimeEvents.TimeEvent = []TimeEvent{{
			Time: ToMilliseconds(end),
			Annotation: Annotation{
				Description: "exception",
				Attributes: map[string]any{
					"exception.type":    errorTypeName(runErr),
					"exception.message": runErr.Error(),
				},
			},
		}}
	}
	return sd
}

// putAttributes copies kvs into m, later keys overwriting earlier ones.
func putAttributes(m map[string]any, kvs []attribute.KeyValue) {
	for _, kv := range kvs {
		m[string(kv.Key)] = kv.Value.AsInterface()
	}
}

// errorTypeName names err's type the way OTel's span.RecordError does for
// exception.type: package path qualified for named types (e.g.
// "github.com/x/pkg.MyErr"), the plain type string for unnamed ones (e.g.
// "*errors.errorString").
func errorTypeName(err error) string {
	t := reflect.TypeOf(err)
	if t.PkgPath() == "" && t.Name() == "" {
		return t.String()
	}
	return fmt.Sprintf("%s.%s", t.PkgPath(), t.Name())
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
