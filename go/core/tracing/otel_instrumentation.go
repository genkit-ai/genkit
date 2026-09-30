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
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// OTelInstrumentation encodes each Genkit span as an OpenTelemetry span with
// the genkit:* attributes. It is the implicit default (see
// [SetInstrumentation]). It installs no TracerProvider itself: with
// TracerProvider unset it reads the global one (otel.SetTracerProvider, set by
// a user's OTel setup or the Google Cloud / Firebase plugins). When nothing is
// configured, OTel's no-op provider yields all-zero, invalid ids, which
// TraceInfo reports as empty. It also puts the OTel span in the context, so
// direct trace.SpanFromContext writes land on it.
//
// Back-compat default; removed as the default in the next major to reach the
// shared "not instrumented by default" goal.
type OTelInstrumentation struct {
	// TracerProvider creates the spans. Nil means the global provider,
	// resolved per span so a provider registered after this value was built
	// is still picked up. Set it to keep Genkit spans on a dedicated provider
	// (a different sampler or exporter) without touching the global.
	TracerProvider trace.TracerProvider

	// tracer caches the tracer for a non-nil TracerProvider.
	tracerOnce sync.Once
	tracer     trace.Tracer
}

const (
	otelTracerName    = "genkit-tracer"
	otelTracerVersion = "v1"
)

func (o *OTelInstrumentation) getTracer() trace.Tracer {
	if o.TracerProvider == nil {
		return otel.GetTracerProvider().Tracer(otelTracerName, trace.WithInstrumentationVersion(otelTracerVersion))
	}
	o.tracerOnce.Do(func() {
		o.tracer = o.TracerProvider.Tracer(otelTracerName, trace.WithInstrumentationVersion(otelTracerVersion))
	})
	return o.tracer
}

// otelSpan is the Span handle over an OTel span.
type otelSpan struct {
	span trace.Span
	sm   *spanMetadata
}

func (s *otelSpan) TraceInfo() TraceInfo {
	sc := s.span.SpanContext()
	// The no-op provider (nothing configured) yields all-zero, invalid ids.
	// Report those as empty so the composite and log/callback correlation treat
	// this provider as tracking no ids, rather than a real zero span.
	if !sc.IsValid() {
		return TraceInfo{}
	}
	return TraceInfo{TraceID: sc.TraceID().String(), SpanID: sc.SpanID().String()}
}

func (s *otelSpan) SetMetadata(md map[string]string) {
	for k, v := range md {
		s.span.SetAttributes(attribute.String(attrPrefix+":metadata:"+k, v))
	}
}

// End reasserts the full attribute set (including the run-determined
// output/state), records error status, and ends the OTel span. Encoding is
// skipped for a non-recording span (the no-op provider, or unsampled), which
// would discard the attributes after paying to JSON-encode them.
func (s *otelSpan) End(res *SpanResult) {
	if s.span.IsRecording() {
		s.span.SetAttributes(s.sm.attributes()...)
		if err := res.Err(); err != nil {
			s.span.RecordError(err)
			s.span.SetStatus(codes.Error, err.Error())
		}
	}
	s.span.End()
}

// StartSpan opens an OTel span seeded with the start-known genkit attributes.
func (o *OTelInstrumentation) StartSpan(ctx context.Context, info *SpanInfo) (context.Context, Span) {
	sm := info.spanMeta()

	var opts []trace.SpanStartOption
	if len(info.labels) > 0 {
		attrs := make([]attribute.KeyValue, 0, len(info.labels))
		for k, v := range info.labels {
			attrs = append(attrs, attribute.String(k, v))
		}
		opts = append(opts, trace.WithAttributes(attrs...))
	}
	// Seed the start-known genkit attributes (including genkit:type) so a
	// live-trace export taken the moment the span starts already carries its
	// name, path, type, and subtype. End reasserts these and adds the
	// run-determined output/state.
	opts = append(opts, trace.WithAttributes(sm.startAttributes()...))
	// Input and init are known now too, but JSON-marshaled, so seed them at
	// start only when a live exporter will read them; otherwise End records
	// them once.
	if realtimeTelemetryEnabled() {
		opts = append(opts, trace.WithAttributes(sm.inputAttributes()...))
	}

	ctx, span := o.getTracer().Start(ctx, sm.Name, opts...)
	return ctx, &otelSpan{span: span, sm: sm}
}
