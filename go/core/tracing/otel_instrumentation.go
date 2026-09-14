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

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// OTelInstrumentation is the default, back-compatible instrumentation: exactly
// what RunInNewSpan did historically via the OpenTelemetry SDK. Pure span
// creation. It reads whatever provider the caller registered via
// otel.SetTracerProvider (a user's OTel setup, or the GCP / Firebase plugins)
// and has no opinion about where that provider comes from; Genkit no longer
// installs one. When nothing is configured, OTel's no-op provider yields
// all-zero, invalid ids, which TraceInfo reports as empty. It also puts an OTel
// span in the context, so direct trace.SpanFromContext writes (e.g. in
// ai/exp/agent.go) keep working while OTel is in the chain.
//
// Back-compat default; removed in the next major to reach the shared "not
// instrumented by default" goal.
type OTelInstrumentation struct{}

// otelSpan is the Span handle over an OTel span.
type otelSpan struct {
	span trace.Span
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

// RunInNewSpan opens an OTel span, seeds the start-known genkit attributes,
// runs next, then reasserts the full attribute set and records error status.
func (o *OTelInstrumentation) RunInNewSpan(ctx context.Context, info *SpanInfo, next NextFunc) (any, error) {
	sm := info.metadata

	var opts []trace.SpanStartOption
	if len(info.Labels) > 0 {
		attrs := make([]attribute.KeyValue, 0, len(info.Labels))
		for k, v := range info.Labels {
			attrs = append(attrs, attribute.String(k, v))
		}
		opts = append(opts, trace.WithAttributes(attrs...))
	}
	// Seed the start-known genkit attributes (including genkit:type) so a
	// live-trace export taken the moment the span starts already carries its
	// name, path, type, and subtype. The deferred end write below reasserts
	// these and adds the run-determined output/state.
	opts = append(opts, trace.WithAttributes(sm.startAttributes()...))
	// Input and init are known now too, but JSON-marshaled, so seed them at
	// start only when a live exporter will read them; otherwise the end write
	// records them once.
	if realtimeTelemetryEnabled() {
		opts = append(opts, trace.WithAttributes(sm.inputAttributes()...))
	}

	// Read the provider the caller configured; do not install or manage one.
	// With nothing configured this is OTel's no-op provider (empty ids).
	tracer := otel.GetTracerProvider().Tracer("genkit-tracer", trace.WithInstrumentationVersion("v1"))
	ctx, span := tracer.Start(ctx, sm.Name, opts...)
	defer span.End()
	// The deferred end write reasserts the full attribute set (including the
	// run-determined output/state). Registered after span.End so it runs first.
	defer func() { span.SetAttributes(sm.attributes()...) }()

	out, err := next(ctx, &otelSpan{span: span})
	if err != nil && !isErrorAlreadyMarked(err) {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	return out, err
}
