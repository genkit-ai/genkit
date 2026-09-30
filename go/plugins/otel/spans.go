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

package otel

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// genaiSpan is the tracing.Span over one OTel span. StartSpan records what is
// known up front; End records the result (response, content, metrics, error)
// and ends the OTel span.
type genaiSpan struct {
	g    *GenAiInstrumentation
	span oteltrace.Span
	// ctx carries span as the active span. Kept because End has no context of
	// its own, and the content log record and metrics must correlate to this
	// span, as they did when they were recorded inside the span's scope.
	ctx   context.Context
	input any
	// model is set for model spans only.
	model *modelCall
}

// modelCall is the model-span state End needs.
type modelCall struct {
	request *ai.ModelRequest
	// metricAttrs are the low-cardinality attributes shared by both metrics.
	metricAttrs []attribute.KeyValue
	start       time.Time
}

func (s *genaiSpan) TraceInfo() tracing.TraceInfo {
	sc := s.span.SpanContext()
	if !sc.IsValid() {
		return tracing.TraceInfo{}
	}
	return tracing.TraceInfo{TraceID: sc.TraceID().String(), SpanID: sc.SpanID().String()}
}

func (s *genaiSpan) SetMetadata(md map[string]string) {
	for k, v := range md {
		s.span.SetAttributes(attribute.String("genkit:metadata:"+k, v))
	}
}

// End records the run's result and ends the OTel span.
//
// A failed model call can still return a response (e.g. output schema
// validation fails after the model already ran and billed tokens), so the
// response is recorded whenever present, not only on success.
func (s *genaiSpan) End(res *tracing.SpanResult) {
	defer s.span.End()
	out, err := res.Output(), res.Err()
	g := s.g

	if m := s.model; m != nil {
		response := asModelResponse(out)
		if response != nil {
			addResponseAttributes(s.span, response, err != nil)
		}
		if g.contentMode != genai.NoContent {
			g.recordContent(s.ctx, s.span, m.request, response, err != nil)
		}
		if g.metrics != nil {
			g.recordModelMetrics(s.ctx, m.start, m.metricAttrs, response, err)
		}
	}
	// Includes a partial output returned alongside an error.
	g.maybeCaptureActionIO(s.span, s.input, out)
	if err != nil {
		g.recordError(s.span, err)
	}
}

// labelAttrs returns the span's telemetry labels as attributes, matching the
// default OTel instrumentation.
func labelAttrs(info *tracing.SpanInfo) []attribute.KeyValue {
	labels := info.Labels()
	attrs := make([]attribute.KeyValue, 0, len(labels))
	for k, v := range labels {
		attrs = append(attrs, attribute.String(k, v))
	}
	return attrs
}

// startSpan opens an OTel span and wraps it.
func (g *GenAiInstrumentation) startSpan(ctx context.Context, info *tracing.SpanInfo, name string, kind oteltrace.SpanKind, attrs []attribute.KeyValue) (context.Context, *genaiSpan) {
	ctx, span := g.tracer.Start(ctx, name,
		oteltrace.WithSpanKind(kind),
		oteltrace.WithAttributes(attrs...))
	g.maybeWarnNotRecording(ctx, span)
	return ctx, &genaiSpan{g: g, span: span, ctx: ctx, input: info.Input()}
}

// startModelSpan opens a gen_ai chat CLIENT span for a model action, carrying
// the request config; End adds the response attributes, content, and metrics.
func (g *GenAiInstrumentation) startModelSpan(ctx context.Context, info *tracing.SpanInfo) (context.Context, tracing.Span) {
	prefix, model := genai.SplitModelName(info.Name())
	provider := genai.DeriveProviderName(prefix)
	request := asModelRequest(info.Input())

	metricAttrs := []attribute.KeyValue{
		attribute.String(genai.AttrOperationName, genai.OperationChat),
		attribute.String(genai.AttrRequestModel, model),
	}
	if provider != "" {
		metricAttrs = append(metricAttrs, attribute.String(genai.AttrProviderName, provider))
	}
	attrs := append(labelAttrs(info), metricAttrs...)
	if request != nil {
		attrs = append(attrs, requestConfigAttributes(request)...)
	}

	start := time.Now()
	ctx, s := g.startSpan(ctx, info, genai.OperationChat+" "+model, oteltrace.SpanKindClient, attrs)
	s.model = &modelCall{request: request, metricAttrs: metricAttrs, start: start}
	return ctx, s
}

// startToolSpan opens an execute_tool INTERNAL span for a tool action.
func (g *GenAiInstrumentation) startToolSpan(ctx context.Context, info *tracing.SpanInfo) (context.Context, tracing.Span) {
	attrs := append(labelAttrs(info),
		attribute.String(genai.AttrOperationName, genai.OperationExecuteTool),
		attribute.String(genai.AttrToolName, info.Name()),
		attribute.String(genai.AttrToolType, "function"),
	)
	return g.startSpan(ctx, info, genai.OperationExecuteTool+" "+info.Name(), oteltrace.SpanKindInternal, attrs)
}

// startGenericSpan opens a plain INTERNAL span for action types with no GenAI
// mapping (flow, util, ...), keeping the trace tree connected.
func (g *GenAiInstrumentation) startGenericSpan(ctx context.Context, info *tracing.SpanInfo, subtype string) (context.Context, tracing.Span) {
	attrs := labelAttrs(info)
	if subtype != "" {
		attrs = append(attrs, attribute.String(genai.AttrGenkitActionType, subtype))
	}
	return g.startSpan(ctx, info, info.Name(), oteltrace.SpanKindInternal, attrs)
}
