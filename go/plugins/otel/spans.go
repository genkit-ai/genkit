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

	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// labelAttrs returns the span's telemetry labels as attributes, matching the
// default OTel instrumentation.
func labelAttrs(info *tracing.SpanInfo) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, len(info.Labels))
	for k, v := range info.Labels {
		attrs = append(attrs, attribute.String(k, v))
	}
	return attrs
}

// runModelSpan opens a gen_ai chat CLIENT span for a model action, records the
// request config and (after next) the response attributes, content, and metrics.
//
// A failed call can still return a response (e.g. output schema validation
// fails after the model already ran and billed tokens), so the response is
// recorded whenever present, not only on success.
func (g *GenAiInstrumentation) runModelSpan(ctx context.Context, info *tracing.SpanInfo, next tracing.NextFunc) (any, error) {
	prefix, model := genai.SplitModelName(info.Name())
	provider := genai.DeriveProviderName(prefix)
	request := asModelRequest(info.Input())

	// Base metric attributes shared by both histograms: low cardinality only.
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
	ctx, span := g.tracer.Start(ctx, genai.OperationChat+" "+model,
		oteltrace.WithSpanKind(oteltrace.SpanKindClient),
		oteltrace.WithAttributes(attrs...))
	defer span.End()
	g.maybeWarnNotRecording(ctx, span)

	out, err := next(ctx, spanHandle(span))
	failed := err != nil

	response := asModelResponse(out)
	if response != nil {
		addResponseAttributes(span, response, failed)
	}
	if g.contentMode != genai.NoContent {
		g.recordContent(ctx, span, request, response, failed)
	}
	g.maybeCaptureActionIO(span, info.Input(), out)
	if failed {
		g.recordError(span, err)
	}
	if g.metrics != nil {
		g.recordModelMetrics(ctx, start, metricAttrs, response, err)
	}
	return out, err
}

// runToolSpan opens an execute_tool INTERNAL span for a tool action.
func (g *GenAiInstrumentation) runToolSpan(ctx context.Context, info *tracing.SpanInfo, next tracing.NextFunc) (any, error) {
	attrs := append(labelAttrs(info),
		attribute.String(genai.AttrOperationName, genai.OperationExecuteTool),
		attribute.String(genai.AttrToolName, info.Name()),
		attribute.String(genai.AttrToolType, "function"),
	)
	ctx, span := g.tracer.Start(ctx, genai.OperationExecuteTool+" "+info.Name(),
		oteltrace.WithSpanKind(oteltrace.SpanKindInternal),
		oteltrace.WithAttributes(attrs...))
	defer span.End()
	g.maybeWarnNotRecording(ctx, span)

	out, err := next(ctx, spanHandle(span))
	return g.finishSpan(span, info, out, err)
}

// runGenericSpan opens a plain INTERNAL span for action types with no GenAI
// mapping (flow, util, ...), keeping the trace tree connected.
func (g *GenAiInstrumentation) runGenericSpan(ctx context.Context, info *tracing.SpanInfo, next tracing.NextFunc, subtype string) (any, error) {
	attrs := labelAttrs(info)
	if subtype != "" {
		attrs = append(attrs, attribute.String(genai.AttrGenkitActionType, subtype))
	}
	ctx, span := g.tracer.Start(ctx, info.Name(),
		oteltrace.WithSpanKind(oteltrace.SpanKindInternal),
		oteltrace.WithAttributes(attrs...))
	defer span.End()
	g.maybeWarnNotRecording(ctx, span)

	out, err := next(ctx, spanHandle(span))
	return g.finishSpan(span, info, out, err)
}

// finishSpan records action IO (including a partial output returned alongside
// an error) and the error status, then passes the result through.
func (g *GenAiInstrumentation) finishSpan(span oteltrace.Span, info *tracing.SpanInfo, out any, err error) (any, error) {
	g.maybeCaptureActionIO(span, info.Input(), out)
	if err != nil {
		g.recordError(span, err)
	}
	return out, err
}
