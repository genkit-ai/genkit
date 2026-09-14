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

// runModelSpan opens a gen_ai chat CLIENT span for a model action, records the
// request config and (after next) the response attributes, content, and metrics.
func (g *GenAiInstrumentation) runModelSpan(ctx context.Context, info *tracing.SpanInfo, next tracing.NextFunc) (any, error) {
	prefix, model := genai.SplitModelName(info.Name())
	provider := genai.DeriveProviderName(prefix)
	request := asModelRequest(info.Input())

	attrs := []attribute.KeyValue{
		attribute.String(genai.AttrOperationName, genai.OperationChat),
		attribute.String(genai.AttrRequestModel, model),
	}
	if provider != "" {
		attrs = append(attrs, attribute.String(genai.AttrProviderName, provider))
	}
	if request != nil {
		attrs = append(attrs, requestConfigAttributes(request)...)
	}

	// Base metric attributes shared by both histograms: low cardinality only.
	metricAttrs := []attribute.KeyValue{
		attribute.String(genai.AttrOperationName, genai.OperationChat),
		attribute.String(genai.AttrRequestModel, model),
	}
	if provider != "" {
		metricAttrs = append(metricAttrs, attribute.String(genai.AttrProviderName, provider))
	}

	start := time.Now()
	ctx, span := g.tracer().Start(ctx, genai.OperationChat+" "+model,
		oteltrace.WithSpanKind(oteltrace.SpanKindClient),
		oteltrace.WithAttributes(attrs...))
	defer span.End()
	g.maybeWarnNotRecording(span)

	out, err := next(ctx, spanHandle(span))
	if err != nil {
		g.recordError(span, err)
		if g.emitMetrics {
			g.recordModelMetrics(ctx, start, metricAttrs, nil, errorTypeOf(err))
		}
		return out, err
	}

	response := asModelResponse(out)
	if response != nil {
		addResponseAttributes(span, response, false)
	}
	if g.contentMode != genai.NoContent {
		g.recordContent(ctx, span, request, response)
	}
	g.maybeCaptureActionIO(span, info.Input(), out)
	if g.emitMetrics {
		g.recordModelMetrics(ctx, start, metricAttrs, response, "")
	}
	return out, nil
}

// runToolSpan opens an execute_tool INTERNAL span for a tool action.
func (g *GenAiInstrumentation) runToolSpan(ctx context.Context, info *tracing.SpanInfo, next tracing.NextFunc) (any, error) {
	attrs := []attribute.KeyValue{
		attribute.String(genai.AttrOperationName, genai.OperationExecuteTool),
		attribute.String(genai.AttrToolName, info.Name()),
		attribute.String(genai.AttrToolType, "function"),
	}
	ctx, span := g.tracer().Start(ctx, genai.OperationExecuteTool+" "+info.Name(),
		oteltrace.WithSpanKind(oteltrace.SpanKindInternal),
		oteltrace.WithAttributes(attrs...))
	defer span.End()
	g.maybeWarnNotRecording(span)

	out, err := next(ctx, spanHandle(span))
	if err != nil {
		g.recordError(span, err)
		return out, err
	}
	g.maybeCaptureActionIO(span, info.Input(), out)
	return out, nil
}

// runGenericSpan opens a plain INTERNAL span for action types with no GenAI
// mapping (flow, util, ...), keeping the trace tree connected.
func (g *GenAiInstrumentation) runGenericSpan(ctx context.Context, info *tracing.SpanInfo, next tracing.NextFunc, subtype string) (any, error) {
	var opts []oteltrace.SpanStartOption
	opts = append(opts, oteltrace.WithSpanKind(oteltrace.SpanKindInternal))
	if subtype != "" {
		opts = append(opts, oteltrace.WithAttributes(attribute.String(genai.AttrGenkitActionType, subtype)))
	}
	ctx, span := g.tracer().Start(ctx, info.Name(), opts...)
	defer span.End()
	g.maybeWarnNotRecording(span)

	out, err := next(ctx, spanHandle(span))
	if err != nil {
		g.recordError(span, err)
		return out, err
	}
	g.maybeCaptureActionIO(span, info.Input(), out)
	return out, nil
}
