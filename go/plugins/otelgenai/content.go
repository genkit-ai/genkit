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

package otelgenai

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	otellog "go.opentelemetry.io/otel/log"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/otelgenai/internal/genai"
)

// recordContent attaches spec-shaped message content to the span (SPAN modes)
// and/or emits a dedicated operation.details event (EVENT modes).
func (g *Instrumentation) recordContent(ctx context.Context, span oteltrace.Span, request *ai.ModelRequest, response *ai.ModelResponse, failed bool) {
	var input *genai.NormalizedMessages
	if request != nil {
		nm := genai.NormalizeMessages(request.Messages)
		input = &nm
	}
	var outputMessages []map[string]any
	if response != nil && response.Message != nil {
		// resolveFinishReasons always returns exactly one element.
		reason := resolveFinishReasons(response, failed)[0]
		outputMessages = append(outputMessages, genai.MapOutputMessage(response.Message, reason))
	}

	if g.captureOnSpan {
		if input != nil {
			setJSONAttribute(span, genai.AttrInputMessages, input.Messages)
			if len(input.SystemInstructions) > 0 {
				setJSONAttribute(span, genai.AttrSystemInstructions, input.SystemInstructions)
			}
		}
		if len(outputMessages) > 0 {
			setJSONAttribute(span, genai.AttrOutputMessages, outputMessages)
		}
	}

	if g.captureOnEvent {
		g.emitOperationDetails(ctx, input, outputMessages)
	}
}

// emitOperationDetails emits a single gen_ai operation.details log event,
// correlated to the active span via ctx, carrying the captured content.
//
// Unlike the span attributes (JSON strings, which is what trace backends
// render), the event carries content as structured MAP / SLICE values, as the
// spec defines for this event.
func (g *Instrumentation) emitOperationDetails(ctx context.Context, input *genai.NormalizedMessages, outputMessages []map[string]any) {
	var rec otellog.Record
	rec.SetTimestamp(time.Now())
	rec.SetEventName(genai.OperationDetailsEvent)
	add := func(key string, v any) {
		rec.AddAttributes(attribute.KeyValue{Key: attribute.Key(key), Value: structuredValue(v)})
	}
	if input != nil {
		add(genai.AttrInputMessages, input.Messages)
		if len(input.SystemInstructions) > 0 {
			add(genai.AttrSystemInstructions, input.SystemInstructions)
		}
	}
	if len(outputMessages) > 0 {
		add(genai.AttrOutputMessages, outputMessages)
	}
	g.logger.Emit(ctx, rec)
}

// recordModelMetrics records the token-usage and operation-duration metrics for
// a model call. Token usage is recorded for a failed call too when it returned
// a response: those tokens were billed. The duration point of a failed call
// carries error.type.
func (g *Instrumentation) recordModelMetrics(ctx context.Context, start time.Time, base []attribute.KeyValue, response *ai.ModelResponse, err error) {
	if response != nil && response.Usage != nil {
		g.metrics.RecordTokenUsage(ctx, base, response.Usage.InputTokens, response.Usage.OutputTokens)
	}
	attrs := base
	if err != nil {
		attrs = append(append([]attribute.KeyValue{}, base...), attribute.String(genai.AttrErrorType, errorTypeOf(err)))
	}
	g.metrics.RecordDuration(ctx, time.Since(start).Seconds(), attrs)
}

// maybeCaptureActionIO records raw Genkit input/output on span as genkit.* JSON
// attributes when CaptureActionIO is enabled. Kept out of the reserved gen_ai.*
// namespace so GenAI-aware backends do not misrender it.
func (g *Instrumentation) maybeCaptureActionIO(span oteltrace.Span, input, output any) {
	if !g.captureActionIO {
		return
	}
	setJSONAttribute(span, genai.AttrGenkitInput, redactInlineMedia(input))
	setJSONAttribute(span, genai.AttrGenkitOutput, redactInlineMedia(output))
}

// redactInlineMedia replaces inline data: media in a model request or response
// with size markers, so a multi-megabyte base64 image is not recorded verbatim:
// it would get truncated mid-JSON by attribute limits and can push the OTLP
// batch past the receiver's size cap, failing every span in it. Other values
// (e.g. flow inputs) have no known shape and pass through.
func redactInlineMedia(v any) any {
	switch v := v.(type) {
	case *ai.ModelRequest:
		if v == nil {
			return v
		}
		if msgs, changed := genai.RedactMessages(v.Messages); changed {
			cp := *v
			cp.Messages = msgs
			return &cp
		}
	case *ai.ModelResponse:
		if v == nil {
			return v
		}
		if msg, changed := genai.RedactMessage(v.Message); changed {
			cp := *v
			cp.Message = msg
			return &cp
		}
	}
	return v
}
