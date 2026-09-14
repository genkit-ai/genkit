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
	otellog "go.opentelemetry.io/otel/log"
	otellogglobal "go.opentelemetry.io/otel/log/global"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// recordContent attaches spec-shaped message content to the span (SPAN modes)
// and/or emits a dedicated operation.details event (EVENT modes).
func (g *GenAiInstrumentation) recordContent(ctx context.Context, span oteltrace.Span, request *ai.ModelRequest, response *ai.ModelResponse) {
	var input *genai.NormalizedMessages
	if request != nil {
		nm := genai.NormalizeMessages(request.Messages)
		input = &nm
	}
	var outputMessages []map[string]any
	if msg := resolveMessage(response); msg != nil {
		reason := resolveFinishReasons(response, false)[0]
		outputMessages = append(outputMessages, genai.MapOutputMessage(msg, reason))
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
func (g *GenAiInstrumentation) emitOperationDetails(ctx context.Context, input *genai.NormalizedMessages, outputMessages []map[string]any) {
	var rec otellog.Record
	rec.SetTimestamp(time.Now())
	rec.SetEventName(genai.OperationDetailsEvent)
	if input != nil {
		rec.AddAttributes(attribute.String(genai.AttrInputMessages, jsonString(input.Messages)))
		if len(input.SystemInstructions) > 0 {
			rec.AddAttributes(attribute.String(genai.AttrSystemInstructions, jsonString(input.SystemInstructions)))
		}
	}
	if len(outputMessages) > 0 {
		rec.AddAttributes(attribute.String(genai.AttrOutputMessages, jsonString(outputMessages)))
	}
	otellogglobal.Logger(g.scopeName).Emit(ctx, rec)
}

// recordModelMetrics records the token-usage and operation-duration metrics for
// a model call. errorType is non-empty for a failed call.
func (g *GenAiInstrumentation) recordModelMetrics(ctx context.Context, start time.Time, base []attribute.KeyValue, response *ai.ModelResponse, errorType string) {
	m := g.genAiMetrics()
	if m == nil {
		return
	}
	if response != nil && response.Usage != nil {
		var in, out *int
		if response.Usage.InputTokens != 0 {
			v := response.Usage.InputTokens
			in = &v
		}
		if response.Usage.OutputTokens != 0 {
			v := response.Usage.OutputTokens
			out = &v
		}
		m.RecordTokenUsage(ctx, base, in, out)
	}
	seconds := time.Since(start).Seconds()
	attrs := base
	if errorType != "" {
		attrs = append(append([]attribute.KeyValue{}, base...), attribute.String(genai.AttrErrorType, errorType))
	}
	m.RecordDuration(ctx, seconds, attrs)
}

// maybeCaptureActionIO records raw Genkit input/output on span as genkit.* JSON
// attributes when CaptureActionIO is enabled. Kept out of the reserved gen_ai.*
// namespace so GenAI-aware backends do not misrender it.
func (g *GenAiInstrumentation) maybeCaptureActionIO(span oteltrace.Span, input, output any) {
	if !g.captureActionIO {
		return
	}
	setJSONAttribute(span, genai.AttrGenkitInput, input)
	setJSONAttribute(span, genai.AttrGenkitOutput, output)
}
