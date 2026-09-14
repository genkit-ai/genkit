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
	"go.opentelemetry.io/otel/attribute"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// requestConfigAttributes maps a model request's config to gen_ai.request.*
// attributes. It reads the config through the common shape; provider-specific
// configs that embed GenerationCommonConfig are also handled.
func requestConfigAttributes(request *ai.ModelRequest) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	cfg := asCommonConfig(request.Config)
	if cfg != nil {
		if cfg.Temperature != 0 {
			attrs = append(attrs, attribute.Float64(genai.AttrRequestTemperature, cfg.Temperature))
		}
		if cfg.TopP != 0 {
			attrs = append(attrs, attribute.Float64(genai.AttrRequestTopP, cfg.TopP))
		}
		if cfg.TopK != 0 {
			attrs = append(attrs, attribute.Int(genai.AttrRequestTopK, cfg.TopK))
		}
		if cfg.MaxOutputTokens != 0 {
			attrs = append(attrs, attribute.Int(genai.AttrRequestMaxTokens, cfg.MaxOutputTokens))
		}
		if len(cfg.StopSequences) > 0 {
			attrs = append(attrs, attribute.StringSlice(genai.AttrRequestStopSequences, cfg.StopSequences))
		}
	}
	if request.Output != nil {
		if ot := genai.DeriveOutputType(request.Output.Format, request.Output.ContentType); ot != "" {
			attrs = append(attrs, attribute.String(genai.AttrOutputType, ot))
		}
	}
	return attrs
}

// addResponseAttributes records finish reasons and token usage on the span.
func addResponseAttributes(span oteltrace.Span, response *ai.ModelResponse, failed bool) {
	if reasons := resolveFinishReasons(response, failed); len(reasons) > 0 {
		span.SetAttributes(attribute.StringSlice(genai.AttrResponseFinishReasons, reasons))
	}
	usage := response.Usage
	if usage == nil {
		return
	}
	if usage.InputTokens != 0 {
		span.SetAttributes(attribute.Int(genai.AttrUsageInputTokens, usage.InputTokens))
	}
	if usage.OutputTokens != 0 {
		span.SetAttributes(attribute.Int(genai.AttrUsageOutputTokens, usage.OutputTokens))
	}
	if usage.ThoughtsTokens != 0 {
		span.SetAttributes(attribute.Int(genai.AttrUsageReasoningOutputTokens, usage.ThoughtsTokens))
	}
	if usage.CachedContentTokens != 0 {
		span.SetAttributes(attribute.Int(genai.AttrUsageCacheReadInputTokens, usage.CachedContentTokens))
	}
}

// resolveFinishReasons maps the response finish reason to the GenAI vocabulary.
// A turn ending in tool calls reports tool_calls, following the OpenAI GenAI
// profile: it is the more informative signal for consumers.
func resolveFinishReasons(response *ai.ModelResponse, failed bool) []string {
	if msg := resolveMessage(response); msg != nil && genai.HasToolRequestPart(msg.Content) {
		return []string{"tool_calls"}
	}
	return []string{genai.MapFinishReason(string(response.FinishReason), failed)}
}
