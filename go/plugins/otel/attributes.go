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

// Config keys per gen_ai.request.* attribute, in lookup order. Configs arrive
// as Genkit's GenerationCommonConfig or as a provider SDK's native type, so
// both the camelCase Genkit/Gemini names and the snake_case wire names of the
// Anthropic and OpenAI SDKs are recognized.
var (
	keysTemperature      = []string{"temperature"}
	keysTopP             = []string{"topP", "top_p"}
	keysTopK             = []string{"topK", "top_k"}
	keysMaxTokens        = []string{"maxOutputTokens", "max_tokens", "max_completion_tokens", "maxTokens"}
	keysStopSequences    = []string{"stopSequences", "stop_sequences", "stop"}
	keysFrequencyPenalty = []string{"frequencyPenalty", "frequency_penalty"}
	keysPresencePenalty  = []string{"presencePenalty", "presence_penalty"}
	keysSeed             = []string{"seed"}
	keysChoiceCount      = []string{"candidateCount", "n"}
)

// requestConfigAttributes maps a model request's config to gen_ai.request.*
// attributes. A param is recorded whenever the config carries it, including
// an explicit 0 (e.g. temperature 0 for deterministic sampling).
func requestConfigAttributes(request *ai.ModelRequest) []attribute.KeyValue {
	var attrs []attribute.KeyValue
	cfg := configMap(request.Config)
	addFloat := func(attr string, keys []string) {
		if v, ok := lookupNumber(cfg, keys); ok {
			attrs = append(attrs, attribute.Float64(attr, v))
		}
	}
	addInt := func(attr string, keys []string) {
		if v, ok := lookupNumber(cfg, keys); ok {
			attrs = append(attrs, attribute.Int64(attr, int64(v)))
		}
	}
	addFloat(genai.AttrRequestTemperature, keysTemperature)
	addFloat(genai.AttrRequestTopP, keysTopP)
	// top_k is a double in the spec; Gemini accepts fractional values.
	addFloat(genai.AttrRequestTopK, keysTopK)
	addInt(genai.AttrRequestMaxTokens, keysMaxTokens)
	if stop := lookupStrings(cfg, keysStopSequences); len(stop) > 0 {
		attrs = append(attrs, attribute.StringSlice(genai.AttrRequestStopSequences, stop))
	}
	addFloat(genai.AttrRequestFrequencyPenalty, keysFrequencyPenalty)
	addFloat(genai.AttrRequestPresencePenalty, keysPresencePenalty)
	addInt(genai.AttrRequestSeed, keysSeed)
	// The spec asks to omit choice.count when it is 1.
	if n, ok := lookupNumber(cfg, keysChoiceCount); ok && n != 1 {
		attrs = append(attrs, attribute.Int64(genai.AttrRequestChoiceCount, int64(n)))
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
	// GenerationUsage cannot tell "0" from "not reported". Input and output are
	// reported by every provider that sets Usage at all, so a 0 there is real;
	// reasoning and cache tokens are commonly absent, so 0 is treated as unset.
	span.SetAttributes(
		attribute.Int(genai.AttrUsageInputTokens, usage.InputTokens),
		attribute.Int(genai.AttrUsageOutputTokens, usage.OutputTokens),
	)
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
	if response == nil {
		return []string{genai.MapFinishReason("", failed)}
	}
	if msg := resolveMessage(response); msg != nil && genai.HasToolRequestPart(msg.Content) {
		return []string{"tool_calls"}
	}
	return []string{genai.MapFinishReason(string(response.FinishReason), failed)}
}
