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

// Package genai holds the mapping from Genkit data to OpenTelemetry GenAI
// semantic conventions: attribute names, value vocabularies, message shapes,
// and the spec's client metrics. Names and values come from the generated
// semconv package wherever it has them, so a spec rename surfaces on upgrade.
//
// See the spec:
// https://github.com/open-telemetry/semantic-conventions-genai
package genai

import (
	"strings"

	semconv "go.opentelemetry.io/otel/semconv/v1.39.0"
)

// SchemaURL is the semantic-conventions schema the emitted telemetry follows.
const SchemaURL = semconv.SchemaURL

// Canonical gen_ai.* attribute names used by this instrumentation.
const (
	AttrOperationName = string(semconv.GenAIOperationNameKey)
	AttrProviderName  = string(semconv.GenAIProviderNameKey)

	AttrRequestModel            = string(semconv.GenAIRequestModelKey)
	AttrRequestTemperature      = string(semconv.GenAIRequestTemperatureKey)
	AttrRequestTopP             = string(semconv.GenAIRequestTopPKey)
	AttrRequestTopK             = string(semconv.GenAIRequestTopKKey)
	AttrRequestMaxTokens        = string(semconv.GenAIRequestMaxTokensKey)
	AttrRequestStopSequences    = string(semconv.GenAIRequestStopSequencesKey)
	AttrRequestFrequencyPenalty = string(semconv.GenAIRequestFrequencyPenaltyKey)
	AttrRequestPresencePenalty  = string(semconv.GenAIRequestPresencePenaltyKey)
	AttrRequestSeed             = string(semconv.GenAIRequestSeedKey)
	AttrRequestChoiceCount      = string(semconv.GenAIRequestChoiceCountKey)

	AttrOutputType = string(semconv.GenAIOutputTypeKey)

	AttrResponseFinishReasons = string(semconv.GenAIResponseFinishReasonsKey)

	// AttrTokenType distinguishes token-usage measurements: input vs output.
	AttrTokenType = string(semconv.GenAITokenTypeKey)

	AttrUsageInputTokens  = string(semconv.GenAIUsageInputTokensKey)
	AttrUsageOutputTokens = string(semconv.GenAIUsageOutputTokensKey)
	// Not in semconv v1.39.0 yet.
	AttrUsageReasoningOutputTokens = "gen_ai.usage.reasoning.output_tokens"
	AttrUsageCacheReadInputTokens  = "gen_ai.usage.cache_read.input_tokens"

	AttrToolName = string(semconv.GenAIToolNameKey)
	AttrToolType = string(semconv.GenAIToolTypeKey)

	// Content attributes (opt-in; may contain PII).
	AttrInputMessages      = string(semconv.GenAIInputMessagesKey)
	AttrOutputMessages     = string(semconv.GenAIOutputMessagesKey)
	AttrSystemInstructions = string(semconv.GenAISystemInstructionsKey)

	AttrErrorType = string(semconv.ErrorTypeKey)
)

// Non-reserved genkit.* attributes. Kept out of the gen_ai.* namespace so
// GenAI-aware backends (e.g. Jaeger's GenAI view) never try to render raw
// Genkit payloads as spec message content.
const (
	// AttrGenkitActionType keeps the span tree connected across Genkit action
	// types that have no GenAI mapping (flow, util, etc.).
	AttrGenkitActionType = "genkit.action.type"

	// Raw Genkit action input/output as JSON strings (opt-in; may contain PII).
	AttrGenkitInput  = "genkit.input"
	AttrGenkitOutput = "genkit.output"
)

// The values below are literals because genaiconv exposes them as vars and
// method results, not constants. TestSemconvValues pins them to genaiconv, so
// a spec rename still fails loudly on upgrade.

// Well-known values for gen_ai.operation.name.
const (
	OperationChat        = "chat"
	OperationExecuteTool = "execute_tool"
)

// Well-known values for gen_ai.token.type.
const (
	TokenTypeInput  = "input"
	TokenTypeOutput = "output"
)

// Well-known values for gen_ai.provider.name that Genkit plugins map to.
const (
	ProviderGCPGemini   = "gcp.gemini"
	ProviderGCPVertexAI = "gcp.vertex_ai"
	ProviderOpenAI      = "openai"
	ProviderAnthropic   = "anthropic"
)

// Canonical gen_ai.* metric instrument names.
const (
	MetricTokenUsage        = "gen_ai.client.token.usage"
	MetricOperationDuration = "gen_ai.client.operation.duration"
)

// OperationDetailsEvent is the dedicated event that carries prompt/response
// content independently of the span, per the spec.
const OperationDetailsEvent = "gen_ai.client.inference.operation.details"

// CaptureContentEnvVar is the spec's canonical opt-in env var for capturing
// message content.
const CaptureContentEnvVar = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

// ContentCapturingMode mirrors the OTel GenAI ContentCapturingMode: where
// captured message content is recorded.
//
// Content may contain PII and is often large, so the default is NoContent.
// EventOnly keeps structured content on a dedicated log event and is preferred
// for production; SpanOnly puts it on span attributes as a JSON string (easy to
// eyeball, but subject to backend attribute/envelope limits), best for
// development.
type ContentCapturingMode string

const (
	NoContent    ContentCapturingMode = "NO_CONTENT"
	SpanOnly     ContentCapturingMode = "SPAN_ONLY"
	EventOnly    ContentCapturingMode = "EVENT_ONLY"
	SpanAndEvent ContentCapturingMode = "SPAN_AND_EVENT"
)

// ParseContentCapturingMode parses the spec env var into a ContentCapturingMode.
// It accepts the spec's enum names (case-insensitive); unset/empty maps to
// NoContent. The second return is false for an unrecognized value so the caller
// can warn.
func ParseContentCapturingMode(raw string) (ContentCapturingMode, bool) {
	switch strings.ToUpper(strings.TrimSpace(raw)) {
	case "", string(NoContent):
		return NoContent, true
	case string(SpanOnly):
		return SpanOnly, true
	case string(EventOnly):
		return EventOnly, true
	case string(SpanAndEvent):
		return SpanAndEvent, true
	default:
		return NoContent, false
	}
}

// SplitModelName splits a fully qualified Genkit model name into (prefix, model).
//
// "googleai/gemini-flash-latest" -> ("googleai", "gemini-flash-latest").
// A name without a "/" yields an empty prefix and the name as the model.
func SplitModelName(name string) (prefix, model string) {
	i := strings.Index(name, "/")
	if i < 0 {
		return "", name
	}
	return name[:i], name[i+1:]
}

// DeriveProviderName derives gen_ai.provider.name from a Genkit model-name
// prefix.
//
// It maps known Genkit plugin prefixes to the spec's well-known provider names,
// and passes unknown prefixes through lowercased so custom plugins still get a
// discriminator. Returns "" when there is no prefix.
func DeriveProviderName(prefix string) string {
	switch strings.ToLower(prefix) {
	case "":
		return ""
	case "googleai", "google-genai", "google_genai":
		// Gemini API (AI Studio), distinct from Vertex AI.
		return ProviderGCPGemini
	case "vertexai", "vertex-ai", "vertex_ai":
		return ProviderGCPVertexAI
	case "openai":
		return ProviderOpenAI
	case "anthropic":
		return ProviderAnthropic
	default:
		return strings.ToLower(prefix)
	}
}

// MapFinishReason maps a Genkit finish reason to the GenAI finish_reasons value.
//
// failed selects the fallback for ambiguous reasons (other/unknown): "error"
// when the span failed, otherwise "stop".
func MapFinishReason(genkitReason string, failed bool) string {
	switch genkitReason {
	case "stop":
		return "stop"
	case "length":
		return "length"
	case "blocked":
		return "content_filter"
	case "interrupted":
		// No exact spec value; treat an interrupted turn as a normal stop.
		return "stop"
	default:
		// Covers other/unknown/"" and anything unrecognized.
		if failed {
			return "error"
		}
		return "stop"
	}
}

// DeriveOutputType derives gen_ai.output.type from an output format / content
// type.
//
// Returns "json" when JSON output was requested, "text" when a text format was
// requested, otherwise "" (omit the attribute).
func DeriveOutputType(format, contentType string) string {
	f := strings.ToLower(format)
	ct := strings.ToLower(contentType)
	if f == "json" || (ct != "" && strings.Contains(ct, "json")) {
		return "json"
	}
	if f == "text" || (ct != "" && strings.HasPrefix(ct, "text/")) {
		return "text"
	}
	return ""
}
