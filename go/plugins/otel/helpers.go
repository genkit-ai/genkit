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
	"encoding/json"
	"fmt"
	"os"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

func getenv(key string) string { return os.Getenv(key) }

// asModelRequest returns the input as an *ai.ModelRequest when it is one. The
// model span's input is an *ai.ModelRequest (or, at the util turn, a value that
// carries Messages); we only need the request shape, so a type assertion is
// enough here.
func asModelRequest(input any) *ai.ModelRequest {
	if req, ok := input.(*ai.ModelRequest); ok {
		return req
	}
	return nil
}

// asModelResponse returns the action output as an *ai.ModelResponse when it is
// one.
func asModelResponse(output any) *ai.ModelResponse {
	if resp, ok := output.(*ai.ModelResponse); ok {
		return resp
	}
	return nil
}

// configMap returns a request config as its JSON object form, or nil.
//
// Configs are Genkit's GenerationCommonConfig or a provider SDK's native type
// (googlegenai GenerateContentConfig, anthropic MessageNewParams, ...), which
// this package cannot depend on. The JSON form is the one shape they all
// share, and decoding into a map (rather than a typed struct) keeps key
// presence, so an explicit 0 is distinguishable from unset, and one field of
// an unexpected type cannot fail the whole decode. It runs once per model
// call, next to a network round trip, so the marshal cost is negligible.
func configMap(config any) map[string]any {
	if config == nil {
		return nil
	}
	if m, ok := config.(map[string]any); ok {
		return m
	}
	b, err := json.Marshal(config)
	if err != nil {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil // Not a JSON object.
	}
	return m
}

// lookupNumber returns the first numeric value present under keys.
func lookupNumber(m map[string]any, keys []string) (float64, bool) {
	for _, k := range keys {
		if v, ok := m[k].(float64); ok {
			return v, true
		}
	}
	return 0, false
}

// lookupStrings returns the first string list (or single string, as OpenAI's
// "stop" allows) present under keys.
func lookupStrings(m map[string]any, keys []string) []string {
	for _, k := range keys {
		switch v := m[k].(type) {
		case string:
			return []string{v}
		case []any:
			out := make([]string, 0, len(v))
			for _, s := range v {
				if s, ok := s.(string); ok {
					out = append(out, s)
				}
			}
			return out
		}
	}
	return nil
}

// resolveMessage resolves the response message. Modern plugins set Message
// directly; there is no candidates fallback in Go's ModelResponse.
func resolveMessage(response *ai.ModelResponse) *ai.Message {
	if response == nil {
		return nil
	}
	return response.Message
}

// spanHandle adapts an OTel span to the tracing.Span the dispatcher expects.
func spanHandle(span oteltrace.Span) tracing.Span {
	return &otelSpan{span: span}
}

// otelSpan is the tracing.Span handle over an OTel span, exposing the span's
// ids and mapping SetMetadata to genkit:metadata:* attributes.
type otelSpan struct {
	span oteltrace.Span
}

func (s *otelSpan) TraceInfo() tracing.TraceInfo {
	sc := s.span.SpanContext()
	if !sc.IsValid() {
		return tracing.TraceInfo{}
	}
	return tracing.TraceInfo{TraceID: sc.TraceID().String(), SpanID: sc.SpanID().String()}
}

func (s *otelSpan) SetMetadata(md map[string]string) {
	for k, v := range md {
		s.span.SetAttributes(attribute.String("genkit:metadata:"+k, v))
	}
}

// recordError sets the error status, error.type attribute, and exception event.
func (g *GenAiInstrumentation) recordError(span oteltrace.Span, err error) {
	span.SetStatus(codes.Error, err.Error())
	span.SetAttributes(attribute.String(genai.AttrErrorType, errorTypeOf(err)))
	span.RecordError(err)
}

// errorTypeOf reports the error.type value: the Genkit status name (e.g.
// "INVALID_ARGUMENT", "RESOURCE_EXHAUSTED"; "INTERNAL" when unclassified).
// It is low cardinality and independent of how the error was wrapped, which a
// Go type name is not (fmt.Errorf alone turns any error into *fmt.wrapError).
func errorTypeOf(err error) string {
	if err == nil {
		return ""
	}
	return string(status.Of(err))
}

// maybeWarnNotRecording warns once when no OpenTelemetry SDK is installed, so
// GenAI telemetry is silently dropped. Without an SDK, a span has an invalid
// span context. A span the SDK sampled out is also non-recording but keeps a
// valid context, so this does not fire for a working, sampled setup.
func (g *GenAiInstrumentation) maybeWarnNotRecording(ctx context.Context, span oteltrace.Span) {
	if span.SpanContext().IsValid() {
		return
	}
	g.warnOnce.Do(func() {
		logger.Warn(ctx, "genai instrumentation is configured but no opentelemetry sdk is installed, telemetry is not exported")
	})
}

// setJSONAttribute records value on span as a JSON string under key, skipping
// nil values, including typed nils (a failed action's zero output), which
// would otherwise encode as "null".
func setJSONAttribute(span oteltrace.Span, key string, value any) {
	if base.IsNil(value) {
		return
	}
	span.SetAttributes(attribute.String(key, jsonString(value)))
}

// jsonString encodes value as JSON, or an "Unable to encode" marker on failure.
func jsonString(value any) string {
	b, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("Unable to encode: %v", err)
	}
	return string(b)
}
