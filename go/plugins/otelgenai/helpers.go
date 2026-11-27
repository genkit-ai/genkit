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
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/otelgenai/internal/genai"
)

// asModelRequest returns the input as an *ai.ModelRequest when it is one.
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
// an unexpected type cannot fail the whole decode. Numbers decode as
// json.Number so integers above 2^53 (e.g. a seed) stay exact. It runs once
// per model call, next to a network round trip, so the marshal cost is
// negligible.
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
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return nil // Not a JSON object.
	}
	return m
}

// configNumber is a numeric config value. Values decoded by configMap are
// json.Number; a map[string]any config passed through as is can hold Go
// numeric types.
type configNumber struct{ v any }

func (n configNumber) float() (float64, bool) {
	switch v := n.v.(type) {
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	}
	return 0, false
}

func (n configNumber) int() (int64, bool) {
	switch v := n.v.(type) {
	case json.Number:
		if i, err := v.Int64(); err == nil {
			return i, true
		}
	case int:
		return int64(v), true
	case int64:
		return v, true
	}
	// Fractional or out-of-range values truncate, as a float would.
	f, ok := n.float()
	return int64(f), ok
}

// lookupNumber returns the first numeric value present under keys.
func lookupNumber(m map[string]any, keys []string) (configNumber, bool) {
	for _, k := range keys {
		n := configNumber{m[k]}
		if _, ok := n.float(); ok {
			return n, true
		}
	}
	return configNumber{}, false
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

// recordError sets the error status, error.type attribute, and exception event.
func (g *Instrumentation) recordError(span oteltrace.Span, err error) {
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

// hasOwnSpanContext reports whether span got span ids of its own. Without an
// SDK, the global tracer returns a non-recording span that carries the
// parent's span context from ctx: invalid at the root, but the parent's ids
// (valid) under a remote parent extracted from a traceparent header. Reporting
// those would give every Genkit span in the request the parent's span id.
func hasOwnSpanContext(parent oteltrace.SpanContext, span oteltrace.Span) bool {
	sc := span.SpanContext()
	return sc.IsValid() && !(parent.IsValid() && sc.SpanID() == parent.SpanID())
}

// maybeWarnNotRecording warns once when no OpenTelemetry SDK is installed, so
// GenAI telemetry is silently dropped. A span the SDK sampled out is also
// non-recording but gets its own ids, so this does not fire for a working,
// sampled setup.
func (g *Instrumentation) maybeWarnNotRecording(ctx context.Context, ownIDs bool) {
	if ownIDs {
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
