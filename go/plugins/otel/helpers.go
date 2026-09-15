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
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"reflect"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/tracing"
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

// asCommonConfig extracts GenerationCommonConfig from a request config that is
// either the common config itself or a provider config that embeds/JSON-matches
// it. It round-trips through JSON so provider-specific configs (which share the
// common field names/tags) map without a hard dependency on their types.
func asCommonConfig(config any) *ai.GenerationCommonConfig {
	if config == nil {
		return nil
	}
	if c, ok := config.(*ai.GenerationCommonConfig); ok {
		return c
	}
	if c, ok := config.(ai.GenerationCommonConfig); ok {
		return &c
	}
	b, err := json.Marshal(config)
	if err != nil {
		return nil
	}
	var c ai.GenerationCommonConfig
	if err := json.Unmarshal(b, &c); err != nil {
		return nil
	}
	return &c
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

// errorTypeOf reports the error type for the error.type attribute: the concrete
// Go type name (e.g. "*errors.errorString"), mirroring JS using the error name.
func errorTypeOf(err error) string {
	if err == nil {
		return ""
	}
	t := reflect.TypeOf(err)
	if t == nil {
		return "error"
	}
	return t.String()
}

// maybeWarnNotRecording warns once if the SDK is not collecting: when no
// TracerProvider is registered, OTel returns a non-recording span, so telemetry
// is silently dropped. Surface that instead of failing quietly.
func (g *GenAiInstrumentation) maybeWarnNotRecording(span oteltrace.Span) {
	if span.IsRecording() {
		return
	}
	g.warnOnce.Do(func() {
		slog.Warn("GenAiInstrumentation is configured but no OpenTelemetry SDK is " +
			"recording, so GenAI telemetry will not be exported. Initialize the " +
			"OTel SDK before constructing Genkit.")
	})
}

// setJSONAttribute records value on span as a JSON string under key, skipping
// nil values.
func setJSONAttribute(span oteltrace.Span, key string, value any) {
	if value == nil {
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
