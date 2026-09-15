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
	"errors"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// setup installs an in-memory OTel span recorder and configures the GenAI
// instrumentation, returning the recorder and restoring global state on cleanup.
func setup(t *testing.T, opts GenAiInstrumentationOptions) *tracetest.SpanRecorder {
	t.Helper()
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)

	tracing.ConfigureInstrumentation(NewGenAiInstrumentation(opts))
	t.Cleanup(func() {
		tracing.ResetInstrumentation()
		otel.SetTracerProvider(prev)
	})
	return sr
}

// attrMap collapses a span's attributes into a map for easy assertions.
func attrMap(span sdktrace.ReadOnlySpan) map[string]attribute.Value {
	m := map[string]attribute.Value{}
	for _, kv := range span.Attributes() {
		m[string(kv.Key)] = kv.Value
	}
	return m
}

func findSpan(t *testing.T, sr *tracetest.SpanRecorder, name string) sdktrace.ReadOnlySpan {
	t.Helper()
	for _, s := range sr.Ended() {
		if s.Name() == name {
			return s
		}
	}
	t.Fatalf("span %q not found; got %v", name, spanNames(sr))
	return nil
}

func spanNames(sr *tracetest.SpanRecorder) []string {
	var names []string
	for _, s := range sr.Ended() {
		names = append(names, s.Name())
	}
	return names
}

func modelRequest() *ai.ModelRequest {
	return &ai.ModelRequest{
		Messages: []*ai.Message{
			ai.NewSystemTextMessage("be nice"),
			ai.NewUserTextMessage("hello"),
		},
		Config: &ai.GenerationCommonConfig{Temperature: 0.5, MaxOutputTokens: 100},
	}
}

func modelResponse() *ai.ModelResponse {
	return &ai.ModelResponse{
		Message:      ai.NewModelTextMessage("hi there"),
		FinishReason: ai.FinishReasonStop,
		Usage:        &ai.GenerationUsage{InputTokens: 3, OutputTokens: 7},
	}
}

func TestModelSpan(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{})

	_, err := tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model"},
		modelRequest(),
		func(ctx context.Context, _ *ai.ModelRequest) (*ai.ModelResponse, error) {
			return modelResponse(), nil
		})
	if err != nil {
		t.Fatal(err)
	}

	span := findSpan(t, sr, "chat gemini-flash-latest")
	if span.SpanKind() != oteltrace.SpanKindClient {
		t.Errorf("span kind = %v, want client", span.SpanKind())
	}
	a := attrMap(span)
	if a[genai.AttrOperationName].AsString() != genai.OperationChat {
		t.Errorf("operation name = %v", a[genai.AttrOperationName].AsString())
	}
	if a[genai.AttrRequestModel].AsString() != "gemini-flash-latest" {
		t.Errorf("request model = %v", a[genai.AttrRequestModel].AsString())
	}
	if a[genai.AttrProviderName].AsString() != "gcp.gemini" {
		t.Errorf("provider = %v", a[genai.AttrProviderName].AsString())
	}
	if a[genai.AttrRequestTemperature].AsFloat64() != 0.5 {
		t.Errorf("temperature = %v", a[genai.AttrRequestTemperature].AsFloat64())
	}
	if a[genai.AttrRequestMaxTokens].AsInt64() != 100 {
		t.Errorf("max tokens = %v", a[genai.AttrRequestMaxTokens].AsInt64())
	}
	if got := a[genai.AttrResponseFinishReasons].AsStringSlice(); len(got) != 1 || got[0] != "stop" {
		t.Errorf("finish reasons = %v", got)
	}
	if a[genai.AttrUsageInputTokens].AsInt64() != 3 {
		t.Errorf("input tokens = %v", a[genai.AttrUsageInputTokens].AsInt64())
	}
	if a[genai.AttrUsageOutputTokens].AsInt64() != 7 {
		t.Errorf("output tokens = %v", a[genai.AttrUsageOutputTokens].AsInt64())
	}
}

func TestModelSpanToolCallsFinishReason(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{})

	resp := &ai.ModelResponse{
		Message: ai.NewMessage(ai.RoleModel, nil,
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "getWeather"})),
		FinishReason: ai.FinishReasonStop,
	}
	_, err := tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model"},
		modelRequest(),
		func(ctx context.Context, _ *ai.ModelRequest) (*ai.ModelResponse, error) {
			return resp, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	span := findSpan(t, sr, "chat gemini-flash-latest")
	got := attrMap(span)[genai.AttrResponseFinishReasons].AsStringSlice()
	if len(got) != 1 || got[0] != "tool_calls" {
		t.Errorf("finish reasons = %v, want [tool_calls]", got)
	}
}

func TestToolSpanDisabledByDefault(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{})

	runTool(t, "getWeather")

	span := findSpan(t, sr, "getWeather")
	if span.SpanKind() != oteltrace.SpanKindInternal {
		t.Errorf("span kind = %v, want internal", span.SpanKind())
	}
	if attrMap(span)[genai.AttrGenkitActionType].AsString() != "tool" {
		t.Errorf("action type attr = %v", attrMap(span)[genai.AttrGenkitActionType].AsString())
	}
}

func TestToolSpanEnabled(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{EmitToolSpans: true})

	runTool(t, "getWeather")

	span := findSpan(t, sr, "execute_tool getWeather")
	a := attrMap(span)
	if a[genai.AttrOperationName].AsString() != genai.OperationExecuteTool {
		t.Errorf("operation = %v", a[genai.AttrOperationName].AsString())
	}
	if a[genai.AttrToolName].AsString() != "getWeather" {
		t.Errorf("tool name = %v", a[genai.AttrToolName].AsString())
	}
	if a[genai.AttrToolType].AsString() != "function" {
		t.Errorf("tool type = %v", a[genai.AttrToolType].AsString())
	}
}

func TestGenericSpan(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{})

	_, err := tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: "myFlow", Type: "action", Subtype: "flow"},
		"in",
		func(ctx context.Context, _ string) (string, error) { return "out", nil })
	if err != nil {
		t.Fatal(err)
	}
	span := findSpan(t, sr, "myFlow")
	if span.SpanKind() != oteltrace.SpanKindInternal {
		t.Errorf("span kind = %v, want internal", span.SpanKind())
	}
	if attrMap(span)[genai.AttrGenkitActionType].AsString() != "flow" {
		t.Errorf("action type = %v", attrMap(span)[genai.AttrGenkitActionType].AsString())
	}
}

func TestContentCaptureNoContentByDefault(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{})
	runModel(t)
	a := attrMap(findSpan(t, sr, "chat gemini-flash-latest"))
	if _, ok := a[genai.AttrInputMessages]; ok {
		t.Error("input messages should be absent in NO_CONTENT mode")
	}
}

func TestContentCaptureSpanOnly(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{ContentCapturingMode: genai.SpanOnly})
	runModel(t)
	a := attrMap(findSpan(t, sr, "chat gemini-flash-latest"))
	if a[genai.AttrInputMessages].AsString() == "" {
		t.Error("expected input messages attribute")
	}
	if a[genai.AttrSystemInstructions].AsString() == "" {
		t.Error("expected system instructions attribute")
	}
	if a[genai.AttrOutputMessages].AsString() == "" {
		t.Error("expected output messages attribute")
	}
}

func TestCaptureActionIO(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{CaptureActionIO: true})
	runModel(t)
	a := attrMap(findSpan(t, sr, "chat gemini-flash-latest"))
	if a[genai.AttrGenkitInput].AsString() == "" {
		t.Error("expected genkit.input attribute")
	}
	if a[genai.AttrGenkitOutput].AsString() == "" {
		t.Error("expected genkit.output attribute")
	}
}

func TestModelSpanError(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{})

	wantErr := errors.New("boom")
	_, err := tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model"},
		modelRequest(),
		func(ctx context.Context, _ *ai.ModelRequest) (*ai.ModelResponse, error) {
			return nil, wantErr
		})
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
	span := findSpan(t, sr, "chat gemini-flash-latest")
	if attrMap(span)[genai.AttrErrorType].AsString() == "" {
		t.Error("expected error.type attribute on failed span")
	}
}

func runTool(t *testing.T, name string) {
	t.Helper()
	_, err := tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: name, Type: "action", Subtype: "tool"},
		"in",
		func(ctx context.Context, _ string) (string, error) { return "out", nil })
	if err != nil {
		t.Fatal(err)
	}
}

func runModel(t *testing.T) {
	t.Helper()
	_, err := tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model"},
		modelRequest(),
		func(ctx context.Context, _ *ai.ModelRequest) (*ai.ModelResponse, error) {
			return modelResponse(), nil
		})
	if err != nil {
		t.Fatal(err)
	}
}
