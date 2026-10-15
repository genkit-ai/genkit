// Copyright 2026 Google LLC
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
	"log/slog"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	oteltrace "go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/otelgenai/internal/genai"
)

// defineToolCallingModel defines a model that requests the getWeather tool on
// its first turn and answers on the second.
func defineToolCallingModel(g *genkit.Genkit) *ai.ModelAction {
	return genkit.DefineModelAction(g, "test/weather-bot",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Tools: true, Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, _ any, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			last := req.Messages[len(req.Messages)-1]
			if last.Role != ai.RoleTool {
				return &ai.ModelResponse{
					Message: ai.NewMessage(ai.RoleModel, nil,
						ai.NewToolRequestPart(&ai.ToolRequest{Name: "getWeather", Input: "SF"})),
					FinishReason: ai.FinishReasonStop,
				}, nil
			}
			return &ai.ModelResponse{
				Message:      ai.NewModelTextMessage("sunny"),
				FinishReason: ai.FinishReasonStop,
				Usage:        &ai.GenerationUsage{InputTokens: 10, OutputTokens: 1},
			}, nil
		})
}

// TestGenkitToolAndModel drives real DefineTool / DefineModel actions through
// genkit.Generate, so span subtypes are the ones Genkit actually produces
// (tools register as tool.v2).
func TestGenkitToolAndModel(t *testing.T) {
	sr := setup(t, Options{EmitToolSpans: true})
	ctx := context.Background()
	g := genkit.Init(ctx)

	tool := genkit.DefineTool(g, "getWeather", "gets the weather",
		func(ctx *ai.ToolContext, city string) (string, error) { return "sunny in " + city, nil })
	model := defineToolCallingModel(g)

	if _, err := genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt("weather?"), ai.WithTools(tool)); err != nil {
		t.Fatal(err)
	}

	a := attrMap(findSpan(t, sr, "execute_tool getWeather"))
	if got := a[genai.AttrToolName].AsString(); got != "getWeather" {
		t.Errorf("tool name = %q, want getWeather", got)
	}
	if got := a[genai.AttrOperationName].AsString(); got != genai.OperationExecuteTool {
		t.Errorf("operation = %q, want %q", got, genai.OperationExecuteTool)
	}
	m := attrMap(findSpan(t, sr, "chat weather-bot"))
	if got := m[genai.AttrProviderName].AsString(); got != "test" {
		t.Errorf("provider = %q, want test", got)
	}
}

func TestTelemetryLabels(t *testing.T) {
	sr := setup(t, Options{EmitToolSpans: true})
	labels := map[string]string{"tenant": "acme"}

	for _, md := range []*tracing.SpanMetadata{
		{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model", TelemetryLabels: labels},
		{Name: "getWeather", Type: "action", Subtype: string(api.ActionTypeToolV2), TelemetryLabels: labels},
		{Name: "myFlow", Type: "action", Subtype: "flow", TelemetryLabels: labels},
	} {
		_, err := tracing.RunInNewSpan(context.Background(), md, modelRequest(),
			func(ctx context.Context, _ *ai.ModelRequest) (*ai.ModelResponse, error) { return modelResponse(), nil })
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"chat gemini-flash-latest", "execute_tool getWeather", "myFlow"} {
		if got := attrMap(findSpan(t, sr, name))["tenant"].AsString(); got != "acme" {
			t.Errorf("%s: tenant label = %q, want acme", name, got)
		}
	}
}

func TestWarnWhenNoSDK(t *testing.T) {
	tests := []struct {
		name      string
		provider  oteltrace.TracerProvider
		wantWarns int
	}{
		{"no sdk warns once", noop.NewTracerProvider(), 1},
		{"sampled-out sdk span does not warn", sdktrace.NewTracerProvider(sdktrace.WithSampler(sdktrace.NeverSample())), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			ctx := logger.WithContext(context.Background(), slog.New(slog.NewTextHandler(&buf, nil)))
			tracing.SetInstrumentation(New(Options{TracerProvider: tt.provider}))
			t.Cleanup(func() { tracing.SetInstrumentation() })

			for range 2 {
				_, err := tracing.RunInNewSpan(ctx, &tracing.SpanMetadata{Name: "myFlow", Type: "action", Subtype: "flow"}, "in",
					func(ctx context.Context, _ string) (string, error) { return "out", nil })
				if err != nil {
					t.Fatal(err)
				}
			}
			if got := strings.Count(buf.String(), "no opentelemetry sdk is installed"); got != tt.wantWarns {
				t.Errorf("warnings = %d, want %d; log:\n%s", got, tt.wantWarns, buf.String())
			}
		})
	}
}

// TestRemoteParentWithoutSDK covers a request carrying a traceparent with no
// SDK installed: the no-op span echoes the remote parent's context, which must
// not be reported as this span's ids (every Genkit span would share them).
func TestRemoteParentWithoutSDK(t *testing.T) {
	var buf bytes.Buffer
	ctx := logger.WithContext(context.Background(), slog.New(slog.NewTextHandler(&buf, nil)))
	const parentSpanID = "00f067aa0ba902b7"
	ctx = propagation.TraceContext{}.Extract(ctx, propagation.MapCarrier{
		"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-" + parentSpanID + "-01",
	})
	tracing.SetInstrumentation(New(Options{TracerProvider: noop.NewTracerProvider()}))
	t.Cleanup(func() { tracing.SetInstrumentation() })

	var spanIDs []string
	ctx = tracing.WithTelemetryCallback(ctx, func(_, spanID string) { spanIDs = append(spanIDs, spanID) })
	for range 2 {
		_, err := tracing.RunInNewSpan(ctx, &tracing.SpanMetadata{Name: "myFlow", Type: "action", Subtype: "flow"}, "in",
			func(ctx context.Context, _ string) (string, error) { return "out", nil })
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(spanIDs) != 2 || spanIDs[0] == parentSpanID || spanIDs[0] == spanIDs[1] {
		t.Errorf("span ids = %v, want distinct ids other than the parent %s", spanIDs, parentSpanID)
	}
	if got := strings.Count(buf.String(), "no opentelemetry sdk is installed"); got != 1 {
		t.Errorf("warnings = %d, want 1", got)
	}
}
