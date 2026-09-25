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

// Demonstrates the OpenTelemetry GenAI instrumentation plugin
// (go/plugins/otel). The app owns OTel SDK setup, configured from the standard
// OTEL_* env vars (the Go analog of JS's `new NodeSDK()`), defaulting to OTLP
// at localhost:4318. It runs a single generate call, which produces a gen_ai
// chat span you can inspect in Jaeger or the Dev UI.
//
// Local collector + Jaeger (see scripts/local-telemetry.ts):
//
//	export GEMINI_API_KEY=...
//	export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
//	go run .
//
// Genkit Dev UI, previewing exactly what your OTel instrumentation emits:
//
//	GEMINI_API_KEY=... genkit start --use-otel -- go run .
package main

import (
	"context"
	"fmt"
	"log"

	"go.opentelemetry.io/contrib/exporters/autoexport"
	"go.opentelemetry.io/otel"
	otellogglobal "go.opentelemetry.io/otel/log/global"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	genaiotel "github.com/firebase/genkit/go/plugins/otel"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

func main() {
	ctx := context.Background()

	// 1. The app owns the OTel SDK.
	shutdown := setupOTel(ctx)
	defer shutdown(ctx)

	// 2. Route Genkit telemetry through the GenAI instrumentation. SPAN_ONLY
	// captures message content on spans (development-friendly; may contain PII).
	tracing.ConfigureInstrumentation(genaiotel.NewGenAiInstrumentation(genaiotel.GenAiInstrumentationOptions{
		ContentCapturingMode: genai.SpanOnly,
		EmitToolSpans:        true,
	}))

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// One-shot generate: produces a gen_ai chat span (plus its enclosing
	// generate span) that you can inspect in Jaeger or the Dev UI. The deferred
	// shutdown flushes it before exit.
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("googleai/gemini-flash-latest"),
		ai.WithPrompt("Tell me a short, clean joke about OpenTelemetry."))
	if err != nil {
		log.Fatalf("failed to generate joke: %v", err)
	}
	fmt.Println(resp.Text())
}

// setupOTel initializes the OpenTelemetry SDK from the standard OTEL_* env vars,
// the Go analog of JS's `new NodeSDK()`. With nothing set it defaults to OTLP
// http/protobuf at localhost:4318 for traces, metrics, and logs; env vars
// (including OTEL_EXPORTER_OTLP_*_PROTOCOL=http/json) override that. Returns a
// shutdown func that flushes buffered telemetry before exit.
func setupOTel(ctx context.Context) func(context.Context) {
	spanExp, err := autoexport.NewSpanExporter(ctx)
	if err != nil {
		log.Fatalf("otel: %v", err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(spanExp))
	otel.SetTracerProvider(tp)

	reader, err := autoexport.NewMetricReader(ctx)
	if err != nil {
		log.Fatalf("otel: %v", err)
	}
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	otel.SetMeterProvider(mp)

	logExp, err := autoexport.NewLogExporter(ctx)
	if err != nil {
		log.Fatalf("otel: %v", err)
	}
	lp := sdklog.NewLoggerProvider(sdklog.WithProcessor(sdklog.NewBatchProcessor(logExp)))
	otellogglobal.SetLoggerProvider(lp)

	return func(ctx context.Context) {
		_ = tp.Shutdown(ctx)
		_ = mp.Shutdown(ctx)
		_ = lp.Shutdown(ctx)
	}
}
