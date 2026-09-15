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

// Package otel provides GenAiInstrumentation, an OpenTelemetry GenAI
// semantic-conventions provider for Genkit.
//
// The application owns SDK setup: configure a TracerProvider / MeterProvider /
// LoggerProvider (e.g. via the OTel SDK) before constructing Genkit. When no
// provider is configured, go.opentelemetry.io/otel returns non-recording spans
// and no-op instruments, so this provider is effectively inert.
//
// Wire it up with [tracing.ConfigureInstrumentation]:
//
//	tracing.ConfigureInstrumentation(otel.NewGenAiInstrumentation(otel.GenAiInstrumentationOptions{
//		ContentCapturingMode: otel.SpanOnly,
//		EmitToolSpans:        true,
//	}))
//	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
//
// It composes with the built-in dev instrumentation, which feeds the Developer
// UI on a separate pipeline.
//
// See the spec:
// https://github.com/open-telemetry/semantic-conventions-genai
package otel

import (
	"context"
	"log/slog"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// GenAiInstrumentationOptions configures a [GenAiInstrumentation].
type GenAiInstrumentationOptions struct {
	// ContentCapturingMode selects where spec-shaped GenAI message content
	// (gen_ai.system_instructions, gen_ai.input.messages, gen_ai.output.messages)
	// is recorded. Content may contain PII, so the default is NoContent. When
	// the zero value is passed, the env var CaptureContentEnvVar is consulted;
	// an explicit value here overrides it.
	ContentCapturingMode genai.ContentCapturingMode

	// CaptureActionIO records raw Genkit action input/output as genkit.input /
	// genkit.output JSON attributes on every span (model, tool, flow, etc.).
	// Independent of ContentCapturingMode. May contain PII, off by default.
	CaptureActionIO bool

	// EmitToolSpans emits execute_tool spans for tool actions. Off by default.
	EmitToolSpans bool

	// DisableMetrics turns off the spec's GenAI client metrics (token usage,
	// operation duration). Metrics are on by default (the zero value): low
	// cardinality and cheap.
	DisableMetrics bool

	// ScopeName is the instrumentation scope name for the tracer/logger/meter.
	// Defaults to "genkit-genai".
	ScopeName string

	// Tracer optionally injects a tracer (escape hatch). When nil the tracer is
	// resolved from the global TracerProvider.
	Tracer oteltrace.Tracer

	// Meter optionally injects a meter (escape hatch). When nil the meter is
	// resolved from the global MeterProvider.
	Meter metric.Meter
}

// GenAiInstrumentation is a [tracing.Instrumentation] that emits OpenTelemetry
// telemetry following the OTel GenAI semantic conventions.
type GenAiInstrumentation struct {
	contentMode     genai.ContentCapturingMode
	captureOnSpan   bool
	captureOnEvent  bool
	captureActionIO bool
	emitToolSpans   bool
	emitMetrics     bool
	scopeName       string

	injectedTracer oteltrace.Tracer
	injectedMeter  metric.Meter

	metricsOnce sync.Once
	metrics     *genai.Metrics

	warnOnce sync.Once
}

// NewGenAiInstrumentation builds a GenAiInstrumentation from options.
func NewGenAiInstrumentation(opts GenAiInstrumentationOptions) *GenAiInstrumentation {
	mode := opts.ContentCapturingMode
	if mode == "" {
		mode = contentCapturingModeFromEnv()
	}
	scope := opts.ScopeName
	if scope == "" {
		scope = "genkit-genai"
	}
	return &GenAiInstrumentation{
		contentMode:     mode,
		captureOnSpan:   mode == genai.SpanOnly || mode == genai.SpanAndEvent,
		captureOnEvent:  mode == genai.EventOnly || mode == genai.SpanAndEvent,
		captureActionIO: opts.CaptureActionIO,
		emitToolSpans:   opts.EmitToolSpans,
		emitMetrics:     !opts.DisableMetrics,
		scopeName:       scope,
		injectedTracer:  opts.Tracer,
		injectedMeter:   opts.Meter,
	}
}

func contentCapturingModeFromEnv() genai.ContentCapturingMode {
	mode, ok := genai.ParseContentCapturingMode(getenv(genai.CaptureContentEnvVar))
	if !ok {
		slog.Warn("invalid content capturing mode env var; defaulting to NO_CONTENT",
			"env", genai.CaptureContentEnvVar)
	}
	return mode
}

func (g *GenAiInstrumentation) tracer() oteltrace.Tracer {
	if g.injectedTracer != nil {
		return g.injectedTracer
	}
	// Resolve lazily so the app's SDK setup, which may run after construction,
	// is picked up. otel.Tracer reads the global provider on each call.
	return otel.Tracer(g.scopeName, oteltrace.WithInstrumentationVersion(genai.SemConvVersion))
}

func (g *GenAiInstrumentation) genAiMetrics() *genai.Metrics {
	g.metricsOnce.Do(func() {
		meter := g.injectedMeter
		if meter == nil {
			meter = otel.Meter(g.scopeName, metric.WithInstrumentationVersion(genai.SemConvVersion))
		}
		m, err := genai.NewMetrics(meter)
		if err != nil {
			slog.Warn("failed to create GenAI metrics; metrics disabled", "error", err)
			return
		}
		g.metrics = m
	})
	return g.metrics
}

// RunInNewSpan dispatches on the Genkit action type and encodes the span using
// the OTel GenAI conventions.
func (g *GenAiInstrumentation) RunInNewSpan(ctx context.Context, info *tracing.SpanInfo, next tracing.NextFunc) (any, error) {
	if info == nil {
		// Nothing to encode; keep the chain intact.
		return next(ctx, nil)
	}
	actionType := info.Subtype()
	if actionType == "" {
		actionType = info.Type()
	}
	switch actionType {
	case "model":
		return g.runModelSpan(ctx, info, next)
	case "tool":
		if g.emitToolSpans {
			return g.runToolSpan(ctx, info, next)
		}
		return g.runGenericSpan(ctx, info, next, actionType)
	default:
		return g.runGenericSpan(ctx, info, next, actionType)
	}
}

var _ tracing.Instrumentation = (*GenAiInstrumentation)(nil)
