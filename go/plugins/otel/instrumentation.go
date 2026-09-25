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
//		ContentCapturingMode: genai.SpanOnly,
//		EmitToolSpans:        true,
//	}))
//	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
//
// Configuration is process-wide on purpose: Genkit actions (e.g. a model
// obtained straight from a plugin) can run without a Genkit instance, and they
// must be instrumented too.
//
// The provider replaces Genkit's default OpenTelemetry encoding (the genkit:*
// span attributes) and composes with the built-in dev instrumentation, which
// feeds the Developer UI on a separate pipeline. Do not combine it with the
// googlecloud or firebase telemetry plugins: those read the genkit:* attributes
// (their metrics stop) and their redaction does not cover gen_ai.* content.
//
// See the spec:
// https://github.com/open-telemetry/semantic-conventions-genai
package otel

import (
	"context"
	"log/slog"
	"sync"

	"go.opentelemetry.io/otel"
	otellog "go.opentelemetry.io/otel/log"
	otellogglobal "go.opentelemetry.io/otel/log/global"
	"go.opentelemetry.io/otel/metric"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal"
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

	tracer oteltrace.Tracer
	logger otellog.Logger
	// metrics is nil when disabled or when instrument creation failed.
	metrics *genai.Metrics

	warnOnce sync.Once
}

// NewGenAiInstrumentation builds a GenAiInstrumentation from options.
//
// The tracer, meter, and logger are resolved here, once. That still picks up
// an SDK installed later: the OTel globals hand out delegating instruments
// that switch to the real provider when it is set. Resolving per span instead
// would take the SDK provider's lock on every span start.
func NewGenAiInstrumentation(opts GenAiInstrumentationOptions) *GenAiInstrumentation {
	mode := opts.ContentCapturingMode
	if mode == "" {
		mode = contentCapturingModeFromEnv()
	}
	scope := opts.ScopeName
	if scope == "" {
		scope = "genkit-genai"
	}
	g := &GenAiInstrumentation{
		contentMode:     mode,
		captureOnSpan:   mode == genai.SpanOnly || mode == genai.SpanAndEvent,
		captureOnEvent:  mode == genai.EventOnly || mode == genai.SpanAndEvent,
		captureActionIO: opts.CaptureActionIO,
		emitToolSpans:   opts.EmitToolSpans,
		tracer:          opts.Tracer,
		logger: otellogglobal.Logger(scope,
			otellog.WithInstrumentationVersion(internal.Version),
			otellog.WithSchemaURL(genai.SchemaURL)),
	}
	if g.tracer == nil {
		g.tracer = otel.Tracer(scope,
			oteltrace.WithInstrumentationVersion(internal.Version),
			oteltrace.WithSchemaURL(genai.SchemaURL))
	}
	if !opts.DisableMetrics {
		meter := opts.Meter
		if meter == nil {
			meter = otel.Meter(scope,
				metric.WithInstrumentationVersion(internal.Version),
				metric.WithSchemaURL(genai.SchemaURL))
		}
		m, err := genai.NewMetrics(meter)
		if err != nil {
			slog.Warn("failed to create genai metrics, metrics disabled", "error", err)
		} else {
			g.metrics = m
		}
	}
	return g
}

// contentCapturingModeFromEnv reads the spec env var. It runs at construction
// with no context, hence slog rather than core/logger.
func contentCapturingModeFromEnv() genai.ContentCapturingMode {
	mode, ok := genai.ParseContentCapturingMode(getenv(genai.CaptureContentEnvVar))
	if !ok {
		slog.Warn("invalid content capturing mode, defaulting to NO_CONTENT",
			"env", genai.CaptureContentEnvVar)
	}
	return mode
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
	case string(api.ActionTypeModel):
		return g.runModelSpan(ctx, info, next)
	case string(api.ActionTypeTool), string(api.ActionTypeToolV2):
		// DefineTool registers tool.v2; plain "tool" is the legacy spelling.
		if g.emitToolSpans {
			return g.runToolSpan(ctx, info, next)
		}
	}
	return g.runGenericSpan(ctx, info, next, actionType)
}

var _ tracing.Instrumentation = (*GenAiInstrumentation)(nil)
