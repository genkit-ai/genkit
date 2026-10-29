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

// Package otelgenai provides an OpenTelemetry GenAI semantic-conventions
// instrumentation for Genkit.
//
// The application owns SDK setup: configure a TracerProvider / MeterProvider /
// LoggerProvider (e.g. via the OTel SDK) before constructing Genkit. When no
// provider is configured, go.opentelemetry.io/otel returns non-recording spans
// and no-op instruments, so this provider is effectively inert.
//
// Wire it up with [tracing.SetInstrumentation]:
//
//	tracing.SetInstrumentation(otelgenai.New(&otelgenai.Options{
//		ContentCapturingMode: otelgenai.SpanOnly,
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
package otelgenai

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync"

	"go.opentelemetry.io/otel"
	otellog "go.opentelemetry.io/otel/log"
	otellogglobal "go.opentelemetry.io/otel/log/global"
	"go.opentelemetry.io/otel/metric"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal"
	"github.com/firebase/genkit/go/plugins/otelgenai/internal/genai"
)

// CaptureContentEnvVar is the spec's canonical opt-in env var for capturing
// message content. It is consulted when [Options.ContentCapturingMode] is unset.
const CaptureContentEnvVar = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

// ContentCapturingMode mirrors the OTel GenAI ContentCapturingMode: where
// captured message content is recorded.
//
// Content may contain PII and is often large, so the default is NoContent.
// EventOnly puts structured content on a dedicated log event and is preferred
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

// parseContentCapturingMode parses the spec env var. It accepts the spec's
// enum names (case-insensitive); unset/empty maps to NoContent. The second
// return is false for an unrecognized value so the caller can warn.
func parseContentCapturingMode(raw string) (ContentCapturingMode, bool) {
	switch m := ContentCapturingMode(strings.ToUpper(strings.TrimSpace(raw))); m {
	case "":
		return NoContent, true
	case NoContent, SpanOnly, EventOnly, SpanAndEvent:
		return m, true
	default:
		return NoContent, false
	}
}

// Options configures an [Instrumentation].
type Options struct {
	// ContentCapturingMode selects where spec-shaped GenAI message content
	// (gen_ai.system_instructions, gen_ai.input.messages, gen_ai.output.messages)
	// is recorded. Content may contain PII, so the default is NoContent. When
	// the zero value is passed, the env var CaptureContentEnvVar is consulted;
	// an explicit value here overrides it.
	ContentCapturingMode ContentCapturingMode

	// CaptureActionIO records raw Genkit action input/output as genkit.input /
	// genkit.output JSON attributes on every span (model, tool, flow, etc.).
	// Independent of ContentCapturingMode. May contain PII, off by default.
	// Inline data: media in model requests and responses is replaced with a
	// size marker; other actions' values (e.g. flow inputs) are recorded as is.
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

	// TracerProvider creates the spans. Nil means the global provider.
	TracerProvider oteltrace.TracerProvider

	// MeterProvider creates the GenAI client metrics. Nil means the global
	// provider.
	MeterProvider metric.MeterProvider

	// LoggerProvider receives the gen_ai.client.inference.operation.details
	// events (EventOnly / SpanAndEvent). Nil means the global provider.
	LoggerProvider otellog.LoggerProvider
}

// Instrumentation is a [tracing.Instrumentation] that emits OpenTelemetry
// telemetry following the OTel GenAI semantic conventions.
type Instrumentation struct {
	contentMode     ContentCapturingMode
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

// New builds an Instrumentation from opts. A nil opts uses the defaults. opts
// is read once, here; changing it afterwards has no effect.
//
// The tracer, meter, and logger are resolved here, once. With the global
// providers that still picks up an SDK installed later: the OTel globals hand
// out delegating instruments that switch to the real provider when it is set.
// Resolving per span instead would take the SDK provider's lock on every span
// start.
func New(opts *Options) *Instrumentation {
	if opts == nil {
		opts = &Options{}
	}
	mode := opts.ContentCapturingMode
	if mode == "" {
		mode = contentCapturingModeFromEnv()
	}
	scope := opts.ScopeName
	if scope == "" {
		scope = "genkit-genai"
	}
	tp := opts.TracerProvider
	if tp == nil {
		tp = otel.GetTracerProvider()
	}
	lp := opts.LoggerProvider
	if lp == nil {
		lp = otellogglobal.GetLoggerProvider()
	}
	g := &Instrumentation{
		contentMode:     mode,
		captureOnSpan:   mode == SpanOnly || mode == SpanAndEvent,
		captureOnEvent:  mode == EventOnly || mode == SpanAndEvent,
		captureActionIO: opts.CaptureActionIO,
		emitToolSpans:   opts.EmitToolSpans,
		tracer: tp.Tracer(scope,
			oteltrace.WithInstrumentationVersion(internal.Version),
			oteltrace.WithSchemaURL(genai.SchemaURL)),
		logger: lp.Logger(scope,
			otellog.WithInstrumentationVersion(internal.Version),
			otellog.WithSchemaURL(genai.SchemaURL)),
	}
	if !opts.DisableMetrics {
		mp := opts.MeterProvider
		if mp == nil {
			mp = otel.GetMeterProvider()
		}
		meter := mp.Meter(scope,
			metric.WithInstrumentationVersion(internal.Version),
			metric.WithSchemaURL(genai.SchemaURL))
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
func contentCapturingModeFromEnv() ContentCapturingMode {
	mode, ok := parseContentCapturingMode(os.Getenv(CaptureContentEnvVar))
	if !ok {
		slog.Warn("invalid content capturing mode, defaulting to NO_CONTENT",
			"env", CaptureContentEnvVar)
	}
	return mode
}

// StartSpan dispatches on the Genkit action type and opens a span encoded with
// the OTel GenAI conventions. The returned span records the result when Genkit
// ends it.
func (g *Instrumentation) StartSpan(ctx context.Context, info *tracing.SpanInfo) (context.Context, tracing.Span) {
	if info == nil {
		// Nothing to encode; the dispatcher tolerates a nil span.
		return ctx, nil
	}
	actionType := info.Subtype()
	if actionType == "" {
		actionType = info.Type()
	}
	switch actionType {
	case string(api.ActionTypeModel):
		return g.startModelSpan(ctx, info)
	case string(api.ActionTypeTool), string(api.ActionTypeToolV2):
		// DefineTool registers tool.v2; plain "tool" is the legacy spelling.
		if g.emitToolSpans {
			return g.startToolSpan(ctx, info)
		}
	}
	return g.startGenericSpan(ctx, info, actionType)
}

var _ tracing.Instrumentation = (*Instrumentation)(nil)
