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

// The tracing package provides support for execution traces.
package tracing

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/internal/base"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// markedError wraps an error to track if it's already been marked as a failure source
type markedError struct {
	error
	marked bool
}

func (e *markedError) Error() string {
	return e.error.Error()
}

func (e *markedError) Unwrap() error {
	return e.error
}

// isErrorAlreadyMarked checks if an error has already been marked as a failure source
func isErrorAlreadyMarked(err error) bool {
	var me *markedError
	if errors.As(err, &me) {
		return me.marked
	}
	return false
}

var (
	providerInitOnce sync.Once
)

// TracerProvider returns the global tracer provider, creating it if needed.
//
// Deprecated: Genkit no longer owns the global OpenTelemetry TracerProvider.
// Configure OpenTelemetry yourself (otel.SetTracerProvider with your own
// exporters, or the contrib autoexport package to honor OTEL_* env vars), or
// use the Google Cloud / Firebase plugins. Retained for backwards compatibility
// and removed in the next major version.
func TracerProvider() *sdktrace.TracerProvider {
	if tp := otel.GetTracerProvider(); tp != nil {
		if sdkTP, ok := tp.(*sdktrace.TracerProvider); ok {
			return sdkTP
		}
	}

	// Lazily create the SDK provider so OTel ids and any user-configured
	// exporters keep working. It no longer auto-registers the telemetry-server
	// processor: feeding the Dev UI is now the Direct instrumentation's job
	// (axis 1), decoupled from tracer-provider bootstrap (axis 2).
	providerInitOnce.Do(func() {
		otel.SetTracerProvider(sdktrace.NewTracerProvider())
	})

	return otel.GetTracerProvider().(*sdktrace.TracerProvider)
}

// Tracer returns a tracer from the global tracer provider.
//
// Deprecated: Genkit no longer manages the global TracerProvider. Get a tracer
// from your own OpenTelemetry setup (otel.GetTracerProvider().Tracer(...)).
// Removed in the next major version.
func Tracer() trace.Tracer {
	return TracerProvider().Tracer("genkit-tracer", trace.WithInstrumentationVersion("v1"))
}

// WriteTelemetryImmediate adds a telemetry server to the global tracer provider.
// Traces are saved immediately as they are finished.
// Use this for a gtrace.Store with a fast Save method,
// such as one that writes to a file.
//
// Deprecated: registers a span processor on the Genkit-managed global
// TracerProvider. Configure OpenTelemetry export yourself, or rely on the dev
// UI's Direct export / the Google Cloud / Firebase plugins. Removed in the next
// major version.
func WriteTelemetryImmediate(client TelemetryClient) {
	e := newTelemetryServerExporter(client)
	TracerProvider().RegisterSpanProcessor(sdktrace.NewSimpleSpanProcessor(e))
}

// WriteTelemetryBatch adds a telemetry server to the global tracer provider.
// Traces are batched before being sent for processing.
// Use this for a gtrace.Store with a potentially expensive Save method,
// such as one that makes an RPC.
//
// Callers must invoke the returned function at the end of the program to flush the final batch
// and perform other cleanup.
//
// Deprecated: registers a span processor on the Genkit-managed global
// TracerProvider. Configure OpenTelemetry export yourself, or rely on the dev
// UI's Direct export / the Google Cloud / Firebase plugins. Removed in the next
// major version.
func WriteTelemetryBatch(client TelemetryClient) (shutdown func(context.Context) error) {
	e := newTelemetryServerExporter(client)
	TracerProvider().RegisterSpanProcessor(sdktrace.NewBatchSpanProcessor(e))
	return TracerProvider().Shutdown
}

// WriteTelemetryRealtime adds a telemetry server to the global tracer provider
// that writes each span both when it starts and when it ends, so a still-running
// trace appears in the dev UI immediately rather than only once its root span
// closes. This matters for agents, whose root span stays open for the lifetime
// of a bidirectional connection. Use this for a fast Save method (such as one
// that writes to a file or a local server); for an end-only export use
// [WriteTelemetryImmediate].
//
// Deprecated: registers a span processor on the Genkit-managed global
// TracerProvider. Configure OpenTelemetry export yourself, or rely on the dev
// UI's Direct export / the Google Cloud / Firebase plugins. Removed in the next
// major version.
func WriteTelemetryRealtime(client TelemetryClient) {
	TracerProvider().RegisterSpanProcessor(newRealtimeSpanProcessor(client))
}

// realtimeTelemetryActive records whether the dev CLI asked runtimes to export
// live traces, read once at startup. `genkit start` sets
// GENKIT_ENABLE_REALTIME_TELEMETRY (unless run with --disable-realtime-telemetry);
// the JS runtime keys its RealtimeSpanProcessor off the same variable. It is
// read on every span (see RunInNewSpan), so it is cached here rather than read
// from the environment each time.
var realtimeTelemetryActive = os.Getenv("GENKIT_ENABLE_REALTIME_TELEMETRY") == "true"

// realtimeTelemetryEnabled reports whether live-trace export is active.
func realtimeTelemetryEnabled() bool {
	return realtimeTelemetryActive
}

const (
	attrPrefix   = "genkit"
	spanTypeAttr = attrPrefix + ":type"
)

// SpanMetadata contains metadata information for creating properly annotated spans
type SpanMetadata struct {
	// Name is the span name
	Name string
	// IsRoot indicates if this is a root span
	IsRoot bool
	// Type represents the kind of span (e.g., "action", "flowStep")
	Type string
	// Subtype provides more specific categorization (e.g., "tool", "flow", "model")
	Subtype string
	// TelemetryLabels are arbitrary key-value pairs set directly as span attributes
	TelemetryLabels map[string]string
	// Metadata are genkit-specific metadata with automatic "genkit:metadata:" prefix
	Metadata map[string]string
	// Init is the initialization data supplied to the action, recorded as the
	// "genkit:init" span attribute when non-nil. It is kept separate from the
	// span input so tooling can distinguish per-call input from session
	// initialization data.
	Init any
}

// RunInNewSpan runs f on input in a new span with the provided metadata.
// The metadata contains all span configuration including name, type, labels, etc.
// If a telemetry callback was set on the context via WithTelemetryCallback,
// it will be called with the trace ID and span ID as soon as the span is created.
func RunInNewSpan[I, O any](
	ctx context.Context,
	metadata *SpanMetadata,
	input I,
	f func(context.Context, I) (O, error),
) (O, error) {
	// TODO: support span links.
	if metadata == nil {
		metadata = &SpanMetadata{}
	}

	parentSM := spanMetaKey.FromContext(ctx)
	isRoot := metadata.IsRoot
	if !isRoot && parentSM == nil {
		// No parent span means this is a root span
		isRoot = true
	}

	sm := &spanMetadata{
		Name:     metadata.Name,
		Input:    input,
		Init:     metadata.Init,
		IsRoot:   isRoot,
		Type:     metadata.Type,
		Subtype:  metadata.Subtype,
		Metadata: metadata.Metadata,
	}

	var parentPath string
	if parentSM != nil {
		parentPath = parentSM.Path
	}

	// Build path with type annotations to maintain compatibility with the
	// TypeScript telemetry format: a flow is annotated by its subtype, every
	// other span by its type, then its subtype when it has one.
	if metadata.Subtype == "flow" {
		sm.Path = buildAnnotatedPath(metadata.Name, parentPath, "flow")
	} else {
		sm.Path = buildAnnotatedPath(metadata.Name, parentPath, metadata.Type)
		if metadata.Subtype != "" {
			sm.Path = decoratePathWithSubtype(sm.Path, metadata.Subtype)
		}
	}

	info := &SpanInfo{Labels: metadata.TelemetryLabels, metadata: sm}

	// runBody is the center of the instrumentation chain: it runs the caller's
	// f under the new span's context and records Genkit's success/error
	// bookkeeping (state, output, failure source). Providers finalize their
	// backend spans around it. All backend-independent semantics stay here so
	// providers only encode.
	var output O
	runBody := func(ctx context.Context, span Span) (any, error) {
		sm.TraceInfo = span.TraceInfo()

		// Fire the telemetry callback the moment ids are known. Omit blank ids
		// so an uninstrumented run does not look like a broken exporter.
		if cb := telemetryCallback(ctx); cb != nil && sm.TraceInfo.TraceID != "" {
			cb(sm.TraceInfo.TraceID, sm.TraceInfo.SpanID)
		}

		ctx = spanMetaKey.NewContext(ctx, sm)
		// Expose the composite span so SetCustomMetadataAttributes can fan
		// mid-run metadata out to every active provider.
		ctx = currentSpanKey.NewContext(ctx, span)

		// These logs run under the new span's context, so they land on this
		// span in the Dev UI. The deferred one fires while the span is still
		// recording. This is the hottest path in the framework, so the log
		// arguments are only built when some handler accepts debug records
		// (the console at GENKIT_LOG_LEVEL=debug, or the Dev UI export sink).
		start := time.Now()
		logDebug := logger.FromContext(ctx).Enabled(ctx, slog.LevelDebug)
		if logDebug {
			startArgs := []any{"name", metadata.Name}
			if metadata.Type != "" {
				startArgs = append(startArgs, "type", metadata.Type)
			}
			if metadata.Subtype != "" {
				startArgs = append(startArgs, "subtype", metadata.Subtype)
			}
			logger.Debug(ctx, "span started", startArgs...)
		}
		defer func() {
			if !logDebug {
				return
			}
			endArgs := []any{"name", metadata.Name, "state", string(sm.State), "duration", time.Since(start).Round(time.Millisecond)}
			if sm.Error != "" {
				endArgs = append(endArgs, "error", sm.Error)
			}
			logger.Debug(ctx, "span finished", endArgs...)
		}()

		var err error
		output, err = f(ctx, input)
		if err != nil {
			sm.State = spanStateError
			sm.Error = err.Error()
			sm.IsFailureSource = true
			// A failure can still carry a result: the generate loop returns the
			// conversation it completed alongside its error. Record it so the
			// span shows what the call produced and not only that it stopped.
			// Guarded, because a function that returns nothing on error would
			// otherwise stamp a null output on every failing span.
			if !base.IsNil(output) {
				sm.Output = output
			}
		} else {
			sm.State = spanStateSuccess
			sm.Output = output
		}
		return output, err
	}

	_, err := dispatch(ctx, activeInstrumentations(), info, runBody)
	return output, err
}

// buildAnnotatedPath creates a path with type annotations
// e.g., /{chatFlow,t:flow}/{generateResponse,t:action}
func buildAnnotatedPath(name, parentPath, spanType string) string {
	pathSegment := name
	if spanType != "" {
		pathSegment = name + ",t:" + spanType
	}
	pathSegment = "{" + pathSegment + "}"
	return parentPath + "/" + pathSegment
}

// decoratePathWithSubtype adds subtype annotation to the final path segment
// e.g., /{flow,t:action}/{step,t:action} -> /{flow,t:action,s:flow}/{step,t:action,s:tool}
func decoratePathWithSubtype(path string, subtype string) string {
	if path == "" || subtype == "" {
		return path
	}

	// Find the last opening brace to locate the final path segment
	lastBraceIndex := strings.LastIndex(path, "{")
	if lastBraceIndex == -1 {
		return path // No braces found, nothing to decorate
	}

	// Find the closing brace after the last opening brace
	closingBraceIndex := strings.Index(path[lastBraceIndex:], "}")
	if closingBraceIndex == -1 {
		return path // No closing brace found
	}
	closingBraceIndex += lastBraceIndex

	// Extract the content of the last segment (without braces)
	segmentContent := path[lastBraceIndex+1 : closingBraceIndex]

	decoratedContent := segmentContent + ",s:" + subtype

	// Rebuild the path with the decorated last segment
	return path[:lastBraceIndex+1] + decoratedContent + path[closingBraceIndex:]
}

// spanState is the completion status of a span.
// An empty spanState indicates that the span has not ended.
type spanState string

const (
	spanStateSuccess spanState = "success"
	spanStateError   spanState = "error"
)

type TraceInfo struct {
	TraceID string
	SpanID  string
}

// spanMetadata holds genkit-specific information about a span.
type spanMetadata struct {
	TraceInfo       TraceInfo
	Name            string
	State           spanState
	IsRoot          bool
	IsFailureSource bool // whether this span is the source of a failure
	Input           any
	Init            any // initialization data for the action, if any
	Output          any
	Error           string            // error message if State is spanStateError
	Path            string            // annotated path with type information
	Type            string            // span type (action, flow, model, etc.)
	Subtype         string            // span subtype (tool, model, flow, etc.)
	Metadata        map[string]string // additional custom metadata
}

// attributes returns some information about the spanMetadata
// as a slice of OpenTelemetry attributes.
func (sm *spanMetadata) attributes() []attribute.KeyValue {
	kvs := []attribute.KeyValue{
		attribute.String("genkit:name", sm.Name),
		attribute.String("genkit:state", string(sm.State)),
		attribute.String("genkit:input", base.JSONString(sm.Input)),
		attribute.String("genkit:path", sm.Path),
	}

	if sm.Init != nil {
		kvs = append(kvs, attribute.String("genkit:init", base.JSONString(sm.Init)))
	}

	if sm.Output != nil {
		kvs = append(kvs, attribute.String("genkit:output", base.JSONString(sm.Output)))
	}

	if sm.Type != "" {
		kvs = append(kvs, attribute.String("genkit:type", sm.Type))
	}

	if sm.Subtype != "" {
		kvs = append(kvs, attribute.String("genkit:metadata:subtype", sm.Subtype))
	}

	if sm.IsRoot {
		kvs = append(kvs, attribute.Bool("genkit:isRoot", sm.IsRoot))
	}

	if sm.IsFailureSource {
		kvs = append(kvs, attribute.Bool("genkit:isFailureSource", true))
	}

	if sm.Metadata != nil {
		for k, v := range sm.Metadata {
			kvs = append(kvs, attribute.String(attrPrefix+":metadata:"+k, v))
		}
	}

	return kvs
}

// startAttributes returns the cheap subset of [spanMetadata.attributes] that is
// already known when the span begins: its identity and shape (name, path, type,
// subtype, root flag, and custom metadata). RunInNewSpan always sets these as
// span-start options so a live-trace exporter, which captures the span the
// instant it starts (before the deferred end write), still sees a span that
// renders with its proper type and place in the tree rather than just
// "genkit:type". State and output are excluded because they are not determined
// until the span finishes; input and init are excluded here because they are
// JSON-marshaled (see inputAttributes). These keys and values match attributes()
// exactly, so the end write simply reasserts them.
func (sm *spanMetadata) startAttributes() []attribute.KeyValue {
	kvs := []attribute.KeyValue{
		attribute.String("genkit:name", sm.Name),
		attribute.String("genkit:path", sm.Path),
	}
	if sm.Type != "" {
		kvs = append(kvs, attribute.String(spanTypeAttr, sm.Type))
	}
	if sm.Subtype != "" {
		kvs = append(kvs, attribute.String("genkit:metadata:subtype", sm.Subtype))
	}
	if sm.IsRoot {
		kvs = append(kvs, attribute.Bool("genkit:isRoot", sm.IsRoot))
	}
	for k, v := range sm.Metadata {
		kvs = append(kvs, attribute.String(attrPrefix+":metadata:"+k, v))
	}
	return kvs
}

// inputAttributes returns the JSON-marshaled input and init attributes. Both are
// known when the span begins, but RunInNewSpan only seeds them as span-start
// options when live-trace export is active: marshaling them is not free, and off
// the live path the deferred end write is the single place they are recorded.
// Seeding them at start lets an in-flight span show what it was invoked with,
// which for a bidi/agent root span (whose input is nil and whose init carries
// the session) is the only call data it has until it ends. The keys and values
// match attributes() so the end write reasserts them.
func (sm *spanMetadata) inputAttributes() []attribute.KeyValue {
	kvs := []attribute.KeyValue{
		attribute.String("genkit:input", base.JSONString(sm.Input)),
	}
	if sm.Init != nil {
		kvs = append(kvs, attribute.String("genkit:init", base.JSONString(sm.Init)))
	}
	return kvs
}

// spanMetaKey is for storing spanMetadatas in a context.
var spanMetaKey = base.NewContextKey[*spanMetadata]()

// currentSpanKey holds the composite Span of the running RunInNewSpan, so
// SetCustomMetadataAttributes can reach it.
var currentSpanKey = base.NewContextKey[Span]()

// telemetryCbKey is the context key for telemetry callbacks.
var telemetryCbKey = base.NewContextKey[func(traceID, spanID string)]()

// telemetryLabelsKey is the context key for telemetry labels.
var telemetryLabelsKey = base.NewContextKey[map[string]string]()

// WithTelemetryCallback returns a context with the telemetry callback attached.
// Used by the reflection server to pass callbacks to actions.
func WithTelemetryCallback(ctx context.Context, cb func(traceID, spanID string)) context.Context {
	return telemetryCbKey.NewContext(ctx, cb)
}

// WithTelemetryLabels returns a context with the telemetry labels attached.
// Used by the reflection server to pass labels to actions.
func WithTelemetryLabels(ctx context.Context, labels map[string]string) context.Context {
	return telemetryLabelsKey.NewContext(ctx, labels)
}

// TelemetryLabelsFromContext retrieves the telemetry labels from context, or nil if not set.
func TelemetryLabelsFromContext(ctx context.Context) map[string]string {
	return telemetryLabelsKey.FromContext(ctx)
}

// telemetryCallback retrieves the telemetry callback from context, or nil if not set.
func telemetryCallback(ctx context.Context) func(traceID, spanID string) {
	return telemetryCbKey.FromContext(ctx)
}

// SpanPath returns the path as recorded in the current span metadata.
func SpanPath(ctx context.Context) string {
	return spanMetaKey.FromContext(ctx).Path
}

// SpanTraceInfo returns the trace info recorded in the current span metadata,
// or the zero TraceInfo when called outside any span (e.g. a log emitted before
// a flow starts).
func SpanTraceInfo(ctx context.Context) TraceInfo {
	if sm := spanMetaKey.FromContext(ctx); sm != nil {
		return sm.TraceInfo
	}
	return TraceInfo{}
}

// SetCustomMetadataAttributes records custom metadata on the current span,
// fanning out to every active instrumentation provider. Each entry is stored as
// a genkit:metadata:<key> span attribute. It is a no-op when called outside a
// span. This is the backend-independent way to annotate the running span; it
// replaces writing to an OpenTelemetry span directly.
func SetCustomMetadataAttributes(ctx context.Context, md map[string]string) {
	if span := currentSpanKey.FromContext(ctx); span != nil {
		span.SetMetadata(md)
	}
}
