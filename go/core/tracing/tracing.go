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
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/internal/base"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
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

// markErrorAsHandled marks an error as already handled for failure source tracking
func markErrorAsHandled(err error) error {
	var me *markedError
	if errors.As(err, &me) {
		me.marked = true
		return me
	}

	return &markedError{error: err, marked: true}
}

// isErrorAlreadyMarked checks if an error has already been marked as a failure source
func isErrorAlreadyMarked(err error) bool {
	var me *markedError
	if errors.As(err, &me) {
		return me.marked
	}
	return false
}

// captureStackTrace captures the current Go stack trace for error reporting
func captureStackTrace() string {
	buf := make([]byte, 4096)
	n := runtime.Stack(buf, false)
	stackTrace := string(buf[:n])
	lines := strings.Split(stackTrace, "\n")
	var cleanLines []string
	skipNext := false

	for _, line := range lines {

		if strings.Contains(line, "github.com/firebase/genkit/go/core/tracing") ||
			strings.Contains(line, "runtime.") ||
			strings.Contains(line, "captureStackTrace") {
			skipNext = true
			continue
		}

		if skipNext {
			skipNext = false
			continue
		}

		cleanLines = append(cleanLines, line)

		if len(cleanLines) > 20 {
			break
		}
	}

	return strings.Join(cleanLines, "\n")
}

var (
	providerInitOnce sync.Once
)

// TracerProvider returns the global tracer provider, creating it if needed.
func TracerProvider() *sdktrace.TracerProvider {
	if tp := otel.GetTracerProvider(); tp != nil {
		if sdkTP, ok := tp.(*sdktrace.TracerProvider); ok {
			return sdkTP
		}
	}

	providerInitOnce.Do(func() {
		otel.SetTracerProvider(sdktrace.NewTracerProvider())
		if telemetryURL := os.Getenv("GENKIT_TELEMETRY_SERVER"); telemetryURL != "" {
			WriteTelemetryImmediate(NewHTTPTelemetryClient(telemetryURL))
		}
	})

	return otel.GetTracerProvider().(*sdktrace.TracerProvider)
}

// Tracer returns a tracer from the global tracer provider.
func Tracer() trace.Tracer {
	return TracerProvider().Tracer("genkit-tracer", trace.WithInstrumentationVersion("v1"))
}

// WriteTelemetryImmediate adds a telemetry server to the global tracer provider.
// Traces are saved immediately as they are finished.
// Use this for a gtrace.Store with a fast Save method,
// such as one that writes to a file.
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
func WriteTelemetryBatch(client TelemetryClient) (shutdown func(context.Context) error) {
	e := newTelemetryServerExporter(client)
	TracerProvider().RegisterSpanProcessor(sdktrace.NewBatchSpanProcessor(e))
	return TracerProvider().Shutdown
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
	log := logger.FromContext(ctx)
	log.Debug("span start", "name", metadata.Name)
	defer log.Debug("span end", "name", metadata.Name)

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
		IsRoot:   isRoot,
		Type:     metadata.Type,
		Subtype:  metadata.Subtype,
		Metadata: metadata.Metadata,
	}

	var parentPath string
	if parentSM != nil {
		parentPath = parentSM.Path
	}

	// Build path with type annotations to maintain compatibility with TypeScript telemetry format
	if metadata.Subtype == "flow" {
		sm.Path = buildAnnotatedPath(metadata.Name, parentPath, "flow")
	} else if metadata.Subtype == "util" {
		sm.Path = buildAnnotatedPath(metadata.Name, parentPath, "util")
	} else {
		sm.Path = buildAnnotatedPath(metadata.Name, parentPath, metadata.Type)
		if metadata.Subtype != "" {
			sm.Path = decoratePathWithSubtype(sm.Path, metadata.Subtype)
		}
	}

	var opts []trace.SpanStartOption
	if metadata.TelemetryLabels != nil {
		var attrs []attribute.KeyValue
		for k, v := range metadata.TelemetryLabels {
			attrs = append(attrs, attribute.String(k, v))
		}
		opts = append(opts, trace.WithAttributes(attrs...))
	}

	if metadata.Type != "" {
		opts = append(opts, trace.WithAttributes(attribute.String(spanTypeAttr, metadata.Type)))
	}

	ctx, span := Tracer().Start(ctx, metadata.Name, opts...)
	sm.TraceInfo = TraceInfo{
		TraceID: span.SpanContext().TraceID().String(),
		SpanID:  span.SpanContext().SpanID().String(),
	}

	// Fire telemetry callback immediately if one was set on the context
	if cb := telemetryCallback(ctx); cb != nil {
		cb(sm.TraceInfo.TraceID, sm.TraceInfo.SpanID)
	}

	defer span.End()
	defer func() { span.SetAttributes(sm.attributes()...) }()
	ctx = spanMetaKey.NewContext(ctx, sm)
	output, err := f(ctx, input)

	if err != nil {
		sm.State = spanStateError
		sm.Error = err.Error()
		sm.IsFailureSource = true
		if !isErrorAlreadyMarked(err) {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
		}
	} else {
		sm.State = spanStateSuccess
		sm.Output = output
	}
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

// spanMetaKey is for storing spanMetadatas in a context.
var spanMetaKey = base.NewContextKey[*spanMetadata]()

// telemetryCbKey is the context key for telemetry callbacks.
var telemetryCbKey = base.NewContextKey[func(traceID, spanID string)]()

// WithTelemetryCallback returns a context with the telemetry callback attached.
// Used by the reflection server to pass callbacks to actions.
func WithTelemetryCallback(ctx context.Context, cb func(traceID, spanID string)) context.Context {
	return telemetryCbKey.NewContext(ctx, cb)
}

// telemetryCallback retrieves the telemetry callback from context, or nil if not set.
func telemetryCallback(ctx context.Context) func(traceID, spanID string) {
	return telemetryCbKey.FromContext(ctx)
}

// SpanPath returns the path as recorded in the current span metadata.
func SpanPath(ctx context.Context) string {
	return spanMetaKey.FromContext(ctx).Path
}

// TraceInfo returns the trace info as recorded in the current span metadata.
func SpanTraceInfo(ctx context.Context) TraceInfo {
	return spanMetaKey.FromContext(ctx).TraceInfo
}
