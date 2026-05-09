// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlecloud

import (
	"context"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// testSpanExporter implements sdktrace.SpanExporter for in-memory testing
type testSpanExporter struct {
	mu             sync.Mutex
	exportedSpans  []sdktrace.ReadOnlySpan
	exportCalls    int
	shutdownCalled bool
	exportSignal   chan struct{}
}

func NewTestSpanExporter() *testSpanExporter {
	return &testSpanExporter{
		exportedSpans: make([]sdktrace.ReadOnlySpan, 0),
		exportSignal:  make(chan struct{}, 100),
	}
}

func (e *testSpanExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exportedSpans = append(e.exportedSpans, spans...)
	e.exportCalls++
	select {
	case e.exportSignal <- struct{}{}:
	default:
	}
	return nil
}

func (e *testSpanExporter) Shutdown(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.shutdownCalled = true
	return nil
}

func (e *testSpanExporter) GetExportedSpans() []sdktrace.ReadOnlySpan {
	e.mu.Lock()
	defer e.mu.Unlock()
	spans := make([]sdktrace.ReadOnlySpan, len(e.exportedSpans))
	copy(spans, e.exportedSpans)
	return spans
}

func (e *testSpanExporter) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exportedSpans = e.exportedSpans[:0]
	e.exportCalls = 0
	e.shutdownCalled = false
	for {
		select {
		case <-e.exportSignal:
		default:
			return
		}
	}
}

func (e *testSpanExporter) GetSpanByName(name string) sdktrace.ReadOnlySpan {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, span := range e.exportedSpans {
		if span.Name() == name {
			return span
		}
	}
	return nil
}

// testError is a simple error implementation for testing
type testError struct {
	msg string
}

func (e *testError) Error() string {
	return e.msg
}

// Test fixture for common test setup
type testFixture struct {
	mockExporter *testSpanExporter
	adjuster     *AdjustingTraceExporter
	tracer       trace.Tracer
	tp           *sdktrace.TracerProvider
	ctx          context.Context
}

// newTestFixture creates a complete test setup with configurable logging
func newTestFixture(t *testing.T, logInputAndOutput bool, modules ...Telemetry) *testFixture {
	mockExporter := NewTestSpanExporter()
	adjuster := &AdjustingTraceExporter{
		exporter:          mockExporter,
		modules:           modules,
		logInputAndOutput: logInputAndOutput,
		projectId:         "test-project",
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(adjuster)),
	)
	t.Cleanup(func() { tp.Shutdown(context.Background()) })

	return &testFixture{
		mockExporter: mockExporter,
		adjuster:     adjuster,
		tracer:       tp.Tracer("test-tracer"),
		tp:           tp,
		ctx:          context.Background(),
	}
}

// spanBuilder helps create spans with attributes fluently
type spanBuilder struct {
	fixture *testFixture
	name    string
	attrs   []attribute.KeyValue
	status  *codes.Code
	err     error
}

func (f *testFixture) createSpan(name string) *spanBuilder {
	return &spanBuilder{
		fixture: f,
		name:    name,
		attrs:   make([]attribute.KeyValue, 0),
	}
}

func (sb *spanBuilder) withAttr(key, value string) *spanBuilder {
	sb.attrs = append(sb.attrs, attribute.String(key, value))
	return sb
}

func (sb *spanBuilder) withStatus(code codes.Code) *spanBuilder {
	sb.status = &code
	return sb
}

func (sb *spanBuilder) withError(err error) *spanBuilder {
	sb.err = err
	return sb
}

func (sb *spanBuilder) build() trace.Span {
	_, span := sb.fixture.tracer.Start(sb.fixture.ctx, sb.name)
	span.SetAttributes(sb.attrs...)

	if sb.status != nil {
		span.SetStatus(*sb.status, "Test status")
	}

	if sb.err != nil {
		span.RecordError(sb.err)
	}

	return span
}

// Test helpers
func (f *testFixture) waitAndGetSpans() []sdktrace.ReadOnlySpan {
	time.Sleep(100 * time.Millisecond) // SimpleSpanProcessor is immediate but allow small delay
	spans := f.mockExporter.GetExportedSpans()
	return spans
}

func (f *testFixture) assertSpanCount(t *testing.T, expected int) []sdktrace.ReadOnlySpan {
	spans := f.waitAndGetSpans()
	if len(spans) != expected {
		t.Errorf("got %d spans, want %d", len(spans), expected)
	}
	return spans
}

func (f *testFixture) assertSpanExists(t *testing.T, name string) sdktrace.ReadOnlySpan {
	span := f.mockExporter.GetSpanByName(name)
	if span == nil {
		t.Fatalf("span %q should exist", name)
	}
	return span
}

// Attribute helpers
func getAttrMap(span sdktrace.ReadOnlySpan) map[string]string {
	attrMap := make(map[string]string)
	for _, attr := range span.Attributes() {
		attrMap[string(attr.Key)] = attr.Value.AsString()
	}
	return attrMap
}

func assertAttr(t *testing.T, span sdktrace.ReadOnlySpan, key, expected string) {
	attrMap := getAttrMap(span)
	got, ok := attrMap[key]
	if !ok {
		t.Errorf("attribute %q not found", key)
		return
	}
	if got != expected {
		t.Errorf("attribute %q = %q, want %q", key, got, expected)
	}
}

// Tests

func TestNewAdjustingTraceExporter(t *testing.T) {
	mockExporter := NewTestSpanExporter()
	adjuster := &AdjustingTraceExporter{
		exporter:          mockExporter,
		modules:           []Telemetry{},
		logInputAndOutput: false,
		projectId:         "test-project",
	}

	if adjuster == nil {
		t.Fatal("adjuster should not be nil")
	}
	if got, want := adjuster.exporter, mockExporter; got != want {
		t.Errorf("exporter = %v, want %v", got, want)
	}
	if got, want := adjuster.projectId, "test-project"; got != want {
		t.Errorf("projectId = %q, want %q", got, want)
	}
	if adjuster.logInputAndOutput {
		t.Error("logInputAndOutput should be false")
	}
}

func TestAdjustingTraceExporter_ExportSpansWithRealTracer(t *testing.T) {
	f := newTestFixture(t, false)

	// Create spans
	f.createSpan("generate").
		withAttr("genkit:model", "gemini-pro").
		withAttr("genkit:type", "generate").
		build().End()

	f.createSpan("feature").
		withAttr("genkit:type", "feature").
		withAttr("genkit:name", "testFeature").
		build().End()

	// Verify
	f.assertSpanCount(t, 2)
	f.assertSpanExists(t, "generate")
	f.assertSpanExists(t, "feature")
}

func TestAdjustingTraceExporter_FailedSpan(t *testing.T) {
	f := newTestFixture(t, false)

	// Create failing span
	f.createSpan("failing-action").
		withAttr("genkit:name", "testAction").
		withStatus(codes.Error).
		withError(&testError{msg: "test failure"}).
		build().End()

	// Verify
	spans := f.assertSpanCount(t, 1)
	if got, want := spans[0].Status().Code, codes.Error; got != want {
		t.Errorf("status code = %v, want %v", got, want)
	}
}

func TestAdjustingTraceExporter_NormalizeLabels(t *testing.T) {
	f := newTestFixture(t, false)

	// Create span with colon attributes
	f.createSpan("label-test").
		withAttr("test:attribute", "value1").
		withAttr("another:key:here", "value2").
		withAttr("normal_attribute", "value3").
		build().End()

	// Verify label normalization
	span := f.assertSpanExists(t, "label-test")

	// Verify colons were converted to slashes
	assertAttr(t, span, "test/attribute", "value1")
	assertAttr(t, span, "another/key/here", "value2")
	assertAttr(t, span, "normal_attribute", "value3")
}

func TestAdjustingTraceExporter_Shutdown(t *testing.T) {
	mockExporter := NewTestSpanExporter()
	adjuster := &AdjustingTraceExporter{
		exporter:          mockExporter,
		modules:           []Telemetry{},
		logInputAndOutput: false,
		projectId:         "test-project",
	}

	err := adjuster.Shutdown(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !mockExporter.shutdownCalled {
		t.Error("shutdownCalled should be true")
	}
}

func TestCompleteSpanProcessingPipeline(t *testing.T) {
	f := newTestFixture(t, false)

	// Create diverse spans
	f.createSpan("generate").
		withAttr("genkit:model", "gemini-pro").
		withAttr("genkit:type", "generate").
		withAttr("test:colon", "should-be-normalized").
		build().End()

	f.createSpan("feature").
		withAttr("genkit:type", "feature").
		withAttr("genkit:name", "testFeature").
		build().End()

	f.createSpan("failed-action").
		withAttr("genkit:name", "failingAction").
		withStatus(codes.Error).
		build().End()

	// Verify all transformations
	f.assertSpanCount(t, 3)

	generateSpan := f.assertSpanExists(t, "generate")
	featureSpan := f.assertSpanExists(t, "feature")
	failedSpan := f.assertSpanExists(t, "failed-action")

	// Verify transformations
	assertAttr(t, generateSpan, "test/colon", "should-be-normalized")
	assertAttr(t, generateSpan, "genkit/model", "gemini-pro")
	assertAttr(t, featureSpan, "genkit/type", "feature")
	if got, want := failedSpan.Status().Code, codes.Error; got != want {
		t.Errorf("status code = %v, want %v", got, want)
	}
}
