// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlecloud

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// TestFeatureTelemetry_PipelineIntegration verifies that feature telemetry
// processes root spans correctly in the full pipeline
func TestFeatureTelemetry_PipelineIntegration(t *testing.T) {
	// This test verifies that feature telemetry works correctly in the full pipeline,
	// only processing root spans

	featureTel := NewFeatureTelemetry()
	f := newTestFixture(t, false, featureTel)

	// Create span using the TracerProvider - this triggers the full pipeline
	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "test-feature-span")

	span.SetAttributes(
		attribute.String("genkit:type", "flow"), // Required for telemetry processing
		attribute.Bool("genkit:isRoot", true),
		attribute.String("genkit:name", "testFeature"),
		attribute.String("genkit:path", "/{testFlow,t:flow}/{testFeature,t:action}"),
		attribute.String("genkit:state", "success"),
	)

	span.End() // This triggers the pipeline

	// Verify the span was exported
	spans := f.waitAndGetSpans()
	if len(spans) != 1 {
		t.Errorf("got %d spans, want 1", len(spans))
	}
}

func TestFeatureTelemetry_MetricCapture(t *testing.T) {
	// Test that verifies we can capture and verify metric calls using OTel's built-in test reader

	testCases := []struct {
		name                   string
		attrs                  map[string]interface{}
		expectCounterMetrics   bool
		expectHistogramMetrics bool
		expectedStatus         string
		expectedError          string
		expectedName           string
	}{
		{
			name: "successful feature captures metrics correctly",
			attrs: map[string]interface{}{
				"genkit:type":   "flow",
				"genkit:isRoot": true,
				"genkit:name":   "chatFlow",
				"genkit:path":   "/{chatFlow,t:flow}/{generateResponse,t:action}",
				"genkit:state":  "success",
			},
			expectCounterMetrics:   true,
			expectHistogramMetrics: true,
			expectedStatus:         "success",
			expectedName:           "chatFlow",
		},
		{
			name: "failed feature captures metrics correctly",
			attrs: map[string]interface{}{
				"genkit:type":   "flow",
				"genkit:isRoot": true,
				"genkit:name":   "codeAssistant",
				"genkit:path":   "/{codeAssistant,t:flow}/{suggestCode,t:action}",
				"genkit:state":  "error",
			},
			expectCounterMetrics:   true,
			expectHistogramMetrics: true,
			expectedStatus:         "failure",
			expectedName:           "codeAssistant",
			expectedError:          "<unknown>",
		},
		{
			name: "non-root span captures no metrics",
			attrs: map[string]interface{}{
				"genkit:isRoot": false,
				"genkit:name":   "subAction",
				"genkit:state":  "success",
			},
			expectCounterMetrics:   false,
			expectHistogramMetrics: false,
		},
		{
			name: "unknown state captures no metrics",
			attrs: map[string]interface{}{
				"genkit:isRoot": true,
				"genkit:name":   "testFeature",
				"genkit:state":  "unknown",
			},
			expectCounterMetrics:   false,
			expectHistogramMetrics: false,
		},
		{
			name: "empty string state captures no metrics",
			attrs: map[string]interface{}{
				"genkit:isRoot": true,
				"genkit:name":   "testFeature",
				"genkit:state":  "", // Explicit empty string
			},
			expectCounterMetrics:   false,
			expectHistogramMetrics: false,
		},
		{
			name: "missing state attribute captures no metrics",
			attrs: map[string]interface{}{
				"genkit:isRoot": true,
				"genkit:name":   "testFeature",
				// No genkit:state attribute at all
			},
			expectCounterMetrics:   false,
			expectHistogramMetrics: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create a fresh ManualReader for each test case to avoid accumulation
			reader := sdkmetric.NewManualReader()

			// Create a MeterProvider with the test reader
			testMeterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))

			// Set the global meter provider temporarily for the test
			originalProvider := otel.GetMeterProvider()
			otel.SetMeterProvider(testMeterProvider)
			defer otel.SetMeterProvider(originalProvider)

			// Create feature telemetry (it will use the global meter provider)
			featureTel := NewFeatureTelemetry()
			f := newTestFixture(t, false, featureTel)

			f.mockExporter.Reset()

			// Create span using the TracerProvider - this will flow through feature telemetry
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-feature-span")

			for key, value := range tc.attrs {
				switch v := value.(type) {
				case string:
					span.SetAttributes(attribute.String(key, v))
				case bool:
					span.SetAttributes(attribute.Bool(key, v))
				case int:
					span.SetAttributes(attribute.Int(key, v))
				case int64:
					span.SetAttributes(attribute.Int64(key, v))
				case float64:
					span.SetAttributes(attribute.Float64(key, v))
				}
			}

			span.End() // This triggers the pipeline including feature telemetry

			// Wait for span to be processed
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}

			// Collect metrics using the manual reader
			var resourceMetrics metricdata.ResourceMetrics
			err := reader.Collect(ctx, &resourceMetrics)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify counter metrics
			if tc.expectCounterMetrics {
				counterMetric := findMetric(&resourceMetrics, "genkit/feature/requests")
				if counterMetric == nil {
					t.Error("Expected counter metric to be recorded")
				} else {
					expectedAttrs := map[string]interface{}{
						"name":   tc.expectedName,
						"status": tc.expectedStatus,
						"source": "go",
					}
					if tc.expectedError != "" {
						expectedAttrs["error"] = tc.expectedError
					}
					verifyCounterMetric(t, counterMetric, expectedAttrs)
				}
			}

			// Verify histogram metrics
			if tc.expectHistogramMetrics {
				histogramMetric := findMetric(&resourceMetrics, "genkit/feature/latency")
				if histogramMetric == nil {
					t.Error("Expected histogram metric to be recorded")
				} else {
					expectedAttrs := map[string]interface{}{
						"name":   tc.expectedName,
						"status": tc.expectedStatus,
						"source": "go",
					}
					if tc.expectedError != "" {
						expectedAttrs["error"] = tc.expectedError
					}
					verifyHistogramMetric(t, histogramMetric, expectedAttrs)
				}
			}

			if !tc.expectCounterMetrics && !tc.expectHistogramMetrics {
				// Should have no feature metrics
				counterMetric := findMetric(&resourceMetrics, "genkit/feature/requests")
				histogramMetric := findMetric(&resourceMetrics, "genkit/feature/latency")
				if counterMetric != nil {
					t.Error("Should not have counter metrics")
				}
				if histogramMetric != nil {
					t.Error("Should not have histogram metrics")
				}
			}
		})
	}
}

func TestFeatureTelemetry_ComprehensiveScenarios(t *testing.T) {
	// Test multiple feature telemetry scenarios using the proper pipeline integration

	featureTel := NewFeatureTelemetry()
	f := newTestFixture(t, false, featureTel)

	testCases := []struct {
		name  string
		attrs map[string]interface{}
	}{
		{
			name: "successful root span",
			attrs: map[string]interface{}{
				"genkit:isRoot": true,
				"genkit:name":   "chatFlow",
				"genkit:path":   "/{chatFlow,t:flow}/{generateResponse,t:action}",
				"genkit:state":  "success",
			},
		},
		{
			name: "failed root span",
			attrs: map[string]interface{}{
				"genkit:isRoot": true,
				"genkit:name":   "codeAssistant",
				"genkit:path":   "/{codeAssistant,t:flow}/{suggestCode,t:action}",
				"genkit:state":  "error",
			},
		},
		{
			name: "non-root span skipped",
			attrs: map[string]interface{}{
				"genkit:isRoot": false,
				"genkit:name":   "subAction",
				"genkit:state":  "success",
			},
		},
		{
			name: "unknown state",
			attrs: map[string]interface{}{
				"genkit:isRoot": true,
				"genkit:name":   "testFeature",
				"genkit:state":  "unknown",
			},
		},
		{
			name: "missing state attribute",
			attrs: map[string]interface{}{
				"genkit:isRoot": true,
				"genkit:name":   "testFeature",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			f.mockExporter.Reset()

			// Create span using the TracerProvider - this triggers the full pipeline
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-feature-span")

			for key, value := range tc.attrs {
				switch v := value.(type) {
				case string:
					span.SetAttributes(attribute.String(key, v))
				case bool:
					span.SetAttributes(attribute.Bool(key, v))
				case int:
					span.SetAttributes(attribute.Int(key, v))
				case int64:
					span.SetAttributes(attribute.Int64(key, v))
				case float64:
					span.SetAttributes(attribute.Float64(key, v))
				}
			}

			span.End() // This triggers the pipeline including feature telemetry

			// Verify spans were processed
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}
		})
	}
}

func TestFeatureTelemetry_InputOutputLogging(t *testing.T) {
	// Test that input/output logging works when logInputOutput is enabled

	featureTel := NewFeatureTelemetry()

	// Create custom fixture with logInputOutput enabled
	mockExporter := NewTestSpanExporter()
	adjuster := &AdjustingTraceExporter{
		exporter:          mockExporter,
		modules:           []Telemetry{featureTel},
		logInputAndOutput: true, // Enable input/output logging for this test
		projectId:         "test-project",
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(sdktrace.NewSimpleSpanProcessor(adjuster)),
	)
	defer tp.Shutdown(context.Background())

	f := &testFixture{
		mockExporter: mockExporter,
		adjuster:     adjuster,
		tracer:       tp.Tracer("test-tracer"),
		tp:           tp,
		ctx:          context.Background(),
	}

	// Set up log capture
	logBuf := setupLogCapture(t)

	// Create span with input/output data
	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "test-feature-span")

	span.SetAttributes(
		attribute.String("genkit:type", "flow"), // Required for telemetry processing
		attribute.Bool("genkit:isRoot", true),
		attribute.String("genkit:name", "testFeature"),
		attribute.String("genkit:path", "/{testFlow,t:flow}/{testFeature,t:action}"),
		attribute.String("genkit:state", "success"),
		attribute.String("genkit:input", `{"prompt": "Hello world"}`),
		attribute.String("genkit:output", `{"response": "Hi there!"}`),
		attribute.String("genkit:sessionId", "session-123"),
		attribute.String("genkit:threadName", "thread-456"),
	)

	span.End() // This triggers the pipeline

	// Get captured logs
	logOutput := logBuf.String()

	// Verify input/output logs are present - we explicitly enabled logInputOutput=true
	if !strings.Contains(logOutput, "Input[") {
		t.Error("Expected input log")
	}
	if !strings.Contains(logOutput, "Output[") {
		t.Error("Expected output log")
	}
	if !strings.Contains(logOutput, "testFeature") {
		t.Error("Expected feature name in logs")
	}

	// Verify spans were processed
	spans := f.waitAndGetSpans()
	if len(spans) != 1 {
		t.Errorf("got %d spans, want 1", len(spans))
	}
}

// Helper function for histogram metric verification (reuses counter verification pattern)
func verifyHistogramMetric(t *testing.T, metric *metricdata.Metrics, expectedAttrs map[string]interface{}) {
	// Verify it's a histogram metric
	histogram, ok := metric.Data.(metricdata.Histogram[float64])
	if !ok {
		t.Errorf("Expected metric to be a Histogram[float64], got %T", metric.Data)
		return
	}

	// Should have exactly one data point for our test
	if len(histogram.DataPoints) != 1 {
		t.Fatalf("got %d data points, want 1", len(histogram.DataPoints))
	}

	if len(histogram.DataPoints) > 0 {
		dp := histogram.DataPoints[0]

		// Verify the count (should be 1 for our test)
		if got, want := dp.Count, uint64(1); got != want {
			t.Errorf("Count = %v, want %v", got, want)
		}

		// Verify the latency value is reasonable for a test span
		if dp.Sum <= 0 {
			t.Errorf("Sum = %v, want > 0", dp.Sum)
		}

		// Verify we have bucket counts (histogram should have buckets)
		if len(dp.BucketCounts) == 0 {
			t.Error("Expected histogram to have bucket counts")
		}

		// Verify the sum of bucket counts equals the total count
		var totalBucketCount uint64
		for _, bucketCount := range dp.BucketCounts {
			totalBucketCount += bucketCount
		}
		if totalBucketCount != dp.Count {
			t.Errorf("Sum of bucket counts = %v, want %v", totalBucketCount, dp.Count)
		}

		// Verify attributes (reuse same pattern as counter verification)
		for expectedKey, expectedValue := range expectedAttrs {
			found := false
			for _, attr := range dp.Attributes.ToSlice() {
				if string(attr.Key) == expectedKey {
					found = true
					switch v := expectedValue.(type) {
					case string:
						if got, want := attr.Value.AsString(), v; got != want {
							t.Errorf("Attribute %s = %q, want %q", expectedKey, got, want)
						}
					case bool:
						if got, want := attr.Value.AsBool(), v; got != want {
							t.Errorf("Attribute %s = %v, want %v", expectedKey, got, want)
						}
					case int64:
						if got, want := attr.Value.AsInt64(), v; got != want {
							t.Errorf("Attribute %s = %v, want %v", expectedKey, got, want)
						}
					default:
						if got, want := attr.Value.AsString(), fmt.Sprintf("%v", v); got != want {
							t.Errorf("Attribute %s = %q, want %q", expectedKey, got, want)
						}
					}
					break
				}
			}
			if !found {
				t.Errorf("Expected attribute %s not found", expectedKey)
			}
		}
	}
}

func TestFeatureTelemetry_LatencyVerification(t *testing.T) {
	// Specific test to verify that latency measurement actually works correctly

	reader := sdkmetric.NewManualReader()
	testMeterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	originalProvider := otel.GetMeterProvider()
	otel.SetMeterProvider(testMeterProvider)
	defer otel.SetMeterProvider(originalProvider)

	featureTel := NewFeatureTelemetry()
	f := newTestFixture(t, false, featureTel)

	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "latency-test-span")

	span.SetAttributes(
		attribute.String("genkit:type", "flow"), // Required for telemetry processing
		attribute.Bool("genkit:isRoot", true),
		attribute.String("genkit:name", "latencyTestFeature"),
		attribute.String("genkit:state", "success"),
	)

	// Add a small delay to ensure measurable latency
	time.Sleep(1 * time.Millisecond)

	span.End()

	// Collect metrics
	var resourceMetrics metricdata.ResourceMetrics
	err := reader.Collect(ctx, &resourceMetrics)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify latency histogram
	histogramMetric := findMetric(&resourceMetrics, "genkit/feature/latency")
	if histogramMetric == nil {
		t.Fatal("Expected latency histogram metric")
	}

	if histogramMetric != nil {
		histogram, ok := histogramMetric.Data.(metricdata.Histogram[float64])
		if !ok {
			t.Errorf("Expected histogram type, got %T", histogramMetric.Data)
		}

		if len(histogram.DataPoints) > 0 {
			dp := histogram.DataPoints[0]

			// More specific latency assertions
			if got, want := dp.Count, uint64(1); got != want {
				t.Errorf("Count = %v, want %v", got, want)
			}
			if dp.Sum < 1.0 {
				t.Errorf("Sum = %v, want >= 1.0", dp.Sum)
			}
			if dp.Sum >= 100.0 {
				t.Errorf("Sum = %v, want < 100.0", dp.Sum)
			}

			// Verify histogram has reasonable structure
			if len(dp.BucketCounts) == 0 {
				t.Error("Should have histogram buckets")
			}
			if len(dp.Bounds) == 0 {
				t.Error("Should have bucket boundaries")
			}

			// At least one bucket should contain our measurement
			hasNonZeroBucket := false
			for _, count := range dp.BucketCounts {
				if count > 0 {
					hasNonZeroBucket = true
					break
				}
			}
			if !hasNonZeroBucket {
				t.Error("At least one bucket should contain the measurement")
			}
		}
	}
}
