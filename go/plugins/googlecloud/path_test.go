// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlecloud

import (
	"context"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

func TestNewPathTelemetry(t *testing.T) {
	pathTel := NewPathTelemetry()

	if pathTel == nil {
		t.Fatal("pathTel should not be nil")
	}
	if pathTel.pathCounter == nil {
		t.Error("pathCounter should not be nil")
	}
	if pathTel.pathLatencies == nil {
		t.Error("pathLatencies should not be nil")
	}
}

// TestPathTelemetry_PipelineIntegration verifies that path telemetry
// processes failing spans correctly in the full pipeline
func TestPathTelemetry_PipelineIntegration(t *testing.T) {
	// This test verifies that path telemetry works correctly in the full pipeline,
	// only processing failing spans that are failure sources

	pathTel := NewPathTelemetry()
	f := newTestFixture(t, true, pathTel) // Enable logging for path telemetry tests

	// Set up log capture
	logBuf := setupLogCapture(t)

	// Create span using the TracerProvider - this triggers the full pipeline
	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "test-path-span")

	span.SetAttributes(
		attribute.String("genkit:type", "flow"), // Required for telemetry processing
		attribute.String("genkit:path", "/{testFlow,t:flow}/{myAction,t:action}"),
		attribute.Bool("genkit:isFailureSource", true),
		attribute.String("genkit:state", "error"),
		attribute.String("genkit:sessionId", "session-123"),
		attribute.String("genkit:threadName", "thread-456"),
	)
	span.SetStatus(codes.Error, "Test error")

	span.End() // This triggers the pipeline

	// Get captured logs
	logOutput := logBuf.String()

	// Verify path telemetry processed the failing span
	if !strings.Contains(logOutput, "Error[") {
		t.Error("logOutput should contain \"Error[\"")
	}

	// Verify the span was exported
	spans := f.waitAndGetSpans()
	if len(spans) != 1 {
		t.Errorf("got %d spans, want 1", len(spans))
	}
}

func TestPathTelemetry_MetricCapture(t *testing.T) {
	// Test that verifies we can capture and verify metric calls using OTel's built-in test reader

	testCases := []struct {
		name                   string
		attrs                  map[string]interface{}
		spanStatus             codes.Code
		expectCounterMetrics   bool
		expectHistogramMetrics bool
		expectedFeatureName    string
		expectedError          string
	}{
		{
			name: "failing span captures metrics correctly",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:path":            "/{chatFlow,t:flow}/{generateResponse,t:action}",
				"genkit:isFailureSource": true,
				"genkit:state":           "error",
				"genkit:sessionId":       "session-123",
				"genkit:threadName":      "thread-456",
			},
			spanStatus:             codes.Error,
			expectCounterMetrics:   true,
			expectHistogramMetrics: true,
			expectedFeatureName:    "chatFlow",
			expectedError:          "Error",
		},
		{
			name: "non-failure-source span captures no metrics",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:path":            "/{testFlow,t:flow}/{myAction,t:action}",
				"genkit:isFailureSource": false,
				"genkit:state":           "error",
			},
			spanStatus:             codes.Error,
			expectCounterMetrics:   false,
			expectHistogramMetrics: false,
		},
		{
			name: "success span captures no metrics",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:path":            "/{testFlow,t:flow}/{myAction,t:action}",
				"genkit:isFailureSource": true,
				"genkit:state":           "success",
			},
			spanStatus:             codes.Ok,
			expectCounterMetrics:   false,
			expectHistogramMetrics: false,
		},
		{
			name: "span without path captures no metrics",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:isFailureSource": true,
				"genkit:state":           "error",
			},
			spanStatus:             codes.Error,
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

			// Create path telemetry (it will use the global meter provider)
			pathTel := NewPathTelemetry()
			f := newTestFixture(t, true, pathTel) // Enable logging for telemetry processing

			f.mockExporter.Reset()

			// Create span using the TracerProvider - this will flow through path telemetry
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-path-span")

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

			span.SetStatus(tc.spanStatus, "Test error")
			span.End() // This triggers the pipeline including path telemetry

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
				counterMetric := findMetric(&resourceMetrics, "genkit/feature/path/requests")
				if counterMetric == nil {
					t.Error("Expected counter metric to be recorded")
				} else {
					expectedAttrs := map[string]interface{}{
						"featureName": tc.expectedFeatureName,
						"status":      "failure",
						"source":      "go",
					}
					if tc.expectedError != "" {
						expectedAttrs["error"] = tc.expectedError
					}
					verifyCounterMetric(t, counterMetric, expectedAttrs)
				}
			}

			// Verify histogram metrics
			if tc.expectHistogramMetrics {
				histogramMetric := findMetric(&resourceMetrics, "genkit/feature/path/latency")
				if histogramMetric == nil {
					t.Error("Expected histogram metric to be recorded")
				} else {
					expectedAttrs := map[string]interface{}{
						"featureName": tc.expectedFeatureName,
						"status":      "failure",
						"source":      "go",
					}
					if tc.expectedError != "" {
						expectedAttrs["error"] = tc.expectedError
					}
					verifyHistogramMetric(t, histogramMetric, expectedAttrs)
				}
			}

			if !tc.expectCounterMetrics && !tc.expectHistogramMetrics {
				// Should have no path metrics
				counterMetric := findMetric(&resourceMetrics, "genkit/feature/path/requests")
				histogramMetric := findMetric(&resourceMetrics, "genkit/feature/path/latency")
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

func TestPathTelemetry_ComprehensiveScenarios(t *testing.T) {
	// Test multiple path telemetry scenarios using the proper pipeline integration

	pathTel := NewPathTelemetry()
	f := newTestFixture(t, true, pathTel) // Enable logging for path telemetry tests

	testCases := []struct {
		name         string
		attrs        map[string]interface{}
		spanStatus   codes.Code
		expectLog    bool
		expectedText string
	}{
		{
			name: "failing span with failure source",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:path":            "/{chatFlow,t:flow}/{generateResponse,t:action}",
				"genkit:isFailureSource": true,
				"genkit:state":           "error",
				"genkit:sessionId":       "session-123",
				"genkit:threadName":      "thread-456",
			},
			spanStatus:   codes.Error,
			expectLog:    true,
			expectedText: "Error[",
		},
		{
			name: "failing span without failure source",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:path":            "/{testFlow,t:flow}/{myAction,t:action}",
				"genkit:isFailureSource": false,
				"genkit:state":           "error",
			},
			spanStatus:   codes.Error,
			expectLog:    false,
			expectedText: "",
		},
		{
			name: "success span skipped",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:path":            "/{testFlow,t:flow}/{myAction,t:action}",
				"genkit:isFailureSource": true,
				"genkit:state":           "success",
			},
			spanStatus:   codes.Ok,
			expectLog:    false,
			expectedText: "",
		},
		{
			name: "span without path skipped",
			attrs: map[string]interface{}{
				"genkit:type":            "flow",
				"genkit:isFailureSource": true,
				"genkit:state":           "error",
			},
			spanStatus:   codes.Error,
			expectLog:    false,
			expectedText: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			f.mockExporter.Reset()

			// Set up log capture
			logBuf := setupLogCapture(t)

			// Create span using the TracerProvider - this triggers the full pipeline
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-path-span")

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

			span.SetStatus(tc.spanStatus, "Test error")
			span.End() // This triggers the pipeline including path telemetry

			// Get captured logs
			logOutput := logBuf.String()

			// Verify spans were processed
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}

			// Check logging behavior
			if tc.expectLog {
				if !strings.Contains(logOutput, tc.expectedText) {
					t.Errorf("Expected log containing %q but got: %q", tc.expectedText, logOutput)
				}
			} else {
				if strings.Contains(logOutput, "Error[") {
					t.Error("Should not log errors for non-qualifying spans")
				}
			}
		})
	}
}

func TestPathTelemetry_LatencyVerification(t *testing.T) {
	// Specific test to verify that latency measurement actually works correctly

	reader := sdkmetric.NewManualReader()
	testMeterProvider := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	originalProvider := otel.GetMeterProvider()
	otel.SetMeterProvider(testMeterProvider)
	defer otel.SetMeterProvider(originalProvider)

	pathTel := NewPathTelemetry()
	f := newTestFixture(t, true, pathTel) // Enable logging for telemetry processing

	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "latency-test-span")

	span.SetAttributes(
		attribute.String("genkit:type", "flow"), // Required for telemetry processing
		attribute.String("genkit:path", "/{testFlow,t:flow}/{latencyTest,t:action}"),
		attribute.Bool("genkit:isFailureSource", true),
		attribute.String("genkit:state", "error"),
		attribute.String("genkit:sessionId", "session-123"),
		attribute.String("genkit:threadName", "thread-456"),
	)
	span.SetStatus(codes.Error, "Test latency error")

	// Add a small delay to ensure measurable latency
	time.Sleep(2 * time.Millisecond)

	span.End()

	// Collect metrics
	var resourceMetrics metricdata.ResourceMetrics
	err := reader.Collect(ctx, &resourceMetrics)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify latency histogram
	histogramMetric := findMetric(&resourceMetrics, "genkit/feature/path/latency")
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
			if dp.Sum < 2.0 {
				t.Errorf("Sum = %v, want >= 2.0", dp.Sum)
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
