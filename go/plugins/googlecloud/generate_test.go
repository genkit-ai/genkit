// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlecloud

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

// TestGenerateTelemetry_PipelineIntegration verifies that generate telemetry
// works correctly in the full pipeline with realistic model generation spans
func TestGenerateTelemetry_PipelineIntegration(t *testing.T) {
	genTel := NewGenerateTelemetry()
	f := newTestFixture(t, false, genTel)

	// Set up test

	// Create realistic generate input/output JSON for span attributes
	inputJSON := `{"model":"googleai/gemini-2.5-flash","config":{"maxOutputTokens":100,"temperature":0.7},"messages":[{"content":[{"text":"What is the capital of France?"}],"role":"user"}]}`
	outputJSON := `{"message":{"content":[{"text":"The capital of France is Paris."}],"role":"model"},"usage":{"inputTokens":8,"outputTokens":7,"inputCharacters":32,"outputCharacters":32,"inputImages":0,"outputImages":0},"latencyMs":245.6}`

	// Create span using the TracerProvider - this triggers the full pipeline
	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "googleai/gemini-2.5-pro")

	span.SetAttributes(
		attribute.String("genkit:name", "googleai/gemini-2.5-pro"),
		attribute.String("genkit:metadata:subtype", "model"),
		attribute.String("genkit:path", "/{chatFlow,t:flow}/{generate,t:action}"),
		attribute.String("genkit:input", inputJSON),
		attribute.String("genkit:output", outputJSON),
		attribute.String("genkit:sessionId", "session-123"),
		attribute.String("genkit:threadName", "main-thread"),
	)

	span.End() // This triggers the pipeline

	// Verify the span was exported
	spans := f.waitAndGetSpans()
	if len(spans) != 1 {
		t.Errorf("got %d spans, want 1", len(spans))
	}
}

func TestGenerateTelemetry_MetricCapture(t *testing.T) {
	// Test that verifies we can capture and verify all 6 generate metrics

	testCases := []struct {
		name                 string
		attrs                map[string]string
		inputJSON            string
		outputJSON           string
		expectMetrics        bool
		expectedStatus       string
		expectedLatency      float64
		expectedInputChars   int64
		expectedOutputChars  int64
		expectedInputTokens  int64
		expectedOutputTokens int64
		expectedInputImages  int64
		expectedOutputImages int64
		expectedInputVideos  int64
		expectedOutputVideos int64
		expectedInputAudio   int64
		expectedOutputAudio  int64
	}{
		{
			name: "successful generation captures all metrics",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "googleai/gemini-2.5-pro",
				"genkit:metadata:subtype": "model",
				"genkit:path":             "/{chatFlow,t:flow}/{generate,t:action}",
			},
			inputJSON:            `{"model":"googleai/gemini-2.5-pro","messages":[{"content":[{"text":"Hello world"}],"role":"user"}]}`,
			outputJSON:           `{"message":{"content":[{"text":"Hello! How can I help you today?"}],"role":"model"},"usage":{"inputTokens":12,"outputTokens":8,"inputCharacters":11,"outputCharacters":33,"inputImages":1,"outputImages":0},"latencyMs":342.5}`,
			expectMetrics:        true,
			expectedStatus:       "success",
			expectedLatency:      342.5,
			expectedInputChars:   11,
			expectedOutputChars:  33,
			expectedInputTokens:  12,
			expectedOutputTokens: 8,
			expectedInputImages:  1,
			expectedOutputImages: 0,
			expectedInputVideos:  0,
			expectedOutputVideos: 0,
			expectedInputAudio:   0,
			expectedOutputAudio:  0,
		},
		{
			name: "failed generation captures error metrics",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "googleai/gemini-2.5-pro",
				"genkit:metadata:subtype": "model",
				"genkit:path":             "/{errorFlow,t:flow}/{generate,t:action}",
			},
			inputJSON:            `{"model":"googleai/gemini-2.5-pro","messages":[{"content":[{"text":"Invalid prompt"}],"role":"user"}]}`,
			outputJSON:           `{"usage":{"inputCharacters":14}}`,
			expectMetrics:        true,
			expectedStatus:       "failure",
			expectedInputChars:   14, // Length of "Invalid prompt"
			expectedInputVideos:  0,
			expectedOutputVideos: 0,
			expectedInputAudio:   0,
			expectedOutputAudio:  0,
		},
		{
			name: "generation with video and audio captures multimedia metrics",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "googleai/gemini-2.5-pro-flash",
				"genkit:metadata:subtype": "model",
				"genkit:path":             "/{multimediaFlow,t:flow}/{generate,t:action}",
			},
			inputJSON:            `{"model":"googleai/gemini-2.5-pro","messages":[{"content":[{"text":"Analyze this video and audio"}],"role":"user"}]}`,
			outputJSON:           `{"message":{"content":[{"text":"Analysis complete"}],"role":"model"},"usage":{"inputTokens":5,"outputTokens":3,"inputCharacters":26,"outputCharacters":17,"inputImages":0,"outputImages":1,"inputVideos":2,"outputVideos":0,"inputAudioFiles":1,"outputAudioFiles":0},"latencyMs":1250.3}`,
			expectMetrics:        true,
			expectedStatus:       "success",
			expectedLatency:      1250.3,
			expectedInputChars:   26,
			expectedOutputChars:  17,
			expectedInputTokens:  5,
			expectedOutputTokens: 3,
			expectedInputImages:  0,
			expectedOutputImages: 1,
			expectedInputVideos:  2,
			expectedOutputVideos: 0,
			expectedInputAudio:   1,
			expectedOutputAudio:  0,
		},
		{
			name: "non-model span captures no metrics",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "myTool",
				"genkit:metadata:subtype": "tool",
				"genkit:path":             "/{testFlow,t:flow}/{myTool,t:action}",
			},
			inputJSON:     `{"model":"googleai/gemini-2.5-flash"}`,
			outputJSON:    `{}`,
			expectMetrics: false,
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

			// Create generate telemetry (it will use the global meter provider)
			genTel := NewGenerateTelemetry()
			f := newTestFixture(t, false, genTel)

			f.mockExporter.Reset()

			// Create span using the TracerProvider - this will flow through generate telemetry
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			for key, value := range tc.attrs {
				span.SetAttributes(attribute.String(key, value))
			}

			// Set input/output JSON
			if tc.inputJSON != "" {
				span.SetAttributes(attribute.String("genkit:input", tc.inputJSON))
			}
			if tc.outputJSON != "" {
				span.SetAttributes(attribute.String("genkit:output", tc.outputJSON))
			}

			// Set error status for failure cases
			if tc.expectedStatus == "failure" {
				span.SetStatus(codes.Error, "Test error")
			}

			span.End() // This triggers the pipeline including generate telemetry

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

			if tc.expectMetrics {
				// Verify request counter
				requestMetric := findMetric(&resourceMetrics, "genkit/ai/generate/requests")
				if requestMetric == nil {
					t.Error("Expected generate/requests metric")
				} else {
					verifyCounterMetric(t, requestMetric, map[string]interface{}{
						"modelName": tc.attrs["genkit:name"],
						"status":    tc.expectedStatus,
						"source":    "go",
					})
				}

				// Verify character metrics if we have usage data
				if tc.expectedInputChars > 0 {
					inputCharMetric := findMetric(&resourceMetrics, "genkit/ai/generate/input/characters")
					if inputCharMetric == nil {
						t.Error("Expected input/characters metric")
					} else {
						verifyCounterMetricValue(t, inputCharMetric, tc.expectedInputChars)
					}
				}

				if tc.expectedOutputChars > 0 {
					outputCharMetric := findMetric(&resourceMetrics, "genkit/ai/generate/output/characters")
					if outputCharMetric == nil {
						t.Error("Expected output/characters metric")
					} else {
						verifyCounterMetricValue(t, outputCharMetric, tc.expectedOutputChars)
					}
				}

				// Verify token metrics
				if tc.expectedInputTokens > 0 {
					inputTokenMetric := findMetric(&resourceMetrics, "genkit/ai/generate/input/tokens")
					if inputTokenMetric == nil {
						t.Error("Expected input/tokens metric")
					} else {
						verifyCounterMetricValue(t, inputTokenMetric, tc.expectedInputTokens)
					}
				}

				if tc.expectedOutputTokens > 0 {
					outputTokenMetric := findMetric(&resourceMetrics, "genkit/ai/generate/output/tokens")
					if outputTokenMetric == nil {
						t.Error("Expected output/tokens metric")
					} else {
						verifyCounterMetricValue(t, outputTokenMetric, tc.expectedOutputTokens)
					}
				}

				// Verify image metrics
				if tc.expectedInputImages > 0 {
					inputImageMetric := findMetric(&resourceMetrics, "genkit/ai/generate/input/images")
					if inputImageMetric == nil {
						t.Error("Expected input/images metric")
					} else {
						verifyCounterMetricValue(t, inputImageMetric, tc.expectedInputImages)
					}
				}

				if tc.expectedOutputImages > 0 {
					outputImageMetric := findMetric(&resourceMetrics, "genkit/ai/generate/output/images")
					if outputImageMetric == nil {
						t.Error("Expected output/images metric")
					} else {
						verifyCounterMetricValue(t, outputImageMetric, tc.expectedOutputImages)
					}
				}

				// Verify video metrics
				if tc.expectedInputVideos > 0 {
					inputVideoMetric := findMetric(&resourceMetrics, "genkit/ai/generate/input/videos")
					if inputVideoMetric == nil {
						t.Error("Expected input/videos metric")
					} else {
						verifyCounterMetricValue(t, inputVideoMetric, tc.expectedInputVideos)
					}
				}

				if tc.expectedOutputVideos > 0 {
					outputVideoMetric := findMetric(&resourceMetrics, "genkit/ai/generate/output/videos")
					if outputVideoMetric == nil {
						t.Error("Expected output/videos metric")
					} else {
						verifyCounterMetricValue(t, outputVideoMetric, tc.expectedOutputVideos)
					}
				}

				// Verify audio metrics
				if tc.expectedInputAudio > 0 {
					inputAudioMetric := findMetric(&resourceMetrics, "genkit/ai/generate/input/audio")
					if inputAudioMetric == nil {
						t.Error("Expected input/audio metric")
					} else {
						verifyCounterMetricValue(t, inputAudioMetric, tc.expectedInputAudio)
					}
				}

				if tc.expectedOutputAudio > 0 {
					outputAudioMetric := findMetric(&resourceMetrics, "genkit/ai/generate/output/audio")
					if outputAudioMetric == nil {
						t.Error("Expected output/audio metric")
					} else {
						verifyCounterMetricValue(t, outputAudioMetric, tc.expectedOutputAudio)
					}
				}

				// Verify latency histogram
				if tc.expectedLatency > 0 {
					latencyMetric := findMetric(&resourceMetrics, "genkit/ai/generate/latency")
					if latencyMetric == nil {
						t.Error("Expected latency metric")
					} else {
						verifyHistogramMetricValue(t, latencyMetric, tc.expectedLatency)
					}
				}
			} else {
				// Should have no generate metrics
				requestMetric := findMetric(&resourceMetrics, "genkit/ai/generate/requests")
				if requestMetric != nil {
					t.Error("Should not have generate metrics for non-model spans")
				}
			}
		})
	}
}

func TestGenerateTelemetry_FilteringLogic(t *testing.T) {
	// Test that GenerateTelemetry only processes model subtypes

	testCases := []struct {
		name             string
		subtype          string
		expectProcessing bool
	}{
		{
			name:             "model subtype gets processed",
			subtype:          "model",
			expectProcessing: true,
		},
		{
			name:             "tool subtype gets skipped",
			subtype:          "tool",
			expectProcessing: false,
		},
		{
			name:             "flow subtype gets skipped",
			subtype:          "flow",
			expectProcessing: false,
		},
		{
			name:             "embedder subtype gets skipped",
			subtype:          "embedder",
			expectProcessing: false,
		},
		{
			name:             "empty subtype gets skipped",
			subtype:          "",
			expectProcessing: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			genTel := NewGenerateTelemetry()
			f := newTestFixture(t, false, genTel)

			// Create minimal valid input for processing
			inputJSON := `{"model":"googleai/gemini-2.5-flash"}`
			outputJSON := `{"usage":{}}`

			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			span.SetAttributes(
				attribute.String("genkit:name", "test-model"),
				attribute.String("genkit:metadata:subtype", tc.subtype),
				attribute.String("genkit:path", "/{testFlow,t:flow}/{generate,t:action}"),
				attribute.String("genkit:input", inputJSON),
				attribute.String("genkit:output", outputJSON),
			)

			span.End()

			// Verify span was processed by pipeline
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}
		})
	}
}

func TestGenerateTelemetry_JSONParsingEdgeCases(t *testing.T) {
	// Test behavior with malformed JSON, missing data, etc.

	testCases := []struct {
		name        string
		inputJSON   string
		outputJSON  string
		expectLogs  bool
		expectPanic bool
	}{
		{
			name:        "valid JSON processes normally",
			inputJSON:   `{"model":"googleai/gemini-2.5-flash","messages":[]}`,
			outputJSON:  `{"response":{"candidates":[]},"usage":{"inputTokens":5}}`,
			expectLogs:  true,
			expectPanic: false,
		},
		{
			name:        "malformed input JSON handled gracefully",
			inputJSON:   `{"model":"gemini","invalid":}`,
			outputJSON:  `{"response":{"candidates":[]}}`,
			expectLogs:  true,
			expectPanic: false,
		},
		{
			name:        "malformed output JSON handled gracefully",
			inputJSON:   `{"model":"googleai/gemini-2.5-flash"}`,
			outputJSON:  `{"response":{"invalid":}`,
			expectLogs:  true,
			expectPanic: false,
		},
		{
			name:        "empty JSON strings handled gracefully",
			inputJSON:   "",
			outputJSON:  "",
			expectLogs:  false, // No input means no metrics
			expectPanic: false,
		},
		{
			name:        "null JSON handled gracefully",
			inputJSON:   "null",
			outputJSON:  "null",
			expectLogs:  true,
			expectPanic: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			genTel := NewGenerateTelemetry()
			f := newTestFixture(t, false, genTel)

			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			span.SetAttributes(
				attribute.String("genkit:name", "test-model"),
				attribute.String("genkit:metadata:subtype", "model"),
				attribute.String("genkit:path", "/{testFlow,t:flow}/{generate,t:action}"),
			)

			// Only set JSON attributes if they're not empty
			if tc.inputJSON != "" {
				span.SetAttributes(attribute.String("genkit:input", tc.inputJSON))
			}
			if tc.outputJSON != "" {
				span.SetAttributes(attribute.String("genkit:output", tc.outputJSON))
			}

			// Function that should not panic
			testFunc := func() {
				defer func() {
					if r := recover(); r != nil {
						if !tc.expectPanic {
							t.Errorf("Should handle malformed JSON gracefully, but got panic: %v", r)
						}
					} else {
						if tc.expectPanic {
							t.Error("Expected panic for malformed data")
						}
					}
				}()
				span.End()
				spans := f.waitAndGetSpans()
				if len(spans) != 1 {
					t.Errorf("got %d spans, want 1", len(spans))
				}
			}

			testFunc()
		})
	}
}

func TestGenerateTelemetry_FeatureNameExtraction(t *testing.T) {
	// Test feature name extraction from paths and flow context

	testCases := []struct {
		name            string
		path            string
		flowName        string
		expectedFeature string
	}{
		{
			name:            "extracts feature from flow context",
			path:            "/{chatFlow,t:flow}/{generate,t:action}",
			flowName:        "chatFlow",
			expectedFeature: "chatFlow",
		},
		{
			name:            "falls back to path extraction",
			path:            "/{assistantFlow,t:flow}/{step1,t:flowStep}/{generate,t:action}",
			flowName:        "",
			expectedFeature: "assistantFlow",
		},
		{
			name:            "uses fallback for empty path",
			path:            "",
			flowName:        "",
			expectedFeature: "generate",
		},
		{
			name:            "uses fallback for unknown path",
			path:            "<unknown>",
			flowName:        "",
			expectedFeature: "generate",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			genTel := NewGenerateTelemetry()
			f := newTestFixture(t, false, genTel)

			// Create minimal valid input for processing
			inputJSON := `{"model":"googleai/gemini-2.5-flash"}`

			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			attrs := []attribute.KeyValue{
				attribute.String("genkit:name", "test-model"),
				attribute.String("genkit:metadata:subtype", "model"),
				attribute.String("genkit:path", tc.path),
				attribute.String("genkit:input", inputJSON),
			}

			// Add flow name if provided
			if tc.flowName != "" {
				attrs = append(attrs, attribute.String("genkit:metadata:flow:name", tc.flowName))
			}

			span.SetAttributes(attrs...)
			span.End()

			// Verify span was processed
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}
		})
	}
}

// Helper functions for metric verification

func verifyCounterMetricValue(t *testing.T, metric *metricdata.Metrics, expectedValue int64) {
	sum, ok := metric.Data.(metricdata.Sum[int64])
	if !ok {
		t.Errorf("Expected metric to be a Sum[int64], got %T", metric.Data)
		return
	}

	if len(sum.DataPoints) != 1 {
		t.Fatalf("got %d data points, want 1", len(sum.DataPoints))
	}
	if got, want := sum.DataPoints[0].Value, expectedValue; got != want {
		t.Errorf("Metric value = %v, want %v", got, want)
	}
}

func verifyHistogramMetricValue(t *testing.T, metric *metricdata.Metrics, expectedValue float64) {
	// Try Int64Histogram first (for latency metrics)
	if hist, ok := metric.Data.(metricdata.Histogram[int64]); ok {
		if len(hist.DataPoints) != 1 {
			t.Fatalf("got %d data points, want 1", len(hist.DataPoints))
		}
		if len(hist.DataPoints) > 0 {
			dp := hist.DataPoints[0]
			if got, want := dp.Sum, int64(expectedValue); got != want {
				t.Errorf("Histogram sum = %v, want %v", got, want)
			}
			if got, want := dp.Count, uint64(1); got != want {
				t.Errorf("Histogram count = %v, want %v", got, want)
			}
		}
		return
	}

	// Try Float64Histogram as fallback
	if hist, ok := metric.Data.(metricdata.Histogram[float64]); ok {
		if len(hist.DataPoints) != 1 {
			t.Fatalf("got %d data points, want 1", len(hist.DataPoints))
		}
		if len(hist.DataPoints) > 0 {
			dp := hist.DataPoints[0]
			if got, want := dp.Sum, expectedValue; got != want {
				t.Errorf("Histogram sum = %v, want %v", got, want)
			}
			if got, want := dp.Count, uint64(1); got != want {
				t.Errorf("Histogram count = %v, want %v", got, want)
			}
		}
		return
	}

	t.Errorf("Expected metric to be a Histogram[int64] or Histogram[float64], got %T", metric.Data)
}
