// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlecloud

import (
	"context"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// TestActionTelemetry_PipelineIntegration verifies that action telemetry
// works correctly in the full pipeline with realistic paths
func TestActionTelemetry_PipelineIntegration(t *testing.T) {
	actionTel := NewActionTelemetry()

	// Create custom fixture with logInputOutput enabled (ActionTelemetry requires this)
	mockExporter := NewTestSpanExporter()
	adjuster := &AdjustingTraceExporter{
		exporter:          mockExporter,
		modules:           []Telemetry{actionTel},
		logInputAndOutput: true, // ActionTelemetry only works when this is enabled
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

	// Create span using the TracerProvider - this triggers the full pipeline
	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "test-tool-span")

	span.SetAttributes(
		attribute.String("genkit:type", "action"), // Required for telemetry processing
		attribute.String("genkit:name", "testTool"),
		attribute.String("genkit:metadata:subtype", "tool"),
		attribute.String("genkit:path", "/{testFlow,t:flow}/{testTool,t:action}"),
		attribute.String("genkit:input", "test input data"),
		attribute.String("genkit:output", "test output data"),
		attribute.String("genkit:sessionId", "session-123"),
		attribute.String("genkit:threadName", "main-thread"),
	)

	span.End() // This triggers the pipeline

	// Get captured logs
	logOutput := logBuf.String()

	// Verify action telemetry worked
	if !strings.Contains(logOutput, "Input[testFlow > testTool, testFlow]") {
		t.Error("logOutput should contain \"Input[testFlow > testTool, testFlow]\"")
	}
	if !strings.Contains(logOutput, "Output[testFlow > testTool, testFlow]") {
		t.Error("logOutput should contain \"Output[testFlow > testTool, testFlow]\"")
	}
	if !strings.Contains(logOutput, "test input data") {
		t.Error("logOutput should contain \"test input data\"")
	}
	if !strings.Contains(logOutput, "test output data") {
		t.Error("logOutput should contain \"test output data\"")
	}

	// Verify the span was exported
	spans := f.waitAndGetSpans()
	if len(spans) != 1 {
		t.Errorf("got %d spans, want 1", len(spans))
	}
}

func TestActionTelemetry_FilteringLogic(t *testing.T) {
	// Test that ActionTelemetry only processes spans under the right conditions

	actionTel := NewActionTelemetry()

	testCases := []struct {
		name              string
		logInputOutput    bool
		subtype           string
		actionName        string
		expectProcessing  bool
		expectedInputLog  string
		expectedOutputLog string
	}{
		{
			name:              "tool subtype with logging enabled",
			logInputOutput:    true,
			subtype:           "tool",
			actionName:        "myTool",
			expectProcessing:  true,
			expectedInputLog:  "Input[",
			expectedOutputLog: "Output[",
		},
		{
			name:              "generate action with logging enabled",
			logInputOutput:    true,
			subtype:           "model",
			actionName:        "generate",
			expectProcessing:  true,
			expectedInputLog:  "Input[",
			expectedOutputLog: "Output[",
		},
		{
			name:              "tool subtype with logging disabled",
			logInputOutput:    false,
			subtype:           "tool",
			actionName:        "myTool",
			expectProcessing:  false,
			expectedInputLog:  "",
			expectedOutputLog: "",
		},
		{
			name:              "flow subtype (not tool/generate)",
			logInputOutput:    true,
			subtype:           "flow",
			actionName:        "myFlow",
			expectProcessing:  false,
			expectedInputLog:  "",
			expectedOutputLog: "",
		},
		{
			name:              "unknown action (not tool/generate)",
			logInputOutput:    true,
			subtype:           "unknown",
			actionName:        "unknownAction",
			expectProcessing:  false,
			expectedInputLog:  "",
			expectedOutputLog: "",
		},
		{
			name:              "missing subtype but is generate",
			logInputOutput:    true,
			subtype:           "",
			actionName:        "generate",
			expectProcessing:  true,
			expectedInputLog:  "Input[",
			expectedOutputLog: "Output[",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create custom fixture with the specified logInputOutput setting
			mockExporter := NewTestSpanExporter()
			adjuster := &AdjustingTraceExporter{
				exporter:          mockExporter,
				modules:           []Telemetry{actionTel},
				logInputAndOutput: tc.logInputOutput,
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

			// Create span
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			span.SetAttributes(
				attribute.String("genkit:type", "action"), // Required for telemetry processing
				attribute.String("genkit:name", tc.actionName),
				attribute.String("genkit:metadata:subtype", tc.subtype),
				attribute.String("genkit:path", "/{testFlow,t:flow}/{testAction,t:action}"),
				attribute.String("genkit:input", "test input"),
				attribute.String("genkit:output", "test output"),
			)

			span.End()

			// Get captured logs
			logOutput := logBuf.String()

			// Verify processing behavior
			if tc.expectProcessing {
				if !strings.Contains(logOutput, tc.expectedInputLog) {
					t.Errorf("Expected input log containing %q but got: %q", tc.expectedInputLog, logOutput)
				}
				if !strings.Contains(logOutput, tc.expectedOutputLog) {
					t.Errorf("Expected output log containing %q but got: %q", tc.expectedOutputLog, logOutput)
				}
			} else {
				if strings.Contains(logOutput, "Input[") {
					t.Error("Should not have input logs")
				}
				if strings.Contains(logOutput, "Output[") {
					t.Error("Should not have output logs")
				}
			}

			// Verify span was processed regardless
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}
		})
	}
}

func TestActionTelemetry_LoggingBehavior(t *testing.T) {
	// Test various logging scenarios including missing input/output

	actionTel := NewActionTelemetry()

	testCases := []struct {
		name                  string
		attrs                 map[string]string
		expectInputLog        bool
		expectOutputLog       bool
		expectedInputLog      string
		expectedOutputLog     string
		expectedInputContent  string
		expectedOutputContent string
	}{
		{
			name: "both input and output present",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "myTool",
				"genkit:metadata:subtype": "tool",
				"genkit:path":             "/{chatFlow,t:flow}/{myTool,t:action}",
				"genkit:input":            "Hello world",
				"genkit:output":           "Hi there!",
				"genkit:sessionId":        "session-123",
				"genkit:threadName":       "thread-456",
			},
			expectInputLog:        true,
			expectOutputLog:       true,
			expectedInputLog:      "Input[chatFlow > myTool, chatFlow]",
			expectedOutputLog:     "Output[chatFlow > myTool, chatFlow]",
			expectedInputContent:  "Hello world",
			expectedOutputContent: "Hi there!",
		},
		{
			name: "only input present",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "myTool",
				"genkit:metadata:subtype": "tool",
				"genkit:path":             "/{testFlow,t:flow}/{myTool,t:action}",
				"genkit:input":            "Hello world",
				"genkit:output":           "", // Empty output
			},
			expectInputLog:        true,
			expectOutputLog:       false,
			expectedInputLog:      "Input[testFlow > myTool, testFlow]",
			expectedOutputLog:     "",
			expectedInputContent:  "Hello world",
			expectedOutputContent: "",
		},
		{
			name: "only output present",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "myTool",
				"genkit:metadata:subtype": "tool",
				"genkit:path":             "/{testFlow,t:flow}/{myTool,t:action}",
				"genkit:input":            "", // Empty input
				"genkit:output":           "Hi there!",
			},
			expectInputLog:        false,
			expectOutputLog:       true,
			expectedInputLog:      "",
			expectedOutputLog:     "Output[testFlow > myTool, testFlow]",
			expectedInputContent:  "",
			expectedOutputContent: "Hi there!",
		},
		{
			name: "neither input nor output present",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "myTool",
				"genkit:metadata:subtype": "tool",
				"genkit:path":             "/{testFlow,t:flow}/{myTool,t:action}",
				"genkit:input":            "",
				"genkit:output":           "",
			},
			expectInputLog:        false,
			expectOutputLog:       false,
			expectedInputLog:      "",
			expectedOutputLog:     "",
			expectedInputContent:  "",
			expectedOutputContent: "",
		},
		{
			name: "generate action",
			attrs: map[string]string{
				"genkit:type":   "action",
				"genkit:name":   "generate",
				"genkit:path":   "/{chatFlow,t:flow}/{generate,t:action}",
				"genkit:input":  "What is the weather?",
				"genkit:output": "The weather is sunny.",
			},
			expectInputLog:        true,
			expectOutputLog:       true,
			expectedInputLog:      "Input[chatFlow > generate, chatFlow]",
			expectedOutputLog:     "Output[chatFlow > generate, chatFlow]",
			expectedInputContent:  "What is the weather?",
			expectedOutputContent: "The weather is sunny.",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create fixture with logging enabled
			mockExporter := NewTestSpanExporter()
			adjuster := &AdjustingTraceExporter{
				exporter:          mockExporter,
				modules:           []Telemetry{actionTel},
				logInputAndOutput: true,
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

			// Create span
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			for key, value := range tc.attrs {
				span.SetAttributes(attribute.String(key, value))
			}

			span.End()

			// Get captured logs
			logOutput := logBuf.String()

			// Verify logging behavior
			if tc.expectInputLog {
				if !strings.Contains(logOutput, tc.expectedInputLog) {
					t.Errorf("Expected input log header %q but got: %q", tc.expectedInputLog, logOutput)
				}
				if tc.expectedInputContent != "" {
					if !strings.Contains(logOutput, tc.expectedInputContent) {
						t.Errorf("Expected input content %q in logs", tc.expectedInputContent)
					}
				}
			} else {
				if strings.Contains(logOutput, "Input[") {
					t.Error("Should not have input log")
				}
				if tc.expectedInputContent != "" {
					if strings.Contains(logOutput, tc.expectedInputContent) {
						t.Errorf("Should not have input content %q in logs", tc.expectedInputContent)
					}
				}
			}

			if tc.expectOutputLog {
				if !strings.Contains(logOutput, tc.expectedOutputLog) {
					t.Errorf("Expected output log header %q but got: %q", tc.expectedOutputLog, logOutput)
				}
				if tc.expectedOutputContent != "" {
					if !strings.Contains(logOutput, tc.expectedOutputContent) {
						t.Errorf("Expected output content %q in logs", tc.expectedOutputContent)
					}
				}
			} else {
				if strings.Contains(logOutput, "Output[") {
					t.Error("Should not have output log")
				}
				if tc.expectedOutputContent != "" {
					if strings.Contains(logOutput, tc.expectedOutputContent) {
						t.Errorf("Should not have output content %q in logs", tc.expectedOutputContent)
					}
				}
			}

			// Verify span was processed
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}
		})
	}
}

func TestActionTelemetry_FeatureNameExtraction(t *testing.T) {
	// Test feature name extraction vs fallback to action name

	actionTel := NewActionTelemetry()

	testCases := []struct {
		name            string
		path            string
		actionName      string
		expectedFeature string
		expectedLog     string
	}{
		{
			name:            "valid path extracts feature name",
			path:            "/{chatFlow,t:flow}/{myTool,t:action}",
			actionName:      "myTool",
			expectedFeature: "chatFlow",
			expectedLog:     "Input[chatFlow > myTool, chatFlow]",
		},
		{
			name:            "empty path falls back to action name",
			path:            "",
			actionName:      "fallbackAction",
			expectedFeature: "fallbackAction",
			expectedLog:     "Input[<unknown>, fallbackAction]",
		},
		{
			name:            "unknown path falls back to action name",
			path:            "<unknown>",
			actionName:      "anotherAction",
			expectedFeature: "anotherAction",
			expectedLog:     "Input[<unknown>, anotherAction]",
		},
		{
			name:            "path without proper format falls back",
			path:            "/simple/path/without/braces",
			actionName:      "simpleTool",
			expectedFeature: "simpleTool",
			expectedLog:     "Input[/simple/path/without/braces, simpleTool]",
		},
		{
			name:            "complex realistic path",
			path:            "/{myApp,t:flow}/{step1,t:flowStep}/{tool,t:action}",
			actionName:      "tool",
			expectedFeature: "myApp",
			expectedLog:     "Input[myApp > step1 > tool, myApp]",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create fixture with logging enabled
			mockExporter := NewTestSpanExporter()
			adjuster := &AdjustingTraceExporter{
				exporter:          mockExporter,
				modules:           []Telemetry{actionTel},
				logInputAndOutput: true,
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

			// Create span
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			span.SetAttributes(
				attribute.String("genkit:type", "action"), // Required for telemetry processing
				attribute.String("genkit:name", tc.actionName),
				attribute.String("genkit:metadata:subtype", "tool"),
				attribute.String("genkit:path", tc.path),
				attribute.String("genkit:input", "test input"),
			)

			span.End()

			// Get captured logs
			logOutput := logBuf.String()

			// Verify feature name extraction worked correctly
			if !strings.Contains(logOutput, tc.expectedLog) {
				t.Errorf("Expected log with correct feature name %q but got: %q", tc.expectedLog, logOutput)
			}

			// Verify span was processed
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}
		})
	}
}

func TestActionTelemetry_EdgeCases(t *testing.T) {
	// Test edge cases like missing attributes, very long content, etc.

	actionTel := NewActionTelemetry()

	testCases := []struct {
		name        string
		attrs       map[string]string
		expectLogs  bool
		expectedLog string
	}{
		{
			name: "missing action name",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:metadata:subtype": "tool",
				"genkit:path":             "/{testFlow,t:flow}/{tool,t:action}",
				"genkit:input":            "test input",
			},
			expectLogs:  true,
			expectedLog: "Input[testFlow > tool, testFlow]", // Should extract from path
		},
		{
			name: "missing path",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "myTool",
				"genkit:metadata:subtype": "tool",
				"genkit:input":            "test input",
			},
			expectLogs:  true,
			expectedLog: "Input[<unknown>, myTool]", // Should fallback to action name
		},
		{
			name: "missing subtype but actionName is generate",
			attrs: map[string]string{
				"genkit:type":  "action",
				"genkit:name":  "generate",
				"genkit:path":  "/{chatFlow,t:flow}/{generate,t:action}",
				"genkit:input": "test input",
			},
			expectLogs:  true,
			expectedLog: "Input[chatFlow > generate, chatFlow]",
		},
		{
			name: "session and thread info included",
			attrs: map[string]string{
				"genkit:type":             "action",
				"genkit:name":             "myTool",
				"genkit:metadata:subtype": "tool",
				"genkit:path":             "/{testFlow,t:flow}/{myTool,t:action}",
				"genkit:input":            "test input",
				"genkit:sessionId":        "session-789",
				"genkit:threadName":       "worker-thread",
			},
			expectLogs:  true,
			expectedLog: "session-789", // Should include session info in log data
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create fixture with logging enabled
			mockExporter := NewTestSpanExporter()
			adjuster := &AdjustingTraceExporter{
				exporter:          mockExporter,
				modules:           []Telemetry{actionTel},
				logInputAndOutput: true,
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

			// Create span
			ctx := context.Background()
			_, span := f.tracer.Start(ctx, "test-span")

			for key, value := range tc.attrs {
				span.SetAttributes(attribute.String(key, value))
			}

			span.End()

			// Get captured logs
			logOutput := logBuf.String()

			// Verify logging behavior
			if tc.expectLogs {
				if !strings.Contains(logOutput, tc.expectedLog) {
					t.Errorf("Expected log content %q but got: %q", tc.expectedLog, logOutput)
				}
			}

			// Verify span was processed
			spans := f.waitAndGetSpans()
			if len(spans) != 1 {
				t.Errorf("got %d spans, want 1", len(spans))
			}
		})
	}
}

func TestActionTelemetry_InputTruncation(t *testing.T) {
	// Test that verifies input content is actually truncated when it exceeds MaxLogContentLength

	actionTel := NewActionTelemetry()

	// Create fixture with logging enabled
	mockExporter := NewTestSpanExporter()
	adjuster := &AdjustingTraceExporter{
		exporter:          mockExporter,
		modules:           []Telemetry{actionTel},
		logInputAndOutput: true,
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

	// Generate content longer than MaxLogContentLength (128,000 chars)
	// Use a smaller size to test more efficiently: 130k chars
	longInput := strings.Repeat("A", 130000) // Simple repeated character, 130k chars

	// Create span with very long input
	ctx := context.Background()
	_, span := f.tracer.Start(ctx, "test-span")

	span.SetAttributes(
		attribute.String("genkit:type", "action"), // Required for telemetry processing
		attribute.String("genkit:name", "myTool"),
		attribute.String("genkit:metadata:subtype", "tool"),
		attribute.String("genkit:path", "/{testFlow,t:flow}/{myTool,t:action}"),
		attribute.String("genkit:input", longInput),
	)

	span.End()

	// Get captured logs
	logOutput := logBuf.String()

	// Verify the log header appears
	if !strings.Contains(logOutput, "Input[testFlow > myTool, testFlow]") {
		t.Error("Expected input log header \"Input[testFlow > myTool, testFlow]\"")
	}

	// Simple verification: check that the original input is not entirely present in logs
	// If truncation worked, the full 130k character string should not appear in logs
	if strings.Contains(logOutput, longInput) {
		t.Error("Full long input should not appear in logs (should be truncated)")
	}

	// Verify that some of the content appears (the beginning should be preserved)
	if !strings.Contains(logOutput, "AAAAAAAAAA") {
		t.Error("Beginning of input should appear in logs")
	}

	// Verify spans were processed
	spans := f.waitAndGetSpans()
	if len(spans) != 1 {
		t.Errorf("got %d spans, want 1", len(spans))
	}
}
