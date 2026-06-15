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

package googlecloud

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/firebase/genkit/go/internal"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// FeatureTelemetry implements telemetry collection for top-level feature requests
type FeatureTelemetry struct {
	featureCounter   *MetricCounter   // genkit/feature/requests
	featureLatencies *MetricHistogram // genkit/feature/latency
}

// NewFeatureTelemetry creates a new feature telemetry module with required metrics
func NewFeatureTelemetry() *FeatureTelemetry {
	n := func(name string) string { return internalMetricNamespaceWrap("feature", name) }

	return &FeatureTelemetry{
		featureCounter: NewMetricCounter(n("requests"), MetricCounterOptions{
			Description: "Counts calls to genkit features.",
			Unit:        "1",
		}),
		featureLatencies: NewMetricHistogram(n("latency"), MetricHistogramOptions{
			Description: "Latencies when calling Genkit features.",
			Unit:        "ms",
		}),
	}
}

// Tick processes a span for feature telemetry
func (f *FeatureTelemetry) Tick(span sdktrace.ReadOnlySpan, logInputOutput bool, projectID string) {
	attributes := span.Attributes()

	isRoot := extractBoolAttribute(attributes, "genkit:isRoot")
	if !isRoot {
		return
	}

	name := extractStringAttribute(attributes, attrGenkitName)
	path := extractStringAttribute(attributes, attrGenkitPath)
	state := extractStringAttribute(attributes, attrGenkitState)

	latencyMs := f.calculateLatencyMs(span)
	switch state {
	case statusSuccess:
		f.writeFeatureSuccess(name, latencyMs)
	case "error":
		errorName := f.extractErrorName(span)
		if errorName == "" {
			errorName = unknownValue
		}
		f.writeFeatureFailure(name, latencyMs, errorName)
	default:
		if state == "" {
			slog.Warn("Unknown feature state", "state", "<missing>", "feature", name)
		} else {
			slog.Warn("Unknown feature state", "state", state, "feature", name)
		}
		return
	}
	if logInputOutput {
		input := truncate(extractStringAttribute(attributes, attrGenkitInput))
		output := truncate(extractStringAttribute(attributes, attrGenkitOutput))
		sessionID := extractStringAttribute(attributes, attrGenkitSessionID)
		threadName := extractStringAttribute(attributes, attrGenkitThreadName)

		if input != "" {
			f.writeLog(span, "Input", name, path, input, projectID, sessionID, threadName)
		}
		if output != "" {
			f.writeLog(span, "Output", name, path, output, projectID, sessionID, threadName)
		}
	}
}

// writeFeatureSuccess records metrics for successful feature calls
func (f *FeatureTelemetry) writeFeatureSuccess(featureName string, latencyMs float64) {
	dimensions := map[string]interface{}{
		"name":             featureName,
		"status":           statusSuccess,
		fieldSource:        sourceGo,
		fieldSourceVersion: internal.Version,
	}

	f.featureCounter.Add(1, dimensions)
	f.featureLatencies.Record(latencyMs, dimensions)
}

// writeFeatureFailure records metrics for failed feature calls
func (f *FeatureTelemetry) writeFeatureFailure(featureName string, latencyMs float64, errorName string) {
	dimensions := map[string]interface{}{
		"name":             featureName,
		"status":           statusFailure,
		fieldSource:        sourceGo,
		fieldSourceVersion: internal.Version,
		"error":            errorName,
	}

	f.featureCounter.Add(1, dimensions)
	f.featureLatencies.Record(latencyMs, dimensions)
}

// writeLog writes structured logs for feature input/output
func (f *FeatureTelemetry) writeLog(span sdktrace.ReadOnlySpan, tag, featureName, qualifiedPath, content, projectID, sessionID, threadName string) {
	ctx := trace.ContextWithSpanContext(context.Background(), span.SpanContext())
	path := truncatePath(toDisplayPath(qualifiedPath))
	sharedMetadata := createCommonLogAttributes(span, projectID)

	logData := map[string]interface{}{
		"path":             path,
		fieldQualifiedPath: qualifiedPath,
		attrFeatureName:    featureName,
		fieldContent:       content,
	}

	if sessionID != "" {
		logData[fieldSessionID] = sessionID
	}
	if threadName != "" {
		logData[fieldThreadName] = threadName
	}

	for k, v := range sharedMetadata {
		logData[k] = v
	}

	slog.InfoContext(ctx, fmt.Sprintf("[genkit] %s[%s, %s]", tag, path, featureName), MetadataKey, logData)
}

// Helper functions

// calculateLatencyMs calculates the latency in milliseconds from span start/end times
func (f *FeatureTelemetry) calculateLatencyMs(span sdktrace.ReadOnlySpan) float64 {
	startTime := span.StartTime()
	endTime := span.EndTime()

	if endTime.IsZero() {
		endTime = time.Now()
	}

	duration := endTime.Sub(startTime)
	return float64(duration.Nanoseconds()) / 1e6
}

// extractErrorName extracts error information from span
func (f *FeatureTelemetry) extractErrorName(span sdktrace.ReadOnlySpan) string {
	if span.Status().Code == codes.Error {
		return span.Status().Description
	}

	for _, event := range span.Events() {
		if event.Name == "exception" {
			for _, attr := range event.Attributes {
				if string(attr.Key) == "exception.type" {
					return attr.Value.AsString()
				}
			}
		}
	}

	return ""
}
