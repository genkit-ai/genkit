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
	"regexp"

	"github.com/firebase/genkit/go/internal"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// EngagementTelemetry implements telemetry collection for user engagement (feedback/acceptance)
type EngagementTelemetry struct {
	feedbackCounter   *MetricCounter // genkit/engagement/feedback
	acceptanceCounter *MetricCounter // genkit/engagement/acceptance
}

// NewEngagementTelemetry creates a new engagement telemetry module with required metrics
func NewEngagementTelemetry() *EngagementTelemetry {
	n := func(name string) string { return internalMetricNamespaceWrap("engagement", name) }

	return &EngagementTelemetry{
		feedbackCounter: NewMetricCounter(n("feedback"), MetricCounterOptions{
			Description: "Counts calls to genkit flows.",
			Unit:        "1",
		}),
		acceptanceCounter: NewMetricCounter(n("acceptance"), MetricCounterOptions{
			Description: "Tracks unique flow paths per flow.",
			Unit:        "1",
		}),
	}
}

// Tick processes a span for engagement telemetry
func (e *EngagementTelemetry) Tick(span sdktrace.ReadOnlySpan, logInputOutput bool, projectID string) {
	attributes := span.Attributes()
	subtype := extractStringAttribute(attributes, "genkit:metadata:subtype")

	switch subtype {
	case "userFeedback":
		e.writeUserFeedback(span, projectID)
	case "userAcceptance":
		e.writeUserAcceptance(span, projectID)
	default:
		return
	}
}

// writeUserFeedback records metrics and logs for user feedback
func (e *EngagementTelemetry) writeUserFeedback(span sdktrace.ReadOnlySpan, projectID string) {
	ctx := trace.ContextWithSpanContext(context.Background(), span.SpanContext())
	attributes := span.Attributes()
	name := e.extractTraceName(attributes)

	feedbackValue := extractStringAttribute(attributes, "genkit:metadata:feedbackValue")
	textFeedback := extractStringAttribute(attributes, "genkit:metadata:textFeedback")
	hasText := textFeedback != ""

	dimensions := map[string]interface{}{
		"name":          name,
		"value":         feedbackValue,
		"hasText":       hasText,
		"source":        "go",
		"sourceVersion": internal.Version,
	}
	e.feedbackCounter.Add(1, dimensions)

	sharedMetadata := createCommonLogAttributes(span, projectID)
	logData := map[string]interface{}{
		"feedbackValue": feedbackValue,
	}

	for k, v := range sharedMetadata {
		logData[k] = v
	}

	if hasText {
		logData["textFeedback"] = truncate(textFeedback)
	}

	slog.InfoContext(ctx, fmt.Sprintf("[genkit] UserFeedback[%s]", name), "data", logData)
}

// writeUserAcceptance records metrics and logs for user acceptance
func (e *EngagementTelemetry) writeUserAcceptance(span sdktrace.ReadOnlySpan, projectID string) {
	ctx := trace.ContextWithSpanContext(context.Background(), span.SpanContext())
	attributes := span.Attributes()
	name := e.extractTraceName(attributes)

	acceptanceValue := extractStringAttribute(attributes, "genkit:metadata:acceptanceValue")

	dimensions := map[string]interface{}{
		"name":          name,
		"value":         acceptanceValue,
		"source":        "go",
		"sourceVersion": internal.Version,
	}
	e.acceptanceCounter.Add(1, dimensions)

	sharedMetadata := createCommonLogAttributes(span, projectID)
	logData := map[string]interface{}{
		"acceptanceValue": acceptanceValue,
	}

	for k, v := range sharedMetadata {
		logData[k] = v
	}

	slog.InfoContext(ctx, fmt.Sprintf("[genkit] UserAcceptance[%s]", name), "data", logData)
}

// Helper functions

// extractTraceName extracts trace name from path using regex
func (e *EngagementTelemetry) extractTraceName(attributes []attribute.KeyValue) string {
	path := extractStringAttribute(attributes, "genkit:path")
	if path == "" || path == "<unknown>" {
		return "<unknown>"
	}

	// Extract the final action name from path using regex pattern to find the last /{...}
	re := regexp.MustCompile(`/{([^}]+)}[^}]*$`)
	matches := re.FindStringSubmatch(path)
	if len(matches) > 1 {
		return matches[1]
	}

	return "<unknown>"
}
