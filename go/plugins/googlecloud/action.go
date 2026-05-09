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

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// ActionTelemetry implements telemetry collection for action input/output logging
type ActionTelemetry struct {
	// Note: Unlike generate and feature telemetry, action telemetry only does logging, no metrics
}

// NewActionTelemetry creates a new action telemetry module
func NewActionTelemetry() *ActionTelemetry {
	return &ActionTelemetry{}
}

// Tick processes a span for action telemetry
func (a *ActionTelemetry) Tick(span sdktrace.ReadOnlySpan, logInputOutput bool, projectID string) {
	if !logInputOutput {
		return
	}

	attributes := span.Attributes()
	actionName := extractStringAttribute(attributes, "genkit:name")
	if actionName == "" {
		actionName = "<unknown>"
	}

	subtype := extractStringAttribute(attributes, "genkit:metadata:subtype")

	if subtype != "tool" && actionName != "generate" {
		return
	}

	path := extractStringAttribute(attributes, "genkit:path")
	if path == "" {
		path = "<unknown>"
	}

	input := truncate(extractStringAttribute(attributes, "genkit:input"))
	output := truncate(extractStringAttribute(attributes, "genkit:output"))
	sessionID := extractStringAttribute(attributes, "genkit:sessionId")
	threadName := extractStringAttribute(attributes, "genkit:threadName")

	featureName := extractOuterFeatureNameFromPath(path)
	if featureName == "" || featureName == "<unknown>" {
		featureName = actionName
	}

	if input != "" {
		a.writeLog(span, "Input", featureName, path, input, projectID, sessionID, threadName)
	}

	if output != "" {
		a.writeLog(span, "Output", featureName, path, output, projectID, sessionID, threadName)
	}
}

// writeLog writes structured logs for action input/output
func (a *ActionTelemetry) writeLog(span sdktrace.ReadOnlySpan, tag, featureName, qualifiedPath, content, projectID, sessionID, threadName string) {
	ctx := trace.ContextWithSpanContext(context.Background(), span.SpanContext())
	path := truncatePath(toDisplayPath(qualifiedPath))
	sharedMetadata := createCommonLogAttributes(span, projectID)

	logData := map[string]interface{}{
		"path":          path,
		"qualifiedPath": qualifiedPath,
		"featureName":   featureName,
		"content":       content,
	}

	if sessionID != "" {
		logData["sessionId"] = sessionID
	}
	if threadName != "" {
		logData["threadName"] = threadName
	}

	for k, v := range sharedMetadata {
		logData[k] = v
	}

	slog.InfoContext(ctx, fmt.Sprintf("[genkit] %s[%s, %s]", tag, path, featureName), MetadataKey, logData)
}
