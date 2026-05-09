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

// The googlecloud package supports telemetry (tracing, metrics and logging) using
// Google Cloud services.
package googlecloud

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"cloud.google.com/go/logging"
)

// TestHandler tests the Google Cloud slog handler functionality.
//
// NOTE: We use a custom test instead of slogtest.TestHandler because our handler
// is intentionally designed for Google Cloud Logging.
// The main requirement we optimized against is matching the bespoke log structure that AIM expects.

func TestHandler(t *testing.T) {
	// Test basic functionality that our handler does support
	var results []logging.Entry
	f := func(e logging.Entry) {
		results = append(results, e)
	}

	handler := newHandler(slog.LevelInfo, f, "test-project")

	// Test basic message logging
	ctx := context.Background()
	record := slog.NewRecord(time.Now(), slog.LevelInfo, "test message", 0)

	err := handler.Handle(ctx, record)
	if err != nil {
		t.Fatalf("Handler.Handle failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 log entry, got %d", len(results))
	}

	entry := results[0]

	// Verify the entry structure matches our expectations
	if entry.Severity != logging.Info {
		t.Errorf("Expected severity Info, got %v", entry.Severity)
	}

	payload, ok := entry.Payload.(map[string]interface{})
	if !ok {
		t.Fatalf("Expected payload to be map[string]interface{}, got %T", entry.Payload)
	}

	if payload["message"] != "test message" {
		t.Errorf("Expected message 'test message', got %v", payload["message"])
	}

	// Test that our handler correctly processes metadata attributes
	results = nil
	record = slog.NewRecord(time.Now(), slog.LevelInfo, "test with metadata", 0)
	record.Add("metadata", map[string]interface{}{"key": "value"})

	err = handler.Handle(ctx, record)
	if err != nil {
		t.Fatalf("Handler.Handle with metadata failed: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 log entry with metadata, got %d", len(results))
	}

	entry = results[0]
	payload, ok = entry.Payload.(map[string]interface{})
	if !ok {
		t.Fatalf("Expected payload with metadata to be map[string]interface{}, got %T", entry.Payload)
	}

	metadata, ok := payload["metadata"].(map[string]interface{})
	if !ok {
		t.Fatalf("Expected metadata to be map[string]interface{}, got %T", payload["metadata"])
	}

	if metadata["key"] != "value" {
		t.Errorf("Expected metadata key 'value', got %v", metadata["key"])
	}
}
