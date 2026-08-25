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

package tracing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
)

// warnTelemetryUnreachableOnce gates a single unreachable-server warning
// shared by the trace and log export paths, so a dev session with a dead
// telemetry server hears about it once instead of once per pipeline (or per
// span).
var warnTelemetryUnreachableOnce sync.Once

// warnTelemetryUnreachable warns once that the telemetry server cannot be
// reached. It writes through diag: export-pipeline diagnostics must stay off
// the export path itself (see [diag]).
func warnTelemetryUnreachable(err error) {
	warnTelemetryUnreachableOnce.Do(func() {
		diag.Warn("cannot reach telemetry server; traces and logs will not appear in the Dev UI", "error", err)
	})
}

// reportTraceSaveError logs a trace-export failure: the first one loudly,
// the rest at debug level.
func reportTraceSaveError(err error) {
	warnTelemetryUnreachable(err)
	slog.Debug("failed to save trace to telemetry server", "error", err)
}

type TelemetryClient interface {
	Save(ctx context.Context, trace *Data) error
}

// TestOnlyTelemetryClient is a test-only implementation of TelemetryClient that stores traces in memory.
type TestOnlyTelemetryClient struct {
	Traces map[string]*Data
}

// NewTestOnlyTelemetryClient creates a new in-memory telemetry client for testing.
func NewTestOnlyTelemetryClient() *TestOnlyTelemetryClient {
	return &TestOnlyTelemetryClient{
		Traces: make(map[string]*Data),
	}
}

// Save saves the data to an in-memory store.
func (c *TestOnlyTelemetryClient) Save(ctx context.Context, trace *Data) error {
	if trace == nil {
		return fmt.Errorf("trace cannot be nil")
	}
	if trace.TraceID == "" {
		return fmt.Errorf("trace ID cannot be empty")
	}
	if existing, ok := c.Traces[trace.TraceID]; ok {
		for _, span := range trace.Spans {
			existing.Spans[span.SpanID] = span
		}
		if existing.DisplayName == "" {
			existing.DisplayName = trace.DisplayName
		}
	} else {
		c.Traces[trace.TraceID] = trace
	}
	return nil
}

type httpTelemetryClient struct {
	url string
}

// NewHTTPTelemetryClient creates a new telemetry client that sends traces to a telemetry server at the given URL.
func NewHTTPTelemetryClient(url string) *httpTelemetryClient {
	return &httpTelemetryClient{url: url}
}

// Save saves the trace data by making a call to the telemetry server.
func (c *httpTelemetryClient) Save(ctx context.Context, trace *Data) error {
	return c.post(ctx, "/api/traces", trace)
}

// SaveLogs sends a batch of log records to the telemetry server, which
// accepts them on its OTLP ingestion endpoint and serves them to the Dev UI.
func (c *httpTelemetryClient) SaveLogs(ctx context.Context, logs *otlpLogsPayload) error {
	return c.post(ctx, "/api/otlp", logs)
}

// post sends v as JSON to the telemetry server endpoint at path.
func (c *httpTelemetryClient) post(ctx context.Context, path string, v any) error {
	if c.url == "" {
		return nil
	}
	body, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("failed to marshal telemetry data: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", c.url+path, bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	return nil
}
