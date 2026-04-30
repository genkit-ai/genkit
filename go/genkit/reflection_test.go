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

package genkit

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
)

func inc(_ context.Context, x int) (int, error) {
	return x + 1, nil
}

func dec(_ context.Context, x int) (int, error) {
	return x - 1, nil
}

func TestReflectionServer(t *testing.T) {
	t.Run("server startup and shutdown", func(t *testing.T) {
		g := Init(context.Background())

		tc := tracing.NewTestOnlyTelemetryClient()
		tracing.WriteTelemetryImmediate(tc)

		errCh := make(chan error, 1)
		serverStartCh := make(chan struct{})

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		srv := startReflectionServer(ctx, g, errCh, serverStartCh)
		if srv == nil {
			t.Fatal("failed to start reflection server")
		}

		select {
		case err := <-errCh:
			t.Fatalf("server failed to start: %v", err)
		case <-serverStartCh:
			// Server started successfully
		case <-time.After(5 * time.Second):
			t.Fatal("timeout waiting for server to start")
		}

		if _, err := os.Stat(srv.RuntimeFilePath); err != nil {
			t.Errorf("runtime file not created: %v", err)
		}

		cancel()
		time.Sleep(100 * time.Millisecond)

		if _, err := os.Stat(srv.RuntimeFilePath); !os.IsNotExist(err) {
			t.Error("runtime file was not cleaned up")
		}
	})
}

func TestServeMux(t *testing.T) {
	g := Init(context.Background())

	tc := tracing.NewTestOnlyTelemetryClient()
	tracing.WriteTelemetryImmediate(tc)

	core.DefineAction(g.reg, "test/inc", api.ActionTypeCustom, nil, nil, inc)
	core.DefineAction(g.reg, "test/dec", api.ActionTypeCustom, nil, nil, dec)

	s := &reflectionServer{
		Server:        &http.Server{},
		activeActions: newActiveActionsMap(),
	}
	ts := httptest.NewServer(serveMux(g, s))
	s.Addr = strings.TrimPrefix(ts.URL, "http://")
	defer ts.Close()

	t.Parallel()

	t.Run("health check", func(t *testing.T) {
		res, err := http.Get(ts.URL + "/api/__health")
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			t.Errorf("health check failed: got status %d, want %d", res.StatusCode, http.StatusOK)
		}

		// Test with correct runtime ID
		res, err = http.Get(ts.URL + "/api/__health?id=" + s.runtimeID())
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			t.Errorf("health check with correct id failed: got status %d, want %d", res.StatusCode, http.StatusOK)
		}

		// Test with incorrect runtime ID
		res, err = http.Get(ts.URL + "/api/__health?id=invalid-id")
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusServiceUnavailable {
			t.Errorf("health check with incorrect id failed: got status %d, want %d", res.StatusCode, http.StatusServiceUnavailable)
		}
	})

	t.Run("list actions", func(t *testing.T) {
		res, err := http.Get(ts.URL + "/api/actions")
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()

		var actions map[string]api.ActionDesc
		if err := json.NewDecoder(res.Body).Decode(&actions); err != nil {
			t.Fatal(err)
		}

		expectedKeys := []string{"/custom/test/inc", "/custom/test/dec"}
		for _, key := range expectedKeys {
			if _, ok := actions[key]; !ok {
				t.Errorf("action %q not found in response", key)
			}
		}
	})

	t.Run("list actions resolves schema references", func(t *testing.T) {
		core.DefineSchema(g.reg, "AgentRequest", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
		})
		core.DefineAction(g.reg, "test/withRef", api.ActionTypeCustom, nil,
			core.SchemaRef("AgentRequest"),
			func(ctx context.Context, in any) (any, error) { return nil, nil })

		res, err := http.Get(ts.URL + "/api/actions")
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()

		var actions map[string]api.ActionDesc
		if err := json.NewDecoder(res.Body).Decode(&actions); err != nil {
			t.Fatal(err)
		}

		desc, ok := actions["/custom/test/withRef"]
		if !ok {
			t.Fatal("action /custom/test/withRef not found in response")
		}
		if _, hasRef := desc.InputSchema["$ref"]; hasRef {
			t.Errorf("InputSchema still contains $ref, expected resolved schema: %v", desc.InputSchema)
		}
		if got, want := desc.InputSchema["type"], "object"; got != want {
			t.Errorf("InputSchema type = %v, want %v", got, want)
		}
	})

	t.Run("run action", func(t *testing.T) {
		tests := []struct {
			name       string
			body       string
			wantStatus int
			wantResult string
		}{
			{
				name:       "valid increment",
				body:       `{"key": "/custom/test/inc", "input": 3}`,
				wantStatus: http.StatusOK,
				wantResult: "4",
			},
			{
				name:       "valid decrement",
				body:       `{"key": "/custom/test/dec", "input": 3}`,
				wantStatus: http.StatusOK,
				wantResult: "2",
			},
			{
				name:       "check telemetry labels",
				body:       `{"key": "/custom/test/dec", "input": 3,"telemetryLabels":{"test_k":"test_v"}}`,
				wantStatus: http.StatusOK,
				wantResult: "2",
			},
			{
				name:       "invalid action key",
				body:       `{"key": "/custom/test/invalid", "input": 3}`,
				wantStatus: http.StatusOK,
			},
			{
				name:       "invalid input type",
				body:       `{"key": "/custom/test/inc", "input": "not a number"}`,
				wantStatus: http.StatusOK,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				res, err := http.Post(ts.URL+"/api/runAction", "application/json", strings.NewReader(tt.body))
				if err != nil {
					t.Fatal(err)
				}
				defer res.Body.Close()

				if res.StatusCode != tt.wantStatus {
					t.Errorf("got status %d, want %d", res.StatusCode, tt.wantStatus)
					return
				}

				if tt.wantResult != "" {
					var resp runActionResponse
					if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
						t.Fatal(err)
					}
					if g := string(resp.Result); g != tt.wantResult {
						t.Errorf("got result %q, want %q", g, tt.wantResult)
					}
					if resp.Telemetry.TraceID == "" {
						t.Error("expected non-empty trace ID")
					}
				}
			})
		}
	})

	t.Run("streaming action", func(t *testing.T) {
		streamingInc := func(_ context.Context, x int, cb streamingCallback[json.RawMessage]) (int, error) {
			for i := range x {
				msg, _ := json.Marshal(i)
				if err := cb(context.Background(), msg); err != nil {
					return 0, err
				}
			}
			return x, nil
		}
		core.DefineStreamingAction(g.reg, "test/streaming", api.ActionTypeCustom, nil, nil, streamingInc)

		body := `{"key": "/custom/test/streaming", "input": 3}`
		req, err := http.NewRequest("POST", ts.URL+"/api/runAction?stream=true", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()

		scanner := bufio.NewScanner(res.Body)

		for i := range 3 {
			if !scanner.Scan() {
				t.Fatalf("expected streaming chunk %d", i)
			}
			got := scanner.Text()
			want := fmt.Sprintf("%d", i)
			if got != want {
				t.Errorf("chunk %d: got %q, want %q", i, got, want)
			}
		}

		if !scanner.Scan() {
			t.Fatal("expected final response")
		}
		var resp runActionResponse
		if err := json.Unmarshal([]byte(scanner.Text()), &resp); err != nil {
			t.Fatal(err)
		}
		if g := string(resp.Result); g != "3" {
			t.Errorf("got final result %q, want %q", g, "3")
		}
		if resp.Telemetry.TraceID == "" {
			t.Error("expected non-empty trace ID")
		}

		if scanner.Scan() {
			t.Errorf("unexpected additional data: %q", scanner.Text())
		}
		if err := scanner.Err(); err != nil {
			t.Errorf("scanner error: %v", err)
		}
	})

	t.Run("notify endpoint", func(t *testing.T) {
		body := `{
			"telemetryServerURL": "http://localhost:9999",
			"reflectionApiSpecVersion": 1
		}`
		res, err := http.Post(ts.URL+"/api/notify", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()

		if res.StatusCode != http.StatusOK {
			t.Errorf("got status %d, want %d", res.StatusCode, http.StatusOK)
		}
	})
}

// TestEarlyTraceIDTransmission verifies that trace ID headers are sent BEFORE the action completes.
//
// The key thing we're testing: headers arrive while the action is still running, not after.
// This allows clients to get the trace ID immediately for cancellation or logging.
func TestEarlyTraceIDTransmission(t *testing.T) {
	g := Init(context.Background())
	tc := tracing.NewTestOnlyTelemetryClient()
	tracing.WriteTelemetryImmediate(tc)

	actionStarted := make(chan struct{})
	actionCanProceed := make(chan struct{})

	// Action that waits for permission to complete - this lets us check headers while it's running
	core.DefineAction(g.reg, "test/slow", api.ActionTypeCustom, nil, nil,
		func(ctx context.Context, input any) (any, error) {
			close(actionStarted) // Signal we've started
			<-actionCanProceed   // Wait for test to say we can finish
			return "completed", nil
		})

	s := &reflectionServer{Server: &http.Server{}, activeActions: newActiveActionsMap()}
	ts := httptest.NewServer(serveMux(g, s))
	defer ts.Close()

	t.Run("headers arrive before body completes", func(t *testing.T) {
		// Channel to receive headers as soon as they arrive
		type headerResult struct {
			traceID string
			spanID  string
			version string
		}
		gotHeaders := make(chan headerResult)

		go func() {
			req, _ := http.NewRequest("POST", ts.URL+"/api/runAction",
				strings.NewReader(`{"key":"/custom/test/slow","input":null}`))
			req.Header.Set("Content-Type", "application/json")

			// Do() returns as soon as headers are received (before body is read)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return
			}
			defer resp.Body.Close()

			// Send headers immediately - body isn't done yet!
			gotHeaders <- headerResult{
				traceID: resp.Header.Get("X-Genkit-Trace-Id"),
				spanID:  resp.Header.Get("X-Genkit-Span-Id"),
				version: resp.Header.Get("X-Genkit-Version"),
			}

			// Now read body (which will block until action completes)
			io.ReadAll(resp.Body)
		}()

		// Wait for action to start
		<-actionStarted

		// Check headers arrived WHILE action is still running
		select {
		case h := <-gotHeaders:
			if h.traceID == "" {
				t.Error("Expected X-Genkit-Trace-Id header")
			}
			if h.spanID == "" {
				t.Error("Expected X-Genkit-Span-Id header")
			}
			if !strings.HasPrefix(h.version, "go/") {
				t.Errorf("Expected X-Genkit-Version to start with 'go/', got %q", h.version)
			}
			t.Logf("Got headers while action running: traceID=%s", h.traceID)
		case <-time.After(1 * time.Second):
			t.Fatal("Headers did not arrive while action was still running")
		}

		// Let action complete
		close(actionCanProceed)
	})

	// Backwards compatability
	t.Run("trace ID in headers matches body", func(t *testing.T) {
		// Reset channels for this subtest
		actionStarted = make(chan struct{})
		actionCanProceed = make(chan struct{})

		// Re-register action for this subtest
		core.DefineAction(g.reg, "test/slow2", api.ActionTypeCustom, nil, nil,
			func(ctx context.Context, input any) (any, error) {
				close(actionStarted)
				<-actionCanProceed
				return "completed", nil
			})

		req, _ := http.NewRequest("POST", ts.URL+"/api/runAction",
			strings.NewReader(`{"key":"/custom/test/slow2","input":null}`))
		req.Header.Set("Content-Type", "application/json")

		// Start request in background
		type result struct {
			headerTraceID string
			bodyTraceID   string
		}
		done := make(chan result)

		go func() {
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				done <- result{}
				return
			}
			defer resp.Body.Close()
			headerTraceID := resp.Header.Get("X-Genkit-Trace-Id")

			var body map[string]interface{}
			json.NewDecoder(resp.Body).Decode(&body)
			bodyTraceID := ""
			if tel, ok := body["telemetry"].(map[string]interface{}); ok {
				bodyTraceID, _ = tel["traceId"].(string)
			}
			done <- result{headerTraceID, bodyTraceID}
		}()

		<-actionStarted
		close(actionCanProceed)

		r := <-done
		if r.headerTraceID == "" {
			t.Error("No trace ID in headers")
		}
		if r.bodyTraceID == "" {
			t.Error("No trace ID in body")
		}
		if r.headerTraceID != r.bodyTraceID {
			t.Errorf("Trace ID mismatch: header=%q, body=%q", r.headerTraceID, r.bodyTraceID)
		}
	})
}

// TestActionCancellation verifies that running actions can be cancelled via /api/cancelAction.
//
// Flow:
//  1. Start a long-running action that sends its trace ID via channel when it starts
//  2. Call POST /api/cancelAction with that trace ID
//  3. Verify: cancel endpoint returns 200, action's ctx.Done() fires, response has error code 1 (gRPC CANCELLED)
func TestActionCancellation(t *testing.T) {
	g := Init(context.Background())
	tc := tracing.NewTestOnlyTelemetryClient()
	tracing.WriteTelemetryImmediate(tc)

	gotTraceID := make(chan string, 1)
	gotCancelled := make(chan struct{})

	// Long-running action that respects cancellation
	core.DefineStreamingAction(g.reg, "test/cancellable", api.ActionTypeCustom, nil, nil,
		func(ctx context.Context, input any, cb func(context.Context, any) error) (any, error) {
			// Send trace ID so test can cancel us
			gotTraceID <- tracing.SpanTraceInfo(ctx).TraceID

			for i := 0; i < 100; i++ {
				select {
				case <-ctx.Done():
					if ctx.Err() != context.Canceled {
						return nil, fmt.Errorf("expected context.Canceled, got %v", ctx.Err())
					}
					close(gotCancelled)
					return nil, ctx.Err()
				case <-time.After(50 * time.Millisecond):
					if cb != nil && i%10 == 0 {
						cb(ctx, fmt.Sprintf("progress: %d", i))
					}
				}
			}
			return "completed", nil
		})

	s := &reflectionServer{Server: &http.Server{}, activeActions: newActiveActionsMap()}
	ts := httptest.NewServer(serveMux(g, s))
	defer ts.Close()

	// Start action in background
	actionDone := make(chan string) // receives response body when done
	go func() {
		req, _ := http.NewRequest("POST", ts.URL+"/api/runAction?stream=true",
			strings.NewReader(`{"key":"/custom/test/cancellable","input":null}`))
		req.Header.Set("Content-Type", "application/json")
		resp, _ := http.DefaultClient.Do(req)
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		actionDone <- string(body)
	}()

	// Wait for action to start
	traceID := <-gotTraceID
	time.Sleep(50 * time.Millisecond) // ensure it's tracked

	// Cancel it
	cancelReq, _ := http.NewRequest("POST", ts.URL+"/api/cancelAction",
		strings.NewReader(fmt.Sprintf(`{"traceId":"%s"}`, traceID)))
	cancelReq.Header.Set("Content-Type", "application/json")
	cancelResp, err := http.DefaultClient.Do(cancelReq)
	if err != nil {
		t.Fatal(err)
	}
	defer cancelResp.Body.Close()

	if cancelResp.StatusCode != http.StatusOK {
		t.Fatalf("Cancel failed with status %d", cancelResp.StatusCode)
	}

	// Verify action acknowledged cancellation
	select {
	case <-gotCancelled:
	case <-time.After(1 * time.Second):
		t.Fatal("Action did not acknowledge cancellation")
	}

	// Verify response indicates cancellation
	responseBody := <-actionDone
	if !strings.Contains(responseBody, "\"code\":1") {
		t.Errorf("Expected error code 1 (gRPC CANCELLED) in response, got: %s", responseBody)
	}
	if !strings.Contains(responseBody, "Action was cancelled") {
		t.Errorf("Expected 'Action was cancelled' message in response, got: %s", responseBody)
	}
}

func TestCancelActionEndpoint(t *testing.T) {
	g := Init(context.Background())

	s := &reflectionServer{
		Server:        &http.Server{},
		activeActions: newActiveActionsMap(),
	}
	ts := httptest.NewServer(serveMux(g, s))
	defer ts.Close()

	t.Run("cancel non-existent action", func(t *testing.T) {
		cancelReq, _ := http.NewRequest("POST", ts.URL+"/api/cancelAction",
			strings.NewReader(`{"traceId":"non-existent-trace-id"}`))
		cancelReq.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(cancelReq)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("Expected 404 for non-existent action, got %d", resp.StatusCode)
		}

		var result map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&result)
		if error, ok := result["error"].(string); !ok || error != "Action not found or already completed" {
			t.Errorf("Unexpected error message: %v", result)
		}
	})

	t.Run("cancel with missing traceId", func(t *testing.T) {
		cancelReq, _ := http.NewRequest("POST", ts.URL+"/api/cancelAction",
			strings.NewReader(`{}`))
		cancelReq.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(cancelReq)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusBadRequest && resp.StatusCode != http.StatusInternalServerError {
			t.Errorf("Expected 400 or 500 for missing traceId, got %d", resp.StatusCode)
		}
	})

	t.Run("cancel active action", func(t *testing.T) {
		// Manually add an action to activeActions
		testTraceID := "test-trace-id-12345"
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		s.activeActions.Set(testTraceID, &activeAction{
			cancel:    cancel,
			startTime: time.Now(),
			traceID:   testTraceID,
		})

		// Send cancel request
		cancelReq, _ := http.NewRequest("POST", ts.URL+"/api/cancelAction",
			strings.NewReader(fmt.Sprintf(`{"traceId":"%s"}`, testTraceID)))
		cancelReq.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(cancelReq)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Errorf("Expected 200 for successful cancellation, got %d", resp.StatusCode)
		}

		var result map[string]interface{}
		json.NewDecoder(resp.Body).Decode(&result)
		if message, ok := result["message"].(string); !ok || message != "Action cancelled" {
			t.Errorf("Expected 'Action cancelled' message, got: %v", result)
		}

		// Verify action was removed from activeActions
		if action, exists := s.activeActions.Get(testTraceID); exists {
			t.Errorf("Action should have been removed from activeActions, but still exists: %v", action)
		}

		// Verify context was cancelled
		select {
		case <-ctx.Done():
			// Good, context was cancelled
		default:
			t.Error("Context should have been cancelled")
		}
	})
}
