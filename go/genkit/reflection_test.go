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
	"errors"
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
	"github.com/firebase/genkit/go/core/status"
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

	defineTestAction(g.reg, "test/inc", api.ActionTypeCustom, nil, nil, inc)
	defineTestAction(g.reg, "test/dec", api.ActionTypeCustom, nil, nil, dec)

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
		g.reg.RegisterSchema("AgentRequest", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
		})
		defineTestAction(g.reg, "test/withRef", api.ActionTypeCustom, nil,
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
		defineTestAction(g.reg, "test/checkLabels", api.ActionTypeCustom, nil,
			nil,
			func(ctx context.Context, in any) (any, error) {
				labels := tracing.TelemetryLabelsFromContext(ctx)
				if labels == nil {
					return nil, errors.New("no telemetry labels found in context")
				}
				if labels["test_k"] != "test_v" {
					return nil, fmt.Errorf("unexpected telemetry labels: %v", labels)
				}
				return "ok", nil
			})

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
				name:       "verify telemetry labels in context",
				body:       `{"key": "/custom/test/checkLabels", "input": null, "telemetryLabels": {"test_k": "test_v"}}`,
				wantStatus: http.StatusOK,
				wantResult: "\"ok\"",
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
		defineTestStreamingAction(g.reg, "test/streaming", api.ActionTypeCustom, nil, nil, streamingInc)

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

	s := &reflectionServer{Server: &http.Server{}, activeActions: newActiveActionsMap()}
	ts := httptest.NewServer(serveMux(g, s))
	defer ts.Close()

	t.Run("headers arrive before body completes", func(t *testing.T) {
		// Subtest-local channels and action so the server goroutine for
		// this subtest doesn't read variables that the next subtest is
		// about to reassign. The previous shared-state setup raced under
		// -race because t.Run only synchronizes with the subtest
		// goroutine, not with the httptest server's request goroutine.
		actionStarted := make(chan struct{})
		actionCanProceed := make(chan struct{})
		defineTestAction(g.reg, "test/slow", api.ActionTypeCustom, nil, nil,
			func(ctx context.Context, input any) (any, error) {
				close(actionStarted)
				<-actionCanProceed
				return "completed", nil
			})
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
		// Subtest-local channels and action; see the comment on the
		// previous subtest.
		actionStarted := make(chan struct{})
		actionCanProceed := make(chan struct{})
		defineTestAction(g.reg, "test/slow2", api.ActionTypeCustom, nil, nil,
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
	defineTestStreamingAction(g.reg, "test/cancellable", api.ActionTypeCustom, nil, nil,
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

// TestRunActionWithInit verifies that the v1 reflection API forwards the
// request's init payload to the action (matching the JS runtime) and that
// init on an action without init support fails loudly.
func TestRunActionWithInit(t *testing.T) {
	g := Init(context.Background())

	tc := tracing.NewTestOnlyTelemetryClient()
	tracing.WriteTelemetryImmediate(tc)

	type initConfig struct {
		Prefix string `json:"prefix"`
	}
	defineTestBidiAction(g.reg, api.ActionTypeCustom, "test/bidi-prefix", nil,
		func(ctx context.Context, cfg initConfig, inCh <-chan string, outCh chan<- string) (string, error) {
			var out string
			for chunk := range inCh {
				out = cfg.Prefix + chunk
			}
			return out, nil
		})
	defineTestAction(g.reg, "test/no-init", api.ActionTypeCustom, nil, nil, inc)

	s := &reflectionServer{
		Server:        &http.Server{},
		activeActions: newActiveActionsMap(),
	}
	ts := httptest.NewServer(serveMux(g, s))
	s.Addr = strings.TrimPrefix(ts.URL, "http://")
	defer ts.Close()

	t.Run("init reaches bidi action", func(t *testing.T) {
		body := `{"key": "/custom/test/bidi-prefix", "input": "hello", "init": {"prefix": ">> "}}`
		res, err := http.Post(ts.URL+"/api/runAction", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(res.Body)
			t.Fatalf("status = %d, body: %s", res.StatusCode, b)
		}
		var resp runActionResponse
		if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		var out string
		if err := json.Unmarshal(resp.Result, &out); err != nil {
			t.Fatal(err)
		}
		if out != ">> hello" {
			t.Errorf("result = %q, want %q", out, ">> hello")
		}
	})

	t.Run("init on non-bidi action is rejected", func(t *testing.T) {
		body := `{"key": "/custom/test/no-init", "input": 1, "init": {"x": 1}}`
		res, err := http.Post(ts.URL+"/api/runAction", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		// The v1 handler reports action errors as an error JSON body
		// (matching the TS runtime), not via the HTTP status.
		var resp struct {
			Error *reflectionError `json:"error"`
		}
		if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		if resp.Error == nil {
			t.Fatal("expected error response for init on non-bidi action")
		}
		if !strings.Contains(resp.Error.Message, "does not accept init") {
			t.Errorf("error message = %q, want mention of init rejection", resp.Error.Message)
		}
	})
}

// TestToReflectionError covers the envelope the reflection API answers a
// failed run with: the code is the canonical status code (the numbering the
// dev UI's Status schema validates, not an HTTP status) taken from the error's
// status however deeply it is wrapped, and the details ride along only when
// the error carried them.
func TestToReflectionError(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		wantCode  int
		wantMsg   string
		wantStack bool
	}{
		{
			"classified error carries its stack",
			status.Errorf(status.ErrInvalidInput, "invalid input to action %q: %w", "/tool/test", errors.New("value must be a number")),
			status.InvalidArgument.Code(),
			`invalid input to action "/tool/test": value must be a number`,
			true,
		},
		{
			"plain error is internal",
			errors.New("plain error"),
			status.Internal.Code(),
			"plain error",
			false,
		},
		{
			"status survives one fmt.Errorf",
			fmt.Errorf("context: %w", status.Errorf(status.ErrNotFound, "not found")),
			status.NotFound.Code(),
			"not found",
			true,
		},
		{
			"status survives two",
			fmt.Errorf("layer2: %w", fmt.Errorf("layer1: %w", status.Errorf(status.ErrPermissionDenied, "denied"))),
			status.PermissionDenied.Code(),
			"denied",
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			re := toReflectionError(tt.err)
			if re.Code != tt.wantCode {
				t.Errorf("Code = %d, want %d", re.Code, tt.wantCode)
			}
			if re.Message != tt.wantMsg {
				t.Errorf("Message = %q, want %q", re.Message, tt.wantMsg)
			}
			gotStack := re.Details != nil && re.Details.Stack != nil
			if gotStack != tt.wantStack {
				t.Errorf("stack present = %v, want %v", gotStack, tt.wantStack)
			}
		})
	}

	t.Run("carries a traceId the error already had", func(t *testing.T) {
		err := status.Errorf(status.ErrInternal, "boom").WithDetails(map[string]any{"traceId": "trace-123"})
		re := toReflectionError(err)
		if re.Details == nil || re.Details.TraceID == nil {
			t.Fatal("expected traceId in details")
		}
		if *re.Details.TraceID != "trace-123" {
			t.Errorf("TraceID = %q, want %q", *re.Details.TraceID, "trace-123")
		}
	})

	// setTraceID is what the handler uses to stamp the run's trace onto an
	// error that arrived without details of its own.
	t.Run("setTraceID allocates missing details", func(t *testing.T) {
		re := toReflectionError(errors.New("plain"))
		if re.Details != nil {
			t.Fatalf("precondition failed, a plain error now carries details: %+v", re.Details)
		}
		re.setTraceID("trace-abc")
		if re.Details == nil || re.Details.TraceID == nil || *re.Details.TraceID != "trace-abc" {
			t.Errorf("setTraceID did not record the trace: %+v", re.Details)
		}
	})

	t.Run("setTraceID ignores an empty trace", func(t *testing.T) {
		re := toReflectionError(errors.New("plain"))
		re.setTraceID("")
		if re.Details != nil {
			t.Errorf("an empty trace allocated details: %+v", re.Details)
		}
	})
}

// TestReflectionErrorCodeYieldsValidHTTPStatus pins the split between the two
// numbering schemes the error envelope touches. The body's code is canonical
// (INVALID_ARGUMENT is 3), while wrapReflectionHandler still has to put an HTTP
// status on the response, which it derives from that code. Feeding a canonical
// code straight to WriteHeader would panic, since 3 is not a valid HTTP status.
func TestReflectionErrorCodeYieldsValidHTTPStatus(t *testing.T) {
	tests := []struct {
		err      error
		wantHTTP int
	}{
		{status.Errorf(status.ErrInvalidArgument, "bad"), http.StatusBadRequest},
		{status.Errorf(status.ErrNotFound, "missing"), http.StatusNotFound},
		{status.Errorf(status.ErrPermissionDenied, "denied"), http.StatusForbidden},
		{status.Errorf(status.ErrUnauthenticated, "who"), http.StatusUnauthorized},
		{status.Errorf(status.ErrResourceExhausted, "slow down"), http.StatusTooManyRequests},
		{errors.New("plain"), http.StatusInternalServerError},
	}
	for _, tt := range tests {
		re := toReflectionError(tt.err)
		if re.Code < 0 || re.Code > 16 {
			t.Errorf("Code = %d, want a canonical status code the dev UI's enum accepts", re.Code)
		}
		got := status.FromCode(re.Code).HTTPCode()
		if got != tt.wantHTTP {
			t.Errorf("HTTP status for code %d = %d, want %d", re.Code, got, tt.wantHTTP)
		}
		if got < 100 || got > 999 {
			t.Errorf("HTTP status %d is outside the range WriteHeader accepts", got)
		}
	}
}

// TestRunActionPlainErrorResponse pins that an action failing with an error the
// framework never classified still produces an error response.
//
// toReflectionError fills in the details envelope only from a stack or trace
// the error itself carried, and leaves the pointer nil otherwise. A plain error
// returned by a plugin or a user's own function hits that case, and the handler
// then wrote the run's trace ID into the nil envelope, panicking the reflection
// server mid-response. That is every unclassified failure, which is the common
// one: a provider SDK rejecting a request reaches here as its own error type.
func TestRunActionPlainErrorResponse(t *testing.T) {
	tc := tracing.NewTestOnlyTelemetryClient()
	tracing.WriteTelemetryImmediate(tc)

	g := Init(context.Background())
	defineTestAction(g.reg, "test/boom", api.ActionTypeCustom, nil, nil,
		func(_ context.Context, x int) (int, error) {
			// Deliberately unclassified, as a provider SDK's error would be.
			return 0, errors.New("provider rejected the request")
		})

	s := &reflectionServer{Server: &http.Server{}, activeActions: newActiveActionsMap()}
	ts := httptest.NewServer(serveMux(g, s))
	s.Addr = strings.TrimPrefix(ts.URL, "http://")
	defer ts.Close()

	for _, tt := range []struct {
		name string
		path string
	}{
		{"non-streaming", "/api/runAction"},
		{"streaming", "/api/runAction?stream=true"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res, err := http.Post(ts.URL+tt.path, "application/json",
				strings.NewReader(`{"key":"/custom/test/boom","input":3}`))
			if err != nil {
				// A panic in the handler closes the connection mid-response,
				// so it surfaces here rather than as a 500.
				t.Fatalf("request failed, the handler likely panicked: %v", err)
			}
			defer res.Body.Close()

			body, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("reading body: %v", err)
			}

			var got struct {
				Error struct {
					Message string `json:"message"`
					Code    int    `json:"code"`
					Details *struct {
						TraceID *string `json:"traceId"`
					} `json:"details"`
				} `json:"error"`
			}
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatalf("response is not the error envelope: %v\nbody: %s", err, body)
			}
			if !strings.Contains(got.Error.Message, "provider rejected the request") {
				t.Errorf("message = %q, want the action's error", got.Error.Message)
			}
			// The trace ID is why the envelope gets written to at all, so an
			// error that arrived without one must still come back carrying it.
			if got.Error.Details == nil || got.Error.Details.TraceID == nil {
				t.Errorf("no trace ID on the error response: %s", body)
			}
		})
	}
}

// defineTestBidiAction creates and registers a bidi action in one call for
// the reflection and server tests in this package.
func defineTestBidiAction[In, Out, Stream, Init any](r api.Registry, atype api.ActionType, name string, opts *core.BidiActionOptions, fn core.BidiFunc[In, Out, Stream, Init]) *core.BidiAction[In, Out, Stream, Init] {
	b := core.NewBidiActionOf(atype, name, opts, fn)
	b.Register(r)
	return b
}

// defineTestAction and defineTestStreamingAction register actions in one
// call for the tests in this package, mirroring the removed registry-taking
// core.Define* helpers.
func defineTestAction[In, Out any](r api.Registry, name string, atype api.ActionType, metadata map[string]any, inputSchema map[string]any, fn core.Func[In, Out]) *core.Action[In, Out, struct{}] {
	a := core.NewActionOf(atype, name, &core.ActionOptions{Metadata: metadata, InputSchema: inputSchema}, fn)
	a.Register(r)
	return a
}

func defineTestStreamingAction[In, Out, Stream any](r api.Registry, name string, atype api.ActionType, metadata map[string]any, inputSchema map[string]any, fn core.StreamingFunc[In, Out, Stream]) *core.Action[In, Out, Stream] {
	a := core.NewStreamingActionOf(atype, name, &core.ActionOptions{Metadata: metadata, InputSchema: inputSchema}, fn)
	a.Register(r)
	return a
}
