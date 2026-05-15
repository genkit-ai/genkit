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
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/x/streaming"
)

func FakeContextProvider(ctx context.Context, req core.RequestData) (core.ActionContext, error) {
	return core.ActionContext{
		"test": "action-context-value",
	}, nil
}

func TestHandler(t *testing.T) {
	g := Init(context.Background())

	successFlow := DefineFlow(g, "handlerSuccess", func(ctx context.Context, input string) (string, error) {
		return "success", nil
	})

	genericErrorFlow := DefineFlow(g, "handlerGenericError", func(ctx context.Context, input string) (string, error) {
		return "", errors.New("generic error message")
	})

	genkitErrorInvalidArgFlow := DefineFlow(g, "handlerGenkitErrorInvalidArg", func(ctx context.Context, input string) (string, error) {
		return "", core.NewError(core.INVALID_ARGUMENT, "invalid argument")
	})

	genkitErrorNotFoundFlow := DefineFlow(g, "handlerGenkitErrorNotFound", func(ctx context.Context, input string) (string, error) {
		return "", core.NewError(core.NOT_FOUND, "resource not found")
	})

	genkitErrorPermissionDeniedFlow := DefineFlow(g, "handlerGenkitErrorPermissionDenied", func(ctx context.Context, input string) (string, error) {
		return "", core.NewError(core.PERMISSION_DENIED, "permission denied")
	})

	userFacingErrorFlow := DefineFlow(g, "handlerUserFacingError", func(ctx context.Context, input string) (string, error) {
		return "", core.NewPublicError(core.INVALID_ARGUMENT, "public error message", nil)
	})

	t.Run("successful request returns 200 with response", func(t *testing.T) {
		handler := Handler(successFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusOK {
			t.Errorf("want status code %d, got %d", http.StatusOK, resp.StatusCode)
		}

		if !strings.Contains(string(body), "success") {
			t.Errorf("want response to contain 'success', got %q", string(body))
		}
	})

	t.Run("generic error returns 500 with error in response body", func(t *testing.T) {
		handler := Handler(genericErrorFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusInternalServerError {
			t.Errorf("want status code %d, got %d", http.StatusInternalServerError, resp.StatusCode)
		}

		if !strings.Contains(string(body), "generic error message") {
			t.Errorf("want error message in response body, got %q", string(body))
		}
	})

	t.Run("GenkitError INVALID_ARGUMENT maps to 400", func(t *testing.T) {
		handler := Handler(genkitErrorInvalidArgFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("want status code %d for INVALID_ARGUMENT, got %d", http.StatusBadRequest, resp.StatusCode)
		}

		if !strings.Contains(string(body), "invalid argument") {
			t.Errorf("want error message in response body, got %q", string(body))
		}
	})

	t.Run("GenkitError NOT_FOUND maps to 404", func(t *testing.T) {
		handler := Handler(genkitErrorNotFoundFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("want status code %d for NOT_FOUND, got %d", http.StatusNotFound, resp.StatusCode)
		}

		if !strings.Contains(string(body), "resource not found") {
			t.Errorf("want error message in response body, got %q", string(body))
		}
	})

	t.Run("GenkitError PERMISSION_DENIED maps to 403", func(t *testing.T) {
		handler := Handler(genkitErrorPermissionDeniedFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusForbidden {
			t.Errorf("want status code %d for PERMISSION_DENIED, got %d", http.StatusForbidden, resp.StatusCode)
		}

		if !strings.Contains(string(body), "permission denied") {
			t.Errorf("want error message in response body, got %q", string(body))
		}
	})

	t.Run("UserFacingError returns internal server error", func(t *testing.T) {
		handler := Handler(userFacingErrorFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusInternalServerError {
			t.Errorf("want status code %d, got %d", http.StatusInternalServerError, resp.StatusCode)
		}

		if !strings.Contains(string(body), "public error message") {
			t.Errorf("want error message in response body, got %q", string(body))
		}
	})

	t.Run("error is written to response not returned", func(t *testing.T) {
		handler := Handler(genericErrorFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()

		// Verify error was written to response
		if resp.StatusCode == http.StatusOK {
			t.Error("want error status code, got 200")
		}
	})
}

func TestHandlerFunc(t *testing.T) {
	g := Init(context.Background())

	echoFlow := DefineFlow(g, "echo", func(ctx context.Context, input string) (string, error) {
		return input, nil
	})

	errorFlow := DefineFlow(g, "error", func(ctx context.Context, input string) (string, error) {
		return "", errors.New("flow error")
	})

	contextReaderFlow := DefineFlow(g, "contextReader", func(ctx context.Context, input []string) (string, error) {
		actionCtx := core.FromContext(ctx)
		if actionCtx == nil {
			return "", errors.New("no action context")
		}

		if len(input) == 0 {
			return "", nil
		}

		var values []string
		for _, key := range input {
			value, ok := actionCtx[key]
			if !ok {
				return "", fmt.Errorf("action context key %q not found", key)
			}

			strValue, ok := value.(string)
			if !ok {
				return "", fmt.Errorf("action context value for key %q is not a string", key)
			}

			values = append(values, strValue)
		}

		return strings.Join(values, ","), nil
	})

	t.Run("basic handler", func(t *testing.T) {
		handlerFunc := HandlerFunc(echoFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test-input"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if !strings.Contains(string(body), `"test-input"`) {
			t.Errorf("want response to contain test-input, got %q", string(body))
		}
	})

	t.Run("action error", func(t *testing.T) {
		handlerFunc := HandlerFunc(errorFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test-input"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err == nil {
			t.Fatal("want error, got nil")
		}

		if !strings.Contains(err.Error(), "flow error") {
			t.Errorf("want error containing 'flow error', got %v", err)
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		handlerFunc := HandlerFunc(echoFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":invalid-json}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err == nil {
			t.Fatal("want error for invalid JSON, got nil")
		}

		if !strings.Contains(err.Error(), "invalid character") {
			t.Errorf("want error about invalid JSON, got %v", err)
		}
	})

	t.Run("with context provider", func(t *testing.T) {
		handlerFunc := HandlerFunc(contextReaderFlow, WithContextProviders(FakeContextProvider))

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":["test"]}`))
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if !strings.Contains(string(body), "action-context-value") {
			t.Errorf("want response to contain action-context-value, got %q", string(body))
		}
	})

	t.Run("multiple context providers", func(t *testing.T) {
		handlerFunc := HandlerFunc(contextReaderFlow, WithContextProviders(
			func(ctx context.Context, req core.RequestData) (core.ActionContext, error) {
				return core.ActionContext{"provider1": "value1"}, nil
			},
			func(ctx context.Context, req core.RequestData) (core.ActionContext, error) {
				return core.ActionContext{"provider2": "value2"}, nil
			},
		))

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":["provider1","provider2"]}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		if !strings.Contains(string(body), "value1,value2") {
			t.Errorf("want response to contain value1,value2, got %q", string(body))
		}
	})
}

func TestStreamingHandlerFunc(t *testing.T) {
	g := Init(context.Background())

	streamingFlow := DefineStreamingFlow(g, "streaming",
		func(ctx context.Context, input string, cb func(context.Context, string) error) (string, error) {
			for _, c := range input {
				if err := cb(ctx, string(c)); err != nil {
					return "", err
				}
			}
			return input + "-end", nil
		})

	errorStreamingFlow := DefineStreamingFlow(g, "errorStreaming",
		func(ctx context.Context, input string, cb func(context.Context, string) error) (string, error) {
			return "", errors.New("streaming error")
		})

	t.Run("streaming response", func(t *testing.T) {
		handlerFunc := HandlerFunc(streamingFlow)

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"hello"}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		expected := `data: {"message":"h"}

data: {"message":"e"}

data: {"message":"l"}

data: {"message":"l"}

data: {"message":"o"}

data: {"result":"hello-end"}

`
		if string(body) != expected {
			t.Errorf("want streaming body:\n%q\n\nGot:\n%q", expected, string(body))
		}
	})

	t.Run("streaming error", func(t *testing.T) {
		handlerFunc := HandlerFunc(errorStreamingFlow)

		req := httptest.NewRequest("POST", "/?stream=true", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		// For streaming, errors are sent as part of the SSE stream, not returned
		if err != nil {
			t.Errorf("want nil error (error should be in stream), got %v", err)
		}

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		expected := `data: {"error":{"status":"INTERNAL_SERVER_ERROR","message":"stream flow error","details":"streaming error"}}

`
		if string(body) != expected {
			t.Errorf("want error body:\n%q\n\nGot:\n%q", expected, string(body))
		}
	})
}

func TestDurableStreamingHandlerFunc(t *testing.T) {
	g := Init(context.Background())

	streamingFlow := DefineStreamingFlow(g, "durableStreaming",
		func(ctx context.Context, input string, cb func(context.Context, string) error) (string, error) {
			for _, c := range input {
				if err := cb(ctx, string(c)); err != nil {
					return "", err
				}
			}
			return input + "-done", nil
		})

	t.Run("returns stream ID header", func(t *testing.T) {
		sm := streaming.NewInMemoryStreamManager()
		defer sm.Close()
		handlerFunc := HandlerFunc(streamingFlow, WithStreamManager(sm))

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"hi"}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp := w.Result()
		body, _ := io.ReadAll(resp.Body)

		streamID := resp.Header.Get("X-Genkit-Stream-Id")
		if streamID == "" {
			t.Error("want X-Genkit-Stream-Id header to be set")
		}

		expected := `data: {"message":"h"}

data: {"message":"i"}

data: {"result":"hi-done"}

`
		if string(body) != expected {
			t.Errorf("want streaming body:\n%q\n\nGot:\n%q", expected, string(body))
		}
	})

	t.Run("subscribe to completed stream", func(t *testing.T) {
		sm := streaming.NewInMemoryStreamManager()
		defer sm.Close()
		handlerFunc := HandlerFunc(streamingFlow, WithStreamManager(sm))

		// First request - run the stream to completion
		req1 := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"ab"}`))
		req1.Header.Set("Content-Type", "application/json")
		req1.Header.Set("Accept", "text/event-stream")
		w1 := httptest.NewRecorder()

		err := handlerFunc(w1, req1)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp1 := w1.Result()
		streamID := resp1.Header.Get("X-Genkit-Stream-Id")
		if streamID == "" {
			t.Fatal("want X-Genkit-Stream-Id header to be set")
		}

		// Second request - subscribe to the completed stream
		req2 := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"ignored"}`))
		req2.Header.Set("Content-Type", "application/json")
		req2.Header.Set("Accept", "text/event-stream")
		req2.Header.Set("X-Genkit-Stream-Id", streamID)
		w2 := httptest.NewRecorder()

		err = handlerFunc(w2, req2)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp2 := w2.Result()
		body2, _ := io.ReadAll(resp2.Body)

		// Should replay all chunks and the final result
		expected := `data: {"message":"a"}

data: {"message":"b"}

data: {"result":"ab-done"}

`
		if string(body2) != expected {
			t.Errorf("want replayed body:\n%q\n\nGot:\n%q", expected, string(body2))
		}
	})

	t.Run("subscribe to non-existent stream returns 204", func(t *testing.T) {
		sm := streaming.NewInMemoryStreamManager()
		defer sm.Close()
		handlerFunc := HandlerFunc(streamingFlow, WithStreamManager(sm))

		req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"test"}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("X-Genkit-Stream-Id", "non-existent-stream-id")
		w := httptest.NewRecorder()

		err := handlerFunc(w, req)

		if err != nil {
			t.Errorf("want nil error, got %v", err)
		}

		resp := w.Result()

		if resp.StatusCode != http.StatusNoContent {
			t.Errorf("want status code %d, got %d", http.StatusNoContent, resp.StatusCode)
		}
	})
}
