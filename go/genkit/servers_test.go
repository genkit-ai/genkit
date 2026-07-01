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
	"github.com/firebase/genkit/go/core/api"
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

		expected := `data: {"error":{"status":"INTERNAL","message":"stream flow error","details":"streaming error"}}

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

// TestHandlerBidiInitEnvelope verifies that an HTTP POST to a bidi action
// handler can supply Init via the request envelope's "init" field, alongside
// the existing "data" field. This is the production HTTP path for bidi
// actions invoked as one-shots.
func TestHandlerBidiInitEnvelope(t *testing.T) {
	g := Init(context.Background())

	type Config struct {
		Prefix string `json:"prefix"`
	}

	bidiAction := core.DefineBidiAction(g.reg, "envelopeBidi", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg Config, inCh <-chan string, outCh chan<- string) (string, error) {
			for in := range inCh {
				outCh <- cfg.Prefix + in
			}
			return "done", nil
		})

	t.Run("non-streaming envelope with init", func(t *testing.T) {
		handler := Handler(bidiAction)

		body := `{"data":"hello","init":{"prefix":">> "}}`
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		respBody, _ := io.ReadAll(resp.Body)

		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, body = %s", resp.StatusCode, string(respBody))
		}
		// Result should be "done"; the prefixed chunk goes to the streaming
		// callback (nil here), so the final output is the function's return.
		if !strings.Contains(string(respBody), `"done"`) {
			t.Errorf("response body = %q, want it to contain \"done\"", string(respBody))
		}
	})

	t.Run("streaming envelope with init delivers prefixed chunk", func(t *testing.T) {
		handler := HandlerFunc(bidiAction)

		// Use an HTML-safe prefix so json.Marshal doesn't escape it; that
		// way the assertion can match the prefix literally in the SSE body.
		body := `{"data":"hello","init":{"prefix":"PFX:"}}`
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		w := httptest.NewRecorder()

		if err := handler(w, req); err != nil {
			t.Fatalf("handler: %v", err)
		}

		respBody, _ := io.ReadAll(w.Result().Body)
		if !strings.Contains(string(respBody), "PFX:hello") {
			t.Errorf("response missing prefixed chunk; body = %q", string(respBody))
		}
		if !strings.Contains(string(respBody), `"done"`) {
			t.Errorf("response missing final result; body = %q", string(respBody))
		}
	})

	t.Run("envelope without init uses zero value", func(t *testing.T) {
		handler := Handler(bidiAction)

		body := `{"data":"hello"}`
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			t.Fatalf("status = %d, body = %s", resp.StatusCode, string(respBody))
		}
	})

	t.Run("envelope with malformed init returns 400", func(t *testing.T) {
		handler := Handler(bidiAction)

		// Init is valid JSON but doesn't match the action's Config (prefix
		// must be a string; here it's a number).
		body := `{"data":"hello","init":{"prefix":42}}`
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusBadRequest {
			respBody, _ := io.ReadAll(resp.Body)
			t.Errorf("status = %d, want %d; body = %s", resp.StatusCode, http.StatusBadRequest, string(respBody))
		}
	})

	t.Run("init on non-bidi flow returns 400", func(t *testing.T) {
		plainFlow := DefineFlow(g, "envelopePlain",
			func(ctx context.Context, in string) (string, error) {
				return "out:" + in, nil
			})
		handler := Handler(plainFlow)

		body := `{"data":"hello","init":{"prefix":">> "}}`
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusBadRequest {
			respBody, _ := io.ReadAll(resp.Body)
			t.Errorf("status = %d, want %d; body = %s", resp.StatusCode, http.StatusBadRequest, string(respBody))
		}
	})

	t.Run("init on non-bidi flow returns 400 on streaming requests", func(t *testing.T) {
		plainFlow := DefineFlow(g, "envelopePlainStream",
			func(ctx context.Context, in string) (string, error) {
				return "out:" + in, nil
			})
		handler := Handler(plainFlow)

		// The rejection must happen before the handler commits to SSE: a
		// streaming client should see an HTTP 400, not a 200 with an
		// in-band error event.
		body := `{"data":"hello","init":{"prefix":">> "}}`
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		w := httptest.NewRecorder()

		handler(w, req)

		resp := w.Result()
		respBody, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("status = %d, want %d; body = %s", resp.StatusCode, http.StatusBadRequest, string(respBody))
		}
		if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "text/event-stream") {
			t.Errorf("Content-Type = %q; response must not commit to SSE before rejecting init", ct)
		}
	})
}
