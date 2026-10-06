// Copyright 2026 Google LLC
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

package ollama

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

func dishRequest() *ai.ModelRequest {
	return &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage("Suggest a dish without peanuts.")}}
}

func noopChunk(context.Context, *ai.ModelResponseChunk) error { return nil }

// A closed listener stands in for a local Ollama server that isn't running.
func downServerURL() string {
	server := httptest.NewServer(http.NotFoundHandler())
	url := server.URL
	server.Close()
	return url
}

func TestGenerateClassifiesUnreachableServer(t *testing.T) {
	url := downServerURL()
	for _, tc := range []struct {
		name string
		cb   func(context.Context, *ai.ModelResponseChunk) error
	}{
		{"unary", nil},
		{"streaming", noopChunk},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := &generator{model: ModelDefinition{Name: "gemma3", Type: "chat"}, serverAddress: url, timeout: 5}
			_, err := g.generate(context.Background(), dishRequest(), tc.cb)
			if got, ok := status.Classified(err); !ok || got != status.Unavailable {
				t.Fatalf("status = %v (classified %v), want Unavailable; err = %v", got, ok, err)
			}
			if !strings.Contains(err.Error(), url) {
				t.Errorf("err = %q, want the server address in the message", err)
			}
		})
	}
}

func TestGenerateClassifiesNon200ByStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"model \"gemma3\" not found, try pulling it first"}`, http.StatusNotFound)
	}))
	defer server.Close()

	for _, tc := range []struct {
		name string
		cb   func(context.Context, *ai.ModelResponseChunk) error
	}{
		{"unary", nil},
		// The streaming branch used to decode the error body as a chunk.
		{"streaming", noopChunk},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := &generator{model: ModelDefinition{Name: "gemma3", Type: "chat"}, serverAddress: server.URL, timeout: 5}
			_, err := g.generate(context.Background(), dishRequest(), tc.cb)
			if got, ok := status.Classified(err); !ok || got != status.NotFound {
				t.Fatalf("status = %v (classified %v), want NotFound; err = %v", got, ok, err)
			}
			if !strings.Contains(err.Error(), "try pulling it first") {
				t.Errorf("err = %q, want the server's message", err)
			}
		})
	}
}

func TestGenerateKeepsCallerCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	g := &generator{model: ModelDefinition{Name: "gemma3", Type: "chat"}, serverAddress: downServerURL(), timeout: 5}
	_, err := g.generate(ctx, dishRequest(), nil)
	if got := status.Of(err); got != status.Cancelled {
		t.Fatalf("status = %v, want Cancelled; err = %v", got, err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want it to wrap context.Canceled", err)
	}
}

func TestEmbedClassifiesFailures(t *testing.T) {
	req := &ai.EmbedRequest{
		Input:   []*ai.Document{ai.DocumentFromText("smoked salmon tartine", nil)},
		Options: &EmbedOptions{Model: "nomic-embed-text"},
	}

	t.Run("unreachable server", func(t *testing.T) {
		_, err := embed(context.Background(), downServerURL(), req)
		if got, ok := status.Classified(err); !ok || got != status.Unavailable {
			t.Fatalf("status = %v (classified %v), want Unavailable; err = %v", got, ok, err)
		}
	})

	t.Run("bad request", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, `{"error":"input exceeds context length"}`, http.StatusBadRequest)
		}))
		defer server.Close()

		_, err := embed(context.Background(), server.URL, req)
		if got, ok := status.Classified(err); !ok || got != status.InvalidArgument {
			t.Fatalf("status = %v (classified %v), want InvalidArgument; err = %v", got, ok, err)
		}
	})
}

type timeoutError struct{}

func (timeoutError) Error() string   { return "i/o timeout" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }

func TestSendErrorClassifiesClientTimeout(t *testing.T) {
	err := sendError(context.Background(), "http://localhost:11434", timeoutError{})
	if got, ok := status.Classified(err); !ok || got != status.DeadlineExceeded {
		t.Fatalf("status = %v (classified %v), want DeadlineExceeded; err = %v", got, ok, err)
	}
	if !errors.Is(err, timeoutError{}) {
		t.Errorf("err = %v, want it to wrap the transport error", err)
	}
}
