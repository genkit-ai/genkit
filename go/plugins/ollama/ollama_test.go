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

package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

var _ api.Plugin = (*Ollama)(nil)
var _ api.DynamicPlugin = (*Ollama)(nil)

func TestConcatMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []*ai.Message
		roles    []ai.Role
		want     string
	}{
		{
			name: "Single message with matching role",
			messages: []*ai.Message{
				{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("Hello, how are you?")},
				},
			},
			roles: []ai.Role{ai.RoleUser},
			want:  "Hello, how are you?",
		},
		{
			name: "Multiple messages with mixed roles",
			messages: []*ai.Message{
				{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("Tell me a joke.")},
				},
				{
					Role:    ai.RoleModel,
					Content: []*ai.Part{ai.NewTextPart("Why did the scarecrow win an award? Because he was outstanding in his field!")},
				},
			},
			roles: []ai.Role{ai.RoleModel},
			want:  "Why did the scarecrow win an award? Because he was outstanding in his field!",
		},
		{
			name: "No matching role",
			messages: []*ai.Message{
				{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("Any suggestions?")},
				},
			},
			roles: []ai.Role{ai.RoleSystem},
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := &ai.ModelRequest{Messages: tt.messages}
			got := concatMessages(input, tt.roles)
			if got != tt.want {
				t.Errorf("concatMessages() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestTranslateGenerateChunk(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    *ai.ModelResponseChunk
		wantErr bool
	}{
		{
			name:  "Valid JSON response",
			input: `{"model": "my-model", "created_at": "2024-06-20T12:34:56Z", "response": "This is a test response."}`,
			want: &ai.ModelResponseChunk{
				Content: []*ai.Part{ai.NewTextPart("This is a test response.")},
			},
			wantErr: false,
		},
		{
			name:    "Invalid JSON",
			input:   `{invalid}`,
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := translateGenerateChunk(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("translateGenerateChunk() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !equalContent(got.Content, tt.want.Content) {
				t.Errorf("translateGenerateChunk() got = %v, want %v", got, tt.want)
			}
		})
	}
}

// Helper function to compare content
func equalContent(a, b []*ai.Part) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].IsText() {
			if !b[i].IsText() || a[i].Text != b[i].Text {
				return false
			}
		} else if a[i].IsReasoning() {
			if !b[i].IsReasoning() || a[i].Text != b[i].Text {
				return false
			}
		} else {
			// For other types, we might need more specific checks,
			// but for now return false if kinds don't match or not handled
			return false
		}
	}
	return true
}

func newTestOllama(serverAddress string) *Ollama {
	o := &Ollama{ServerAddress: serverAddress, Timeout: 30}
	o.Init(context.Background())
	return o
}

func modelSupportsMetadata(t *testing.T, desc api.ActionDesc) map[string]any {
	t.Helper()
	modelMetadata, ok := desc.Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata has type %T, want map[string]any", desc.Metadata["model"])
	}
	supports, ok := modelMetadata["supports"].(map[string]any)
	if !ok {
		t.Fatalf("model supports metadata has type %T, want map[string]any", modelMetadata["supports"])
	}
	return supports
}

func TestDynamicPlugin(t *testing.T) {
	t.Run("listLocalModels", func(t *testing.T) {
		tests := []struct {
			name       string
			response   ollamaTagsResponse
			statusCode int
			wantCount  int
			wantErr    bool
		}{
			{
				name: "successful response with multiple models",
				response: ollamaTagsResponse{
					Models: []ollamaLocalModel{
						{Name: "llama3:latest", Model: "llama3:latest"},
						{Name: "mistral:7b", Model: "mistral:7b"},
					},
				},
				statusCode: http.StatusOK,
				wantCount:  2,
			},
			{
				name:       "empty model list",
				response:   ollamaTagsResponse{Models: []ollamaLocalModel{}},
				statusCode: http.StatusOK,
				wantCount:  0,
			},
			{
				name:       "server error",
				statusCode: http.StatusInternalServerError,
				wantErr:    true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != "/api/tags" {
						t.Errorf("unexpected path: %s", r.URL.Path)
					}
					if r.Method != http.MethodGet {
						t.Errorf("unexpected method: %s", r.Method)
					}
					w.WriteHeader(tt.statusCode)
					if tt.statusCode == http.StatusOK {
						json.NewEncoder(w).Encode(tt.response)
					}
				}))
				defer server.Close()

				o := newTestOllama(server.URL)
				models, err := o.listLocalModels(context.Background())
				if (err != nil) != tt.wantErr {
					t.Errorf("listLocalModels() error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if !tt.wantErr && len(models) != tt.wantCount {
					t.Errorf("listLocalModels() returned %d models, want %d", len(models), tt.wantCount)
				}
			})
		}
	})

	t.Run("ListActions", func(t *testing.T) {
		t.Run("filters embed models", func(t *testing.T) {
			response := ollamaTagsResponse{
				Models: []ollamaLocalModel{
					{Name: "llama3:latest", Model: "llama3:latest"},
					{Name: "nomic-embed-text:latest", Model: "nomic-embed-text:latest"},
					{Name: "moondream:v2", Model: "moondream:v2"},
				},
			}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				json.NewEncoder(w).Encode(response)
			}))
			defer server.Close()

			o := newTestOllama(server.URL)
			actions := o.ListActions(context.Background())

			if len(actions) != 2 {
				t.Fatalf("ListActions() returned %d actions, want 2", len(actions))
			}

			names := make(map[string]bool)
			for _, a := range actions {
				names[a.Name] = true
			}
			if !names["ollama/llama3:latest"] {
				t.Error("ListActions() missing ollama/llama3:latest")
			}
			if !names["ollama/moondream:v2"] {
				t.Error("ListActions() missing ollama/moondream:v2")
			}
			if names["ollama/nomic-embed-text:latest"] {
				t.Error("ListActions() should have filtered out embed model")
			}
		})

		t.Run("server unreachable", func(t *testing.T) {
			o := newTestOllama("http://localhost:0")
			actions := o.ListActions(context.Background())
			if actions != nil {
				t.Errorf("ListActions() should return nil when server is unreachable, got %v", actions)
			}
		})

		t.Run("preserves dynamic model defaults when capabilities are unavailable", func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/api/tags" {
					json.NewEncoder(w).Encode(ollamaTagsResponse{
						Models: []ollamaLocalModel{{Name: "brand-new-model"}},
					})
					return
				}
				http.NotFound(w, r)
			}))
			defer server.Close()

			actions := newTestOllama(server.URL).ListActions(context.Background())
			if len(actions) != 1 {
				t.Fatalf("ListActions() returned %d actions, want 1", len(actions))
			}
			supports := modelSupportsMetadata(t, actions[0])
			if supports["tools"] != true || supports["media"] != true {
				t.Errorf("fallback supports = %v, want tools and media enabled", supports)
			}
		})

		t.Run("does not fall back for an explicitly empty capabilities list", func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					json.NewEncoder(w).Encode(ollamaTagsResponse{
						Models: []ollamaLocalModel{{Name: "qwen2.5"}},
					})
				case "/api/show":
					json.NewEncoder(w).Encode(map[string]any{"capabilities": []string{}})
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			actions := newTestOllama(server.URL).ListActions(context.Background())
			if len(actions) != 1 {
				t.Fatalf("ListActions() returned %d actions, want 1", len(actions))
			}
			supports := modelSupportsMetadata(t, actions[0])
			if supports["tools"] != false || supports["media"] != false {
				t.Errorf("supports = %v, want tools and media disabled", supports)
			}
		})

		t.Run("caches capabilities by model digest", func(t *testing.T) {
			var tagsCalls atomic.Int32
			var showCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					digest := "digest-1"
					if tagsCalls.Add(1) == 3 {
						digest = "digest-2"
					}
					_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{{
						Name: "qwen2.5", Digest: digest,
					}}})
				case "/api/show":
					showCalls.Add(1)
					_ = json.NewEncoder(w).Encode(map[string]any{"capabilities": []string{"completion", "tools"}})
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			o := newTestOllama(server.URL)
			for range 3 {
				if actions := o.ListActions(t.Context()); len(actions) != 1 {
					t.Fatalf("ListActions() returned %d actions, want 1", len(actions))
				}
			}
			if got := showCalls.Load(); got != 2 {
				t.Fatalf("/api/show calls = %d, want 2 (initial digest and changed digest)", got)
			}
		})

		t.Run("shares discovered capabilities with ResolveAction", func(t *testing.T) {
			var showCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{{
						Name: "dynamic-model", Digest: "digest-1",
					}}})
				case "/api/show":
					showCalls.Add(1)
					_ = json.NewEncoder(w).Encode(map[string]any{"capabilities": []string{"completion"}})
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			o := newTestOllama(server.URL)
			if actions := o.ListActions(t.Context()); len(actions) != 1 {
				t.Fatalf("ListActions() returned %d actions, want 1", len(actions))
			}
			action := o.ResolveAction(api.ActionTypeModel, "dynamic-model")
			if action == nil {
				t.Fatal("ResolveAction() returned nil")
			}
			supports := modelSupportsMetadata(t, action.Desc())
			if supports["tools"] != false || supports["media"] != false {
				t.Errorf("resolved supports = %v, want discovered capabilities", supports)
			}
			if got := showCalls.Load(); got != 1 {
				t.Errorf("/api/show calls = %d, want 1", got)
			}
		})

		t.Run("shares latest-tag capabilities with bare model names", func(t *testing.T) {
			var showCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{{
						Name: "moondream:latest", Digest: "digest-1",
					}}})
				case "/api/show":
					showCalls.Add(1)
					_ = json.NewEncoder(w).Encode(map[string]any{
						"capabilities": []string{"completion", "vision"},
					})
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			o := newTestOllama(server.URL)
			if actions := o.ListActions(t.Context()); len(actions) != 1 {
				t.Fatalf("ListActions() returned %d actions, want 1", len(actions))
			}

			resolved := o.ResolveAction(api.ActionTypeModel, "moondream")
			if resolved == nil {
				t.Fatal("ResolveAction() returned nil")
			}
			resolvedSupports := modelSupportsMetadata(t, resolved.Desc())
			if resolvedSupports["tools"] != false || resolvedSupports["media"] != true {
				t.Errorf("resolved supports = %v, want detected tools=false and media=true", resolvedSupports)
			}

			g := genkit.Init(t.Context())
			defined := o.DefineModel(g, ModelDefinition{Name: "moondream", Type: "chat"}, nil)
			definedAction, ok := defined.(api.Action)
			if !ok {
				t.Fatal("defined model does not implement api.Action")
			}
			definedSupports := modelSupportsMetadata(t, definedAction.Desc())
			if definedSupports["tools"] != false || definedSupports["media"] != true {
				t.Errorf("defined supports = %v, want detected tools=false and media=true", definedSupports)
			}
			if got := showCalls.Load(); got != 1 {
				t.Errorf("/api/show calls = %d, want 1", got)
			}
		})

		t.Run("retries capability detection after a cached failure expires", func(t *testing.T) {
			var showCalls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{{
						Name: "recovering-model", Digest: "digest-1",
					}}})
				case "/api/show":
					if showCalls.Add(1) == 1 {
						http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
						return
					}
					_ = json.NewEncoder(w).Encode(map[string]any{"capabilities": []string{"completion"}})
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			o := newTestOllama(server.URL)
			for range 2 {
				actions := o.ListActions(t.Context())
				if len(actions) != 1 {
					t.Fatalf("ListActions() returned %d actions, want 1", len(actions))
				}
				if got := modelSupportsMetadata(t, actions[0])["tools"]; got != true {
					t.Fatalf("fallback tools support = %v, want true", got)
				}
			}
			if got := showCalls.Load(); got != 1 {
				t.Fatalf("/api/show calls before cache expiry = %d, want 1", got)
			}

			o.capMu.Lock()
			entry := o.capabilitiesCache["recovering-model"]
			entry.expires = time.Now().Add(-time.Second)
			o.capabilitiesCache["recovering-model"] = entry
			o.capMu.Unlock()

			actions := o.ListActions(t.Context())
			if len(actions) != 1 {
				t.Fatalf("ListActions() after recovery returned %d actions, want 1", len(actions))
			}
			if got := modelSupportsMetadata(t, actions[0])["tools"]; got != false {
				t.Errorf("detected tools support after recovery = %v, want false", got)
			}
			if got := showCalls.Load(); got != 2 {
				t.Errorf("/api/show calls after cache expiry = %d, want 2", got)
			}
		})

		t.Run("queries uncached models concurrently", func(t *testing.T) {
			started := make(chan struct{}, 2)
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{
						{Name: "model-a", Digest: "a"},
						{Name: "model-b", Digest: "b"},
					}})
				case "/api/show":
					started <- struct{}{}
					<-release
					_ = json.NewEncoder(w).Encode(map[string]any{"capabilities": []string{"completion"}})
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			result := make(chan []api.ActionDesc, 1)
			go func() { result <- newTestOllama(server.URL).ListActions(t.Context()) }()
			for range 2 {
				select {
				case <-started:
				case <-time.After(time.Second):
					close(release)
					t.Fatal("capability requests did not run concurrently")
				}
			}
			close(release)
			if actions := <-result; len(actions) != 2 {
				t.Fatalf("ListActions() returned %d actions, want 2", len(actions))
			}
		})

		t.Run("bounds concurrent capability queries", func(t *testing.T) {
			var active atomic.Int32
			var peak atomic.Int32
			release := make(chan struct{})
			started := make(chan struct{}, maxConcurrentCapabilityQueries+1)
			models := make([]ollamaLocalModel, maxConcurrentCapabilityQueries+2)
			for i := range models {
				models[i] = ollamaLocalModel{
					Name: fmt.Sprintf("model-%d", i), Digest: fmt.Sprintf("digest-%d", i),
				}
			}

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: models})
				case "/api/show":
					current := active.Add(1)
					for {
						previous := peak.Load()
						if current <= previous || peak.CompareAndSwap(previous, current) {
							break
						}
					}
					started <- struct{}{}
					<-release
					active.Add(-1)
					_ = json.NewEncoder(w).Encode(map[string]any{"capabilities": []string{"completion"}})
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			result := make(chan []api.ActionDesc, 1)
			go func() { result <- newTestOllama(server.URL).ListActions(t.Context()) }()
			for range maxConcurrentCapabilityQueries {
				select {
				case <-started:
				case <-time.After(time.Second):
					close(release)
					t.Fatal("capability queries did not reach the concurrency limit")
				}
			}
			select {
			case <-started:
				close(release)
				t.Fatalf("more than %d capability queries ran concurrently", maxConcurrentCapabilityQueries)
			case <-time.After(50 * time.Millisecond):
			}
			close(release)
			if actions := <-result; len(actions) != len(models) {
				t.Fatalf("ListActions() returned %d actions, want %d", len(actions), len(models))
			}
			if got := peak.Load(); got != maxConcurrentCapabilityQueries {
				t.Errorf("peak concurrent queries = %d, want %d", got, maxConcurrentCapabilityQueries)
			}
		})

		t.Run("cancellation does not return a partial list", func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/tags":
					_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{
						{Name: "model-a", Digest: "a"}, {Name: "model-b", Digest: "b"},
					}})
				case "/api/show":
					cancel()
					_ = json.NewEncoder(w).Encode(map[string]any{"capabilities": []string{"completion"}})
				}
			}))
			defer server.Close()

			if actions := newTestOllama(server.URL).ListActions(ctx); actions != nil {
				t.Fatalf("ListActions() returned partial actions after cancellation: %v", actions)
			}
		})
	})

	t.Run("ResolveAction", func(t *testing.T) {
		o := newTestOllama("http://localhost:11434")

		t.Run("model action type", func(t *testing.T) {
			action := o.ResolveAction(api.ActionTypeModel, "llama3:latest")
			if action == nil {
				t.Fatal("ResolveAction() returned nil for model type")
			}
			desc := action.Desc()
			if desc.Name != "ollama/llama3:latest" {
				t.Errorf("ResolveAction() name = %q, want %q", desc.Name, "ollama/llama3:latest")
			}
		})

		t.Run("non-model action type", func(t *testing.T) {
			action := o.ResolveAction(api.ActionTypeExecutablePrompt, "llama3:latest")
			if action != nil {
				t.Error("ResolveAction() should return nil for non-model action type")
			}
		})

		t.Run("preserves dynamic model defaults when capabilities are unavailable", func(t *testing.T) {
			server := httptest.NewServer(http.NotFoundHandler())
			defer server.Close()

			action := newTestOllama(server.URL).ResolveAction(api.ActionTypeModel, "brand-new-model")
			if action == nil {
				t.Fatal("ResolveAction() returned nil for model type")
			}
			supports := modelSupportsMetadata(t, action.Desc())
			if supports["tools"] != true || supports["media"] != true {
				t.Errorf("fallback supports = %v, want tools and media enabled", supports)
			}
		})

		t.Run("works before Init", func(t *testing.T) {
			var called atomic.Bool
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called.Store(true)
				http.Error(w, "unexpected request", http.StatusInternalServerError)
			}))
			defer server.Close()

			o := &Ollama{ServerAddress: server.URL}
			action := o.ResolveAction(api.ActionTypeModel, "new-model")
			if action == nil {
				t.Fatal("ResolveAction() returned nil for model type")
			}
			supports := modelSupportsMetadata(t, action.Desc())
			if supports["tools"] != true {
				t.Errorf("resolved model supports = %v, want historical dynamic defaults", supports)
			}
			if called.Load() {
				t.Error("ResolveAction() performed network I/O")
			}
		})
	})

	t.Run("newModel", func(t *testing.T) {
		o := newTestOllama("http://localhost:11434")
		model := o.newModel("test-model", ai.ModelOptions{Supports: &defaultOllamaSupports})
		if model == nil {
			t.Fatal("newModel() returned nil")
		}
		action, ok := model.(api.Action)
		if !ok {
			t.Fatal("newModel() result does not implement api.Action")
		}
		desc := action.Desc()
		if desc.Name != "ollama/test-model" {
			t.Errorf("newModel() name = %q, want %q", desc.Name, "ollama/test-model")
		}
	})
}

func TestParseThinking(t *testing.T) {
	tests := []struct {
		name         string
		content      string
		wantThinking string
		wantRest     string
	}{
		{
			name:         "Single think tag",
			content:      "<think>I am thinking</think>Hello world",
			wantThinking: "I am thinking",
			wantRest:     "Hello world",
		},
		{
			name:         "Single thinking tag",
			content:      "<thinking>I am thinking</thinking>Hello world",
			wantThinking: "I am thinking",
			wantRest:     "Hello world",
		},
		{
			name:         "Multiple think tags",
			content:      "<think>First thought</think> Some text <think>Second thought</think> Final text",
			wantThinking: "First thought\n\nSecond thought",
			wantRest:     "Some text  Final text",
		},
		{
			name:         "Mixed think and thinking tags",
			content:      "<think>First thought</think> <thinking>Second thought</thinking> Final text",
			wantThinking: "First thought\n\nSecond thought",
			wantRest:     "Final text",
		},
		{
			name:         "Multiline thinking",
			content:      "<think>\nLine 1\nLine 2\n</think>Hello",
			wantThinking: "Line 1\nLine 2",
			wantRest:     "Hello",
		},
		{
			name:         "No thinking tags",
			content:      "Just plain text",
			wantThinking: "",
			wantRest:     "Just plain text",
		},
		{
			name:         "Case insensitive tags",
			content:      "<THINK>Shouting thoughts</THINK>Quiet response",
			wantThinking: "Shouting thoughts",
			wantRest:     "Quiet response",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotThinking, gotRest := parseThinking(tt.content)
			if gotThinking != tt.wantThinking {
				t.Errorf("parseThinking() gotThinking = %q, want %q", gotThinking, tt.wantThinking)
			}
			if gotRest != tt.wantRest {
				t.Errorf("parseThinking() gotRest = %q, want %q", gotRest, tt.wantRest)
			}
		})
	}
}

func TestTranslateChatResponse(t *testing.T) {
	tests := []struct {
		name            string
		input           string
		thinkingEnabled bool
		want            *ai.ModelResponse
		wantReasoning   string
		wantErr         bool
	}{
		{
			name:            "Thinking field present (always honored regardless of thinkingEnabled)",
			input:           `{"model": "deepseek-r1", "created_at": "2024-06-20T12:34:56Z", "message": {"role": "assistant", "content": "Hello", "thinking": "I should say hello"}}`,
			thinkingEnabled: false,
			want: &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewReasoningPart("I should say hello", nil),
						ai.NewTextPart("Hello"),
					},
				},
			},
			wantReasoning: "I should say hello",
		},
		{
			name:            "Thinking in content tag with thinking enabled",
			input:           `{"model": "deepseek-r1", "created_at": "2024-06-20T12:34:56Z", "message": {"role": "assistant", "content": "<think>I should say hello</think>Hello"}}`,
			thinkingEnabled: true,
			want: &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewReasoningPart("I should say hello", nil),
						ai.NewTextPart("Hello"),
					},
				},
			},
			wantReasoning: "I should say hello",
		},
		{
			name:            "Think tags in content NOT parsed when thinking disabled",
			input:           `{"model": "llama3", "created_at": "2024-06-20T12:34:56Z", "message": {"role": "assistant", "content": "<think>Not reasoning</think>Hello"}}`,
			thinkingEnabled: false,
			want: &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewTextPart("<think>Not reasoning</think>Hello"),
					},
				},
			},
			wantReasoning: "",
		},
		{
			name:            "Thinking in thinking tag with thinking enabled",
			input:           `{"model": "ollama-model", "created_at": "2024-06-20T12:34:56Z", "message": {"role": "assistant", "content": "<thinking>I am thinking</thinking>Hello"}}`,
			thinkingEnabled: true,
			want: &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewReasoningPart("I am thinking", nil),
						ai.NewTextPart("Hello"),
					},
				},
			},
			wantReasoning: "I am thinking",
		},
		{
			name:            "Multiple thinking blocks",
			input:           `{"model": "ollama-model", "created_at": "2024-06-20T12:34:56Z", "message": {"role": "assistant", "content": "<think>First</think><think>Second</think>Hello"}}`,
			thinkingEnabled: true,
			want: &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewReasoningPart("First\n\nSecond", nil),
						ai.NewTextPart("Hello"),
					},
				},
			},
			wantReasoning: "First\n\nSecond",
		},
		{
			name:            "Only thinking in content",
			input:           `{"model": "deepseek-r1", "created_at": "2024-06-20T12:34:56Z", "message": {"role": "assistant", "content": "<think>Just thinking</think>"}}`,
			thinkingEnabled: true,
			want: &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewReasoningPart("Just thinking", nil),
					},
				},
			},
			wantReasoning: "Just thinking",
		},
		{
			name:            "No thinking",
			input:           `{"model": "llama3", "created_at": "2024-06-20T12:34:56Z", "message": {"role": "assistant", "content": "Hello"}}`,
			thinkingEnabled: false,
			want: &ai.ModelResponse{
				Message: &ai.Message{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewTextPart("Hello"),
					},
				},
			},
			wantReasoning: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := translateChatResponse([]byte(tt.input), tt.thinkingEnabled)
			if (err != nil) != tt.wantErr {
				t.Errorf("translateChatResponse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if got.Reasoning() != tt.wantReasoning {
					t.Errorf("translateChatResponse() Reasoning = %q, want %q", got.Reasoning(), tt.wantReasoning)
				}
				if !equalContent(got.Message.Content, tt.want.Message.Content) {
					t.Errorf("translateChatResponse() got = %v, want %v", got.Message.Content, tt.want.Message.Content)
				}
			}
		})
	}
}

func TestGetModelCapabilities(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" {
			var req map[string]string
			json.NewDecoder(r.Body).Decode(&req)
			caps := map[string][]string{
				"gemma4:e2b":  {"completion", "vision", "audio", "tools", "thinking"},
				"llama3.2":    {"completion", "tools"},
				"nomic-embed": {"embedding"},
				"empty-model": {},
			}
			json.NewEncoder(w).Encode(map[string]any{"capabilities": caps[req["model"]]})
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	o := &Ollama{ServerAddress: server.URL, client: &http.Client{}, initted: true}

	t.Run("gemma4 reports tools capability", func(t *testing.T) {
		caps, err := o.getModelCapabilities(context.Background(), "gemma4:e2b")
		if err != nil {
			t.Fatalf("getModelCapabilities() error = %v", err)
		}
		if !slices.Contains(caps, "tools") {
			t.Errorf("expected 'tools' in capabilities, got %v", caps)
		}
	})

	t.Run("embed model has no tools", func(t *testing.T) {
		caps, err := o.getModelCapabilities(context.Background(), "nomic-embed")
		if err != nil {
			t.Fatalf("getModelCapabilities() error = %v", err)
		}
		if slices.Contains(caps, "tools") {
			t.Error("embed model should not have tools capability")
		}
	})

	t.Run("empty capabilities are detected", func(t *testing.T) {
		caps, err := o.getModelCapabilities(context.Background(), "empty-model")
		if err != nil {
			t.Fatalf("getModelCapabilities() error = %v", err)
		}
		if len(caps) != 0 {
			t.Errorf("expected empty capabilities, got %v", caps)
		}
	})

	t.Run("missing capabilities are not detected", func(t *testing.T) {
		caps, err := o.getModelCapabilities(context.Background(), "unknown-model")
		if err == nil {
			t.Errorf("expected missing capabilities to return an error, got %v", caps)
		}
	})

	t.Run("uses default HTTP client before Init", func(t *testing.T) {
		uninitialized := &Ollama{ServerAddress: server.URL}
		caps, err := uninitialized.getModelCapabilities(context.Background(), "llama3.2")
		if err != nil || !slices.Contains(caps, "tools") {
			t.Errorf("capabilities = %v, error = %v; want tools detected", caps, err)
		}
	})

	t.Run("normalizes trailing slash", func(t *testing.T) {
		withSlash := &Ollama{ServerAddress: server.URL + "/"}
		caps, err := withSlash.getModelCapabilities(context.Background(), "llama3.2")
		if err != nil || !slices.Contains(caps, "tools") {
			t.Errorf("capabilities = %v, error = %v; want tools detected", caps, err)
		}
	})
}

func TestModelCapabilitiesContext(t *testing.T) {
	tests := []struct {
		name        string
		timeout     int
		wantTimeout time.Duration
	}{
		{name: "unset timeout", timeout: 0, wantTimeout: modelCapabilitiesTimeout},
		{name: "long generation timeout", timeout: 30, wantTimeout: modelCapabilitiesTimeout},
		{name: "short configured timeout", timeout: 2, wantTimeout: 2 * time.Second},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := &Ollama{Timeout: tt.timeout}
			ctx, cancel := o.modelCapabilitiesContext(context.Background())
			defer cancel()

			deadline, ok := ctx.Deadline()
			if !ok {
				t.Fatal("capability context has no deadline")
			}
			remaining := time.Until(deadline)
			if remaining > tt.wantTimeout || remaining < tt.wantTimeout-time.Second {
				t.Errorf("capability timeout = %v, want approximately %v", remaining, tt.wantTimeout)
			}
		})
	}
}

func TestModelSupportsFromCapabilities(t *testing.T) {
	t.Run("dynamic capabilities with tools and vision", func(t *testing.T) {
		s := modelSupportsFromCapabilities([]string{"completion", "vision", "tools"})
		if !s.Tools {
			t.Error("expected Tools=true")
		}
		if !s.Media {
			t.Error("expected Media=true (vision)")
		}
	})

	t.Run("no tools in capabilities", func(t *testing.T) {
		s := modelSupportsFromCapabilities([]string{"completion"})
		if s.Tools {
			t.Error("expected Tools=false")
		}
	})

	t.Run("audio alone does not advertise unsupported media input", func(t *testing.T) {
		s := modelSupportsFromCapabilities([]string{"completion", "audio"})
		if s.Media {
			t.Error("expected Media=false when only audio is reported")
		}
	})

	t.Run("empty capabilities do not fall back", func(t *testing.T) {
		s := modelSupportsFromCapabilities([]string{})
		if s.Tools || s.Media {
			t.Errorf("expected empty capabilities to disable tools and media, got %+v", s)
		}
	})
}

func TestModelSupportsFromStaticLists(t *testing.T) {
	t.Run("qwen2.5 supports tools", func(t *testing.T) {
		s := modelSupportsFromStaticLists("qwen2.5")
		if !s.Tools {
			t.Error("expected Tools=true from static fallback")
		}
	})

	t.Run("unknown model has no optional capabilities", func(t *testing.T) {
		s := modelSupportsFromStaticLists("brand-new-model")
		if s.Tools {
			t.Error("expected Tools=false for unknown model")
		}
	})

	t.Run("tagged tool model preserves exact-match fallback", func(t *testing.T) {
		s := modelSupportsFromStaticLists("qwen2.5:7b")
		if s.Tools {
			t.Error("expected tagged qwen2.5 model to preserve legacy Tools=false fallback")
		}
	})

	t.Run("tagged media model preserves exact-match fallback", func(t *testing.T) {
		s := modelSupportsFromStaticLists("llava:34b")
		if s.Media {
			t.Error("expected tagged llava model to preserve legacy Media=false fallback")
		}
	})
}

func TestDefineModelNonChatDoesNotSupportTools(t *testing.T) {
	var called atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called.Store(true)
		http.Error(w, "unexpected request", http.StatusInternalServerError)
	}))
	defer server.Close()

	o := newTestOllama(server.URL)
	g := genkit.Init(context.Background())
	model := o.DefineModel(g, ModelDefinition{Name: "qwen2.5", Type: "generate"}, nil)
	action, ok := model.(api.Action)
	if !ok {
		t.Fatal("defined model does not implement api.Action")
	}
	supports := modelSupportsMetadata(t, action.Desc())
	if supports["tools"] != false {
		t.Errorf("non-chat model supports tools: %v", supports)
	}
	if called.Load() {
		t.Error("DefineModel() performed network I/O")
	}
}

func TestDefineModelPreservesStaticFallback(t *testing.T) {
	o := newTestOllama("http://localhost:11434")
	// An unavailable capability response uses the static fallback for explicitly
	// registered models, not the permissive dynamic fallback.
	o.cacheModelSupports("qwen2.5", "digest-1", &defaultOllamaSupports, false)
	g := genkit.Init(t.Context())
	model := o.DefineModel(g, ModelDefinition{Name: "qwen2.5", Type: "chat"}, nil)
	action, ok := model.(api.Action)
	if !ok {
		t.Fatal("defined model does not implement api.Action")
	}
	supports := modelSupportsMetadata(t, action.Desc())
	if supports["tools"] != true {
		t.Errorf("defined model supports = %v, want static tools fallback", supports)
	}
}

func TestDefineModelReusesDiscoveredCapabilities(t *testing.T) {
	var showCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{{
				Name: "brand-new-model", Digest: "digest-1",
			}}})
		case "/api/show":
			showCalls.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"capabilities": []string{"completion", "tools"},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	o := newTestOllama(server.URL)
	if actions := o.ListActions(t.Context()); len(actions) != 1 {
		t.Fatalf("ListActions() returned %d actions, want 1", len(actions))
	}
	g := genkit.Init(t.Context())
	model := o.DefineModel(g, ModelDefinition{Name: "brand-new-model", Type: "chat"}, nil)
	action, ok := model.(api.Action)
	if !ok {
		t.Fatal("defined model does not implement api.Action")
	}
	supports := modelSupportsMetadata(t, action.Desc())
	if supports["tools"] != true {
		t.Errorf("defined model supports = %v, want discovered tools capability", supports)
	}
	if got := showCalls.Load(); got != 1 {
		t.Errorf("/api/show calls = %d, want 1; DefineModel must not perform I/O", got)
	}
}
