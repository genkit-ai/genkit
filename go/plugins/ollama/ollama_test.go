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
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
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
			}
			json.NewEncoder(w).Encode(map[string]any{"capabilities": caps[req["model"]]})
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	o := &Ollama{ServerAddress: server.URL, client: &http.Client{}, initted: true}

	t.Run("gemma4 reports tools capability", func(t *testing.T) {
		caps := o.getModelCapabilities(context.Background(), "gemma4:e2b")
		if !slices.Contains(caps, "tools") {
			t.Errorf("expected 'tools' in capabilities, got %v", caps)
		}
	})

	t.Run("embed model has no tools", func(t *testing.T) {
		caps := o.getModelCapabilities(context.Background(), "nomic-embed")
		if slices.Contains(caps, "tools") {
			t.Error("embed model should not have tools capability")
		}
	})

	t.Run("unknown model returns empty", func(t *testing.T) {
		caps := o.getModelCapabilities(context.Background(), "unknown-model")
		if len(caps) > 0 {
			t.Errorf("expected empty capabilities for unknown model, got %v", caps)
		}
	})
}

func TestModelSupportsFromCapabilities(t *testing.T) {
	t.Run("dynamic capabilities with tools and vision", func(t *testing.T) {
		s := modelSupportsFromCapabilities([]string{"completion", "vision", "tools"}, "gemma4")
		if !s.Tools {
			t.Error("expected Tools=true")
		}
		if !s.Media {
			t.Error("expected Media=true (vision)")
		}
	})

	t.Run("no tools in capabilities", func(t *testing.T) {
		s := modelSupportsFromCapabilities([]string{"completion"}, "some-model")
		if s.Tools {
			t.Error("expected Tools=false")
		}
	})

	t.Run("fallback to static list for qwen2.5", func(t *testing.T) {
		s := modelSupportsFromCapabilities(nil, "qwen2.5")
		if !s.Tools {
			t.Error("expected Tools=true from static fallback")
		}
	})

	t.Run("fallback for unknown model", func(t *testing.T) {
		s := modelSupportsFromCapabilities(nil, "brand-new-model")
		if s.Tools {
			t.Error("expected Tools=false for unknown model")
		}
	})
}
