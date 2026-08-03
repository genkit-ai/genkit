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

package groq_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/groq"
)

func TestPluginRegistersGroqModelsAndForwardsConfig(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/openai/v1/chat/completions" && r.URL.Path != "/v1/chat/completions" {
			// BaseURL may include /openai/v1 or just /v1 depending on how the test sets it.
			if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
				t.Errorf("path = %q, want .../chat/completions", r.URL.Path)
			}
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if got := body["model"]; got != groq.ModelLlama3370bVersatile {
			t.Errorf("model = %v, want %q", got, groq.ModelLlama3370bVersatile)
		}

		stream, _ := body["stream"].(bool)
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"llama-3.3-70b-versatile","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			} {
				_, _ = io.WriteString(w, "data: "+event+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		if got := body["reasoning_format"]; got != "parsed" {
			t.Errorf("reasoning_format = %v, want %q", got, "parsed")
		}
		if got := body["service_tier"]; got != "on_demand" {
			t.Errorf("service_tier = %v, want %q", got, "on_demand")
		}
		if got := body["reasoning_effort"]; got != "high" {
			t.Errorf("reasoning_effort = %v, want %q", got, "high")
		}
		if got := body["include_reasoning"]; got != true {
			t.Errorf("include_reasoning = %v, want true", got)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1",
			"object":"chat.completion",
			"created":1,
			"model":"llama-3.3-70b-versatile",
			"choices":[{
				"index":0,
				"message":{"role":"assistant","content":"Final answer"},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &groq.Groq{
		APIKey:  "test-key",
		BaseURL: server.URL + "/openai/v1",
	}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("groq/"+groq.ModelLlama3370bVersatile),
	)

	if plugin.Name() != "groq" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "groq")
	}

	for _, model := range []string{
		groq.ModelLlama318bInstant,
		groq.ModelLlama3370bVersatile,
		groq.ModelGPTOss120b,
		groq.ModelGPTOss20b,
		groq.ModelCompound,
		groq.ModelCompoundMini,
		groq.ModelQwen3627b,
	} {
		if plugin.Model(g, model) == nil {
			t.Errorf("Model(%q) = nil", model)
		}
	}

	qwenMeta := plugin.Model(g, groq.ModelQwen3627b).(api.Action).
		Desc().Metadata["model"].(map[string]any)
	supports := qwenMeta["supports"].(map[string]any)
	if got := supports["media"]; got != true {
		t.Errorf("%s media support = %v, want true", groq.ModelQwen3627b, got)
	}
	llamaMeta := plugin.Model(g, groq.ModelLlama3370bVersatile).(api.Action).
		Desc().Metadata["model"].(map[string]any)
	llamaSupports := llamaMeta["supports"].(map[string]any)
	if got := llamaSupports["media"]; got != false {
		t.Errorf("%s media support = %v, want false", groq.ModelLlama3370bVersatile, got)
	}

	t.Run("complete with groq config extras", func(t *testing.T) {
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Solve this."),
			ai.WithConfig(map[string]any{
				"reasoning_effort":  "high",
				"reasoning_format":  "parsed",
				"include_reasoning": true,
				"service_tier":      "on_demand",
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Text(); got != "Final answer" {
			t.Fatalf("Text() = %q, want %q", got, "Final answer")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var text strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Say hello."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				text.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := text.String(); got != "Hello world" {
			t.Fatalf("streamed text = %q, want %q", got, "Hello world")
		}
		if got := resp.Text(); got != text.String() {
			t.Fatalf("final text = %q, want streamed %q", got, text.String())
		}
	})

	if requests != 2 {
		t.Fatalf("requests = %d, want 2", requests)
	}
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("GROQ_API_KEY", "")

	defer func() {
		got := recover()
		if got != "groq plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&groq.Groq{}).Init(context.Background())
}

func TestListActionsFiltersNonChatModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/models") {
			t.Errorf("path = %q, want .../models", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"object":"list",
			"data":[
				{"id":"llama-3.3-70b-versatile","object":"model","created":1,"owned_by":"groq"},
				{"id":"whisper-large-v3","object":"model","created":1,"owned_by":"groq"},
				{"id":"nomic-embed-text","object":"model","created":1,"owned_by":"groq"},
				{"id":"canopy-orpheus","object":"model","created":1,"owned_by":"groq"}
			]
		}`)
	}))
	defer server.Close()

	plugin := &groq.Groq{
		APIKey:  "test-key",
		BaseURL: server.URL + "/openai/v1",
	}
	_ = plugin.Init(context.Background())

	actions := plugin.ListActions(context.Background())
	if len(actions) != 1 {
		t.Fatalf("ListActions() len = %d, want 1; got %#v", len(actions), actions)
	}
	if got := actions[0].Name; got != "groq/"+groq.ModelLlama3370bVersatile {
		t.Fatalf("ListActions()[0].Name = %q, want chat model", got)
	}
}
