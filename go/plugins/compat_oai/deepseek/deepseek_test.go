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

package deepseek_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
)

func TestPluginRegistersModelsAndHandlesReasoning(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body struct {
			Model     string `json:"model"`
			Stream    bool   `json:"stream"`
			MaxTokens int    `json:"max_tokens"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != deepseek.ModelDeepSeekReasoner {
			t.Errorf("model = %q, want %q", body.Model, deepseek.ModelDeepSeekReasoner)
		}
		if body.MaxTokens != 8192 {
			t.Errorf("max_tokens = %d, want 8192", body.MaxTokens)
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"Final answer"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			} {
				_, _ = io.WriteString(w, "data: "+event+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1",
			"object":"chat.completion",
			"created":1,
			"model":"deepseek-reasoner",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"reasoning_content":"Think carefully.",
					"content":"Final answer"
				},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	t.Setenv("DEEPSEEK_BASE_URL", server.URL)

	ctx := context.Background()
	plugin := &deepseek.DeepSeek{}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("deepseek/"+deepseek.ModelDeepSeekReasoner),
	)

	if plugin.Name() != "deepseek" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "deepseek")
	}
	for _, modelName := range []string{deepseek.ModelDeepSeekChat, deepseek.ModelDeepSeekReasoner} {
		model := plugin.Model(g, modelName)
		if model == nil {
			t.Errorf("Model(%q) = nil", modelName)
			continue
		}
		metadata := model.(api.Action).Desc().Metadata["model"].(map[string]any)
		supports := metadata["supports"].(map[string]any)
		if got := supports["media"]; got != false {
			t.Errorf("%s media support = %v, want false", modelName, got)
		}
		if got := supports["tools"]; got != true {
			t.Errorf("%s tools support = %v, want true", modelName, got)
		}
		configSchema := metadata["customOptions"].(map[string]any)
		properties := configSchema["properties"].(map[string]any)
		if _, ok := properties["maxOutputTokens"]; !ok {
			t.Errorf("%s customOptions does not contain maxOutputTokens", modelName)
		}
	}
	config := map[string]any{"maxOutputTokens": 8192}

	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g, ai.WithPrompt("Solve this."), ai.WithConfig(config))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Fatalf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "Final answer" {
			t.Fatalf("Text() = %q, want %q", got, "Final answer")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var reasoning, text strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Solve this."),
			ai.WithConfig(config),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				reasoning.WriteString(chunk.Reasoning())
				text.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := reasoning.String(); got != "Think carefully." {
			t.Fatalf("streamed reasoning = %q, want %q", got, "Think carefully.")
		}
		if got := text.String(); got != "Final answer" {
			t.Fatalf("streamed text = %q, want %q", got, "Final answer")
		}
		if resp.Reasoning() != reasoning.String() {
			t.Fatalf("final reasoning = %q, want streamed %q", resp.Reasoning(), reasoning.String())
		}
		if resp.Text() != text.String() {
			t.Fatalf("final text = %q, want streamed %q", resp.Text(), text.String())
		}
	})

	if requests != 2 {
		t.Fatalf("requests = %d, want 2", requests)
	}
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")

	defer func() {
		if got := recover(); got != "deepseek plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()
	(&deepseek.DeepSeek{}).Init(context.Background())
}

func TestDynamicModelsUseDeepSeekMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/models")
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"deepseek-custom","object":"model","created":1,"owned_by":"deepseek"}]}`)
	}))
	defer server.Close()

	plugin := &deepseek.DeepSeek{APIKey: "test-key", BaseURL: server.URL}
	plugin.Init(context.Background())

	descs := plugin.ListActions(context.Background())
	if len(descs) != 1 {
		t.Fatalf("ListActions() returned %d actions, want 1", len(descs))
	}
	assertDeepSeekMetadata(t, descs[0])

	resolved := plugin.ResolveAction(api.ActionTypeModel, "deepseek-custom")
	if resolved == nil {
		t.Fatal("ResolveAction(model) = nil")
	}
	assertDeepSeekMetadata(t, resolved.Desc())
	if got := plugin.ResolveAction(api.ActionTypeEmbedder, "deepseek-custom"); got != nil {
		t.Errorf("ResolveAction(embedder) = %v, want nil", got)
	}
}

func assertDeepSeekMetadata(t *testing.T, desc api.ActionDesc) {
	t.Helper()
	metadata := desc.Metadata["model"].(map[string]any)
	supports := metadata["supports"].(map[string]any)
	if got := supports["media"]; got != false {
		t.Errorf("media support = %v, want false", got)
	}
	if got := supports["tools"]; got != true {
		t.Errorf("tools support = %v, want true", got)
	}
	properties := metadata["customOptions"].(map[string]any)["properties"].(map[string]any)
	if _, ok := properties["maxOutputTokens"]; !ok {
		t.Error("customOptions does not contain maxOutputTokens")
	}
}
