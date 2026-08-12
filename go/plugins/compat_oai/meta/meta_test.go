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

package meta_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/meta"
)

func TestPluginRegistersMuseSparkAndGenerates(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body struct {
			Model                string `json:"model"`
			PromptCacheRetention string `json:"prompt_cache_retention"`
			ReasoningEffort      string `json:"reasoning_effort"`
			Stream               bool   `json:"stream"`
			ToolChoice           string `json:"tool_choice"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != meta.ModelMuseSpark11 {
			t.Errorf("model = %q, want %q", body.Model, meta.ModelMuseSpark11)
		}
		if body.ReasoningEffort != "high" {
			t.Errorf("reasoning_effort = %q, want %q", body.ReasoningEffort, "high")
		}
		if body.PromptCacheRetention != "24h" {
			t.Errorf("prompt_cache_retention = %q, want %q", body.PromptCacheRetention, "24h")
		}
		if body.ToolChoice != "required" {
			t.Errorf("tool_choice = %q, want %q", body.ToolChoice, "required")
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"muse-spark-1.1","choices":[{"index":0,"delta":{"role":"assistant","content":"Muse "},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"muse-spark-1.1","choices":[{"index":0,"delta":{"content":"streaming works"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"muse-spark-1.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
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
			"model":"muse-spark-1.1",
			"choices":[{
				"index":0,
				"message":{"role":"assistant","content":"Muse completion works"},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	t.Setenv("META_API_KEY", "stale-alias-key")
	t.Setenv("MODEL_API_KEY", "test-key")
	t.Setenv("META_BASE_URL", server.URL+"/v1")

	ctx := context.Background()
	plugin := &meta.Meta{}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("meta/"+meta.ModelMuseSpark11),
	)

	if plugin.Name() != "meta" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "meta")
	}
	model := plugin.Model(g, meta.ModelMuseSpark11)
	if model == nil {
		t.Fatal("Muse Spark model was not registered")
	}
	modelMetadata := model.(api.Action).Desc().Metadata["model"].(map[string]any)
	supports := modelMetadata["supports"].(map[string]any)
	for capability, want := range map[string]any{
		"media":      true,
		"multiturn":  true,
		"systemRole": true,
		"tools":      true,
		"toolChoice": true,
	} {
		if got := supports[capability]; got != want {
			t.Errorf("%s support = %v, want %v", capability, got, want)
		}
	}

	lookup := genkit.DefineTool(
		g,
		"lookup",
		"Looks up a value.",
		func(_ *ai.ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return "result for " + input.Value, nil
		},
	)
	config := map[string]any{
		"prompt_cache_retention": "24h",
		"reasoning_effort":       "high",
	}
	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Reply with a short confirmation."),
			ai.WithConfig(config),
			ai.WithTools(lookup),
			ai.WithToolChoice(ai.ToolChoiceRequired),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Text(); got != "Muse completion works" {
			t.Fatalf("Text() = %q, want %q", got, "Muse completion works")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var streamed strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Reply with a short streaming confirmation."),
			ai.WithConfig(config),
			ai.WithTools(lookup),
			ai.WithToolChoice(ai.ToolChoiceRequired),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				streamed.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := streamed.String(); got != "Muse streaming works" {
			t.Fatalf("streamed text = %q, want %q", got, "Muse streaming works")
		}
		if got := resp.Text(); got != streamed.String() {
			t.Fatalf("final text = %q, want streamed %q", got, streamed.String())
		}
	})

	if requests != 2 {
		t.Fatalf("requests = %d, want 2", requests)
	}
}

func TestPluginInitializesWithoutAPIKey(t *testing.T) {
	t.Setenv("META_API_KEY", "")
	t.Setenv("MODEL_API_KEY", "")
	// The OpenAI SDK reads this automatically. The Meta plugin must override
	// that fallback rather than sending an unrelated credential to Meta.
	t.Setenv("OPENAI_API_KEY", "openai-key-must-not-leak")
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1","object":"chat.completion","created":1,
			"model":"muse-spark-1.1",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&meta.Meta{BaseURL: server.URL + "/v1"}))
	if model := genkit.LookupModel(g, "meta/"+meta.ModelMuseSpark11); model == nil {
		t.Fatal("Muse Spark model was not registered without an API key")
	}
	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("meta/"+meta.ModelMuseSpark11),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if strings.Contains(authorization, "openai-key-must-not-leak") {
		t.Fatalf("Authorization leaked OPENAI_API_KEY: %q", authorization)
	}
}
