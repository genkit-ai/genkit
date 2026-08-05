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

package kimi_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
)

func TestPluginRegistersKimiModelsAndHandlesReasoning(t *testing.T) {
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
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != kimi.ModelKimiK26 {
			t.Errorf("model = %q, want %q", body.Model, kimi.ModelKimiK26)
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"content":"Final answer"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k2.6","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
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
			"model":"kimi-k2.6",
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

	ctx := context.Background()
	plugin := &kimi.Kimi{
		APIKey:  "test-key",
		BaseURL: server.URL + "/v1",
	}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("kimi/"+kimi.ModelKimiK26),
	)

	if plugin.Name() != "kimi" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "kimi")
	}
	for _, model := range []string{
		kimi.ModelKimiK3,
		kimi.ModelKimiK25,
		kimi.ModelKimiK26,
		kimi.ModelKimiK27Code,
		kimi.ModelKimiK27CodeHighspeed,
	} {
		if plugin.Model(g, model) == nil {
			t.Errorf("Model(%q) = nil", model)
		}
	}
	for _, model := range []string{
		kimi.ModelKimiK3,
		kimi.ModelKimiK25,
		kimi.ModelKimiK26,
		kimi.ModelKimiK27Code,
		kimi.ModelKimiK27CodeHighspeed,
	} {
		action := plugin.Model(g, model).(api.Action)
		modelMetadata := action.Desc().Metadata["model"].(map[string]any)
		supports := modelMetadata["supports"].(map[string]any)
		if got := supports["media"]; got != true {
			t.Errorf("%s media support = %v, want true", model, got)
		}
	}
	k25Metadata := plugin.Model(g, kimi.ModelKimiK25).(api.Action).
		Desc().Metadata["model"].(map[string]any)
	if got := k25Metadata["stage"]; got != ai.ModelStageDeprecated {
		t.Errorf("%s stage = %v, want %q", kimi.ModelKimiK25, got, ai.ModelStageDeprecated)
	}

	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g, ai.WithPrompt("Solve this."))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Fatalf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "Final answer" {
			t.Fatalf("Text() = %q, want %q", got, "Final answer")
		}
		if len(resp.Message.Content) != 2 ||
			!resp.Message.Content[0].IsReasoning() ||
			!resp.Message.Content[1].IsText() {
			t.Fatalf("content = %#v, want reasoning followed by text", resp.Message.Content)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var reasoning, text strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Solve this."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				for _, part := range chunk.Content {
					switch {
					case part.IsReasoning():
						reasoning.WriteString(part.Text)
					case part.IsText():
						text.WriteString(part.Text)
					}
				}
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
		if got := resp.Reasoning(); got != reasoning.String() {
			t.Fatalf("final reasoning = %q, want streamed %q", got, reasoning.String())
		}
		if got := resp.Text(); got != text.String() {
			t.Fatalf("final text = %q, want streamed %q", got, text.String())
		}
	})

	if requests != 2 {
		t.Fatalf("requests = %d, want 2", requests)
	}
}

func TestPluginPreservesReasoningAndConfigAcrossToolCalls(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body struct {
			Messages   []map[string]any `json:"messages"`
			Model      string           `json:"model"`
			Thinking   map[string]any   `json:"thinking"`
			ToolChoice string           `json:"tool_choice"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != kimi.ModelKimiK26 {
			t.Errorf("model = %q, want %q", body.Model, kimi.ModelKimiK26)
		}
		if got := body.Thinking["type"]; got != "enabled" {
			t.Errorf("thinking.type = %v, want %q", got, "enabled")
		}
		if got := body.Thinking["keep"]; got != "all" {
			t.Errorf("thinking.keep = %v, want %q", got, "all")
		}
		if body.ToolChoice != "required" {
			t.Errorf("tool_choice = %q, want %q", body.ToolChoice, "required")
		}

		w.Header().Set("Content-Type", "application/json")
		if requests == 1 {
			_, _ = io.WriteString(w, `{
				"id":"chatcmpl-tool-1",
				"object":"chat.completion",
				"created":1,
				"model":"kimi-k2.6",
				"choices":[{
					"index":0,
					"message":{
						"role":"assistant",
						"reasoning_content":"I should call the lookup tool.",
						"content":null,
						"tool_calls":[{
							"id":"call-1",
							"type":"function",
							"function":{"name":"lookup","arguments":"{\"value\":\"question\"}"}
						}]
					},
					"finish_reason":"tool_calls"
				}]
			}`)
			return
		}

		var assistant map[string]any
		for _, message := range body.Messages {
			if message["role"] == "assistant" {
				assistant = message
				break
			}
		}
		if assistant == nil {
			t.Error("second request has no assistant message")
		} else {
			if got := assistant["reasoning_content"]; got != "I should call the lookup tool." {
				t.Errorf("assistant reasoning_content = %v, want preserved reasoning", got)
			}
			if got := assistant["content"]; got == "I should call the lookup tool." {
				t.Errorf("assistant content incorrectly contains reasoning: %v", got)
			}
		}

		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-tool-2",
			"object":"chat.completion",
			"created":1,
			"model":"kimi-k2.6",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"reasoning_content":"The tool returned the result.",
					"content":"Tool loop complete"
				},
				"finish_reason":"stop"
			}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &kimi.Kimi{APIKey: "test-key", BaseURL: server.URL + "/v1"}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("kimi/"+kimi.ModelKimiK26),
	)
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

	resp, err := genkit.Generate(
		ctx,
		g,
		ai.WithPrompt("Use the lookup tool."),
		ai.WithTools(lookup),
		ai.WithToolChoice(ai.ToolChoiceRequired),
		ai.WithConfig(map[string]any{
			"thinking": map[string]any{
				"type": "enabled",
				"keep": "all",
			},
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "Tool loop complete" {
		t.Errorf("Text() = %q, want %q", got, "Tool loop complete")
	}
	if got := resp.Reasoning(); got != "The tool returned the result." {
		t.Errorf("Reasoning() = %q, want final-turn reasoning", got)
	}
	if requests != 2 {
		t.Errorf("requests = %d, want 2", requests)
	}
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("KIMI_API_KEY", "")
	t.Setenv("MOONSHOT_API_KEY", "")

	defer func() {
		got := recover()
		if got != "kimi plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&kimi.Kimi{}).Init(context.Background())
}
