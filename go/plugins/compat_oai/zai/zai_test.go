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

package zai_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/zai"
)

func TestPluginRegistersGLMModelsAndHandlesReasoning(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/api/paas/v4/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/api/paas/v4/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body struct {
			Model    string         `json:"model"`
			Stream   bool           `json:"stream"`
			Thinking map[string]any `json:"thinking"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != zai.ModelGLM51 {
			t.Errorf("model = %q, want %q", body.Model, zai.ModelGLM51)
		}
		if got := body.Thinking["type"]; got != "enabled" {
			t.Errorf("thinking.type = %v, want %q", got, "enabled")
		}
		if got := body.Thinking["clear_thinking"]; got != false {
			t.Errorf("thinking.clear_thinking = %v, want false", got)
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"glm-5.1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"glm-5.1","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"glm-5.1","choices":[{"index":0,"delta":{"content":"GLM streaming works"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"glm-5.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
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
			"model":"glm-5.1",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"reasoning_content":"Think carefully.",
					"content":"GLM completion works"
				},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	t.Setenv("ZAI_API_KEY", "test-key")
	t.Setenv("ZAI_BASE_URL", server.URL+"/api/paas/v4")

	ctx := context.Background()
	plugin := &zai.ZAI{}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("zai/"+zai.ModelGLM51),
	)

	if plugin.Name() != "zai" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "zai")
	}

	textModels := []string{
		zai.ModelGLM51,
		zai.ModelGLM5Turbo,
		zai.ModelGLM5,
		zai.ModelGLM47,
		zai.ModelGLM47Flash,
		zai.ModelGLM47FlashX,
		zai.ModelGLM46,
		zai.ModelGLM45,
		zai.ModelGLM45Air,
		zai.ModelGLM45X,
		zai.ModelGLM45AirX,
		zai.ModelGLM45Flash,
		zai.ModelGLM432B0414128K,
	}
	visionModels := []string{
		zai.ModelGLM5VTurbo,
		zai.ModelGLM46V,
		zai.ModelGLM46VFlash,
		zai.ModelGLM46VFlashX,
		zai.ModelGLM45V,
	}
	for _, group := range []struct {
		models    []string
		wantMedia bool
	}{
		{models: textModels, wantMedia: false},
		{models: visionModels, wantMedia: true},
	} {
		for _, modelName := range group.models {
			model := plugin.Model(g, modelName)
			if model == nil {
				t.Errorf("Model(%q) = nil", modelName)
				continue
			}
			modelMetadata := model.(api.Action).Desc().Metadata["model"].(map[string]any)
			supports := modelMetadata["supports"].(map[string]any)
			if got := supports["media"]; got != group.wantMedia {
				t.Errorf("%s media support = %v, want %v", modelName, got, group.wantMedia)
			}
			if got := supports["tools"]; got != true {
				t.Errorf("%s tools support = %v, want true", modelName, got)
			}
			if got := supports["toolChoice"]; got != false {
				t.Errorf("%s toolChoice support = %v, want false", modelName, got)
			}
		}
	}

	config := map[string]any{
		"thinking": map[string]any{
			"type":           "enabled",
			"clear_thinking": false,
		},
	}
	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Reply with a short confirmation."),
			ai.WithConfig(config),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Fatalf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "GLM completion works" {
			t.Fatalf("Text() = %q, want %q", got, "GLM completion works")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var reasoning, text strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Reply with a short streaming confirmation."),
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
		if got := text.String(); got != "GLM streaming works" {
			t.Fatalf("streamed text = %q, want %q", got, "GLM streaming works")
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

func TestPluginPreservesReasoningAcrossToolCalls(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body struct {
			Messages []map[string]any `json:"messages"`
			Thinking map[string]any   `json:"thinking"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if got := body.Thinking["clear_thinking"]; got != false {
			t.Errorf("thinking.clear_thinking = %v, want false", got)
		}

		w.Header().Set("Content-Type", "application/json")
		if requests == 1 {
			_, _ = io.WriteString(w, `{
				"id":"chatcmpl-tool-1",
				"object":"chat.completion",
				"created":1,
				"model":"glm-5.1",
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
		} else if got := assistant["reasoning_content"]; got != "I should call the lookup tool." {
			t.Errorf("assistant reasoning_content = %v, want preserved reasoning", got)
		}

		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-tool-2",
			"object":"chat.completion",
			"created":1,
			"model":"glm-5.1",
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
	plugin := &zai.ZAI{APIKey: "test-key", BaseURL: server.URL + "/api/paas/v4"}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("zai/"+zai.ModelGLM51),
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
		ai.WithConfig(map[string]any{
			"thinking": map[string]any{
				"type":           "enabled",
				"clear_thinking": false,
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

func TestPluginShapesJSONAndVisionRequests(t *testing.T) {
	const imageDataURI = "data:image/png;base64,iVBORw0KGgo="

	tests := []struct {
		name      string
		model     string
		options   []ai.GenerateOption
		checkBody func(*testing.T, map[string]any)
		response  string
	}{
		{
			name:  "json output",
			model: zai.ModelGLM51,
			options: []ai.GenerateOption{
				ai.WithPrompt("Return a JSON object."),
				ai.WithOutputFormat(ai.OutputFormatJSON),
			},
			checkBody: func(t *testing.T, body map[string]any) {
				responseFormat, ok := body["response_format"].(map[string]any)
				if !ok {
					t.Fatalf("response_format = %#v, want object", body["response_format"])
				}
				if got := responseFormat["type"]; got != "json_object" {
					t.Errorf("response_format.type = %v, want %q", got, "json_object")
				}
			},
			response: `{"answer":"ok"}`,
		},
		{
			name:  "vision input",
			model: zai.ModelGLM5VTurbo,
			options: []ai.GenerateOption{
				ai.WithMessages(ai.NewUserMessage(
					ai.NewMediaPart("image/png", imageDataURI),
					ai.NewTextPart("Describe this image."),
				)),
			},
			checkBody: func(t *testing.T, body map[string]any) {
				messages, ok := body["messages"].([]any)
				if !ok || len(messages) != 1 {
					t.Fatalf("messages = %#v, want one message", body["messages"])
				}
				message, ok := messages[0].(map[string]any)
				if !ok {
					t.Fatalf("message = %#v, want object", messages[0])
				}
				content, ok := message["content"].([]any)
				if !ok || len(content) != 2 {
					t.Fatalf("content = %#v, want image and text parts", message["content"])
				}
				imagePart, ok := content[0].(map[string]any)
				if !ok {
					t.Fatalf("image part = %#v, want object", content[0])
				}
				if got := imagePart["type"]; got != "image_url" {
					t.Errorf("image part type = %v, want %q", got, "image_url")
				}
				imageURL, ok := imagePart["image_url"].(map[string]any)
				if !ok {
					t.Fatalf("image_url = %#v, want object", imagePart["image_url"])
				}
				if got := imageURL["url"]; got != imageDataURI {
					t.Errorf("image_url.url = %v, want %q", got, imageDataURI)
				}
			},
			response: "A test image.",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var requests int
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests++
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("decode request: %v", err)
					return
				}
				if got := body["model"]; got != test.model {
					t.Errorf("model = %v, want %q", got, test.model)
				}
				test.checkBody(t, body)

				w.Header().Set("Content-Type", "application/json")
				encodedResponse, err := json.Marshal(test.response)
				if err != nil {
					t.Errorf("marshal response: %v", err)
					return
				}
				_, _ = io.WriteString(w, `{
					"id":"chatcmpl-shaping",
					"object":"chat.completion",
					"created":1,
					"model":"`+test.model+`",
					"choices":[{
						"index":0,
						"message":{"role":"assistant","content":`+string(encodedResponse)+`},
						"finish_reason":"stop"
					}]
				}`)
			}))
			defer server.Close()

			ctx := context.Background()
			plugin := &zai.ZAI{APIKey: "test-key", BaseURL: server.URL + "/api/paas/v4"}
			g := genkit.Init(ctx, genkit.WithPlugins(plugin))
			options := append([]ai.GenerateOption{
				ai.WithModelName("zai/" + test.model),
			}, test.options...)

			resp, err := genkit.Generate(ctx, g, options...)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if got := resp.Text(); got != test.response {
				t.Errorf("Text() = %q, want %q", got, test.response)
			}
			if requests != 1 {
				t.Errorf("requests = %d, want 1", requests)
			}
		})
	}
}

func TestPluginRejectsUnsupportedToolChoice(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		t.Error("unexpected HTTP request")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &zai.ZAI{APIKey: "test-key", BaseURL: server.URL + "/api/paas/v4"}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("zai/"+zai.ModelGLM51),
	)

	_, err := genkit.Generate(
		ctx,
		g,
		ai.WithPrompt("Use a tool."),
		ai.WithToolChoice(ai.ToolChoiceRequired),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want unsupported tool choice error")
	}
	if !strings.Contains(err.Error(), "does not support tool choice") {
		t.Fatalf("Generate() error = %q, want unsupported tool choice error", err)
	}
	if requests != 0 {
		t.Errorf("requests = %d, want 0", requests)
	}
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("ZAI_API_KEY", "")

	defer func() {
		got := recover()
		if got != "zai plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&zai.ZAI{}).Init(context.Background())
}
