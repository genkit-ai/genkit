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

package dashscope_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
)

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("DASHSCOPE_API_KEY", "")
	// An OPENAI_API_KEY must never be picked up as a fallback: sending it to
	// DashScope would silently authenticate with the wrong provider's key.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		got := recover()
		if got != "dashscope plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&dashscope.DashScope{}).Init(context.Background())
}

func TestPluginConfigPrecedence(t *testing.T) {
	var rightHit, wrongHit bool
	var gotAuth string

	right := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rightHit = true
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"qwen-plus",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer right.Close()

	wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wrongHit = true
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer wrong.Close()

	// Struct fields must win over env vars for both APIKey and BaseURL.
	t.Setenv("DASHSCOPE_API_KEY", "env-key")
	t.Setenv("DASHSCOPE_BASE_URL", wrong.URL+"/compatible-mode/v1")

	ctx := context.Background()
	plugin := &dashscope.DashScope{
		APIKey:  "struct-key",
		BaseURL: right.URL + "/compatible-mode/v1",
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	if _, err := genkit.Generate(ctx, g, ai.WithPrompt("hi")); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if !rightHit || wrongHit {
		t.Fatalf("rightHit = %v, wrongHit = %v, want struct fields to take precedence over env vars", rightHit, wrongHit)
	}
	if gotAuth != "Bearer struct-key" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer struct-key")
	}
}

func TestPluginRegistersModelsAndHandlesReasoning(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/compatible-mode/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/compatible-mode/v1/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body struct {
			Model          string `json:"model"`
			Stream         bool   `json:"stream"`
			EnableThinking bool   `json:"enable_thinking"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body.Model != "qwen-plus" {
			t.Errorf("model = %q, want %q", body.Model, "qwen-plus")
		}
		if !body.EnableThinking {
			t.Errorf("enable_thinking = %v, want true", body.EnableThinking)
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"qwen-plus","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"qwen-plus","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"qwen-plus","choices":[{"index":0,"delta":{"content":"Qwen streaming works"},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"qwen-plus","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			} {
				_, _ = io.WriteString(w, "data: "+event+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"qwen-plus",
			"choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"Think carefully.","content":"Qwen completion works"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	plugin := &dashscope.DashScope{APIKey: "test-key", BaseURL: server.URL + "/compatible-mode/v1"}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	if plugin.Name() != "dashscope" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "dashscope")
	}

	// Mirrors the Media flags set in supportedModels.
	textModels := []string{"qwen-flash", "qwen-plus", "qwen3.7-max", "qwen3-max", "qwen3-coder-plus"}
	mediaModels := []string{"qwen3.5-flash", "qwen3.5-plus", "qwen3.6-flash", "qwen3.6-plus", "qwen3.7-plus", "qwen3-vl-plus"}
	for _, group := range []struct {
		models    []string
		wantMedia bool
	}{
		{models: textModels, wantMedia: false},
		{models: mediaModels, wantMedia: true},
	} {
		for _, modelID := range group.models {
			model := plugin.Model(g, modelID)
			if model == nil {
				t.Errorf("Model(%q) = nil", modelID)
				continue
			}
			desc := model.(api.Action).Desc()
			if got, want := desc.Name, "dashscope/"+modelID; got != want {
				t.Errorf("%s Desc().Name = %q, want %q", modelID, got, want)
			}
			modelMetadata := desc.Metadata["model"].(map[string]any)
			supports := modelMetadata["supports"].(map[string]any)
			if got := supports["media"]; got != group.wantMedia {
				t.Errorf("%s media support = %v, want %v", modelID, got, group.wantMedia)
			}
			if got := supports["tools"]; got != true {
				t.Errorf("%s tools support = %v, want true", modelID, got)
			}
			if got := supports["toolChoice"]; got != false {
				t.Errorf("%s toolChoice support = %v, want false", modelID, got)
			}
			output, _ := supports["output"].([]string)
			if !slices.Equal(output, []string{"text", "json"}) {
				t.Errorf("%s output = %v, want [text json]", modelID, output)
			}
		}
	}

	config := map[string]any{"enable_thinking": true}
	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g, ai.WithPrompt("Say hi."), ai.WithConfig(config))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Fatalf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "Qwen completion works" {
			t.Fatalf("Text() = %q, want %q", got, "Qwen completion works")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var reasoning, text strings.Builder
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Say hi, streamed."),
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
		if got := text.String(); got != "Qwen streaming works" {
			t.Fatalf("streamed text = %q, want %q", got, "Qwen streaming works")
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

func TestPluginHandlesToolCalls(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body struct {
			Messages []map[string]any `json:"messages"`
			Tools    []map[string]any `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		if requests == 1 {
			if len(body.Tools) != 1 {
				t.Fatalf("tools = %#v, want one tool", body.Tools)
			}
			fn, ok := body.Tools[0]["function"].(map[string]any)
			if !ok || fn["name"] != "lookup" {
				t.Errorf("tool function = %#v, want name %q", body.Tools[0], "lookup")
			}

			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{
				"id":"c-tool-1","object":"chat.completion","created":1,"model":"qwen-plus",
				"choices":[{
					"index":0,
					"message":{
						"role":"assistant",
						"content":null,
						"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"value\":\"question\"}"}}]
					},
					"finish_reason":"tool_calls"
				}]
			}`)
			return
		}

		var assistant, toolResult map[string]any
		for _, m := range body.Messages {
			switch m["role"] {
			case "assistant":
				assistant = m
			case "tool":
				toolResult = m
			}
		}
		if assistant == nil {
			t.Error("second request has no assistant message")
		}
		if toolResult == nil || toolResult["tool_call_id"] != "call-1" {
			t.Errorf("tool result = %#v, want tool_call_id %q", toolResult, "call-1")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c-tool-2","object":"chat.completion","created":1,"model":"qwen-plus",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Tool loop complete"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &dashscope.DashScope{APIKey: "test-key", BaseURL: server.URL + "/compatible-mode/v1"}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	lookup := genkit.DefineTool(g, "lookup", "Looks up a value.",
		func(_ *ai.ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return "result for " + input.Value, nil
		},
	)

	resp, err := genkit.Generate(ctx, g, ai.WithPrompt("Use the lookup tool."), ai.WithTools(lookup))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "Tool loop complete" {
		t.Errorf("Text() = %q, want %q", got, "Tool loop complete")
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
			model: "qwen-plus",
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
			model: "qwen3-vl-plus",
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
					t.Fatalf("decode request: %v", err)
				}
				if got := body["model"]; got != test.model {
					t.Errorf("model = %v, want %q", got, test.model)
				}
				test.checkBody(t, body)

				w.Header().Set("Content-Type", "application/json")
				encodedResponse, err := json.Marshal(test.response)
				if err != nil {
					t.Fatalf("marshal response: %v", err)
				}
				_, _ = io.WriteString(w, `{
					"id":"c-shaping","object":"chat.completion","created":1,"model":"`+test.model+`",
					"choices":[{"index":0,"message":{"role":"assistant","content":`+string(encodedResponse)+`},"finish_reason":"stop"}]
				}`)
			}))
			defer server.Close()

			ctx := context.Background()
			plugin := &dashscope.DashScope{APIKey: "test-key", BaseURL: server.URL + "/compatible-mode/v1"}
			g := genkit.Init(ctx, genkit.WithPlugins(plugin))
			options := append([]ai.GenerateOption{ai.WithModelName("dashscope/" + test.model)}, test.options...)

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
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("unexpected HTTP request")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &dashscope.DashScope{APIKey: "test-key", BaseURL: server.URL + "/compatible-mode/v1"}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	_, err := genkit.Generate(ctx, g, ai.WithPrompt("Use a tool."), ai.WithToolChoice(ai.ToolChoiceRequired))
	if err == nil {
		t.Fatal("Generate() error = nil, want unsupported tool choice error")
	}
	if !strings.Contains(err.Error(), "does not support tool choice") {
		t.Fatalf("Generate() error = %q, want unsupported tool choice error", err)
	}
}

// TestPluginValidatesModelVersions locks in the registration metadata and
// version-validation behavior. It intentionally does not assert what the
// outbound "model" field is when a dated version is requested: the shared
// compat_oai adapter currently always sends the base model id regardless of
// the selected version (a known gap tracked separately, not specific to
// dashscope), so pinning that behavior here would be asserting a bug rather
// than a contract.
func TestPluginValidatesModelVersions(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c-version","object":"chat.completion","created":1,"model":"qwen-plus",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &dashscope.DashScope{APIKey: "test-key", BaseURL: server.URL + "/compatible-mode/v1"}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	model := plugin.Model(g, "qwen-plus")
	desc := model.(api.Action).Desc()
	modelMetadata := desc.Metadata["model"].(map[string]any)
	versions, _ := modelMetadata["versions"].([]string)
	wantVersions := []string{"qwen-plus", "qwen-plus-2025-07-28", "qwen-plus-2025-09-11", "qwen-plus-2025-12-01"}
	if !slices.Equal(versions, wantVersions) {
		t.Fatalf("versions = %v, want %v", versions, wantVersions)
	}

	t.Run("supported version is accepted", func(t *testing.T) {
		_, err := genkit.Generate(ctx, g,
			ai.WithPrompt("hi"),
			ai.WithConfig(map[string]any{"version": "qwen-plus-2025-09-11"}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
	})

	t.Run("unsupported version is rejected", func(t *testing.T) {
		before := requests
		_, err := genkit.Generate(ctx, g,
			ai.WithPrompt("hi"),
			ai.WithConfig(map[string]any{"version": "qwen-plus-9999-01-01"}),
		)
		if err == nil {
			t.Fatal("Generate() error = nil, want unsupported version error")
		}
		if requests != before {
			t.Errorf("requests = %d, want no additional request for rejected version", requests)
		}
	})
}
