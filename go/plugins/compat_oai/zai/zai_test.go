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
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/compat_oai/zai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestPluginRegistersGLMModelsAndHandlesReasoning(t *testing.T) {
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		mu.Unlock()
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
		if body.Model != "glm-5.1" {
			t.Errorf("model = %q, want %q", body.Model, "glm-5.1")
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
		genkit.WithDefaultModel("zai/glm-5.1"),
	)

	if plugin.Name() != "zai" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "zai")
	}

	textModels := []string{
		"glm-5.1",
		"glm-5-turbo",
		"glm-5",
		"glm-4.7",
		"glm-4.7-flash",
		"glm-4.7-flashx",
		"glm-4.6",
		"glm-4.5",
		"glm-4.5-air",
		"glm-4.5-x",
		"glm-4.5-airx",
		"glm-4.5-flash",
		"glm-4-32b-0414-128k",
	}
	visionModels := []string{
		"glm-5v-turbo",
		"glm-4.6v",
		"glm-4.6v-flash",
		"glm-4.6v-flashx",
		"glm-4.5v",
	}
	for _, group := range []struct {
		models    []string
		wantMedia bool
	}{
		{models: textModels, wantMedia: false},
		{models: visionModels, wantMedia: true},
	} {
		for _, modelName := range group.models {
			model := genkit.LookupModel(g, "zai/"+modelName)
			if model == nil {
				t.Errorf("LookupModel(%q) = nil", modelName)
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

	// The Genkit config speaks the plugin's camelCase contract; the handler
	// above asserts it reaches the wire as clear_thinking.
	config := map[string]any{
		"thinking": map[string]any{
			"type":          "enabled",
			"clearThinking": false,
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

	mu.Lock()
	defer mu.Unlock()
	if requests != 2 {
		t.Fatalf("requests = %d, want 2", requests)
	}
}

func TestPluginPreservesReasoningAcrossToolCalls(t *testing.T) {
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		reqNum := requests
		mu.Unlock()
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
		if reqNum == 1 {
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
	plugin := &zai.ZAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/api/paas/v4")}}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("zai/glm-5.1"),
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
				"type":          "enabled",
				"clearThinking": false,
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
	mu.Lock()
	defer mu.Unlock()
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
			model: "glm-5.1",
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
			model: "glm-5v-turbo",
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
		{
			name:  "extra passthrough",
			model: "glm-5.1",
			options: []ai.GenerateOption{
				ai.WithPrompt("hi"),
				ai.WithConfig(map[string]any{
					"thinking": map[string]any{"type": "disabled"},
					"extra": map[string]any{
						"thinking": map[string]any{"type": "enabled"},
						"user_id":  "app-user-1",
					},
				}),
			},
			checkBody: func(t *testing.T, body map[string]any) {
				if got := body["user_id"]; got != "app-user-1" {
					t.Errorf("user_id = %v, want the undeclared field on the wire", got)
				}
				thinking, _ := body["thinking"].(map[string]any)
				if got := thinking["type"]; got != "enabled" {
					t.Errorf("thinking.type = %v, want the caller's extra winning over the config's own thinking", got)
				}
			},
			response: "Extra fields ride.",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var mu sync.Mutex
			var requests int
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				requests++
				mu.Unlock()
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
			plugin := &zai.ZAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/api/paas/v4")}}
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
			mu.Lock()
			defer mu.Unlock()
			if requests != 1 {
				t.Errorf("requests = %d, want 1", requests)
			}
		})
	}
}

func TestPluginRejectsUnsupportedToolChoice(t *testing.T) {
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		mu.Unlock()
		t.Error("unexpected HTTP request")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &zai.ZAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/api/paas/v4")}}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("zai/glm-5.1"),
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
	mu.Lock()
	defer mu.Unlock()
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

// TestModelRefAndConfigSchema pins the call-site surface: the ref carries the
// prefixed name and the typed config, and the registered models advertise the
// camelCase config contract including the Z.ai-specific fields.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &zai.ChatConfig{Thinking: &zai.ThinkingConfig{Type: "enabled"}}
	for _, name := range []string{"glm-5", "zai/glm-5"} {
		ref := zai.ModelRef(name, cfg)
		if want := "zai/glm-5"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	t.Setenv("ZAI_API_KEY", "test-key")
	plugin := &zai.ZAI{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	m := genkit.LookupModel(g, "zai/glm-5")
	if m == nil {
		t.Fatal("glm-5 not registered by Init")
	}
	model := m.(api.Action).Desc().Metadata["model"].(map[string]any)
	schema, ok := model["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions missing, got %v", model["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions has no properties: %v", schema)
	}
	for _, key := range []string{"temperature", "maxOutputTokens", "stopSequences", "thinking", "doSample", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}

	// The constraints Z.ai documents ride on the schema, where the framework
	// enforces them: temperature stops at 1 rather than OpenAI's 2, and topP
	// starts at 0.01.
	for field, want := range map[string]map[string]any{
		"temperature":     {"minimum": 0.0, "maximum": 1.0},
		"topP":            {"minimum": 0.01, "maximum": 1.0},
		"maxOutputTokens": {"minimum": 1.0, "maximum": 131072.0},
		"stopSequences":   {"maxItems": 4.0},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
	thinking, _ := props["thinking"].(map[string]any)
	thinkingProps, _ := thinking["properties"].(map[string]any)
	thinkingType, _ := thinkingProps["type"].(map[string]any)
	if got, want := thinkingType["enum"], []any{
		string(zai.ThinkingTypeEnabled), string(zai.ThinkingTypeDisabled)}; !reflect.DeepEqual(got, want) {
		t.Errorf("thinking.type enum = %#v, want %#v", got, want)
	}
}

// TestModelRefConfigReachesTheWire pins the whole typed path: a ModelRef
// carrying a typed ChatConfig passes the framework's schema validation, the
// common fields land on their snake_case wire names, and the Z.ai extras ride
// along, all in one Generate call.
func TestModelRefConfigReachesTheWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model       string         `json:"model"`
			Temperature float64        `json:"temperature"`
			MaxTokens   int            `json:"max_tokens"`
			Thinking    map[string]any `json:"thinking"`
			DoSample    *bool          `json:"do_sample"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "glm-5-20260101" {
			t.Errorf("model = %q, want the pinned version %q", body.Model, "glm-5-20260101")
		}
		if body.Temperature != 0.3 {
			t.Errorf("temperature = %v, want 0.3", body.Temperature)
		}
		if body.MaxTokens != 512 {
			t.Errorf("max_tokens = %v, want 512", body.MaxTokens)
		}
		if got := body.Thinking["type"]; got != "enabled" {
			t.Errorf("thinking.type = %v, want enabled", got)
		}
		if body.DoSample == nil || *body.DoSample != true {
			t.Errorf("do_sample = %v, want true", body.DoSample)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"glm-5",
			"choices":[{"index":0,"message":{"role":"assistant","content":"typed config works"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &zai.ZAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(zai.ModelRef("glm-5", &zai.ChatConfig{
			RequestConfig:   compat_oai.RequestConfig{Version: "glm-5-20260101"},
			Temperature:     openai.Ptr(0.3),
			MaxOutputTokens: 512,
			Thinking:        &zai.ThinkingConfig{Type: "enabled"},
			DoSample:        openai.Ptr(true),
		})),
		ai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "typed config works" {
		t.Fatalf("Text() = %q, want %q", got, "typed config works")
	}
}

// TestPerRequestAPIKey pins the credential override path: a typed config
// carrying an APIKey authenticates that request alone with the override, the
// plugin's key serves requests without one, and the key never appears in the
// request body.
func TestPerRequestAPIKey(t *testing.T) {
	var mu sync.Mutex
	var auths []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		auths = append(auths, r.Header.Get("Authorization"))
		mu.Unlock()

		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		if strings.Contains(string(body), "override-key") || strings.Contains(string(body), "apiKey") {
			t.Errorf("request body leaks the API key: %s", body)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"glm-5",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &zai.ZAI{APIKey: "plugin-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	if _, err := genkit.Generate(ctx, g,
		ai.WithModel(zai.ModelRef("glm-5", &zai.ChatConfig{
			RequestConfig: compat_oai.RequestConfig{APIKey: "override-key"},
		})),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() with override error = %v", err)
	}

	if _, err := genkit.Generate(ctx, g,
		ai.WithModel(zai.ModelRef("glm-5", nil)),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() without override error = %v", err)
	}

	want := []string{"Bearer override-key", "Bearer plugin-key"}
	mu.Lock()
	defer mu.Unlock()
	if len(auths) != 2 || auths[0] != want[0] || auths[1] != want[1] {
		t.Fatalf("Authorization headers = %v, want %v", auths, want)
	}
}
