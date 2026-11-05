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
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
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
	var mu sync.Mutex
	var rightHit, wrongHit bool
	var gotAuth string

	right := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		rightHit = true
		gotAuth = r.Header.Get("Authorization")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"qwen-plus",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer right.Close()

	wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		wrongHit = true
		mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer wrong.Close()

	// Explicit configuration must win over env vars: the struct field for the
	// key, and an Opts [option.WithBaseURL] for the endpoint, which the
	// plugin applies after its own defaults.
	t.Setenv("DASHSCOPE_API_KEY", "env-key")
	t.Setenv("DASHSCOPE_BASE_URL", wrong.URL+"/compatible-mode/v1")

	ctx := context.Background()
	plugin := &dashscope.DashScope{
		APIKey: "struct-key",
		Opts:   []option.RequestOption{option.WithBaseURL(right.URL + "/compatible-mode/v1")},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	if _, err := genkit.Generate(ctx, g, ai.WithPrompt("hi")); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !rightHit || wrongHit {
		t.Fatalf("rightHit = %v, wrongHit = %v, want explicit configuration to take precedence over env vars", rightHit, wrongHit)
	}
	if gotAuth != "Bearer struct-key" {
		t.Errorf("Authorization = %q, want %q", gotAuth, "Bearer struct-key")
	}
}

func TestPluginRegistersModelsAndHandlesReasoning(t *testing.T) {
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		mu.Unlock()
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

	plugin := &dashscope.DashScope{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/compatible-mode/v1")}}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	if plugin.Name() != "dashscope" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "dashscope")
	}

	// Mirrors the Media and Output claims in supportedModels: qwen3.7-max and
	// qwen3-coder-plus document structured output as unsupported so they
	// advertise text only, and the 2026-06-08 max snapshot alone adds media.
	textAndJSON := []string{"text", "json"}
	textOutputOnly := []string{"text"}
	for _, group := range []struct {
		models     []string
		wantMedia  bool
		wantOutput []string
	}{
		{models: []string{"qwen-flash", "qwen-plus", "qwen3-max"},
			wantMedia: false, wantOutput: textAndJSON},
		{models: []string{"qwen3.5-flash", "qwen3.5-plus", "qwen3.6-flash", "qwen3.6-plus", "qwen3.7-plus", "qwen3-vl-plus"},
			wantMedia: true, wantOutput: textAndJSON},
		{models: []string{"qwen3.7-max", "qwen3-coder-plus"},
			wantMedia: false, wantOutput: textOutputOnly},
		{models: []string{"qwen3.7-max-2026-06-08"},
			wantMedia: true, wantOutput: textOutputOnly},
	} {
		for _, modelID := range group.models {
			model := genkit.LookupModel(g, "dashscope/"+modelID)
			if model == nil {
				t.Errorf("LookupModel(%q) = nil", modelID)
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
			if !slices.Equal(output, group.wantOutput) {
				t.Errorf("%s output = %v, want %v", modelID, output, group.wantOutput)
			}
		}
	}

	// The Genkit config speaks the plugin's camelCase contract; the handler
	// above asserts it reaches the wire as enable_thinking.
	config := map[string]any{"enableThinking": true}
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

	mu.Lock()
	defer mu.Unlock()
	if requests != 2 {
		t.Fatalf("requests = %d, want 2", requests)
	}
}

func TestPluginHandlesToolCalls(t *testing.T) {
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		reqNum := requests
		mu.Unlock()
		var body struct {
			Messages []map[string]any `json:"messages"`
			Tools    []map[string]any `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		if reqNum == 1 {
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
	plugin := &dashscope.DashScope{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/compatible-mode/v1")}}
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
	mu.Lock()
	defer mu.Unlock()
	if requests != 2 {
		t.Errorf("requests = %d, want 2", requests)
	}
}

func TestPluginShapesJSONAndVisionRequests(t *testing.T) {
	const imageDataURI = "data:image/png;base64,iVBORw0KGgo="

	checkImageBody := func(t *testing.T, body map[string]any) {
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
	}

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
			// qwen3.7-max documents structured output as unsupported, so a
			// JSON request must reach it without the response_format DashScope
			// would fail on, carried by the framework's format instructions.
			name:  "json output without structured output support",
			model: "qwen3.7-max",
			options: []ai.GenerateOption{
				ai.WithPrompt("Return a JSON object."),
				ai.WithOutputFormat(ai.OutputFormatJSON),
			},
			checkBody: func(t *testing.T, body map[string]any) {
				if got, ok := body["response_format"]; ok {
					t.Errorf("response_format = %#v, want none for a model without structured output", got)
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
			checkBody: checkImageBody,
			response:  "A test image.",
		},
		{
			// The dated snapshot is the one qwen3.7-max entry that takes
			// media, so an image request through it must pass validation and
			// reach the wire.
			name:  "vision input on the multimodal max snapshot",
			model: "qwen3.7-max-2026-06-08",
			options: []ai.GenerateOption{
				ai.WithMessages(ai.NewUserMessage(
					ai.NewMediaPart("image/png", imageDataURI),
					ai.NewTextPart("Describe this image."),
				)),
			},
			checkBody: checkImageBody,
			response:  "A test image.",
		},
		{
			name:  "extra passthrough",
			model: "qwen-plus",
			options: []ai.GenerateOption{
				ai.WithPrompt("hi"),
				ai.WithConfig(map[string]any{
					"enableSearch": false,
					"extra": map[string]any{
						"enable_search":  true,
						"search_options": map[string]any{"forced_search": true},
					},
				}),
			},
			checkBody: func(t *testing.T, body map[string]any) {
				if got := body["enable_search"]; got != true {
					t.Errorf("enable_search = %v, want the extra winning over the extra the config wrote", got)
				}
				searchOptions, _ := body["search_options"].(map[string]any)
				if got := searchOptions["forced_search"]; got != true {
					t.Errorf("search_options = %#v, want the undeclared field on the wire", body["search_options"])
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
			plugin := &dashscope.DashScope{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/compatible-mode/v1")}}
			g := genkit.Init(ctx, genkit.WithPlugins(plugin))
			options := append([]ai.GenerateOption{ai.WithModelName("dashscope/" + test.model)}, test.options...)

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
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("unexpected HTTP request")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &dashscope.DashScope{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/compatible-mode/v1")}}
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
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c-version","object":"chat.completion","created":1,"model":"qwen-plus",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &dashscope.DashScope{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/compatible-mode/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("dashscope/qwen-plus"))

	model := genkit.LookupModel(g, "dashscope/qwen-plus")
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
		mu.Lock()
		before := requests
		mu.Unlock()
		_, err := genkit.Generate(ctx, g,
			ai.WithPrompt("hi"),
			ai.WithConfig(map[string]any{"version": "qwen-plus-9999-01-01"}),
		)
		if err == nil {
			t.Fatal("Generate() error = nil, want unsupported version error")
		}
		mu.Lock()
		defer mu.Unlock()
		if requests != before {
			t.Errorf("requests = %d, want no additional request for rejected version", requests)
		}
	})
}

// TestModelRefAndConfigSchema pins the call-site surface: the ref carries the
// prefixed name and the typed config, and the registered models advertise the
// camelCase config contract including the DashScope-specific fields.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &dashscope.ChatConfig{EnableThinking: openai.Ptr(true)}
	for _, name := range []string{"qwen-plus", "dashscope/qwen-plus"} {
		ref := dashscope.ModelRef(name, cfg)
		if want := "dashscope/qwen-plus"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	plugin := &dashscope.DashScope{APIKey: "test-key"}
	g := genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	m := genkit.LookupModel(g, "dashscope/qwen-plus")
	if m == nil {
		t.Fatal("qwen-plus not registered by Init")
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
	for _, key := range []string{"temperature", "maxOutputTokens", "stopSequences", "seed", "enableThinking", "thinkingBudget", "enableSearch", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}

	// The constraints DashScope documents ride on the schema, where the
	// framework enforces them, exclusive ends included: temperature is [0, 2)
	// and topP is (0, 1].
	for field, want := range map[string]map[string]any{
		"temperature":     {"minimum": 0.0, "exclusiveMaximum": 2.0},
		"topP":            {"exclusiveMinimum": 0.0, "maximum": 1.0},
		"presencePenalty": {"minimum": -2.0, "maximum": 2.0},
		"seed":            {"minimum": 0.0, "maximum": 2147483647.0},
		"maxOutputTokens": {"minimum": 1.0},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
}

// TestDynamicListingAndResolution pins the on-demand surface: models the
// endpoint reports are listed with the plugin's config schema, and generating
// with an uncurated name resolves it instead of failing with model-not-found.
func TestDynamicListingAndResolution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/compatible-mode/v1/models" {
			_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"qwen-brand-new","object":"model"}]}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"qwen-brand-new",
			"choices":[{"index":0,"message":{"role":"assistant","content":"resolved"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &dashscope.DashScope{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/compatible-mode/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	var listed *api.ActionDesc
	for _, desc := range plugin.ListActions(ctx) {
		if desc.Name == "dashscope/qwen-brand-new" {
			listed = &desc
			break
		}
	}
	if listed == nil {
		t.Fatal("ListActions() does not include the endpoint's model")
	}
	model := listed.Metadata["model"].(map[string]any)
	schema, ok := model["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("listed model has no customOptions: %v", model)
	}
	props, _ := schema["properties"].(map[string]any)
	if props["enableThinking"] == nil {
		t.Error("listed model schema is missing the plugin's enableThinking property")
	}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("dashscope/qwen-brand-new"),
		ai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("Generate() with an uncurated model error = %v", err)
	}
	if got := resp.Text(); got != "resolved" {
		t.Fatalf("Text() = %q, want %q", got, "resolved")
	}
}
