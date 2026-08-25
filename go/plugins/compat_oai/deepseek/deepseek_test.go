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
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")
	// An OPENAI_API_KEY must never be picked up as a fallback: sending it to
	// DeepSeek would silently authenticate with the wrong provider's key.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		got := recover()
		if got != "deepseek plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&deepseek.DeepSeek{}).Init(context.Background())
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
			"id":"c1","object":"chat.completion","created":1,"model":"deepseek-v4-pro",
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
	t.Setenv("DEEPSEEK_API_KEY", "env-key")
	t.Setenv("DEEPSEEK_BASE_URL", wrong.URL)

	ctx := context.Background()
	plugin := &deepseek.DeepSeek{APIKey: "struct-key", Opts: []option.RequestOption{option.WithBaseURL(right.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("deepseek/deepseek-v4-pro"))

	if _, err := genkit.Generate(ctx, g, ai.WithPrompt("hi")); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !rightHit || wrongHit {
		t.Fatalf("rightHit = %v, wrongHit = %v, want struct fields to take precedence over env vars", rightHit, wrongHit)
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
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body struct {
			Model           string         `json:"model"`
			Stream          bool           `json:"stream"`
			Thinking        map[string]any `json:"thinking"`
			ReasoningEffort string         `json:"reasoning_effort"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "deepseek-v4-pro" {
			t.Errorf("model = %q, want %q", body.Model, "deepseek-v4-pro")
		}
		if got := body.Thinking["type"]; got != "enabled" {
			t.Errorf("thinking.type = %v, want %q", got, "enabled")
		}
		// DeepSeek reads the effort only as the top-level OpenAI field;
		// nested inside thinking the service silently ignores it.
		if body.ReasoningEffort != "max" {
			t.Errorf("reasoning_effort = %q, want %q", body.ReasoningEffort, "max")
		}
		if got, ok := body.Thinking["reasoning_effort"]; ok {
			t.Errorf("thinking.reasoning_effort = %v, want the top-level field only", got)
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"content":"DeepSeek streaming works"},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			} {
				_, _ = io.WriteString(w, "data: "+event+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"deepseek-v4-pro",
			"choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"Think carefully.","content":"DeepSeek completion works"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	t.Setenv("DEEPSEEK_BASE_URL", server.URL)

	ctx := context.Background()
	plugin := &deepseek.DeepSeek{}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("deepseek/deepseek-v4-pro"))

	if plugin.Name() != "deepseek" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "deepseek")
	}

	for _, modelID := range []string{"deepseek-v4-flash", "deepseek-v4-pro"} {
		model := genkit.LookupModel(g, "deepseek/"+modelID)
		if model == nil {
			t.Errorf("LookupModel(%q) = nil", modelID)
			continue
		}
		desc := model.(api.Action).Desc()
		if got, want := desc.Name, "deepseek/"+modelID; got != want {
			t.Errorf("%s Desc().Name = %q, want %q", modelID, got, want)
		}
		supports := desc.Metadata["model"].(map[string]any)["supports"].(map[string]any)
		// The V4 models are text-only, so media support must stay off.
		for field, want := range map[string]bool{"media": false, "tools": true, "toolChoice": true, "multiturn": true} {
			if got := supports[field]; got != want {
				t.Errorf("%s %s support = %v, want %v", modelID, field, got, want)
			}
		}
		output, _ := supports["output"].([]string)
		if !slices.Equal(output, []string{"text", "json"}) {
			t.Errorf("%s output = %v, want [text json]", modelID, output)
		}
	}

	// The Genkit config speaks the plugin's camelCase contract; the handler
	// above asserts the effort reaches the wire as the top-level
	// reasoning_effort DeepSeek reads.
	config := map[string]any{"reasoningEffort": "max", "thinking": map[string]any{"type": "enabled"}}
	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g, ai.WithPrompt("Say hi."), ai.WithConfig(config))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Fatalf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "DeepSeek completion works" {
			t.Fatalf("Text() = %q, want %q", got, "DeepSeek completion works")
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
		if got := text.String(); got != "DeepSeek streaming works" {
			t.Fatalf("streamed text = %q, want %q", got, "DeepSeek streaming works")
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
			Tools    []map[string]any `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if reqNum == 1 {
			if len(body.Tools) != 1 {
				t.Fatalf("tools = %#v, want one tool", body.Tools)
			}
			_, _ = io.WriteString(w, `{
				"id":"c-tool-1","object":"chat.completion","created":1,"model":"deepseek-v4-pro",
				"choices":[{
					"index":0,
					"message":{
						"role":"assistant",
						"reasoning_content":"I should call the lookup tool.",
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
		} else if got := assistant["reasoning_content"]; got != "I should call the lookup tool." {
			t.Errorf("assistant reasoning_content = %v, want preserved reasoning", got)
		}
		if toolResult == nil || toolResult["tool_call_id"] != "call-1" {
			t.Errorf("tool result = %#v, want tool_call_id %q", toolResult, "call-1")
		}

		_, _ = io.WriteString(w, `{
			"id":"c-tool-2","object":"chat.completion","created":1,"model":"deepseek-v4-pro",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Tool loop complete"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &deepseek.DeepSeek{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("deepseek/deepseek-v4-pro"))

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

// TestModelRefAndConfigSchema pins the call-site surface: the ref carries the
// prefixed name and the typed config, and the registered models advertise the
// camelCase config contract including the DeepSeek-specific fields.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &deepseek.ChatConfig{Thinking: &deepseek.ThinkingConfig{Type: "disabled"}}
	for _, name := range []string{"deepseek-v4-pro", "deepseek/deepseek-v4-pro"} {
		ref := deepseek.ModelRef(name, cfg)
		if want := "deepseek/deepseek-v4-pro"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	plugin := &deepseek.DeepSeek{APIKey: "test-key"}
	g := genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	m := genkit.LookupModel(g, "deepseek/deepseek-v4-pro")
	if m == nil {
		t.Fatalf("%s not registered by Init", "deepseek-v4-pro")
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
	for _, key := range []string{"temperature", "maxOutputTokens", "stopSequences", "reasoningEffort", "thinking", "userId", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}

	// The constraints DeepSeek documents ride on the schema, where the
	// framework enforces them.
	for field, want := range map[string]map[string]any{
		"temperature":     {"minimum": 0.0, "maximum": 2.0},
		"topP":            {"maximum": 1.0},
		"maxOutputTokens": {"minimum": 1.0},
		"stopSequences":   {"maxItems": 16.0},
		"topLogProbs":     {"minimum": 0.0, "maximum": 20.0},
		"userId":          {"maxLength": 512.0, "pattern": "^[a-zA-Z0-9_-]*$"},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
	// topP has no documented lower bound, so the schema declares none.
	topP, _ := props["topP"].(map[string]any)
	if got, has := topP["minimum"]; has {
		t.Errorf("topP minimum = %v, want none since DeepSeek documents only the upper bound", got)
	}
	effort, _ := props["reasoningEffort"].(map[string]any)
	if got, want := effort["enum"], []any{string(deepseek.ReasoningEffortLow),
		string(deepseek.ReasoningEffortHigh), string(deepseek.ReasoningEffortMax)}; !reflect.DeepEqual(got, want) {
		t.Errorf("reasoningEffort enum = %#v, want %#v", got, want)
	}
	thinking, _ := props["thinking"].(map[string]any)
	thinkingProps, _ := thinking["properties"].(map[string]any)
	thinkingType, _ := thinkingProps["type"].(map[string]any)
	if got, want := thinkingType["enum"], []any{string(deepseek.ThinkingTypeEnabled),
		string(deepseek.ThinkingTypeDisabled)}; !reflect.DeepEqual(got, want) {
		t.Errorf("thinking.type enum = %#v, want %#v", got, want)
	}
}

// TestModelRefConfigReachesTheWire pins the whole typed path: a ModelRef
// carrying a typed ChatConfig passes the framework's schema validation, the
// common fields land on their wire names (maxOutputTokens on the max_tokens
// DeepSeek reads, not max_completion_tokens), and the thinking controls ride
// along, all in one Generate call.
func TestModelRefConfigReachesTheWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model               string         `json:"model"`
			Temperature         float64        `json:"temperature"`
			MaxTokens           int            `json:"max_tokens"`
			MaxCompletionTokens *int           `json:"max_completion_tokens"`
			Thinking            map[string]any `json:"thinking"`
			UserID              string         `json:"user_id"`
			User                string         `json:"user"`
			Logprobs            *bool          `json:"logprobs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "deepseek-v4-pro-0731" {
			t.Errorf("model = %q, want the pinned version %q", body.Model, "deepseek-v4-pro-0731")
		}
		if body.Temperature != 0.3 {
			t.Errorf("temperature = %v, want 0.3", body.Temperature)
		}
		if body.MaxTokens != 512 {
			t.Errorf("max_tokens = %v, want 512", body.MaxTokens)
		}
		if body.MaxCompletionTokens != nil {
			t.Errorf("max_completion_tokens = %v, want DeepSeek's max_tokens instead", *body.MaxCompletionTokens)
		}
		if got := body.Thinking["type"]; got != "disabled" {
			t.Errorf("thinking.type = %v, want %q", got, "disabled")
		}
		// DeepSeek partitions its cache by user_id; OpenAI's user is a
		// different field it does not read.
		if body.UserID != "tenant-42" {
			t.Errorf("user_id = %q, want %q", body.UserID, "tenant-42")
		}
		if body.User != "" {
			t.Errorf("user = %q, want it sent as user_id instead", body.User)
		}
		if body.Logprobs == nil || !*body.Logprobs {
			t.Errorf("logprobs = %v, want the config's extra field on the wire", body.Logprobs)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"deepseek-v4-pro",
			"choices":[{"index":0,"message":{"role":"assistant","content":"typed config works"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &deepseek.DeepSeek{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(deepseek.ModelRef("deepseek-v4-pro", &deepseek.ChatConfig{
			RequestConfig: compat_oai.RequestConfig{
				Version: "deepseek-v4-pro-0731",
				// logprobs is a field the config does not declare, riding
				// the inherited passthrough.
				Extra: map[string]any{"logprobs": true},
			},
			Temperature:     openai.Ptr(0.3),
			MaxOutputTokens: 512,
			UserID:          "tenant-42",
			Thinking:        &deepseek.ThinkingConfig{Type: "disabled"},
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

// TestDynamicListingAndResolution pins the on-demand surface: models the
// endpoint reports are listed with the plugin's config schema, and generating
// with an uncurated name resolves it instead of failing with model-not-found.
func TestDynamicListingAndResolution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/models" {
			_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"deepseek-brand-new","object":"model","owned_by":"deepseek"}]}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"deepseek-brand-new",
			"choices":[{"index":0,"message":{"role":"assistant","content":"resolved"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &deepseek.DeepSeek{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	var listed *api.ActionDesc
	for _, desc := range plugin.ListActions(ctx) {
		if desc.Name == "deepseek/deepseek-brand-new" {
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
	if props["thinking"] == nil {
		t.Error("listed model schema is missing the plugin's thinking property")
	}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("deepseek/deepseek-brand-new"),
		ai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("Generate() with an uncurated model error = %v", err)
	}
	if got := resp.Text(); got != "resolved" {
		t.Fatalf("Text() = %q, want %q", got, "resolved")
	}
}
