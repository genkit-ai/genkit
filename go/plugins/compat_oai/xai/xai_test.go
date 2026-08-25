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

package xai_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("XAI_API_KEY", "")
	// An OPENAI_API_KEY must never be picked up as a fallback: sending it to
	// xAI would silently authenticate with the wrong provider's key.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		got := recover()
		if got != "xai plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&xai.XAI{}).Init(context.Background())
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
			"id":"c1","object":"chat.completion","created":1,"model":"grok-4.5",
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
	t.Setenv("XAI_API_KEY", "env-key")
	t.Setenv("XAI_BASE_URL", wrong.URL+"/v1")

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "struct-key", Opts: []option.RequestOption{option.WithBaseURL(right.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("xai/grok-4.5"))

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
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		var body struct {
			Model           string `json:"model"`
			Stream          bool   `json:"stream"`
			ReasoningEffort string `json:"reasoning_effort"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "grok-4.3" {
			t.Errorf("model = %q, want %q", body.Model, "grok-4.3")
		}
		if body.ReasoningEffort != "high" {
			t.Errorf("reasoning_effort = %q, want %q", body.ReasoningEffort, "high")
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"grok-4.3","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"grok-4.3","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"grok-4.3","choices":[{"index":0,"delta":{"content":"Grok streaming works"},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"grok-4.3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			} {
				_, _ = io.WriteString(w, "data: "+event+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"grok-4.3",
			"choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"Think carefully.","content":"Grok completion works"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	t.Setenv("XAI_API_KEY", "test-key")
	t.Setenv("XAI_BASE_URL", server.URL+"/v1")

	ctx := context.Background()
	plugin := &xai.XAI{}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("xai/grok-4.3"))

	if plugin.Name() != "xai" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "xai")
	}

	for _, modelID := range []string{
		"grok-4.5",
		"grok-4.3",
		"grok-4.20-0309-reasoning",
		"grok-4.20-0309-non-reasoning",
		"grok-build-0.1",
	} {
		model := genkit.LookupModel(g, "xai/"+modelID)
		if model == nil {
			t.Errorf("LookupModel(%q) = nil", modelID)
			continue
		}
		desc := model.(api.Action).Desc()
		if got, want := desc.Name, "xai/"+modelID; got != want {
			t.Errorf("%s Desc().Name = %q, want %q", modelID, got, want)
		}
		supports := desc.Metadata["model"].(map[string]any)["supports"].(map[string]any)
		for field, want := range map[string]bool{"media": true, "tools": true, "toolChoice": true, "multiturn": true} {
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
	// above asserts it reaches the wire as reasoning_effort.
	config := map[string]any{"reasoningEffort": "high"}
	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g, ai.WithPrompt("Say hi."), ai.WithConfig(config))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Fatalf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "Grok completion works" {
			t.Fatalf("Text() = %q, want %q", got, "Grok completion works")
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
		if got := text.String(); got != "Grok streaming works" {
			t.Fatalf("streamed text = %q, want %q", got, "Grok streaming works")
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
			Messages   []map[string]any `json:"messages"`
			Tools      []map[string]any `json:"tools"`
			ToolChoice string           `json:"tool_choice"`
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
			fn, ok := body.Tools[0]["function"].(map[string]any)
			if !ok || fn["name"] != "lookup" {
				t.Errorf("tool function = %#v, want name %q", body.Tools[0], "lookup")
			}
			// Grok models advertise tool choice, so the request carries it.
			if body.ToolChoice != "required" {
				t.Errorf("tool_choice = %q, want %q", body.ToolChoice, "required")
			}

			_, _ = io.WriteString(w, `{
				"id":"c-tool-1","object":"chat.completion","created":1,"model":"grok-4.5",
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

		var toolResult map[string]any
		for _, m := range body.Messages {
			if m["role"] == "tool" {
				toolResult = m
			}
		}
		if toolResult == nil || toolResult["tool_call_id"] != "call-1" {
			t.Errorf("tool result = %#v, want tool_call_id %q", toolResult, "call-1")
		}

		_, _ = io.WriteString(w, `{
			"id":"c-tool-2","object":"chat.completion","created":1,"model":"grok-4.5",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Tool loop complete"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("xai/grok-4.5"))

	lookup := genkit.DefineTool(g, "lookup", "Looks up a value.",
		func(_ *ai.ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return "result for " + input.Value, nil
		},
	)

	resp, err := genkit.Generate(ctx, g,
		ai.WithPrompt("Use the lookup tool."),
		ai.WithTools(lookup),
		ai.WithToolChoice(ai.ToolChoiceRequired),
	)
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
// camelCase config contract including the xAI-specific fields.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &xai.ChatConfig{ReasoningEffort: "low"}
	for _, name := range []string{"grok-4.5", "xai/grok-4.5"} {
		ref := xai.ModelRef(name, cfg)
		if want := "xai/grok-4.5"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	plugin := &xai.XAI{APIKey: "test-key"}
	g := genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	m := genkit.LookupModel(g, "xai/grok-4.5")
	if m == nil {
		t.Fatalf("%s not registered by Init", "grok-4.5")
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
	for _, key := range []string{"temperature", "maxOutputTokens", "stopSequences", "reasoningEffort", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}

	// The constraints xAI documents ride on the schema, where the framework
	// enforces them; the values are the ones docs.x.ai gives.
	for field, want := range map[string]map[string]any{
		"temperature":      {"minimum": 0.0, "maximum": 2.0},
		"frequencyPenalty": {"minimum": -2.0, "maximum": 2.0},
		"presencePenalty":  {"minimum": -2.0, "maximum": 2.0},
		"topLogProbs":      {"minimum": 0.0, "maximum": 8.0},
		"maxOutputTokens":  {"minimum": 1.0},
		"stopSequences":    {"maxItems": 4.0},
		"reasoningEffort":  {"enum": []any{"none", "low", "medium", "high", "xhigh"}},
		"serviceTier":      {"enum": []any{"default", "priority"}},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
	// topP deliberately carries no range: xAI documents none for it.
	topP, _ := props["topP"].(map[string]any)
	for _, key := range []string{"minimum", "maximum"} {
		if got, has := topP[key]; has {
			t.Errorf("topP %s = %v, want none since xAI documents no range", key, got)
		}
	}
	// searchParameters is deliberately absent: xAI retired live search, and
	// the API answers any request carrying it with 410 Gone.
	if props["searchParameters"] != nil {
		t.Error("config schema still declares searchParameters, which the API rejects with 410 Gone")
	}
}

// TestConfigConstraintsRejected pins that a documented constraint is enforced,
// not just advertised: validation runs against the schema the model declares,
// so an out-of-range value or an unknown level fails the request before it is
// sent and billed. The server answers success so that, were validation to let
// one through, the test fails on the nil error rather than on a network call.
func TestConfigConstraintsRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"grok-4.5",
			"choices":[{"index":0,"message":{"role":"assistant","content":"should not be reached"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	for name, tc := range map[string]struct {
		config map[string]any
		field  string
	}{
		"temperature above 2":     {map[string]any{"temperature": 3}, "temperature"},
		"unknown reasoning level": {map[string]any{"reasoningEffort": "ultra"}, "reasoningEffort"},
		"fifth stop sequence":     {map[string]any{"stopSequences": []any{"a", "b", "c", "d", "e"}}, "stopSequences"},
		"retired search field":    {map[string]any{"searchParameters": map[string]any{"mode": "on"}}, "searchParameters"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := genkit.Generate(ctx, g,
				ai.WithModelName("xai/grok-4.5"),
				ai.WithConfig(tc.config),
				ai.WithPrompt("hi"),
			)
			if err == nil {
				t.Fatal("Generate() error = nil, want the config rejected by schema validation")
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Errorf("error = %v, want it to name %q", err, tc.field)
			}
		})
	}
}

// TestModelRefConfigReachesTheWire pins the whole typed path: a ModelRef
// carrying a typed ChatConfig passes the framework's schema validation, the
// common fields land on their wire names (with maxOutputTokens moved to the
// max_completion_tokens xAI wants instead of the deprecated max_tokens), all
// in one Generate call.
func TestModelRefConfigReachesTheWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model               string  `json:"model"`
			Temperature         float64 `json:"temperature"`
			MaxTokens           *int    `json:"max_tokens"`
			MaxCompletionTokens int     `json:"max_completion_tokens"`
			ReasoningEffort     string  `json:"reasoning_effort"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "grok-4.5-latest" {
			t.Errorf("model = %q, want the pinned version %q", body.Model, "grok-4.5-latest")
		}
		if body.Temperature != 0.3 {
			t.Errorf("temperature = %v, want 0.3", body.Temperature)
		}
		if body.MaxTokens != nil {
			t.Errorf("max_tokens = %v, want it moved to max_completion_tokens", *body.MaxTokens)
		}
		if body.MaxCompletionTokens != 512 {
			t.Errorf("max_completion_tokens = %v, want 512", body.MaxCompletionTokens)
		}
		if body.ReasoningEffort != "high" {
			t.Errorf("reasoning_effort = %q, want %q", body.ReasoningEffort, "high")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"grok-4.5",
			"choices":[{"index":0,"message":{"role":"assistant","content":"typed config works"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(xai.ModelRef("grok-4.5", &xai.ChatConfig{
			RequestConfig:   compat_oai.RequestConfig{Version: "grok-4.5-latest"},
			Temperature:     openai.Ptr(0.3),
			MaxOutputTokens: 512,
			ReasoningEffort: "high",
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

// TestRequestControlsReachTheWire pins the xAI request controls that are not
// generation settings. The config type is the only way to set them, since the
// schema it advertises admits no properties of its own, so a field missing
// here is a field no caller can reach.
func TestRequestControlsReachTheWire(t *testing.T) {
	var mu sync.Mutex
	var body struct {
		ParallelToolCalls *bool  `json:"parallel_tool_calls"`
		User              string `json:"user"`
		ServiceTier       string `json:"service_tier"`
		PromptCacheKey    string `json:"prompt_cache_key"`
		ReasoningEffort   string `json:"reasoning_effort"`
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		err := json.NewDecoder(r.Body).Decode(&body)
		mu.Unlock()
		if err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"grok-4.6",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"end_turn"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(xai.ModelRef("grok-4.6", &xai.ChatConfig{
			ReasoningEffort:   xai.ReasoningEffortXHigh,
			ParallelToolCalls: openai.Ptr(false),
			User:              "user-7",
			ServiceTier:       xai.ServiceTierPriority,
			PromptCacheKey:    "prefix-a",
		})),
		ai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	// end_turn is xAI's own finish reason for an answer the model chose to end.
	if resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishReasonStop)
	}
	mu.Lock()
	defer mu.Unlock()
	if body.ParallelToolCalls == nil || *body.ParallelToolCalls {
		t.Errorf("parallel_tool_calls = %v, want false", body.ParallelToolCalls)
	}
	for _, tc := range []struct{ field, got, want string }{
		{"user", body.User, "user-7"},
		{"service_tier", body.ServiceTier, "priority"},
		{"prompt_cache_key", body.PromptCacheKey, "prefix-a"},
		{"reasoning_effort", body.ReasoningEffort, "xhigh"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %q, want %q", tc.field, tc.got, tc.want)
		}
	}
}

// TestCitationsReachTheCaller pins the response-metadata mapping: a
// completion carrying a citations array and a num_sources_used usage field
// hands both to the caller, the sources on the response metadata and the
// count on the usage's custom counters.
func TestCitationsReachTheCaller(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"grok-4.6",
			"citations":["https://x.ai/news","https://x.com/xai"],
			"choices":[{"index":0,"message":{"role":"assistant","content":"searched"},"finish_reason":"end_turn"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"num_sources_used":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("xai/grok-4.6"),
		ai.WithPrompt("What did xAI announce?"),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	custom, _ := resp.Custom.(map[string]any)
	got, _ := custom["citations"].([]any)
	want := []any{"https://x.ai/news", "https://x.com/xai"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("custom[citations] = %#v, want %#v", got, want)
	}
	if n := resp.Usage.Custom["numSourcesUsed"]; n != 2 {
		t.Errorf("Usage.Custom[numSourcesUsed] = %v, want 2", n)
	}
}

// TestJSONConfigExtraRidesTheWire pins the JSON contract of the inherited
// passthrough: a map config (what the Dev UI and cross-runtime callers send)
// passes this plugin's schema validation and its extra fields reach the wire
// under xAI's own names.
func TestJSONConfigExtraRidesTheWire(t *testing.T) {
	var mu sync.Mutex
	var deferred *bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Deferred *bool `json:"deferred"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		deferred = body.Deferred
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"grok-4.5",
			"choices":[{"index":0,"message":{"role":"assistant","content":"searched"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("xai/grok-4.5"))

	if _, err := genkit.Generate(ctx, g,
		ai.WithPrompt("What is new?"),
		ai.WithConfig(map[string]any{
			// deferred is a field the config does not declare, riding the
			// inherited passthrough.
			"extra": map[string]any{"deferred": true},
		}),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if deferred == nil || !*deferred {
		t.Errorf("deferred = %v, want the config's extra field on the wire", deferred)
	}
}

// TestDynamicListingAndResolution pins the on-demand surface: models the
// endpoint reports are listed with the plugin's config schema, and generating
// with an uncurated name resolves it instead of failing with model-not-found.
func TestDynamicListingAndResolution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"grok-brand-new","object":"model"}]}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"grok-brand-new",
			"choices":[{"index":0,"message":{"role":"assistant","content":"resolved"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	var listed *api.ActionDesc
	for _, desc := range plugin.ListActions(ctx) {
		if desc.Name == "xai/grok-brand-new" {
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
	if props["reasoningEffort"] == nil {
		t.Error("listed model schema is missing the plugin's reasoningEffort property")
	}

	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("xai/grok-brand-new"),
		ai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("Generate() with an uncurated model error = %v", err)
	}
	if got := resp.Text(); got != "resolved" {
		t.Fatalf("Text() = %q, want %q", got, "resolved")
	}
}

// TestConstrainedSupport pins the Grok 4 family carve-out. xAI documents that
// structured outputs combined with tools is only available on Grok 4 family
// models, so anything else must advertise no-tools rather than all, and a
// model resolved dynamically takes the narrower value since it may not be
// Grok 4. Claiming all where it does not hold would drop the schema
// instructions Genkit injects into the prompt for a request carrying tools.
func TestConstrainedSupport(t *testing.T) {
	t.Setenv("XAI_API_KEY", "test-key")

	ctx := context.Background()
	plugin := &xai.XAI{}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	constrained := func(a api.Action) ai.ConstrainedSupport {
		t.Helper()
		supports := a.Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
		got, _ := supports["constrained"].(ai.ConstrainedSupport)
		return got
	}

	for id, want := range map[string]ai.ConstrainedSupport{
		"grok-4.6":                     ai.ConstrainedSupportAll,
		"grok-4.5":                     ai.ConstrainedSupportAll,
		"grok-4.3":                     ai.ConstrainedSupportAll,
		"grok-4.20-0309-reasoning":     ai.ConstrainedSupportAll,
		"grok-4.20-0309-non-reasoning": ai.ConstrainedSupportAll,
		"grok-build-0.1":               ai.ConstrainedSupportNoTools,
	} {
		m := genkit.LookupModel(g, "xai/"+id)
		if m == nil {
			t.Errorf("LookupModel(%q) = nil", id)
			continue
		}
		if got := constrained(m.(api.Action)); got != want {
			t.Errorf("%s constrained = %q, want %q", id, got, want)
		}
	}

	// A model xAI adds later may sit outside the Grok 4 family.
	resolved := plugin.ResolveAction(api.ActionTypeModel, "grok-not-yet-released")
	if resolved == nil {
		t.Fatal("ResolveAction returned nil for an unknown model")
	}
	if got := constrained(resolved); got != ai.ConstrainedSupportNoTools {
		t.Errorf("dynamic constrained = %q, want %q", got, ai.ConstrainedSupportNoTools)
	}
}
