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
//
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// TestChatConfigApply pins the wire contract of the Claude compat config: the
// camelCase fields land on their snake_case counterparts and thinking rides
// as the endpoint's extra field.
func TestChatConfigApply(t *testing.T) {
	cfg := ChatConfig{
		Temperature:     openai.Ptr(0.5),
		MaxOutputTokens: 1024,
		StopSequences:   []string{"END"},
		Thinking:        &ThinkingConfig{Type: "enabled", BudgetTokens: 2000},
	}

	var params openai.ChatCompletionNewParams
	cfg.ApplyToChatCompletion(&params)

	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if got := request["temperature"]; got != 0.5 {
		t.Errorf("temperature = %v, want 0.5", got)
	}
	if got := request["max_tokens"]; got != float64(1024) {
		t.Errorf("max_tokens = %v, want 1024", got)
	}
	thinking, ok := request["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking = %#v, want object", request["thinking"])
	}
	if got := thinking["type"]; got != "enabled" {
		t.Errorf("thinking.type = %v, want enabled", got)
	}
	if got := thinking["budget_tokens"]; got != float64(2000) {
		t.Errorf("thinking.budget_tokens = %v, want 2000", got)
	}
}

// TestModelConfigSchema pins what the models advertise: the curated camelCase
// config contract, without the OpenAI fields Anthropic documents as ignored.
func TestModelConfigSchema(t *testing.T) {
	a := &Anthropic{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	m := genkit.LookupModel(g, "anthropic/claude-3-5-haiku-20241022")
	if m == nil {
		t.Fatal("claude-3-5-haiku-20241022 not registered by Init")
	}
	model, ok := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata missing")
	}
	schema, ok := model["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions missing, got %v", model["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions has no properties: %v", schema)
	}
	for _, key := range []string{"temperature", "maxOutputTokens", "topP", "stopSequences", "thinking", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}
	for _, key := range []string{"frequencyPenalty", "presencePenalty", "logProbs", "topLogProbs"} {
		if props[key] != nil {
			t.Errorf("config schema advertises %q, which the endpoint ignores", key)
		}
	}

	// The constraints the endpoint documents ride on the schema, where the
	// framework enforces them: temperature runs 0 to 1, not OpenAI's 2, since
	// the endpoint caps greater values rather than honoring them.
	temp, _ := props["temperature"].(map[string]any)
	if got := temp["minimum"]; got != 0.0 {
		t.Errorf("temperature minimum = %#v, want 0", got)
	}
	if got := temp["maximum"]; got != 1.0 {
		t.Errorf("temperature maximum = %#v, want the endpoint's 1", got)
	}
	// topP carries no range and thinking.type no enum: the endpoint documents
	// neither a range nor a closed set, and Anthropic's thinking types have
	// grown before, so a list here would reject a value the endpoint accepts.
	topP, _ := props["topP"].(map[string]any)
	for _, key := range []string{"minimum", "maximum"} {
		if got, has := topP[key]; has {
			t.Errorf("topP %s = %v, want none since the endpoint documents no range", key, got)
		}
	}
	thinking, _ := props["thinking"].(map[string]any)
	thinkingProps, _ := thinking["properties"].(map[string]any)
	thinkingType, _ := thinkingProps["type"].(map[string]any)
	if got, has := thinkingType["enum"]; has {
		t.Errorf("thinking.type enum = %v, want none since the set is Anthropic's to grow", got)
	}
	// The API rejects thinking budgets under 1,024 tokens, so the schema
	// carries the floor.
	budget, _ := thinkingProps["budgetTokens"].(map[string]any)
	if got := budget["minimum"]; got != 1024.0 {
		t.Errorf("thinking.budgetTokens minimum = %#v, want Anthropic's 1024 floor", got)
	}
}

// TestThinkingBudgetFloorRejected pins the floor end to end: the API rejects
// budgets under 1,024 ("Input should be greater than or equal to 1024"), so
// boundary validation catches one before the wire.
func TestThinkingBudgetFloorRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request reached the server, want boundary validation to reject the config")
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &Anthropic{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	_, err := genkit.Generate(ctx, g,
		ai.WithModel(ModelRef("claude-3-5-haiku-20241022", &ChatConfig{
			Thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 512},
		})),
		ai.WithPrompt("hi"),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want the 512-token budget rejected")
	}
	if !strings.Contains(err.Error(), "budgetTokens") {
		t.Errorf("error = %v, want it to name budgetTokens", err)
	}
}

// TestModelRef pins the name a ref carries and that the typed config rides
// along.
func TestModelRef(t *testing.T) {
	cfg := &ChatConfig{MaxOutputTokens: 1024}

	for _, name := range []string{"claude-3-5-haiku-20241022", "anthropic/claude-3-5-haiku-20241022"} {
		ref := ModelRef(name, cfg)
		if want := "anthropic/claude-3-5-haiku-20241022"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}
}

// TestUncuratedModelResolves covers a model outside the curated list: it
// resolves on demand under either name form, and a [Anthropic.Models] entry
// describes it without anything having to register it first.
func TestUncuratedModelResolves(t *testing.T) {
	a := &Anthropic{Models: map[string]ai.ModelOptions{
		"claude-something-new": {Label: "Something New"},
	}}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	for _, name := range []string{"claude-something-new", "anthropic/claude-something-new"} {
		m := genkit.LookupModel(g, "anthropic/"+strings.TrimPrefix(name, "anthropic/"))
		if m == nil {
			t.Fatalf("LookupModel(%q) = nil, want the model resolved on demand", name)
		}
		if got := m.(*ai.ModelAction).Desc().Metadata["model"].(map[string]any)["label"]; got != "Something New" {
			t.Errorf("%s label = %v, want the Models entry's", name, got)
		}
	}
}

// TestDynamicListingAndResolution pins the on-demand surface: the full,
// cursor-paged models list is returned (the models list is a native
// Anthropic endpoint, so requests carry x-api-key alongside the bearer token
// and page through has_more/last_id rather than OpenAI-style), models are
// described with the plugin's config schema, and generating with an
// uncurated name resolves it instead of failing with model-not-found. The
// server answers only version-prefixed paths, pinning that an origin-form
// ANTHROPIC_BASE_URL gains the /v1 segment.
func TestDynamicListingAndResolution(t *testing.T) {
	var mu sync.Mutex
	var modelsAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/models" {
			mu.Lock()
			modelsAuth = r.Header.Get("x-api-key")
			mu.Unlock()
			switch r.URL.Query().Get("after_id") {
			case "":
				_, _ = io.WriteString(w, `{"data":[{"id":"claude-brand-new","type":"model"}],"has_more":true,"last_id":"claude-brand-new"}`)
			case "claude-brand-new":
				_, _ = io.WriteString(w, `{"data":[{"id":"claude-second-page","type":"model"}],"has_more":false,"last_id":"claude-second-page"}`)
			default:
				t.Errorf("unexpected after_id %q", r.URL.Query().Get("after_id"))
				_, _ = io.WriteString(w, `{"data":[],"has_more":false}`)
			}
			return
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"claude-brand-new",
			"choices":[{"index":0,"message":{"role":"assistant","content":"resolved"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("ANTHROPIC_BASE_URL", server.URL)

	ctx := context.Background()
	plugin := &Anthropic{}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	listed := map[string]bool{}
	for _, desc := range plugin.ListActions(ctx) {
		listed[desc.Name] = true
		if desc.Name == "anthropic/claude-brand-new" {
			model := desc.Metadata["model"].(map[string]any)
			schema, ok := model["customOptions"].(map[string]any)
			if !ok {
				t.Fatalf("listed model has no customOptions: %v", model)
			}
			props, _ := schema["properties"].(map[string]any)
			if props["thinking"] == nil {
				t.Error("listed model schema is missing the plugin's thinking property")
			}
		}
	}
	if !listed["anthropic/claude-brand-new"] || !listed["anthropic/claude-second-page"] {
		t.Fatalf("ListActions() = %v, want every page of the endpoint's models", listed)
	}
	mu.Lock()
	if modelsAuth != "test-key" {
		t.Errorf("models request x-api-key = %q, want the API key", modelsAuth)
	}
	mu.Unlock()

	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("anthropic/claude-brand-new"),
		ai.WithPrompt("hi"),
	)
	if err != nil {
		t.Fatalf("Generate() with an uncurated model error = %v", err)
	}
	if got := resp.Text(); got != "resolved" {
		t.Fatalf("Text() = %q, want %q", got, "resolved")
	}
}

// TestPluginConfigPrecedence pins the credential contract: the struct field
// wins over the environment, and either authenticates both surfaces the
// plugin speaks to, the chat endpoint as the bearer token and the native
// models list as x-api-key. A key arriving only through Opts is opaque to
// the plugin and not part of this contract.
func TestPluginConfigPrecedence(t *testing.T) {
	var mu sync.Mutex
	var chatAuth, modelsKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/models" {
			mu.Lock()
			modelsKey = r.Header.Get("x-api-key")
			mu.Unlock()
			_, _ = io.WriteString(w, `{"data":[],"has_more":false}`)
			return
		}
		mu.Lock()
		chatAuth = r.Header.Get("Authorization")
		mu.Unlock()
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"claude-3-5-haiku-20241022",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	t.Setenv("ANTHROPIC_BASE_URL", server.URL)

	ctx := context.Background()
	plugin := &Anthropic{APIKey: "struct-key"}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	plugin.ListActions(ctx)
	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("anthropic/claude-3-5-haiku-20241022"),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if chatAuth != "Bearer struct-key" {
		t.Errorf("chat Authorization = %q, want the struct key as the bearer token", chatAuth)
	}
	if modelsKey != "struct-key" {
		t.Errorf("models x-api-key = %q, want the struct key over the environment's", modelsKey)
	}
}

// TestExtraPassthroughRidesTheWire pins the inherited passthrough on this
// plugin's own config: an undeclared field lands at the top level of the
// request, and a collision with the extra the plugin itself writes (thinking)
// resolves toward the caller's.
func TestExtraPassthroughRidesTheWire(t *testing.T) {
	var mu sync.Mutex
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"claude-3-5-haiku-20241022",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("ANTHROPIC_BASE_URL", server.URL)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Anthropic{}))

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("anthropic/claude-3-5-haiku-20241022"),
		ai.WithPrompt("hi"),
		ai.WithConfig(map[string]any{
			"thinking": map[string]any{"type": "enabled", "budgetTokens": 2048},
			"extra": map[string]any{
				"thinking":       map[string]any{"type": "disabled"},
				"context_window": "long",
			},
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if got := body["context_window"]; got != "long" {
		t.Errorf("context_window = %v, want the undeclared field on the wire", got)
	}
	thinking, _ := body["thinking"].(map[string]any)
	if got := thinking["type"]; got != "disabled" {
		t.Errorf("thinking.type = %v, want the caller's extra winning over the config's own thinking", got)
	}
}

// TestListingStopsOnStuckCursor pins the pager's exit on a misbehaving
// endpoint: a page that reports has_more but repeats the same last_id would
// otherwise be fetched forever.
func TestListingStopsOnStuckCursor(t *testing.T) {
	var mu sync.Mutex
	pages := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		pages++
		n := pages
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if n > 2 {
			t.Errorf("page request %d, want the walk to stop once the cursor fails to advance", n)
			_, _ = io.WriteString(w, `{"data":[],"has_more":false}`)
			return
		}
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-stuck","type":"model"}],"has_more":true,"last_id":"claude-stuck"}`)
	}))
	defer server.Close()

	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("ANTHROPIC_BASE_URL", server.URL)

	ctx := context.Background()
	plugin := &Anthropic{}
	genkit.Init(ctx, genkit.WithPlugins(plugin))

	listed := false
	for _, desc := range plugin.ListActions(ctx) {
		if desc.Name == "anthropic/claude-stuck" {
			listed = true
		}
	}
	if !listed {
		t.Error("ListActions() dropped the stuck page's model")
	}
	mu.Lock()
	defer mu.Unlock()
	if pages != 2 {
		t.Errorf("page requests = %d, want the first page and the one that proves the cursor is stuck", pages)
	}
}
