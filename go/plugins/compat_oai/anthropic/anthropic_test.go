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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go"
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
	for _, key := range []string{"temperature", "maxOutputTokens", "topP", "stopSequences", "thinking", "version"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}
	for _, key := range []string{"frequencyPenalty", "presencePenalty", "logProbs", "topLogProbs"} {
		if props[key] != nil {
			t.Errorf("config schema advertises %q, which the endpoint ignores", key)
		}
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

// TestDefineModelNilOptions covers the nil ModelOptions path for a model
// outside the curated list, and that registration makes the lookup helpers
// find it under either name form.
func TestDefineModelNilOptions(t *testing.T) {
	a := &Anthropic{}
	g := genkit.Init(context.Background(), genkit.WithPlugins(a))

	if _, err := a.RegisterModel(g, "anthropic/claude-something-new", nil); err != nil {
		t.Fatalf("RegisterModel() error = %v", err)
	}
	for _, name := range []string{"claude-something-new", "anthropic/claude-something-new"} {
		if !IsDefinedModel(g, name) {
			t.Errorf("IsDefinedModel(%q) = false, want the model registered under either form", name)
		}
		if a.Model(g, name) == nil {
			t.Errorf("Model(%q) = nil, want the model registered under either form", name)
		}
	}
}

// TestDynamicListingAndResolution pins the on-demand surface: the full,
// cursor-paged models list is returned (the models list is a native
// Anthropic endpoint, so requests carry x-api-key alongside the bearer token
// and page through has_more/last_id rather than OpenAI-style), models are
// described with the plugin's config schema, and generating with an
// uncurated name resolves it instead of failing with model-not-found.
func TestDynamicListingAndResolution(t *testing.T) {
	var modelsAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/models" {
			modelsAuth = r.Header.Get("x-api-key")
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
	if modelsAuth != "test-key" {
		t.Errorf("models request x-api-key = %q, want the API key", modelsAuth)
	}

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
