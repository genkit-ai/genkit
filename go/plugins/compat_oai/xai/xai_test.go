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
	"slices"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
)

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("XAI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		if got := recover(); got != "xai plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()
	(&xai.XAI{}).Init(context.Background())
}

func TestPluginRegistersModels(t *testing.T) {
	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key"}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	if got := plugin.Name(); got != "xai" {
		t.Fatalf("Name() = %q, want xai", got)
	}
	for _, tc := range []struct {
		model          string
		wantMedia      bool
		wantMultiturn  bool
		wantSystemRole bool
	}{
		{model: xai.ModelGrok3, wantMultiturn: true, wantSystemRole: true},
		{model: xai.ModelGrok3Fast, wantMultiturn: true, wantSystemRole: true},
		{model: xai.ModelGrok3Mini, wantMultiturn: true, wantSystemRole: true},
		{model: xai.ModelGrok3MiniFast, wantMultiturn: true, wantSystemRole: true},
		{model: xai.ModelGrok2Vision1212, wantMedia: true},
	} {
		t.Run(tc.model, func(t *testing.T) {
			model := plugin.Model(g, tc.model)
			if model == nil {
				t.Fatalf("Model(%q) = nil", tc.model)
			}
			desc := model.(api.Action).Desc()
			if got, want := desc.Name, "xai/"+tc.model; got != want {
				t.Errorf("Desc().Name = %q, want %q", got, want)
			}
			metadata := desc.Metadata["model"].(map[string]any)
			supports := metadata["supports"].(map[string]any)
			for _, support := range []struct {
				name string
				want bool
			}{
				{name: "media", want: tc.wantMedia},
				{name: "multiturn", want: tc.wantMultiturn},
				{name: "systemRole", want: tc.wantSystemRole},
				{name: "tools", want: true},
				{name: "toolChoice", want: false},
			} {
				got, ok := supports[support.name].(bool)
				if !ok || got != support.want {
					t.Errorf("%s support = %v, want %v", support.name, supports[support.name], support.want)
				}
			}
			output, _ := supports["output"].([]string)
			if !slices.Equal(output, []string{"text", "json"}) {
				t.Errorf("output = %v, want [text json]", output)
			}
		})
	}

	metadata := plugin.Model(g, xai.ModelGrok3).(api.Action).Desc().Metadata["model"].(map[string]any)
	properties := metadata["customOptions"].(map[string]any)["properties"].(map[string]any)
	reasoningEffort := properties["reasoningEffort"].(map[string]any)
	if enumValues, _ := reasoningEffort["enum"].([]any); !slices.Equal(enumValues, []any{"low", "medium", "high"}) {
		t.Errorf("reasoningEffort enum = %v, want [low medium high]", enumValues)
	}
	for _, name := range []string{"deferred", "webSearchOptions"} {
		if _, ok := properties[name]; !ok {
			t.Errorf("config schema is missing %s", name)
		}
	}
}

func TestPluginTranslatesConfig(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want bearer token", got)
		}

		data, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		var body struct {
			Model            string `json:"model"`
			Deferred         bool   `json:"deferred"`
			ReasoningEffort  string `json:"reasoning_effort"`
			WebSearchOptions struct {
				SearchContextSize string `json:"search_context_size"`
			} `json:"web_search_options"`
		}
		if err := json.Unmarshal(data, &body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != xai.ModelGrok3Mini {
			t.Errorf("model = %q, want %q", body.Model, xai.ModelGrok3Mini)
		}
		if !body.Deferred {
			t.Error("deferred = false, want true")
		}
		if body.ReasoningEffort != "high" {
			t.Errorf("reasoning_effort = %q, want high", body.ReasoningEffort)
		}
		if got := body.WebSearchOptions.SearchContextSize; got != "high" {
			t.Errorf("web_search_options.search_context_size = %q, want high", got)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Errorf("decode request fields: %v", err)
			return
		}
		for _, name := range []string{"reasoningEffort", "webSearchOptions"} {
			if _, ok := fields[name]; ok {
				t.Errorf("request contains unconverted %s field", name)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1",
			"object":"chat.completion",
			"created":1,
			"model":"grok-3-mini",
			"choices":[{
				"index":0,
				"message":{"role":"assistant","content":"Grok works"},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &xai.XAI{APIKey: "test-key", BaseURL: server.URL + "/v1"}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("xai/"+xai.ModelGrok3Mini),
	)

	resp, err := genkit.Generate(
		ctx,
		g,
		ai.WithPrompt("Say hi."),
		ai.WithConfig(map[string]any{
			"deferred":        true,
			"reasoningEffort": "high",
			"webSearchOptions": map[string]any{
				"search_context_size": "high",
			},
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "Grok works" {
		t.Fatalf("Text() = %q, want %q", got, "Grok works")
	}
	if requests != 1 {
		t.Fatalf("requests = %d, want 1", requests)
	}
}
