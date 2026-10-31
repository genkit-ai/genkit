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

package orcarouter_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/orcarouter"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// completion answers any chat request with a fixed completion, so a test can
// assert on what was sent rather than on what came back.
const completion = `{
	"id":"c1","object":"chat.completion","created":1,"model":"anthropic/claude-sonnet-4.5",
	"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
}`

func completionServer(t *testing.T, capture func(r *http.Request, body map[string]any)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if capture != nil {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode request: %v", err)
			}
			capture(r, body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("ORCAROUTER_API_KEY", "")
	// An OPENAI_API_KEY must never be picked up as a fallback: sending it to
	// OrcaRouter would silently authenticate with the wrong provider's key.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		got := recover()
		if got != "orcarouter plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&orcarouter.OrcaRouter{}).Init(context.Background())
}

func TestPluginConfigPrecedence(t *testing.T) {
	var mu sync.Mutex
	var rightHit, wrongHit bool
	var gotAuth string

	right := completionServer(t, func(r *http.Request, _ map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		rightHit = true
		gotAuth = r.Header.Get("Authorization")
	})
	defer right.Close()

	wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		wrongHit = true
		mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer wrong.Close()

	// Explicit configuration must win over env vars: the struct field for the
	// key, and an Opts [option.WithBaseURL] for the endpoint, which the plugin
	// applies after its own defaults.
	t.Setenv("ORCAROUTER_API_KEY", "env-key")
	t.Setenv("ORCAROUTER_BASE_URL", wrong.URL+"/api/v1")

	ctx := context.Background()
	plugin := &orcarouter.OrcaRouter{
		APIKey: "struct-key",
		Opts:   []option.RequestOption{option.WithBaseURL(right.URL)},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("orcarouter/anthropic/claude-sonnet-4.5"))

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

// TestVendorPrefixedModelIDs pins the naming that makes a gateway work: an
// OrcaRouter model ID carries its upstream vendor's prefix, so the Genkit
// action name has two slashes. The plugin's own prefix must be the only one
// stripped, and the vendor prefix must survive onto the wire, or the request
// names a model OrcaRouter does not serve.
func TestVendorPrefixedModelIDs(t *testing.T) {
	var mu sync.Mutex
	var gotModel string
	server := completionServer(t, func(_ *http.Request, body map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		gotModel, _ = body["model"].(string)
	})
	defer server.Close()

	const id = "anthropic/claude-sonnet-4.5"
	for _, name := range []string{id, "orcarouter/" + id} {
		ref := orcarouter.ModelRef(name, nil)
		if want := "orcarouter/" + id; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
	}

	ctx := context.Background()
	plugin := &orcarouter.OrcaRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orcarouter.ModelRef(id, nil)), ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Text(); got != "ok" {
		t.Errorf("Text() = %q, want %q", got, "ok")
	}
	mu.Lock()
	defer mu.Unlock()
	if gotModel != id {
		t.Errorf("model = %q, want the vendor-prefixed ID %q", gotModel, id)
	}
}

func TestConfigSchema(t *testing.T) {
	plugin := &orcarouter.OrcaRouter{APIKey: "test-key"}
	genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	resolved := plugin.ResolveAction(api.ActionTypeModel, "anthropic/claude-sonnet-4.5")
	if resolved == nil {
		t.Fatal("ResolveAction returned nil")
	}
	model := resolved.Desc().Metadata["model"].(map[string]any)
	schema, ok := model["customOptions"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions missing, got %v", model["customOptions"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("customOptions has no properties: %v", schema)
	}

	for _, key := range []string{
		"temperature", "topP", "maxOutputTokens", "stopSequences",
		"frequencyPenalty", "presencePenalty", "seed", "logProbs",
		"topLogProbs", "parallelToolCalls", "user", "reasoningEffort",
		"version", "extra",
	} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}

	// The constraints OrcaRouter's OpenAI-compatible surface documents ride on
	// the schema, where the framework enforces them before a request is sent
	// and billed.
	for field, want := range map[string]map[string]any{
		"temperature":      {"minimum": 0.0, "maximum": 2.0},
		"topP":             {"minimum": 0.0, "maximum": 1.0},
		"frequencyPenalty": {"minimum": -2.0, "maximum": 2.0},
		"presencePenalty":  {"minimum": -2.0, "maximum": 2.0},
		"topLogProbs":      {"minimum": 0.0, "maximum": 20.0},
		"maxOutputTokens":  {"minimum": 1.0},
		"stopSequences":    {"maxItems": 4.0},
		"reasoningEffort":  {"enum": []any{"low", "medium", "high"}},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}

	// n is deliberately absent: Genkit reads the first choice only.
	if props["n"] != nil {
		t.Error("config schema declares n, which must not be offered")
	}
}

// TestConfigConstraintsRejected pins that a documented constraint is enforced,
// not just advertised. The server answers success so that, were validation to
// let one through, the test fails on the nil error rather than on a network
// call.
func TestConfigConstraintsRejected(t *testing.T) {
	server := completionServer(t, nil)
	defer server.Close()

	ctx := context.Background()
	plugin := &orcarouter.OrcaRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	for name, tc := range map[string]struct {
		config map[string]any
		field  string
	}{
		"temperature above 2":     {map[string]any{"temperature": 3}, "temperature"},
		"presence penalty high":   {map[string]any{"presencePenalty": 2.5}, "presencePenalty"},
		"fifth stop sequence":     {map[string]any{"stopSequences": []any{"a", "b", "c", "d", "e"}}, "stopSequences"},
		"unknown reasoning level": {map[string]any{"reasoningEffort": "ultra"}, "reasoningEffort"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := genkit.Generate(ctx, g,
				ai.WithModelName("orcarouter/anthropic/claude-sonnet-4.5"),
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

// TestSamplingFieldsReachTheWire pins that the sampling fields the config
// declares land on the request body OrcaRouter sees.
func TestSamplingFieldsReachTheWire(t *testing.T) {
	var mu sync.Mutex
	var got map[string]any
	server := completionServer(t, func(_ *http.Request, body map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		got = body
	})
	defer server.Close()

	ctx := context.Background()
	plugin := &orcarouter.OrcaRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	deny := false
	seed := 42
	if _, err := genkit.Generate(ctx, g,
		ai.WithModel(orcarouter.ModelRef("deepseek/deepseek-v4-flash", &orcarouter.ChatConfig{
			Temperature:       openai.Ptr(0.5),
			MaxOutputTokens:   512,
			Seed:              &seed,
			User:              "u-1",
			ParallelToolCalls: &deny,
			ReasoningEffort:   orcarouter.ReasoningEffortLow,
		})),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	for field, want := range map[string]any{
		"temperature":         0.5,
		"max_tokens":          512.0,
		"seed":                42.0,
		"user":                "u-1",
		"parallel_tool_calls": false,
		"reasoning_effort":    "low",
	} {
		if !reflect.DeepEqual(got[field], want) {
			t.Errorf("%s = %#v, want %#v", field, got[field], want)
		}
	}
	// A field left at its zero value must not be sent.
	if _, has := got["top_p"]; has {
		t.Errorf("top_p = %#v, want it absent when unset", got["top_p"])
	}
}

// TestListActionsIsEmpty pins a deliberate choice rather than an oversight.
// OrcaRouter serves hundreds of models and a descriptor carries full request
// and response schemas, so listing the catalog would put megabytes on every
// reflection poll. Models stay reachable by name.
func TestListActionsIsEmpty(t *testing.T) {
	ctx := context.Background()
	plugin := &orcarouter.OrcaRouter{APIKey: "test-key"}
	genkit.Init(ctx, genkit.WithPlugins(plugin))

	if got := plugin.ListActions(ctx); len(got) != 0 {
		t.Errorf("ListActions() returned %d descriptors, want none", len(got))
	}
	if plugin.ResolveAction(api.ActionTypeModel, "openai/gpt-4o") == nil {
		t.Error("ResolveAction returned nil, so an unlisted model is unreachable")
	}
	// Only models resolve; asking for another action type must not invent one.
	if got := plugin.ResolveAction(api.ActionTypeEmbedder, "openai/text-embedding-3-small"); got != nil {
		t.Errorf("ResolveAction(embedder) = %v, want nil", got)
	}
}

// TestDynamicCapabilitiesAndOverride pins the capability policy. Every model
// resolves permissive, because a capability declared too narrow is refused by
// Genkit locally and blocks a model that works, while one declared too wide
// fails at OrcaRouter with the real reason. Constrained output is the
// exception, left unset so structured output falls back to prompt
// instructions that every model handles. Models is the correction.
func TestDynamicCapabilitiesAndOverride(t *testing.T) {
	ctx := context.Background()
	plugin := &orcarouter.OrcaRouter{
		APIKey: "test-key",
		Models: map[string]ai.ModelOptions{
			"mistralai/mistral-7b-instruct": {Supports: &ai.ModelSupports{
				Multiturn: true, Tools: true, SystemRole: true,
			}},
		},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	supports := func(id string) map[string]any {
		t.Helper()
		resolved := plugin.ResolveAction(api.ActionTypeModel, id)
		if resolved == nil {
			t.Fatalf("ResolveAction(%q) = nil", id)
		}
		return resolved.Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
	}

	got := supports("openai/gpt-4o")
	for _, key := range []string{"multiturn", "tools", "systemRole", "media", "toolChoice"} {
		if got[key] != true {
			t.Errorf("dynamic supports[%q] = %v, want true", key, got[key])
		}
	}
	if c, _ := got["constrained"].(ai.ConstrainedSupport); c != "" {
		t.Errorf("dynamic constrained = %q, want it unset so structured output falls back to prompt instructions", c)
	}

	if got := supports("mistralai/mistral-7b-instruct"); got["media"] != false {
		t.Errorf("overridden supports[media] = %v, want the Models entry to narrow it", got["media"])
	}

	// The narrowed entry is enforced, not just advertised: Genkit refuses the
	// media locally rather than paying for the upstream rejection.
	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("orcarouter/mistralai/mistral-7b-instruct"),
		ai.WithMessages(ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgo="))),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want the media refused by the overridden capabilities")
	}
	if !strings.Contains(err.Error(), "does not support media") {
		t.Errorf("error = %v, want it to name the missing media support", err)
	}
}
