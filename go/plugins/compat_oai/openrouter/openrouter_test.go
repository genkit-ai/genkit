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

package openrouter_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/openrouter"
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
	t.Setenv("OPENROUTER_API_KEY", "")
	// An OPENAI_API_KEY must never be picked up as a fallback: sending it to
	// OpenRouter would silently authenticate with the wrong provider's key.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		got := recover()
		if got != "openrouter plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&openrouter.OpenRouter{}).Init(context.Background())
}

func TestPluginConfigPrecedenceAndAttribution(t *testing.T) {
	var mu sync.Mutex
	var rightHit, wrongHit bool
	var gotAuth, gotReferer, gotTitle string

	right := completionServer(t, func(r *http.Request, _ map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		rightHit = true
		gotAuth = r.Header.Get("Authorization")
		gotReferer = r.Header.Get("HTTP-Referer")
		gotTitle = r.Header.Get("X-Title")
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
	t.Setenv("OPENROUTER_API_KEY", "env-key")
	t.Setenv("OPENROUTER_BASE_URL", wrong.URL+"/api/v1")

	ctx := context.Background()
	plugin := &openrouter.OpenRouter{
		APIKey:  "struct-key",
		SiteURL: "https://example.test",
		AppName: "Genkit Test",
		Opts:    []option.RequestOption{option.WithBaseURL(right.URL)},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("openrouter/anthropic/claude-sonnet-4.5"))

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
	if gotReferer != "https://example.test" {
		t.Errorf("HTTP-Referer = %q, want the plugin's SiteURL", gotReferer)
	}
	if gotTitle != "Genkit Test" {
		t.Errorf("X-Title = %q, want the plugin's AppName", gotTitle)
	}
}

// TestVendorPrefixedModelIDs pins the naming that makes a gateway work: an
// OpenRouter model ID carries its upstream vendor's prefix, so the Genkit
// action name has two slashes. The plugin's own prefix must be the only one
// stripped, and the vendor prefix must survive onto the wire, or the request
// names a model OpenRouter does not serve.
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
	for _, name := range []string{id, "openrouter/" + id} {
		ref := openrouter.ModelRef(name, nil)
		if want := "openrouter/" + id; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
	}

	ctx := context.Background()
	plugin := &openrouter.OpenRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	resp, err := genkit.Generate(ctx, g, ai.WithModel(openrouter.ModelRef(id, nil)), ai.WithPrompt("hi"))
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
	plugin := &openrouter.OpenRouter{APIKey: "test-key"}
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
		"temperature", "topP", "topK", "maxOutputTokens", "stopSequences",
		"repetitionPenalty", "minP", "topA", "models", "provider", "reasoning",
		"plugins", "transforms", "sessionId", "serviceTier", "metadata",
		"version", "extra",
	} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}

	// The constraints OpenRouter documents ride on the schema, where the
	// framework enforces them before a request is sent and billed.
	for field, want := range map[string]map[string]any{
		"temperature":       {"minimum": 0.0, "maximum": 2.0},
		"topP":              {"minimum": 0.0, "maximum": 1.0},
		"topK":              {"minimum": 0.0},
		"repetitionPenalty": {"minimum": 0.0, "maximum": 2.0},
		"minP":              {"minimum": 0.0, "maximum": 1.0},
		"topA":              {"minimum": 0.0, "maximum": 1.0},
		"frequencyPenalty":  {"minimum": -2.0, "maximum": 2.0},
		"presencePenalty":   {"minimum": -2.0, "maximum": 2.0},
		"topLogProbs":       {"minimum": 0.0, "maximum": 20.0},
		"maxOutputTokens":   {"minimum": 1.0},
		"stopSequences":     {"maxItems": 4.0},
		"serviceTier":       {"enum": []any{"auto", "default", "fast", "flex", "priority", "scale"}},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}

	// Nested routing and reasoning objects carry their own documented sets.
	routing, _ := props["provider"].(map[string]any)
	routingProps, _ := routing["properties"].(map[string]any)
	for _, key := range []string{
		"order", "only", "ignore", "allowFallbacks", "requireParameters",
		"dataCollection", "zdr", "sort", "quantizations", "maxPrice",
		"preferredMinThroughput", "preferredMaxLatency",
	} {
		if routingProps[key] == nil {
			t.Errorf("provider schema is missing the %q property", key)
		}
	}
	if got := routingProps["sort"].(map[string]any)["enum"]; !reflect.DeepEqual(got, []any{"price", "throughput", "latency"}) {
		t.Errorf("provider.sort enum = %#v", got)
	}
	// Quantizations deliberately carries no enum: OpenRouter adds levels as
	// hardware gains them, and a closed list would reject one it accepts.
	if _, has := routingProps["quantizations"].(map[string]any)["enum"]; has {
		t.Error("provider.quantizations declares an enum, which would reject a level OpenRouter accepts")
	}

	reasoning, _ := props["reasoning"].(map[string]any)
	reasoningProps, _ := reasoning["properties"].(map[string]any)
	want := []any{"none", "minimal", "low", "medium", "high", "xhigh", "max"}
	if got := reasoningProps["effort"].(map[string]any)["enum"]; !reflect.DeepEqual(got, want) {
		t.Errorf("reasoning.effort enum = %#v, want %#v", got, want)
	}

	// n and route are deliberately absent: Genkit reads the first choice only,
	// and OpenRouter deprecated route in favor of provider sorting.
	for _, key := range []string{"n", "route", "usage"} {
		if props[key] != nil {
			t.Errorf("config schema declares %q, which must not be offered", key)
		}
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
	plugin := &openrouter.OpenRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	for name, tc := range map[string]struct {
		config map[string]any
		field  string
	}{
		"temperature above 2":     {map[string]any{"temperature": 3}, "temperature"},
		"repetition penalty high": {map[string]any{"repetitionPenalty": 2.5}, "repetitionPenalty"},
		"fifth stop sequence":     {map[string]any{"stopSequences": []any{"a", "b", "c", "d", "e"}}, "stopSequences"},
		"unknown reasoning level": {map[string]any{"reasoning": map[string]any{"effort": "ultra"}}, "effort"},
		"unknown service tier":    {map[string]any{"serviceTier": "turbo"}, "serviceTier"},
		"unknown data collection": {map[string]any{"provider": map[string]any{"dataCollection": "maybe"}}, "dataCollection"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := genkit.Generate(ctx, g,
				ai.WithModelName("openrouter/anthropic/claude-sonnet-4.5"),
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

// TestGatewayControlsReachTheWire pins the fields that are the reason to route
// through OpenRouter at all: routing preferences, the fallback chain, the
// reasoning object, and the sampling knobs the OpenAI schema has no home for.
func TestGatewayControlsReachTheWire(t *testing.T) {
	var mu sync.Mutex
	var got map[string]any
	server := completionServer(t, func(_ *http.Request, body map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		got = body
	})
	defer server.Close()

	ctx := context.Background()
	plugin := &openrouter.OpenRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	deny := false
	topK := 40
	minP := 0.05
	if _, err := genkit.Generate(ctx, g,
		ai.WithModel(openrouter.ModelRef("anthropic/claude-sonnet-4.5", &openrouter.ChatConfig{
			TopK:       &topK,
			MinP:       &minP,
			Models:     []string{"openai/gpt-5", "google/gemini-3-pro"},
			Transforms: []string{"middle-out"},
			Plugins:    []map[string]any{{"id": "web", "max_results": 3}},
			SessionID:  "session-42",
			Metadata:   map[string]string{"tenant": "acme"},
			Provider: &openrouter.ProviderRouting{
				Order:          []string{"anthropic", "google-vertex"},
				AllowFallbacks: &deny,
				Sort:           openrouter.ProviderSortThroughput,
				DataCollection: openrouter.DataCollectionDeny,
				MaxPrice:       &openrouter.MaxPrice{Prompt: openai.Ptr(3.0)},
			},
			Reasoning: &openrouter.ReasoningConfig{
				Effort:    openrouter.ReasoningEffortHigh,
				MaxTokens: 2048,
			},
		})),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	for field, want := range map[string]any{
		"top_k":      40.0,
		"min_p":      0.05,
		"models":     []any{"openai/gpt-5", "google/gemini-3-pro"},
		"transforms": []any{"middle-out"},
		"plugins":    []any{map[string]any{"id": "web", "max_results": 3.0}},
		"session_id": "session-42",
		"metadata":   map[string]any{"tenant": "acme"},
		"provider": map[string]any{
			"order":           []any{"anthropic", "google-vertex"},
			"allow_fallbacks": false,
			"sort":            "throughput",
			"data_collection": "deny",
			"max_price":       map[string]any{"prompt": 3.0},
		},
		"reasoning": map[string]any{"effort": "high", "max_tokens": 2048.0},
	} {
		if !reflect.DeepEqual(got[field], want) {
			t.Errorf("%s = %#v, want %#v", field, got[field], want)
		}
	}
	// A config that sets none of them must not send an empty object that
	// would enable a feature the caller never asked for.
	if _, has := got["top_a"]; has {
		t.Errorf("top_a = %#v, want it absent when unset", got["top_a"])
	}
}

// TestExtraWinsOverDeclaredRouting pins the escape hatch for the request
// shapes this config does not declare, such as the object form of the
// provider sort. An extra collides with the declared provider object and must
// replace it wholesale, so a caller is never blocked by a mapping the plugin
// has not caught up with.
func TestExtraWinsOverDeclaredRouting(t *testing.T) {
	var mu sync.Mutex
	var got map[string]any
	server := completionServer(t, func(_ *http.Request, body map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		got = body
	})
	defer server.Close()

	ctx := context.Background()
	plugin := &openrouter.OpenRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("openrouter/anthropic/claude-sonnet-4.5"),
		ai.WithConfig(map[string]any{
			"provider": map[string]any{"sort": "price"},
			"extra": map[string]any{
				"provider": map[string]any{"sort": map[string]any{"by": "price", "partition": "model"}},
				"debug":    true,
			},
		}),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	want := map[string]any{"sort": map[string]any{"by": "price", "partition": "model"}}
	if !reflect.DeepEqual(got["provider"], want) {
		t.Errorf("provider = %#v, want the extra to replace the declared object: %#v", got["provider"], want)
	}
	if got["debug"] != true {
		t.Errorf("debug = %#v, want an undeclared field to ride the extra map", got["debug"])
	}
}

// TestReasoningReachesTheCaller pins the response half of the reasoning knob.
// OpenRouter normalizes every vendor's thinking onto the reasoning field,
// which is not the reasoning_content the OpenAI-compatible providers that came
// before it use, so a config asking for reasoning would otherwise be billed
// for output the caller never sees.
func TestReasoningReachesTheCaller(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream    bool           `json:"stream"`
			Reasoning map[string]any `json:"reasoning"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if got := body.Reasoning["effort"]; got != "high" {
			t.Errorf("reasoning.effort = %v, want %q", got, "high")
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"Think "},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning":"carefully."},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"42"},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			} {
				_, _ = io.WriteString(w, "data: "+event+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","reasoning":"Think carefully.","content":"42"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &openrouter.OpenRouter{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	model := openrouter.ModelRef("openai/gpt-5", &openrouter.ChatConfig{
		Reasoning: &openrouter.ReasoningConfig{Effort: openrouter.ReasoningEffortHigh},
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Reasoning(); got != "Think carefully." {
		t.Errorf("Reasoning() = %q, want %q", got, "Think carefully.")
	}
	if got := resp.Text(); got != "42" {
		t.Errorf("Text() = %q, want %q", got, "42")
	}

	var streamed strings.Builder
	resp, err = genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt("hi"),
		ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
			for _, part := range chunk.Content {
				if part.IsReasoning() {
					streamed.WriteString(part.Text)
				}
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("streaming Generate() error = %v", err)
	}
	if got := streamed.String(); got != "Think carefully." {
		t.Errorf("streamed reasoning = %q, want %q", got, "Think carefully.")
	}
	if got := resp.Reasoning(); got != "Think carefully." {
		t.Errorf("accumulated Reasoning() = %q, want %q", got, "Think carefully.")
	}
}

// TestListActionsIsEmpty pins a deliberate choice rather than an oversight.
// OpenRouter serves hundreds of models and a descriptor carries full request
// and response schemas, so listing the catalog would put megabytes on every
// reflection poll. Models stay reachable by name.
func TestListActionsIsEmpty(t *testing.T) {
	ctx := context.Background()
	plugin := &openrouter.OpenRouter{APIKey: "test-key"}
	genkit.Init(ctx, genkit.WithPlugins(plugin))

	if got := plugin.ListActions(ctx); len(got) != 0 {
		t.Errorf("ListActions() returned %d descriptors, want none", len(got))
	}
	if plugin.ResolveAction(api.ActionTypeModel, "openai/gpt-5") == nil {
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
// fails at OpenRouter with the real reason. Constrained output is the
// exception, left unset so structured output falls back to prompt
// instructions that every model handles. Models is the correction.
func TestDynamicCapabilitiesAndOverride(t *testing.T) {
	ctx := context.Background()
	plugin := &openrouter.OpenRouter{
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

	got := supports("openai/gpt-5")
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
		ai.WithModelName("openrouter/mistralai/mistral-7b-instruct"),
		ai.WithMessages(ai.NewUserMessage(ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgo="))),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want the media refused by the overridden capabilities")
	}
	if !strings.Contains(err.Error(), "does not support media") {
		t.Errorf("error = %v, want it to name the missing media support", err)
	}
}
