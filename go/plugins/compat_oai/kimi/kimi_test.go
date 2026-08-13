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

package kimi_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
	"github.com/openai/openai-go/option"
)

func TestPluginRegistersKimiModelsAndHandlesReasoning(t *testing.T) {
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
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "kimi-k3" {
			t.Errorf("model = %q, want %q", body.Model, "kimi-k3")
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k3","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k3","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k3","choices":[{"index":0,"delta":{"content":"Final answer"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"kimi-k3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
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
			"model":"kimi-k3",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"reasoning_content":"Think carefully.",
					"content":"Final answer"
				},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &kimi.Kimi{
		APIKey: "test-key",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL + "/v1")},
	}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("kimi/kimi-k3"),
	)

	if plugin.Name() != "kimi" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "kimi")
	}
	// Tool-choice steering is K3's alone: the K2 generation rejects
	// tool_choice required as incompatible with thinking, which is on by
	// default.
	for model, wantToolChoice := range map[string]bool{
		"kimi-k3":                  true,
		"kimi-k2.5":                false,
		"kimi-k2.6":                false,
		"kimi-k2.7-code":           false,
		"kimi-k2.7-code-highspeed": false,
	} {
		m := genkit.LookupModel(g, "kimi/"+model)
		if m == nil {
			t.Errorf("LookupModel(%q) = nil", model)
			continue
		}
		supports := m.(api.Action).Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
		if got := supports["toolChoice"]; got != wantToolChoice {
			t.Errorf("%s toolChoice support = %v, want %v", model, got, wantToolChoice)
		}
	}
	for _, model := range []string{
		"kimi-k3",
		"kimi-k2.5",
		"kimi-k2.6",
		"kimi-k2.7-code",
		"kimi-k2.7-code-highspeed",
	} {
		action := genkit.LookupModel(g, "kimi/"+model).(api.Action)
		modelMetadata := action.Desc().Metadata["model"].(map[string]any)
		supports := modelMetadata["supports"].(map[string]any)
		if got := supports["media"]; got != true {
			t.Errorf("%s media support = %v, want true", model, got)
		}
	}
	k25Metadata := genkit.LookupModel(g, "kimi/kimi-k2.5").(api.Action).
		Desc().Metadata["model"].(map[string]any)
	if got := k25Metadata["stage"]; got != ai.ModelStageDeprecated {
		t.Errorf("%s stage = %v, want %q", "kimi-k2.5", got, ai.ModelStageDeprecated)
	}

	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g, ai.WithPrompt("Solve this."))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Fatalf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "Final answer" {
			t.Fatalf("Text() = %q, want %q", got, "Final answer")
		}
		if len(resp.Message.Content) != 2 ||
			!resp.Message.Content[0].IsReasoning() ||
			!resp.Message.Content[1].IsText() {
			t.Fatalf("content = %#v, want reasoning followed by text", resp.Message.Content)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var reasoning, text strings.Builder
		resp, err := genkit.Generate(
			ctx,
			g,
			ai.WithPrompt("Solve this."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				for _, part := range chunk.Content {
					switch {
					case part.IsReasoning():
						reasoning.WriteString(part.Text)
					case part.IsText():
						text.WriteString(part.Text)
					}
				}
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := reasoning.String(); got != "Think carefully." {
			t.Fatalf("streamed reasoning = %q, want %q", got, "Think carefully.")
		}
		if got := text.String(); got != "Final answer" {
			t.Fatalf("streamed text = %q, want %q", got, "Final answer")
		}
		if got := resp.Reasoning(); got != reasoning.String() {
			t.Fatalf("final reasoning = %q, want streamed %q", got, reasoning.String())
		}
		if got := resp.Text(); got != text.String() {
			t.Fatalf("final text = %q, want streamed %q", got, text.String())
		}
	})

	mu.Lock()
	defer mu.Unlock()
	if requests != 2 {
		t.Fatalf("requests = %d, want 2", requests)
	}
}

func TestPluginPreservesReasoningAndConfigAcrossToolCalls(t *testing.T) {
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		reqNum := requests
		mu.Unlock()
		var body struct {
			Messages   []map[string]any `json:"messages"`
			Model      string           `json:"model"`
			Thinking   map[string]any   `json:"thinking"`
			ToolChoice string           `json:"tool_choice"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != "kimi-k3" {
			t.Errorf("model = %q, want %q", body.Model, "kimi-k3")
		}
		if got := body.Thinking["type"]; got != "enabled" {
			t.Errorf("thinking.type = %v, want %q", got, "enabled")
		}
		if got := body.Thinking["keep"]; got != "all" {
			t.Errorf("thinking.keep = %v, want %q", got, "all")
		}
		if body.ToolChoice != "required" {
			t.Errorf("tool_choice = %q, want %q", body.ToolChoice, "required")
		}

		w.Header().Set("Content-Type", "application/json")
		if reqNum == 1 {
			_, _ = io.WriteString(w, `{
				"id":"chatcmpl-tool-1",
				"object":"chat.completion",
				"created":1,
				"model":"kimi-k3",
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
		} else {
			if got := assistant["reasoning_content"]; got != "I should call the lookup tool." {
				t.Errorf("assistant reasoning_content = %v, want preserved reasoning", got)
			}
			if got := assistant["content"]; got == "I should call the lookup tool." {
				t.Errorf("assistant content incorrectly contains reasoning: %v", got)
			}
		}

		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-tool-2",
			"object":"chat.completion",
			"created":1,
			"model":"kimi-k3",
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
	plugin := &kimi.Kimi{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("kimi/kimi-k3"),
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
		ai.WithToolChoice(ai.ToolChoiceRequired),
		ai.WithConfig(map[string]any{
			"thinking": map[string]any{
				"type": "enabled",
				"keep": "all",
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
			"id":"c1","object":"chat.completion","created":1,"model":"kimi-k3",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &kimi.Kimi{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("kimi/kimi-k3"),
		ai.WithPrompt("hi"),
		ai.WithConfig(map[string]any{
			"thinking": map[string]any{"type": "disabled"},
			"extra": map[string]any{
				"thinking":  map[string]any{"type": "enabled"},
				"kimi_beta": "on",
			},
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if got := body["kimi_beta"]; got != "on" {
		t.Errorf("kimi_beta = %v, want the undeclared field on the wire", got)
	}
	thinking, _ := body["thinking"].(map[string]any)
	if got := thinking["type"]; got != "enabled" {
		t.Errorf("thinking.type = %v, want the caller's extra winning over the config's own thinking", got)
	}
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("KIMI_API_KEY", "")
	t.Setenv("MOONSHOT_API_KEY", "")

	defer func() {
		got := recover()
		if got != "kimi plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&kimi.Kimi{}).Init(context.Background())
}

// TestModelRefAndConfigSchema pins the call-site surface: the ref carries the
// prefixed name and the typed config, and the registered models advertise the
// camelCase config contract including the Kimi-specific fields.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &kimi.ChatConfig{ReasoningEffort: "high"}
	for _, name := range []string{"kimi-k3", "kimi/kimi-k3"} {
		ref := kimi.ModelRef(name, cfg)
		if want := "kimi/kimi-k3"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}

	plugin := &kimi.Kimi{APIKey: "test-key"}
	g := genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	m := genkit.LookupModel(g, "kimi/kimi-k3")
	if m == nil {
		t.Fatal("kimi-k3 not registered by Init")
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
	for _, key := range []string{"maxOutputTokens", "stopSequences", "logProbs", "topLogProbs", "thinking", "reasoningEffort", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}
	// Moonshot documents these for the legacy moonshot-v1 family only, so the
	// K-series models this plugin serves must not advertise them.
	for _, key := range []string{"temperature", "topP", "frequencyPenalty", "presencePenalty"} {
		if props[key] != nil {
			t.Errorf("config schema advertises %q, which the Kimi K-series does not take", key)
		}
	}

	// The constraints Moonshot documents ride on the schema, where the
	// framework enforces them.
	for field, want := range map[string]map[string]any{
		"maxOutputTokens": {"minimum": 1.0},
		"stopSequences":   {"maxItems": 5.0},
		"topLogProbs":     {"minimum": 0.0, "maximum": 20.0},
		"reasoningEffort": {"enum": []any{
			string(kimi.ReasoningEffortLow), string(kimi.ReasoningEffortHigh),
			string(kimi.ReasoningEffortMax)}},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
	// The per-entry limit lands on the items schema, where each sequence is
	// checked against Moonshot's 32 maximum.
	stop, _ := props["stopSequences"].(map[string]any)
	stopItems, _ := stop["items"].(map[string]any)
	if got := stopItems["maxLength"]; got != 32.0 {
		t.Errorf("stopSequences items maxLength = %v, want 32 to match Moonshot's per-entry limit", got)
	}
	thinking, _ := props["thinking"].(map[string]any)
	thinkingProps, _ := thinking["properties"].(map[string]any)
	thinkingType, _ := thinkingProps["type"].(map[string]any)
	if got, want := thinkingType["enum"], []any{
		string(kimi.ThinkingTypeEnabled), string(kimi.ThinkingTypeDisabled)}; !reflect.DeepEqual(got, want) {
		t.Errorf("thinking.type enum = %#v, want %#v", got, want)
	}
	// keep carries no enum: Moonshot documents a single value today, and a
	// list of one would reject whatever it adds next.
	keep, _ := thinkingProps["keep"].(map[string]any)
	if got, has := keep["enum"]; has {
		t.Errorf("thinking.keep enum = %v, want none since the set is Moonshot's to grow", got)
	}
}

// TestToolChoiceRejectedOnK2Generation pins the capability split: Moonshot
// rejects tool_choice required on the K2 generation as "incompatible with
// thinking enabled", and thinking is on by default, so those models advertise
// no tool-choice steering and Genkit fails the request before the wire.
func TestToolChoiceRejectedOnK2Generation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request reached the server, want local rejection")
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &kimi.Kimi{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	lookup := genkit.DefineTool(g, "lookup", "Looks up a value.",
		func(_ *ai.ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return "result for " + input.Value, nil
		})

	for _, model := range []string{"kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code", "kimi-k2.7-code-highspeed"} {
		if _, err := genkit.Generate(ctx, g,
			ai.WithModelName("kimi/"+model),
			ai.WithPrompt("Use the lookup tool."),
			ai.WithTools(lookup),
			ai.WithToolChoice(ai.ToolChoiceRequired),
		); err == nil {
			t.Errorf("%s Generate() error = nil, want the tool choice rejected", model)
		}
	}
}

// TestStopSequenceLengthRejected pins the per-entry limit end to end:
// Moonshot rejects a stop sequence over 32 ("stop sequence must not be longer
// than 32"), so the schema catches one before the wire.
func TestStopSequenceLengthRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request reached the server, want boundary validation to reject the config")
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &kimi.Kimi{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("kimi/kimi-k3"),
		ai.WithPrompt("Say OK."),
		ai.WithConfig(map[string]any{"stopSequences": []string{strings.Repeat("x", 33)}}),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want the 33-character stop sequence rejected")
	}
	if !strings.Contains(err.Error(), "stopSequences") {
		t.Errorf("error = %v, want it to name stopSequences", err)
	}
}
