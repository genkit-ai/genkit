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

package nvidia_test

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
	"github.com/firebase/genkit/go/plugins/compat_oai/nvidia"
	"github.com/openai/openai-go/option"
)

const testModel = "nvidia/nemotron-3.5-super-vl-120b-a12b"

// okResponse is a minimal non-streaming chat completion.
const okResponse = `{
	"id":"c1","object":"chat.completion","created":1,"model":"` + testModel + `",
	"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
}`

func TestPluginRegistersNVIDIAModelsAndHandlesReasoning(t *testing.T) {
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
		// The full publisher/model ID reaches NIM: the plugin's own
		// "nvidia/" prefix is stripped, the publisher's is kept.
		if body.Model != testModel {
			t.Errorf("model = %q, want %q", body.Model, testModel)
		}

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, event := range []string{
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"` + testModel + `","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Think "},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"` + testModel + `","choices":[{"index":0,"delta":{"reasoning_content":"carefully."},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"` + testModel + `","choices":[{"index":0,"delta":{"content":"Final answer"},"finish_reason":null}]}`,
				`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"` + testModel + `","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
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
			"model":"`+testModel+`",
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
	plugin := &nvidia.NVIDIA{
		APIKey: "test-key",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL + "/v1")},
	}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("nvidia/"+testModel),
	)

	if plugin.Name() != "nvidia" {
		t.Fatalf("Name() = %q, want %q", plugin.Name(), "nvidia")
	}
	// Image input is claimed for the vision-capable models only:
	// Lightning rejects multimodal content.
	for model, wantMedia := range map[string]bool{
		"nvidia/nemotron-3.5-super-vl-120b-a12b": true,
		"nvidia/nemotron-3.5-lightning-30b-a3b":  false,
		"nvidia/nemotron-3.5-bolt-e4b":           true,
	} {
		m := genkit.LookupModel(g, "nvidia/"+model)
		if m == nil {
			t.Errorf("LookupModel(%q) = nil", model)
			continue
		}
		supports := m.(api.Action).Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
		if got := supports["media"]; got != wantMedia {
			t.Errorf("%s media support = %v, want %v", model, got, wantMedia)
		}
		if got := supports["toolChoice"]; got != true {
			t.Errorf("%s toolChoice support = %v, want true", model, got)
		}
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
			Messages    []map[string]any `json:"messages"`
			Model       string           `json:"model"`
			MaxTokens   int              `json:"max_tokens"`
			Temperature float64          `json:"temperature"`
			ToolChoice  string           `json:"tool_choice"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if body.Model != testModel {
			t.Errorf("model = %q, want %q", body.Model, testModel)
		}
		if body.MaxTokens != 1024 {
			t.Errorf("max_tokens = %d, want %d", body.MaxTokens, 1024)
		}
		if body.Temperature != 0.2 {
			t.Errorf("temperature = %v, want %v", body.Temperature, 0.2)
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
				"model":"`+testModel+`",
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
			"model":"`+testModel+`",
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
	plugin := &nvidia.NVIDIA{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("nvidia/"+testModel),
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
			"maxOutputTokens": 1024,
			"temperature":     0.2,
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
// request, and a collision with a field the config itself writes
// (max_tokens) resolves toward the caller's extra.
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
		_, _ = io.WriteString(w, okResponse)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &nvidia.NVIDIA{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("nvidia/"+testModel),
		ai.WithPrompt("hi"),
		ai.WithConfig(map[string]any{
			"maxOutputTokens": 16,
			"extra": map[string]any{
				"max_tokens": 32,
				"nim_beta":   "on",
			},
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if got := body["nim_beta"]; got != "on" {
		t.Errorf("nim_beta = %v, want the undeclared field on the wire", got)
	}
	if got := body["max_tokens"]; got != 32.0 {
		t.Errorf("max_tokens = %v, want the caller's extra winning over the config's own maxOutputTokens", got)
	}
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("NVIDIA_API_KEY", "")

	defer func() {
		got := recover()
		if got != "nvidia plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&nvidia.NVIDIA{}).Init(context.Background())
}

// TestPluginReadsKeyAndBaseURLFromEnv pins the environment defaults: with no
// APIKey or Opts, NVIDIA_API_KEY authenticates and NVIDIA_BASE_URL points the
// plugin at a self-hosted NIM.
func TestPluginReadsKeyAndBaseURLFromEnv(t *testing.T) {
	var mu sync.Mutex
	var path, auth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		path, auth = r.URL.Path, r.Header.Get("Authorization")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, okResponse)
	}))
	defer server.Close()
	t.Setenv("NVIDIA_API_KEY", "env-key")
	t.Setenv("NVIDIA_BASE_URL", server.URL+"/v1")

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&nvidia.NVIDIA{}))
	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("nvidia/"+testModel),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if path != "/v1/chat/completions" {
		t.Errorf("path = %q, want %q", path, "/v1/chat/completions")
	}
	if auth != "Bearer env-key" {
		t.Errorf("Authorization = %q, want the NVIDIA_API_KEY bearer token", auth)
	}
}

// TestOptsBaseURLWinsOverEnv pins the precedence the Opts doc promises: a
// caller's option.WithBaseURL overrides NVIDIA_BASE_URL.
func TestOptsBaseURLWinsOverEnv(t *testing.T) {
	wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request reached NVIDIA_BASE_URL, want the Opts base URL")
	}))
	defer wrong.Close()
	var mu sync.Mutex
	var hits int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		hits++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, okResponse)
	}))
	defer server.Close()
	t.Setenv("NVIDIA_BASE_URL", wrong.URL+"/v1")

	ctx := context.Background()
	plugin := &nvidia.NVIDIA{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("nvidia/"+testModel),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}
}

// recordingTransport answers every request locally and records its URL, so
// the default endpoint can be checked without reaching the network.
type recordingTransport struct {
	mu   sync.Mutex
	urls []string
}

func (rt *recordingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	rt.mu.Lock()
	rt.urls = append(rt.urls, r.URL.String())
	rt.mu.Unlock()
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(okResponse)),
		Request:    r,
	}, nil
}

// TestDefaultBaseURL pins where requests go with no override: NVIDIA's hosted
// NIM API.
func TestDefaultBaseURL(t *testing.T) {
	t.Setenv("NVIDIA_BASE_URL", "")
	rt := &recordingTransport{}

	ctx := context.Background()
	plugin := &nvidia.NVIDIA{APIKey: "test-key", Opts: []option.RequestOption{
		option.WithHTTPClient(&http.Client{Transport: rt}),
	}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))
	if _, err := genkit.Generate(ctx, g,
		ai.WithModelName("nvidia/"+testModel),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	rt.mu.Lock()
	defer rt.mu.Unlock()
	want := "https://integrate.api.nvidia.com/v1/chat/completions"
	if len(rt.urls) != 1 || rt.urls[0] != want {
		t.Errorf("request URLs = %v, want [%s]", rt.urls, want)
	}
}

// TestDynamicModelResolves pins the open catalog: a NIM model the plugin does
// not curate resolves on demand, its full publisher/model ID reaches the
// wire, and it claims no image input until a Models entry says so.
func TestDynamicModelResolves(t *testing.T) {
	var mu sync.Mutex
	var model string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		mu.Lock()
		model = body.Model
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, okResponse)
	}))
	defer server.Close()

	const uncurated = "nvidia/nemotron-3-ultra-550b-a55b"
	ctx := context.Background()
	plugin := &nvidia.NVIDIA{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	if _, err := genkit.Generate(ctx, g,
		ai.WithModel(nvidia.ModelRef(uncurated, nil)),
		ai.WithPrompt("hi"),
	); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	mu.Lock()
	if model != uncurated {
		t.Errorf("model = %q, want %q", model, uncurated)
	}
	mu.Unlock()

	supports := genkit.LookupModel(g, "nvidia/"+uncurated).(api.Action).
		Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)
	if got := supports["media"]; got != false {
		t.Errorf("uncurated media support = %v, want false", got)
	}

	// A Models entry widens one model, under either key form.
	for _, key := range []string{uncurated, "nvidia/" + uncurated} {
		p := &nvidia.NVIDIA{APIKey: "test-key", Models: map[string]ai.ModelOptions{
			key: {Supports: &compat_oai.Multimodal},
		}}
		genkit.Init(ctx, genkit.WithPlugins(p))
		action := p.ResolveAction(api.ActionTypeModel, uncurated)
		got := action.Desc().Metadata["model"].(map[string]any)["supports"].(map[string]any)["media"]
		if got != true {
			t.Errorf("Models[%q] media support = %v, want the override's true", key, got)
		}
	}
}

// TestModelRefAndConfigSchema pins the call-site surface: the ref carries the
// prefixed name and the typed config, and the registered models advertise the
// camelCase config contract.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &nvidia.ChatConfig{MaxOutputTokens: 1024}
	for _, name := range []string{testModel, "nvidia/" + testModel} {
		ref := nvidia.ModelRef(name, cfg)
		if want := "nvidia/" + testModel; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}
	// Another publisher's model keeps its own prefix.
	if got, want := nvidia.ModelRef("google/gemma-3-12b-it", nil).Name(), "nvidia/google/gemma-3-12b-it"; got != want {
		t.Errorf("ModelRef(google/gemma-3-12b-it).Name() = %q, want %q", got, want)
	}

	plugin := &nvidia.NVIDIA{APIKey: "test-key"}
	g := genkit.Init(context.Background(), genkit.WithPlugins(plugin))

	m := genkit.LookupModel(g, "nvidia/"+testModel)
	if m == nil {
		t.Fatalf("%s not registered by Init", testModel)
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
	for _, key := range []string{"temperature", "topP", "maxOutputTokens", "stopSequences", "frequencyPenalty", "presencePenalty", "seed", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}
	// No provider-specific reasoning controls: NIM documents none that hold
	// across the models it serves.
	for _, key := range []string{"thinking", "reasoningEffort"} {
		if props[key] != nil {
			t.Errorf("config schema advertises %q, which NIM does not document", key)
		}
	}

	for field, want := range map[string]map[string]any{
		"maxOutputTokens":  {"minimum": 1.0},
		"topP":             {"minimum": 0.0, "maximum": 1.0},
		"frequencyPenalty": {"minimum": -2.0, "maximum": 2.0},
		"presencePenalty":  {"minimum": -2.0, "maximum": 2.0},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
}

// TestTopPRangeRejected pins a documented limit end to end: the schema
// catches a topP above 1 before the wire.
func TestTopPRangeRejected(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request reached the server, want boundary validation to reject the config")
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := &nvidia.NVIDIA{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL + "/v1")}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	_, err := genkit.Generate(ctx, g,
		ai.WithModelName("nvidia/"+testModel),
		ai.WithPrompt("Say OK."),
		ai.WithConfig(map[string]any{"topP": 1.5}),
	)
	if err == nil {
		t.Fatal("Generate() error = nil, want topP 1.5 rejected")
	}
	if !strings.Contains(err.Error(), "topP") {
		t.Errorf("error = %v, want it to name topP", err)
	}
}
