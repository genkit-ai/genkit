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

package mistral_test

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
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/compat_oai/mistral"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// chatRequestFields are the top-level fields of Mistral's
// ChatCompletionRequest schema, which is closed: a request carrying any other
// field is rejected. The fake server enforces the same, which is what proves
// the plugin never sends OpenAI-only fields such as stream_options,
// max_completion_tokens, seed or user.
var chatRequestFields = []string{
	"model", "messages", "temperature", "top_p", "max_tokens", "stream", "stop",
	"random_seed", "metadata", "response_format", "tools", "tool_choice",
	"presence_penalty", "frequency_penalty", "min_tokens", "repetition_penalty",
	"top_k", "logprobs", "top_logprobs", "prompt_logprobs", "top_prompt_logprobs",
	"n", "prediction", "parallel_tool_calls", "reasoning_effort", "prompt_mode",
	"guardrails", "prompt_cache_key", "service_tier", "safe_prompt",
}

// messageFields are the fields of each of Mistral's message schemas, by role.
// The assistant message has no reasoning_content.
var messageFields = map[string][]string{
	"system":    {"role", "content"},
	"user":      {"role", "content"},
	"assistant": {"role", "content", "tool_calls", "prefix"},
	"tool":      {"role", "content", "tool_call_id", "name"},
}

// embeddingRequestFields are the fields of Mistral's EmbeddingRequest schema.
var embeddingRequestFields = []string{"model", "input", "metadata", "output_dimension", "output_dtype", "encoding_format"}

// checkClosed fails for every key of body outside allowed, as Mistral does.
func checkClosed(t *testing.T, what string, body map[string]any, allowed []string) {
	t.Helper()
	for key := range body {
		if !slices.Contains(allowed, key) {
			t.Errorf("%s carries %q, which Mistral's closed schema rejects", what, key)
		}
	}
}

// decodeChatRequest decodes a chat completion request and checks it against
// Mistral's closed schemas.
func decodeChatRequest(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Errorf("decode request: %v", err)
		return nil
	}
	checkClosed(t, "request", body, chatRequestFields)
	messages, _ := body["messages"].([]any)
	for _, m := range messages {
		msg, _ := m.(map[string]any)
		role, _ := msg["role"].(string)
		allowed, known := messageFields[role]
		if !known {
			t.Errorf("message has role %q, which Mistral does not take", role)
			continue
		}
		checkClosed(t, role+" message", msg, allowed)
	}
	return body
}

func writeSSE(w http.ResponseWriter, events ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, event := range events {
		_, _ = io.WriteString(w, "data: "+event+"\n\n")
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
}

func newPlugin(server *httptest.Server) *mistral.Mistral {
	return &mistral.Mistral{APIKey: "test-key", Opts: []option.RequestOption{option.WithBaseURL(server.URL)}}
}

func TestPluginRequiresAPIKey(t *testing.T) {
	t.Setenv("MISTRAL_API_KEY", "")
	// An OPENAI_API_KEY must never be picked up as a fallback: sending it to
	// Mistral would silently authenticate with the wrong provider's key.
	t.Setenv("OPENAI_API_KEY", "sk-should-not-be-used")

	defer func() {
		got := recover()
		if got != "mistral plugin initialization failed: apiKey is required" {
			t.Fatalf("panic = %v, want missing API key error", got)
		}
	}()

	(&mistral.Mistral{}).Init(context.Background())
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
			"id":"c1","object":"chat.completion","created":1,"model":"mistral-small-latest",
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
	t.Setenv("MISTRAL_API_KEY", "env-key")
	t.Setenv("MISTRAL_BASE_URL", wrong.URL)

	ctx := context.Background()
	plugin := &mistral.Mistral{APIKey: "struct-key", Opts: []option.RequestOption{option.WithBaseURL(right.URL)}}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin), genkit.WithDefaultModel("mistral/mistral-small-latest"))

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

func TestPluginRegistersModelsAndEmbedders(t *testing.T) {
	g := genkit.Init(context.Background(), genkit.WithPlugins(&mistral.Mistral{APIKey: "test-key"}))

	for modelID, media := range map[string]bool{
		"mistral-large-latest":  true,
		"mistral-large-4":       true,
		"mistral-medium-latest": true,
		"mistral-small-latest":  true,
		"ministral-14b-latest":  true,
		"ministral-8b-latest":   true,
		"ministral-3b-latest":   true,
		"codestral-latest":      false,
	} {
		model := genkit.LookupModel(g, "mistral/"+modelID)
		if model == nil {
			t.Errorf("LookupModel(%q) = nil", modelID)
			continue
		}
		meta := model.(api.Action).Desc().Metadata["model"].(map[string]any)
		supports := meta["supports"].(map[string]any)
		for field, want := range map[string]bool{"media": media, "tools": true, "toolChoice": true, "multiturn": true, "systemRole": true} {
			if got := supports[field]; got != want {
				t.Errorf("%s %s support = %v, want %v", modelID, field, got, want)
			}
		}
		if got := supports["constrained"]; got != ai.ConstrainedSupportNoTools {
			t.Errorf("%s constrained = %v, want %q", modelID, got, ai.ConstrainedSupportNoTools)
		}
		output, _ := supports["output"].([]string)
		if !slices.Equal(output, []string{"text", "json"}) {
			t.Errorf("%s output = %v, want [text json]", modelID, output)
		}
		if label, _ := meta["label"].(string); label == "" {
			t.Errorf("%s has no label", modelID)
		}
		// Mistral deprecated the models these two aliases name on
		// 2026-10-31; the rest are generally available.
		var wantStage ai.ModelStage
		if modelID == "mistral-large-latest" || modelID == "mistral-medium-latest" {
			wantStage = ai.ModelStageDeprecated
		}
		if got := meta["stage"]; got != wantStage {
			t.Errorf("%s stage = %v, want %q", modelID, got, wantStage)
		}
	}

	for id, dims := range map[string]int{"mistral-embed": 1024, "codestral-embed": 1536} {
		e := genkit.LookupEmbedder(g, "mistral/"+id)
		if e == nil {
			t.Errorf("LookupEmbedder(%q) = nil", id)
			continue
		}
		info, _ := e.(api.Action).Desc().Metadata["info"].(map[string]any)
		if got := info["dimensions"]; got != dims {
			t.Errorf("%s dimensions = %v, want %d", id, got, dims)
		}
	}
}

// TestModelRefConfigReachesTheWire pins the typed path end to end: a ModelRef
// carrying a typed ChatConfig passes the framework's schema validation, and
// every field lands on the name Mistral's closed schema declares. The fields
// OpenAI names differently are the point: max_tokens rather than
// max_completion_tokens, random_seed rather than seed, and no user.
func TestModelRefConfigReachesTheWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeChatRequest(t, r)
		for key, want := range map[string]any{
			"model":               "mistral-small-2603",
			"temperature":         0.3,
			"top_p":               0.9,
			"max_tokens":          512.0,
			"stop":                []any{"END"},
			"frequency_penalty":   0.1,
			"presence_penalty":    0.2,
			"random_seed":         7.0,
			"reasoning_effort":    "high",
			"parallel_tool_calls": false,
			"safe_prompt":         true,
			"prompt_cache_key":    "tenant-42",
			"service_tier":        "standard_only",
			"min_tokens":          2.0,
		} {
			if got := body[key]; !reflect.DeepEqual(got, want) {
				t.Errorf("%s = %#v, want %#v", key, got, want)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"mistral-small-2603",
			"choices":[{"index":0,"message":{"role":"assistant","content":"typed config works"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(newPlugin(server)))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(mistral.ModelRef("mistral-small-latest", &mistral.ChatConfig{
			RequestConfig: compat_oai.RequestConfig{
				Version: "mistral-small-2603",
				// min_tokens is a field the config does not declare, riding
				// the inherited passthrough.
				Extra: map[string]any{"min_tokens": 2},
			},
			Temperature:       openai.Ptr(0.3),
			TopP:              openai.Ptr(0.9),
			MaxOutputTokens:   512,
			StopSequences:     []string{"END"},
			FrequencyPenalty:  openai.Ptr(0.1),
			PresencePenalty:   openai.Ptr(0.2),
			Seed:              openai.Ptr(7),
			ReasoningEffort:   mistral.ReasoningEffortHigh,
			ParallelToolCalls: openai.Ptr(false),
			SafePrompt:        openai.Ptr(true),
			PromptCacheKey:    "tenant-42",
			ServiceTier:       mistral.ServiceTierStandardOnly,
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

// TestStreamingOmitsStreamOptions pins the streaming request: Mistral rejects
// stream_options, which every other provider in the plugin is sent, and
// reports usage on the final chunk unasked, which is still read.
func TestStreamingOmitsStreamOptions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeChatRequest(t, r)
		if body["stream"] != true {
			t.Errorf("stream = %v, want true", body["stream"])
		}
		if _, ok := body["stream_options"]; ok {
			t.Errorf("stream_options = %v, want it left out", body["stream_options"])
		}
		writeSSE(w,
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(newPlugin(server)), genkit.WithDefaultModel("mistral/mistral-small-latest"))

	var streamed strings.Builder
	resp, err := genkit.Generate(ctx, g,
		ai.WithPrompt("hi"),
		ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
			streamed.WriteString(chunk.Text())
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := streamed.String(); got != "Hello there" {
		t.Errorf("streamed text = %q, want %q", got, "Hello there")
	}
	if got := resp.Text(); got != "Hello there" {
		t.Errorf("Text() = %q, want %q", got, "Hello there")
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 2 || resp.Usage.TotalTokens != 7 {
		t.Errorf("Usage = %+v, want 5 in, 2 out, 7 total from the final chunk", resp.Usage)
	}
}

// TestThinkingChunks pins how reasoning crosses the wire: Mistral answers with
// a content array whose thinking chunk holds the reasoning, which the plugin
// surfaces as a reasoning part rather than as raw JSON in the text, on both
// the complete and the streamed path.
func TestThinkingChunks(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeChatRequest(t, r)
		if body["stream"] == true {
			writeSSE(w,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"mistral-large-4","choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"Think "}]}]},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"mistral-large-4","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"carefully."}]}]},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"mistral-large-4","choices":[{"index":0,"delta":{"content":[{"type":"text","text":"Streamed "}]},"finish_reason":null}]}`,
				`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"mistral-large-4","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`,
			)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"mistral-large-4",
			"choices":[{"index":0,"message":{"role":"assistant","content":[
				{"type":"thinking","thinking":[{"type":"text","text":"Think carefully."}]},
				{"type":"text","text":"Complete answer"}
			]},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(newPlugin(server)))
	model := ai.WithModel(mistral.ModelRef("mistral-large-4", &mistral.ChatConfig{ReasoningEffort: mistral.ReasoningEffortHigh}))

	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g, model, ai.WithPrompt("Say hi."))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Errorf("Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "Complete answer" {
			t.Errorf("Text() = %q, want %q", got, "Complete answer")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var reasoning, text strings.Builder
		resp, err := genkit.Generate(ctx, g, model,
			ai.WithPrompt("Say hi, streamed."),
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
			t.Errorf("streamed reasoning = %q, want %q", got, "Think carefully.")
		}
		if got := text.String(); got != "Streamed answer" {
			t.Errorf("streamed text = %q, want %q", got, "Streamed answer")
		}
		if got := resp.Reasoning(); got != "Think carefully." {
			t.Errorf("final Reasoning() = %q, want %q", got, "Think carefully.")
		}
		if got := resp.Text(); got != "Streamed answer" {
			t.Errorf("final Text() = %q, want %q", got, "Streamed answer")
		}
	})
}

// TestReasoningGoesBackAsThinkingChunk pins the turn after a reasoning tool
// call: the model turn in the history carries its reasoning as the thinking
// chunk Mistral reads, since its assistant message schema has no
// reasoning_content field and rejects one.
func TestReasoningGoesBackAsThinkingChunk(t *testing.T) {
	var mu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		reqNum := requests
		mu.Unlock()
		body := decodeChatRequest(t, r)

		w.Header().Set("Content-Type", "application/json")
		if reqNum == 1 {
			_, _ = io.WriteString(w, `{
				"id":"c-tool-1","object":"chat.completion","created":1,"model":"mistral-large-4",
				"choices":[{"index":0,"message":{
					"role":"assistant",
					"content":[{"type":"thinking","thinking":[{"type":"text","text":"I should call the lookup tool."}],"signature":"sig-1"}],
					"tool_calls":[{"id":"call1","type":"function","function":{"name":"lookup","arguments":"{\"value\":\"question\"}"}}]
				},"finish_reason":"tool_calls"}]
			}`)
			return
		}

		var assistant map[string]any
		messages, _ := body["messages"].([]any)
		for _, m := range messages {
			if msg, _ := m.(map[string]any); msg["role"] == "assistant" {
				assistant = msg
			}
		}
		want := []any{map[string]any{
			"type":      "thinking",
			"thinking":  []any{map[string]any{"type": "text", "text": "I should call the lookup tool."}},
			"signature": "sig-1",
		}}
		if assistant == nil {
			t.Error("second request has no assistant message")
		} else {
			if got := assistant["content"]; !reflect.DeepEqual(got, want) {
				t.Errorf("assistant content = %#v, want the reasoning as a thinking chunk %#v", got, want)
			}
			if calls, _ := assistant["tool_calls"].([]any); len(calls) != 1 {
				t.Errorf("assistant tool_calls = %#v, want the one call", assistant["tool_calls"])
			}
		}

		_, _ = io.WriteString(w, `{
			"id":"c-tool-2","object":"chat.completion","created":1,"model":"mistral-large-4",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Tool loop complete"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(newPlugin(server)), genkit.WithDefaultModel("mistral/mistral-large-4"))

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

// TestStructuredOutputRequest pins how a constrained JSON request reaches
// Mistral. Without tools the output schema rides as a json_schema
// response_format. With tools no response_format is sent at all, since
// Mistral is not confirmed to honor structured output and tools together:
// the schema reaches the model as prompt instructions instead.
func TestStructuredOutputRequest(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := decodeChatRequest(t, r)
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"mistral-small-latest",
			"choices":[{"index":0,"message":{"role":"assistant","content":"{\"city\":\"Paris\"}"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(newPlugin(server)), genkit.WithDefaultModel("mistral/mistral-small-latest"))
	lookup := genkit.DefineTool(g, "lookup", "Looks up a value.",
		func(_ *ai.ToolContext, input struct {
			Value string `json:"value"`
		}) (string, error) {
			return input.Value, nil
		},
	)

	type answer struct {
		City string `json:"city"`
	}
	for name, opts := range map[string][]ai.GenerateOption{
		"without tools": {ai.WithPrompt("Where?")},
		"with tools":    {ai.WithPrompt("Where?"), ai.WithTools(lookup), ai.WithToolChoice(ai.ToolChoiceAuto)},
	} {
		mu.Lock()
		bodies = nil
		mu.Unlock()
		got, _, err := genkit.GenerateData[answer](ctx, g, opts...)
		if err != nil {
			t.Fatalf("%s: GenerateData() error = %v", name, err)
		}
		if got.City != "Paris" {
			t.Errorf("%s: City = %q, want %q", name, got.City, "Paris")
		}
		mu.Lock()
		if len(bodies) != 1 {
			t.Fatalf("%s: requests = %d, want 1", name, len(bodies))
		}
		body := bodies[0]
		mu.Unlock()
		format, sent := body["response_format"].(map[string]any)
		switch name {
		case "without tools":
			if format["type"] != "json_schema" {
				t.Errorf("%s: response_format = %#v, want a json_schema format", name, body["response_format"])
			}
		case "with tools":
			if sent {
				t.Errorf("%s: response_format = %#v, want none beside tools", name, format)
			}
			if tools, _ := body["tools"].([]any); len(tools) != 1 {
				t.Errorf("%s: tools = %#v, want the one tool", name, body["tools"])
			}
		}
	}
}

// TestModelLengthFinishReason pins Mistral's model_length finish reason, a
// generation cut off by the model's context length, as a length finish.
func TestModelLengthFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"mistral-small-latest",
			"choices":[{"index":0,"message":{"role":"assistant","content":"cut"},"finish_reason":"model_length"}]
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(newPlugin(server)), genkit.WithDefaultModel("mistral/mistral-small-latest"))
	resp, err := genkit.Generate(ctx, g, ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if resp.FinishReason != ai.FinishReasonLength {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishReasonLength)
	}
}

// TestEmbedderRequest pins the embeddings request: the output size goes out
// as Mistral's output_dimension rather than OpenAI's dimensions, the
// passthrough merges beside it rather than replacing it, and nothing outside
// Mistral's closed schema is sent.
func TestEmbedderRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/embeddings")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		checkClosed(t, "embeddings request", body, embeddingRequestFields)
		for key, want := range map[string]any{
			"model":            "codestral-embed",
			"input":            []any{"hello", "world"},
			"output_dimension": 3.0,
			"output_dtype":     "float",
			"encoding_format":  "float",
		} {
			if got := body[key]; !reflect.DeepEqual(got, want) {
				t.Errorf("%s = %#v, want %#v", key, got, want)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"e1","object":"list","model":"codestral-embed",
			"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},{"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]}],
			"usage":{"prompt_tokens":2,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(newPlugin(server)))
	res, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(mistral.EmbedderRef("codestral-embed", &mistral.EmbedConfig{
			Dimensions: 3,
			Extra:      map[string]any{"output_dtype": "float"},
		})),
		ai.WithTextDocs("hello", "world"),
	)
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(res.Embeddings) != 2 || !slices.Equal(res.Embeddings[1].Embedding, []float32{0.4, 0.5, 0.6}) {
		t.Errorf("Embeddings = %+v, want the two vectors the server sent", res.Embeddings)
	}
}

// TestModelRefAndConfigSchema pins the call-site surface: the refs carry the
// prefixed name and the typed config, and the registered actions advertise
// the camelCase config contract.
func TestModelRefAndConfigSchema(t *testing.T) {
	cfg := &mistral.ChatConfig{ReasoningEffort: mistral.ReasoningEffortHigh}
	for _, name := range []string{"mistral-small-latest", "mistral/mistral-small-latest"} {
		ref := mistral.ModelRef(name, cfg)
		if want := "mistral/mistral-small-latest"; ref.Name() != want {
			t.Errorf("ModelRef(%q).Name() = %q, want %q", name, ref.Name(), want)
		}
		if ref.Config() != cfg {
			t.Errorf("ModelRef(%q).Config() = %v, want the config it was built with", name, ref.Config())
		}
	}
	embedCfg := &mistral.EmbedConfig{Dimensions: 256}
	if ref := mistral.EmbedderRef("mistral/codestral-embed", embedCfg); ref.Name() != "mistral/codestral-embed" || ref.Config() != embedCfg {
		t.Errorf("EmbedderRef() = %q, %v, want the prefixed name and the config", ref.Name(), ref.Config())
	}

	g := genkit.Init(context.Background(), genkit.WithPlugins(&mistral.Mistral{APIKey: "test-key"}))

	m := genkit.LookupModel(g, "mistral/mistral-small-latest")
	if m == nil {
		t.Fatal("mistral-small-latest not registered by Init")
	}
	model := m.(api.Action).Desc().Metadata["model"].(map[string]any)
	schema, _ := model["customOptions"].(map[string]any)
	props, _ := schema["properties"].(map[string]any)
	for _, key := range []string{"temperature", "topP", "maxOutputTokens", "stopSequences", "frequencyPenalty",
		"presencePenalty", "seed", "reasoningEffort", "parallelToolCalls", "safePrompt", "promptCacheKey",
		"serviceTier", "version", "extra"} {
		if props[key] == nil {
			t.Errorf("config schema is missing the %q property", key)
		}
	}
	for _, key := range []string{"user", "n", "topK"} {
		if props[key] != nil {
			t.Errorf("config schema advertises %q, which Mistral does not take", key)
		}
	}
	effort, _ := props["reasoningEffort"].(map[string]any)
	if got, want := effort["enum"], []any{"none", "low", "medium", "high", "max"}; !reflect.DeepEqual(got, want) {
		t.Errorf("reasoningEffort enum = %#v, want %#v", got, want)
	}
	// The ranges Mistral's OpenAPI spec declares ride on the schema, where
	// the framework enforces them.
	for field, want := range map[string]map[string]any{
		"temperature":      {"minimum": 0.0, "maximum": 1.5},
		"topP":             {"exclusiveMinimum": 0.0, "maximum": 1.0},
		"frequencyPenalty": {"minimum": -2.0, "maximum": 2.0},
		"presencePenalty":  {"minimum": -2.0, "maximum": 2.0},
		"seed":             {"minimum": 0.0},
		"maxOutputTokens":  {"minimum": 1.0},
	} {
		prop, _ := props[field].(map[string]any)
		for key, value := range want {
			if got := prop[key]; !reflect.DeepEqual(got, value) {
				t.Errorf("%s %s = %#v, want %#v", field, key, got, value)
			}
		}
	}
	tier, _ := props["serviceTier"].(map[string]any)
	if got, want := tier["enum"], []any{"auto", "standard_only"}; !reflect.DeepEqual(got, want) {
		t.Errorf("serviceTier enum = %#v, want %#v", got, want)
	}

	e := genkit.LookupEmbedder(g, "mistral/codestral-embed")
	if e == nil {
		t.Fatal("codestral-embed not registered by Init")
	}
	embedder, _ := e.(api.Action).Desc().Metadata["embedder"].(map[string]any)
	embedSchema, _ := embedder["customOptions"].(map[string]any)
	embedProps, _ := embedSchema["properties"].(map[string]any)
	for _, key := range []string{"dimensions", "encodingFormat", "extra"} {
		if embedProps[key] == nil {
			t.Errorf("embedder config schema is missing the %q property", key)
		}
	}
	for _, key := range []string{"user", "apiKey"} {
		if embedProps[key] != nil {
			t.Errorf("embedder config schema advertises %q", key)
		}
	}
}

// TestDynamicListingAndResolution pins the on-demand surface: the chat models
// the endpoint reports are listed with the plugin's config schema, the models
// that do not chat are left out, and generating with an uncurated name
// resolves it instead of failing with model-not-found.
func TestDynamicListingAndResolution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/models" {
			_, _ = io.WriteString(w, `{"object":"list","data":[
				{"id":"mistral-brand-new","object":"model","created":1,"owned_by":"mistralai","capabilities":{"completion_chat":true,"function_calling":true}},
				{"id":"mistral-embed","object":"model","created":1,"owned_by":"mistralai","capabilities":{"completion_chat":false}},
				{"id":"mistral-ocr-latest","object":"model","created":1,"owned_by":"mistralai","capabilities":{"completion_chat":false,"ocr":true}}
			]}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"id":"c1","object":"chat.completion","created":1,"model":"mistral-brand-new",
			"choices":[{"index":0,"message":{"role":"assistant","content":"resolved"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`)
	}))
	defer server.Close()

	ctx := context.Background()
	plugin := newPlugin(server)
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	var names []string
	var listed *api.ActionDesc
	for _, desc := range plugin.ListActions(ctx) {
		names = append(names, desc.Name)
		if desc.Name == "mistral/mistral-brand-new" {
			listed = &desc
		}
	}
	if !slices.Equal(names, []string{"mistral/mistral-brand-new"}) {
		t.Fatalf("ListActions() = %v, want the one chat model", names)
	}
	model := listed.Metadata["model"].(map[string]any)
	schema, _ := model["customOptions"].(map[string]any)
	props, _ := schema["properties"].(map[string]any)
	if props["safePrompt"] == nil {
		t.Error("listed model schema is missing the plugin's safePrompt property")
	}
	// Mistral documents structured output on every model it serves, so an
	// uncurated model takes the curated constrained support.
	supports, _ := model["supports"].(map[string]any)
	if got := supports["constrained"]; got != ai.ConstrainedSupportNoTools {
		t.Errorf("listed model constrained = %v, want %q", got, ai.ConstrainedSupportNoTools)
	}

	resp, err := genkit.Generate(ctx, g, ai.WithModelName("mistral/mistral-brand-new"), ai.WithPrompt("hi"))
	if err != nil {
		t.Fatalf("Generate() with an uncurated model error = %v", err)
	}
	if got := resp.Text(); got != "resolved" {
		t.Fatalf("Text() = %q, want %q", got, "resolved")
	}
}

// Mistral answers an unknown or retired model with 400 and an invalid_model
// error at the top level of the body. The plugin reports NOT_FOUND for it,
// streamed or not, and leaves any other 400 as INVALID_ARGUMENT.
func TestUnknownModelIsNotFound(t *testing.T) {
	for body, want := range map[string]status.Name{
		`{"object":"error","message":"Invalid model: no-such-model","type":"invalid_model","param":null,"code":"1500","raw_status_code":400}`: status.NotFound,
		`{"object":"error","message":"Invalid max_tokens value","type":"invalid_request_error","param":null,"code":"3000"}`:                   status.InvalidArgument,
	} {
		for _, stream := range []bool{false, true} {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, body)
			}))
			g := genkit.Init(context.Background(), genkit.WithPlugins(newPlugin(server)))
			opts := []ai.GenerateOption{ai.WithModelName("mistral/no-such-model"), ai.WithPrompt("hi")}
			if stream {
				opts = append(opts, ai.WithStreaming(func(context.Context, *ai.ModelResponseChunk) error { return nil }))
			}
			_, err := genkit.Generate(context.Background(), g, opts...)
			server.Close()
			if got := status.Of(err); got != want {
				t.Errorf("stream %v, body %s: status = %q, want %q: %v", stream, body, got, want, err)
			}
		}
	}
}
