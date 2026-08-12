// Copyright 2025 Google LLC
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

package compat_oai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

func TestConvertChatCompletionToModelResponseReasoningContent(t *testing.T) {
	var completion openai.ChatCompletion
	if err := json.Unmarshal([]byte(`{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"created": 1,
		"model": "reasoning-model",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"reasoning_content": "Let me think...",
				"content": "Final answer"
			},
			"finish_reason": "stop"
		}]
	}`), &completion); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	resp, err := convertChatCompletionToModelResponse(&completion)
	if err != nil {
		t.Fatalf("convertChatCompletionToModelResponse() error = %v", err)
	}
	if got := resp.Reasoning(); got != "Let me think..." {
		t.Errorf("Reasoning() = %q, want %q", got, "Let me think...")
	}
	if got := resp.Text(); got != "Final answer" {
		t.Errorf("Text() = %q, want %q", got, "Final answer")
	}
	if len(resp.Message.Content) != 2 ||
		!resp.Message.Content[0].IsReasoning() ||
		!resp.Message.Content[1].IsText() {
		t.Fatalf("content = %#v, want reasoning followed by text", resp.Message.Content)
	}
}

// newGen returns a ModelGenerator with a nil client; only local tool-shaping
// logic is exercised, so no network call is made.
func newGen() *ModelGenerator {
	return NewModelGenerator((*openai.Client)(nil), "test-model")
}

// newStubGen returns a ModelGenerator pointed at a stub OpenAI-compatible
// endpoint that replies with body, so both generate paths can be exercised
// without a live API key.
func newStubGen(t *testing.T, contentType, body string) *ModelGenerator {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	client := openai.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("stub"))
	return NewModelGenerator(&client, "test-model")
}

// Regression test for #4683: the streaming path used to leave Request as an
// empty &ai.ModelRequest{}, so History() dropped every input message.
func TestGeneratePreservesRequest(t *testing.T) {
	messages := []*ai.Message{
		ai.NewUserTextMessage("first user turn"),
		ai.NewModelTextMessage("first model turn"),
		ai.NewUserTextMessage("second user turn"),
	}

	const streamBody = `data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":null}]}

data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	const completeBody = `{"id":"1","object":"chat.completion","created":1,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}]}`

	for _, tc := range []struct {
		name        string
		contentType string
		body        string
		handleChunk func(context.Context, *ai.ModelResponseChunk) error
	}{
		{"streaming", "text/event-stream", streamBody, func(context.Context, *ai.ModelResponseChunk) error { return nil }},
		{"complete", "application/json", completeBody, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &ai.ModelRequest{Messages: messages}
			resp, err := newStubGen(t, tc.contentType, tc.body).
				WithMessages(messages).
				Generate(context.Background(), req, tc.handleChunk)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if resp.Request != req {
				t.Fatalf("Request = %#v, want the originating request", resp.Request)
			}
			if got := len(resp.History()); got != len(messages)+1 {
				t.Errorf("len(History()) = %d, want %d (inputs plus model reply)", got, len(messages)+1)
			}
		})
	}
}

func TestConvertChatCompletionToModelResponseProviderFinishReasons(t *testing.T) {
	for finishReason, want := range map[string]ai.FinishReason{
		"sensitive":                     ai.FinishReasonBlocked,
		"model_context_window_exceeded": ai.FinishReasonLength,
		"network_error":                 ai.FinishReasonOther,
		"insufficient_system_resource":  ai.FinishReasonOther,
		// xAI documents three finish reasons and this is one of them, so an
		// unmapped end_turn would report ordinary answers as unknown.
		"end_turn": ai.FinishReasonStop,
	} {
		t.Run(finishReason, func(t *testing.T) {
			completion := &openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{{
					FinishReason: finishReason,
					Message: openai.ChatCompletionMessage{
						Role: "assistant",
					},
				}},
			}
			resp, err := convertChatCompletionToModelResponse(completion)
			if err != nil {
				t.Fatalf("convertChatCompletionToModelResponse() error = %v", err)
			}
			if resp.FinishReason != want {
				t.Errorf("FinishReason = %q, want %q", resp.FinishReason, want)
			}
		})
	}
}

// TestConvertChatCompletionToModelResponseCachedTokens covers both shapes a
// provider reports prompt cache hits in: OpenAI's prompt_tokens_details
// breakdown and the top-level field DeepSeek uses, which returns no
// prompt_tokens_details at all.
func TestConvertChatCompletionToModelResponseCachedTokens(t *testing.T) {
	for _, tc := range []struct {
		name  string
		usage string
		want  int
	}{
		{
			name:  "openai prompt_tokens_details",
			usage: `{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":6}}`,
			want:  6,
		},
		{
			name:  "deepseek prompt_cache_hit_tokens",
			usage: `{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_cache_hit_tokens":6,"prompt_cache_miss_tokens":4}`,
			want:  6,
		},
		{
			name:  "no cache hit reported",
			usage: `{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":10}`,
			want:  0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var completion openai.ChatCompletion
			if err := json.Unmarshal([]byte(`{
				"id":"1","object":"chat.completion","created":1,"model":"test-model",
				"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
				"usage":`+tc.usage+`
			}`), &completion); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}

			resp, err := convertChatCompletionToModelResponse(&completion)
			if err != nil {
				t.Fatalf("convertChatCompletionToModelResponse() error = %v", err)
			}
			if got := resp.Usage.CachedContentTokens; got != tc.want {
				t.Errorf("CachedContentTokens = %d, want %d", got, tc.want)
			}
			if got := resp.Usage.InputTokens; got != 10 {
				t.Errorf("InputTokens = %d, want 10", got)
			}
		})
	}
}

// TestConvertChatCompletionToModelResponseCitations pins that the sources
// behind a live-search answer survive the conversion. They arrive in a
// response field of xAI's own that the SDK does not model, and a caller who
// asked for citations has no other way to read them back.
func TestConvertChatCompletionToModelResponseCitations(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want []any
	}{
		{
			name: "citations alongside a system fingerprint",
			body: `"system_fingerprint":"fp_1","citations":["https://x.com/xai","https://x.ai"]`,
			want: []any{"https://x.com/xai", "https://x.ai"},
		},
		{
			// The fingerprint is nullable, and gating the metadata on it would
			// drop the citations of every response that omits it.
			name: "citations without a system fingerprint",
			body: `"citations":["https://x.ai"]`,
			want: []any{"https://x.ai"},
		},
		{
			name: "no citations",
			body: `"system_fingerprint":"fp_1"`,
			want: nil,
		},
		{
			name: "null citations",
			body: `"citations":null`,
			want: nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var completion openai.ChatCompletion
			if err := json.Unmarshal([]byte(`{
				"id":"1","object":"chat.completion","created":1,"model":"grok-4.6",`+tc.body+`,
				"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
			}`), &completion); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}

			resp, err := convertChatCompletionToModelResponse(&completion)
			if err != nil {
				t.Fatalf("convertChatCompletionToModelResponse() error = %v", err)
			}
			custom, _ := resp.Custom.(map[string]any)
			got, _ := custom["citations"].([]any)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("custom[citations] = %#v, want %#v", got, tc.want)
			}
			// Custom is deprecated in favor of Raw, so both carry the metadata.
			if tc.want != nil && !reflect.DeepEqual(resp.Raw, resp.Custom) {
				t.Errorf("Raw = %#v, want it to match Custom %#v", resp.Raw, resp.Custom)
			}
		})
	}
}

// TestConvertChatCompletionToModelResponseSearchUsage pins the usage counts
// xAI reports that OpenAI's shape has no field for.
func TestConvertChatCompletionToModelResponseSearchUsage(t *testing.T) {
	var completion openai.ChatCompletion
	if err := json.Unmarshal([]byte(`{
		"id":"1","object":"chat.completion","created":1,"model":"grok-4.6",
		"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"end_turn"}],
		"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"num_sources_used":3,
			"prompt_tokens_details":{"image_tokens":7}}
	}`), &completion); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	resp, err := convertChatCompletionToModelResponse(&completion)
	if err != nil {
		t.Fatalf("convertChatCompletionToModelResponse() error = %v", err)
	}
	for name, want := range map[string]float64{"numSourcesUsed": 3, "imageTokens": 7} {
		if got := resp.Usage.Custom[name]; got != want {
			t.Errorf("Usage.Custom[%q] = %v, want %v", name, got, want)
		}
	}
}

// TestGenerateStreamReportsUsage pins that a stream asks for its token usage
// and reports every part of it. The usage rides on a final chunk the request
// has to opt into, and [openai.ChatCompletionAccumulator] sums only the three
// top-level counts, so without both halves a streamed response reports zero
// tokens or loses the reasoning and cache breakdowns.
func TestGenerateStreamReportsUsage(t *testing.T) {
	var includeUsage bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			StreamOptions struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		includeUsage = body.StreamOptions.IncludeUsage

		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":null},{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":7,"total_tokens":17,"prompt_cache_hit_tokens":6,"prompt_cache_miss_tokens":4,"completion_tokens_details":{"reasoning_tokens":5}}}`,
		} {
			_, _ = io.WriteString(w, "data: "+event+"\n\n")
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	client := openai.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("stub"))
	messages := []*ai.Message{ai.NewUserTextMessage("hi")}
	resp, err := NewModelGenerator(&client, "test-model").
		WithMessages(messages).
		Generate(context.Background(), &ai.ModelRequest{Messages: messages},
			func(context.Context, *ai.ModelResponseChunk) error { return nil })
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if !includeUsage {
		t.Error("stream_options.include_usage = false, want the stream to ask for its usage")
	}
	for _, tc := range []struct {
		field string
		got   int
		want  int
	}{
		{"InputTokens", resp.Usage.InputTokens, 10},
		{"OutputTokens", resp.Usage.OutputTokens, 7},
		{"TotalTokens", resp.Usage.TotalTokens, 17},
		{"ThoughtsTokens", resp.Usage.ThoughtsTokens, 5},
		{"CachedContentTokens", resp.Usage.CachedContentTokens, 6},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.field, tc.got, tc.want)
		}
	}
}

// TestGenerateStreamReportsCitations pins that a streamed live search reports
// its sources too. [openai.ChatCompletionAccumulator] rebuilds the completion
// from the fields it models, so citations that arrive on a chunk are lost
// unless they are carried out of the stream directly.
func TestGenerateStreamReportsCitations(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"grok-4.6","choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":null}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"grok-4.6","choices":[{"index":0,"delta":{},"finish_reason":"end_turn"}],"citations":["https://x.ai"]}`,
		} {
			_, _ = io.WriteString(w, "data: "+event+"\n\n")
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	client := openai.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("stub"))
	messages := []*ai.Message{ai.NewUserTextMessage("what is new?")}
	resp, err := NewModelGenerator(&client, "grok-4.6").
		WithMessages(messages).
		Generate(context.Background(), &ai.ModelRequest{Messages: messages},
			func(context.Context, *ai.ModelResponseChunk) error { return nil })
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	custom, _ := resp.Custom.(map[string]any)
	got, _ := custom["citations"].([]any)
	if want := []any{"https://x.ai"}; !reflect.DeepEqual(got, want) {
		t.Errorf("custom[citations] = %#v, want %#v", got, want)
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishReasonStop)
	}
}

func TestWithMessagesPreservesReasoningContent(t *testing.T) {
	g := newGen().WithMessages([]*ai.Message{{
		Role: ai.RoleModel,
		Content: []*ai.Part{
			ai.NewReasoningPart("Think carefully.", nil),
			ai.NewTextPart("Final answer"),
		},
	}})
	if g.err != nil {
		t.Fatalf("WithMessages() error = %v", g.err)
	}

	data, err := json.Marshal(g.messages[0])
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var message map[string]any
	if err := json.Unmarshal(data, &message); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got := message["reasoning_content"]; got != "Think carefully." {
		t.Errorf("reasoning_content = %v, want %q", got, "Think carefully.")
	}
	if got := message["content"]; got != "Final answer" {
		t.Errorf("content = %v, want %q", got, "Final answer")
	}
}

func TestWithMessagesSkipsNilMessagesAndParts(t *testing.T) {
	g := newGen().WithMessages([]*ai.Message{
		nil,
		{
			Role:    ai.RoleSystem,
			Content: []*ai.Part{nil, ai.NewTextPart("System prompt")},
		},
		{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				nil,
				ai.NewReasoningPart("Reasoning", nil),
				ai.NewTextPart("Answer"),
			},
		},
		{
			Role:    ai.RoleTool,
			Content: []*ai.Part{nil},
		},
		{
			Role:    ai.RoleUser,
			Content: []*ai.Part{nil, ai.NewTextPart("Question")},
		},
	})
	if g.err != nil {
		t.Fatalf("WithMessages() error = %v", g.err)
	}
	if got := len(g.messages); got != 3 {
		t.Fatalf("messages = %d, want 3 non-empty messages", got)
	}
}

// TestWithParams pins how a request is built on top of the config's params:
// a model the params carry pins the served model while an empty one falls
// back to the generator's, SDK-modeled fields land on their wire names, and
// provider-specific extra fields ride along.
func TestWithParams(t *testing.T) {
	if got := newGen().WithParams(openai.ChatCompletionNewParams{}).GetRequest().Model; got != "test-model" {
		t.Errorf("model = %q for empty params, want the generator's %q", got, "test-model")
	}

	params := openai.ChatCompletionNewParams{
		Model:       "test-model-2026-01-01",
		Temperature: openai.Float(0.3),
		TopP:        openai.Float(0.8),
	}
	params.SetExtraFields(map[string]any{
		"thinking": map[string]any{
			"type":           "enabled",
			"clear_thinking": false,
		},
	})

	g := newGen().WithParams(params)
	if g.err != nil {
		t.Fatalf("WithParams() error = %v", g.err)
	}

	request := marshalParams(t, *g.GetRequest())
	if got := request["model"]; got != "test-model-2026-01-01" {
		t.Errorf("model = %v, want the pinned %q", got, "test-model-2026-01-01")
	}
	if got := request["temperature"]; got != 0.3 {
		t.Errorf("temperature = %v, want 0.3", got)
	}
	if got := request["top_p"]; got != 0.8 {
		t.Errorf("top_p = %v, want 0.8", got)
	}
	thinking, ok := request["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking = %#v, want object", request["thinking"])
	}
	if got := thinking["type"]; got != "enabled" {
		t.Errorf("thinking.type = %v, want %q", got, "enabled")
	}
	if got := thinking["clear_thinking"]; got != false {
		t.Errorf("thinking.clear_thinking = %v, want false", got)
	}
}

func TestConcatenateContentSkipsNilParts(t *testing.T) {
	parts := []*ai.Part{
		nil,
		ai.NewTextPart("visible"),
		ai.NewReasoningPart("thought", nil),
		nil,
		ai.NewDataPart(" data"),
	}
	if got := concatenateTextContent(parts); got != "visible data" {
		t.Errorf("concatenateTextContent() = %q, want %q", got, "visible data")
	}
	if got := concatenateReasoningContent(parts); got != "thought" {
		t.Errorf("concatenateReasoningContent() = %q, want %q", got, "thought")
	}
}

func TestWithToolChoice(t *testing.T) {
	for _, toolChoice := range []ai.ToolChoice{
		ai.ToolChoiceAuto,
		ai.ToolChoiceNone,
		ai.ToolChoiceRequired,
	} {
		t.Run(string(toolChoice), func(t *testing.T) {
			g := newGen().WithToolChoice(toolChoice)
			if g.err != nil {
				t.Fatalf("WithToolChoice() error = %v", g.err)
			}
			if !g.request.ToolChoice.OfAuto.Valid() {
				t.Fatal("tool choice was not set")
			}
			if got := g.request.ToolChoice.OfAuto.Value; got != string(toolChoice) {
				t.Errorf("tool choice = %q, want %q", got, toolChoice)
			}
		})
	}
}

func TestWithToolChoiceRejectsUnsupportedValue(t *testing.T) {
	g := newGen().WithToolChoice(ai.ToolChoice("sometimes"))
	if g.err == nil {
		t.Fatal("WithToolChoice() error = nil, want unsupported tool choice error")
	}
}

// TestWithTools_StrictDefaultsOff verifies the default behavior of sending
// strict: false to OpenAI when the tool has no strict metadata.
func TestWithTools_StrictDefaultsOff(t *testing.T) {
	g := newGen().WithTools([]*ai.ToolDefinition{
		{
			Name:        "ping",
			Description: "ping",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"msg": map[string]any{"type": "string"}},
			},
		},
	})
	if len(g.tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(g.tools))
	}
	fn := g.tools[0].Function
	if !fn.Strict.Valid() || fn.Strict.Value {
		t.Errorf("expected Strict=false by default, got valid=%v value=%v", fn.Strict.Valid(), fn.Strict.Value)
	}
	if _, present := fn.Parameters["additionalProperties"]; present {
		t.Errorf("expected no additionalProperties when strict is off, got %v", fn.Parameters["additionalProperties"])
	}
}

// TestWithTools_StrictOptIn verifies a tool with Metadata["strict"]=true is
// sent with Strict=true and additionalProperties: false applied recursively.
func TestWithTools_StrictOptIn(t *testing.T) {
	g := newGen().WithTools([]*ai.ToolDefinition{
		{
			Name:        "weather",
			Description: "get weather",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city": map[string]any{"type": "string"},
						},
					},
				},
			},
			Metadata: map[string]any{"strict": true},
		},
	})
	if len(g.tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(g.tools))
	}
	fn := g.tools[0].Function
	if !fn.Strict.Valid() || !fn.Strict.Value {
		t.Errorf("expected Strict=true, got valid=%v value=%v", fn.Strict.Valid(), fn.Strict.Value)
	}
	if fn.Parameters["additionalProperties"] != false {
		t.Errorf("expected top-level additionalProperties: false, got %v", fn.Parameters["additionalProperties"])
	}
	props, _ := fn.Parameters["properties"].(map[string]any)
	loc, _ := props["location"].(map[string]any)
	if loc["additionalProperties"] != false {
		t.Errorf("expected nested additionalProperties: false on location, got %v", loc["additionalProperties"])
	}
}

// TestWithTools_StrictExplicitFalse verifies an explicit opt-out matches the
// default: Strict=false and the schema is not enforced.
func TestWithTools_StrictExplicitFalse(t *testing.T) {
	g := newGen().WithTools([]*ai.ToolDefinition{
		{
			Name:        "ping",
			Description: "ping",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"msg": map[string]any{"type": "string"}},
			},
			Metadata: map[string]any{"strict": false},
		},
	})
	fn := g.tools[0].Function
	if !fn.Strict.Valid() || fn.Strict.Value {
		t.Errorf("expected Strict=false, got valid=%v value=%v", fn.Strict.Valid(), fn.Strict.Value)
	}
	if _, present := fn.Parameters["additionalProperties"]; present {
		t.Errorf("expected no additionalProperties when strict is off, got %v", fn.Parameters["additionalProperties"])
	}
}

// TestWithTools_StrictDoesNotMutateCallerSchema guards against the caller's
// input schema being mutated in place by strict-mode enforcement.
func TestWithTools_StrictDoesNotMutateCallerSchema(t *testing.T) {
	original := map[string]any{
		"type":       "object",
		"properties": map[string]any{"msg": map[string]any{"type": "string"}},
	}
	def := &ai.ToolDefinition{
		Name:        "ping",
		Description: "ping",
		InputSchema: original,
		Metadata:    map[string]any{"strict": true},
	}
	newGen().WithTools([]*ai.ToolDefinition{def})
	if _, present := original["additionalProperties"]; present {
		t.Errorf("caller schema was mutated: additionalProperties leaked into original input schema")
	}
}

// TestWithConfigDeprecatedPath pins the behavior the released generator has,
// which plugins built on this package still call: a camelCase map is rewritten
// to the SDK's wire names, keys the SDK does not model ride as extra fields,
// and the generator's model survives the config.
func TestWithConfigDeprecatedPath(t *testing.T) {
	client := openai.NewClient()
	gen := NewModelGenerator(&client, "test-model").WithConfig(map[string]any{
		"topP":             0.4,
		"frequencyPenalty": 0.25,
		"maxOutputTokens":  512,
		"enable_search":    true,
	})

	if gen.err != nil {
		t.Fatalf("WithConfig() error = %v", gen.err)
	}
	if got := gen.request.Model; got != "test-model" {
		t.Errorf("model = %q, want the generator's %q", got, "test-model")
	}
	if got := gen.request.TopP.Or(0); got != 0.4 {
		t.Errorf("top_p = %v, want 0.4", got)
	}
	if got := gen.request.FrequencyPenalty.Or(0); got != 0.25 {
		t.Errorf("frequency_penalty = %v, want 0.25", got)
	}
	// maxOutputTokens is dropped by this path, as it always has been.
	if gen.request.MaxTokens.Valid() {
		t.Errorf("max_tokens = %v, want the key dropped", gen.request.MaxTokens)
	}
	if got := gen.request.ExtraFields()["enable_search"]; got != true {
		t.Errorf("enable_search extra field = %v, want true", got)
	}

	// A typed SDK config passes through, and an unusable one errors.
	gen = NewModelGenerator(&client, "test-model").WithConfig(openai.ChatCompletionNewParams{Seed: openai.Int(7)})
	if gen.err != nil {
		t.Fatalf("WithConfig(params) error = %v", gen.err)
	}
	if got := gen.request.Seed.Or(0); got != 7 {
		t.Errorf("seed = %v, want 7", got)
	}
	if gen := NewModelGenerator(&client, "test-model").WithConfig("nonsense"); gen.err == nil {
		t.Error("WithConfig(string) error = nil, want an unexpected-config-type error")
	}
}

// managedFieldParams is a config setting every request field Genkit owns,
// including the deprecated function pair OpenAI used before tools.
func managedFieldParams() openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Temperature: openai.Float(0.5),
		Messages:    []openai.ChatCompletionMessageParamUnion{openai.UserMessage("smuggled")},
		Tools: []openai.ChatCompletionToolParam{{
			Function: shared.FunctionDefinitionParam{Name: "smuggled_tool"},
		}},
		ToolChoice:   openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")},
		Functions:    []openai.ChatCompletionNewParamsFunction{{Name: "smuggled_function"}},
		FunctionCall: openai.ChatCompletionNewParamsFunctionCallUnion{OfFunctionCallMode: openai.String("auto")},
	}
	// A managed name among the extra fields would override the struct field
	// at marshal time, so the clearing has to sweep the extras too; the
	// benign neighbor proves the sweep is surgical.
	params.SetExtraFields(map[string]any{"messages": "smuggled", "safe_mode": true})
	return params
}

// assertManagedFieldsCleared checks that nothing a config set in a
// Genkit-managed field survived onto the request.
func assertManagedFieldsCleared(t *testing.T, gen *ModelGenerator) {
	t.Helper()
	if gen.err != nil {
		t.Fatalf("builder error = %v", gen.err)
	}
	if len(gen.request.Messages) != 0 {
		t.Errorf("messages = %v, want cleared: Genkit builds them from the request", gen.request.Messages)
	}
	if len(gen.request.Tools) != 0 {
		t.Errorf("tools = %v, want cleared: a config must not offer a tool the framework cannot dispatch", gen.request.Tools)
	}
	if !reflect.DeepEqual(gen.request.ToolChoice, openai.ChatCompletionToolChoiceOptionUnionParam{}) {
		t.Errorf("tool_choice = %v, want cleared", gen.request.ToolChoice)
	}
	if len(gen.request.Functions) != 0 {
		t.Errorf("functions = %v, want cleared", gen.request.Functions)
	}
	if !reflect.DeepEqual(gen.request.FunctionCall, openai.ChatCompletionNewParamsFunctionCallUnion{}) {
		t.Errorf("function_call = %v, want cleared", gen.request.FunctionCall)
	}
	extras := gen.request.ExtraFields()
	if _, ok := extras["messages"]; ok {
		t.Error(`extra field "messages" survived, want managed names swept from the extras too`)
	}
	// Clearing is scoped to the managed fields; the rest of the config rides.
	if got := gen.request.Temperature.Or(0); got != 0.5 {
		t.Errorf("temperature = %v, want the config's 0.5 preserved", got)
	}
	if extras["safe_mode"] != true {
		t.Error(`extra field "safe_mode" swept, want the sweep scoped to managed names`)
	}
}

// TestWithParamsClearsManagedFields pins that a config cannot smuggle a tool,
// a message, or the deprecated function pair onto the wire. The model schema
// rejects those keys before a request reaches the generator, so this is the
// layer that holds for callers driving the generator directly, where nothing
// validates the params first.
func TestWithParamsClearsManagedFields(t *testing.T) {
	client := openai.NewClient()
	gen := NewModelGenerator(&client, "test-model").WithParams(managedFieldParams())
	assertManagedFieldsCleared(t, gen)

	if got := gen.request.Model; got != "test-model" {
		t.Errorf("model = %q, want the generator's %q", got, "test-model")
	}
}

// TestWithConfigClearsManagedFields is [TestWithParamsClearsManagedFields] for
// the deprecated entry point, which is still exported and still reachable.
func TestWithConfigClearsManagedFields(t *testing.T) {
	client := openai.NewClient()
	gen := NewModelGenerator(&client, "test-model").WithConfig(managedFieldParams())
	assertManagedFieldsCleared(t, gen)
}

// TestGenerateOmitsManagedFieldsOnTheWire is the end-to-end form: a request
// whose config carried tools, with no Genkit tools to replace them, must send
// none.
func TestGenerateOmitsManagedFieldsOnTheWire(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("unmarshal request: %v (%s)", err, raw)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"1","object":"chat.completion","model":"test-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	client := openai.NewClient(option.WithBaseURL(server.URL), option.WithAPIKey("test-key"))
	_, err := NewModelGenerator(&client, "test-model").
		WithParams(managedFieldParams()).
		WithMessages([]*ai.Message{ai.NewUserTextMessage("hello")}).
		WithTools(nil).
		WithToolChoice("").
		Generate(context.Background(), &ai.ModelRequest{}, nil)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	for _, field := range []string{"tools", "tool_choice", "functions", "function_call"} {
		if got, ok := body[field]; ok {
			t.Errorf("request sent %q = %v, want the field omitted", field, got)
		}
	}
	msgs, ok := body["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("messages = %v, want only the one Genkit built", body["messages"])
	}
	if raw, _ := json.Marshal(msgs[0]); strings.Contains(string(raw), "smuggled") {
		t.Errorf("messages carried the config's own message: %s", raw)
	}
}

// TestApplyResponseFormatHonorsDeclaredOutputs pins the response_format
// policy: a schema keeps its json_schema form no matter what the model
// declares, a schema-less JSON request only sends json_object to a model
// whose declared output formats include "json" (none declared permits every
// format), and the text format never sends anything since it would only
// restate the default and some compatible endpoints reject the parameter.
func TestApplyResponseFormatHonorsDeclaredOutputs(t *testing.T) {
	client := openai.NewClient()
	jsonOutput := &ai.ModelOutputConfig{Format: "json"}
	schemaOutput := &ai.ModelOutputConfig{
		Format: "json",
		Schema: map[string]any{"type": "object"},
	}

	g := NewModelGenerator(&client, "m")
	g.applyResponseFormat(jsonOutput)
	if g.request.ResponseFormat.OfJSONObject == nil {
		t.Error("no declaration: json_object should be sent")
	}

	g = NewModelGenerator(&client, "m").WithOutputFormats([]string{"text", "json"})
	g.applyResponseFormat(jsonOutput)
	if g.request.ResponseFormat.OfJSONObject == nil {
		t.Error("declared json: json_object should be sent")
	}

	g = NewModelGenerator(&client, "m").WithOutputFormats([]string{"text"})
	g.applyResponseFormat(jsonOutput)
	if g.request.ResponseFormat.OfJSONObject != nil {
		t.Error("declared without json: json_object should be dropped")
	}

	g = NewModelGenerator(&client, "m").WithOutputFormats([]string{"text"})
	g.applyResponseFormat(schemaOutput)
	if g.request.ResponseFormat.OfJSONSchema == nil {
		t.Error("a schema keeps its json_schema form regardless of the declaration")
	}

	g = NewModelGenerator(&client, "m")
	g.applyResponseFormat(&ai.ModelOutputConfig{Format: "text"})
	if g.request.ResponseFormat.OfText != nil {
		t.Error("the text format should send nothing")
	}
}
