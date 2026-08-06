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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
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
