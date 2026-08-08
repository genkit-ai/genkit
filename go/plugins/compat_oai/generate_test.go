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

func TestOpenAICompatibleSnapshotsConfigAliasesAtInit(t *testing.T) {
	aliases := map[string]string{"maxTokens": "max_tokens"}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if got := body["max_tokens"]; got != float64(64) {
			t.Errorf("max_tokens = %v, want 64", got)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chatcmpl-1",
			"object":"chat.completion",
			"created":1,
			"model":"test-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]
		}`)
	}))
	defer server.Close()

	plugin := &OpenAICompatible{
		Provider:      "snapshot",
		APIKey:        "test-key",
		BaseURL:       server.URL,
		ConfigAliases: aliases,
	}
	plugin.Init(context.Background())
	aliases["maxTokens"] = "changed_after_init"

	model := plugin.DefineModel("snapshot", "test-model", ai.ModelOptions{})
	_, err := model.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("hello")},
		Config:   map[string]any{"maxTokens": 64},
	}, nil)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
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

func TestWithConfigPreservesProviderSpecificFields(t *testing.T) {
	g := newGen().WithConfig(map[string]any{
		"maxOutputTokens": 123,
		"model":           "must-not-override",
		"temperature":     0.3,
		"topP":            0.8,
		"thinking": map[string]any{
			"type":           "enabled",
			"clear_thinking": false,
		},
	})
	if g.err != nil {
		t.Fatalf("WithConfig() error = %v", g.err)
	}

	data, err := json.Marshal(g.GetRequest())
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got := request["model"]; got != "test-model" {
		t.Errorf("model = %v, want %q", got, "test-model")
	}
	if got := request["temperature"]; got != 0.3 {
		t.Errorf("temperature = %v, want 0.3", got)
	}
	if got := request["top_p"]; got != 0.8 {
		t.Errorf("top_p = %v, want 0.8", got)
	}
	extraFields := g.GetRequest().ExtraFields()
	if _, ok := extraFields["temperature"]; ok {
		t.Error("temperature was added as an extra field")
	}
	if _, ok := extraFields["top_p"]; ok {
		t.Error("top_p was added as an extra field")
	}
	if _, ok := extraFields["model"]; ok {
		t.Error("model was added as an extra field")
	}
	if _, ok := request["topP"]; ok {
		t.Error("request contains unconverted topP field")
	}
	if _, ok := request["maxOutputTokens"]; ok {
		t.Error("request contains unsupported maxOutputTokens field")
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
	if _, ok := extraFields["thinking"]; !ok {
		t.Error("thinking was not preserved as a provider-specific extra field")
	}
}

func TestWithConfigAppliesProviderAliases(t *testing.T) {
	g := newGen().
		WithConfigAliases(map[string]string{
			"reasoningEffort":  "reasoning_effort",
			"webSearchOptions": "web_search_options",
		}).
		WithConfig(map[string]any{
			"deferred":         true,
			"reasoningEffort":  "high",
			"webSearchOptions": map[string]any{"search_context_size": "high"},
		})
	if g.err != nil {
		t.Fatalf("WithConfig() error = %v", g.err)
	}

	data, err := json.Marshal(g.GetRequest())
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got := request["reasoning_effort"]; got != "high" {
		t.Errorf("reasoning_effort = %v, want high", got)
	}
	webSearch, ok := request["web_search_options"].(map[string]any)
	if !ok {
		t.Fatalf("web_search_options = %#v, want object", request["web_search_options"])
	}
	if got := webSearch["search_context_size"]; got != "high" {
		t.Errorf("web_search_options.search_context_size = %v, want high", got)
	}
	if got := request["deferred"]; got != true {
		t.Errorf("deferred = %v, want true", got)
	}
	for _, name := range []string{"reasoningEffort", "webSearchOptions"} {
		if _, ok := request[name]; ok {
			t.Errorf("request contains unconverted %s field", name)
		}
	}
	for _, name := range []string{"reasoning_effort", "web_search_options"} {
		if _, ok := g.GetRequest().ExtraFields()[name]; ok {
			t.Errorf("SDK-supported field %s was retained as an extra", name)
		}
	}
}

func TestWithConfigPreservesIdentityAlias(t *testing.T) {
	g := newGen().
		WithConfigAliases(map[string]string{"deferred": "deferred"}).
		WithConfig(map[string]any{"deferred": true})
	if g.err != nil {
		t.Fatalf("WithConfig() error = %v", g.err)
	}

	data, err := json.Marshal(g.GetRequest())
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var request map[string]any
	if err := json.Unmarshal(data, &request); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if got := request["deferred"]; got != true {
		t.Errorf("deferred = %v, want true", got)
	}
}

func TestWithConfigIgnoresNilPointer(t *testing.T) {
	var config *openai.ChatCompletionNewParams
	g := newGen().WithConfig(config)
	if g.err != nil {
		t.Fatalf("WithConfig() error = %v", g.err)
	}
	if got := g.GetRequest().Model; got != "test-model" {
		t.Errorf("model = %q, want %q", got, "test-model")
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
