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
//
// SPDX-License-Identifier: Apache-2.0

package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/cohere-ai/cohere-go/v2/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

func TestResolveAPIKey(t *testing.T) {
	t.Setenv("COHERE_API_KEY", "cohere-key")
	t.Setenv("CO_API_KEY", "sdk-key")

	if got := resolveAPIKey("explicit-key"); got != "explicit-key" {
		t.Fatalf("explicit key precedence: got %q", got)
	}
	if got := resolveAPIKey(""); got != "cohere-key" {
		t.Fatalf("COHERE_API_KEY precedence: got %q", got)
	}

	t.Setenv("COHERE_API_KEY", "")
	if got := resolveAPIKey(""); got != "sdk-key" {
		t.Fatalf("CO_API_KEY fallback: got %q", got)
	}

	t.Setenv("CO_API_KEY", "")
	if got := resolveAPIKey(""); got != "" {
		t.Fatalf("empty environment: got %q", got)
	}
}

func TestCuratedModelsUseActiveIDs(t *testing.T) {
	for _, retired := range []string{"command-r", "command-r-plus"} {
		if _, ok := cohereChatModels[retired]; ok {
			t.Errorf("retired model %q must not be curated", retired)
		}
	}
	for _, active := range []string{
		"command-a-plus-05-2026",
		"command-a-03-2025",
		"command-a-reasoning-08-2025",
		"command-r-plus-08-2024",
		"command-r-08-2024",
		"command-r7b-12-2024",
	} {
		if _, ok := cohereChatModels[active]; !ok {
			t.Errorf("active model %q is not curated", active)
		}
	}
}

func TestCuratedEmbeddersIncludeLightModels(t *testing.T) {
	for _, tc := range []struct {
		id         string
		dimensions int
	}{
		{id: "embed-english-light-v3.0", dimensions: 384},
		{id: "embed-multilingual-light-v3.0", dimensions: 384},
	} {
		info, ok := cohereEmbedders[tc.id]
		if !ok {
			t.Errorf("light embedder %q is not curated", tc.id)
			continue
		}
		if info.Dimensions != tc.dimensions {
			t.Errorf("embedder %q dimensions = %d, want %d", tc.id, info.Dimensions, tc.dimensions)
		}
	}
}

func TestModelRef(t *testing.T) {
	config := &ChatOptions{}
	ref := ModelRef("command-a-03-2025", config)
	if ref.Name() != "cohere/command-a-03-2025" {
		t.Fatalf("ref name = %q", ref.Name())
	}
	if ref.Config() != config {
		t.Fatalf("ref config = %#v, want original pointer", ref.Config())
	}
	prefixed := ModelRef("cohere/command-a-03-2025", nil)
	if prefixed.Name() != "cohere/command-a-03-2025" {
		t.Fatalf("prefixed ref name = %q", prefixed.Name())
	}
}

func TestPluginListsAndResolvesActions(t *testing.T) {
	plugin := &Cohere{}
	actions := plugin.ListActions(context.Background())
	want := len(cohereChatModels) + len(cohereEmbedders)
	if len(actions) != want {
		t.Fatalf("ListActions returned %d actions, want %d", len(actions), want)
	}
	if action := plugin.ResolveAction(api.ActionTypeModel, "command-a-03-2025"); action == nil {
		t.Fatal("failed to resolve known model")
	}
	if action := plugin.ResolveAction(api.ActionTypeEmbedder, "embed-v4.0"); action == nil {
		t.Fatal("failed to resolve known embedder")
	}
	if action := plugin.ResolveAction(api.ActionTypeRetriever, "anything"); action != nil {
		t.Fatalf("resolved unsupported action type: %T", action)
	}
}

func TestPluginInitRequiresAuthAndRejectsSecondInit(t *testing.T) {
	t.Run("missing auth", func(t *testing.T) {
		t.Setenv("COHERE_API_KEY", "")
		t.Setenv("CO_API_KEY", "")
		defer func() {
			if recover() == nil {
				t.Fatal("Init did not panic without authentication")
			}
		}()
		(&Cohere{}).Init(context.Background())
	})

	t.Run("second init", func(t *testing.T) {
		plugin := &Cohere{APIKey: "test-key"}
		plugin.Init(context.Background())
		defer func() {
			if recover() == nil {
				t.Fatal("second Init did not panic")
			}
		}()
		plugin.Init(context.Background())
	})
}

func TestModelAndEmbedderOptionFallbacks(t *testing.T) {
	knownModel := GetModelOptions("command-a-03-2025")
	if knownModel.Label == "" || knownModel.Supports == nil {
		t.Fatalf("known model options = %+v", knownModel)
	}
	unknownModel := GetModelOptions("future-command")
	if unknownModel.Label != "Cohere - future-command" || unknownModel.Supports == nil {
		t.Fatalf("unknown model options = %+v", unknownModel)
	}
	knownEmbedder := GetEmbedderOptions("embed-v4.0")
	if knownEmbedder.Dimensions != 1536 {
		t.Fatalf("known embedder options = %+v", knownEmbedder)
	}
	unknownEmbedder := GetEmbedderOptions("future-embed")
	if unknownEmbedder.Label != "Cohere - future-embed" || unknownEmbedder.Dimensions != 0 {
		t.Fatalf("unknown embedder options = %+v", unknownEmbedder)
	}
}

// TestGenerateStreamToolCall exercises Cohere's actual SSE decoding path. In
// particular, it verifies that argument fragments are joined by stream index,
// emitted once at tool-call-end, and preserved in the aggregate response.
func TestGenerateStreamToolCall(t *testing.T) {
	events := []string{
		`{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\""}}}}}`,
		`{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"Paris\"}"}}}}}`,
		`{"type":"tool-call-end","index":0}`,
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/chat" {
			t.Errorf("request path = %q, want /v2/chat", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	client := cohereclient.NewClient(
		option.WithToken("test-key"),
		option.WithBaseURL(server.URL),
	)
	input := &ai.ModelRequest{Messages: []*ai.Message{{
		Role:    ai.RoleUser,
		Content: []*ai.Part{ai.NewTextPart("weather in Paris")},
	}}}

	var chunks []*ai.ModelResponseChunk
	response, err := generate(context.Background(), client, "command-r", input, ChatOptions{},
		func(_ context.Context, chunk *ai.ModelResponseChunk) error {
			chunks = append(chunks, chunk)
			return nil
		})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(chunks) != 1 || len(chunks[0].Content) != 1 || !chunks[0].Content[0].IsToolRequest() {
		t.Fatalf("streamed chunks = %#v, want one tool request", chunks)
	}
	assertToolRequest(t, chunks[0].Content[0].ToolRequest)

	if response.Message == nil || len(response.Message.Content) != 1 || !response.Message.Content[0].IsToolRequest() {
		t.Fatalf("aggregate content = %#v, want one tool request", response.Message)
	}
	assertToolRequest(t, response.Message.Content[0].ToolRequest)
}

// TestChatOptionsPassActionSchema guards the action boundary that validates a
// config after JSON serialization. Using V2ChatRequest here used to inject a
// stream property that its inferred schema rejected before the API was called.
func TestChatOptionsPassActionSchema(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/chat" {
			t.Errorf("request path = %q, want /v2/chat", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"response_1","finish_reason":"COMPLETE","message":{"role":"assistant","content":[{"type":"text","text":"ok"}]}}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Cohere{
		APIKey:  "test-key",
		BaseURL: server.URL,
	}))
	if IsDefinedModel(g, "command-r") {
		t.Fatal("IsDefinedModel resolved an unregistered model")
	}
	maxTokens := 32
	temperature := 0.2
	response, err := genkit.Generate(ctx, g,
		ai.WithModel(ModelRef("command-r", &ChatOptions{MaxTokens: &maxTokens, Temperature: &temperature})),
		ai.WithPrompt("say ok"),
	)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := response.Text(); got != "ok" {
		t.Fatalf("response text = %q, want ok", got)
	}
	if !IsDefinedModel(g, "command-r") {
		t.Fatal("resolved model was not registered")
	}
}

func TestChatOptionsMapEveryField(t *testing.T) {
	strict := true
	safety := cohere.V2ChatRequestSafetyModeContextual
	maxTokens := 128
	temperature := 0.4
	seed := 7
	frequencyPenalty := 0.1
	presencePenalty := 0.2
	k := 20
	p := 0.8
	logprobs := true
	toolChoice := cohere.V2ChatRequestToolChoiceRequired
	priority := 1
	options := ChatOptions{
		StrictTools:      &strict,
		Documents:        []*cohere.V2ChatRequestDocumentsItem{{String: "source"}},
		CitationOptions:  &cohere.CitationOptions{},
		ResponseFormat:   &cohere.ResponseFormatV2{Type: "json_object"},
		SafetyMode:       &safety,
		MaxTokens:        &maxTokens,
		StopSequences:    []string{"STOP"},
		Temperature:      &temperature,
		Seed:             &seed,
		FrequencyPenalty: &frequencyPenalty,
		PresencePenalty:  &presencePenalty,
		K:                &k,
		P:                &p,
		Logprobs:         &logprobs,
		ToolChoice:       &toolChoice,
		Thinking:         &cohere.Thinking{},
		Priority:         &priority,
	}
	request := options.request()

	if request.StrictTools != options.StrictTools || len(request.Documents) != 1 || request.Documents[0].String != "source" {
		t.Errorf("strict tools/documents not mapped: %+v", request)
	}
	if request.CitationOptions != options.CitationOptions || request.ResponseFormat != options.ResponseFormat || request.Thinking != options.Thinking {
		t.Errorf("complex options not mapped: %+v", request)
	}
	if request.SafetyMode != options.SafetyMode || request.ToolChoice != options.ToolChoice {
		t.Errorf("enum options not mapped: %+v", request)
	}
	if request.MaxTokens != options.MaxTokens || request.Temperature != options.Temperature || request.Seed != options.Seed ||
		request.FrequencyPenalty != options.FrequencyPenalty || request.PresencePenalty != options.PresencePenalty ||
		request.K != options.K || request.P != options.P || request.Logprobs != options.Logprobs || request.Priority != options.Priority {
		t.Errorf("scalar options not mapped: %+v", request)
	}
	if len(request.StopSequences) != 1 || request.StopSequences[0] != "STOP" {
		t.Errorf("stop sequences = %v", request.StopSequences)
	}
}

func TestStructuredOutputOverridesResponseFormat(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
	}
	request, err := toCohereRequest(&ai.ModelRequest{
		Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("answer")}}},
		Output:   &ai.ModelOutputConfig{Format: "json", Schema: schema, Constrained: true},
	}, ChatOptions{ResponseFormat: &cohere.ResponseFormatV2{Type: "text"}})
	if err != nil {
		t.Fatalf("toCohereRequest: %v", err)
	}
	if request.ResponseFormat == nil || request.ResponseFormat.Type != "json_object" || request.ResponseFormat.JsonObject == nil {
		t.Fatalf("response format = %+v", request.ResponseFormat)
	}
	if request.ResponseFormat.JsonObject.JsonSchema["type"] != "object" {
		t.Fatalf("JSON schema = %#v", request.ResponseFormat.JsonObject.JsonSchema)
	}
}

func TestGenerateStreamAggregatesReasoningTextCitationsAndUsage(t *testing.T) {
	client := newSSETestClient(t, []string{
		`{"type":"content-delta","index":0,"delta":{"message":{"content":{"thinking":"reason "}}}}`,
		`{"type":"content-delta","index":0,"delta":{"message":{"content":{"thinking":"carefully"}}}}`,
		`{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"the answer"}}}}`,
		`{"type":"citation-start","index":0,"delta":{"message":{"citations":{"start":0,"end":10,"text":"the answer","sources":[],"type":"TEXT_CONTENT"}}}}`,
		`{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"tokens":{"input_tokens":3,"output_tokens":2}}}}`,
	})

	var chunks []*ai.ModelResponseChunk
	response, err := generate(context.Background(), client, "command-a-03-2025", &ai.ModelRequest{
		Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("answer")}}},
	}, ChatOptions{}, func(_ context.Context, chunk *ai.ModelResponseChunk) error {
		chunks = append(chunks, chunk)
		return nil
	})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(chunks) != 3 || !chunks[0].Content[0].IsReasoning() || !chunks[1].Content[0].IsReasoning() || !chunks[2].Content[0].IsText() {
		t.Fatalf("chunks = %#v", chunks)
	}
	if len(response.Message.Content) != 2 || response.Message.Content[0].Text != "reason carefully" || response.Message.Content[1].Text != "the answer" {
		t.Fatalf("aggregate content = %#v", response.Message.Content)
	}
	if response.FinishReason != ai.FinishReasonStop || response.Usage == nil || response.Usage.InputTokens != 3 || response.Usage.OutputTokens != 2 {
		t.Fatalf("finish/usage = %q/%+v", response.FinishReason, response.Usage)
	}
	custom, ok := response.Custom.(map[string]any)
	if !ok {
		t.Fatalf("custom = %T, want map", response.Custom)
	}
	citations, ok := custom["citations"].([]*cohere.Citation)
	if !ok || len(citations) != 1 || citations[0].Text == nil || *citations[0].Text != "the answer" {
		t.Fatalf("citations = %#v", custom["citations"])
	}
}

func TestGenerateStreamReturnsCallbackError(t *testing.T) {
	client := newSSETestClient(t, []string{
		`{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"hello"}}}}`,
	})
	want := errors.New("stop streaming")
	_, err := generate(context.Background(), client, "command-a-03-2025", &ai.ModelRequest{
		Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hello")}}},
	}, ChatOptions{}, func(context.Context, *ai.ModelResponseChunk) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("generate error = %v, want %v", err, want)
	}
}

func TestGenerateStreamRejectsMalformedToolArguments(t *testing.T) {
	client := newSSETestClient(t, []string{
		`{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"call_bad","type":"function","function":{"name":"lookup","arguments":"{bad"}}}}}`,
		`{"type":"tool-call-end","index":0}`,
	})
	_, err := generate(context.Background(), client, "command-a-03-2025", &ai.ModelRequest{
		Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("lookup")}}},
	}, ChatOptions{}, func(context.Context, *ai.ModelResponseChunk) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "unable to parse tool call arguments") {
		t.Fatalf("generate error = %v", err)
	}
}

func newSSETestClient(t *testing.T, events []string) *cohereclient.Client {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	return cohereclient.NewClient(option.WithToken("test-key"), option.WithBaseURL(server.URL))
}

func assertToolRequest(t *testing.T, request *ai.ToolRequest) {
	t.Helper()
	if request.Ref != "call_1" || request.Name != "get_weather" {
		t.Fatalf("tool request identity = %+v", request)
	}
	input, ok := request.Input.(map[string]any)
	if !ok || input["city"] != "Paris" {
		t.Fatalf("tool request input = %#v", request.Input)
	}
}

func TestToCohereMessages(t *testing.T) {
	messages := []*ai.Message{
		{Role: ai.RoleSystem, Content: []*ai.Part{ai.NewTextPart("be terse")}},
		{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hi there")}},
		{Role: ai.RoleModel, Content: []*ai.Part{
			ai.NewTextPart("let me check"),
			ai.NewToolRequestPart(&ai.ToolRequest{Ref: "call_1", Name: "lookup", Input: map[string]any{"q": "weather"}}),
		}},
		{Role: ai.RoleTool, Content: []*ai.Part{
			ai.NewToolResponsePart(&ai.ToolResponse{Ref: "call_1", Name: "lookup", Output: map[string]any{"temp": 21}}),
		}},
	}

	out, err := toCohereMessages(messages)
	if err != nil {
		t.Fatalf("toCohereMessages: %v", err)
	}
	if len(out) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(out))
	}

	if out[0].Role != "system" || out[0].System == nil || out[0].System.Content.String != "be terse" {
		t.Errorf("system message mapped incorrectly: %+v", out[0])
	}
	if out[1].Role != "user" || out[1].User == nil || out[1].User.Content.String != "hi there" {
		t.Errorf("user message mapped incorrectly: %+v", out[1])
	}

	asst := out[2]
	if asst.Role != "assistant" || asst.Assistant == nil {
		t.Fatalf("expected assistant message, got %+v", asst)
	}
	if asst.Assistant.Content == nil || asst.Assistant.Content.String != "let me check" {
		t.Errorf("assistant text mapped incorrectly: %+v", asst.Assistant.Content)
	}
	if len(asst.Assistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(asst.Assistant.ToolCalls))
	}
	tc := asst.Assistant.ToolCalls[0]
	if tc.Id != "call_1" || tc.Function == nil || tc.Function.Name == nil || *tc.Function.Name != "lookup" {
		t.Errorf("tool call mapped incorrectly: %+v", tc)
	}
	if tc.Function.Arguments == nil || *tc.Function.Arguments != `{"q":"weather"}` {
		t.Errorf("tool call arguments mapped incorrectly: %v", tc.Function.Arguments)
	}

	tool := out[3]
	if tool.Role != "tool" || tool.Tool == nil || tool.Tool.ToolCallId != "call_1" {
		t.Fatalf("expected tool message with id call_1, got %+v", tool)
	}
	if tool.Tool.Content == nil || tool.Tool.Content.String != `{"temp":21}` {
		t.Errorf("tool result mapped incorrectly: %v", tool.Tool.Content)
	}
}

func TestToCohereTools(t *testing.T) {
	tools := []*ai.ToolDefinition{
		{
			Name:        "get_weather",
			Description: "look up the weather",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		},
		{Name: "ping"}, // empty schema -> default object schema
	}

	out, err := toCohereTools(tools)
	if err != nil {
		t.Fatalf("toCohereTools: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(out))
	}
	if out[0].Function.Name != "get_weather" || out[0].Function.Description == nil || *out[0].Function.Description != "look up the weather" {
		t.Errorf("tool[0] mapped incorrectly: %+v", out[0].Function)
	}
	if out[0].Function.Parameters["type"] != "object" {
		t.Errorf("tool[0] parameters not passed through: %+v", out[0].Function.Parameters)
	}
	if out[1].Function.Parameters["type"] != "object" {
		t.Errorf("tool[1] should get default object schema, got: %+v", out[1].Function.Parameters)
	}

	if _, err := toCohereTools([]*ai.ToolDefinition{{Name: ""}}); err == nil {
		t.Error("expected error for tool with empty name")
	}

	if out, err := toCohereTools(nil); err != nil || out != nil {
		t.Errorf("nil tools should map to nil, no error; got %v, %v", out, err)
	}
}

func TestToGenkitResponse(t *testing.T) {
	name := "get_weather"
	args := `{"city":"SF"}`
	inTok := 12.0
	outTok := 7.0
	resp := &cohere.V2ChatResponse{
		FinishReason: cohere.ChatFinishReasonComplete,
		Message: &cohere.AssistantMessageResponse{
			Content: []*cohere.AssistantMessageResponseContentItem{
				{Type: "text", Text: &cohere.ChatTextContent{Text: "here you go"}},
			},
			ToolCalls: []*cohere.ToolCallV2{
				{Id: "call_9", Function: &cohere.ToolCallV2Function{Name: &name, Arguments: &args}},
			},
			Citations: []*cohere.Citation{{Text: strPtr("cited snippet")}},
		},
		Usage: &cohere.Usage{Tokens: &cohere.UsageTokens{InputTokens: &inTok, OutputTokens: &outTok}},
	}

	r, err := toGenkitResponse(resp)
	if err != nil {
		t.Fatalf("toGenkitResponse: %v", err)
	}
	if r.FinishReason != ai.FinishReasonStop {
		t.Errorf("finish reason = %q, want stop", r.FinishReason)
	}
	if r.Message.Role != ai.RoleModel {
		t.Errorf("role = %q, want model", r.Message.Role)
	}
	if len(r.Message.Content) != 2 {
		t.Fatalf("expected 2 content parts (text + tool), got %d", len(r.Message.Content))
	}
	if !r.Message.Content[0].IsText() || r.Message.Content[0].Text != "here you go" {
		t.Errorf("text part mapped incorrectly: %+v", r.Message.Content[0])
	}
	if !r.Message.Content[1].IsToolRequest() {
		t.Fatalf("expected tool request part, got %+v", r.Message.Content[1])
	}
	tr := r.Message.Content[1].ToolRequest
	if tr.Ref != "call_9" || tr.Name != "get_weather" {
		t.Errorf("tool request mapped incorrectly: %+v", tr)
	}
	if input, ok := tr.Input.(map[string]any); !ok || input["city"] != "SF" {
		t.Errorf("tool request input not parsed: %#v", tr.Input)
	}
	if r.Usage == nil || r.Usage.InputTokens != 12 || r.Usage.OutputTokens != 7 {
		t.Errorf("usage mapped incorrectly: %+v", r.Usage)
	}
	custom, ok := r.Custom.(map[string]any)
	if !ok {
		t.Fatalf("expected Custom map, got %T", r.Custom)
	}
	if _, ok := custom["citations"]; !ok {
		t.Errorf("citations not preserved in Custom: %+v", custom)
	}
}

func TestToGenkitResponseReasoning(t *testing.T) {
	resp := &cohere.V2ChatResponse{
		FinishReason: cohere.ChatFinishReasonComplete,
		Message: &cohere.AssistantMessageResponse{
			Content: []*cohere.AssistantMessageResponseContentItem{
				{Type: "thinking", Thinking: &cohere.ChatThinkingContent{Thinking: "let me reason"}},
				{Type: "text", Text: &cohere.ChatTextContent{Text: "the answer"}},
			},
		},
	}

	r, err := toGenkitResponse(resp)
	if err != nil {
		t.Fatalf("toGenkitResponse: %v", err)
	}
	if len(r.Message.Content) != 2 {
		t.Fatalf("expected reasoning + text parts, got %d", len(r.Message.Content))
	}
	if !r.Message.Content[0].IsReasoning() || r.Message.Content[0].Text != "let me reason" {
		t.Errorf("first part should be reasoning: %+v", r.Message.Content[0])
	}
	if !r.Message.Content[1].IsText() || r.Message.Content[1].Text != "the answer" {
		t.Errorf("second part should be text: %+v", r.Message.Content[1])
	}
}

func TestToGenkitFinishReason(t *testing.T) {
	cases := map[cohere.ChatFinishReason]ai.FinishReason{
		cohere.ChatFinishReasonComplete:     ai.FinishReasonStop,
		cohere.ChatFinishReasonStopSequence: ai.FinishReasonStop,
		cohere.ChatFinishReasonToolCall:     ai.FinishReasonStop,
		cohere.ChatFinishReasonMaxTokens:    ai.FinishReasonLength,
		cohere.ChatFinishReasonError:        ai.FinishReasonUnknown,
		cohere.ChatFinishReasonTimeout:      ai.FinishReasonUnknown,
	}
	for in, want := range cases {
		if got := toGenkitFinishReason(in); got != want {
			t.Errorf("toGenkitFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestToolCallPart(t *testing.T) {
	part, err := toolCallPart("ref1", "fn", `{"a":1}`)
	if err != nil {
		t.Fatalf("toolCallPart: %v", err)
	}
	if !part.IsToolRequest() || part.ToolRequest.Ref != "ref1" || part.ToolRequest.Name != "fn" {
		t.Errorf("tool call part mapped incorrectly: %+v", part)
	}
	if input, ok := part.ToolRequest.Input.(map[string]any); !ok || input["a"] != float64(1) {
		t.Errorf("input not parsed: %#v", part.ToolRequest.Input)
	}

	// Empty args -> nil input, no error.
	empty, err := toolCallPart("ref2", "fn", "")
	if err != nil {
		t.Fatalf("toolCallPart empty: %v", err)
	}
	if empty.ToolRequest.Input != nil {
		t.Errorf("expected nil input for empty args, got %#v", empty.ToolRequest.Input)
	}

	// Malformed args -> error.
	if _, err := toolCallPart("ref3", "fn", "{not json"); err == nil {
		t.Error("expected error for malformed tool arguments")
	}
}

func TestToStreamRequest(t *testing.T) {
	temp := 0.5
	maxTok := 256
	safety := cohere.V2ChatRequestSafetyModeStrict
	choice := cohere.V2ChatRequestToolChoiceRequired
	req := &cohere.V2ChatRequest{
		Model:       "command-r",
		Temperature: &temp,
		MaxTokens:   &maxTok,
		SafetyMode:  &safety,
		ToolChoice:  &choice,
		Messages: cohere.ChatMessages{
			{Role: "user", User: &cohere.UserMessageV2{Content: &cohere.UserMessageV2Content{String: "hi"}}},
		},
	}

	got, err := toStreamRequest(req)
	if err != nil {
		t.Fatalf("toStreamRequest: %v", err)
	}
	if got.Model != "command-r" {
		t.Errorf("model = %q, want command-r", got.Model)
	}
	if got.Temperature == nil || *got.Temperature != 0.5 {
		t.Errorf("temperature not copied: %v", got.Temperature)
	}
	if got.MaxTokens == nil || *got.MaxTokens != 256 {
		t.Errorf("max tokens not copied: %v", got.MaxTokens)
	}
	if got.SafetyMode == nil || *got.SafetyMode != cohere.V2ChatStreamRequestSafetyModeStrict {
		t.Errorf("safety mode not copied: %v", got.SafetyMode)
	}
	if got.ToolChoice == nil || *got.ToolChoice != cohere.V2ChatStreamRequestToolChoiceRequired {
		t.Errorf("tool choice not copied: %v", got.ToolChoice)
	}
	if len(got.Messages) != 1 {
		t.Errorf("messages not copied: %d", len(got.Messages))
	}
}

// TestToCohereMessagesSkipsEmpty guards against the SDK content-union marshal
// error: an empty user/system message must be dropped, not allocated with an
// empty string that fails to serialize.
func TestToCohereMessagesSkipsEmpty(t *testing.T) {
	out, err := toCohereMessages([]*ai.Message{
		{Role: ai.RoleSystem, Content: []*ai.Part{ai.NewTextPart("")}},
		{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("")}},
		{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("real question")}},
	})
	if err != nil {
		t.Fatalf("toCohereMessages: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected empty messages to be skipped, got %d", len(out))
	}
	if out[0].User.Content.String != "real question" {
		t.Errorf("wrong surviving message: %+v", out[0])
	}
}

// TestRequestMarshals exercises the full request build + JSON encode, the path
// the SDK takes before hitting the wire. It would have caught the empty-content
// marshal crash.
func TestRequestMarshals(t *testing.T) {
	req, err := toCohereRequest(&ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleSystem, Content: []*ai.Part{ai.NewTextPart("be terse")}},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hello")}},
		},
		Tools: []*ai.ToolDefinition{{Name: "t", Description: "d", InputSchema: map[string]any{"type": "object"}}},
	}, ChatOptions{})
	if err != nil {
		t.Fatalf("toCohereRequest: %v", err)
	}
	req.Model = "command-r"

	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	got := string(b)
	for _, want := range []string{`"model":"command-r"`, `"role":"system"`, `"role":"user"`, `"type":"function"`} {
		if !strings.Contains(got, want) {
			t.Errorf("marshaled request missing %q: %s", want, got)
		}
	}
}

// TestNewModelSchema confirms a model (and its reflected config schema) can be
// built without a client or Init — the registration path used by ListActions.
func TestNewModelSchema(t *testing.T) {
	if (&Cohere{}).newModel("command-r") == nil {
		t.Fatal("newModel returned nil")
	}
	if schema := core.InferSchemaMap(ChatOptions{}); len(schema) == 0 {
		t.Fatal("empty config schema")
	}
}

func strPtr(s string) *string { return &s }
