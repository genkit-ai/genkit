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
	"encoding/json"
	"strings"
	"testing"

	cohere "github.com/cohere-ai/cohere-go/v2"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
)

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

func TestConfigFromRequest(t *testing.T) {
	temp := 0.7
	t.Run("nil", func(t *testing.T) {
		got, err := configFromRequest(&ai.ModelRequest{})
		if err != nil || got == nil {
			t.Fatalf("nil config: got %v, err %v", got, err)
		}
	})
	t.Run("struct value", func(t *testing.T) {
		got, err := configFromRequest(&ai.ModelRequest{Config: cohere.V2ChatRequest{Temperature: &temp}})
		if err != nil || got.Temperature == nil || *got.Temperature != 0.7 {
			t.Fatalf("struct config: got %+v, err %v", got, err)
		}
	})
	t.Run("pointer", func(t *testing.T) {
		got, err := configFromRequest(&ai.ModelRequest{Config: &cohere.V2ChatRequest{Temperature: &temp}})
		if err != nil || got.Temperature == nil || *got.Temperature != 0.7 {
			t.Fatalf("pointer config: got %+v, err %v", got, err)
		}
	})
	t.Run("map", func(t *testing.T) {
		got, err := configFromRequest(&ai.ModelRequest{Config: map[string]any{"temperature": 0.7}})
		if err != nil || got.Temperature == nil || *got.Temperature != 0.7 {
			t.Fatalf("map config: got %+v, err %v", got, err)
		}
	})
	t.Run("unsupported", func(t *testing.T) {
		if _, err := configFromRequest(&ai.ModelRequest{Config: 42}); err == nil {
			t.Error("expected error for unsupported config type")
		}
	})
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
	})
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
	if schema := core.InferSchemaMap(cohere.V2ChatRequest{}); len(schema) == 0 {
		t.Fatal("empty config schema")
	}
}

func strPtr(s string) *string { return &s }
