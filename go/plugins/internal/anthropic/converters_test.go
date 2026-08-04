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

package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
)

func TestToGenkitResponseServerTools(t *testing.T) {
	raw := `{
		"id": "msg_test",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [
			{
				"type": "server_tool_use",
				"id": "toolu_search",
				"name": "web_search",
				"input": {"query": "genkit anthropic"},
				"caller": {"type": "direct"}
			},
			{
				"type": "web_search_tool_result",
				"tool_use_id": "toolu_search",
				"content": [
					{
						"type": "web_search_result",
						"url": "https://example.com",
						"title": "Example",
						"encrypted_content": "abc",
						"page_age": null
					}
				],
				"caller": {"type": "direct"}
			},
			{
				"type": "text",
				"text": "Here is what I found."
			}
		],
		"stop_reason": "end_turn",
		"stop_sequence": null,
		"usage": {"input_tokens": 10, "output_tokens": 20}
	}`

	var msg anthropic.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}

	got, err := toGenkitResponse(&msg)
	if err != nil {
		t.Fatalf("toGenkitResponse: %v", err)
	}
	if got.FinishReason != ai.FinishReasonStop {
		t.Errorf("FinishReason = %v, want stop", got.FinishReason)
	}
	if len(got.Message.Content) != 3 {
		t.Fatalf("got %d parts, want 3", len(got.Message.Content))
	}

	use := got.Message.Content[0]
	if !strings.Contains(use.Text, "[Anthropic server tool web_search]") {
		t.Errorf("server tool text = %q", use.Text)
	}
	metaUse, ok := use.Metadata["anthropicServerToolUse"].(map[string]any)
	if !ok {
		t.Fatalf("missing anthropicServerToolUse metadata: %#v", use.Metadata)
	}
	if metaUse["id"] != "toolu_search" || metaUse["name"] != "web_search" {
		t.Errorf("server tool metadata = %#v", metaUse)
	}

	result := got.Message.Content[1]
	if !strings.Contains(result.Text, "[Anthropic server tool result toolu_search]") {
		t.Errorf("result text = %q", result.Text)
	}
	metaResult, ok := result.Metadata["anthropicServerToolResult"].(map[string]any)
	if !ok {
		t.Fatalf("missing anthropicServerToolResult metadata: %#v", result.Metadata)
	}
	if metaResult["type"] != "web_search_tool_result" || metaResult["toolUseId"] != "toolu_search" {
		t.Errorf("result metadata = %#v", metaResult)
	}

	if got.Message.Content[2].Text != "Here is what I found." {
		t.Errorf("text part = %q", got.Message.Content[2].Text)
	}
}

func TestToBetaGenkitResponseStopReasons(t *testing.T) {
	tests := []struct {
		reason anthropic.BetaStopReason
		want   ai.FinishReason
	}{
		{anthropic.BetaStopReasonMaxTokens, ai.FinishReasonLength},
		{anthropic.BetaStopReasonModelContextWindowExceeded, ai.FinishReasonLength},
		{anthropic.BetaStopReasonEndTurn, ai.FinishReasonStop},
		{anthropic.BetaStopReasonPauseTurn, ai.FinishReasonStop},
		{anthropic.BetaStopReasonRefusal, ai.FinishReasonOther},
		{"", ai.FinishReasonUnknown},
		{"something-new", ai.FinishReasonOther},
	}
	for _, tt := range tests {
		t.Run(string(tt.reason), func(t *testing.T) {
			got, err := toBetaGenkitResponse(&anthropic.BetaMessage{
				StopReason: tt.reason,
				Content:    []anthropic.BetaContentBlockUnion{},
			})
			if err != nil {
				t.Fatalf("toBetaGenkitResponse: %v", err)
			}
			if got.FinishReason != tt.want {
				t.Errorf("FinishReason = %q, want %q", got.FinishReason, tt.want)
			}
		})
	}
}

func TestToBetaGenkitResponseServerTools(t *testing.T) {
	raw := `{
		"id": "msg_beta",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-4-20250514",
		"content": [
			{
				"type": "server_tool_use",
				"id": "toolu_1",
				"name": "web_search",
				"input": {"query": "go genkit"},
				"caller": {"type": "direct"}
			},
			{
				"type": "web_search_tool_result",
				"tool_use_id": "toolu_1",
				"content": [{"type": "web_search_result", "url": "https://go.dev", "title": "Go", "encrypted_content": "x"}],
				"caller": {"type": "direct"}
			}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 1, "output_tokens": 2}
	}`

	var msg anthropic.BetaMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal beta message: %v", err)
	}
	got, err := toBetaGenkitResponse(&msg)
	if err != nil {
		t.Fatalf("toBetaGenkitResponse: %v", err)
	}
	if len(got.Message.Content) != 2 {
		t.Fatalf("got %d parts, want 2", len(got.Message.Content))
	}
	if _, ok := got.Message.Content[0].Metadata["anthropicServerToolUse"]; !ok {
		t.Errorf("missing server tool use metadata")
	}
	if _, ok := got.Message.Content[1].Metadata["anthropicServerToolResult"]; !ok {
		t.Errorf("missing server tool result metadata")
	}
}

func TestResolveAPIVersion(t *testing.T) {
	tests := []struct {
		name           string
		pluginDefault  string
		config         any
		wantVersion    string
		wantBetasCount int
		wantBetasNil   bool
	}{
		{"default stable", "", nil, APIVersionStable, 0, true},
		{"plugin beta", APIVersionBeta, nil, APIVersionBeta, 0, true},
		{"request overrides plugin", APIVersionStable, map[string]any{"apiVersion": "beta"}, APIVersionBeta, 0, true},
		{"request betas", "", map[string]any{"apiVersion": "beta", "betas": []any{"files-api-2025-04-14"}}, APIVersionBeta, 1, false},
		{"explicit empty betas", "", map[string]any{"apiVersion": "beta", "betas": []any{}}, APIVersionBeta, 0, false},
		{"typed params ignore routing", APIVersionBeta, &anthropic.MessageNewParams{MaxTokens: 1}, APIVersionBeta, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotVersion, gotBetas := resolveAPIVersion(&ai.ModelRequest{Config: tt.config}, tt.pluginDefault)
			if gotVersion != tt.wantVersion {
				t.Errorf("version = %q, want %q", gotVersion, tt.wantVersion)
			}
			if len(gotBetas) != tt.wantBetasCount {
				t.Errorf("betas = %#v, want len %d", gotBetas, tt.wantBetasCount)
			}
			if tt.wantBetasNil != (gotBetas == nil) {
				t.Errorf("betas nil = %v, want %v", gotBetas == nil, tt.wantBetasNil)
			}
		})
	}
}

func TestConfigFromRequestStripsRoutingFields(t *testing.T) {
	req := &ai.ModelRequest{
		Config: map[string]any{
			"max_tokens": 123,
			"apiVersion": "beta",
			"betas":      []any{"files-api-2025-04-14"},
		},
	}
	got, err := configFromRequest(req)
	if err != nil {
		t.Fatalf("configFromRequest: %v", err)
	}
	if got.MaxTokens != 123 {
		t.Errorf("MaxTokens = %d, want 123", got.MaxTokens)
	}
}

func TestShouldEmitOnContentBlockStop(t *testing.T) {
	tests := []struct {
		name string
		part *ai.Part
		want bool
	}{
		{"nil", nil, false},
		{"text", ai.NewTextPart("hi"), false},
		{"reasoning with signature", ai.NewReasoningPart("think", []byte("sig")), false},
		{"redacted thinking", ai.NewCustomPart(map[string]any{"redactedThinking": "x"}), false},
		{"tool request", ai.NewToolRequestPart(&ai.ToolRequest{Name: "x"}), true},
		{"server tool use", serverToolUseToPart("id", "web_search", map[string]any{"q": "a"}), true},
		{"server tool result", webSearchToolResultToPart("id", []any{}), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldEmitOnContentBlockStop(tt.part); got != tt.want {
				t.Errorf("shouldEmitOnContentBlockStop() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestToBetaRequestSetsDefaultBetas(t *testing.T) {
	req := &anthropic.MessageNewParams{
		MaxTokens: 100,
		Model:     "claude-sonnet-4-20250514",
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	}
	got, err := toBetaRequest(req, nil)
	if err != nil {
		t.Fatalf("toBetaRequest: %v", err)
	}
	if got.MaxTokens != 100 {
		t.Errorf("MaxTokens = %d", got.MaxTokens)
	}
	if len(got.Betas) == 0 {
		t.Fatal("expected default betas")
	}

	empty, err := toBetaRequest(req, []anthropic.AnthropicBeta{})
	if err != nil {
		t.Fatalf("toBetaRequest empty: %v", err)
	}
	if empty.Betas == nil || len(empty.Betas) != 0 {
		t.Fatalf("explicit empty betas = %#v, want non-nil empty", empty.Betas)
	}
}
