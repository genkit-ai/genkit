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

func TestCacheControlOnTextAndSystem(t *testing.T) {
	cached := ai.NewTextPart("cached system")
	cached.Metadata = map[string]any{
		"cache_control": map[string]any{
			"type": "ephemeral",
			"ttl":  "5m",
		},
	}
	uncached := ai.NewTextPart("plain system")

	got, err := toAnthropicRequest("anthropic", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleSystem, Content: []*ai.Part{uncached, cached}},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hi")}},
		},
		Config: map[string]any{"max_tokens": 10},
	})
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	if len(got.System) != 2 {
		t.Fatalf("System len = %d, want 2", len(got.System))
	}
	sysWire := wireJSON(t, got.System)
	if !strings.Contains(sysWire, `"text":"plain system"`) {
		t.Fatalf("missing uncached system text: %s", sysWire)
	}
	if !strings.Contains(sysWire, `"text":"cached system"`) {
		t.Fatalf("missing cached system text: %s", sysWire)
	}
	if !strings.Contains(sysWire, `"cache_control"`) {
		t.Fatalf("missing cache_control on system: %s", sysWire)
	}
	if !strings.Contains(sysWire, `"ttl":"5m"`) {
		t.Fatalf("missing ttl on system cache_control: %s", sysWire)
	}
	// First system block should not carry cache_control.
	firstWire := wireJSON(t, got.System[0])
	if strings.Contains(firstWire, "cache_control") {
		t.Fatalf("uncached system block unexpectedly has cache_control: %s", firstWire)
	}
}

func TestCacheControlOnUserContentBlocks(t *testing.T) {
	text := ai.NewTextPart("hello")
	text.Metadata = map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	}
	img := ai.NewMediaPart("image/png", "data:image/png;base64,AAAA")
	img.Metadata = map[string]any{
		"cache_control": map[string]any{"type": "ephemeral", "ttl": "1h"},
	}
	toolResp := ai.NewToolResponsePart(&ai.ToolResponse{
		Ref:    "call-1",
		Name:   "lookup",
		Output: map[string]any{"ok": true},
	})
	toolResp.Metadata = map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	}

	blocks, err := toAnthropicParts([]*ai.Part{text, img, toolResp})
	if err != nil {
		t.Fatalf("toAnthropicParts: %v", err)
	}
	wire := wireJSON(t, blocks)
	if strings.Count(wire, `"cache_control"`) < 3 {
		t.Fatalf("expected cache_control on text/image/tool_result, got %s", wire)
	}
	if !strings.Contains(wire, `"ttl":"1h"`) {
		t.Fatalf("missing 1h ttl on image: %s", wire)
	}
}

func TestCacheControlSkippedOnReasoning(t *testing.T) {
	p := ai.NewReasoningPart("think", []byte("sig"))
	if p.Metadata == nil {
		p.Metadata = map[string]any{}
	}
	p.Metadata["cache_control"] = map[string]any{"type": "ephemeral"}
	blocks, err := toAnthropicParts([]*ai.Part{p})
	if err != nil {
		t.Fatalf("toAnthropicParts: %v", err)
	}
	wire := wireJSON(t, blocks)
	if strings.Contains(wire, "cache_control") {
		t.Fatalf("thinking block must not carry cache_control: %s", wire)
	}
}

func TestCacheControlInvalidTTL(t *testing.T) {
	p := ai.NewTextPart("x")
	p.Metadata = map[string]any{
		"cache_control": map[string]any{"type": "ephemeral", "ttl": "10m"},
	}
	_, err := toAnthropicParts([]*ai.Part{p})
	if err == nil || !strings.Contains(err.Error(), "ttl") {
		t.Fatalf("error = %v, want ttl validation error", err)
	}
}

func TestToGenkitResponseCacheUsage(t *testing.T) {
	msg := &anthropic.Message{
		StopReason: anthropic.StopReasonEndTurn,
		Content: []anthropic.ContentBlockUnion{
			{Type: "text", Text: "hi"},
		},
		Usage: anthropic.Usage{
			InputTokens:              100,
			OutputTokens:             10,
			CacheCreationInputTokens: 80,
			CacheReadInputTokens:     20,
			CacheCreation: anthropic.CacheCreation{
				Ephemeral5mInputTokens: 70,
				Ephemeral1hInputTokens: 10,
			},
		},
	}
	// Content union needs proper JSON for AsAny in some SDK versions; use
	// Unmarshal if Text field alone isn't enough.
	raw, _ := json.Marshal(map[string]any{
		"id":          "msg_1",
		"type":        "message",
		"role":        "assistant",
		"model":       "claude",
		"stop_reason": "end_turn",
		"content":     []any{map[string]any{"type": "text", "text": "hi"}},
		"usage": map[string]any{
			"input_tokens":                100,
			"output_tokens":               10,
			"cache_creation_input_tokens": 80,
			"cache_read_input_tokens":     20,
			"cache_creation": map[string]any{
				"ephemeral_5m_input_tokens": 70,
				"ephemeral_1h_input_tokens": 10,
			},
		},
	})
	if err := json.Unmarshal(raw, msg); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}

	got, err := toGenkitResponse(msg)
	if err != nil {
		t.Fatalf("toGenkitResponse: %v", err)
	}
	if got.Usage.CachedContentTokens != 20 {
		t.Errorf("CachedContentTokens = %d, want 20", got.Usage.CachedContentTokens)
	}
	if got.Usage.Custom["cache_creation_input_tokens"] != 80 {
		t.Errorf("cache_creation_input_tokens = %v, want 80", got.Usage.Custom["cache_creation_input_tokens"])
	}
	if got.Usage.Custom["cache_read_input_tokens"] != 20 {
		t.Errorf("cache_read_input_tokens = %v, want 20", got.Usage.Custom["cache_read_input_tokens"])
	}
	if got.Usage.Custom["ephemeral_5m_input_tokens"] != 70 {
		t.Errorf("ephemeral_5m_input_tokens = %v, want 70", got.Usage.Custom["ephemeral_5m_input_tokens"])
	}
	if got.Usage.Custom["ephemeral_1h_input_tokens"] != 10 {
		t.Errorf("ephemeral_1h_input_tokens = %v, want 10", got.Usage.Custom["ephemeral_1h_input_tokens"])
	}
}
