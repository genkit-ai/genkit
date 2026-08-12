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
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/go-cmp/cmp"
)

func TestAnthropic(t *testing.T) {
	t.Run("to anthropic role", func(t *testing.T) {
		tests := []struct {
			role        ai.Role
			want        anthropic.MessageParamRole
			expectedErr string
		}{
			{ai.RoleModel, anthropic.MessageParamRoleAssistant, ""},
			{ai.RoleUser, anthropic.MessageParamRoleUser, ""},
			{ai.RoleSystem, "", "unknown role given"},
			{ai.RoleTool, anthropic.MessageParamRoleAssistant, ""},
			{"unknown", "", "unknown role given"},
		}

		for _, tt := range tests {
			got, err := toAnthropicRole(tt.role)
			if tt.expectedErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.expectedErr) {
					t.Errorf("toAnthropicRole(%q) error = %v, want error containing %q", tt.role, err, tt.expectedErr)
				}
				continue
			}
			if err != nil {
				t.Errorf("toAnthropicRole(%q) unexpected error: %v", tt.role, err)
			}
			if got != tt.want {
				t.Errorf("toAnthropicRole(%q) = %q, want %q", tt.role, got, tt.want)
			}
		}
	})
}

type modelRequestTestCase struct {
	name        string
	req         *ai.ModelRequest
	config      anthropic.MessageNewParams
	expected    *anthropic.MessageNewParams
	expectedErr string
}

// validateConfig runs a config through the same check the action boundary
// performs: the config schema the model advertises is enforced on every call.
func validateConfig(t *testing.T, inputSchema map[string]any, config any) error {
	t.Helper()
	return base.ValidateValue(&ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hello"))},
		Config:   config,
	}, inputSchema)
}

// TestModelConfig pins the contract between the schema a model advertises and
// the SDK type the framework deserializes into: every config form the action
// boundary accepts must convert to [anthropic.MessageNewParams], and the
// wrapper types the SDK uses for optional primitives must survive the trip.
// The schema is enforced on every request, so a form that the two disagree on
// is either a request rejected for no reason or a value silently dropped.
func TestModelConfig(t *testing.T) {
	desc := NewModel(anthropic.Client{}, "anthropic", "claude-opus-4-5", ai.ModelOptions{}).Desc()

	sampled := anthropic.MessageNewParams{
		Temperature: anthropic.Float(1.0),
		TopK:        anthropic.Int(1),
	}

	accepted := []struct {
		name   string
		config any
		want   anthropic.MessageNewParams
	}{
		{"struct config", sampled, sampled},
		{"pointer config", &sampled, sampled},
		{"map config", map[string]any{"temperature": 1.0, "top_k": 1}, sampled},
		{"empty map config", map[string]any{}, anthropic.MessageNewParams{}},
		{"nil config", nil, anthropic.MessageNewParams{}},
		// A typed nil marshals to JSON null, which the config slot tolerates.
		{"typed nil config", (*anthropic.MessageNewParams)(nil), anthropic.MessageNewParams{}},
		{"thinking union", map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 1024}}, anthropic.MessageNewParams{
			Thinking: anthropic.ThinkingConfigParamOfEnabled(1024),
		}},
	}
	for _, tt := range accepted {
		t.Run("accepts "+tt.name, func(t *testing.T) {
			if err := validateConfig(t, desc.InputSchema, tt.config); err != nil {
				t.Fatalf("config rejected at the action boundary: %v", err)
			}
			got, err := base.ConvertToExact[anthropic.MessageNewParams](tt.config)
			if err != nil {
				t.Fatalf("config accepted by the schema but not deserializable: %v", err)
			}
			// Compared as JSON: the SDK's param structs carry the raw
			// message they were deserialized from, so two values that send
			// the same request are not deeply equal.
			if diff := cmp.Diff(wireJSON(t, tt.want), wireJSON(t, got)); diff != "" {
				t.Errorf("deserialized config mismatch (-want +got):\n%s", diff)
			}
		})
	}

	rejected := []struct {
		name   string
		config any
	}{
		{"unknown field", map[string]any{"nope": 1}},
		// The SDK's wire names are snake_case; camelCase would deserialize to
		// nothing at all.
		{"camelCase field name", map[string]any{"maxTokens": 10}},
		{"mistyped value", map[string]any{"temperature": "hot"}},
		// The SDK's params struct decodes a bare scalar into a zero value
		// rather than failing, so the schema is what refuses one.
		{"scalar config", 123},
	}
	for _, tt := range rejected {
		t.Run("rejects "+tt.name, func(t *testing.T) {
			if err := validateConfig(t, desc.InputSchema, tt.config); err == nil {
				t.Error("expected the action boundary to reject this config")
			}
		})
	}

	// Another provider's config never reaches the model function. Its fields
	// are omitempty and the ones it sets overlap with Claude's, so it marshals
	// to an object the schema accepts; the deserialization step is what refuses
	// it, and without that the model function would run on an all-zero config.
	type otherProviderConfig struct {
		Temperature float64 `json:"temperature,omitempty"`
	}
	other := otherProviderConfig{Temperature: 1.0}
	if err := validateConfig(t, desc.InputSchema, other); err != nil {
		t.Fatalf("premise no longer holds, the schema rejects it on its own: %v", err)
	}
	if _, err := base.ConvertToExact[anthropic.MessageNewParams](other); !errors.Is(err, base.ErrTypeMismatch) {
		t.Errorf("ConvertToExact(%T) error = %v, want one wrapping ErrTypeMismatch", other, err)
	}
}

// TestModelLabel covers the label a model advertises: the caller's when set,
// otherwise one derived from the provider and the model name.
func TestModelLabel(t *testing.T) {
	tests := []struct {
		provider string
		name     string
		opts     ai.ModelOptions
		want     string
	}{
		{"anthropic", "claude-opus-4-5", ai.ModelOptions{Label: "Anthropic - Claude Opus 4.5"}, "Anthropic - Claude Opus 4.5"},
		{"anthropic", "claude-something-new", ai.ModelOptions{}, "Anthropic - claude-something-new"},
		{"vertexai", "claude-opus-4-5", ai.ModelOptions{Label: "Claude Opus 4.5"}, "Claude Opus 4.5"},
		{"vertexai", "claude-something-new", ai.ModelOptions{}, "Vertex AI - claude-something-new"},
	}

	for _, tt := range tests {
		t.Run(tt.provider+"/"+tt.name, func(t *testing.T) {
			desc := NewModel(anthropic.Client{}, tt.provider, tt.name, tt.opts).Desc()
			got := desc.Metadata["model"].(map[string]any)["label"]
			if got != tt.want {
				t.Errorf("label = %v, want %q", got, tt.want)
			}
		})
	}
}

func TestToAnthropicTools(t *testing.T) {
	tests := []struct {
		name        string
		tools       []*ai.ToolDefinition
		check       func(t *testing.T, got []anthropic.ToolUnionParam)
		expectedErr string
	}{
		{
			name: "valid tool",
			tools: []*ai.ToolDefinition{
				{
					Name:        "my-tool",
					Description: "my tool description",
				},
			},
			check: func(t *testing.T, got []anthropic.ToolUnionParam) {
				if len(got) != 1 {
					t.Fatalf("expected 1 tool, got %d", len(got))
				}
				tool := got[0].OfTool
				if tool.Name != "my-tool" {
					t.Errorf("got name %q, want %q", tool.Name, "my-tool")
				}
				if desc := tool.Description.Value; desc != "my tool description" {
					t.Errorf("got description %q, want %q", desc, "my tool description")
				}
				if !tool.Strict.Value {
					t.Error("expected Strict to be true")
				}
				if tool.InputSchema.ExtraFields["additionalProperties"] != false {
					t.Errorf("expected additionalProperties: false in ExtraFields, got %v", tool.InputSchema.ExtraFields["additionalProperties"])
				}
			},
		},
		{
			name: "valid tool with schema",
			tools: []*ai.ToolDefinition{
				{
					Name:        "weather",
					Description: "get weather",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type": "string",
							},
						},
						"required": []string{"location"},
					},
				},
			},
			check: func(t *testing.T, got []anthropic.ToolUnionParam) {
				if len(got) != 1 {
					t.Fatalf("expected 1 tool, got %d", len(got))
				}
				tool := got[0].OfTool
				if tool.Name != "weather" {
					t.Errorf("got name %q, want %q", tool.Name, "weather")
				}
				if tool.InputSchema.ExtraFields["additionalProperties"] != false {
					t.Errorf("expected additionalProperties: false in ExtraFields, got %v", tool.InputSchema.ExtraFields["additionalProperties"])
				}
			},
		},
		{
			name: "tool with strict opt-out omits the strict field",
			tools: []*ai.ToolDefinition{
				{
					Name:        "loose-tool",
					Description: "tool that opts out of strict",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"items": map[string]any{
								"type":     "array",
								"maxItems": 5,
							},
						},
					},
					Metadata: map[string]any{"strict": false},
				},
			},
			check: func(t *testing.T, got []anthropic.ToolUnionParam) {
				if len(got) != 1 {
					t.Fatalf("expected 1 tool, got %d", len(got))
				}
				tool := got[0].OfTool
				if tool.Strict.Valid() {
					t.Errorf("expected Strict to be omitted, got value=%v", tool.Strict.Value)
				}
				if _, ok := tool.InputSchema.ExtraFields["additionalProperties"]; ok {
					t.Errorf("expected additionalProperties to be absent, got %v", tool.InputSchema.ExtraFields["additionalProperties"])
				}
				// maxItems must be preserved when strict is off.
				items, _ := tool.InputSchema.Properties.(map[string]any)["items"].(map[string]any)
				if items["maxItems"] != float64(5) {
					t.Errorf("expected maxItems to be preserved, got %v", items["maxItems"])
				}
			},
		},
		{
			name: "tool with explicit strict=true still enforces additionalProperties",
			tools: []*ai.ToolDefinition{
				{
					Name:        "explicit-strict",
					Description: "tool that opts in to strict explicitly",
					InputSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"nested": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"x": map[string]any{"type": "string"},
								},
							},
						},
					},
					Metadata: map[string]any{"strict": true},
				},
			},
			check: func(t *testing.T, got []anthropic.ToolUnionParam) {
				if len(got) != 1 {
					t.Fatalf("expected 1 tool, got %d", len(got))
				}
				tool := got[0].OfTool
				if !tool.Strict.Valid() || !tool.Strict.Value {
					t.Errorf("expected Strict=true, got valid=%v value=%v", tool.Strict.Valid(), tool.Strict.Value)
				}
				if tool.InputSchema.ExtraFields["additionalProperties"] != false {
					t.Errorf("expected top-level additionalProperties: false, got %v", tool.InputSchema.ExtraFields["additionalProperties"])
				}
				// Nested object must also have additionalProperties: false.
				props, _ := tool.InputSchema.Properties.(map[string]any)
				nested, _ := props["nested"].(map[string]any)
				if nested["additionalProperties"] != false {
					t.Errorf("expected nested additionalProperties: false, got %v", nested["additionalProperties"])
				}
			},
		},
		{
			name: "empty tool name",
			tools: []*ai.ToolDefinition{
				{
					Name:        "",
					Description: "my tool description",
				},
			},
			expectedErr: "tool name is required",
		},
		{
			name: "invalid tool name",
			tools: []*ai.ToolDefinition{
				{
					Name:        "invalid tool name",
					Description: "my tool description",
				},
			},
			expectedErr: "tool name must match regex",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toAnthropicTools("anthropic", tt.tools)
			if checkError(t, err, tt.expectedErr) {
				return
			}
			if tt.check != nil {
				tt.check(t, got)
			}
		})
	}
}

// TestToAnthropicToolsVertex verifies the strict field is never set when the
// provider is "vertexai", regardless of the tool's strict metadata.
func TestToAnthropicToolsVertex(t *testing.T) {
	tests := []struct {
		name string
		tool *ai.ToolDefinition
	}{
		{
			name: "default strict is downgraded on vertex",
			tool: &ai.ToolDefinition{
				Name:        "default-tool",
				Description: "default strict behavior",
				InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
			},
		},
		{
			name: "explicit strict=true is downgraded on vertex",
			tool: &ai.ToolDefinition{
				Name:        "explicit-strict-vertex",
				Description: "user explicitly opted into strict but vertex does not support it",
				InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
				Metadata:    map[string]any{"strict": true},
			},
		},
		{
			name: "explicit strict=false is also downgraded on vertex",
			tool: &ai.ToolDefinition{
				Name:        "explicit-loose-vertex",
				Description: "user opted out and vertex doesn't support strict anyway",
				InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
				Metadata:    map[string]any{"strict": false},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toAnthropicTools("vertexai", []*ai.ToolDefinition{tt.tool})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != 1 {
				t.Fatalf("expected 1 tool, got %d", len(got))
			}
			tool := got[0].OfTool
			if tool.Strict.Valid() {
				t.Errorf("expected Strict to be unset on vertex, got value=%v", tool.Strict.Value)
			}
			if _, ok := tool.InputSchema.ExtraFields["additionalProperties"]; ok {
				t.Errorf("expected additionalProperties to be absent on vertex, got %v", tool.InputSchema.ExtraFields["additionalProperties"])
			}
		})
	}
}

func TestToAnthropicParts(t *testing.T) {
	tests := []struct {
		name        string
		parts       []*ai.Part
		expected    []anthropic.ContentBlockParamUnion
		expectedErr string
	}{
		{
			name: "text part",
			parts: []*ai.Part{
				ai.NewTextPart("hello"),
			},
			expected: []anthropic.ContentBlockParamUnion{
				anthropic.NewTextBlock("hello"),
			},
		},
		{
			name: "tool request part",
			parts: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{
					Ref:   "ref1",
					Input: map[string]any{"arg": "value"},
					Name:  "tool1",
				}),
			},
			expected: []anthropic.ContentBlockParamUnion{
				anthropic.NewToolUseBlock("ref1", map[string]any{"arg": "value"}, "tool1"),
			},
		},
		{
			name: "tool response part",
			parts: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{
					Ref:    "ref1",
					Output: map[string]any{"result": "ok"},
				}),
			},
			expected: []anthropic.ContentBlockParamUnion{
				anthropic.NewToolResultBlock("ref1", `{"result":"ok"}`, false),
			},
		},
		{
			name: "multipart tool response keeps output and content parts",
			parts: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{
					Ref:    "ref1",
					Output: map[string]any{"result": "ok"},
					Content: []*ai.Part{
						ai.NewTextPart("here is the chart"),
						ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgo="),
					},
				}),
			},
			expected: []anthropic.ContentBlockParamUnion{
				{OfToolResult: &anthropic.ToolResultBlockParam{
					ToolUseID: "ref1",
					IsError:   anthropic.Bool(false),
					Content: []anthropic.ToolResultBlockParamContentUnion{
						{OfText: &anthropic.TextBlockParam{Text: `{"result":"ok"}`}},
						{OfText: &anthropic.TextBlockParam{Text: "here is the chart"}},
						{OfImage: anthropic.NewImageBlockBase64("image/png", "iVBORw0KGgo=").OfImage},
					},
				}},
			},
		},
		{
			name: "multipart tool response without output omits the null text block",
			parts: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{
					Ref: "ref1",
					Content: []*ai.Part{
						ai.NewMediaPart("application/pdf", "data:application/pdf;base64,JVBERi0xLjQK"),
					},
				}),
			},
			expected: []anthropic.ContentBlockParamUnion{
				{OfToolResult: &anthropic.ToolResultBlockParam{
					ToolUseID: "ref1",
					IsError:   anthropic.Bool(false),
					Content: []anthropic.ToolResultBlockParamContentUnion{
						{OfDocument: anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{Data: "JVBERi0xLjQK"}).OfDocument},
					},
				}},
			},
		},
		{
			name: "tool response content with unsupported media type is rejected",
			parts: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{
					Ref:     "ref1",
					Content: []*ai.Part{ai.NewMediaPart("audio/mpeg", "data:audio/mpeg;base64,SUQzAw==")},
				}),
			},
			expectedErr: `unsupported media content type "audio/mpeg"`,
		},
		{
			name: "tool response content with unsupported part kind is rejected",
			parts: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{
					Ref:     "ref1",
					Content: []*ai.Part{ai.NewReasoningPart("thinking", nil)},
				}),
			},
			expectedErr: "unsupported part in tool response content",
		},
		{
			// Only reachable by hand-constructing the part: unmarshaling never
			// sets the kind without the payload.
			name:        "tool response part without a tool response is rejected",
			parts:       []*ai.Part{{Kind: ai.PartToolResponse}},
			expectedErr: "tool response part carries no tool response",
		},
		{
			name: "nil part in tool response content is rejected",
			parts: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{
					Ref:     "ref1",
					Content: []*ai.Part{nil},
				}),
			},
			expectedErr: "unsupported part in tool response content",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toAnthropicParts(tt.parts)
			if checkError(t, err, tt.expectedErr) {
				return
			}
			if !reflect.DeepEqual(tt.expected, got) {
				t.Errorf("toAnthropicParts() mismatch, got = %+v, want %+v", got, tt.expected)
			}
		})
	}
}

func TestToAnthropicRequest(t *testing.T) {
	tests := []modelRequestTestCase{
		{
			name: "simple request",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role:    ai.RoleUser,
						Content: []*ai.Part{ai.NewTextPart("hello")},
					},
				},
			},
			config: anthropic.MessageNewParams{MaxTokens: 10},
			expected: &anthropic.MessageNewParams{
				MaxTokens: 10,
				System:    []anthropic.TextBlockParam{},
				Messages: []anthropic.MessageParam{
					anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
				},
			},
		},
		{
			name: "with system prompt",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role:    ai.RoleSystem,
						Content: []*ai.Part{ai.NewTextPart("system prompt")},
					},
					{
						Role:    ai.RoleUser,
						Content: []*ai.Part{ai.NewTextPart("hello")},
					},
				},
			},
			config: anthropic.MessageNewParams{MaxTokens: 10},
			expected: &anthropic.MessageNewParams{
				MaxTokens: 10,
				System: []anthropic.TextBlockParam{
					{Text: "system prompt"},
				},
				Messages: []anthropic.MessageParam{
					anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
				},
			},
		},
		{
			// max_tokens is required by the API, so an unset value falls back to
			// DefaultMaxOutputTokens rather than failing the request.
			name: "no max tokens falls back to the default",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role:    ai.RoleUser,
						Content: []*ai.Part{ai.NewTextPart("hello")},
					},
				},
			},
			expected: &anthropic.MessageNewParams{
				MaxTokens: DefaultMaxOutputTokens,
				Messages: []anthropic.MessageParam{
					anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toAnthropicRequest("anthropic", tt.req, tt.config)
			if checkError(t, err, tt.expectedErr) {
				return
			}
			if got.MaxTokens != tt.expected.MaxTokens {
				t.Errorf("MaxTokens = %d, want %d", got.MaxTokens, tt.expected.MaxTokens)
			}
			if (len(tt.expected.System) > 0 || len(got.System) > 0) && !reflect.DeepEqual(tt.expected.System, got.System) {
				t.Errorf("System mismatch, got = %+v, want %+v", got.System, tt.expected.System)
			}
			if (len(tt.expected.Messages) > 0 || len(got.Messages) > 0) && !reflect.DeepEqual(tt.expected.Messages, got.Messages) {
				t.Errorf("Messages mismatch, got = %+v, want %+v", got.Messages, tt.expected.Messages)
			}
		})
	}
}

func TestToAnthropicRequest_StructuredOutput(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []string{"answer"},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role:    ai.RoleUser,
				Content: []*ai.Part{ai.NewTextPart("hello")},
			},
		},
		Output: &ai.ModelOutputConfig{
			Format:      "json",
			Schema:      schema,
			Constrained: true,
		},
	}

	got, err := toAnthropicRequest("anthropic", req, anthropic.MessageNewParams{MaxTokens: 100})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got.OutputConfig.Format.Schema == nil {
		t.Fatal("expected OutputConfig schema to be present")
	}

	// Verify the schema has additionalProperties: false added by enforceStrictSchema
	wantSchema := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []any{"answer"},
	}

	if diff := cmp.Diff(wantSchema, got.OutputConfig.Format.Schema); diff != "" {
		t.Errorf("OutputConfig schema mismatch (-want +got):\n%s", diff)
	}
}

// userRequest builds a minimal request carrying a single user message. The
// config travels beside the request now, so it is not part of it.
func userRequest() *ai.ModelRequest {
	return &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(ai.NewTextPart("hi"))},
	}
}

// wireJSON marshals v the way it would be sent to the Anthropic API.
func wireJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

func TestToAnthropicPartsMediaTypes(t *testing.T) {
	tests := []struct {
		name        string
		part        *ai.Part
		want        string
		expectedErr string
	}{
		{
			name: "png stays an image block",
			part: ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgo="),
			want: `"type":"image"`,
		},
		{
			// Anthropic only accepts images in an image block; a PDF sent as one
			// is rejected by the API.
			name: "pdf becomes a document block",
			part: ai.NewMediaPart("application/pdf", "data:application/pdf;base64,JVBERi0xLjQK"),
			want: `"type":"document"`,
		},
		{
			name: "plain text becomes a document block",
			part: ai.NewMediaPart("text/plain", "data:text/plain;base64,aGVsbG8="),
			want: `"type":"document"`,
		},
		{
			name:        "unsupported type is rejected",
			part:        ai.NewMediaPart("audio/mpeg", "data:audio/mpeg;base64,SUQzAw=="),
			expectedErr: `unsupported media content type "audio/mpeg"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toAnthropicParts([]*ai.Part{tt.part})
			if checkError(t, err, tt.expectedErr) {
				return
			}
			if wire := wireJSON(t, got); !strings.Contains(wire, tt.want) {
				t.Errorf("got %s, want it to contain %s", wire, tt.want)
			}
		})
	}
}

func TestToAnthropicPartsReasoningSignature(t *testing.T) {
	part := ai.NewReasoningPart("thinking...", []byte("sig-abc"))

	t.Run("in process", func(t *testing.T) {
		got, err := toAnthropicParts([]*ai.Part{part})
		if err != nil {
			t.Fatalf("toAnthropicParts: %v", err)
		}
		if wire := wireJSON(t, got); !strings.Contains(wire, `"signature":"sig-abc"`) {
			t.Errorf("signature missing: %s", wire)
		}
	})

	// A part read back from persisted history holds the signature as a base64
	// string rather than []byte. Dropping it makes Anthropic reject the
	// replayed thinking block.
	t.Run("after a JSON roundtrip", func(t *testing.T) {
		raw, err := json.Marshal(part)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var restored ai.Part
		if err := json.Unmarshal(raw, &restored); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		got, err := toAnthropicParts([]*ai.Part{&restored})
		if err != nil {
			t.Fatalf("toAnthropicParts: %v", err)
		}
		if wire := wireJSON(t, got); !strings.Contains(wire, `"signature":"sig-abc"`) {
			t.Errorf("signature lost across roundtrip: %s", wire)
		}
	})
}

func TestToAnthropicRequestPreservesConfig(t *testing.T) {
	// Server-side tools can only be expressed through the config, so the
	// genkit tool list must merge with them rather than replace them. The same
	// config value is reused across both calls, which also pins that a request
	// amends its own copy rather than the caller's.
	t.Run("config tools survive", func(t *testing.T) {
		config := anthropic.MessageNewParams{
			MaxTokens: 100,
			Tools: []anthropic.ToolUnionParam{
				{OfWebSearchTool20250305: &anthropic.WebSearchTool20250305Param{}},
			},
		}
		got, err := toAnthropicRequest("anthropic", userRequest(), config)
		if err != nil {
			t.Fatalf("toAnthropicRequest: %v", err)
		}
		if len(got.Tools) != 1 {
			t.Fatalf("got %d tools, want the config tool preserved", len(got.Tools))
		}

		req := userRequest()
		req.Tools = []*ai.ToolDefinition{{
			Name:        "my_tool",
			Description: "d",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		}}
		got, err = toAnthropicRequest("anthropic", req, config)
		if err != nil {
			t.Fatalf("toAnthropicRequest: %v", err)
		}
		if len(got.Tools) != 2 {
			t.Errorf("got %d tools, want the config tool plus the genkit tool", len(got.Tools))
		}
	})

	// The request's config is a shallow copy, so its Tools header still points
	// at the caller's backing array. Appending in place would write the genkit
	// tools into that array's spare capacity, which concurrent requests over a
	// hoisted config race over. Length alone does not catch it: an in-place
	// append leaves the caller's length untouched while overwriting the slots
	// past it.
	t.Run("appending tools leaves the caller's array alone", func(t *testing.T) {
		configTools := make([]anthropic.ToolUnionParam, 1, 4)
		configTools[0] = anthropic.ToolUnionParam{OfWebSearchTool20250305: &anthropic.WebSearchTool20250305Param{}}
		config := anthropic.MessageNewParams{MaxTokens: 100, Tools: configTools}

		req := userRequest()
		req.Tools = []*ai.ToolDefinition{{
			Name:        "my_tool",
			Description: "d",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		}}

		got, err := toAnthropicRequest("anthropic", req, config)
		if err != nil {
			t.Fatalf("toAnthropicRequest: %v", err)
		}
		if len(got.Tools) != 2 {
			t.Fatalf("got %d tools, want the config tool plus the genkit tool", len(got.Tools))
		}
		if &got.Tools[0] == &configTools[0] {
			t.Error("the request's tools share the caller's backing array")
		}
		for i, tool := range configTools[:cap(configTools)] {
			if i == 0 {
				continue
			}
			if tool.OfTool != nil {
				t.Errorf("the caller's spare capacity was written at index %d", i)
			}
		}
	})

	// Assigning a fresh OutputConfig would drop a config-provided effort.
	t.Run("output config effort survives structured output", func(t *testing.T) {
		config := anthropic.MessageNewParams{
			MaxTokens:    100,
			OutputConfig: anthropic.OutputConfigParam{Effort: anthropic.OutputConfigEffort("high")},
		}
		req := userRequest()
		req.Output = &ai.ModelOutputConfig{
			Format:      "json",
			Constrained: true,
			Schema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"a": map[string]any{"type": "string"}},
			},
		}
		got, err := toAnthropicRequest("anthropic", req, config)
		if err != nil {
			t.Fatalf("toAnthropicRequest: %v", err)
		}
		wire := wireJSON(t, got.OutputConfig)
		if !strings.Contains(wire, `"effort":"high"`) {
			t.Errorf("effort dropped: %s", wire)
		}
		if !strings.Contains(wire, `"json_schema"`) {
			t.Errorf("format missing: %s", wire)
		}
	})

	t.Run("config tool choice survives when unset", func(t *testing.T) {
		got, err := toAnthropicRequest("anthropic", userRequest(), anthropic.MessageNewParams{
			MaxTokens:  100,
			ToolChoice: anthropic.ToolChoiceUnionParam{OfTool: &anthropic.ToolChoiceToolParam{Name: "pinned"}},
		})
		if err != nil {
			t.Fatalf("toAnthropicRequest: %v", err)
		}
		if wire := wireJSON(t, got.ToolChoice); !strings.Contains(wire, "pinned") {
			t.Errorf("config tool choice dropped: %s", wire)
		}
	})

	// Anthropic rejects an empty content array, and an empty tools array
	// conflicts with a config-provided tool_choice.
	t.Run("no empty arrays on the wire", func(t *testing.T) {
		got, err := toAnthropicRequest("anthropic", userRequest(), anthropic.MessageNewParams{})
		if err != nil {
			t.Fatalf("toAnthropicRequest: %v", err)
		}
		wire := wireJSON(t, got)
		if strings.Contains(wire, `"tools":[]`) || strings.Contains(wire, `"system":[]`) {
			t.Errorf("empty arrays present: %s", wire)
		}
	})
}

func TestToAnthropicRequestToolChoice(t *testing.T) {
	tests := []struct {
		choice ai.ToolChoice
		want   string
	}{
		{ai.ToolChoiceAuto, `"type":"auto"`},
		{ai.ToolChoiceRequired, `"type":"any"`},
		{ai.ToolChoiceNone, `"type":"none"`},
	}

	for _, tt := range tests {
		t.Run(string(tt.choice), func(t *testing.T) {
			req := userRequest()
			req.ToolChoice = tt.choice
			got, err := toAnthropicRequest("anthropic", req, anthropic.MessageNewParams{})
			if err != nil {
				t.Fatalf("toAnthropicRequest: %v", err)
			}
			if wire := wireJSON(t, got.ToolChoice); !strings.Contains(wire, tt.want) {
				t.Errorf("got %s, want it to contain %s", wire, tt.want)
			}
		})
	}
}

// A message whose content is empty must be skipped rather than indexed into.
func TestToAnthropicRequestSkipsEmptyMessages(t *testing.T) {
	got, err := toAnthropicRequest("anthropic", &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleModel, Content: nil},
			ai.NewUserMessage(ai.NewTextPart("hi")),
		},
	}, anthropic.MessageNewParams{MaxTokens: 100})
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	if len(got.Messages) != 1 {
		t.Errorf("got %d messages, want the empty one skipped", len(got.Messages))
	}
}

func checkError(t *testing.T, err error, expectedErr string) bool {
	t.Helper()
	if expectedErr != "" {
		if err == nil {
			t.Errorf("expecting error containing %q, got nil", expectedErr)
		} else if !strings.Contains(err.Error(), expectedErr) {
			t.Errorf("expecting error to contain %q, but got: %q", expectedErr, err.Error())
		}
		return true
	}
	if err != nil {
		t.Errorf("expected no error, got: %v", err)
		return true
	}
	return false
}
