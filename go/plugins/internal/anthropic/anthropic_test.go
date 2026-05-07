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
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
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
	expected    *anthropic.MessageNewParams
	expectedErr string
}

func TestAnthropicConfig(t *testing.T) {
	emptyConfig := anthropic.MessageNewParams{}
	expectedConfig := anthropic.MessageNewParams{
		Temperature: anthropic.Float(1.0),
		TopK:        anthropic.Int(1),
	}

	tests := []modelRequestTestCase{
		{
			name: "Input is anthropic.MessageNewParams struct",
			req: &ai.ModelRequest{
				Config: anthropic.MessageNewParams{
					Temperature: anthropic.Float(1.0),
					TopK:        anthropic.Int(1),
				},
			},
			expected: &expectedConfig,
		},
		{
			name: "Input is *anthropic.MessageNewParams struct",
			req: &ai.ModelRequest{
				Config: &anthropic.MessageNewParams{
					Temperature: anthropic.Float(1.0),
					TopK:        anthropic.Int(1),
				},
			},
			expected: &expectedConfig,
		},
		{
			name: "Input is map[string]any",
			req: &ai.ModelRequest{
				Config: map[string]any{
					"temperature": 1.0,
					"top_k":       1,
				},
			},
			expected: &expectedConfig,
		},
		{
			name: "Input is map[string]any (empty)",
			req: &ai.ModelRequest{
				Config: map[string]any{},
			},
			expected: &emptyConfig,
		},
		{
			name: "Input is nil",
			req: &ai.ModelRequest{
				Config: nil,
			},
			expected: &emptyConfig,
		},
		{
			name: "Input is an unexpected type",
			req: &ai.ModelRequest{
				Config: 123,
			},
			expectedErr: "unexpected config type: int",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := configFromRequest(tt.req)
			if checkError(t, err, tt.expectedErr) {
				return
			}
			if !reflect.DeepEqual(tt.expected, got) {
				t.Errorf("configFromRequest() got = %+v, want %+v", got, tt.expected)
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
					Strict: func() *bool { b := false; return &b }(),
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
				Config: map[string]any{
					"max_tokens": 10,
				},
			},
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
				Config: map[string]any{
					"max_tokens": 10,
				},
			},
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
			name: "no max tokens",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role:    ai.RoleUser,
						Content: []*ai.Part{ai.NewTextPart("hello")},
					},
				},
			},
			expectedErr: "maxTokens not set",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toAnthropicRequest("anthropic", tt.req)
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
		Config: map[string]any{
			"max_tokens": 100,
		},
		Output: &ai.ModelOutputConfig{
			Format:      "json",
			Schema:      schema,
			Constrained: true,
		},
	}

	got, err := toAnthropicRequest("anthropic", req)
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
