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

package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

func TestToOpenAIResponseParams_SystemMessage(t *testing.T) {
	msg := &ai.Message{
		Role:    ai.RoleSystem,
		Content: []*ai.Part{ai.NewTextPart("system instruction")},
	}
	req := &ai.ModelRequest{
		Messages: []*ai.Message{msg},
	}

	var inputItems []responses.ResponseInputItemUnionParam
	for _, m := range req.Messages {
		items, err := toOpenAIInputItems(m)
		if err != nil {
			t.Errorf("error converting messages to input items: %v", err)
		}
		inputItems = append(inputItems, items...)
	}
	if len(inputItems) == 0 {
		t.Errorf("expecting input items, got: 0")
	}

	params := responses.ResponseNewParams{}
	params.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: inputItems,
	}
	for _, m := range params.Input.OfInputItemList {
		if *m.GetRole() != "developer" {
			t.Errorf("want system role: developer, got: %s", *m.GetRole())
		}
	}
}

func TestToOpenAIInputItems_JSON(t *testing.T) {
	tests := []struct {
		name string
		msg  *ai.Message
		want []string // substrings to match in JSON
	}{
		{
			name: "user text message",
			msg: &ai.Message{
				Role:    ai.RoleUser,
				Content: []*ai.Part{ai.NewTextPart("user query")},
			},
			want: []string{`"role":"user"`, `"type":"input_text"`, `"text":"user query"`},
		},
		{
			name: "model text message",
			msg: &ai.Message{
				Role:    ai.RoleModel,
				Content: []*ai.Part{ai.NewTextPart("model response")},
			},
			want: []string{`"role":"assistant"`, `"type":"output_text"`, `"text":"model response"`, `"status":"completed"`},
		},
		{
			name: "tool request",
			msg: &ai.Message{
				Role: ai.RoleModel,
				Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
					Name:  "myTool",
					Ref:   "call_123",
					Input: map[string]string{"arg": "val"},
				})},
			},
			want: []string{`"type":"function_call"`, `"name":"myTool"`, `"call_id":"call_123"`, `"arguments":"{\"arg\":\"val\"}"`},
		},
		{
			name: "tool response",
			msg: &ai.Message{
				Role: ai.RoleTool,
				Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{
					Name:   "myTool",
					Ref:    "call_123",
					Output: map[string]string{"res": "ok"},
				})},
			},
			want: []string{`"type":"function_call_output"`, `"call_id":"call_123"`, `"output":"{\"res\":\"ok\"}"`},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			items, err := toOpenAIInputItems(tc.msg)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			b, err := json.Marshal(items)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			jsonStr := string(b)

			for _, w := range tc.want {
				if !strings.Contains(jsonStr, w) {
					t.Errorf("JSON output missing %q. Got: %s", w, jsonStr)
				}
			}
		})
	}
}

func TestToOpenAITools(t *testing.T) {
	tests := []struct {
		name      string
		tools     []*ai.ToolDefinition
		wantCount int
		check     func(*testing.T, []responses.ToolUnionParam)
	}{
		{
			name: "basic function tool",
			tools: []*ai.ToolDefinition{
				{
					Name:        "myTool",
					Description: "does something",
					InputSchema: map[string]any{"type": "object"},
				},
			},
			wantCount: 1,
			check: func(t *testing.T, got []responses.ToolUnionParam) {
				tool := got[0]
				// We need to marshal to check fields since they are hidden in UnionParam
				b, _ := json.Marshal(tool)
				s := string(b)
				if !strings.Contains(s, `"name":"myTool"`) {
					t.Errorf("missing name: %s", s)
				}
				if !strings.Contains(s, `"type":"function"`) {
					t.Errorf("missing type function: %s", s)
				}
			},
		},
		{
			name: "empty name tool ignored",
			tools: []*ai.ToolDefinition{
				{Name: ""},
			},
			wantCount: 0,
			check:     func(t *testing.T, got []responses.ToolUnionParam) {},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := toOpenAITools(tc.tools)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != tc.wantCount {
				t.Errorf("got %d tools, want %d", len(got), tc.wantCount)
			}
			tc.check(t, got)
		})
	}
}

func TestTranslateResponse(t *testing.T) {
	// this is a workaround function to bypass Union types used in the openai-go SDK
	createResponse := func(jsonStr string) *responses.Response {
		var r responses.Response
		if err := json.Unmarshal([]byte(jsonStr), &r); err != nil {
			t.Fatalf("failed to create mock response: %v", err)
		}
		return &r
	}

	tests := []struct {
		name       string
		respJSON   string
		wantReason ai.FinishReason
		check      func(*testing.T, *ai.ModelResponse)
	}{
		{
			name: "text response completed",
			respJSON: `{
				"id": "resp_1",
				"status": "completed",
				"usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
				"output": [
					{
						"type": "message",
						"role": "assistant",
						"content": [{"type": "output_text", "text": "Hello world"}]
					}
				]
			}`,
			wantReason: ai.FinishReasonStop,
			check: func(t *testing.T, m *ai.ModelResponse) {
				if len(m.Message.Content) != 1 {
					t.Fatalf("got %d parts, want 1", len(m.Message.Content))
				}
				if got := m.Message.Content[0].Text; got != "Hello world" {
					t.Errorf("got text %q, want 'Hello world'", got)
				}
				if m.Usage.TotalTokens != 30 {
					t.Errorf("got usage %d, want 30", m.Usage.TotalTokens)
				}
			},
		},
		{
			name: "incomplete response",
			respJSON: `{
				"id": "resp_2",
				"status": "incomplete",
				"output": []
			}`,
			wantReason: ai.FinishReasonLength, // mapped from Incomplete
			check:      func(t *testing.T, m *ai.ModelResponse) {},
		},
		{
			name: "refusal response",
			respJSON: `{
				"id": "resp_3",
				"status": "completed",
				"output": [
					{
						"type": "message",
						"role": "assistant",
						"content": [{"type": "refusal", "refusal": "I cannot do that"}]
					}
				]
			}`,
			wantReason: ai.FinishReasonBlocked,
			check: func(t *testing.T, m *ai.ModelResponse) {
				if m.FinishMessage != "I cannot do that" {
					t.Errorf("got FinishMessage %q, want 'I cannot do that'", m.FinishMessage)
				}
			},
		},
		{
			name: "tool call",
			respJSON: `{
				"id": "resp_4",
				"status": "completed",
				"output": [
					{
						"type": "function_call",
						"call_id": "call_abc",
						"name": "weather",
						"arguments": "{\"loc\":\"SFO\"}"
					}
				]
			}`,
			wantReason: ai.FinishReasonStop,
			check: func(t *testing.T, m *ai.ModelResponse) {
				if len(m.Message.Content) != 1 {
					t.Fatalf("got %d parts, want 1", len(m.Message.Content))
				}
				p := m.Message.Content[0]
				if !p.IsToolRequest() {
					t.Fatalf("expected tool request part")
				}
				if p.ToolRequest.Name != "weather" {
					t.Errorf("got tool name %q, want 'weather'", p.ToolRequest.Name)
				}
				args := p.ToolRequest.Input.(map[string]any)
				if args["loc"] != "SFO" {
					t.Errorf("got arg loc %v, want 'SFO'", args["loc"])
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := createResponse(tc.respJSON)
			got, err := translateResponse(r)
			if err != nil {
				t.Fatalf("translateResponse() unexpected error: %v", err)
			}
			if got.FinishReason != tc.wantReason {
				t.Errorf("got reason %v, want %v", got.FinishReason, tc.wantReason)
			}
			tc.check(t, got)
		})
	}
}

func TestConfigFromRequest(t *testing.T) {
	tests := []struct {
		name    string
		input   any
		wantErr bool
		check   func(*testing.T, *responses.ResponseNewParams)
	}{
		{
			name: "struct config",
			input: responses.ResponseNewParams{
				Model: shared.ResponsesModel("gpt-4o"),
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				if got.Model != shared.ResponsesModel("gpt-4o") {
					t.Errorf("got model %v, want %v", got.Model, shared.ResponsesModel("gpt-4o"))
				}
			},
		},
		{
			name: "map config",
			input: map[string]any{
				"model": "gpt-4o",
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				if got.Model != "gpt-4o" {
					t.Errorf("got model %v, want gpt-4o", got.Model)
				}
			},
		},
		{
			name:    "invalid type",
			input:   "some string",
			wantErr: true,
			check:   func(t *testing.T, got *responses.ResponseNewParams) {},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := configFromRequest(tc.input)
			if (err != nil) != tc.wantErr {
				t.Errorf("configFromRequest() error = %v, wantErr %v", err, tc.wantErr)
				return
			}
			if !tc.wantErr {
				tc.check(t, got)
			}
		})
	}
}
