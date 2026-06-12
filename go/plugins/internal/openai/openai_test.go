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

package openai

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
)

func TestOpenAIConfig(t *testing.T) {
	emptyConfig := responses.ResponseNewParams{}
	expectedConfig := responses.ResponseNewParams{
		MaxOutputTokens:    openai.Int(128),
		PreviousResponseID: openai.String("resp_123"),
	}

	tests := []struct {
		name        string
		req         *ai.ModelRequest
		expected    *responses.ResponseNewParams
		expectedErr string
	}{
		{
			name: "Input is responses.ResponseNewParams struct",
			req: &ai.ModelRequest{
				Config: responses.ResponseNewParams{
					MaxOutputTokens:    openai.Int(128),
					PreviousResponseID: openai.String("resp_123"),
				},
			},
			expected: &expectedConfig,
		},
		{
			name: "Input is *responses.ResponseNewParams struct",
			req: &ai.ModelRequest{
				Config: &responses.ResponseNewParams{
					MaxOutputTokens:    openai.Int(128),
					PreviousResponseID: openai.String("resp_123"),
				},
			},
			expected: &expectedConfig,
		},
		{
			name: "Input is map[string]any",
			req: &ai.ModelRequest{
				Config: map[string]any{
					"max_output_tokens":    128,
					"previous_response_id": "resp_123",
				},
			},
			expected: &expectedConfig,
		},
		{
			name:     "Input is nil",
			req:      &ai.ModelRequest{},
			expected: &emptyConfig,
		},
		{
			name:        "Input is unexpected type",
			req:         &ai.ModelRequest{Config: 123},
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

func TestToOpenAITools(t *testing.T) {
	tests := []struct {
		name        string
		tools       []*ai.ToolDefinition
		check       func(t *testing.T, got []responses.ToolUnionParam)
		expectedErr string
	}{
		{
			name: "valid tool",
			tools: []*ai.ToolDefinition{{
				Name:        "weather",
				Description: "get weather",
			}},
			check: func(t *testing.T, got []responses.ToolUnionParam) {
				if len(got) != 1 {
					t.Fatalf("expected 1 tool, got %d", len(got))
				}
				tool := got[0].OfFunction
				if tool.Name != "weather" {
					t.Errorf("got name %q, want %q", tool.Name, "weather")
				}
				if !tool.Strict.Valid() || !tool.Strict.Value {
					t.Errorf("expected strict=true, got valid=%v value=%v", tool.Strict.Valid(), tool.Strict.Value)
				}
			},
		},
		{
			name: "tool with strict opt-out",
			tools: []*ai.ToolDefinition{{
				Name:        "loose",
				Description: "loose tool",
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
			}},
			check: func(t *testing.T, got []responses.ToolUnionParam) {
				tool := got[0].OfFunction
				if !tool.Strict.Valid() || tool.Strict.Value {
					t.Errorf("expected strict=false, got valid=%v value=%v", tool.Strict.Valid(), tool.Strict.Value)
				}
				props, _ := tool.Parameters["properties"].(map[string]any)
				items, _ := props["items"].(map[string]any)
				if items["maxItems"] != 5 {
					t.Errorf("expected maxItems preserved, got %v", items["maxItems"])
				}
			},
		},
		{
			name:        "empty tool name",
			tools:       []*ai.ToolDefinition{{Description: "missing name"}},
			expectedErr: "tool name is required",
		},
		{
			name: "invalid tool name",
			tools: []*ai.ToolDefinition{{
				Name:        "weather.tool",
				Description: "invalid name",
			}},
			expectedErr: "tool name must match regex",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toOpenAITools(tt.tools)
			if checkError(t, err, tt.expectedErr) {
				return
			}
			if tt.check != nil {
				tt.check(t, got)
			}
		})
	}
}

func TestToOpenAIRequest(t *testing.T) {
	tests := []struct {
		name        string
		req         *ai.ModelRequest
		expectedErr string
		check       func(t *testing.T, got *responses.ResponseNewParams)
	}{
		{
			name: "simple request",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("hello")},
				}},
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				if len(got.Input.OfInputItemList) != 1 {
					t.Fatalf("expected 1 input item, got %d", len(got.Input.OfInputItemList))
				}
				msg := got.Input.OfInputItemList[0].OfInputMessage
				if msg == nil {
					t.Fatal("expected input message")
				}
				if msg.Role != "user" {
					t.Errorf("got role %q, want user", msg.Role)
				}
				if len(msg.Content) != 1 {
					t.Fatalf("expected 1 message content item, got %d", len(msg.Content))
				}
				if text := msg.Content[0].GetText(); text == nil || *text != "hello" {
					t.Errorf("got text %v, want hello", text)
				}
			},
		},
		{
			name: "with system prompt and previous response id",
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
					"previous_response_id": "resp_prev",
				},
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				if !got.PreviousResponseID.Valid() || got.PreviousResponseID.Value != "resp_prev" {
					t.Fatalf("expected previous_response_id=resp_prev, got %+v", got.PreviousResponseID)
				}
				if len(got.Input.OfInputItemList) != 2 {
					t.Fatalf("expected 2 input items, got %d", len(got.Input.OfInputItemList))
				}
				if got.Input.OfInputItemList[0].OfInputMessage.Role != "system" {
					t.Errorf("expected first message to be system")
				}
			},
		},
		{
			name: "with tool choice required",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{{
					Role:    ai.RoleUser,
					Content: []*ai.Part{ai.NewTextPart("hello")},
				}},
				ToolChoice: ai.ToolChoiceRequired,
				Tools: []*ai.ToolDefinition{{
					Name:        "weather",
					Description: "get weather",
				}},
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				if !got.ToolChoice.OfToolChoiceMode.Valid() {
					t.Fatal("expected tool choice to be set")
				}
				if got.ToolChoice.OfToolChoiceMode.Value != responses.ToolChoiceOptionsRequired {
					t.Fatalf("expected required, got %v", got.ToolChoice.OfToolChoiceMode.Value)
				}
			},
		},
		{
			name: "with model history tool request and reasoning",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleModel,
						Content: []*ai.Part{
							ai.NewReasoningPart("thinking", nil),
							ai.NewToolRequestPart(&ai.ToolRequest{
								Ref:   "call_123",
								Name:  "weather",
								Input: map[string]any{"location": "Paris"},
							}),
						},
					},
				},
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				if len(got.Input.OfInputItemList) != 2 {
					t.Fatalf("expected 2 input items, got %d", len(got.Input.OfInputItemList))
				}
				if got.Input.OfInputItemList[0].OfReasoning == nil && got.Input.OfInputItemList[1].OfReasoning == nil {
					t.Fatal("expected reasoning item in history")
				}
				foundTool := false
				for _, item := range got.Input.OfInputItemList {
					if item.OfFunctionCall != nil {
						foundTool = true
						if item.OfFunctionCall.CallID != "call_123" {
							t.Fatalf("expected call id call_123, got %s", item.OfFunctionCall.CallID)
						}
					}
				}
				if !foundTool {
					t.Fatal("expected function call history item")
				}
			},
		},
		{
			name: "with data input and interleaved model history",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewDataPart("raw data"),
						},
					},
					{
						Role: ai.RoleModel,
						Content: []*ai.Part{
							ai.NewTextPart("before tool"),
							ai.NewToolResponsePart(&ai.ToolResponse{
								Ref:    "call_456",
								Output: map[string]any{"ok": true},
							}),
							ai.NewTextPart("after tool"),
							func() *ai.Part {
								part := ai.NewReasoningPart("chain", nil)
								part.Metadata["id"] = "rs_1"
								part.Metadata["encrypted_content"] = "enc"
								return part
							}(),
						},
					},
				},
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				items := got.Input.OfInputItemList
				if len(items) != 5 {
					t.Fatalf("expected 5 input items, got %d", len(items))
				}

				userMsg := items[0].OfInputMessage
				if userMsg == nil || len(userMsg.Content) != 1 || userMsg.Content[0].OfInputFile == nil {
					t.Fatal("expected first user content to be input_file")
				}
				if !userMsg.Content[0].OfInputFile.FileData.Valid() || userMsg.Content[0].OfInputFile.FileData.Value != "raw data" {
					t.Fatalf("unexpected file data: %+v", userMsg.Content[0].OfInputFile.FileData)
				}

				if items[1].OfOutputMessage == nil {
					t.Fatal("expected output message history item")
				}
				if text := items[1].OfOutputMessage.Content[0].GetText(); text == nil || *text != "before tool" {
					t.Fatalf("unexpected first output text: %v", text)
				}
				if items[2].OfFunctionCallOutput == nil {
					t.Fatal("expected function_call_output history item")
				}
				if items[2].OfFunctionCallOutput.CallID != "call_456" {
					t.Fatalf("unexpected tool response call id: %s", items[2].OfFunctionCallOutput.CallID)
				}
				if items[2].OfFunctionCallOutput.Status != "completed" {
					t.Fatalf("unexpected tool response status: %s", items[2].OfFunctionCallOutput.Status)
				}
				if items[3].OfOutputMessage == nil {
					t.Fatal("expected second output message history item")
				}
				if text := items[3].OfOutputMessage.Content[0].GetText(); text == nil || *text != "after tool" {
					t.Fatalf("unexpected second output text: %v", text)
				}
				if items[4].OfReasoning == nil {
					t.Fatal("expected reasoning history item")
				}
				if items[4].OfReasoning.ID != "rs_1" {
					t.Fatalf("unexpected reasoning id: %s", items[4].OfReasoning.ID)
				}
				if !items[4].OfReasoning.EncryptedContent.Valid() || items[4].OfReasoning.EncryptedContent.Value != "enc" {
					t.Fatalf("unexpected encrypted content: %+v", items[4].OfReasoning.EncryptedContent)
				}
			},
		},
		{
			name: "tool role response status is completed",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Ref:    "call_tool",
							Output: "ok",
						}),
					},
				}},
			},
			check: func(t *testing.T, got *responses.ResponseNewParams) {
				items := got.Input.OfInputItemList
				if len(items) != 1 || items[0].OfFunctionCallOutput == nil {
					t.Fatalf("expected function_call_output, got %+v", items)
				}
				if items[0].OfFunctionCallOutput.Status != "completed" {
					t.Fatalf("unexpected tool response status: %s", items[0].OfFunctionCallOutput.Status)
				}
			},
		},
		{
			name: "user message rejects tool request part",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleUser,
						Content: []*ai.Part{
							ai.NewToolRequestPart(&ai.ToolRequest{
								Ref:   "call_1",
								Name:  "weather",
								Input: map[string]any{"location": "Paris"},
							}),
						},
					},
				},
			},
			expectedErr: "unsupported part type in OpenAI input message",
		},
		{
			name: "tool role rejects non tool response part",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleTool,
						Content: []*ai.Part{
							ai.NewTextPart("bad"),
						},
					},
				},
			},
			expectedErr: "unsupported tool message part",
		},
		{
			name: "tool role marshal error includes call id",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleTool,
						Content: []*ai.Part{
							ai.NewToolResponsePart(&ai.ToolResponse{
								Ref:    "call_bad",
								Output: func() {},
							}),
						},
					},
				},
			},
			expectedErr: "failed to marshal tool response output for call_bad",
		},
		{
			name: "model history rejects custom part",
			req: &ai.ModelRequest{
				Messages: []*ai.Message{
					{
						Role: ai.RoleModel,
						Content: []*ai.Part{
							ai.NewTextPart("before"),
							ai.NewCustomPart(map[string]any{"type": "custom"}),
						},
					},
				},
			},
			expectedErr: "unsupported part type in OpenAI model history",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toOpenAIRequest(tt.req)
			if checkError(t, err, tt.expectedErr) {
				return
			}
			if tt.check != nil {
				tt.check(t, got)
			}
		})
	}
}

func TestToOpenAIRequest_StructuredOutput(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []string{"answer"},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{{
			Role:    ai.RoleUser,
			Content: []*ai.Part{ai.NewTextPart("hello")},
		}},
		Output: &ai.ModelOutputConfig{
			Format:      "json",
			Schema:      schema,
			Constrained: true,
		},
	}

	got, err := toOpenAIRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Text.Format.OfJSONSchema == nil {
		t.Fatal("expected JSON schema output format")
	}

	wantSchema := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []any{"answer"},
	}
	if diff := cmp.Diff(wantSchema, got.Text.Format.OfJSONSchema.Schema); diff != "" {
		t.Errorf("schema mismatch (-want +got):\n%s", diff)
	}
}

func TestHandleStreamEvent(t *testing.T) {
	ctx := context.Background()

	t.Run("text delta", func(t *testing.T) {
		state := newStreamState()
		event := mustStreamEvent(t, `{"type":"response.output_text.delta","content_index":0,"delta":"hello","item_id":"msg_1","output_index":0,"sequence_number":1}`)
		var got []*ai.ModelResponseChunk
		err := handleStreamEvent(ctx, state, event, func(_ context.Context, c *ai.ModelResponseChunk) error {
			got = append(got, c)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0].Text() != "hello" {
			t.Fatalf("unexpected chunks: %+v", got)
		}
	})

	t.Run("refusal delta", func(t *testing.T) {
		state := newStreamState()
		event := mustStreamEvent(t, `{"type":"response.refusal.delta","content_index":0,"delta":"blocked","item_id":"msg_1","output_index":0,"sequence_number":1}`)
		var text string
		err := handleStreamEvent(ctx, state, event, func(_ context.Context, c *ai.ModelResponseChunk) error {
			text += c.Text()
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if text != "blocked" {
			t.Fatalf("unexpected refusal text: %q", text)
		}
	})

	t.Run("reasoning summary done", func(t *testing.T) {
		state := newStreamState()
		event := mustStreamEvent(t, `{"type":"response.reasoning_summary.done","item_id":"rs_1","output_index":0,"sequence_number":1,"summary_index":0,"text":"think"}`)
		var reasoning string
		err := handleStreamEvent(ctx, state, event, func(_ context.Context, c *ai.ModelResponseChunk) error {
			reasoning += c.Content[0].Text
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if reasoning != "think" {
			t.Fatalf("unexpected reasoning text: %q", reasoning)
		}
	})

	t.Run("function call via item added and arguments done", func(t *testing.T) {
		state := newStreamState()
		added := mustStreamEvent(t, `{"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"weather","arguments":"","status":"in_progress"}}`)
		if err := handleStreamEvent(ctx, state, added, func(_ context.Context, _ *ai.ModelResponseChunk) error { return nil }); err != nil {
			t.Fatal(err)
		}

		done := mustStreamEvent(t, `{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"sequence_number":2,"arguments":"{\"location\":\"Paris\"}"}`)
		var got *ai.ToolRequest
		err := handleStreamEvent(ctx, state, done, func(_ context.Context, c *ai.ModelResponseChunk) error {
			got = c.Content[0].ToolRequest
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if got == nil {
			t.Fatal("expected tool request")
		}
		if got.Name != "weather" || got.Ref != "call_1" {
			t.Fatalf("unexpected tool request: %+v", got)
		}
		input, _ := got.Input.(map[string]any)
		if input["location"] != "Paris" {
			t.Fatalf("unexpected tool input: %+v", got.Input)
		}
	})

	t.Run("function call is emitted only once", func(t *testing.T) {
		state := newStreamState()
		added := mustStreamEvent(t, `{"type":"response.output_item.added","output_index":0,"sequence_number":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"weather","arguments":"","status":"in_progress"}}`)
		if err := handleStreamEvent(ctx, state, added, func(_ context.Context, _ *ai.ModelResponseChunk) error { return nil }); err != nil {
			t.Fatal(err)
		}
		doneArgs := mustStreamEvent(t, `{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"sequence_number":2,"arguments":"{\"location\":\"Paris\"}"}`)
		outputDone := mustStreamEvent(t, `{"type":"response.output_item.done","output_index":0,"sequence_number":3,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"weather","arguments":"{\"location\":\"Paris\"}","status":"completed"}}`)

		calls := 0
		cb := func(_ context.Context, _ *ai.ModelResponseChunk) error {
			calls++
			return nil
		}
		if err := handleStreamEvent(ctx, state, doneArgs, cb); err != nil {
			t.Fatal(err)
		}
		if err := handleStreamEvent(ctx, state, outputDone, cb); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("expected 1 tool call emission, got %d", calls)
		}
	})

	t.Run("stream error event returns error", func(t *testing.T) {
		state := newStreamState()
		event := mustStreamEvent(t, `{"type":"error","code":"bad_request","message":"boom","param":"model","sequence_number":1}`)
		err := handleStreamEvent(ctx, state, event, func(_ context.Context, _ *ai.ModelResponseChunk) error { return nil })
		if err == nil || !strings.Contains(err.Error(), "boom") {
			t.Fatalf("expected stream error, got %v", err)
		}
	})

	t.Run("text done is not duplicated after delta", func(t *testing.T) {
		state := newStreamState()
		delta := mustStreamEvent(t, `{"type":"response.output_text.delta","content_index":0,"delta":"hel","item_id":"msg_1","output_index":0,"sequence_number":1}`)
		done := mustStreamEvent(t, `{"type":"response.output_text.done","content_index":0,"text":"hello","item_id":"msg_1","output_index":0,"sequence_number":2}`)
		calls := 0
		cb := func(_ context.Context, _ *ai.ModelResponseChunk) error {
			calls++
			return nil
		}
		if err := handleStreamEvent(ctx, state, delta, cb); err != nil {
			t.Fatal(err)
		}
		if err := handleStreamEvent(ctx, state, done, cb); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("expected 1 text emission, got %d", calls)
		}
	})

	t.Run("reasoning done is not duplicated after summary delta", func(t *testing.T) {
		state := newStreamState()
		delta := mustStreamEvent(t, `{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"sequence_number":1,"summary_index":0,"delta":"rea"}`)
		done := mustStreamEvent(t, `{"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"sequence_number":2,"summary_index":0,"text":"reason"}`)
		calls := 0
		cb := func(_ context.Context, _ *ai.ModelResponseChunk) error {
			calls++
			return nil
		}
		if err := handleStreamEvent(ctx, state, delta, cb); err != nil {
			t.Fatal(err)
		}
		if err := handleStreamEvent(ctx, state, done, cb); err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("expected 1 reasoning emission, got %d", calls)
		}
	})
}

func TestToGenkitResponse(t *testing.T) {
	t.Run("maps finish reason and usage", func(t *testing.T) {
		resp := mustResponse(t, `{
			"id":"resp_1",
			"object":"response",
			"model":"gpt-5.4",
			"status":"incomplete",
			"incomplete_details":{"reason":"content_filter"},
			"output":[
				{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[],"logprobs":[]}]},
				{"id":"fc_1","type":"function_call","call_id":"call_1","name":"weather","arguments":"{\"location\":\"Paris\"}","status":"completed"},
				{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"reason"}],"encrypted_content":"enc","status":"completed"}
			],
			"parallel_tool_calls":true,
			"tool_choice":"auto",
			"tools":[],
			"text":{"format":{"type":"text"}},
			"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":3},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2},"total_tokens":15}
		}`)

		got, err := toGenkitResponse(resp)
		if err != nil {
			t.Fatal(err)
		}
		if got.FinishReason != ai.FinishReasonBlocked {
			t.Fatalf("expected blocked finish reason, got %q", got.FinishReason)
		}
		if got.Usage == nil || got.Usage.CachedContentTokens != 3 || got.Usage.ThoughtsTokens != 2 || got.Usage.TotalTokens != 15 {
			t.Fatalf("unexpected usage: %+v", got.Usage)
		}
		if got.Text() != "hello" {
			t.Fatalf("unexpected text: %q", got.Text())
		}
		if len(got.Message.Content) != 3 {
			t.Fatalf("expected 3 parts, got %d", len(got.Message.Content))
		}
		if !got.Message.Content[1].IsToolRequest() || got.Message.Content[1].ToolRequest.Name != "weather" {
			t.Fatalf("unexpected tool part: %+v", got.Message.Content[1])
		}
		if !got.Message.Content[2].IsReasoning() {
			t.Fatalf("unexpected reasoning part: %+v", got.Message.Content[2])
		}
		if got.Message.Content[2].Metadata["encrypted_content"] != "enc" {
			t.Fatalf("unexpected reasoning metadata: %+v", got.Message.Content[2].Metadata)
		}
	})
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

func mustStreamEvent(t *testing.T, raw string) responses.ResponseStreamEventUnion {
	t.Helper()
	var event responses.ResponseStreamEventUnion
	if err := json.Unmarshal([]byte(raw), &event); err != nil {
		t.Fatalf("failed to unmarshal stream event: %v", err)
	}
	return event
}

func mustResponse(t *testing.T, raw string) *responses.Response {
	t.Helper()
	var resp responses.Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	return &resp
}
