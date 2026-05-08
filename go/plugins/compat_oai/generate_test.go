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

package compat_oai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/respjson"
)

func TestStreamResponseCollector(t *testing.T) {
	collector := &streamResponseCollector{}

	chunks := []openai.ChatCompletionChunk{
		{
			ID: "chatcmpl-test",
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoiceDelta{
						Content: "Checking weather. ",
						ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{
							{
								Index: 0,
								ID:    "call_1",
								Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
									Name:      "get_weather",
									Arguments: `{"city":"Par`,
								},
							},
						},
					},
				},
			},
		},
		{
			ID: "chatcmpl-test",
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index:        0,
					FinishReason: "tool_calls",
					Delta: openai.ChatCompletionChunkChoiceDelta{
						ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{
							{
								Index: 0,
								Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
									Arguments: `is"}`,
								},
							},
						},
					},
				},
			},
		},
	}
	chunks[0].Choices[0].Delta.JSON.ExtraFields = map[string]respjson.Field{
		"reasoning_content": respjson.NewField(`"Need location lookup. "`),
	}
	chunks[1].Choices[0].Delta.JSON.ExtraFields = map[string]respjson.Field{
		"reasoning_content": respjson.NewField(`"Calling the tool."`),
	}

	var reasoningChunks []string
	for _, chunk := range chunks {
		modelChunk, ok := collector.AddChunk(chunk)
		if ok && modelChunk != nil && modelChunk.Reasoning() != "" {
			reasoningChunks = append(reasoningChunks, modelChunk.Reasoning())
		}
	}

	resp, err := collector.ToModelResponse()
	if err != nil {
		t.Fatalf("ToModelResponse() error: %v", err)
	}

	if got, want := strings.Join(reasoningChunks, ""), "Need location lookup. Calling the tool."; got != want {
		t.Errorf("stream reasoning mismatch: got %q want %q", got, want)
	}

	if strings.Contains(resp.Text(), resp.Reasoning()) {
		t.Errorf("response text contains reasoning")
	}

	var reasoningParts int
	for _, part := range resp.Message.Content {
		if part.IsReasoning() {
			reasoningParts++
		}
	}
	if reasoningParts != 1 {
		t.Errorf("expected 1 reasoning part, got %d", reasoningParts)
	}
	if got := len(resp.ToolRequests()); got != 1 {
		t.Errorf("expected 1 tool request, got %d", got)
	}
	if got, want := resp.ToolRequests()[0].Name, "get_weather"; got != want {
		t.Errorf("tool name mismatch: got %q want %q", got, want)
	}
}

// TestDuplicateToolCallIDsInHistory verifies that model messages containing
// partial tool requests (with empty ref/name) do not create duplicate or
// empty tool call IDs when converted back to OpenAI format.
// This reproduces the bug where accumulated streaming chunks create
// partial tool requests that result in "tool call id duplicated" errors.
func TestDuplicateToolCallIDsInHistory(t *testing.T) {
	var requests [][]byte
	var mu sync.Mutex

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		requests = append(requests, body)
		mu.Unlock()

		isStream := strings.Contains(string(body), `"stream":true`)
		if isStream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			chunks := []string{
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n",
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}` + "\n\n",
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"65","type":"function","function":{"name":"get_weather"}}]},"finish_reason":null}]}` + "\n\n",
				`data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"kimi-k2.5","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n",
				"data: [DONE]\n\n",
			}
			for _, chunk := range chunks {
				w.Write([]byte(chunk))
				w.(http.Flusher).Flush()
			}
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"id":"chatcmpl-123","object":"chat.completion","created":1694268190,"model":"kimi-k2.5","choices":[{"index":0,"message":{"role":"assistant","content":"The weather in Paris is sunny."},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":10,"total_tokens":20}}`))
		}
	}))
	defer ts.Close()

	client := openai.NewClient(option.WithBaseURL(ts.URL), option.WithAPIKey("test"))
	g := NewModelGenerator(&client, "kimi-k2.5")

	// Simulate a history message that contains partial tool request fragments
	// (as would happen if a consumer appended streaming chunks to history)
	historyMsg := &ai.Message{
		Role: ai.RoleModel,
		Content: []*ai.Part{
			ai.NewTextPart("Hello"),
			// Complete tool request
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "get_weather", Ref: "65", Input: map[string]any{"city": "Paris"}}),
			// Partial fragments that were emitted by raw streaming deltas
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "", Ref: "", Input: `{"city": `}),
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "", Ref: "", Input: `"Paris"}`}),
			// Another duplicate of the complete one
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "get_weather", Ref: "65", Input: map[string]any{"city": "Paris"}}),
		},
	}

	_, err := g.WithMessages([]*ai.Message{
		ai.NewUserTextMessage("What's the weather?"),
		historyMsg,
	}).WithTools([]*ai.ToolDefinition{{
		Name:        "get_weather",
		Description: "Get weather",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		},
	}}).Generate(context.Background(), ai.NewModelRequest(nil), nil)

	if err != nil {
		t.Fatalf("generation failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if len(requests) != 1 {
		t.Fatalf("expected 1 request, got %d", len(requests))
	}

	var reqBody map[string]any
	if err := json.Unmarshal(requests[0], &reqBody); err != nil {
		t.Fatalf("failed to parse request: %v", err)
	}

	messages, _ := reqBody["messages"].([]any)
	for _, msg := range messages {
		m, _ := msg.(map[string]any)
		if m["role"] != "assistant" {
			continue
		}
		toolCalls, _ := m["tool_calls"].([]any)
		ids := make(map[string]int)
		for _, tc := range toolCalls {
			tcMap, _ := tc.(map[string]any)
			id, _ := tcMap["id"].(string)
			ids[id]++
			if ids[id] > 1 {
				t.Errorf("duplicate tool call id found: %q", id)
			}
			if id == "" {
				t.Errorf("empty tool call id found")
			}
		}
		if len(toolCalls) != 1 {
			t.Errorf("expected 1 tool call, got %d", len(toolCalls))
		}
	}
}
