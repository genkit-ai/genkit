package compat_oai

import (
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/respjson"
)

func TestStreamResponseCollectorPreservesReasoning(t *testing.T) {
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
		modelChunk, err := collector.AddChunk(chunk)
		if err != nil {
			t.Fatalf("AddChunk() error: %v", err)
		}
		if modelChunk != nil && modelChunk.Reasoning() != "" {
			reasoningChunks = append(reasoningChunks, modelChunk.Reasoning())
		}
	}

	resp, err := collector.Response()
	if err != nil {
		t.Fatalf("Response() error: %v", err)
	}

	if got, want := strings.Join(reasoningChunks, ""), "Need location lookup. Calling the tool."; got != want {
		t.Fatalf("stream reasoning mismatch: got %q want %q", got, want)
	}
	if got, want := resp.Reasoning(), "Need location lookup. Calling the tool."; got != want {
		t.Fatalf("final reasoning mismatch: got %q want %q", got, want)
	}
	var reasoningParts int
	for _, part := range resp.Message.Content {
		if part.IsReasoning() {
			reasoningParts++
		}
	}
	if reasoningParts != 1 {
		t.Fatalf("expected 1 reasoning part, got %d", reasoningParts)
	}
	toolRequests := resp.ToolRequests()
	if len(toolRequests) != 1 {
		t.Fatalf("expected 1 tool request, got %d", len(toolRequests))
	}
	if got, want := toolRequests[0].Name, "get_weather"; got != want {
		t.Fatalf("tool name mismatch: got %q want %q", got, want)
	}
}

func TestStreamResponseCollectorResponsePrefersAccumulatedCompletionReasoning(t *testing.T) {
	collector := &streamResponseCollector{}
	collector.accumulator.ChatCompletion = openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{},
			},
		},
	}
	collector.accumulator.ChatCompletion.Choices[0].Message.JSON.ExtraFields = map[string]respjson.Field{
		"reasoning_content": respjson.NewField(`"from completion"`),
	}
	collector.reasoning.content.WriteString("from chunks")
	collector.reasoning.key = "reasoning_content"

	resp, err := collector.Response()
	if err != nil {
		t.Fatalf("Response() error: %v", err)
	}

	if got, want := resp.Reasoning(), "from completion"; got != want {
		t.Fatalf("final reasoning mismatch: got %q want %q", got, want)
	}
}

func TestStreamResponseCollectorReturnsReasoningParseErrors(t *testing.T) {
	collector := &streamResponseCollector{}

	chunk := openai.ChatCompletionChunk{
		ID: "chatcmpl-test",
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index: 0,
				Delta: openai.ChatCompletionChunkChoiceDelta{},
			},
		},
	}
	chunk.Choices[0].Delta.JSON.ExtraFields = map[string]respjson.Field{
		"reasoning_content": respjson.NewField(`{"bad":"json"}`),
	}

	_, err := collector.AddChunk(chunk)
	if err == nil {
		t.Fatal("AddChunk() error = nil, want parse failure")
	}
	if !strings.Contains(err.Error(), "could not parse reasoning field") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestFragmentedToolCalls verifies that fragmented tool calls in streaming mode
// do not create multiple partial/empty tool request parts.
//
// This is a regression test for a bug where each chunk of a tool call would
// create a separate tool request part, resulting in many tool requests with
// empty names and refs.
func TestFragmentedToolCalls(t *testing.T) {
	collector := &streamResponseCollector{}

	// Simulate a tool call arriving in fragments across multiple chunks
	// This mimics how OpenAI-compatible APIs actually stream tool calls
	chunks := []openai.ChatCompletionChunk{
		// First chunk: ID and name arrive, no arguments yet
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{
							{
								Index: 0,
								ID:    "call_abc123",
								Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
									Name:      "get_weather",
									Arguments: "",
								},
							},
						},
					},
				},
			},
		},
		// Second chunk: partial arguments (empty name/id as they were already sent)
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{
							{
								Index: 0,
								ID:    "", // Empty in subsequent chunks
								Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
									Name:      "", // Empty in subsequent chunks
									Arguments: `{"city": "`,
								},
							},
						},
					},
				},
			},
		},
		// Third chunk: rest of arguments
		{
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Delta: openai.ChatCompletionChunkChoiceDelta{
						ToolCalls: []openai.ChatCompletionChunkChoiceDeltaToolCall{
							{
								Index: 0,
								ID:    "",
								Function: openai.ChatCompletionChunkChoiceDeltaToolCallFunction{
									Name:      "",
									Arguments: `Paris"}`,
								},
							},
						},
					},
				},
			},
		},
	}

	var totalToolRequestParts int
	var emptyNameToolRequests int
	var emptyRefToolRequests int

	for i, chunk := range chunks {
		modelChunk, err := collector.AddChunk(chunk)
		if err != nil {
			t.Fatalf("AddChunk %d failed: %v", i, err)
		}
		if modelChunk != nil {
			for _, part := range modelChunk.Content {
				if part.IsToolRequest() {
					totalToolRequestParts++
					t.Logf("Chunk %d: ToolRequest(name=%q, ref=%q, input=%q)",
						i, part.ToolRequest.Name, part.ToolRequest.Ref, part.ToolRequest.Input)
					if part.ToolRequest.Name == "" {
						emptyNameToolRequests++
					}
					if part.ToolRequest.Ref == "" {
						emptyRefToolRequests++
					}
				}
			}
		}
	}

	t.Logf("Total tool request parts emitted: %d", totalToolRequestParts)
	t.Logf("Empty name tool requests: %d", emptyNameToolRequests)
	t.Logf("Empty ref tool requests: %d", emptyRefToolRequests)

	// BEFORE THE FIX: This creates 2 tool request parts with empty names/refs
	// (chunks 2 and 3 have empty name/ref because they are fragments)
	// AFTER THE FIX: Should create 0-1 tool request parts with no empty names/refs
	if emptyNameToolRequests > 0 {
		t.Errorf("Got %d tool requests with empty names - fragmented tool calls not properly handled", emptyNameToolRequests)
	}
	if emptyRefToolRequests > 0 {
		t.Errorf("Got %d tool requests with empty refs - fragmented tool calls not properly handled", emptyRefToolRequests)
	}

	// Check the final response
	resp, err := collector.Response()
	if err != nil {
		t.Fatalf("Response() failed: %v", err)
	}

	toolReqs := resp.ToolRequests()
	t.Logf("Final response has %d tool requests", len(toolReqs))

	// We should have exactly 1 tool request
	if len(toolReqs) != 1 {
		t.Errorf("Expected 1 tool request in final response, got %d", len(toolReqs))
	}

	if len(toolReqs) > 0 {
		toolReq := toolReqs[0]
		if toolReq.Name != "get_weather" {
			t.Errorf("Expected tool name 'get_weather', got %q", toolReq.Name)
		}
		if toolReq.Ref != "call_abc123" {
			t.Errorf("Expected tool ref 'call_abc123', got %q", toolReq.Ref)
		}
	}
}
