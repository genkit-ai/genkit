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

package compat_oai_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// setupFunc creates a ModelGenerator for testing reasoning_content compatibility.
// Each setup function should check for required environment variables and skip
// the test if they are not configured.
type setupFunc func(t *testing.T) *compat_oai.ModelGenerator

var reasoningProviders = []struct {
	name  string
	setup setupFunc
	cb ai.ModelStreamCallback
}{
	{
		name:  "Moonshot",
		setup: setupMoonshotCompatibleTestClient,
	},
	{
		name:  "MoonshotStreaming",
		setup: setupMoonshotCompatibleTestClient,
		cb: func(context.Context, *ai.ModelResponseChunk) error {
			return nil
		},
	},
}

// TestToolCallReasoning verifies that the compat_oai plugin correctly handles
// reasoning_content for OpenAI-compatible backends that require it
func TestToolCallReasoning(t *testing.T) {
	for _, tc := range reasoningProviders {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.setup(t)
			resp1 := generateToolCall(t, g, tc.cb)
			resp2 := continueToolCall(t, g, resp1)
			assertFinalText(t, resp2)
		})
	}
}

// generateToolCall prompts LLM to make a tool request
// if cb is not nil - streaming mode is used
func generateToolCall(t *testing.T, g *compat_oai.ModelGenerator, cb ai.ModelStreamCallback) *ai.ModelResponse {
	t.Helper()

	resp, err := g.WithMessages([]*ai.Message{
		ai.NewUserTextMessage("Call the get_weather tool for Paris."),
	}).WithTools([]*ai.ToolDefinition{weatherTool}).Generate(t.Context(), ai.NewModelRequest(nil), cb)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	if got := len(resp.ToolRequests()); got != 1 {
		t.Fatalf("Expected 1 tool request, got %d", got)
	}
	if resp.Reasoning() == "" {
		t.Fatal("Expected reasoning on first response, got empty")
	}

	return resp
}

// continueToolCall appends tool response to the respose and calls LLM with it
func continueToolCall(t *testing.T, g *compat_oai.ModelGenerator, resp1 *ai.ModelResponse) *ai.ModelResponse {
	t.Helper()
	toolRequest := resp1.ToolRequests()[0]
	history := append(resp1.History(), &ai.Message{
		Role: ai.RoleTool,
		Content: []*ai.Part{
			ai.NewToolResponsePart(&ai.ToolResponse{
				Name:   toolRequest.Name,
				Ref:    toolRequest.Ref,
				Output: map[string]any{
					"temp_c": 21,
					"condition": "sunny",
				},
			}),
		},
	})
	resp, err := g.WithMessages(history).Generate(t.Context(), ai.NewModelRequest(nil), nil)
	if err != nil {
		t.Fatalf("rejected request with tool response: %v", err)
	}
	return resp
}

func assertFinalText(t *testing.T, resp *ai.ModelResponse) {
	t.Helper()
	if resp.Text() == "" {
		t.Fatalf("empty response from the model: %#v", resp)
	}
	t.Logf("model response: %v", resp.Text())
}

// setupMoonshotCompatibleTestClient creates a ModelGenerator for
// Moonshot-compatible backends.
//
// Required environment variables:
//   - MOONSHOT_API_KEY
//   - MOONSHOT_BASE_URL (e.g., "https://api.moonshot.cn/v1")
//   - MOONSHOT_MODEL (e.g., "kimi-k2.5")
//
// Tested with:
//   - Base URL: https://opencode.ai/zen/go/v1
//   - Model: kimi-k2.5
func setupMoonshotCompatibleTestClient(t *testing.T) *compat_oai.ModelGenerator {
	t.Helper()

	vars := []string{
		"MOONSHOT_API_KEY",
		"MOONSHOT_BASE_URL",
		"MOONSHOT_MODEL",
	}
	var missing []string

	for i, envVar := range vars {
		v := os.Getenv(envVar)
		if v == "" {
			missing = append(missing, envVar)
			continue
		}
		vars[i] = v
	}
	if len(missing) != 0 {
		t.Skipf("missing env vars: %s", strings.Join(missing, ", "))
	}

	client := openai.NewClient(
		option.WithAPIKey(vars[0]),
		option.WithBaseURL(vars[1]),
	)
	return compat_oai.NewModelGenerator(
		&client, vars[2])
}

var weatherTool = &ai.ToolDefinition{
	Name:        "get_weather",
	Description: "Get the current weather for a location. Always use this tool when asked about weather.",
	InputSchema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{
				"type":        "string",
				"description": "The city name to get weather for",
			},
		},
		"required": []string{"city"},
	},
}

// TestFragmentedToolCalls verifies that fragmented streaming tool calls
// are not emitted as duplicate/partial tool requests.
// This is a regression test for a bug where each chunk of a tool call
// would create a separate tool request part, resulting in many tool
// requests with empty names and refs.
func TestFragmentedToolCalls(t *testing.T) {
	g := setupMoonshotCompatibleTestClient(t)

	var chunkCount int
	var toolRequestCount int
	var emptyToolRequestCount int

	cb := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		chunkCount++
		for _, part := range chunk.Content {
			if part.IsToolRequest() {
				toolRequestCount++
				if part.ToolRequest.Name == "" || part.ToolRequest.Ref == "" {
					emptyToolRequestCount++
					t.Logf("Empty tool request found: name=%q ref=%q input=%q",
						part.ToolRequest.Name, part.ToolRequest.Ref, part.ToolRequest.Input)
				}
			}
		}
		return nil
	}

	resp, err := g.WithMessages([]*ai.Message{
		ai.NewUserTextMessage("Call the get_weather tool for Paris."),
	}).WithTools([]*ai.ToolDefinition{weatherTool}).Generate(t.Context(), ai.NewModelRequest(nil), cb)

	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	t.Logf("Total chunks: %d, Total tool request parts: %d, Empty tool requests: %d",
		chunkCount, toolRequestCount, emptyToolRequestCount)

	// We should NOT have any empty tool requests (name="" or ref="")
	if emptyToolRequestCount > 0 {
		t.Errorf("Got %d empty tool request parts - streaming tool calls are fragmented", emptyToolRequestCount)
	}

	// We should have exactly 1 tool request in the final response
	if got := len(resp.ToolRequests()); got != 1 {
		t.Errorf("Expected 1 tool request in final response, got %d", got)
	}

	// The tool request should have a valid name and ref
	toolReq := resp.ToolRequests()[0]
	if toolReq.Name == "" {
		t.Error("Tool request has empty name")
	}
	if toolReq.Ref == "" {
		t.Error("Tool request has empty ref")
	}
}
