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

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/mcp"
)

func TestMCPToolAvailableToDotPrompt(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPromptFS(promptsFS))
	client, err := mcp.NewGenkitMCPClient(mcp.MCPClientOptions{
		Name: "demo",
		Stdio: &mcp.StdioConfig{
			Command: "go",
			Args:    []string{"run", "../server.go"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := client.Disconnect(); err != nil {
			t.Error(err)
		}
	})

	tools, err := client.GetActiveTools(ctx, g)
	if err != nil {
		t.Fatal(err)
	}
	for _, tool := range tools {
		genkit.RegisterAction(g, tool)
	}

	prompt := genkit.LookupPrompt(g, "encode")
	if prompt == nil {
		t.Fatal("could not find encode.prompt")
	}
	rendered, err := prompt.Render(ctx, map[string]any{"text": "Hello World"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(rendered.Tools, "demo_text_encode") {
		t.Fatalf("rendered prompt tools = %v, want demo_text_encode", rendered.Tools)
	}
	tool := genkit.LookupTool(g, "demo_text_encode")
	if tool == nil {
		t.Fatal("DotPrompt tool is not registered")
	}
	modelCalls := 0
	model := genkit.DefineModelAction(g, "test/mcp-tool-caller",
		&ai.ModelOptions{Supports: &ai.ModelSupports{Tools: true, Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, _ struct{}, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			modelCalls++
			if len(req.Tools) != 1 || req.Tools[0].Name != "demo_text_encode" {
				return nil, fmt.Errorf("model tools = %v, want demo_text_encode", req.Tools)
			}
			if modelCalls == 1 {
				return &ai.ModelResponse{
					Request: req,
					Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "demo_text_encode",
							Input: map[string]any{"text": "Hello World", "method": "base64_encode"},
						}),
					}},
				}, nil
			}
			for _, message := range req.Messages {
				if message.Role != ai.RoleTool {
					continue
				}
				for _, part := range message.Content {
					if part.ToolResponse == nil {
						continue
					}
					data, err := json.Marshal(part.ToolResponse.Output)
					if err != nil {
						return nil, err
					}
					if strings.Contains(string(data), "SGVsbG8gV29ybGQ=") {
						return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("encoded")}, nil
					}
				}
			}
			return nil, fmt.Errorf("model did not receive the encoded MCP tool response")
		})
	response, err := prompt.Execute(ctx,
		ai.WithInput(map[string]any{"text": "Hello World"}),
		ai.WithModel(model),
	)
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "encoded" || modelCalls != 2 {
		t.Fatalf("response = %q after %d model calls, want encoded after 2 calls", response.Text(), modelCalls)
	}
}
