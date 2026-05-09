// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//
// Run with: export GOOGLE_AI_API_KEY=your_key && go run client.go
package main

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/mcp"
)

func client() {
	ctx := context.Background()

	// Initialize Genkit with Google AI
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Connect to server
	client, err := mcp.NewGenkitMCPClient(mcp.MCPClientOptions{
		Name: "client",
		Stdio: &mcp.StdioConfig{
			Command: "go",
			Args:    []string{"run", "server.go"},
		},
	})
	if err != nil {
		logger.FromContext(ctx).Error("Failed to connect to MCP server", "error", err)
		return
	}
	defer client.Disconnect()

	// Import tools
	tools, err := client.GetActiveTools(ctx, g)
	if err != nil {
		logger.FromContext(ctx).Error("Failed to get tools from MCP server", "error", err)
		return
	}

	logger.FromContext(ctx).Info("Connected to MCP server", "tools", getToolNames(tools))

	// Convert to ToolRef
	var toolRefs []ai.ToolRef
	for _, tool := range tools {
		toolRefs = append(toolRefs, tool)
	}

	// Use tools with AI
	logger.FromContext(ctx).Info("Starting demo: Fetch and summarize content")

	response, err := genkit.Generate(ctx, g,
		ai.WithModelName("googleai/gemini-2.5-pro"),
		ai.WithPrompt("Fetch content from https://httpbin.org/json and give me a summary of what you find"),
		ai.WithTools(toolRefs...),
		ai.WithToolChoice(ai.ToolChoiceAuto),
	)

	if err != nil {
		logger.FromContext(ctx).Error("Generation failed", "error", err)
	} else {
		logger.FromContext(ctx).Info("Generation completed", "result", response.Text())
	}
}

func getToolNames(tools []ai.Tool) []string {
	var names []string
	for _, tool := range tools {
		names = append(names, tool.Name())
	}
	return names
}
