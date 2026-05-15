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

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/mcp"
)

// MCP Client Example - connects to time server
func clientExample() {
	ctx := context.Background()

	// Initialize Genkit with Google AI
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Create and connect to MCP time server
	client, err := mcp.NewGenkitMCPClient(mcp.MCPClientOptions{
		Name:    "mcp-time",
		Version: "1.0.0",
		Stdio: &mcp.StdioConfig{
			Command: "uvx",
			Args:    []string{"mcp-server-time", "--local-timezone=America/New_York"},
		},
	})
	if err != nil {
		logger.FromContext(ctx).Error("Failed to create MCP client", "error", err)
		return
	}

	// Get tools and generate response
	tools, _ := client.GetActiveTools(ctx, g)
	logger.FromContext(ctx).Info("Found MCP time tools", "count", len(tools), "client", "mcp-time")

	var toolRefs []ai.ToolRef
	for _, tool := range tools {
		toolRefs = append(toolRefs, tool)
	}

	response, err := genkit.Generate(ctx, g,
		ai.WithModelName("googleai/gemini-2.5-pro"),
		ai.WithPrompt("Convert the current time from New York to London timezone."),
		ai.WithTools(toolRefs...),
		ai.WithToolChoice(ai.ToolChoiceAuto),
	)
	if err != nil {
		logger.FromContext(ctx).Error("Generation failed", "error", err)
	} else {
		logger.FromContext(ctx).Info("Generation completed", "response", response.Text())
	}

	// Disconnect from server
	logger.FromContext(ctx).Info("Disconnecting from MCP server", "client", "mcp-time")
	client.Disconnect()
	logger.FromContext(ctx).Info("Disconnected from MCP server", "client", "mcp-time")
}

// MCP Host Example - connects to time server and demonstrates both tools and resources
func managerExample() {
	ctx := context.Background()

	// Initialize Genkit with Google AI
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Create and connect to MCP time server
	host, _ := mcp.NewMCPHost(g, mcp.MCPHostOptions{
		Name: "time-example",
		MCPServers: []mcp.MCPServerConfig{
			{
				Name: "time",
				Config: mcp.MCPClientOptions{
					Name:    "mcp-time",
					Version: "1.0.0",
					Stdio: &mcp.StdioConfig{
						Command: "uvx",
						Args:    []string{"mcp-server-time", "--local-timezone=America/New_York"},
					},
				},
			},
		},
	})

	// Get tools and resources from MCP servers
	tools, _ := host.GetActiveTools(ctx, g)
	logger.FromContext(ctx).Info("Found MCP tools", "count", len(tools))

	var toolRefs []ai.ToolRef
	for _, tool := range tools {
		toolRefs = append(toolRefs, tool)
	}

	response, err := genkit.Generate(ctx, g,
		ai.WithModelName("googleai/gemini-2.5-pro"),
		ai.WithPrompt("What time is it in New York and Tokyo?"),
		ai.WithTools(toolRefs...),
		ai.WithToolChoice(ai.ToolChoiceAuto),
	)
	if err != nil {
		logger.FromContext(ctx).Error("Generation failed", "error", err)
	} else {
		logger.FromContext(ctx).Info("Generation completed", "response", response.Text())
	}

	// Disconnect from server
	host.Disconnect(ctx, "time")
	logger.FromContext(ctx).Info("Disconnected from MCP server", "server", "time")
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go [manager|client]")
		fmt.Println("  manager        - MCP Host example with time server (tools & resources)")
		fmt.Println("  client         - MCP Client example with time server")
		os.Exit(1)
	}

	ctx := context.Background()

	switch os.Args[1] {
	case "client":
		logger.FromContext(ctx).Info("Running MCP Client example")
		clientExample()
	case "manager":
		logger.FromContext(ctx).Info("Running MCP Host example")
		managerExample()
	default:
		fmt.Printf("Unknown example: %s\n", os.Args[1])
		fmt.Println("Use 'client' or 'manager'")
		os.Exit(1)
	}
}
