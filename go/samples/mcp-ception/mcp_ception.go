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
	"os/signal"
	"syscall"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/mcp"
)

// MCP self-hosting example: Genkit serves itself through MCP
// 1. Start a Go MCP server that exposes Genkit resources
// 2. Connect to that server as an MCP client
// 3. Use the resources from the server for AI generation

// Create the MCP Server (runs in background)
func createMCPServer() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	logger.FromContext(ctx).Info("Starting Genkit MCP Server")

	// Initialize Genkit for the server
	g := genkit.Init(ctx)

	// Define a tool that generates creative content (this will be auto-exposed via MCP)
	genkit.DefineTool(g, "genkit-brainstorm", "Generate creative ideas about a topic",
		func(ctx *ai.ToolContext, input struct {
			Topic string `json:"topic" description:"The topic to brainstorm about"`
		}) (map[string]interface{}, error) {
			logger.FromContext(ctx.Context).Debug("Executing genkit-brainstorm tool", "topic", input.Topic)

			ideas := fmt.Sprintf(`Creative Ideas for "%s":

1. Interactive Experience: Create an immersive, hands-on workshop
2. Digital Innovation: Develop a mobile app or web platform
3. Community Building: Start a local meetup or online community
4. Educational Content: Design a course or tutorial series
5. Collaborative Project: Partner with others for cross-pollination
6. Storytelling Approach: Create narratives around the topic
7. Gamification: Turn learning into an engaging game
8. Real-world Application: Find practical, everyday uses
9. Creative Challenge: Host competitions or hackathons
10. Multi-media Approach: Combine video, audio, and interactive elements

These ideas can be mixed, matched, and customized for "%s".`, input.Topic, input.Topic)

			return map[string]interface{}{
				"topic": input.Topic,
				"ideas": ideas,
			}, nil
		})

	// Define a resource that contains Genkit knowledge (this will be auto-exposed via MCP)
	genkit.DefineResource(g, "genkit-knowledge", &ai.ResourceOptions{
		URI: "knowledge://genkit-docs",
	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
		knowledge := `# Genkit Knowledge Base

## What is Genkit?
Genkit is Firebase's open-source framework for building AI-powered applications.

## Key Features:
- Multi-modal AI generation (text, images, audio)
- Tool calling and function execution
- RAG (Retrieval Augmented Generation) support
- Evaluation and testing frameworks
- Multi-language support (TypeScript, Go, Python)

## Popular Models:
- Google AI: Gemini 1.5 Flash, Gemini 2.0 Flash
- Vertex AI: All Gemini models
- OpenAI Compatible models via plugins

## Use Cases:
- Chatbots and conversational AI
- Content generation and editing
- Code analysis and generation
- Document processing and summarization
- Creative applications (story writing, brainstorming)

## Architecture:
Genkit follows a plugin-based architecture where models, retrievers, evaluators, and other components are provided by plugins.`

		return &ai.ResourceOutput{
			Content: []*ai.Part{ai.NewTextPart(knowledge)},
		}, nil
	})

	// Create MCP server (automatically exposes all defined tools and resources)
	server := mcp.NewMCPServer(g, mcp.MCPServerOptions{
		Name:    "genkit-mcp-server",
		Version: "1.0.0",
	})

	logger.FromContext(ctx).Info("Genkit MCP Server configured successfully")
	logger.FromContext(ctx).Info("Starting MCP server on stdio")
	logger.FromContext(ctx).Info("Registered tools", "count", len(server.ListRegisteredTools()))
	logger.FromContext(ctx).Info("Registered resources", "count", len(server.ListRegisteredResources()))

	// Start the server
	if err := server.ServeStdio(); err != nil && err != context.Canceled {
		logger.FromContext(ctx).Error("MCP server failed", "error", err)
		os.Exit(1)
	}
}

// Create the MCP Client that connects to our server
func mcpSelfConnection() {
	ctx := context.Background()

	logger.FromContext(ctx).Info("")
	logger.FromContext(ctx).Info("MCP Demo: Use your own MCP server from your own client")
	logger.FromContext(ctx).Info("This demo will: connect to a local MCP server, use a resource, and call a tool")
	logger.FromContext(ctx).Info("Connecting to local MCP server (the sample will spawn it for you)")

	// Initialize Genkit with Google AI for the client
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
	)

	// Server process is spawned automatically via stdio

	// Create MCP Host that connects to our Genkit server
	host, err := mcp.NewMCPHost(g, mcp.MCPHostOptions{
		Name: "mcp-ception-host",
		MCPServers: []mcp.MCPServerConfig{
			{
				Name: "genkit-server",
				Config: mcp.MCPClientOptions{
					Name:    "genkit-mcp-server",
					Version: "1.0.0",
					Stdio: &mcp.StdioConfig{
						Command: "go",
						Args:    []string{"run", "mcp_ception.go", "server"},
					},
				},
			},
		},
	})
	if err != nil {
		logger.FromContext(ctx).Error("Failed to create MCP host", "error", err)
		return
	}

	// Get resources from our Genkit server
	logger.FromContext(ctx).Info("Getting resources from Genkit MCP server")
	resources, err := host.GetActiveResources(ctx)
	if err != nil {
		logger.FromContext(ctx).Error("Failed to get resources", "error", err)
		return
	}
	logger.FromContext(ctx).Info("Retrieved resources from server", "count", len(resources))
	for _, r := range resources {
		logger.FromContext(ctx).Info("Resource discovered", "name", r.Name())
	}
	logger.FromContext(ctx).Info("")

	// Debug: examine retrieved resources
	for i, resource := range resources {
		logger.FromContext(ctx).Info("Resource details", "index", i, "name", resource.Name())
		// Test if the resource matches our target URI
		matches := resource.Matches("knowledge://genkit-docs")
		logger.FromContext(ctx).Info("Resource URI matching", "matches_target_uri", matches)
	}

	// Get tools from our Genkit server
	logger.FromContext(ctx).Info("Getting tools from Genkit MCP server")
	tools, err := host.GetActiveTools(ctx, g)
	if err != nil {
		logger.FromContext(ctx).Error("Failed to get tools", "error", err)
		return
	}
	logger.FromContext(ctx).Info("Retrieved tools from server", "count", len(tools))
	for _, t := range tools {
		logger.FromContext(ctx).Info("Tool discovered", "name", t.Name())
	}
	logger.FromContext(ctx).Info("")

	// Convert tools to refs
	var toolRefs []ai.ToolRef
	for _, tool := range tools {
		toolRefs = append(toolRefs, tool)
	}

	// Guided demos
	runResourceDemo(ctx, g, resources)
	logger.FromContext(ctx).Info("")
	runToolDemo(ctx, g, toolRefs)
	logger.FromContext(ctx).Info("")

	logger.FromContext(ctx).Info("MCP demos finished")

	// Clean disconnect (skip for now to avoid hanging)
	logger.FromContext(ctx).Info("MCP self-connection complete")
}

// runResourceDemo demonstrates referencing a resource from the MCP server in a model request.
func runResourceDemo(ctx context.Context, g *genkit.Genkit, resources []ai.Resource) {
	// Explain
	logger.FromContext(ctx).Info("")
	logger.FromContext(ctx).Info("Resource demo: Use a resource from our own MCP server")
	logger.FromContext(ctx).Info("We will reference URI 'knowledge://genkit-docs' and pass detached resources via WithResources")

	// Select and log target resource
	selectedURI := "knowledge://genkit-docs"
	matchedName := ""
	matched := false
	for _, r := range resources {
		if r.Matches(selectedURI) {
			matched = true
			matchedName = r.Name()
			break
		}
	}
	logger.FromContext(ctx).Info("resource_demo: selected_resource", "uri", selectedURI, "matched", matched, "resource", matchedName)

	// Call generate using resource
	resp, err := genkit.Generate(ctx, g,
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Based on this Genkit knowledge:"),
			ai.NewResourcePart(selectedURI),
			ai.NewTextPart("What is Genkit and what are its key features?"),
		)),
		ai.WithResources(resources...),
	)
	if err != nil {
		logger.FromContext(ctx).Error("resource_demo: generation failed", "error", err)
		return
	}

	// Diagnostics and preview
	toolReqCount := len(resp.ToolRequests())
	preview := resp.Text()
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	logger.FromContext(ctx).Info("resource_demo: completed", "latency_ms", resp.LatencyMs, "tool_requests", toolReqCount, "output_chars", len(resp.Text()))
	logger.FromContext(ctx).Info("resource_demo: response_preview", "text", preview)
	logger.FromContext(ctx).Info("")
}

// runToolDemo demonstrates enabling and calling a tool from the MCP server.
func runToolDemo(ctx context.Context, g *genkit.Genkit, toolRefs []ai.ToolRef) {
	// Explain
	logger.FromContext(ctx).Info("")
	logger.FromContext(ctx).Info("Tool demo: Use a tool from our own MCP server")
	logger.FromContext(ctx).Info("We will enable tools from our MCP server and let the model call one (ToolChoice=auto)")

	// List tools we are enabling
	toolNames := []string{}
	for _, t := range toolRefs {
		toolNames = append(toolNames, t.Name())
	}
	logger.FromContext(ctx).Info("tool_demo: starting", "tool_choice", string(ai.ToolChoiceAuto), "tool_count", len(toolRefs), "tools", toolNames)
	for _, n := range toolNames {
		logger.FromContext(ctx).Info("tool_demo: tool_enabled", "name", n)
	}

	// Call generate with tools enabled
	resp, err := genkit.Generate(ctx, g,
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Use the brainstorm tool to generate creative ideas for \"AI-powered cooking assistant\""),
		)),
		ai.WithTools(toolRefs...),
		ai.WithToolChoice(ai.ToolChoiceAuto),
	)
	if err != nil {
		logger.FromContext(ctx).Error("tool_demo: generation failed", "error", err)
		return
	}

	// Which tools executed?
	calledTools := []string{}
	for _, msg := range resp.History() {
		if msg.Role != ai.RoleTool {
			continue
		}
		for _, p := range msg.Content {
			if p.IsToolResponse() && p.ToolResponse != nil {
				calledTools = append(calledTools, p.ToolResponse.Name)
			}
		}
	}
	preview := resp.Text()
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	logger.FromContext(ctx).Info("tool_demo: executed_tools", "tools_called", calledTools, "count", len(calledTools))
	logger.FromContext(ctx).Info("tool_demo: completed", "latency_ms", resp.LatencyMs, "output_chars", len(resp.Text()))
	logger.FromContext(ctx).Info("tool_demo: response_preview", "text", preview)
	logger.FromContext(ctx).Info("")
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run mcp_ception.go [server|demo]")
		fmt.Println("  server - Run as MCP server (exposes Genkit resources)")
		fmt.Println("  demo   - Run MCP self-connection demo (connects to server)")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "server":
		createMCPServer()
	case "demo":
		mcpSelfConnection()
	default:
		fmt.Printf("Unknown command: %s\n", os.Args[1])
		fmt.Println("Use 'server' or 'demo'")
		os.Exit(1)
	}
}
