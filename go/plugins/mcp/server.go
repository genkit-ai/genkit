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

package mcp

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// MCPServerOptions holds configuration for GenkitMCPServer
type MCPServerOptions struct {
	// Name for this server instance - used for MCP identification
	Name string
	// Version number for this server (defaults to "1.0.0" if empty)
	Version string
}

// GenkitMCPServer represents an MCP server that exposes Genkit tools, prompts, and resources
type GenkitMCPServer struct {
	genkit    *genkit.Genkit
	options   MCPServerOptions
	mcpServer *server.MCPServer

	// Discovered actions from Genkit registry
	toolActions     []ai.Tool
	resourceActions []api.Action
	actionsResolved bool
}

// NewMCPServer creates a new GenkitMCPServer with the provided options
func NewMCPServer(g *genkit.Genkit, options MCPServerOptions) *GenkitMCPServer {
	// Set default values
	if options.Version == "" {
		options.Version = "1.0.0"
	}

	server := &GenkitMCPServer{
		genkit:  g,
		options: options,
	}

	return server
}

// setup initializes the MCP server and discovers actions
func (s *GenkitMCPServer) setup() error {
	if s.actionsResolved {
		return nil
	}

	// Create MCP server with all capabilities
	s.mcpServer = server.NewMCPServer(
		s.options.Name,
		s.options.Version,
		server.WithToolCapabilities(true),
		server.WithResourceCapabilities(true, true), // subscribe and listChanged capabilities
	)

	// Discover and categorize actions from Genkit registry
	toolActions, resourceActions, err := s.discoverAndCategorizeActions()
	if err != nil {
		return fmt.Errorf("failed to discover actions: %w", err)
	}

	// Store discovered actions
	s.toolActions = toolActions
	s.resourceActions = resourceActions

	// Register tools with the MCP server
	for _, tool := range toolActions {
		mcpTool := s.convertGenkitToolToMCP(tool)
		s.mcpServer.AddTool(mcpTool, s.createToolHandler(tool))
	}

	// Register resources with the MCP server
	for _, resourceAction := range resourceActions {
		if err := s.registerResourceWithMCP(resourceAction); err != nil {
			slog.Warn("Failed to register resource", "resource", resourceAction.Desc().Name, "error", err)
		}
	}

	s.actionsResolved = true
	slog.Info("MCP Server setup complete",
		"name", s.options.Name,
		"tools", len(s.toolActions),
		"resources", len(s.resourceActions))
	return nil
}

// discoverAndCategorizeActions discovers all actions from Genkit registry and categorizes them
func (s *GenkitMCPServer) discoverAndCategorizeActions() ([]ai.Tool, []api.Action, error) {
	// Use the existing List functions which properly handle the registry access
	toolActions := genkit.ListTools(s.genkit)
	resources := genkit.ListResources(s.genkit)

	// Convert ai.Resource to api.Action
	resourceActions := make([]api.Action, len(resources))
	for i, resource := range resources {
		if resourceAction, ok := resource.(api.Action); ok {
			resourceActions[i] = resourceAction
		} else {
			return nil, nil, fmt.Errorf("resource %s does not implement api.Action", resource.Name())
		}
	}

	return toolActions, resourceActions, nil
}

// convertGenkitToolToMCP converts a Genkit tool to MCP format
func (s *GenkitMCPServer) convertGenkitToolToMCP(tool ai.Tool) mcp.Tool {
	def := tool.Definition()

	// Start with basic options
	options := []mcp.ToolOption{mcp.WithDescription(def.Description)}

	// Convert input schema if available
	if def.InputSchema != nil {
		// Parse the JSON schema and convert to MCP tool options
		if properties, ok := def.InputSchema["properties"].(map[string]interface{}); ok {
			// Convert each property to appropriate MCP option
			for propName, propDef := range properties {
				if propMap, ok := propDef.(map[string]interface{}); ok {
					propType, _ := propMap["type"].(string)

					switch propType {
					case "string":
						options = append(options, mcp.WithString(propName))
					case "integer", "number":
						options = append(options, mcp.WithNumber(propName))
					case "boolean":
						options = append(options, mcp.WithBoolean(propName))
					}
				}
			}
		}
	}

	return mcp.NewTool(def.Name, options...)
}

// createToolHandler creates an MCP tool handler for a Genkit tool
func (s *GenkitMCPServer) createToolHandler(tool ai.Tool) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		// Execute the Genkit tool
		result, err := tool.RunRaw(ctx, request.Params.Arguments)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}

		// Convert result to MCP format
		switch v := result.(type) {
		case string:
			return mcp.NewToolResultText(v), nil
		case nil:
			return mcp.NewToolResultText(""), nil
		default:
			// Convert complex types to string
			return mcp.NewToolResultText(fmt.Sprintf("%v", v)), nil
		}
	}
}

// registerResourceWithMCP registers a Genkit resource with the MCP server
func (s *GenkitMCPServer) registerResourceWithMCP(resourceAction api.Action) error {
	desc := resourceAction.Desc()
	resourceName := strings.TrimPrefix(desc.Key, "/resource/")

	// Extract original URI/template from metadata
	var originalURI string
	var isTemplate bool

	if resourceMeta, ok := desc.Metadata["resource"].(map[string]any); ok {
		if uri, ok := resourceMeta["uri"].(string); ok && uri != "" {
			originalURI = uri
			isTemplate = false
		} else if template, ok := resourceMeta["template"].(string); ok && template != "" {
			originalURI = template
			isTemplate = true
		}
	}

	// Fallback to synthetic URI if no original URI found (shouldn't happen normally)
	if originalURI == "" {
		originalURI = fmt.Sprintf("genkit://%s", resourceName)
		isTemplate = false
	}

	// Create resource handler
	handler := func(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {

		// Find matching resource for the URI and execute it
		resourceAction, input, err := genkit.FindMatchingResource(s.genkit, request.Params.URI)
		if err != nil {
			return nil, fmt.Errorf("no resource found for URI %s: %w", request.Params.URI, err)
		}

		// Execute the resource
		result, err := resourceAction.Execute(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("resource execution failed: %w", err)
		}

		// Convert result to MCP content format
		var contents []mcp.ResourceContents
		for _, part := range result.Content {
			if part.Text != "" {
				contents = append(contents, mcp.TextResourceContents{
					URI:      request.Params.URI,
					MIMEType: "text/plain",
					Text:     part.Text,
				})
			}
			// Handle other part types (media, data, etc.) if needed
		}

		return contents, nil
	}

	// Register as template resource or static resource based on type
	if isTemplate {
		// Create MCP template resource
		mcpTemplate := mcp.NewResourceTemplate(
			originalURI,  // Template URI like "user://profile/{id}"
			resourceName, // Name
			mcp.WithTemplateDescription(desc.Description),
		)
		s.mcpServer.AddResourceTemplate(mcpTemplate, handler)
	} else {
		// Create MCP static resource
		mcpResource := mcp.NewResource(
			originalURI,  // Static URI
			resourceName, // Name
			mcp.WithResourceDescription(desc.Description),
		)
		s.mcpServer.AddResource(mcpResource, handler)
	}

	return nil
}

// ServeStdio starts the MCP server using stdio transport
func (s *GenkitMCPServer) ServeStdio() error {
	if err := s.setup(); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	return server.ServeStdio(s.mcpServer)
}

// Serve starts the MCP server with a custom transport
func (s *GenkitMCPServer) Serve(transport interface{}) error {
	if err := s.setup(); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	// For now, only stdio is supported through the server.ServeStdio function
	return server.ServeStdio(s.mcpServer)
}

// Close shuts down the MCP server
func (s *GenkitMCPServer) Close() error {
	// The mcp-go server handles cleanup internally
	return nil
}

// GetServer returns the underlying MCP server instance
func (s *GenkitMCPServer) GetServer() *server.MCPServer {
	return s.mcpServer
}

// ListRegisteredTools returns the names of all discovered tools
func (s *GenkitMCPServer) ListRegisteredTools() []string {
	var toolNames []string
	for _, tool := range s.toolActions {
		toolNames = append(toolNames, tool.Name())
	}
	return toolNames
}

// ListRegisteredResources returns the names of all discovered resources
func (s *GenkitMCPServer) ListRegisteredResources() []string {
	var resourceNames []string
	for _, resourceAction := range s.resourceActions {
		desc := resourceAction.Desc()
		resourceName := strings.TrimPrefix(desc.Key, "/resource/")
		resourceNames = append(resourceNames, resourceName)
	}
	return resourceNames
}
