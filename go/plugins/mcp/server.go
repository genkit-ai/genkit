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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/uri"
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

// GenkitMCPServer represents an MCP server that exposes Genkit tools and resources
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
			slog.Warn("failed to register resource with the MCP server, skipping it", "resource", resourceAction.Desc().Name, "error", err)
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

// convertGenkitToolToMCP converts a Genkit tool to MCP format, advertising
// the tool's full input JSON schema.
func (s *GenkitMCPServer) convertGenkitToolToMCP(tool ai.Tool) mcp.Tool {
	def := tool.Definition()
	return mcp.NewToolWithRawSchema(def.Name, def.Description, inputSchemaForMCP(def))
}

// inputSchemaForMCP returns the JSON schema of the tool's input. MCP requires
// an object schema, so a tool without one or whose input is not an object
// gets an empty object schema; a strict client would otherwise reject the
// whole tools/list response.
func inputSchemaForMCP(def *ai.ToolDefinition) json.RawMessage {
	emptyObject := json.RawMessage(`{"type":"object"}`)
	if def.InputSchema == nil {
		return emptyObject
	}
	if t, _ := def.InputSchema["type"].(string); t != "object" {
		slog.Warn("MCP tool input must be an object, advertising an empty object schema instead", "tool", def.Name, "type", def.InputSchema["type"])
		return emptyObject
	}
	schema, err := json.Marshal(def.InputSchema)
	if err != nil {
		slog.Warn("failed to encode tool input schema, advertising an empty object schema instead", "tool", def.Name, "error", err)
		return emptyObject
	}
	return schema
}

// createToolHandler creates an MCP tool handler for a Genkit tool
func (s *GenkitMCPServer) createToolHandler(tool ai.Tool) func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		resp, err := tool.RunRawMultipart(ctx, request.Params.Arguments)
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		return toolResultToMCP(tool.Name(), resp)
	}
}

// toolResultToMCP converts a Genkit tool response to an MCP tool result. A
// string output is sent as is and any other output as its JSON encoding.
// Content parts follow the output: text parts as text, data parts as their
// JSON encoding, and image or audio media given as a data: URI as image or
// audio content. MCP has no content type for other media or for media given
// by URL, so those parts are dropped with a warning.
func toolResultToMCP(name string, resp *ai.MultipartToolResponse) (*mcp.CallToolResult, error) {
	if resp == nil {
		resp = &ai.MultipartToolResponse{}
	}
	var content []mcp.Content
	switch v := resp.Output.(type) {
	case nil:
	case string:
		content = append(content, mcp.NewTextContent(v))
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("encoding output of tool %q: %w", name, err)
		}
		content = append(content, mcp.NewTextContent(string(b)))
	}
	for _, p := range resp.Content {
		if p == nil {
			continue
		}
		c, err := partToMCP(p)
		if err != nil {
			slog.Warn("dropping tool response part that MCP cannot carry", "tool", name, "kind", p.Kind, "error", err)
			continue
		}
		content = append(content, c)
	}
	if len(content) == 0 {
		content = append(content, mcp.NewTextContent(""))
	}
	return &mcp.CallToolResult{Content: content}, nil
}

// partToMCP converts one content part of a tool response to MCP content.
func partToMCP(p *ai.Part) (mcp.Content, error) {
	switch {
	case p.IsText():
		return mcp.NewTextContent(p.Text), nil
	case p.IsData():
		return mcp.NewTextContent(p.DataString()), nil
	case p.IsMedia():
		if !strings.HasPrefix(p.Text, "data:") {
			return nil, fmt.Errorf("media given by URL is not supported")
		}
		contentType, data, err := uri.Data(p)
		if err != nil {
			return nil, err
		}
		encoded := base64.StdEncoding.EncodeToString(data)
		switch {
		case strings.HasPrefix(contentType, "image/"):
			return mcp.NewImageContent(encoded, contentType), nil
		case strings.HasPrefix(contentType, "audio/"):
			return mcp.NewAudioContent(encoded, contentType), nil
		}
		return nil, fmt.Errorf("media type %q is not supported", contentType)
	}
	return nil, fmt.Errorf("part kind is not supported")
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

// Serve starts the MCP server over stdio. It is the same as [GenkitMCPServer.ServeStdio]
// and exists for compatibility: transport must be nil, and any other value
// returns an error. To serve over HTTP, wrap [GenkitMCPServer.GetServer] in a
// transport of the mcp-go server package, such as server.NewStreamableHTTPServer
// or server.NewSSEServer.
func (s *GenkitMCPServer) Serve(transport interface{}) error {
	if transport != nil {
		return fmt.Errorf("unsupported MCP transport %T: pass nil to serve over stdio, or wrap GetServer() in an mcp-go server transport for HTTP", transport)
	}
	return s.ServeStdio()
}

// Close shuts down the MCP server
func (s *GenkitMCPServer) Close() error {
	// The mcp-go server handles cleanup internally
	return nil
}

// GetServer returns the underlying MCP server instance, with the Genkit tools
// and resources registered on it. It returns nil if they cannot be discovered.
func (s *GenkitMCPServer) GetServer() *server.MCPServer {
	if err := s.setup(); err != nil {
		slog.Error("MCP server setup failed", "name", s.options.Name, "error", err)
		return nil
	}
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
