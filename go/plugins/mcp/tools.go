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

// Package mcp provides a client for integration with the Model Context Protocol.
package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

// GetActiveTools retrieves all tools available from the MCP server
func (c *GenkitMCPClient) GetActiveTools(ctx context.Context, g *genkit.Genkit) ([]ai.Tool, error) {
	if !c.IsEnabled() || c.server == nil {
		return nil, nil
	}

	// Get all MCP tools
	mcpTools, err := c.getTools(ctx)
	if err != nil {
		return nil, err
	}

	// Create tools from MCP server
	return c.createTools(mcpTools)
}

// createTools creates Genkit tools from MCP tools
func (c *GenkitMCPClient) createTools(mcpTools []mcp.Tool) ([]ai.Tool, error) {
	var tools []ai.Tool
	for _, mcpTool := range mcpTools {
		tool, err := c.createTool(mcpTool)
		if err != nil {
			return nil, err
		}
		if tool != nil {
			tools = append(tools, tool)
		}
	}
	return tools, nil
}

// getInputSchema returns the MCP input schema as a generic map for Genkit
func (c *GenkitMCPClient) getInputSchema(mcpTool mcp.Tool) (map[string]any, error) {
	var out map[string]any
	schemaBytes, err := json.Marshal(mcpTool.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal MCP input schema for tool %s: %w", mcpTool.Name, err)
	}
	if err := json.Unmarshal(schemaBytes, &out); err != nil {
		// Fall back to empty map if unmarshalling fails
		out = map[string]any{}
	}
	if out == nil {
		out = map[string]any{}
	}
	return out, nil
}

// createTool converts a single MCP tool to a Genkit tool
func (c *GenkitMCPClient) createTool(mcpTool mcp.Tool) (ai.Tool, error) {
	// Use namespaced tool name
	namespacedToolName := c.GetToolNameWithNamespace(mcpTool.Name)

	toolFunc := c.createToolFunction(mcpTool)
	inputSchema, err := c.getInputSchema(mcpTool)
	if err != nil {
		return nil, fmt.Errorf("failed to get input schema for tool %s: %w", mcpTool.Name, err)
	}
	var tool ai.Tool
	if len(inputSchema) > 0 {
		tool = ai.NewTool(
			namespacedToolName,
			mcpTool.Description,
			toolFunc,
			ai.WithInputSchema(inputSchema),
		)
	} else {
		tool = ai.NewTool(
			namespacedToolName,
			mcpTool.Description,
			toolFunc,
		)
	}
	return tool, nil
}

// getTools retrieves all tools from the MCP server by paginating through results
func (c *GenkitMCPClient) getTools(ctx context.Context) ([]mcp.Tool, error) {
	var allMcpTools []mcp.Tool
	var cursor mcp.Cursor

	// Paginate through all available tools from the MCP server
	for {
		// Fetch a page of tools
		mcpTools, nextCursor, err := c.fetchToolsPage(ctx, cursor)
		if err != nil {
			return nil, err
		}

		allMcpTools = append(allMcpTools, mcpTools...)

		// Check if we've reached the last page
		cursor = nextCursor
		if cursor == "" {
			break
		}
	}

	return allMcpTools, nil
}

// fetchToolsPage retrieves a single page of tools from the MCP server
func (c *GenkitMCPClient) fetchToolsPage(ctx context.Context, cursor mcp.Cursor) ([]mcp.Tool, mcp.Cursor, error) {
	listReq := mcp.ListToolsRequest{
		PaginatedRequest: mcp.PaginatedRequest{
			Params: struct {
				Cursor mcp.Cursor `json:"cursor,omitempty"`
			}{
				Cursor: cursor,
			},
		},
	}

	result, err := c.server.Client.ListTools(ctx, listReq)
	if err != nil {
		return nil, "", fmt.Errorf("failed to list tools: %w", err)
	}

	return result.Tools, result.NextCursor, nil
}

// createToolFunction creates a Genkit tool function that will execute the MCP tool
func (c *GenkitMCPClient) createToolFunction(mcpTool mcp.Tool) func(*ai.ToolContext, interface{}) (interface{}, error) {
	// Capture mcpTool by value for the closure
	currentMCPTool := mcpTool
	client := c.server.Client

	return func(toolCtx *ai.ToolContext, args interface{}) (interface{}, error) {
		ctx := toolCtx.Context // Get context from tool context

		// Convert the arguments to the format expected by MCP
		callToolArgs, err := prepareToolArguments(currentMCPTool, args)
		if err != nil {
			return nil, err
		}

		// Create and execute the MCP tool call request
		mcpResult, err := executeToolCall(ctx, client, currentMCPTool.Name, callToolArgs)
		if err != nil {
			return nil, fmt.Errorf("failed to call tool %s: %w", currentMCPTool.Name, err)
		}

		return mcpResult, nil
	}
}

// prepareToolArguments converts Genkit tool arguments to MCP format
// and validates required fields based on the tool's schema
func prepareToolArguments(mcpTool mcp.Tool, args interface{}) (map[string]interface{}, error) {
	var callToolArgs map[string]interface{}
	if args != nil {
		jsonBytes, err := json.Marshal(args)
		if err != nil {
			return nil, fmt.Errorf("tool arguments must be marshallable to map[string]interface{}, got %T: %w", args, err)
		}

		if err := json.Unmarshal(jsonBytes, &callToolArgs); err != nil {
			return nil, fmt.Errorf("tool arguments could not be converted to map[string]interface{} for tool %s: %w", mcpTool.Name, err)
		}
	} else {
		callToolArgs = make(map[string]interface{})
	}

	// Validate required fields
	if err := validateRequiredArguments(mcpTool, callToolArgs); err != nil {
		return nil, err
	}

	return callToolArgs, nil
}

// validateRequiredArguments checks if all required arguments are present
func validateRequiredArguments(mcpTool mcp.Tool, args map[string]interface{}) error {
	if mcpTool.InputSchema.Required != nil {
		for _, required := range mcpTool.InputSchema.Required {
			if _, exists := args[required]; !exists {
				return fmt.Errorf("required field %q missing for tool %q", required, mcpTool.Name)
			}
		}
	}
	return nil
}

// executeToolCall makes the actual MCP tool call
func executeToolCall(ctx context.Context, client *client.Client, toolName string, args map[string]interface{}) (*mcp.CallToolResult, error) {
	callReq := mcp.CallToolRequest{
		Params: struct {
			Name      string    `json:"name"`
			Arguments any       `json:"arguments,omitempty"`
			Meta      *mcp.Meta `json:"_meta,omitempty"`
		}{
			Name:      toolName,
			Arguments: args,
			Meta:      nil,
		},
	}

	result, err := client.CallTool(ctx, callReq)

	if err != nil {
		return nil, err
	}

	return result, nil
}
