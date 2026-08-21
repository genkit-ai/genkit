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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
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
	if mcpTool.RawInputSchema != nil {
		var out map[string]any
		if err := json.Unmarshal(mcpTool.RawInputSchema, &out); err != nil {
			return nil, fmt.Errorf("failed to unmarshal MCP input schema for tool %s: %w", mcpTool.Name, err)
		}
		if out == nil {
			return nil, fmt.Errorf("MCP input schema for tool %s must be a JSON object", mcpTool.Name)
		}
		return out, nil
	}
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

// getOutputSchema returns the MCP output schema as a generic map for Genkit.
// The boolean distinguishes an omitted schema from an explicitly empty schema.
func (c *GenkitMCPClient) getOutputSchema(mcpTool mcp.Tool) (map[string]any, bool, error) {
	if mcpTool.RawOutputSchema != nil {
		var out map[string]any
		if err := json.Unmarshal(mcpTool.RawOutputSchema, &out); err != nil {
			return nil, false, fmt.Errorf("failed to unmarshal MCP output schema for tool %s: %w", mcpTool.Name, err)
		}
		if out == nil {
			return nil, false, fmt.Errorf("MCP output schema for tool %s must be a JSON object", mcpTool.Name)
		}
		return out, true, nil
	}
	if mcpTool.OutputSchema.Type == "" {
		return nil, false, nil
	}

	var out map[string]any
	schemaBytes, err := json.Marshal(mcpTool.OutputSchema)
	if err != nil {
		return nil, false, fmt.Errorf("failed to marshal MCP output schema for tool %s: %w", mcpTool.Name, err)
	}
	if err := json.Unmarshal(schemaBytes, &out); err != nil {
		return nil, false, fmt.Errorf("failed to unmarshal MCP output schema for tool %s: %w", mcpTool.Name, err)
	}
	return out, true, nil
}

// createTool converts a single MCP tool to a Genkit tool
func (c *GenkitMCPClient) createTool(mcpTool mcp.Tool) (ai.Tool, error) {
	// Use namespaced tool name
	namespacedToolName := c.GetToolNameWithNamespace(mcpTool.Name)

	inputSchema, err := c.getInputSchema(mcpTool)
	if err != nil {
		return nil, fmt.Errorf("failed to get input schema for tool %s: %w", mcpTool.Name, err)
	}
	outputSchema, hasOutputSchema, err := c.getOutputSchema(mcpTool)
	if err != nil {
		return nil, fmt.Errorf("failed to get output schema for tool %s: %w", mcpTool.Name, err)
	}
	toolFunc := c.createToolFunction(mcpTool, outputSchema, hasOutputSchema, namespacedToolName)

	var opts []ai.ToolOption
	if len(inputSchema) > 0 {
		opts = append(opts, ai.WithInputSchema(inputSchema))
	}
	if hasOutputSchema {
		opts = append(opts, ai.WithOutputSchema(outputSchema))
	}
	return ai.NewMultipartTool(namespacedToolName, mcpTool.Description, toolFunc, opts...), nil
}

func validateMCPToolResult(result *mcp.CallToolResult, outputSchema map[string]any) error {
	if result == nil {
		return fmt.Errorf("expected *mcp.CallToolResult, got nil")
	}
	// MCP tool-level errors are model-visible results, not successful structured
	// output. They commonly omit structuredContent and must not be schema-validated.
	if result.IsError {
		return nil
	}
	return base.ValidateValue(result.StructuredContent, outputSchema)
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
	if !c.IsEnabled() {
		return nil, "", fmt.Errorf("failed to list tools: client disabled")
	}
	if c.server == nil {
		return nil, "", fmt.Errorf("failed to list tools: client not initialized")
	}
	if c.server.Error != "" {
		return nil, "", fmt.Errorf("failed to list tools: %s", c.server.Error)
	}
	if c.server.Client == nil || c.server.Transport == nil {
		return nil, "", fmt.Errorf("failed to list tools: client not initialized")
	}

	listReq := mcp.ListToolsRequest{
		PaginatedRequest: mcp.PaginatedRequest{
			Params: struct {
				Cursor mcp.Cursor `json:"cursor,omitempty"`
			}{
				Cursor: cursor,
			},
		},
	}

	// Decode the tools/list response here instead of through mcp-go's typed
	// ListTools result. Its ToolOutputSchema only models a subset of JSON Schema
	// and would otherwise discard extension and unsupported keywords. Remove
	// this bypass once https://github.com/mark3labs/mcp-go/issues/563 is available
	// in the version used here with lossless input and output schema decoding.
	response, err := c.server.Transport.SendRequest(ctx, transport.JSONRPCRequest{
		JSONRPC: mcp.JSONRPC_VERSION,
		ID:      mcp.NewRequestId(fmt.Sprintf("genkit-list-tools-%d", c.listToolsRequestID.Add(1))),
		Method:  "tools/list",
		Params:  listReq.Params,
	})
	if err != nil {
		return nil, "", fmt.Errorf("failed to list tools: %w", transport.NewError(err))
	}
	if response == nil {
		return nil, "", fmt.Errorf("failed to list tools: empty response")
	}
	if response.Error != nil {
		return nil, "", fmt.Errorf("failed to list tools: %w", response.Error.AsError())
	}

	var result rawListToolsResult
	if err := json.Unmarshal(response.Result, &result); err != nil {
		return nil, "", fmt.Errorf("failed to decode tools list: %w", err)
	}
	tools := make([]mcp.Tool, len(result.Tools))
	for i, tool := range result.Tools {
		tools[i] = tool.Tool
	}

	return tools, result.NextCursor, nil
}

type rawListToolsResult struct {
	Tools      []toolWithRawSchemas `json:"tools"`
	NextCursor mcp.Cursor           `json:"nextCursor,omitempty"`
}

// toolWithRawSchemas retains the exact schemas sent by the MCP server.
// mcp.Tool's default decoder only keeps the JSON Schema fields represented by
// mcp.ToolInputSchema and mcp.ToolOutputSchema.
type toolWithRawSchemas struct {
	mcp.Tool
}

func (t *toolWithRawSchemas) UnmarshalJSON(data []byte) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	rawInputSchema, hasInputSchema := fields["inputSchema"]
	rawOutputSchema, hasOutputSchema := fields["outputSchema"]
	delete(fields, "inputSchema")
	delete(fields, "outputSchema")
	toolData, err := json.Marshal(fields)
	if err != nil {
		return err
	}

	var tool mcp.Tool
	if err := json.Unmarshal(toolData, &tool); err != nil {
		return err
	}
	if hasInputSchema && !bytes.Equal(bytes.TrimSpace(rawInputSchema), []byte("null")) {
		tool.RawInputSchema = append(json.RawMessage(nil), rawInputSchema...)
	}
	if hasOutputSchema && !bytes.Equal(bytes.TrimSpace(rawOutputSchema), []byte("null")) {
		tool.RawOutputSchema = append(json.RawMessage(nil), rawOutputSchema...)
	}
	t.Tool = tool
	return nil
}

// createToolFunction creates a Genkit tool function that will execute the MCP tool
func (c *GenkitMCPClient) createToolFunction(mcpTool mcp.Tool, outputSchema map[string]any, hasOutputSchema bool, namespacedToolName string) func(*ai.ToolContext, interface{}) (*ai.MultipartToolResponse, error) {
	// Capture mcpTool by value for the closure
	currentMCPTool := mcpTool
	client := c.server.Client

	return func(toolCtx *ai.ToolContext, args interface{}) (*ai.MultipartToolResponse, error) {
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

		return convertMCPToolResult(ctx, mcpResult, outputSchema, hasOutputSchema, namespacedToolName)
	}
}

func convertMCPToolResult(ctx context.Context, result *mcp.CallToolResult, outputSchema map[string]any, hasOutputSchema bool, toolName string) (*ai.MultipartToolResponse, error) {
	if result == nil {
		return nil, fmt.Errorf("MCP tool %q returned an empty result", toolName)
	}

	response := &ai.MultipartToolResponse{Metadata: mcpMetaMap(result.Meta)}
	if result.IsError {
		response.Output = map[string]any{"error": mcpTextContent(result.Content)}
		return response, nil
	}

	text, parts := convertMCPContent(result.Content)
	response.Content = parts
	response.Output = result.StructuredContent
	parsedTextJSON := false
	if response.Output == nil && text != "" {
		response.Output, parsedTextJSON = parseMCPTextOutput(text)
	}

	if hasOutputSchema {
		if result.StructuredContent != nil || parsedTextJSON {
			validationResult := &mcp.CallToolResult{StructuredContent: response.Output}
			if err := validateMCPToolResult(validationResult, outputSchema); err != nil {
				return nil, core.NewError(core.INTERNAL, "invalid output from tool %q: %v", toolName, err)
			}
		} else {
			logger.FromContext(ctx).Warn("MCP tool declared an output schema but returned no structured content; skipping output validation", "tool", toolName)
		}
	}

	return response, nil
}

func convertMCPContent(contents []mcp.Content) (string, []*ai.Part) {
	var text strings.Builder
	var parts []*ai.Part
	appendText := func(value string) {
		if value == "" {
			return
		}
		if text.Len() > 0 {
			text.WriteString("\n\n")
		}
		text.WriteString(value)
	}

	for _, content := range contents {
		switch value := content.(type) {
		case mcp.TextContent:
			appendText(value.Text)
		case mcp.ImageContent:
			parts = append(parts, ai.NewMediaPart(value.MIMEType, fmt.Sprintf("data:%s;base64,%s", value.MIMEType, value.Data)))
		case mcp.AudioContent:
			parts = append(parts, ai.NewMediaPart(value.MIMEType, fmt.Sprintf("data:%s;base64,%s", value.MIMEType, value.Data)))
		case mcp.ResourceLink:
			parts = append(parts, ai.NewResourcePart(value.URI))
		case mcp.EmbeddedResource:
			switch resource := value.Resource.(type) {
			case mcp.TextResourceContents:
				appendText(fmt.Sprintf("Resource (%s):\n%s", resource.URI, resource.Text))
			case mcp.BlobResourceContents:
				parts = append(parts, ai.NewMediaPart(resource.MIMEType, fmt.Sprintf("data:%s;base64,%s", resource.MIMEType, resource.Blob)))
			}
		}
	}
	return text.String(), parts
}

func parseMCPTextOutput(text string) (any, bool) {
	trimmed := strings.TrimSpace(text)
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		var output any
		if json.Unmarshal([]byte(trimmed), &output) == nil {
			return output, true
		}
	}
	return text, false
}

func mcpTextContent(contents []mcp.Content) string {
	text, _ := convertMCPContent(contents)
	return text
}

func mcpMetaMap(meta *mcp.Meta) map[string]any {
	if meta == nil {
		return nil
	}
	data, err := json.Marshal(meta)
	if err != nil {
		return nil
	}
	var result map[string]any
	if json.Unmarshal(data, &result) != nil {
		return nil
	}
	return result
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
	requiredFields := mcpTool.InputSchema.Required
	if mcpTool.RawInputSchema != nil {
		var schema struct {
			Required []string `json:"required"`
		}
		if json.Unmarshal(mcpTool.RawInputSchema, &schema) == nil {
			requiredFields = schema.Required
		}
	}
	if requiredFields != nil {
		for _, required := range requiredFields {
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
