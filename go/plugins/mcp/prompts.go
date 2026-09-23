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
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/mcp"
)

const dynamicPromptMetadataKey = "genkitMCPDynamicPrompt"

// GetPrompt retrieves a prompt from the MCP server
func (c *GenkitMCPClient) GetPrompt(ctx context.Context, g *genkit.Genkit, promptName string, args map[string]string) (ai.Prompt, error) {
	if !c.IsEnabled() || c.server == nil {
		return nil, fmt.Errorf("MCP client is disabled or not connected")
	}

	// Check if prompt already exists
	namespacedPromptName := c.GetPromptNameWithNamespace(promptName)
	if existingPrompt := genkit.LookupPrompt(g, namespacedPromptName); existingPrompt != nil {
		if desc, ok := existingPrompt.(interface{ Desc() api.ActionDesc }); ok && desc.Desc().Metadata[dynamicPromptMetadataKey] == true {
			return nil, fmt.Errorf("prompt %q is registered as a dynamic MCP prompt", namespacedPromptName)
		}
		return existingPrompt, nil
	}

	// Fetch prompt from MCP server
	mcpPrompt, err := c.fetchMCPPrompt(ctx, promptName, args)
	if err != nil {
		return nil, err
	}

	// Convert and register the prompt
	return c.createGenkitPrompt(g, namespacedPromptName, mcpPrompt)
}

// GetDynamicPrompt registers an MCP prompt whose messages are fetched each time
// it is rendered or executed. Supply MCP arguments as string-valued input (for
// example, map[string]any{"topic": "Go"}) when calling Render or Execute.
// Unlike GetPrompt, it does not reuse a prompt already registered under the
// same namespaced name.
func (c *GenkitMCPClient) GetDynamicPrompt(ctx context.Context, g *genkit.Genkit, promptName string) (ai.Prompt, error) {
	if !c.IsEnabled() || c.server == nil {
		return nil, fmt.Errorf("mcp client is disabled or not connected")
	}
	if promptName == "" {
		return nil, fmt.Errorf("mcp prompt name is required")
	}

	namespacedPromptName := c.GetPromptNameWithNamespace(promptName)
	if genkit.LookupPrompt(g, namespacedPromptName) != nil {
		return nil, fmt.Errorf("prompt %q is already registered", namespacedPromptName)
	}

	prompts, err := c.getPrompts(ctx)
	if err != nil {
		return nil, err
	}
	var selected *mcp.Prompt
	for i := range prompts {
		if prompts[i].Name == promptName {
			selected = &prompts[i]
			break
		}
	}
	if selected == nil {
		return nil, fmt.Errorf("mcp prompt %q was not found", promptName)
	}

	opts := []ai.PromptOption{
		ai.WithDescription(selected.Description),
		ai.WithMetadata(map[string]any{dynamicPromptMetadataKey: true}),
		ai.WithMessagesFn(func(renderCtx context.Context, input map[string]any) ([]*ai.Message, error) {
			if !c.IsEnabled() || c.server == nil {
				return nil, fmt.Errorf("mcp client is disabled or not connected")
			}
			var args map[string]string
			if len(input) > 0 {
				args = make(map[string]string, len(input))
				for name, value := range input {
					arg, ok := value.(string)
					if !ok {
						return nil, fmt.Errorf("mcp prompt %q argument %q must be a string", promptName, name)
					}
					args[name] = arg
				}
			}
			result, err := c.fetchMCPPrompt(renderCtx, promptName, args)
			if err != nil {
				return nil, err
			}
			return c.convertMCPMessages(result.Messages), nil
		}),
	}
	if len(selected.Arguments) > 0 {
		properties := make(map[string]any, len(selected.Arguments))
		var required []string
		for _, arg := range selected.Arguments {
			properties[arg.Name] = map[string]any{"type": "string", "description": arg.Description}
			if arg.Required {
				required = append(required, arg.Name)
			}
		}
		schema := map[string]any{
			"type":                 "object",
			"properties":           properties,
			"additionalProperties": map[string]any{"type": "string"},
		}
		if len(required) > 0 {
			schema["required"] = required
		}
		opts = append(opts, ai.WithInputSchema(schema))
	}

	return genkit.DefinePrompt(g, namespacedPromptName, opts...), nil
}

// fetchMCPPrompt retrieves a prompt from the MCP server
func (c *GenkitMCPClient) fetchMCPPrompt(ctx context.Context, promptName string, args map[string]string) (*mcp.GetPromptResult, error) {
	req := mcp.GetPromptRequest{
		Params: mcp.GetPromptParams{
			Name:      promptName,
			Arguments: args,
		},
	}

	result, err := c.server.Client.GetPrompt(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to get prompt %s: %w", promptName, err)
	}

	return result, nil
}

// createGenkitPrompt converts MCP prompt to Genkit prompt and registers it
func (c *GenkitMCPClient) createGenkitPrompt(g *genkit.Genkit, promptName string, mcpPrompt *mcp.GetPromptResult) (ai.Prompt, error) {
	messages := c.convertMCPMessages(mcpPrompt.Messages)

	promptOpts := []ai.PromptOption{
		ai.WithDescription(mcpPrompt.Description),
	}

	if len(messages) > 0 {
		promptOpts = append(promptOpts, ai.WithMessages(messages...))
	}

	prompt := genkit.DefinePrompt(g, promptName, promptOpts...)

	return prompt, nil
}

// convertMCPMessages converts MCP messages to Genkit messages
func (c *GenkitMCPClient) convertMCPMessages(mcpMessages []mcp.PromptMessage) []*ai.Message {
	var messages []*ai.Message

	for _, msg := range mcpMessages {
		text := ExtractTextFromContent(msg.Content)
		if text == "" {
			continue
		}

		switch msg.Role {
		case mcp.RoleUser:
			messages = append(messages, ai.NewUserTextMessage(text))
		case mcp.RoleAssistant:
			messages = append(messages, ai.NewModelTextMessage(text))
		}
	}

	return messages
}

// GetActivePrompts retrieves all prompts available from the MCP server
func (c *GenkitMCPClient) GetActivePrompts(ctx context.Context) ([]mcp.Prompt, error) {
	if !c.IsEnabled() || c.server == nil {
		return nil, nil
	}

	// Get all MCP prompts
	return c.getPrompts(ctx)
}

// getPrompts retrieves all prompts from the MCP server by paginating through results
func (c *GenkitMCPClient) getPrompts(ctx context.Context) ([]mcp.Prompt, error) {
	var allMcpPrompts []mcp.Prompt
	var cursor mcp.Cursor

	// Paginate through all available prompts from the MCP server
	for {
		// Fetch a page of prompts
		mcpPrompts, nextCursor, err := c.fetchPromptsPage(ctx, cursor)
		if err != nil {
			return nil, err
		}

		allMcpPrompts = append(allMcpPrompts, mcpPrompts...)

		// Check if we've reached the last page
		cursor = nextCursor
		if cursor == "" {
			break
		}
	}

	return allMcpPrompts, nil
}

// fetchPromptsPage retrieves a single page of prompts from the MCP server
func (c *GenkitMCPClient) fetchPromptsPage(ctx context.Context, cursor mcp.Cursor) ([]mcp.Prompt, mcp.Cursor, error) {
	listReq := mcp.ListPromptsRequest{
		PaginatedRequest: mcp.PaginatedRequest{
			Params: mcp.PaginatedParams{
				Cursor: cursor,
			},
		},
	}

	result, err := c.server.Client.ListPrompts(ctx, listReq)
	if err != nil {
		return nil, "", fmt.Errorf("failed to list prompts: %w", err)
	}

	return result.Prompts, result.NextCursor, nil
}
