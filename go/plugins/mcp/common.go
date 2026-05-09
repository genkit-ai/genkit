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
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
)

// GetPromptNameWithNamespace returns a prompt name prefixed with the client's namespace
func (c *GenkitMCPClient) GetPromptNameWithNamespace(promptName string) string {
	return fmt.Sprintf("%s_%s", c.options.Name, promptName)
}

// GetToolNameWithNamespace returns a tool name prefixed with the client's namespace
func (c *GenkitMCPClient) GetToolNameWithNamespace(toolName string) string {
	return fmt.Sprintf("%s_%s", c.options.Name, toolName)
}

// GetResourceNameWithNamespace returns a resource name prefixed with the client's namespace
func (c *GenkitMCPClient) GetResourceNameWithNamespace(resourceName string) string {
	return fmt.Sprintf("%s_%s", c.options.Name, resourceName)
}

// ExtractTextFromContent extracts text content from MCP Content
func ExtractTextFromContent(content mcp.Content) string {
	if textContent, ok := content.(mcp.TextContent); ok && textContent.Type == "text" {
		return textContent.Text
	} else if resourceContent, ok := content.(mcp.EmbeddedResource); ok {
		if textResource, ok := resourceContent.Resource.(mcp.TextResourceContents); ok {
			return textResource.Text
		}
	}
	return ""
}
