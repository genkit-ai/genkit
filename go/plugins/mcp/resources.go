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

	"github.com/firebase/genkit/go/ai"
	"github.com/mark3labs/mcp-go/mcp"
)

// GetActiveResources fetches resources from the MCP server
func (c *GenkitMCPClient) GetActiveResources(ctx context.Context) ([]ai.Resource, error) {
	if !c.IsEnabled() || c.server == nil {
		return nil, fmt.Errorf("MCP client is disabled or not connected")
	}

	var resources []ai.Resource

	// Fetch static resources
	staticResources, err := c.getStaticResources(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get resources from %s: %w", c.options.Name, err)
	}
	resources = append(resources, staticResources...)

	// Fetch template resources (optional - not all servers support templates)
	templateResources, err := c.getTemplateResources(ctx)
	if err != nil {
		// Templates not supported by all servers, continue without them
		return resources, nil
	}
	resources = append(resources, templateResources...)

	return resources, nil
}

// getStaticResources retrieves and converts static MCP resources to Genkit resources
func (c *GenkitMCPClient) getStaticResources(ctx context.Context) ([]ai.Resource, error) {
	mcpResources, err := c.getResources(ctx)
	if err != nil {
		return nil, err
	}

	var resources []ai.Resource
	for _, mcpResource := range mcpResources {
		resource, err := c.toGenkitResource(mcpResource)
		if err != nil {
			return nil, fmt.Errorf("failed to create resource %s: %w", mcpResource.Name, err)
		}
		resources = append(resources, resource)
	}
	return resources, nil
}

// getTemplateResources retrieves and converts MCP resource templates to Genkit resources
func (c *GenkitMCPClient) getTemplateResources(ctx context.Context) ([]ai.Resource, error) {
	mcpTemplates, err := c.getResourceTemplates(ctx)
	if err != nil {
		return nil, err
	}

	var resources []ai.Resource
	for _, mcpTemplate := range mcpTemplates {
		resource, err := c.toGenkitResourceTemplate(mcpTemplate)
		if err != nil {
			return nil, fmt.Errorf("failed to create resource template %s: %w", mcpTemplate.Name, err)
		}
		resources = append(resources, resource)
	}
	return resources, nil
}

// toGenkitResource creates a Genkit resource from an MCP static resource
func (c *GenkitMCPClient) toGenkitResource(mcpResource mcp.Resource) (ai.Resource, error) {
	// Create namespaced resource name
	resourceName := c.GetResourceNameWithNamespace(mcpResource.Name)

	// Create Genkit resource that bridges to MCP
	return ai.NewResource(resourceName, &ai.ResourceOptions{
		URI:         mcpResource.URI,
		Description: mcpResource.Description,
		Metadata: map[string]any{
			"mcp_server": c.options.Name,
			"mcp_name":   mcpResource.Name,
			"source":     "mcp",
			"mime_type":  mcpResource.MIMEType,
		},
	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
		output, err := c.readMCPResource(ctx, input.URI)
		if err != nil {
			return nil, err
		}
		return &ai.ResourceOutput{Content: output.Content}, nil
	}), nil
}

// toGenkitResourceTemplate creates a Genkit template resource from MCP template
func (c *GenkitMCPClient) toGenkitResourceTemplate(mcpTemplate mcp.ResourceTemplate) (ai.Resource, error) {
	resourceName := c.GetResourceNameWithNamespace(mcpTemplate.Name)

	// Convert URITemplate to string - extract the raw template string
	var templateStr string
	if mcpTemplate.URITemplate != nil && mcpTemplate.URITemplate.Template != nil {
		templateStr = mcpTemplate.URITemplate.Template.Raw()
	}

	// Validate template - return error instead of panicking
	if templateStr == "" {
		return nil, fmt.Errorf("MCP resource template %s has empty URI template", mcpTemplate.Name)
	}

	return ai.NewResource(resourceName, &ai.ResourceOptions{
		Template:    templateStr,
		Description: mcpTemplate.Description,
		Metadata: map[string]any{
			"mcp_server":   c.options.Name,
			"mcp_name":     mcpTemplate.Name,
			"mcp_template": templateStr,
			"source":       "mcp",
			"mime_type":    mcpTemplate.MIMEType,
		},
	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
		output, err := c.readMCPResource(ctx, input.URI)
		if err != nil {
			return nil, err
		}
		return &ai.ResourceOutput{Content: output.Content}, nil
	}), nil
}

// readMCPResource fetches content from MCP server for a given URI
func (c *GenkitMCPClient) readMCPResource(ctx context.Context, uri string) (ai.ResourceOutput, error) {
	if !c.IsEnabled() || c.server == nil {
		return ai.ResourceOutput{}, fmt.Errorf("MCP client is disabled or not connected")
	}

	// Create ReadResource request
	readReq := mcp.ReadResourceRequest{
		Params: struct {
			URI       string                 `json:"uri"`
			Arguments map[string]interface{} `json:"arguments,omitempty"`
		}{
			URI:       uri,
			Arguments: nil,
		},
	}

	// Call the MCP server to read the resource
	readResp, err := c.server.Client.ReadResource(ctx, readReq)
	if err != nil {
		return ai.ResourceOutput{}, fmt.Errorf("failed to read resource from MCP server %s: %w", c.options.Name, err)
	}

	// Convert MCP ResourceContents to Genkit Parts
	parts, err := convertMCPResourceContentsToGenkitParts(readResp.Contents)
	if err != nil {
		return ai.ResourceOutput{}, fmt.Errorf("failed to convert MCP resource contents to Genkit parts: %w", err)
	}

	return ai.ResourceOutput{Content: parts}, nil
}

// getResources retrieves all resources from the MCP server by paginating through results
func (c *GenkitMCPClient) getResources(ctx context.Context) ([]mcp.Resource, error) {
	var allResources []mcp.Resource
	var cursor mcp.Cursor

	// Paginate through all available resources from the MCP server
	for {
		// Fetch a page of resources
		resources, nextCursor, err := c.fetchResourcesPage(ctx, cursor)
		if err != nil {
			return nil, err
		}

		allResources = append(allResources, resources...)

		// Check if we've reached the last page
		cursor = nextCursor
		if cursor == "" {
			break
		}
	}

	return allResources, nil
}

// fetchResourcesPage retrieves a single page of resources from the MCP server
func (c *GenkitMCPClient) fetchResourcesPage(ctx context.Context, cursor mcp.Cursor) ([]mcp.Resource, mcp.Cursor, error) {
	// Build the list request - include cursor if we have one for pagination
	listReq := mcp.ListResourcesRequest{}
	listReq.PaginatedRequest = mcp.PaginatedRequest{
		Params: struct {
			Cursor mcp.Cursor `json:"cursor,omitempty"`
		}{
			Cursor: cursor,
		},
	}

	// Ask the MCP server for resources
	result, err := c.server.Client.ListResources(ctx, listReq)
	if err != nil {
		return nil, "", fmt.Errorf("failed to list resources from MCP server %s: %w", c.options.Name, err)
	}

	return result.Resources, result.NextCursor, nil
}

// getResourceTemplates retrieves all resource templates from the MCP server by paginating through results
func (c *GenkitMCPClient) getResourceTemplates(ctx context.Context) ([]mcp.ResourceTemplate, error) {
	var allTemplates []mcp.ResourceTemplate
	var cursor mcp.Cursor

	// Paginate through all available resource templates from the MCP server
	for {
		// Fetch a page of resource templates
		templates, nextCursor, err := c.fetchResourceTemplatesPage(ctx, cursor)
		if err != nil {
			return nil, err
		}

		allTemplates = append(allTemplates, templates...)

		// Check if we've reached the last page
		cursor = nextCursor
		if cursor == "" {
			break
		}
	}

	return allTemplates, nil
}

// fetchResourceTemplatesPage retrieves a single page of resource templates from the MCP server
func (c *GenkitMCPClient) fetchResourceTemplatesPage(ctx context.Context, cursor mcp.Cursor) ([]mcp.ResourceTemplate, mcp.Cursor, error) {
	listReq := mcp.ListResourceTemplatesRequest{
		PaginatedRequest: mcp.PaginatedRequest{
			Params: struct {
				Cursor mcp.Cursor `json:"cursor,omitempty"`
			}{
				Cursor: cursor,
			},
		},
	}

	result, err := c.server.Client.ListResourceTemplates(ctx, listReq)
	if err != nil {
		return nil, "", fmt.Errorf("failed to list resource templates from MCP server %s: %w", c.options.Name, err)
	}

	return result.ResourceTemplates, result.NextCursor, nil
}

// convertMCPResourceContentsToGenkitParts converts MCP ResourceContents to Genkit Parts
func convertMCPResourceContentsToGenkitParts(mcpContents []mcp.ResourceContents) ([]*ai.Part, error) {
	var parts []*ai.Part

	for _, content := range mcpContents {
		// Handle TextResourceContents
		if textContent, ok := content.(mcp.TextResourceContents); ok {
			parts = append(parts, ai.NewTextPart(textContent.Text))
			continue
		}

		// Handle BlobResourceContents
		if blobContent, ok := content.(mcp.BlobResourceContents); ok {
			// Create media part using ai.NewMediaPart for binary data
			parts = append(parts, ai.NewMediaPart(blobContent.MIMEType, blobContent.Blob))
			continue
		}

		// Handle unknown resource content types as text
		parts = append(parts, ai.NewTextPart(fmt.Sprintf("[Unknown MCP resource content type: %T]", content)))
	}

	return parts, nil
}
