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
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
)

// MCPServerConfig holds configuration for a single MCP server
type MCPServerConfig struct {
	// Name for this server - used as the key for lookups
	Name string
	// Config holds the client configuration options
	Config MCPClientOptions
}

// MCPHostOptions holds configuration for MCPHost
type MCPHostOptions struct {
	// Name for this host instance - used for logging and identification
	Name string
	// Version number for this host (defaults to "1.0.0" if empty)
	Version string
	// MCPServers is an array of server configurations
	MCPServers []MCPServerConfig
}

// MCPHost manages connections to multiple MCP servers
// This matches the naming convention used in the JavaScript implementation
type MCPHost struct {
	name    string
	version string
	clients map[string]*GenkitMCPClient // Internal map for efficient lookups
}

// NewMCPHost creates a new MCPHost with the given options
func NewMCPHost(g *genkit.Genkit, options MCPHostOptions) (*MCPHost, error) {
	// Set default values
	if options.Name == "" {
		options.Name = "genkit-mcp"
	}
	if options.Version == "" {
		options.Version = "1.0.0"
	}

	host := &MCPHost{
		name:    options.Name,
		version: options.Version,
		clients: make(map[string]*GenkitMCPClient),
	}

	// Connect to all servers synchronously during initialization
	ctx := context.Background()
	for _, serverConfig := range options.MCPServers {
		if err := host.Connect(ctx, g, serverConfig.Name, serverConfig.Config); err != nil {
			logger.FromContext(ctx).Error("Failed to connect to MCP server", "server", serverConfig.Name, "host", host.name, "error", err)
			// Continue with other servers
		}
	}

	return host, nil
}

// Connect connects to a single MCP server with the provided configuration
// and automatically registers tools, prompts, and resources from the server
func (h *MCPHost) Connect(ctx context.Context, g *genkit.Genkit, serverName string, config MCPClientOptions) error {
	// If a client with this name already exists, disconnect it first
	if existingClient, exists := h.clients[serverName]; exists {
		if err := existingClient.Disconnect(); err != nil {
			logger.FromContext(ctx).Warn("Error disconnecting existing MCP client", "server", serverName, "host", h.name, "error", err)
		}
	}

	logger.FromContext(ctx).Info("Connecting to MCP server", "server", serverName, "host", h.name)

	// Set the server name in the config
	if config.Name == "" {
		config.Name = serverName
	}

	// Create and connect the client
	client, err := NewGenkitMCPClient(config)
	if err != nil {
		return fmt.Errorf("error connecting to server %s: %w", serverName, err)
	}

	h.clients[serverName] = client

	return nil
}

// Disconnect disconnects from a specific MCP server
func (h *MCPHost) Disconnect(ctx context.Context, serverName string) error {
	client, exists := h.clients[serverName]
	if !exists {
		return fmt.Errorf("no client found with name '%s'", serverName)
	}

	logger.FromContext(ctx).Info("Disconnecting MCP server", "server", serverName, "host", h.name)

	err := client.Disconnect()
	delete(h.clients, serverName)
	return err
}

// Reconnect restarts a specific MCP server connection
func (h *MCPHost) Reconnect(ctx context.Context, serverName string) error {
	client, exists := h.clients[serverName]
	if !exists {
		return fmt.Errorf("no client found with name '%s'", serverName)
	}

	logger.FromContext(ctx).Info("Reconnecting MCP server", "server", serverName, "host", h.name)
	return client.Restart(ctx)
}

// GetActiveTools retrieves all tools from all connected and enabled MCP clients
func (h *MCPHost) GetActiveTools(ctx context.Context, gk *genkit.Genkit) ([]ai.Tool, error) {
	var allTools []ai.Tool

	for name, client := range h.clients {
		if !client.IsEnabled() {
			continue
		}

		tools, err := client.GetActiveTools(ctx, gk)
		if err != nil {
			logger.FromContext(ctx).Error("Error fetching tools from MCP client", "client", name, "host", h.name, "error", err)
			continue
		}

		allTools = append(allTools, tools...)
	}

	return allTools, nil
}

// GetActiveResources retrieves detached resources from all connected and enabled MCP clients
func (h *MCPHost) GetActiveResources(ctx context.Context) ([]ai.Resource, error) {
	var allResources []ai.Resource

	for name, client := range h.clients {
		if !client.IsEnabled() {
			continue
		}

		resources, err := client.GetActiveResources(ctx)
		if err != nil {
			logger.FromContext(ctx).Error("Error fetching resources from MCP client", "client", name, "host", h.name, "error", err)
			continue
		}
		allResources = append(allResources, resources...)
	}

	return allResources, nil
}

// GetPrompt retrieves a specific prompt from a specific server
func (h *MCPHost) GetPrompt(ctx context.Context, gk *genkit.Genkit, serverName, promptName string, args map[string]string) (ai.Prompt, error) {
	client, exists := h.clients[serverName]
	if !exists {
		return nil, fmt.Errorf("no client found with name '%s'", serverName)
	}

	if !client.IsEnabled() {
		return nil, fmt.Errorf("client '%s' is disabled", serverName)
	}

	return client.GetPrompt(ctx, gk, promptName, args)
}
