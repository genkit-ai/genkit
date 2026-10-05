// Copyright 2026 Google LLC
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
	"sort"
	"time"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/mark3labs/mcp-go/mcp"
)

var _ api.DynamicPlugin = (*GenkitMCPClient)(nil)

const dynamicToolDiscoveryTimeout = 10 * time.Second

// Init makes a connected MCP client usable as a Genkit dynamic plugin. Tools
// are resolved on demand, so there are no actions to register at initialization.
func (c *GenkitMCPClient) Init(context.Context) []api.Action { return nil }

// ListActions describes the tools currently exposed by the MCP server. Genkit's
// reflection API uses these descriptors to show the tools in the Dev UI.
func (c *GenkitMCPClient) ListActions(ctx context.Context) []api.ActionDesc {
	if !c.isConnected() {
		return nil
	}
	lookupCtx, cancel := context.WithTimeout(ctx, dynamicToolDiscoveryTimeout)
	defer cancel()
	tools, err := c.getTools(lookupCtx)
	if err != nil {
		logger.Warn(ctx, "unable to list MCP tools", "client", c.Name(), "error", err)
		return nil
	}

	actions := make([]api.ActionDesc, 0, len(tools))
	for _, remoteTool := range tools {
		action := c.dynamicToolAction(ctx, remoteTool)
		if action != nil {
			actions = append(actions, action.Desc())
		}
	}
	sort.Slice(actions, func(i, j int) bool { return actions[i].Name < actions[j].Name })
	return actions
}

// ResolveAction resolves an MCP tool by its server-side name. The Genkit
// registry registers the returned action under this client's provider name.
func (c *GenkitMCPClient) ResolveAction(atype api.ActionType, id string) api.Action {
	if (atype != api.ActionTypeToolV2 && atype != api.ActionTypeTool) || id == "" || !c.isConnected() {
		return nil
	}
	// DynamicPlugin.ResolveAction has no context parameter. Bound the remote
	// lookup so registry resolution cannot wait indefinitely on a server.
	ctx, cancel := context.WithTimeout(context.Background(), dynamicToolDiscoveryTimeout)
	defer cancel()
	tools, err := c.getTools(ctx)
	if err != nil {
		logger.Warn(ctx, "unable to resolve MCP tool", "client", c.Name(), "tool", id, "error", err)
		return nil
	}
	for _, remoteTool := range tools {
		if remoteTool.Name == id {
			return c.dynamicToolAction(ctx, remoteTool)
		}
	}
	return nil
}

func (c *GenkitMCPClient) dynamicToolAction(ctx context.Context, remoteTool mcp.Tool) api.Action {
	tool, err := c.createToolNamed(remoteTool, api.NewName(c.Name(), remoteTool.Name))
	if err != nil {
		logger.Warn(ctx, "unable to convert MCP tool to Genkit action", "client", c.Name(), "tool", remoteTool.Name, "error", err)
		return nil
	}
	action, ok := tool.(api.Action)
	if !ok {
		logger.Warn(ctx, "converted MCP tool is not a Genkit action", "client", c.Name(), "tool", remoteTool.Name)
		return nil
	}
	return action
}
