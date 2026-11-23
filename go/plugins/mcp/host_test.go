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
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func newInProcessMCPClient(t *testing.T, name string, srv *server.MCPServer, disabled bool) *GenkitMCPClient {
	t.Helper()
	ctx := context.Background()
	c, err := client.NewInProcessClient(srv)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	if _, err := c.Initialize(ctx, initReq); err != nil {
		t.Fatal(err)
	}
	return &GenkitMCPClient{
		options: MCPClientOptions{Name: name, Disabled: disabled},
		server:  &ServerRef{Client: c},
	}
}

func TestHostActiveTools(t *testing.T) {
	ctx := context.Background()

	enabledSrv := server.NewMCPServer("enabled", "1.0.0", server.WithToolCapabilities(true))
	enabledSrv.AddTool(mcp.NewTool("ping"), func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		return mcp.NewToolResultText("pong"), nil
	})

	disabledSrv := server.NewMCPServer("disabled", "1.0.0", server.WithToolCapabilities(true))
	disabledSrv.AddTool(mcp.NewTool("hidden"), func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		return mcp.NewToolResultText("should not appear"), nil
	})

	enabledClient := newInProcessMCPClient(t, "srv", enabledSrv, false)
	disabledClient := newInProcessMCPClient(t, "disabled", disabledSrv, true)

	host, err := NewHost(MCPHostOptions{Name: "test-host"})
	if err != nil {
		t.Fatalf("NewHost failed: %v", err)
	}
	host.clients["srv"] = enabledClient
	host.clients["disabled"] = disabledClient

	// Host.ActiveTools returns only tools from enabled clients and invokes them end-to-end.
	tools, err := host.ActiveTools(ctx)
	if err != nil {
		t.Fatalf("host.ActiveTools failed: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("got %d tools from host.ActiveTools, want 1", len(tools))
	}
	if got := tools[0].Name(); got != "srv_ping" {
		t.Fatalf("got tool name %q, want %q", got, "srv_ping")
	}

	rawResult, err := tools[0].RunRaw(ctx, map[string]any{})
	if err != nil {
		t.Fatalf("tool execution failed: %v", err)
	}
	resMap, ok := rawResult.(map[string]any)
	if !ok {
		t.Fatalf("got result type %T, want map[string]any", rawResult)
	}
	content, ok := resMap["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("unexpected content in tool result: %#v", rawResult)
	}
	firstPart, ok := content[0].(map[string]any)
	if !ok || firstPart["text"] != "pong" {
		t.Fatalf("got tool content %#v, want text %q", content[0], "pong")
	}

	// Deprecated host.GetActiveTools delegates to ActiveTools and ignores the Genkit argument.
	g := genkit.Init(ctx)
	deprecatedTools, err := host.GetActiveTools(ctx, g)
	if err != nil {
		t.Fatalf("host.GetActiveTools failed: %v", err)
	}
	if len(deprecatedTools) != 1 || deprecatedTools[0].Name() != "srv_ping" {
		t.Fatalf("got %d tools from host.GetActiveTools, want [srv_ping]", len(deprecatedTools))
	}
	if genkit.LookupTool(g, "srv_ping") != nil {
		t.Fatal("host.GetActiveTools should return detached tools, not register them on g")
	}

	// Detached tools can be explicitly registered with genkit.RegisterAction when needed.
	genkit.RegisterAction(g, tools[0])
	if genkit.LookupTool(g, "srv_ping") == nil {
		t.Fatal("expected srv_ping to be registered after genkit.RegisterAction")
	}

	// Deprecated client.GetActiveTools also delegates to client.ActiveTools and accepts nil g.
	clientTools, err := enabledClient.GetActiveTools(ctx, nil)
	if err != nil {
		t.Fatalf("client.GetActiveTools failed: %v", err)
	}
	if len(clientTools) != 1 || clientTools[0].Name() != "srv_ping" {
		t.Fatalf("got %d tools from client.GetActiveTools, want [srv_ping]", len(clientTools))
	}

	// Deprecated NewMCPHost accepts nil g and preserves host defaults.
	compatHost, err := NewMCPHost(nil, MCPHostOptions{})
	if err != nil {
		t.Fatalf("NewMCPHost(nil, ...) failed: %v", err)
	}
	if compatHost.name != "genkit-mcp" || compatHost.version != "1.0.0" {
		t.Fatalf("unexpected defaults on compatHost: name=%q version=%q", compatHost.name, compatHost.version)
	}
}
