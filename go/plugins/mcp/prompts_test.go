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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	protocol "github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/mcptest"
	"github.com/mark3labs/mcp-go/server"
)

func testPromptClient(t *testing.T, handler server.PromptHandlerFunc) *GenkitMCPClient {
	t.Helper()
	srv := mcptest.NewUnstartedServer(t)
	srv.AddPrompt(protocol.Prompt{
		Name:        "greeting",
		Description: "A greeting from the MCP server",
		Arguments: []protocol.PromptArgument{
			{Name: "name", Required: true},
		},
	}, handler)
	if err := srv.Start(context.Background()); err != nil {
		t.Fatalf("start MCP server: %v", err)
	}
	t.Cleanup(srv.Close)
	return &GenkitMCPClient{
		options: MCPClientOptions{Name: "demo"},
		server:  &ServerRef{Client: srv.Client()},
	}
}

func TestGetDynamicPromptFetchesOnEveryRender(t *testing.T) {
	var calls int
	client := testPromptClient(t, func(_ context.Context, request protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		calls++
		return &protocol.GetPromptResult{Messages: []protocol.PromptMessage{{
			Role:    protocol.RoleUser,
			Content: protocol.TextContent{Type: "text", Text: "Hello " + request.Params.Arguments["name"] + " {{literal}}"},
		}}}, nil
	})
	ctx := context.Background()
	g := genkit.Init(ctx)
	prompt, err := client.GetDynamicPrompt(ctx, g, "greeting")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}
	if prompt.Name() != "demo_greeting" {
		t.Fatalf("prompt name = %q, want demo_greeting", prompt.Name())
	}
	for i, tc := range []struct {
		name  string
		input any
		want  string
	}{
		{"Ada", map[string]any{"name": "Ada"}, "Hello Ada {{literal}}"},
		{"Grace", map[string]string{"name": "Grace"}, "Hello Grace {{literal}}"},
	} {
		toRender := prompt
		if i == 1 {
			toRender = genkit.LookupPrompt(g, prompt.Name())
		}
		rendered, err := toRender.Render(ctx, tc.input)
		if err != nil {
			t.Fatalf("Render(%q): %v", tc.name, err)
		}
		if len(rendered.Messages) != 1 || rendered.Messages[0].Text() != tc.want {
			t.Fatalf("Render(%q) messages = %v, want %q", tc.name, rendered.Messages, tc.want)
		}
		if calls != i+1 {
			t.Fatalf("after Render(%q), MCP calls = %d, want %d", tc.name, calls, i+1)
		}
	}
}

func TestGetDynamicPromptWithoutArguments(t *testing.T) {
	srv := mcptest.NewUnstartedServer(t)
	var calls int
	srv.AddPrompt(protocol.Prompt{Name: "clock"}, func(_ context.Context, _ protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		calls++
		return &protocol.GetPromptResult{Messages: []protocol.PromptMessage{{
			Role:    protocol.RoleUser,
			Content: protocol.TextContent{Type: "text", Text: "reading"},
		}}}, nil
	})
	if err := srv.Start(context.Background()); err != nil {
		t.Fatalf("start MCP server: %v", err)
	}
	t.Cleanup(srv.Close)
	client := &GenkitMCPClient{options: MCPClientOptions{Name: "demo"}, server: &ServerRef{Client: srv.Client()}}
	ctx := context.Background()
	prompt, err := client.GetDynamicPrompt(ctx, genkit.Init(ctx), "clock")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}
	if _, err := prompt.Render(ctx, nil); err != nil {
		t.Fatalf("Render without arguments: %v", err)
	}
	if calls != 1 {
		t.Fatalf("MCP calls = %d, want 1", calls)
	}
}

func TestGetDynamicPromptRejectsInvalidInputAndCollisions(t *testing.T) {
	var calls int
	client := testPromptClient(t, func(_ context.Context, _ protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		calls++
		return &protocol.GetPromptResult{Messages: []protocol.PromptMessage{{
			Role:    protocol.RoleUser,
			Content: protocol.TextContent{Type: "text", Text: "Hello"},
		}}}, nil
	})
	ctx := context.Background()
	g := genkit.Init(ctx)
	prompt, err := client.GetDynamicPrompt(ctx, g, "greeting")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}
	if _, err := prompt.Render(ctx, map[string]any{"name": 42}); err == nil {
		t.Fatal("Render with a non-string argument succeeded")
	}
	if calls != 0 {
		t.Fatalf("invalid input made %d MCP calls, want none", calls)
	}
	if _, err := client.GetPrompt(ctx, g, "greeting", nil); err == nil || !strings.Contains(err.Error(), "dynamic") {
		t.Fatalf("GetPrompt after dynamic registration error = %v, want collision", err)
	}
	if _, err := client.GetDynamicPrompt(ctx, g, "greeting"); err == nil || !strings.Contains(err.Error(), "already registered") {
		t.Fatalf("second GetDynamicPrompt error = %v, want collision", err)
	}

	staticRegistry := genkit.Init(ctx)
	if _, err := client.GetPrompt(ctx, staticRegistry, "greeting", map[string]string{"name": "Ada"}); err != nil {
		t.Fatalf("GetPrompt: %v", err)
	}
	if _, err := client.GetDynamicPrompt(ctx, staticRegistry, "greeting"); err == nil || !strings.Contains(err.Error(), "already registered") {
		t.Fatalf("GetDynamicPrompt after snapshot error = %v, want collision", err)
	}
}
