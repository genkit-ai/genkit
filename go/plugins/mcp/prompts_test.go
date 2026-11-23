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
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	protocol "github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/mcptest"
	"github.com/mark3labs/mcp-go/server"
)

var greetingPrompt = protocol.Prompt{
	Name:        "greeting",
	Description: "A greeting from the MCP server",
	Arguments:   []protocol.PromptArgument{{Name: "name", Required: true}},
}

func textResult(text string) *protocol.GetPromptResult {
	return &protocol.GetPromptResult{Messages: []protocol.PromptMessage{{
		Role:    protocol.RoleUser,
		Content: protocol.TextContent{Type: "text", Text: text},
	}}}
}

func testPromptClient(t *testing.T, prompt protocol.Prompt, handler server.PromptHandlerFunc) *GenkitMCPClient {
	t.Helper()
	srv := mcptest.NewUnstartedServer(t)
	srv.AddPrompt(prompt, handler)
	if err := srv.Start(context.Background()); err != nil {
		t.Fatalf("start MCP server: %v", err)
	}
	t.Cleanup(srv.Close)
	client := &GenkitMCPClient{
		options: MCPClientOptions{Name: "demo"},
		server:  &ServerRef{Client: srv.Client()},
	}
	client.active.Store(srv.Client())
	return client
}

func inputSchema(p ai.Prompt) map[string]any {
	return p.(interface{ Desc() api.ActionDesc }).Desc().InputSchema
}

func TestGetDynamicPromptFetchesOnEveryRender(t *testing.T) {
	var calls int
	client := testPromptClient(t, greetingPrompt, func(_ context.Context, request protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		calls++
		return textResult("Hello " + request.Params.Arguments["name"] + " {{literal}}"), nil
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
	type cityName string
	for i, tc := range []struct {
		name  string
		input any
		want  string
	}{
		{"Ada", map[string]any{"name": "Ada"}, "Hello Ada {{literal}}"},
		{"Grace", map[string]string{"name": "Grace"}, "Hello Grace {{literal}}"},
		{"Seattle", map[string]any{"name": cityName("Seattle")}, "Hello Seattle {{literal}}"},
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
	var calls int
	client := testPromptClient(t, protocol.Prompt{Name: "clock"}, func(_ context.Context, _ protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		calls++
		return textResult("reading"), nil
	})
	ctx := context.Background()
	prompt, err := client.GetDynamicPrompt(ctx, genkit.Init(ctx), "clock")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}
	if got := inputSchema(prompt)["type"]; !reflect.DeepEqual(got, []any{"object", "null"}) {
		t.Fatalf("input schema type = %#v, want [object null]", got)
	}
	if _, err := prompt.Render(ctx, nil); err != nil {
		t.Fatalf("Render without arguments: %v", err)
	}
	if _, err := prompt.Render(ctx, "scalar"); err == nil {
		t.Fatal("Render with scalar input succeeded, want schema validation error")
	}
	if calls != 1 {
		t.Fatalf("MCP calls = %d, want 1", calls)
	}
}

func TestGetDynamicPromptOptionalArguments(t *testing.T) {
	client := testPromptClient(t, protocol.Prompt{
		Name: "summarize",
		Arguments: []protocol.PromptArgument{
			{Name: "style", Description: "Output style"},
			{Name: "tone"},
		},
	}, func(_ context.Context, request protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		return textResult("style=" + request.Params.Arguments["style"]), nil
	})
	ctx := context.Background()
	g := genkit.Init(ctx)
	echo := genkit.DefineModel(g, "test/echo", &ai.ModelOptions{Supports: &ai.ModelSupports{}},
		func(_ context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{Message: ai.NewModelTextMessage(req.Messages[0].Text())}, nil
		})
	prompt, err := client.GetDynamicPrompt(ctx, g, "summarize")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}
	wantProperties := map[string]any{
		"style": map[string]any{"type": "string", "description": "Output style"},
		"tone":  map[string]any{"type": "string"},
	}
	if got := inputSchema(prompt)["properties"]; !reflect.DeepEqual(got, wantProperties) {
		t.Fatalf("input schema properties = %#v, want %#v", got, wantProperties)
	}

	rendered, err := prompt.Render(ctx, nil)
	if err != nil || len(rendered.Messages) != 1 || rendered.Messages[0].Text() != "style=" {
		t.Fatalf("Render(nil) = (%v, %v), want style=", rendered, err)
	}
	resp, err := prompt.Execute(ctx, ai.WithModel(echo))
	if err != nil || resp.Text() != "style=" {
		t.Fatalf("Execute without input = (%v, %v), want style=", resp, err)
	}
}

func TestGetDynamicPromptRejectsInvalidInputAndCollisions(t *testing.T) {
	var calls int
	client := testPromptClient(t, greetingPrompt, func(_ context.Context, _ protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		calls++
		return textResult("Hello"), nil
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

func TestGetDynamicPromptAfterDisconnect(t *testing.T) {
	client := testPromptClient(t, greetingPrompt, func(_ context.Context, _ protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		return textResult("Hello"), nil
	})
	ctx := context.Background()
	prompt, err := client.GetDynamicPrompt(ctx, genkit.Init(ctx), "greeting")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}
	if err := client.Disconnect(); err != nil {
		t.Fatalf("Disconnect: %v", err)
	}
	if _, err := prompt.Render(ctx, map[string]any{"name": "Ada"}); err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Fatalf("Render after disconnect error = %v, want disconnected error", err)
	}
}

func TestGetDynamicPromptDisconnectDuringRender(t *testing.T) {
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	client := testPromptClient(t, greetingPrompt, func(_ context.Context, _ protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		entered <- struct{}{}
		<-release
		return textResult("Hello"), nil
	})
	ctx := context.Background()
	prompt, err := client.GetDynamicPrompt(ctx, genkit.Init(ctx), "greeting")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}

	renderDone := make(chan error, 1)
	go func() {
		_, err := prompt.Render(ctx, map[string]any{"name": "Ada"})
		renderDone <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("MCP prompt handler was not reached")
	}

	disconnectDone := make(chan error, 1)
	go func() { disconnectDone <- client.Disconnect() }()
	close(release)
	select {
	case <-renderDone: // The in-flight request may finish or be interrupted.
	case <-time.After(5 * time.Second):
		t.Fatal("Render did not finish after disconnect")
	}
	select {
	case err := <-disconnectDone:
		if err != nil {
			t.Fatalf("Disconnect: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Disconnect did not finish")
	}
	if _, err := prompt.Render(ctx, map[string]any{"name": "Grace"}); err == nil || !strings.Contains(err.Error(), "not connected") {
		t.Fatalf("Render after disconnect error = %v, want disconnected error", err)
	}
}

func TestGetDynamicPromptSurvivesReconnect(t *testing.T) {
	mcpServer := server.NewMCPServer("test", "1.0.0", server.WithPromptCapabilities(true))
	mcpServer.AddPrompt(greetingPrompt, func(_ context.Context, request protocol.GetPromptRequest) (*protocol.GetPromptResult, error) {
		return textResult("Hello " + request.Params.Arguments["name"]), nil
	})
	ts := server.NewTestStreamableHTTPServer(mcpServer)
	t.Cleanup(ts.Close)
	client, err := NewGenkitMCPClient(MCPClientOptions{Name: "demo", StreamableHTTP: &StreamableHTTPConfig{BaseURL: ts.URL}})
	if err != nil {
		t.Fatalf("NewGenkitMCPClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Disconnect() })
	ctx := context.Background()
	prompt, err := client.GetDynamicPrompt(ctx, genkit.Init(ctx), "greeting")
	if err != nil {
		t.Fatalf("GetDynamicPrompt: %v", err)
	}

	if err := client.Restart(ctx); err != nil {
		t.Fatalf("Restart: %v", err)
	}
	rendered, err := prompt.Render(ctx, map[string]string{"name": "Linus"})
	if err != nil || len(rendered.Messages) != 1 || rendered.Messages[0].Text() != "Hello Linus" {
		t.Fatalf("Render after Restart = (%v, %v), want Hello Linus", rendered, err)
	}
}
