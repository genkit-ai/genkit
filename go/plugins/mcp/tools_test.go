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
	"encoding/json"
	"errors"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

func asMap(t *testing.T, v any, label string) map[string]any {
	t.Helper()
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("%s: want map[string]any, got %T", label, v)
	}
	return m
}

func eqStr(t *testing.T, got any, want string, label string) {
	t.Helper()
	s, ok := got.(string)
	if !ok || s != want {
		t.Fatalf("%s: got %v", label, got)
	}
}

func eqNum(t *testing.T, got any, want int, label string) {
	t.Helper()
	f, ok := got.(float64)
	if !ok || int(f) != want {
		t.Fatalf("%s: got %v", label, got)
	}
}

func reqContains(t *testing.T, v any, want string) {
	t.Helper()
	switch arr := v.(type) {
	case []any:
		for _, x := range arr {
			if s, ok := x.(string); ok && s == want {
				return
			}
		}
	case []string:
		for _, s := range arr {
			if s == want {
				return
			}
		}
	default:
		t.Fatalf("required: unexpected type %T", v)
	}
	t.Fatalf("required does not contain %q: %v", want, v)
}

// TestCreateTool tests the createTool function.
func TestCreateTool(t *testing.T) {
	client := &GenkitMCPClient{options: MCPClientOptions{Name: "srv"}}
	client.server = &ServerRef{} // avoid nil deref in createToolFunction

	var m mcp.Tool
	toolJSON := `{
        "name": "echo",
        "description": "Echo",
        "inputSchema": {
            "type": "object",
            "required": ["q"],
            "properties": {
                "q": {"type": "string", "description": "query"},
                "n": {"type": "number", "minimum": 1, "maximum": 10},
                "arr": {"type": "array", "minItems": 2, "maxItems": 4}
            }
        }
    }`
	if err := json.Unmarshal([]byte(toolJSON), &m); err != nil {
		t.Fatalf("failed to unmarshal tool JSON: %v", err)
	}

	tool, err := client.createTool(m)
	if err != nil {
		t.Fatalf("createTool error: %v", err)
	}

	// Validate that the tool is namespaced
	def := tool.Definition()
	if def.Name != "srv_echo" {
		t.Fatalf("namespacing failed: got %q", def.Name)
	}
	if def.Description != "Echo" {
		t.Fatalf("description mismatch: %q", def.Description)
	}
	if def.InputSchema == nil {
		t.Fatalf("input schema missing")
	}

	// Validate that the schema is propagated correctly.
	eqStr(t, def.InputSchema["type"], "object", "schema.type")
	props := asMap(t, def.InputSchema["properties"], "schema.properties")

	q := asMap(t, props["q"], "properties.q")
	eqStr(t, q["type"], "string", "q.type")
	eqStr(t, q["description"], "query", "q.description")

	n := asMap(t, props["n"], "properties.n")
	eqStr(t, n["type"], "number", "n.type")
	eqNum(t, n["minimum"], 1, "n.minimum")
	eqNum(t, n["maximum"], 10, "n.maximum")

	arr := asMap(t, props["arr"], "properties.arr")
	eqStr(t, arr["type"], "array", "arr.type")
	eqNum(t, arr["minItems"], 2, "arr.minItems")
	eqNum(t, arr["maxItems"], 4, "arr.maxItems")

	reqContains(t, def.InputSchema["required"], "q")
}

// TestPrepareToolArguments tests the prepareToolArguments function.
// Ensures that required fields are validated.
func TestPrepareToolArguments(t *testing.T) {
	var tool mcp.Tool
	toolJSON := `{
        "name": "echo",
        "inputSchema": {
            "type": "object",
            "required": ["q"]
        }
    }`
	if err := json.Unmarshal([]byte(toolJSON), &tool); err != nil {
		t.Fatalf("failed to unmarshal tool JSON: %v", err)
	}

	okArgs := map[string]any{"q": "hi"}
	got, err := prepareToolArguments(tool, okArgs)
	if err != nil {
		t.Fatalf("unexpected error for valid args: %v", err)
	}
	if got["q"] != "hi" {
		t.Fatalf("args not preserved: %v", got)
	}

	_, err = prepareToolArguments(tool, map[string]any{})
	if err == nil {
		t.Fatalf("expected error for missing required field")
	}
	_, err = prepareToolArguments(tool, nil)
	if err == nil {
		t.Fatalf("expected error for nil args with required field")
	}
}

// inProcessTools serves an MCP server in process and returns its tools as
// Genkit tools, keyed by name, as GetActiveTools builds them.
func inProcessTools(t *testing.T, srv *server.MCPServer) map[string]ai.Tool {
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
	init := mcp.InitializeRequest{}
	init.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	if _, err := c.Initialize(ctx, init); err != nil {
		t.Fatal(err)
	}

	gc := &GenkitMCPClient{options: MCPClientOptions{Name: "srv"}, server: &ServerRef{Client: c}}
	tools, err := gc.GetActiveTools(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	byName := make(map[string]ai.Tool, len(tools))
	for _, tl := range tools {
		byName[tl.Name()] = tl
	}
	return byName
}

// TestToolErrorsAnswerTheCall checks which MCP tool errors are marked for the
// model: errors the server reports in the result are, protocol errors are not.
func TestToolErrorsAnswerTheCall(t *testing.T) {
	srv := server.NewMCPServer("test", "1.0.0", server.WithToolCapabilities(true))
	srv.AddTool(mcp.NewTool("lookup", mcp.WithString("city", mcp.Required())),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultError("no such city"), nil
		})
	srv.AddTool(mcp.NewTool("broken"),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return nil, errors.New("database unreachable")
		})
	tools := inProcessTools(t, srv)
	ctx := context.Background()

	tests := []struct {
		name     string
		tool     string
		input    map[string]any
		wantFail string // message the model reads; empty for an error that stops the loop
	}{
		{name: "result error", tool: "srv_lookup", input: map[string]any{"city": "Atlantis"}, wantFail: "no such city"},
		{name: "protocol error", tool: "srv_broken", input: map[string]any{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tools[tt.tool].RunRawMultipart(ctx, tt.input)
			if err == nil {
				t.Fatal("got no error")
			}
			var fail *base.ToolFailError
			marked := errors.As(err, &fail)
			if tt.wantFail == "" {
				if marked {
					t.Errorf("err = %v is marked for the model, want it to stop the loop", err)
				}
				return
			}
			if !marked {
				t.Fatalf("err = %v is not marked for the model", err)
			}
			if fail.Error() != tt.wantFail {
				t.Errorf("model reads %q, want %q", fail.Error(), tt.wantFail)
			}
		})
	}
}
