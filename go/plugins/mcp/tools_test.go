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
	"testing"

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

// TestToolOutputSchema tests that both input and output schemas are correctly retrieved
// from the MCP server.
func TestToolOutputSchema(t *testing.T) {
	// Start a test MCP server with a tool that has an input and output schema.
	type InputSchema struct {
		City string
	}
	type OutputSchema struct {
		Weather     string
		Temperature int
	}
	mcpServer := server.NewMCPServer("test", "1.0.0",
		server.WithToolCapabilities(true),
	)
	mcpServer.AddTool(
		mcp.NewTool("getWeather",
			mcp.WithInputSchema[InputSchema](),
			mcp.WithOutputSchema[OutputSchema](),
		),
		func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultStructured(
				OutputSchema{Weather: "Sunny, 25°C", Temperature: 25},
				"{\"weather\": \"Sunny, 25°C\", \"temperature\": 25}",
			), nil
		},
	)
	// Start the stdio server
	sseServer := server.NewTestServer(mcpServer)
	defer sseServer.Close()
	client, err := NewGenkitMCPClient(MCPClientOptions{
		Name: "test",
		SSE: &SSEConfig{
			BaseURL: sseServer.URL + "/sse",
		},
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Disconnect()
	// Retrieve tools from the MCP server
	tools, err := client.GetActiveTools(context.Background(), nil)
	if err != nil {
		t.Fatalf("GetActiveTools error: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	for _, tool := range tools {
		if tool.Name() != "test_getWeather" {
			t.Fatalf("unexpected tool: %s", tool.Name())
		}
		inputSchema := tool.Definition().InputSchema
		assertSchemaProperty(t, inputSchema, "City", "string")

		outputSchema := tool.Definition().OutputSchema
		assertSchemaProperty(t, outputSchema, "Weather", "string")
		assertSchemaProperty(t, outputSchema, "Temperature", "integer")

		result, err := tool.RunRaw(t.Context(), InputSchema{
			City: "Paris",
		})
		if err != nil {
			t.Fatalf("RunRaw error: %v", err)
		}
		if result == nil {
			t.Fatalf("RunRaw result is nil")
		}
		toolResult := ParseMapToStruct[mcp.CallToolResult](t, result)
		toolResultOutput := ParseMapToStruct[OutputSchema](t, toolResult.StructuredContent)
		if toolResultOutput.Weather != "Sunny, 25°C" {
			t.Fatalf("unexpected weather: %s", toolResultOutput.Weather)
		}
		if toolResultOutput.Temperature != 25 {
			t.Fatalf("unexpected temperature: %d", toolResultOutput.Temperature)
		}
	}
}

func ParseMapToStruct[T any](t *testing.T, v any) T {
	t.Helper()
	var result T
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("failed to marshal map to JSON: %v", err)
	}
	err = json.Unmarshal(jsonBytes, &result)
	if err != nil {
		t.Fatalf("failed to unmarshal JSON to struct: %v", err)
	}
	return result
}

// assertSchemaProperty asserts that a property in a schema is present and of the expected type.
func assertSchemaProperty(t *testing.T, schema map[string]any, propName string, propType string) {
	t.Helper()
	if schema == nil {
		t.Fatalf("schema is nil")
	}
	if props, ok := schema["properties"].(map[string]any); !ok {
		t.Fatalf("schema properties is nil")
	} else if propValue, ok := props[propName].(map[string]any); !ok {
		t.Fatalf("schema property %s is nil. schema: %v", propName, schema)
	} else if propValue["type"] != propType {
		t.Fatalf("schema property %s type is %s, expected %s",
			propName, propValue["type"], propType)
	}
}
