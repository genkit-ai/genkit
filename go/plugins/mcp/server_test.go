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
	"encoding/json"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/go-cmp/cmp"
)

type weatherLocation struct {
	City    string `json:"city" jsonschema:"description=City name"`
	Country string `json:"country,omitempty"`
}

type weatherInput struct {
	Location weatherLocation `json:"location" jsonschema:"description=Where to look"`
	Units    string          `json:"units,omitempty" jsonschema:"enum=c,enum=f"`
	Days     []int           `json:"days,omitempty"`
}

type weatherReport struct {
	City       string  `json:"city"`
	TempC      float64 `json:"temp_c"`
	Conditions string  `json:"conditions"`
}

// rpc sends one JSON-RPC request to the server and returns the result as
// plain JSON values.
func rpc(t *testing.T, s *GenkitMCPServer, method string, params any) map[string]any {
	t.Helper()
	req, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(s.GetServer().HandleMessage(context.Background(), req))
	if err != nil {
		t.Fatal(err)
	}
	var resp struct {
		Result map[string]any `json:"result"`
		Error  any            `json:"error"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Error != nil {
		t.Fatalf("%s: error %v", method, resp.Error)
	}
	return resp.Result
}

// listedSchema returns the input schema tools/list advertises for the tool.
func listedSchema(t *testing.T, s *GenkitMCPServer, name string) map[string]any {
	t.Helper()
	for _, tool := range rpc(t, s, "tools/list", nil)["tools"].([]any) {
		tool := tool.(map[string]any)
		if tool["name"] == name {
			return tool["inputSchema"].(map[string]any)
		}
	}
	t.Fatalf("tools/list has no tool %q", name)
	return nil
}

// callContent calls the tool and returns the content of its result.
func callContent(t *testing.T, s *GenkitMCPServer, name string, args any) []any {
	t.Helper()
	result := rpc(t, s, "tools/call", map[string]any{"name": name, "arguments": args})
	if result["isError"] == true {
		t.Fatalf("tools/call %s: error result %v", name, result["content"])
	}
	return result["content"].([]any)
}

func TestMCPServerAdvertisesFullInputSchema(t *testing.T) {
	g := genkit.Init(context.Background())
	tool := genkit.DefineTool(g, "weather", "Gets the weather", func(_ *ai.ToolContext, in weatherInput) (string, error) {
		return "", nil
	})
	s := NewMCPServer(g, MCPServerOptions{Name: "test"})

	want, err := json.Marshal(tool.Definition().InputSchema)
	if err != nil {
		t.Fatal(err)
	}
	var wantSchema map[string]any
	if err := json.Unmarshal(want, &wantSchema); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(wantSchema, listedSchema(t, s, "weather")); diff != "" {
		t.Errorf("inputSchema mismatch (-want +got):\n%s", diff)
	}
}

func TestMCPServerNonObjectInputAdvertisesObject(t *testing.T) {
	g := genkit.Init(context.Background())
	genkit.DefineTool(g, "echo", "Echoes", func(_ *ai.ToolContext, in string) (string, error) {
		return in, nil
	})
	s := NewMCPServer(g, MCPServerOptions{Name: "test"})

	if got := listedSchema(t, s, "echo"); got["type"] != "object" {
		t.Errorf("inputSchema = %v, want an object schema", got)
	}
}

func TestMCPServerToolOutput(t *testing.T) {
	g := genkit.Init(context.Background())
	genkit.DefineTool(g, "report", "Reports", func(_ *ai.ToolContext, in weatherInput) (weatherReport, error) {
		return weatherReport{City: in.Location.City, TempC: 18.5, Conditions: "cloudy"}, nil
	})
	genkit.DefineTool(g, "greet", "Greets", func(_ *ai.ToolContext, in struct{}) (string, error) {
		return "hello", nil
	})
	genkit.DefineMultipartTool(g, "snapshot", "Snapshots", func(_ *ai.ToolContext, in struct{}) (*ai.MultipartToolResponse, error) {
		return &ai.MultipartToolResponse{
			Output: map[string]any{"ok": true},
			Content: []*ai.Part{
				ai.NewTextPart("caption"),
				ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgo="),
				ai.NewMediaPart("audio/wav", "data:audio/wav;base64,UklGRg=="),
				ai.NewMediaPart("image/png", "https://example.com/remote.png"),
				ai.NewMediaPart("application/pdf", "data:application/pdf;base64,JVBERi0="),
			},
		}, nil
	})
	s := NewMCPServer(g, MCPServerOptions{Name: "test"})

	tests := []struct {
		name string
		tool string
		args any
		want []any
	}{
		{
			name: "struct output is JSON",
			tool: "report",
			args: map[string]any{"location": map[string]any{"city": "Paris"}},
			want: []any{
				map[string]any{"type": "text", "text": `{"city":"Paris","conditions":"cloudy","temp_c":18.5}`},
			},
		},
		{
			name: "string output is sent as is",
			tool: "greet",
			args: map[string]any{},
			want: []any{map[string]any{"type": "text", "text": "hello"}},
		},
		{
			name: "multipart keeps text, image and audio and drops the rest",
			tool: "snapshot",
			args: map[string]any{},
			want: []any{
				map[string]any{"type": "text", "text": `{"ok":true}`},
				map[string]any{"type": "text", "text": "caption"},
				map[string]any{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
				map[string]any{"type": "audio", "data": "UklGRg==", "mimeType": "audio/wav"},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := callContent(t, s, tc.tool, tc.args)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("content mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMCPServerServeRejectsTransport(t *testing.T) {
	g := genkit.Init(context.Background())
	s := NewMCPServer(g, MCPServerOptions{Name: "test"})

	// A nil transport would serve stdio and block, so only the rejection is
	// testable here.
	if err := s.Serve("sse"); err == nil {
		t.Fatal("Serve(\"sse\") = nil, want an unsupported transport error")
	}
}
