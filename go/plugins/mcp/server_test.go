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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/mcp"
)

func TestMCPServerToolFilterExcludesCalls(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)
	publicCalls, privateCalls := 0, 0
	genkit.DefineTool(g, "public", "Public tool", func(*ai.ToolContext, struct{}) (string, error) {
		publicCalls++
		return "public result", nil
	})
	genkit.DefineTool(g, "private", "Private tool", func(*ai.ToolContext, struct{}) (string, error) {
		privateCalls++
		return "private result", nil
	})

	s := NewMCPServer(g, MCPServerOptions{
		Name: "filtered",
		ToolFilter: func(tool ai.Tool) bool {
			return tool.Name() == "public"
		},
	})
	if err := s.setup(); err != nil {
		t.Fatal(err)
	}
	if genkit.LookupTool(g, "private") == nil {
		t.Fatal("filter removed private tool from Genkit registry")
	}

	listed := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}`))
	listResponse, ok := listed.(mcp.JSONRPCResponse)
	if !ok {
		t.Fatalf("tools/list response = %T, want JSONRPCResponse", listed)
	}
	tools, ok := listResponse.Result.(mcp.ListToolsResult)
	if !ok || len(tools.Tools) != 1 || tools.Tools[0].Name != "public" {
		t.Fatalf("tools/list result = %v, want only public", listResponse.Result)
	}

	denied := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"private","arguments":{}}}`))
	callError, ok := denied.(mcp.JSONRPCError)
	if !ok || callError.Error.Code != mcp.INVALID_PARAMS {
		t.Fatalf("private tools/call response = %v, want invalid params error", denied)
	}
	if privateCalls != 0 {
		t.Fatalf("private tool called %d times", privateCalls)
	}

	allowed := s.GetServer().HandleMessage(ctx, []byte(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"public","arguments":{}}}`))
	if _, ok := allowed.(mcp.JSONRPCResponse); !ok || publicCalls != 1 {
		t.Fatalf("public tools/call response = %v after %d calls, want success", allowed, publicCalls)
	}
}
