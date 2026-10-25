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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

func TestMCPClientDynamicActions(t *testing.T) {
	serverBinary := filepath.Join(t.TempDir(), "basic_server")
	if output, err := exec.Command("go", "build", "-o", serverBinary, "./fixtures/basic_server").CombinedOutput(); err != nil {
		t.Fatalf("build MCP fixture: %v\n%s", err, output)
	}

	client, err := NewGenkitMCPClient(MCPClientOptions{
		Name:  "remote",
		Stdio: &StdioConfig{Command: serverBinary},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Disconnect() })

	ctx := context.Background()
	if got := client.ListActions(ctx); len(got) != 1 || got[0].Key != "/tool.v2/remote/echo" || got[0].Description != "Echo text" {
		t.Fatalf("dynamic action descriptors = %+v, want remote/echo tool", got)
	}
	if action := client.ResolveAction(api.ActionTypeModel, "echo"); action != nil {
		t.Fatalf("model action resolved from MCP tool: %+v", action)
	}
	if action := client.ResolveAction(api.ActionTypeToolV2, "missing"); action != nil {
		t.Fatalf("unknown MCP tool resolved: %+v", action)
	}

	g := genkit.Init(ctx, genkit.WithPlugins(client))
	tool := genkit.LookupTool(g, "remote/echo")
	if tool == nil {
		t.Fatal("Genkit registry did not resolve remote/echo")
	}
	result, err := tool.RunRaw(ctx, map[string]any{"text": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(result)
	if err != nil || !strings.Contains(string(encoded), "hello") {
		t.Fatalf("MCP tool result = %s, err = %v; want hello", encoded, err)
	}

	if err := client.Restart(ctx); err != nil {
		t.Fatalf("restart MCP client: %v", err)
	}
	result, err = tool.RunRaw(ctx, map[string]any{"text": "after restart"})
	if err != nil {
		t.Fatalf("call registered tool after restart: %v", err)
	}
	encoded, err = json.Marshal(result)
	if err != nil || !strings.Contains(string(encoded), "after restart") {
		t.Fatalf("MCP tool result after restart = %s, err = %v", encoded, err)
	}
	callCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	callErrors := make(chan error, 8)
	for range 8 {
		go func() {
			_, err := tool.RunRaw(callCtx, map[string]any{"text": "during restart"})
			callErrors <- err
		}()
	}
	if err := client.Restart(ctx); err != nil {
		t.Fatalf("restart MCP client during tool calls: %v", err)
	}
	for range 8 {
		// Closing the old transport can interrupt a call that was in flight.
		<-callErrors
	}
	result, err = tool.RunRaw(ctx, map[string]any{"text": "after concurrent restart"})
	if err != nil {
		t.Fatalf("call registered tool after concurrent restart: %v", err)
	}
	encoded, err = json.Marshal(result)
	if err != nil || !strings.Contains(string(encoded), "after concurrent restart") {
		t.Fatalf("MCP tool result after concurrent restart = %s, err = %v", encoded, err)
	}

	slashedClient, err := NewGenkitMCPClient(MCPClientOptions{
		Name:  "team/filesystem",
		Stdio: &StdioConfig{Command: serverBinary},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = slashedClient.Disconnect() })
	if got := slashedClient.ListActions(ctx); len(got) != 1 || got[0].Key != "/tool.v2/team%2Ffilesystem/echo" {
		t.Fatalf("slashed client action descriptors = %+v", got)
	}
	slashedGenkit := genkit.Init(ctx, genkit.WithPlugins(slashedClient))
	if tool := genkit.LookupTool(slashedGenkit, "team%2Ffilesystem/echo"); tool == nil {
		t.Fatal("Genkit registry did not resolve tool for slash-containing client name")
	}
}
