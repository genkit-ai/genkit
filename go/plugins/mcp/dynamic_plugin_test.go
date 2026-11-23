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

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// newDynamicTestClient connects a client named "remote" to the basic_server
// fixture, which exposes a single "echo" tool.
func newDynamicTestClient(t *testing.T) *GenkitMCPClient {
	t.Helper()
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
	return client
}

// lookupEcho resolves remote/echo through a Genkit registry, which caches the
// resolved action for the life of the Genkit instance.
func lookupEcho(t *testing.T, client *GenkitMCPClient) ai.Tool {
	t.Helper()
	g := genkit.Init(context.Background(), genkit.WithPlugins(client))
	tool := genkit.LookupTool(g, "remote/echo")
	if tool == nil {
		t.Fatal("Genkit registry did not resolve remote/echo")
	}
	return tool
}

func assertEcho(t *testing.T, tool ai.Tool, text string) {
	t.Helper()
	result, err := tool.RunRaw(context.Background(), map[string]any{"text": text})
	if err != nil {
		t.Fatalf("call remote/echo with %q: %v", text, err)
	}
	encoded, err := json.Marshal(result)
	if err != nil || !strings.Contains(string(encoded), text) {
		t.Fatalf("remote/echo result = %s, err = %v; want it to contain %q", encoded, err, text)
	}
}

func TestMCPClientDynamicActions(t *testing.T) {
	client := newDynamicTestClient(t)

	got := client.ListActions(context.Background())
	if len(got) != 1 || got[0].Key != "/tool.v2/remote/echo" || got[0].Description != "Echo text" {
		t.Fatalf("ListActions() = %+v, want only remote/echo", got)
	}
	if action := client.ResolveAction(api.ActionTypeModel, "echo"); action != nil {
		t.Errorf("ResolveAction(model, echo) = %v, want nil", action.Name())
	}
	if action := client.ResolveAction(api.ActionTypeToolV2, "missing"); action != nil {
		t.Errorf("ResolveAction(tool.v2, missing) = %v, want nil", action.Name())
	}
	if action := client.ResolveAction(api.ActionTypeToolV2, "echo"); action == nil || action.Name() != "remote/echo" {
		t.Fatalf("ResolveAction(tool.v2, echo) = %v, want remote/echo", action)
	}
	if action := client.ResolveAction(api.ActionTypeTool, "echo"); action == nil || action.Name() != "remote/echo" {
		t.Fatalf("ResolveAction(tool, echo) = %v, want remote/echo", action)
	}

	assertEcho(t, lookupEcho(t, client), "hello")
}

// The registry keeps the first resolved action, so the tool must use the
// connection that is current when it is called, not the one it was built with.
func TestMCPClientDynamicToolSurvivesRestart(t *testing.T) {
	client := newDynamicTestClient(t)
	tool := lookupEcho(t, client)
	assertEcho(t, tool, "before restart")

	for i := range 2 {
		if err := client.Restart(context.Background()); err != nil {
			t.Fatalf("restart %d: %v", i+1, err)
		}
		assertEcho(t, tool, "after restart")
	}
}

func TestMCPClientDynamicActionsWhileDisconnected(t *testing.T) {
	client := newDynamicTestClient(t)
	tool := lookupEcho(t, client)
	if err := client.Disconnect(); err != nil {
		t.Fatalf("disconnect: %v", err)
	}

	if got := client.ListActions(context.Background()); got != nil {
		t.Errorf("ListActions() while disconnected = %+v, want nil", got)
	}
	if action := client.ResolveAction(api.ActionTypeToolV2, "echo"); action != nil {
		t.Errorf("ResolveAction() while disconnected = %v, want nil", action.Name())
	}
	_, err := tool.RunRaw(context.Background(), map[string]any{"text": "hello"})
	if err == nil || !strings.Contains(err.Error(), `MCP client "remote" is disconnected`) {
		t.Fatalf("cached tool call while disconnected: err = %v, want disconnected error", err)
	}

	if err := client.Restart(context.Background()); err != nil {
		t.Fatalf("reconnect: %v", err)
	}
	assertEcho(t, tool, "reconnected")
}
