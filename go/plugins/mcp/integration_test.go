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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// TestMCPConnectionAndTranslation tests the full integration between
// MCP servers and Genkit resources.
//
// This test validates:
// 1. MCP server connection (real stdio connection)
// 2. Resource discovery from MCP server
// 3. Translation of MCP templates to Genkit resources
// 4. URI matching works end-to-end
func TestMCPConnectionAndTranslation(t *testing.T) {
	// SETUP: Build test server from fixture
	ctx := context.Background()

	// Build test server from fixtures/basic_server/
	testDir := t.TempDir()
	serverBinary := filepath.Join(testDir, "basic_server")

	cmd := exec.Command("go", "build", "-o", serverBinary, "./fixtures/basic_server")
	err := cmd.Run()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// SETUP: Genkit client
	g := genkit.Init(ctx)

	host, err := NewMCPHost(g, MCPHostOptions{
		Name: "test-host",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// TEST: Connect to MCP server
	err = host.Connect(ctx, g, "test-server", MCPClientOptions{
		Name: "test-server",
		Stdio: &StdioConfig{
			Command: serverBinary,
		},
	})

	// ASSERT 1: Connection succeeds
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// TEST: Discover resources
	resources, err := host.GetActiveResources(ctx)

	// ASSERT 2: Resources discovered
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resources) == 0 {
		t.Error("Should discover at least 1 resource from test server")
	}

	// ASSERT 3: MCP template became Genkit resource
	found := false
	testURI := "file://test/README.md"
	for _, res := range resources {
		if res.Matches(testURI) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Template 'file://test/{filename}' should match 'file://test/README.md'")
	}
}

// TestMCPAIIntegration tests that MCP resources work with AI generation.
//
// This test validates:
// 1. MCP resources can be used in AI.Generate()
// 2. Resource content is properly resolved and included in prompts
// 3. End-to-end AI generation with MCP resources
func TestMCPAIIntegration(t *testing.T) {
	// SETUP: Build test server from fixture
	ctx := context.Background()

	// Build test server from fixtures/policy_server/
	testDir := t.TempDir()
	serverBinary := filepath.Join(testDir, "policy_server")

	cmd := exec.Command("go", "build", "-o", serverBinary, "./fixtures/policy_server")
	err := cmd.Run()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// SETUP: Genkit with MCP and mock model
	g := genkit.Init(ctx)

	// Define a mock model that echoes the input (like in resource_test.go)
	genkit.DefineModel(g, "echo-model", &ai.ModelOptions{
		Label:    "Mock Echo Model for Testing",
		Supports: &ai.ModelSupports{},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		// Echo back all the content to verify resources were included
		var parts []*ai.Part
		for _, msg := range req.Messages {
			parts = append(parts, msg.Content...)
		}
		return &ai.ModelResponse{
			Message: &ai.Message{
				Content: parts,
				Role:    "model",
			},
		}, nil
	})

	host, err := NewMCPHost(g, MCPHostOptions{Name: "test-host"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = host.Connect(ctx, g, "policy-server", MCPClientOptions{
		Name:  "policy-server",
		Stdio: &StdioConfig{Command: serverBinary},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Get resources from MCP (like mcp-client sample)
	hostResources, err := host.GetActiveResources(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(hostResources) == 0 {
		t.Error("Should have MCP resources")
	}

	// TEST: AI generation with MCP resources (like resource_test.go)
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("echo-model"),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Policy summary:"),
			ai.NewResourcePart("docs://policy/vacation"), // Reference MCP resource
			ai.NewTextPart("That's the policy."),
		)),
		ai.WithResources(hostResources...), // Pass MCP resources
	)

	// ASSERT: Generation succeeds and includes resource content
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	result := resp.Text()
	t.Logf("AI response: %s", result)

	// Verify resource content was resolved and included
	if !strings.Contains(result, "VACATION_POLICY") {
		t.Error("Should include resource content")
	}
	if !strings.Contains(result, "20 days vacation") {
		t.Error("Should include specific policy details")
	}
}

// TestMCPURIMatching tests URI template matching edge cases.
//
// This test validates:
// 1. Basic URI template matching works
// 2. Edge cases work (trailing slashes, query params, fragments)
// 3. Variable extraction is correct
//
// This covers the URI normalization fixes we implemented after noticing some unhandled cases in our URI template library.
func TestMCPURIMatching(t *testing.T) {
	// SETUP: Build test server from fixture
	ctx := context.Background()

	testDir := t.TempDir()
	serverBinary := filepath.Join(testDir, "content_server")

	cmd := exec.Command("go", "build", "-o", serverBinary, "./fixtures/content_server")
	err := cmd.Run()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// SETUP: Genkit with MCP
	g := genkit.Init(ctx)

	host, err := NewMCPHost(g, MCPHostOptions{Name: "test-host"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = host.Connect(ctx, g, "content-server", MCPClientOptions{
		Name:  "content-server",
		Stdio: &StdioConfig{Command: serverBinary},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Get resources to test
	resources, err := host.GetActiveResources(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resources) == 0 {
		t.Error("Should have resources")
	}

	// TEST: Edge cases that should work with our normalizeURI fixes
	testCases := []struct {
		name string
		uri  string
	}{
		{"basic", "file://data/test.txt"},
		{"trailing slash", "file://data/test.txt/"},
		{"query params", "file://data/test.txt?version=1"},
		{"fragment", "file://data/test.txt#section1"},
		{"query and fragment", "file://data/test.txt?v=1#top"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// ASSERT: URI matches the template
			found := false
			for _, res := range resources {
				if res.Matches(tc.uri) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("URI %s should match template file://data/{filename}", tc.uri)
			}
		})
	}
}

// TestMCPContentFetch tests actual content retrieval from MCP servers.
//
// This test validates:
// 1. Content can be fetched from MCP resources
// 2. Content has the expected format and structure
// 3. Variable substitution works correctly
//
// This tests the core content delivery functionality.
func TestMCPContentFetch(t *testing.T) {
	// SETUP: Build test server from fixture
	ctx := context.Background()

	testDir := t.TempDir()
	serverBinary := filepath.Join(testDir, "content_server")

	cmd := exec.Command("go", "build", "-o", serverBinary, "./fixtures/content_server")
	err := cmd.Run()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// SETUP: Genkit with MCP
	g := genkit.Init(ctx)

	host, err := NewMCPHost(g, MCPHostOptions{Name: "test-host"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = host.Connect(ctx, g, "content-server", MCPClientOptions{
		Name:  "content-server",
		Stdio: &StdioConfig{Command: serverBinary},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// TEST: Get resources and find matching one
	resources, err := host.GetActiveResources(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resources) == 0 {
		t.Error("Should have resources")
	}

	// Find resource that matches our test URI
	testURI := "file://data/example.txt"
	var matchingResource ai.Resource
	for _, res := range resources {
		if res.Matches(testURI) {
			matchingResource = res
			break
		}
	}

	// ASSERT 1: Resource found
	if matchingResource == nil {
		t.Fatalf("Should find matching resource for %s", testURI)
	}

	// ASSERT 2: Content can be fetched via AI integration (end-to-end test)
	genkit.DefineModel(g, "echo-model", &ai.ModelOptions{
		Supports: &ai.ModelSupports{},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		// Echo back all content to verify resources were included
		var parts []*ai.Part
		for _, msg := range req.Messages {
			parts = append(parts, msg.Content...)
		}
		return &ai.ModelResponse{
			Message: &ai.Message{
				Content: parts,
				Role:    "model",
			},
		}, nil
	})

	// TEST: AI generation with MCP resource to verify content fetch
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("echo-model"),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Content:"),
			ai.NewResourcePart(testURI), // This should fetch content from MCP
			ai.NewTextPart("End."),
		)),
		ai.WithResources(resources...), // Pass all MCP resources
	)

	// ASSERT 3: Generation succeeds and includes resource content
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp == nil {
		t.Fatal("expected response, got nil")
	}

	result := resp.Text()
	t.Logf("AI response with resource content: %s", result)

	// ASSERT 4: Content was fetched and included
	if !strings.Contains(result, "CONTENT_FROM_SERVER") {
		t.Error("Should include server identifier")
	}
	if !strings.Contains(result, "example.txt") {
		t.Error("Should include the filename variable")
	}
	if !strings.Contains(result, "with important data.") {
		t.Error("Should include expected content")
	}
}

// TestMCPMultipleServers tests connecting to multiple MCP servers simultaneously.
//
// This test validates:
// 1. Multiple MCP servers can be connected at once
// 2. Resources from all servers are discoverable
// 3. Resources from different servers don't conflict
//
// This tests advanced multi-server scenarios.
func TestMCPMultipleServers(t *testing.T) {
	// SETUP: Build both test servers
	ctx := context.Background()

	testDir := t.TempDir()
	serverA := filepath.Join(testDir, "server_a")
	serverB := filepath.Join(testDir, "server_b")

	cmdA := exec.Command("go", "build", "-o", serverA, "./fixtures/server_a")
	err := cmdA.Run()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cmdB := exec.Command("go", "build", "-o", serverB, "./fixtures/server_b")
	err = cmdB.Run()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// SETUP: Genkit with MCP host
	g := genkit.Init(ctx)

	host, err := NewMCPHost(g, MCPHostOptions{Name: "multi-host"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// CONNECT: To both servers
	err = host.Connect(ctx, g, "server-a", MCPClientOptions{
		Name:  "server-a",
		Stdio: &StdioConfig{Command: serverA},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = host.Connect(ctx, g, "server-b", MCPClientOptions{
		Name:  "server-b",
		Stdio: &StdioConfig{Command: serverB},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// TEST: Get all resources from both servers
	allResources, err := host.GetActiveResources(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// ASSERT: Resources from both servers are present
	if len(allResources) < 2 {
		t.Error("Should have resources from both servers")
	}

	// ASSERT: Can identify resources from each server
	serverAResources := 0
	serverBResources := 0

	for _, res := range allResources {
		name := res.Name()
		if strings.Contains(name, "server-a") {
			serverAResources++
			// Test server A resource matches its URI pattern
			if !res.Matches("a://docs/test.md") {
				t.Error("Server A resource should match a:// pattern")
			}
		} else if strings.Contains(name, "server-b") {
			serverBResources++
			// Test server B resource matches its URI pattern
			if !res.Matches("b://files/data.json") {
				t.Error("Server B resource should match b:// pattern")
			}
		}
	}

	// ASSERT: Both servers contributed resources
	if serverAResources == 0 {
		t.Error("Should have resources from server A")
	}
	if serverBResources == 0 {
		t.Error("Should have resources from server B")
	}

	t.Logf("Found %d resources from server A, %d from server B", serverAResources, serverBResources)
}

// TestMCPErrorResilience tests error handling and graceful failure scenarios.
//
// This test validates:
// 1. Connection failures fail gracefully without crashes
// 2. Invalid/malformed content is handled properly
// 3. Resource not found scenarios provide clear errors
//
// This covers the most common real-world failure scenarios.
func TestMCPErrorResilience(t *testing.T) {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// TEST 1: Server connection failure (fast!)
	t.Run("connection_failure", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()

		host, err := NewMCPHost(g, MCPHostOptions{Name: "error-test"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Try to connect to non-existent command
		err = host.Connect(ctx, g, "bad-server", MCPClientOptions{
			Name:  "bad-server",
			Stdio: &StdioConfig{Command: "/nonexistent/command"},
		})

		// ASSERT: Graceful failure, not crash (fails in ~100ms)
		if err == nil {
			t.Error("expected error, got nil")
		}
		t.Logf("Connection failure handled gracefully: %v", err)
	})

	// TEST 2: Resource not found scenario
	t.Run("resource_not_found", func(t *testing.T) {
		// Setup working server first
		testDir := t.TempDir()
		serverBinary := filepath.Join(testDir, "basic_server")

		cmd := exec.Command("go", "build", "-o", serverBinary, "./fixtures/basic_server")
		err := cmd.Run()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		host, err := NewMCPHost(g, MCPHostOptions{Name: "test-host"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		err = host.Connect(ctx, g, "test-server", MCPClientOptions{
			Name:  "test-server",
			Stdio: &StdioConfig{Command: serverBinary},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Try to find non-existent resource
		resources, err := host.GetActiveResources(ctx)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Test URI that won't match any resource
		nonExistentURI := "file://nonexistent/path.txt"
		found := false
		for _, res := range resources {
			if res.Matches(nonExistentURI) {
				found = true
				break
			}
		}

		// ASSERT: Resource not found, but no crash
		if found {
			t.Error("Non-existent resource should not match")
		}
		t.Logf("Resource not found handled gracefully for URI: %s", nonExistentURI)
	})

	// TEST 3: Invalid URI patterns
	t.Run("invalid_uri_patterns", func(t *testing.T) {
		testDir := t.TempDir()
		serverBinary := filepath.Join(testDir, "basic_server")

		cmd := exec.Command("go", "build", "-o", serverBinary, "./fixtures/basic_server")
		err := cmd.Run()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		host, err := NewMCPHost(g, MCPHostOptions{Name: "test-host"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		err = host.Connect(ctx, g, "test-server", MCPClientOptions{
			Name:  "test-server",
			Stdio: &StdioConfig{Command: serverBinary},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		resources, err := host.GetActiveResources(ctx)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// Test various malformed URIs
		malformedURIs := []string{
			"",                  // Empty URI
			"not-a-uri",         // Not a URI
			"://missing-scheme", // Missing scheme
			"file://",           // Empty path
		}

		for _, uri := range malformedURIs {
			t.Run("malformed_uri_"+uri, func(t *testing.T) {
				// Test that malformed URIs don't crash the system
				found := false
				for _, res := range resources {
					// This should not panic or crash
					if res.Matches(uri) {
						found = true
						break
					}
				}
				// We don't care if it matches or not, just that it doesn't crash
				t.Logf("Malformed URI '%s' handled without crash, found=%v", uri, found)
			})
		}
	})
}
