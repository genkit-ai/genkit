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
	"fmt"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/mark3labs/mcp-go/mcp"
)

// TestMCPTemplateTranslation tests the translation of MCP ResourceTemplate
// objects to Genkit ai.Resource objects.
//
// This test validates:
// 1. Template string extraction from MCP ResourceTemplate objects
// 2. Working Genkit ai.Resource objects
// 3. URI pattern matching with extracted templates
// 4. Variable extraction from matched URIs
//
// This translation step happens inside GetActiveResources()
// when users fetch resources from MCP servers. If template extraction fails,
// the resulting resources won't match any URIs and will be unusable.
func TestMCPTemplateTranslation(t *testing.T) {
	testCases := []struct {
		name         string
		templateURI  string
		testURI      string
		shouldMatch  bool
		expectedVars map[string]string
	}{
		{
			name:         "user profile template",
			templateURI:  "user://profile/{id}",
			testURI:      "user://profile/alice",
			shouldMatch:  true,
			expectedVars: map[string]string{"id": "alice"},
		},
		{
			name:        "user profile no match",
			templateURI: "user://profile/{id}",
			testURI:     "api://different/path",
			shouldMatch: false,
		},
		{
			name:         "api service template",
			templateURI:  "api://{service}/{version}",
			testURI:      "api://users/v1",
			shouldMatch:  true,
			expectedVars: map[string]string{"service": "users", "version": "v1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Simulates what GetActiveResources() receives from MCP server
			mcpTemplate := mcp.NewResourceTemplate(tc.templateURI, "test-resource")

			if mcpTemplate.URITemplate != nil && mcpTemplate.URITemplate.Template != nil {
				rawString := mcpTemplate.URITemplate.Template.Raw()
				if rawString != tc.templateURI {
					t.Errorf("Raw() extraction failed: expected %q, got %q", tc.templateURI, rawString)
					t.Errorf("This indicates the MCP SDK Raw() method is broken!")
				}
			} else {
				t.Fatal("URITemplate structure is nil - MCP SDK structure changed!")
			}

			// Create client for testing translation
			client := &GenkitMCPClient{
				options: MCPClientOptions{Name: "test-client"},
			}

			// Test the MCP → Genkit translation step
			detachedResource, err := client.toGenkitResourceTemplate(mcpTemplate)
			if err != nil {
				t.Fatalf("MCP → Genkit translation failed: %v", err)
			}

			// Verify the translated resource can match URIs correctly
			actualMatch := detachedResource.Matches(tc.testURI)
			if actualMatch != tc.shouldMatch {
				t.Errorf("Template matching failed: template %s vs URI %s: expected match=%v, got %v",
					tc.templateURI, tc.testURI, tc.shouldMatch, actualMatch)
				t.Errorf("This indicates template extraction or URI matching is broken!")
			}

			if tc.shouldMatch && tc.expectedVars != nil {
				variables, err := detachedResource.ExtractVariables(tc.testURI)
				if err != nil {
					t.Errorf("Variable extraction failed after translation: %v", err)
				}

				for key, expectedValue := range tc.expectedVars {
					if variables[key] != expectedValue {
						t.Errorf("Variable %s: expected %s, got %s", key, expectedValue, variables[key])
					}
				}
			}
		})
	}
}

// TestMCPTemplateEdgeCases tests malformed inputs
func TestMCPTemplateEdgeCases(t *testing.T) {
	testCases := []struct {
		name         string
		templateURI  string
		testURI      string
		expectError  bool
		expectMatch  bool
		expectedVars map[string]string
		description  string
	}{
		{
			name:        "empty template",
			templateURI: "",
			testURI:     "user://profile/alice",
			expectError: true,
			description: "Should fail with empty template",
		},
		{
			name:        "malformed template - missing closing brace",
			templateURI: "user://profile/{id",
			testURI:     "user://profile/alice",
			expectError: true,
			description: "Should fail with malformed template syntax",
		},
		{
			name:        "malformed template - missing opening brace",
			templateURI: "user://profile/id}",
			testURI:     "user://profile/alice",
			expectError: true,
			description: "Should fail with malformed template syntax",
		},
		{
			name:        "template with special characters",
			templateURI: "api://v1/{resource-name}/data",
			testURI:     "api://v1/user-profiles/data",
			expectError: true, // MCP SDK rejects this template
			description: "Should handle SDK template rejections gracefully",
		},
		{
			name:         "template with encoded characters",
			templateURI:  "file://docs/{filename}",
			testURI:      "file://docs/hello%20world.pdf",
			expectMatch:  true,
			expectedVars: map[string]string{"filename": "hello world.pdf"},
			description:  "URL decoding occurs during variable extraction",
		},
		{
			name:         "URI with query parameters",
			templateURI:  "api://search/{query}",
			testURI:      "api://search/hello?limit=10&offset=0",
			expectMatch:  true, // Query parameters are stripped before matching
			expectedVars: map[string]string{"query": "hello"},
			description:  "Query parameters are stripped, template matches path portion",
		},
		{
			name:        "case sensitivity",
			templateURI: "user://profile/{id}",
			testURI:     "USER://PROFILE/ALICE",
			expectMatch: false, // URI schemes are case-sensitive
			description: "Should be case-sensitive for scheme",
		},
		{
			name:         "multiple variables same pattern",
			templateURI:  "api://{service}/{service}",
			testURI:      "api://users/users",
			expectMatch:  true,
			expectedVars: map[string]string{"service": ""}, // BUG: Returns empty instead of "users"
			description:  "Duplicate variable names have buggy behavior (should return 'users', not '')",
		},
		{
			name:         "empty variable value",
			templateURI:  "api://{service}/data",
			testURI:      "api:///data", // Empty service name
			expectMatch:  true,          // RFC 6570 allows empty variables
			expectedVars: map[string]string{"service": ""},
			description:  "Empty variable values are valid per RFC 6570",
		},
		{
			name:        "nested path variables",
			templateURI: "file:///{folder}/{subfolder}/{filename}",
			testURI:     "file:///docs/api/readme.md",
			expectMatch: true,
			expectedVars: map[string]string{
				"folder":    "docs",
				"subfolder": "api",
				"filename":  "readme.md",
			},
			description: "Should handle multiple path segments",
		},
		{
			name:         "trailing slash in URI (common user issue)",
			templateURI:  "api://users/{id}",
			testURI:      "api://users/123/", // User adds trailing slash
			expectMatch:  true,               // Fixed! Trailing slashes are now stripped
			expectedVars: map[string]string{"id": "123"},
			description:  "Trailing slashes are stripped for better UX",
		},
		{
			name:         "URI with fragment (common in docs/web)",
			templateURI:  "docs://page/{id}",
			testURI:      "docs://page/intro#section1", // Common in documentation
			expectMatch:  true,                         // Fixed! Fragments are now stripped
			expectedVars: map[string]string{"id": "intro"},
			description:  "URI fragments are stripped like query parameters",
		},
		{
			name:         "file extension in template",
			templateURI:  "file://docs/{filename}",
			testURI:      "file://docs/README.md",
			expectMatch:  true,
			expectedVars: map[string]string{"filename": "README.md"},
			description:  "File extensions should be captured in variables",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Handle empty template as special case
			if tc.templateURI == "" {
				client := &GenkitMCPClient{
					options: MCPClientOptions{Name: "test-client"},
				}

				mcpTemplate := mcp.NewResourceTemplate("", "test-resource")
				_, err := client.toGenkitResourceTemplate(mcpTemplate)

				if tc.expectError && err == nil {
					t.Error("Expected error for empty template, but got none")
				}
				if !tc.expectError && err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				return
			}

			// Test template creation (may panic for malformed templates)
			var mcpTemplate mcp.ResourceTemplate
			var templateErr error

			func() {
				defer func() {
					if r := recover(); r != nil {
						templateErr = fmt.Errorf("template creation panicked: %v", r)
					}
				}()
				mcpTemplate = mcp.NewResourceTemplate(tc.templateURI, "test-resource")
			}()

			// Create client for testing translation
			client := &GenkitMCPClient{
				options: MCPClientOptions{Name: "test-client"},
			}

			// Test the MCP → Genkit translation step
			var resource ai.Resource
			var err error

			if templateErr != nil {
				err = templateErr
			} else {
				resource, err = client.toGenkitResourceTemplate(mcpTemplate)
			}

			if tc.expectError {
				if err == nil {
					t.Errorf("Expected error for %s, but got none", tc.description)
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error for %s: %v", tc.description, err)
				return
			}

			// Test URI matching
			actualMatch := resource.Matches(tc.testURI)
			if actualMatch != tc.expectMatch {
				t.Errorf("URI matching failed for %s: template %s vs URI %s: expected match=%v, got %v",
					tc.description, tc.templateURI, tc.testURI, tc.expectMatch, actualMatch)
			}

			// Test variable extraction if match is expected
			if tc.expectMatch && tc.expectedVars != nil {
				variables, err := resource.ExtractVariables(tc.testURI)
				if err != nil {
					t.Errorf("Variable extraction failed for %s: %v", tc.description, err)
					return
				}

				for key, expectedValue := range tc.expectedVars {
					if variables[key] != expectedValue {
						t.Errorf("Variable %s: expected %q, got %q", key, expectedValue, variables[key])
					}
				}
			}
		})
	}
}
