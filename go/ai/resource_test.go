// Copyright 2024 Google LLC
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

package ai

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/internal/registry"
)

func TestStaticResource(t *testing.T) {
	g := registry.New()

	// Define static resource
	DefineResource(g, "test-doc", &ResourceOptions{
		URI: "file:///test.txt",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{
			Content: []*Part{NewTextPart("test content")},
		}, nil
	})

	// Find and execute
	resource, input, err := FindMatchingResource(g, "file:///test.txt")
	if err != nil {
		t.Fatalf("resource not found: %v", err)
	}

	output, err := resource.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("resource execution failed: %v", err)
	}

	if len(output.Content) != 1 || output.Content[0].Text != "test content" {
		t.Fatalf("unexpected output: %v", output.Content)
	}
}

func TestDynamicResourceWithTemplate(t *testing.T) {
	r := registry.New()

	dynResource := NewResource("user-profile", &ResourceOptions{
		Template: "user://profile/{userID}",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		userID := input.Variables["userID"]
		return &ResourceOutput{
			Content: []*Part{NewTextPart("User: " + userID)},
		}, nil
	})

	// Register the resource to set up tracing state properly
	dynResource.(*resource).Register(r)

	// Test URI matching
	if !dynResource.Matches("user://profile/123") {
		t.Fatal("should match user://profile/123")
	}

	if dynResource.Matches("user://different/123") {
		t.Fatal("should not match different URI scheme")
	}

	// Test variable extraction and execution
	variables, err := dynResource.ExtractVariables("user://profile/alice")
	if err != nil {
		t.Fatalf("failed to extract variables: %v", err)
	}

	if variables["userID"] != "alice" {
		t.Fatalf("expected userID=alice, got %s", variables["userID"])
	}

	// Execute with extracted variables
	input := &ResourceInput{
		URI:       "user://profile/alice",
		Variables: variables,
	}

	output, err := dynResource.Execute(context.Background(), input)
	if err != nil {
		t.Fatalf("execution failed: %v", err)
	}

	if len(output.Content) != 1 || output.Content[0].Text != "User: alice" {
		t.Fatalf("unexpected output: %v", output.Content)
	}
}

func TestResourceInGeneration(t *testing.T) {
	r := registry.New()

	// Configure default formats
	ConfigureFormats(r)

	// Define mock model
	DefineModel(r, "test", nil, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		// Extract resource parts from the prompt
		var responseText strings.Builder
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.Text != "" {
					responseText.WriteString(part.Text + " ")
				}
			}
		}

		return &ModelResponse{
			Request: req,
			Message: &Message{
				Content: []*Part{NewTextPart("Model response: " + responseText.String())},
				Role:    "model",
			},
		}, nil
	})

	// Define resource
	DefineResource(r, "policy", &ResourceOptions{
		URI: "file:///policy.txt",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{
			Content: []*Part{NewTextPart("POLICY_CONTENT")},
		}, nil
	})

	// Generate with resource reference
	resp, err := Generate(context.Background(), r,
		WithModelName("test"),
		WithMessages(NewUserMessage(
			NewTextPart("Read this:"),
			NewResourcePart("file:///policy.txt"),
			NewTextPart("Done."),
		)),
		WithOutputFormat(OutputFormatText), // Add explicit output format
	)

	if err != nil {
		t.Fatalf("generation failed: %v", err)
	}

	if !contains(resp.Text(), "Model response") {
		t.Fatalf("expected model response, got: %s", resp.Text())
	}
}

func TestDynamicResourceInGeneration(t *testing.T) {
	r := registry.New()

	// Configure default formats
	ConfigureFormats(r)

	// Define mock model
	DefineModel(r, "test", nil, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		var responseText strings.Builder
		for _, msg := range req.Messages {
			for _, part := range msg.Content {
				if part.Text != "" {
					responseText.WriteString(part.Text + " ")
				}
			}
		}

		return &ModelResponse{
			Request: req,
			Message: &Message{
				Content: []*Part{NewTextPart("Model response: " + responseText.String())},
				Role:    "model",
			},
		}, nil
	})

	// Create dynamic resource (not registered in registry)
	dynResource := NewResource("dynamic-policy", &ResourceOptions{
		URI: "dynamic://policy",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{
			Content: []*Part{NewTextPart("DYNAMIC_CONTENT")},
		}, nil
	})

	// Generate with dynamic resource reference using WithResources
	resp, err := Generate(context.Background(), r,
		WithModelName("test"),
		WithMessages(NewUserMessage(
			NewTextPart("Read this:"),
			NewResourcePart("dynamic://policy"),
			NewTextPart("Done."),
		)),
		WithResources(dynResource),
		WithOutputFormat(OutputFormatText), // Add explicit output format
	)

	if err != nil {
		t.Fatalf("generation failed: %v", err)
	}

	if !contains(resp.Text(), "Model response") {
		t.Fatalf("expected model response, got: %s", resp.Text())
	}
}

func TestMultipleDynamicResourcesInGeneration(t *testing.T) {
	r := registry.New()

	// Configure default formats
	ConfigureFormats(r)

	// Define mock model
	DefineModel(r, "test", nil, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Request: req,
			Message: &Message{
				Content: []*Part{NewTextPart("Model processed all resources")},
				Role:    "model",
			},
		}, nil
	})

	// Create multiple dynamic resources
	userResource := NewResource("user-data", &ResourceOptions{
		Template: "user://profile/{id}",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{
			Content: []*Part{NewTextPart("User: " + input.Variables["id"])},
		}, nil
	})

	projectResource := NewResource("project-data", &ResourceOptions{
		URI: "project://settings",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{
			Content: []*Part{NewTextPart("Project settings")},
		}, nil
	})

	// Generate with multiple dynamic resources
	resp, err := Generate(context.Background(), r,
		WithModelName("test"),
		WithMessages(NewUserMessage(
			NewTextPart("User:"),
			NewResourcePart("user://profile/alice"),
			NewTextPart("Project:"),
			NewResourcePart("project://settings"),
			NewTextPart("Done."),
		)),
		WithResources(userResource, projectResource),
		WithOutputFormat(OutputFormatText), // Add explicit output format
	)

	if err != nil {
		t.Fatalf("generation failed: %v", err)
	}

	if !contains(resp.Text(), "Model processed") {
		t.Fatalf("expected model response, got: %s", resp.Text())
	}
}

func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}

func TestLookupResource(t *testing.T) {
	t.Run("finds registered resource", func(t *testing.T) {
		r := registry.New()
		DefineResource(r, "test/lookup", &ResourceOptions{
			URI: "lookup://test",
		}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
			return &ResourceOutput{
				Content: []*Part{NewTextPart("found")},
			}, nil
		})

		found := LookupResource(r, "test/lookup")
		if found == nil {
			t.Fatal("LookupResource returned nil")
		}
		if found.Name() != "test/lookup" {
			t.Errorf("Name() = %q, want %q", found.Name(), "test/lookup")
		}
	})

	t.Run("returns nil for non-existent resource", func(t *testing.T) {
		r := registry.New()

		found := LookupResource(r, "test/nonexistent")
		if found != nil {
			t.Errorf("LookupResource returned %v, want nil", found)
		}
	})

	t.Run("resource can be executed after lookup", func(t *testing.T) {
		r := registry.New()
		DefineResource(r, "test/executable", &ResourceOptions{
			URI: "exec://test",
		}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
			return &ResourceOutput{
				Content: []*Part{NewTextPart("executed: " + input.URI)},
			}, nil
		})

		found := LookupResource(r, "test/executable")
		if found == nil {
			t.Fatal("LookupResource returned nil")
		}

		output, err := found.Execute(context.Background(), &ResourceInput{URI: "exec://test", Variables: map[string]string{}})
		if err != nil {
			t.Fatalf("Execute error: %v", err)
		}
		if len(output.Content) != 1 || output.Content[0].Text != "executed: exec://test" {
			t.Errorf("unexpected output: %v", output.Content)
		}
	})

	t.Run("resource matches and extracts variables after lookup", func(t *testing.T) {
		r := registry.New()
		DefineResource(r, "test/template", &ResourceOptions{
			Template: "template://item/{id}",
		}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
			return &ResourceOutput{
				Content: []*Part{NewTextPart("item " + input.Variables["id"])},
			}, nil
		})

		found := LookupResource(r, "test/template")
		if found == nil {
			t.Fatal("LookupResource returned nil")
		}

		if !found.Matches("template://item/123") {
			t.Error("Matches() = false, want true")
		}

		vars, err := found.ExtractVariables("template://item/456")
		if err != nil {
			t.Fatalf("ExtractVariables error: %v", err)
		}
		if vars["id"] != "456" {
			t.Errorf("vars[id] = %q, want %q", vars["id"], "456")
		}
	})
}
