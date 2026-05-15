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

package compat_oai_test

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const defaultModel = "gpt-4o-mini"

func setupTestClient(t *testing.T) *compat_oai.ModelGenerator {
	t.Helper()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: OPENAI_API_KEY environment variable not set")
	}

	client := openai.NewClient(option.WithAPIKey(apiKey))
	return compat_oai.NewModelGenerator(&client, defaultModel)
}

func TestGenerator_Complete(t *testing.T) {
	g := setupTestClient(t)
	messages := []*ai.Message{
		{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewTextPart("Tell me a joke"),
			},
		},
		{
			Role: ai.RoleModel,
			Content: []*ai.Part{
				ai.NewTextPart("Why did the scarecrow win an award?"),
			},
		},
		{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewTextPart("Why?"),
			},
		},
	}
	req := &ai.ModelRequest{
		Messages: messages,
	}

	resp, err := g.WithMessages(messages).Generate(context.Background(), req, nil)
	if err != nil {
		t.Error(err)
	}
	if len(resp.Message.Content) == 0 {
		t.Error("empty messages content, got 0")
	}
	if resp.Message.Role != ai.RoleModel {
		t.Errorf("unexpected role, got: %q, want: %q", resp.Message.Role, ai.RoleModel)
	}
}

func TestGenerator_Stream(t *testing.T) {
	g := setupTestClient(t)
	messages := []*ai.Message{
		{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewTextPart("Count from 1 to 3"),
			},
		},
	}
	req := &ai.ModelRequest{
		Messages: messages,
	}

	var chunks []string
	handleChunk := func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		for _, part := range chunk.Content {
			chunks = append(chunks, part.Text)
		}
		return nil
	}

	_, err := g.WithMessages(messages).Generate(context.Background(), req, handleChunk)
	if err != nil {
		t.Error(err)
	}
	if len(chunks) == 0 {
		t.Error("expecting stream chunks, got: 0")
	}

	// Verify we got the full response
	fullText := strings.Join(chunks, "")
	if !strings.Contains(fullText, "1") {
		t.Errorf("expecting chunk to contain: \"1\", got: %q", fullText)
	}
	if !strings.Contains(fullText, "2") {
		t.Errorf("expecting chunk to contain: \"2\", got: %q", fullText)
	}
	if !strings.Contains(fullText, "3") {
		t.Errorf("expecting chunk to contain: \"3\", got: %q", fullText)
	}
}

func TestWithConfig(t *testing.T) {
	tests := []struct {
		name     string
		config   any
		err      error
		validate func(*testing.T, *openai.ChatCompletionNewParams)
	}{
		{
			name:   "nil config",
			config: nil,
			validate: func(t *testing.T, request *openai.ChatCompletionNewParams) {
				// For nil config, we expect config fields to be unset (not nil, but with its zero value)
				if request.Temperature.Value != 0 {
					t.Errorf("expecting empty in temperature, got: %v", request.Temperature.Value)
				}
				if request.MaxCompletionTokens.Value != 0 {
					t.Errorf("expecting empty max completion tokens, got: %v", request.MaxCompletionTokens.Value)
				}
				if request.TopP.Value != 0 {
					t.Errorf("expecting empty in topP, got: %v", request.TopP.Value)
				}
				if len(request.Stop.OfStringArray) != 0 {
					t.Errorf("expecting empty stop reasons, got: %v", request.Stop)
				}
			},
		},
		{
			name:   "empty openai config",
			config: openai.ChatCompletionNewParams{},
			validate: func(t *testing.T, request *openai.ChatCompletionNewParams) {
				if request.Temperature.Value != 0 {
					t.Errorf("expecting empty in temperature, got: %v", request.Temperature.Value)
				}
				if request.MaxCompletionTokens.Value != 0 {
					t.Errorf("expecting empty max completion tokens, got: %v", request.MaxCompletionTokens.Value)
				}
				if request.TopP.Value != 0 {
					t.Errorf("expecting empty in topP, got: %v", request.TopP.Value)
				}
				if len(request.Stop.OfStringArray) != 0 {
					t.Errorf("expecting empty stop reasons, got: %v", request.Stop)
				}
			},
		},
		{
			name: "valid config with all supported fields",
			config: openai.ChatCompletionNewParams{
				Temperature:         openai.Float(0.7),
				MaxCompletionTokens: openai.Int(100),
				TopP:                openai.Float(0.9),
				Stop: openai.ChatCompletionNewParamsStopUnion{
					OfStringArray: []string{"stop1", "stop2"},
				},
			},
			validate: func(t *testing.T, request *openai.ChatCompletionNewParams) {
				// Check that fields are present and have correct values
				stopReasons := []string{"stop1, stop2"}
				if request.Temperature.Value != 0.7 {
					t.Errorf("expecting empty in temperature, got: %v", request.Temperature.Value)
				}
				if request.MaxCompletionTokens.Value != 100 {
					t.Errorf("expecting empty max completion tokens, got: %v", request.MaxCompletionTokens.Value)
				}
				if request.TopP.Value != 0.9 {
					t.Errorf("expecting empty in topP, got: %v", request.TopP.Value)
				}
				if slices.Equal(request.Stop.OfStringArray, stopReasons) {
					t.Errorf("diff in stop reasons, got: %v, want: %v", request.Stop.OfStringArray, stopReasons)
				}
			},
		},
		{
			name: "valid config as map",
			config: map[string]any{
				"temperature":           0.7,
				"max_completion_tokens": 100,
				"top_p":                 0.9,
				"stop":                  []string{"stop1", "stop2"},
			},
			validate: func(t *testing.T, request *openai.ChatCompletionNewParams) {
				stopReasons := []string{"stop1, stop2"}
				if request.Temperature.Value != 0.7 {
					t.Errorf("expecting empty in temperature, got: %v", request.Temperature.Value)
				}
				if request.MaxCompletionTokens.Value != 100 {
					t.Errorf("expecting empty max completion tokens, got: %v", request.MaxCompletionTokens.Value)
				}
				if request.TopP.Value != 0.9 {
					t.Errorf("expecting empty in topP, got: %v", request.TopP.Value)
				}
				if slices.Equal(request.Stop.OfStringArray, stopReasons) {
					t.Errorf("diff in stop reasons, got: %v, want: %v", request.Stop.OfStringArray, stopReasons)
				}
			},
		},
		{
			name:   "invalid config type",
			config: "not a config",
			err:    fmt.Errorf("unexpected config type: string"),
		},
	}

	// define simple messages for testing
	messages := []*ai.Message{
		{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewTextPart("Tell me a joke"),
			},
		},
	}
	req := &ai.ModelRequest{
		Messages: messages,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := setupTestClient(t)
			result, err := generator.WithMessages(messages).WithConfig(tt.config).Generate(context.Background(), req, nil)

			if tt.err != nil {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if got, want := err.Error(), tt.err.Error(); got != want {
					t.Errorf("error message = %q, want %q", got, want)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result == nil {
				t.Fatal("expected result, got nil")
			}

			// validate the input request was transformed correctly
			if tt.validate != nil {
				tt.validate(t, generator.GetRequest())
			}
		})
	}
}

func TestResponseFormat(t *testing.T) {
	tests := []struct {
		name     string
		output   *ai.ModelOutputConfig
		prompt   string
		validate func(*testing.T, *openai.ChatCompletionNewParams)
	}{
		{
			name: "json_schema",
			output: &ai.ModelOutputConfig{
				Format: "json",
				Schema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"joke": map[string]any{
							"type":        "string",
							"description": "A funny joke",
						},
					},
					"required":             []string{"joke"},
					"additionalProperties": false,
				},
			},
			prompt: "Tell me a joke",
			validate: func(t *testing.T, request *openai.ChatCompletionNewParams) {
				if request.ResponseFormat.OfJSONSchema == nil {
					t.Error("Expected ResponseFormat.OfJSONSchema to be set")
				}
				if request.ResponseFormat.OfJSONObject != nil {
					t.Error("Expected ResponseFormat.OfJSONObject to be nil for json_schema mode")
				}
				if request.ResponseFormat.OfText != nil {
					t.Error("Expected ResponseFormat.OfText to be nil for json_schema mode")
				}
			},
		},
		{
			name: "json_object",
			output: &ai.ModelOutputConfig{
				Format: "json",
			},
			prompt: "Generate a JSON object with a joke",
			validate: func(t *testing.T, request *openai.ChatCompletionNewParams) {
				if request.ResponseFormat.OfJSONObject == nil {
					t.Error("Expected ResponseFormat.OfJSONObject to be set")
				}
				if request.ResponseFormat.OfJSONSchema != nil {
					t.Error("Expected ResponseFormat.OfJSONSchema to be nil for json_object mode")
				}
				if request.ResponseFormat.OfText != nil {
					t.Error("Expected ResponseFormat.OfText to be nil for json_object mode")
				}
			},
		},
		{
			name: "text",
			output: &ai.ModelOutputConfig{
				Format: "text",
			},
			prompt: "Tell me a joke",
			validate: func(t *testing.T, request *openai.ChatCompletionNewParams) {
				if request.ResponseFormat.OfText == nil {
					t.Error("Expected ResponseFormat.OfText to be set")
				}
				if request.ResponseFormat.OfJSONObject != nil {
					t.Error("Expected ResponseFormat.OfJSONObject to be nil for text mode")
				}
				if request.ResponseFormat.OfJSONSchema != nil {
					t.Error("Expected ResponseFormat.OfJSONSchema to be nil for text mode")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := setupTestClient(t)

			messages := []*ai.Message{
				{
					Role: ai.RoleUser,
					Content: []*ai.Part{
						ai.NewTextPart(tt.prompt),
					},
				},
			}

			req := &ai.ModelRequest{
				Messages: messages,
				Output:   tt.output,
			}

			resp, err := g.WithMessages(messages).Generate(context.Background(), req, nil)
			if err != nil {
				t.Fatalf("Generate failed: %v", err)
			}

			if len(resp.Message.Content) == 0 {
				t.Fatal("Expected response content, got empty")
			}

			// Validate the request output was configured correctly
			if tt.validate != nil {
				tt.validate(t, g.GetRequest())
			}
		})
	}
}
