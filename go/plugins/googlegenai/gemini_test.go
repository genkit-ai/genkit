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
//
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"errors"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

func TestConvertRequest(t *testing.T) {
	text := "hello"
	tool := &ai.ToolDefinition{
		Description: "this is a dummy tool",
		Name:        "myTool",
		InputSchema: map[string]any{
			"additionalProperties": bool(false),
			"properties":           map[string]any{"Test": map[string]any{"type": string("string")}},
			"required":             []any{string("Test")},
			"type":                 string("object"),
		},
		OutputSchema: map[string]any{"type": string("string")},
	}

	req := &ai.ModelRequest{
		Config: genai.GenerateContentConfig{
			MaxOutputTokens: 10,
			StopSequences:   []string{"stop"},
			Temperature:     genai.Ptr[float32](0.4),
			TopK:            genai.Ptr[float32](0.1),
			TopP:            genai.Ptr[float32](1.0),
			Tools: []*genai.Tool{
				{GoogleSearch: &genai.GoogleSearch{}},
			},
			ThinkingConfig: &genai.ThinkingConfig{
				IncludeThoughts: false,
				ThinkingBudget:  genai.Ptr[int32](0),
			},
		},
		Tools:      []*ai.ToolDefinition{tool},
		ToolChoice: ai.ToolChoiceAuto,
		Output: &ai.ModelOutputConfig{
			Constrained: true,
			Schema: map[string]any{
				"type": string("object"),
				"properties": map[string]any{
					"string": map[string]any{
						"type": string("string"),
					},
					"boolean": map[string]any{
						"type": string("boolean"),
					},
					"float": map[string]any{
						"type": string("float64"),
					},
					"number": map[string]any{
						"type": string("number"),
					},
					"array": map[string]any{
						"type": string("array"),
					},
					"object": map[string]any{
						"type": string("object"),
					},
					"domain": map[string]any{
						"anyOf": []map[string]any{
							{
								"type": string("string"),
							},
							{
								"type": string("null"),
							},
						},
						"default": "null",
						"title":   string("Domain"),
					},
				},
			},
		},
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					{Text: text},
				},
			},
			{
				Role: ai.RoleSystem,
				Content: []*ai.Part{
					{Text: text},
				},
			},
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					{Text: text},
				},
			},
			{
				Role: ai.RoleSystem,
				Content: []*ai.Part{
					{Text: text},
				},
			},
		},
	}
	t.Run("convert request", func(t *testing.T) {
		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatal(err)
		}
		if gcc.SystemInstruction == nil {
			t.Error("expecting system instructions to be populated")
		}
		if len(gcc.SystemInstruction.Parts) != 2 {
			t.Errorf("got: %d, want: 2", len(gcc.SystemInstruction.Parts))
		}
		if gcc.SystemInstruction.Role != string(ai.RoleSystem) {
			t.Errorf(" system instruction role: got: %q, want: %q", gcc.SystemInstruction.Role, string(ai.RoleSystem))
		}
		// this is explicitly set to 1 in source
		if gcc.CandidateCount == 0 {
			t.Error("candidate count: got: 0, want: 1")
		}
		ogCfg, ok := req.Config.(genai.GenerateContentConfig)
		if !ok {
			t.Fatalf("request config should have been of type: genai.GenerateContentConfig, got: %T", req.Config)
		}
		if gcc.MaxOutputTokens == 0 {
			t.Errorf("max output tokens: got: 0, want %d", ogCfg.MaxOutputTokens)
		}
		if len(gcc.StopSequences) == 0 {
			t.Errorf("stop sequences: got: 0, want: %d", len(ogCfg.StopSequences))
		}
		if gcc.Temperature == nil {
			t.Errorf("temperature: got: nil, want %f", *ogCfg.Temperature)
		}
		if gcc.TopP == nil {
			t.Errorf("topP: got: nil, want %f", *ogCfg.TopP)
		}
		if gcc.TopK == nil {
			t.Errorf("topK: got: nil, want %d", ogCfg.TopK)
		}
		if gcc.ResponseMIMEType != "" {
			t.Errorf("ResponseMIMEType should be empty if tools are present")
		}
		if gcc.ResponseSchema != nil {
			t.Errorf("ResponseSchema should be nil when tools are present (JSON mode is not compatible with tools)")
		}
		if gcc.ThinkingConfig == nil {
			t.Errorf("ThinkingConfig should not be empty")
		}
		// With the merge fix, we should have 2 tools:
		// - GoogleSearch from config.Tools (preserved)
		// - FunctionDeclarations from input.Tools (merged)
		if len(gcc.Tools) != 2 {
			t.Errorf("tools should have been: 2, got: %d", len(gcc.Tools))
		}
		// Verify GoogleSearch was preserved
		hasGoogleSearch := false
		hasFunctionDecl := false
		for _, tool := range gcc.Tools {
			if tool.GoogleSearch != nil {
				hasGoogleSearch = true
			}
			if tool.FunctionDeclarations != nil {
				hasFunctionDecl = true
			}
		}
		if !hasGoogleSearch {
			t.Error("GoogleSearch tool was dropped during merge")
		}
		if !hasFunctionDecl {
			t.Error("FunctionDeclarations were not added")
		}
	})
	t.Run("use valid tools outside genkit", func(t *testing.T) {
		badCfg := genai.GenerateContentConfig{
			Temperature: genai.Ptr[float32](1.0),
			Tools: []*genai.Tool{
				{
					CodeExecution: &genai.ToolCodeExecution{},
					GoogleSearch:  &genai.GoogleSearch{},
				},
			},
		}
		req := ai.ModelRequest{
			Config: badCfg,
		}
		_, err := toGeminiRequest(&req, nil)
		if err != nil {
			t.Fatalf("expected nil, got: %v", err)
		}
	})
	t.Run("forbidden primitives outside genkit", func(t *testing.T) {
		type testCase struct {
			name string
			cfg  genai.GenerateContentConfig
			err  error
		}
		tests := []testCase{
			{
				name: "use system instruction outside genkit",
				cfg: genai.GenerateContentConfig{
					Temperature:       genai.Ptr[float32](1.0),
					SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "talk like a pirate"}}},
				},
				err: errors.New("system instruction should be set using Genkit features"),
			},
			// Note: FunctionDeclarations in config.Tools are now allowed and merged
			// with ai.WithTools() declarations instead of being rejected.
			{
				name: "use cache outside genkit",
				cfg: genai.GenerateContentConfig{
					CachedContent: "some cache uuid",
				},
				err: errors.New("cache contents should be set using Genkit features"),
			},
			{
				name: "use response schema outside genkit",
				cfg: genai.GenerateContentConfig{
					ResponseSchema: &genai.Schema{
						Description: "some schema",
					},
				},
				err: errors.New("response schema should be set using Genkit features"),
			},
			{
				name: "use response MIME type outside genkit",
				cfg: genai.GenerateContentConfig{
					ResponseMIMEType: "image/png",
				},
				err: errors.New("response schema should be set using Genkit features"),
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				req := ai.ModelRequest{
					Config: tc.cfg,
				}
				_, err := toGeminiRequest(&req, nil)
				if err == nil {
					t.Fatalf("expected an error: '%v' but got nil", tc.err)
				}
			})
		}
	})
	t.Run("invalid config map", func(t *testing.T) {
		req := ai.ModelRequest{
			Config: map[string]any{
				"temperature": "not a number", // This should fail map->struct conversion
			},
		}
		_, err := toGeminiRequest(&req, nil)
		if err == nil {
			t.Fatal("expected error for invalid config map")
		}
	})
	t.Run("convert tools with valid tool", func(t *testing.T) {
		tools := []*ai.ToolDefinition{tool}
		gt, err := toGeminiTools(tools)
		if err != nil {
			t.Fatalf("expected tool convertion but got error: %v", err)
		}
		for _, tt := range gt {
			for _, fd := range tt.FunctionDeclarations {
				if fd.Description == "" {
					t.Error("expecting tool description, got empty")
				}
				if fd.Name == "" {
					t.Error("expecting tool name, got empty")
				}
				if fd.Parameters == nil {
					t.Error("expecting parameters, got empty")
				}
			}
		}
	})
	t.Run("convert tools with empty tools", func(t *testing.T) {
		tools := []*ai.ToolDefinition{}
		gt, err := toGeminiTools(tools)
		if err != nil {
			t.Fatal("should not expect errors")
		}
		if gt != nil {
			t.Fatalf("should expect an empty tool list, got %#v", gt)
		}
	})
	t.Run("convert tools with invalid name", func(t *testing.T) {
		tools := []*ai.ToolDefinition{{
			Description:  tool.Description,
			InputSchema:  tool.InputSchema,
			OutputSchema: tool.OutputSchema,
			Name:         "something/myTool", // '/' is not a valid character for a Gemini tool name
		}}
		_, err := toGeminiTools(tools)
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
	})
}

// TestToolMerging tests that ai.WithTools() merges with existing Gemini-specific tools
// instead of replacing them. This enables using Genkit tools alongside FileSearch,
// GoogleSearch, and CodeExecution.
func TestToolMerging(t *testing.T) {
	genkitTool := &ai.ToolDefinition{
		Name:        "my_function",
		Description: "A test function for tool merging",
		InputSchema: map[string]any{"type": "object"},
	}

	t.Run("preserves Retrieval when adding Genkit tools", func(t *testing.T) {
		req := &ai.ModelRequest{
			Config: genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.5),
				Tools: []*genai.Tool{
					{
						Retrieval: &genai.Retrieval{
							VertexAISearch: &genai.VertexAISearch{
								Datastore: "test-datastore",
							},
						},
					},
				},
			},
			Tools: []*ai.ToolDefinition{genkitTool},
			Messages: []*ai.Message{
				{Role: ai.RoleUser, Content: []*ai.Part{{Text: "test"}}},
			},
		}

		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatalf("toGeminiRequest failed: %v", err)
		}

		hasRetrieval := false
		hasFunctionDecl := false

		for _, tool := range gcc.Tools {
			if tool.Retrieval != nil {
				hasRetrieval = true
				// Verify Retrieval content was preserved
				if tool.Retrieval.VertexAISearch == nil ||
					tool.Retrieval.VertexAISearch.Datastore != "test-datastore" {
					t.Error("Retrieval datastore was modified")
				}
			}
			if tool.FunctionDeclarations != nil {
				hasFunctionDecl = true
			}
		}

		if !hasRetrieval {
			t.Error("Retrieval tool was dropped during merge")
		}
		if !hasFunctionDecl {
			t.Error("Function declarations were not added")
		}
	})

	t.Run("preserves GoogleSearch when adding Genkit tools", func(t *testing.T) {
		req := &ai.ModelRequest{
			Config: genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.5),
				Tools: []*genai.Tool{
					{GoogleSearch: &genai.GoogleSearch{}},
				},
			},
			Tools: []*ai.ToolDefinition{genkitTool},
			Messages: []*ai.Message{
				{Role: ai.RoleUser, Content: []*ai.Part{{Text: "test"}}},
			},
		}

		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatalf("toGeminiRequest failed: %v", err)
		}

		hasGoogleSearch := false
		for _, tool := range gcc.Tools {
			if tool.GoogleSearch != nil {
				hasGoogleSearch = true
				break
			}
		}

		if !hasGoogleSearch {
			t.Error("GoogleSearch tool was dropped during merge")
		}
	})

	t.Run("preserves CodeExecution when adding Genkit tools", func(t *testing.T) {
		req := &ai.ModelRequest{
			Config: genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.5),
				Tools: []*genai.Tool{
					{CodeExecution: &genai.ToolCodeExecution{}},
				},
			},
			Tools: []*ai.ToolDefinition{genkitTool},
			Messages: []*ai.Message{
				{Role: ai.RoleUser, Content: []*ai.Part{{Text: "test"}}},
			},
		}

		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatalf("toGeminiRequest failed: %v", err)
		}

		hasCodeExec := false
		for _, tool := range gcc.Tools {
			if tool.CodeExecution != nil {
				hasCodeExec = true
				break
			}
		}

		if !hasCodeExec {
			t.Error("CodeExecution tool was dropped during merge")
		}
	})

	t.Run("preserves multiple Gemini tools when adding Genkit tools", func(t *testing.T) {
		req := &ai.ModelRequest{
			Config: genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.5),
				Tools: []*genai.Tool{
					{
						Retrieval: &genai.Retrieval{
							VertexAISearch: &genai.VertexAISearch{
								Datastore: "test-datastore",
							},
						},
					},
					{GoogleSearch: &genai.GoogleSearch{}},
					{CodeExecution: &genai.ToolCodeExecution{}},
				},
			},
			Tools: []*ai.ToolDefinition{genkitTool},
			Messages: []*ai.Message{
				{Role: ai.RoleUser, Content: []*ai.Part{{Text: "test"}}},
			},
		}

		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatalf("toGeminiRequest failed: %v", err)
		}

		hasRetrieval := false
		hasGoogleSearch := false
		hasCodeExec := false
		hasFunctionDecl := false

		for _, tool := range gcc.Tools {
			if tool.Retrieval != nil {
				hasRetrieval = true
			}
			if tool.GoogleSearch != nil {
				hasGoogleSearch = true
			}
			if tool.CodeExecution != nil {
				hasCodeExec = true
			}
			if tool.FunctionDeclarations != nil {
				hasFunctionDecl = true
			}
		}

		if !hasRetrieval {
			t.Error("Retrieval tool was dropped during merge")
		}
		if !hasGoogleSearch {
			t.Error("GoogleSearch tool was dropped during merge")
		}
		if !hasCodeExec {
			t.Error("CodeExecution tool was dropped during merge")
		}
		if !hasFunctionDecl {
			t.Error("Function declarations were not added")
		}
	})

	t.Run("works when no existing tools in config", func(t *testing.T) {
		req := &ai.ModelRequest{
			Config: genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.5),
			},
			Tools: []*ai.ToolDefinition{genkitTool},
			Messages: []*ai.Message{
				{Role: ai.RoleUser, Content: []*ai.Part{{Text: "test"}}},
			},
		}

		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatalf("toGeminiRequest failed: %v", err)
		}

		if len(gcc.Tools) != 1 {
			t.Errorf("expected 1 tool, got %d", len(gcc.Tools))
		}

		hasFunctionDecl := false
		for _, tool := range gcc.Tools {
			if tool.FunctionDeclarations != nil {
				hasFunctionDecl = true
			}
		}

		if !hasFunctionDecl {
			t.Error("Function declarations were not added")
		}
	})

	t.Run("merges multiple Genkit tools correctly", func(t *testing.T) {
		anotherTool := &ai.ToolDefinition{
			Name:        "another_function",
			Description: "Another test function",
			InputSchema: map[string]any{"type": "object"},
		}

		req := &ai.ModelRequest{
			Config: genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.5),
				Tools: []*genai.Tool{
					{
						Retrieval: &genai.Retrieval{
							VertexAISearch: &genai.VertexAISearch{
								Datastore: "test-datastore",
							},
						},
					},
				},
			},
			Tools: []*ai.ToolDefinition{genkitTool, anotherTool},
			Messages: []*ai.Message{
				{Role: ai.RoleUser, Content: []*ai.Part{{Text: "test"}}},
			},
		}

		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatalf("toGeminiRequest failed: %v", err)
		}

		hasRetrieval := false
		funcDeclCount := 0

		for _, tool := range gcc.Tools {
			if tool.Retrieval != nil {
				hasRetrieval = true
			}
			if tool.FunctionDeclarations != nil {
				funcDeclCount += len(tool.FunctionDeclarations)
			}
		}

		if !hasRetrieval {
			t.Error("Retrieval tool was dropped during merge")
		}
		if funcDeclCount != 2 {
			t.Errorf("expected 2 function declarations, got %d", funcDeclCount)
		}
	})

	t.Run("merges FunctionDeclarations from config with Genkit tools", func(t *testing.T) {
		// This tests the case where FunctionDeclarations exist in both
		// config.Tools AND input.Tools - they should all be merged.
		configFuncDecl := &genai.FunctionDeclaration{
			Name:        "config_function",
			Description: "A function from config",
		}

		req := &ai.ModelRequest{
			Config: genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.5),
				Tools: []*genai.Tool{
					{
						FunctionDeclarations: []*genai.FunctionDeclaration{configFuncDecl},
						GoogleSearch:         &genai.GoogleSearch{}, // hybrid tool
					},
				},
			},
			Tools: []*ai.ToolDefinition{genkitTool},
			Messages: []*ai.Message{
				{Role: ai.RoleUser, Content: []*ai.Part{{Text: "test"}}},
			},
		}

		gcc, err := toGeminiRequest(req, nil)
		if err != nil {
			t.Fatalf("toGeminiRequest failed: %v", err)
		}

		// Should have: 1 tool with all FunctionDeclarations, 1 tool with GoogleSearch
		hasGoogleSearch := false
		funcDeclCount := 0
		var funcNames []string

		for _, tool := range gcc.Tools {
			if tool.GoogleSearch != nil {
				hasGoogleSearch = true
			}
			if tool.FunctionDeclarations != nil {
				for _, fd := range tool.FunctionDeclarations {
					funcDeclCount++
					funcNames = append(funcNames, fd.Name)
				}
			}
		}

		if !hasGoogleSearch {
			t.Error("GoogleSearch was dropped during merge")
		}
		if funcDeclCount != 2 {
			t.Errorf("expected 2 function declarations (1 from config + 1 from input.Tools), got %d: %v", funcDeclCount, funcNames)
		}
	})
}

func TestValidToolName(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name:     "Valid single letter",
			input:    "a",
			expected: true,
		},
		{
			name:     "Valid single underscore",
			input:    "_",
			expected: true,
		},
		{
			name:     "Valid alphanumeric with underscore",
			input:    "my_tool",
			expected: true,
		},
		{
			name:     "Valid alphanumeric with dot and hyphen",
			input:    "user.name-id",
			expected: true,
		},
		{
			name:     "Valid max length",
			input:    "a" + genToolName(63, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"),
			expected: true,
		},
		{
			name:     "Invalid starts with digit",
			input:    "1tool",
			expected: false,
		},
		{
			name:     "Invalid starts with hyphen",
			input:    "-tool",
			expected: false,
		},
		{
			name:     "Invalid starts with dot",
			input:    ".tool",
			expected: false,
		},
		{
			name:     "Invalid contains space",
			input:    "my tool",
			expected: false,
		},
		{
			name:     "Invalid contains special character",
			input:    "my$tool",
			expected: false,
		},
		{
			name:     "Invalid empty string",
			input:    "",
			expected: false,
		},
		{
			name:     "Invalid too long",
			input:    "a" + genToolName(64, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"),
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := validToolName(tc.input)
			if got != tc.expected {
				t.Errorf("Test %q failed: expected: %v, got: %v", tc.name, tc.expected, got)
			}
		})
	}
}

func TestToGeminiParts_MultipartToolResponse(t *testing.T) {
	t.Run("ValidPartType", func(t *testing.T) {
		// Create a tool response with both output and additional content (media)
		toolResp := &ai.ToolResponse{
			Name:   "generateImage",
			Output: map[string]any{"status": "success"},
			Content: []*ai.Part{
				ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="),
			},
		}

		// create a mock ToolResponsePart, setting "multipart" to true is required
		part := ai.NewToolResponsePart(toolResp)
		part.Metadata = map[string]any{"multipart": true}

		geminiParts, err := toGeminiParts([]*ai.Part{part})
		if err != nil {
			t.Fatalf("toGeminiParts failed: %v", err)
		}

		// Expecting 1 part which contains the function response with internal parts
		if len(geminiParts) != 1 {
			t.Fatalf("expected 1 Gemini part, got %d", len(geminiParts))
		}

		if geminiParts[0].FunctionResponse == nil {
			t.Error("expected first part to be FunctionResponse")
		}
		if geminiParts[0].FunctionResponse.Name != "generateImage" {
			t.Errorf("expected function name 'generateImage', got %q", geminiParts[0].FunctionResponse.Name)
		}
	})

	t.Run("UnsupportedPartType", func(t *testing.T) {
		// Create a tool response with text content (unsupported for multipart)
		toolResp := &ai.ToolResponse{
			Name:   "generateText",
			Output: map[string]any{"status": "success"},
			Content: []*ai.Part{
				ai.NewTextPart("Generated text"),
			},
		}

		part := ai.NewToolResponsePart(toolResp)
		part.Metadata = map[string]any{"multipart": true}

		_, err := toGeminiParts([]*ai.Part{part})
		if err == nil {
			t.Fatal("expected error for unsupported text part in multipart response, got nil")
		}
	})
}

func TestToGeminiParts_SimpleToolResponse(t *testing.T) {
	// Create a simple tool response (no content)
	toolResp := &ai.ToolResponse{
		Name:   "search",
		Output: map[string]any{"result": "foo"},
	}

	part := ai.NewToolResponsePart(toolResp)

	geminiParts, err := toGeminiParts([]*ai.Part{part})
	if err != nil {
		t.Fatalf("toGeminiParts failed: %v", err)
	}

	if len(geminiParts) != 1 {
		t.Fatalf("expected 1 Gemini part, got %d", len(geminiParts))
	}

	if geminiParts[0].FunctionResponse == nil {
		t.Error("expected part to be FunctionResponse")
	}
}

// genToolName generates a string of a specified length using only
// the valid characters for a Gemini Tool name
func genToolName(length int, chars string) string {
	r := make([]byte, length)

	for i := range length {
		r[i] = chars[i%len(chars)]
	}
	return string(r)
}

// TestThoughtSignatureRoundTrip tests that thought signatures are properly preserved
// when converting between Genkit and Gemini part formats.
func TestThoughtSignatureRoundTrip(t *testing.T) {
	testSignature := []byte("test-thought-signature-abc123")

	t.Run("text part preserves signature", func(t *testing.T) {
		// Create a Genkit text part with a signature
		genkitPart := ai.NewTextPart("Hello world")
		genkitPart.Metadata = map[string]any{"signature": testSignature}

		// Convert to Gemini part
		geminiPart, err := toGeminiPart(genkitPart)
		if err != nil {
			t.Fatalf("toGeminiPart failed: %v", err)
		}

		// Verify signature was restored
		if geminiPart.ThoughtSignature == nil {
			t.Error("expected ThoughtSignature to be set on Gemini part")
		}
		if string(geminiPart.ThoughtSignature) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", geminiPart.ThoughtSignature, testSignature)
		}
	})

	t.Run("reasoning part preserves signature", func(t *testing.T) {
		// Create a Genkit reasoning part (signature is embedded via NewReasoningPart)
		genkitPart := ai.NewReasoningPart("I'm thinking about this...", testSignature)

		// Convert to Gemini part
		geminiPart, err := toGeminiPart(genkitPart)
		if err != nil {
			t.Fatalf("toGeminiPart failed: %v", err)
		}

		// Verify it's marked as a thought
		if !geminiPart.Thought {
			t.Error("expected Thought to be true on Gemini part")
		}

		// Verify signature was restored
		if geminiPart.ThoughtSignature == nil {
			t.Error("expected ThoughtSignature to be set on Gemini part")
		}
		if string(geminiPart.ThoughtSignature) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", geminiPart.ThoughtSignature, testSignature)
		}
	})

	t.Run("tool request part preserves signature", func(t *testing.T) {
		// Create a Genkit tool request part with a signature
		genkitPart := ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  "myTool",
			Input: map[string]any{"arg": "value"},
		})
		genkitPart.Metadata = map[string]any{"signature": testSignature}

		// Convert to Gemini part
		geminiPart, err := toGeminiPart(genkitPart)
		if err != nil {
			t.Fatalf("toGeminiPart failed: %v", err)
		}

		// Verify it's a function call
		if geminiPart.FunctionCall == nil {
			t.Fatal("expected FunctionCall to be set on Gemini part")
		}

		// Verify signature was restored
		if geminiPart.ThoughtSignature == nil {
			t.Error("expected ThoughtSignature to be set on Gemini part")
		}
		if string(geminiPart.ThoughtSignature) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", geminiPart.ThoughtSignature, testSignature)
		}
	})

	t.Run("tool response part preserves signature", func(t *testing.T) {
		// Create a Genkit tool response part with a signature
		genkitPart := ai.NewToolResponsePart(&ai.ToolResponse{
			Name:   "myTool",
			Output: map[string]any{"result": "success"},
		})
		genkitPart.Metadata = map[string]any{"signature": testSignature}

		// Convert to Gemini part
		geminiPart, err := toGeminiPart(genkitPart)
		if err != nil {
			t.Fatalf("toGeminiPart failed: %v", err)
		}

		// Verify it's a function response
		if geminiPart.FunctionResponse == nil {
			t.Fatal("expected FunctionResponse to be set on Gemini part")
		}

		// Verify signature was restored
		if geminiPart.ThoughtSignature == nil {
			t.Error("expected ThoughtSignature to be set on Gemini part")
		}
		if string(geminiPart.ThoughtSignature) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", geminiPart.ThoughtSignature, testSignature)
		}
	})

	t.Run("multipart tool response preserves signature", func(t *testing.T) {
		// Create a multipart tool response with media content
		genkitPart := ai.NewToolResponsePart(&ai.ToolResponse{
			Name:   "generateImage",
			Output: map[string]any{"status": "success"},
			Content: []*ai.Part{
				ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="),
			},
		})
		genkitPart.Metadata = map[string]any{
			"multipart": true,
			"signature": testSignature,
		}

		// Convert to Gemini part
		geminiPart, err := toGeminiPart(genkitPart)
		if err != nil {
			t.Fatalf("toGeminiPart failed: %v", err)
		}

		// Verify it's a function response
		if geminiPart.FunctionResponse == nil {
			t.Fatal("expected FunctionResponse to be set on Gemini part")
		}

		// Verify signature was restored
		if geminiPart.ThoughtSignature == nil {
			t.Error("expected ThoughtSignature to be set on Gemini part")
		}
		if string(geminiPart.ThoughtSignature) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", geminiPart.ThoughtSignature, testSignature)
		}
	})
}

// TestTranslateCandidateThoughtSignature tests that thought signatures from Gemini
// responses are properly extracted and stored in Genkit parts.
func TestTranslateCandidateThoughtSignature(t *testing.T) {
	testSignature := []byte("response-thought-signature-xyz789")

	t.Run("extracts signature from text part", func(t *testing.T) {
		candidate := &genai.Candidate{
			FinishReason: genai.FinishReasonStop,
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						Text:             "Hello world",
						ThoughtSignature: testSignature,
					},
				},
			},
		}

		resp, err := translateCandidate(candidate)
		if err != nil {
			t.Fatalf("translateCandidate failed: %v", err)
		}

		if len(resp.Message.Content) != 1 {
			t.Fatalf("expected 1 part, got %d", len(resp.Message.Content))
		}

		part := resp.Message.Content[0]
		if part.Metadata == nil {
			t.Fatal("expected Metadata to be set")
		}

		sig, ok := part.Metadata["signature"].([]byte)
		if !ok {
			t.Fatal("expected signature in metadata")
		}
		if string(sig) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", sig, testSignature)
		}
	})

	t.Run("extracts signature from thought part", func(t *testing.T) {
		candidate := &genai.Candidate{
			FinishReason: genai.FinishReasonStop,
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						Text:             "Let me think about this...",
						Thought:          true,
						ThoughtSignature: testSignature,
					},
				},
			},
		}

		resp, err := translateCandidate(candidate)
		if err != nil {
			t.Fatalf("translateCandidate failed: %v", err)
		}

		if len(resp.Message.Content) != 1 {
			t.Fatalf("expected 1 part, got %d", len(resp.Message.Content))
		}

		part := resp.Message.Content[0]
		if !part.IsReasoning() {
			t.Error("expected part to be reasoning")
		}
		if part.Metadata == nil {
			t.Fatal("expected Metadata to be set")
		}

		sig, ok := part.Metadata["signature"].([]byte)
		if !ok {
			t.Fatal("expected signature in metadata")
		}
		if string(sig) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", sig, testSignature)
		}
	})

	t.Run("extracts signature from function call part", func(t *testing.T) {
		candidate := &genai.Candidate{
			FinishReason: genai.FinishReasonStop,
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							Name: "myTool",
							Args: map[string]any{"arg": "value"},
						},
						ThoughtSignature: testSignature,
					},
				},
			},
		}

		resp, err := translateCandidate(candidate)
		if err != nil {
			t.Fatalf("translateCandidate failed: %v", err)
		}

		if len(resp.Message.Content) != 1 {
			t.Fatalf("expected 1 part, got %d", len(resp.Message.Content))
		}

		part := resp.Message.Content[0]
		if !part.IsToolRequest() {
			t.Error("expected part to be tool request")
		}
		if part.Metadata == nil {
			t.Fatal("expected Metadata to be set")
		}

		sig, ok := part.Metadata["signature"].([]byte)
		if !ok {
			t.Fatal("expected signature in metadata")
		}
		if string(sig) != string(testSignature) {
			t.Errorf("signature mismatch: got %q, want %q", sig, testSignature)
		}
	})
}

// TestFinishReasonMapping tests the mapping of Gemini finish reasons to Genkit finish reasons.
func TestFinishReasonMapping(t *testing.T) {
	testCases := []struct {
		name           string
		geminiReason   genai.FinishReason
		expectedReason ai.FinishReason
	}{
		{"stop", genai.FinishReasonStop, ai.FinishReasonStop},
		{"max tokens", genai.FinishReasonMaxTokens, ai.FinishReasonLength},
		{"safety", genai.FinishReasonSafety, ai.FinishReasonBlocked},
		{"recitation", genai.FinishReasonRecitation, ai.FinishReasonBlocked},
		{"language", genai.FinishReasonLanguage, ai.FinishReasonBlocked},
		{"blocklist", genai.FinishReasonBlocklist, ai.FinishReasonBlocked},
		{"prohibited content", genai.FinishReasonProhibitedContent, ai.FinishReasonBlocked},
		{"spii", genai.FinishReasonSPII, ai.FinishReasonBlocked},
		{"image safety", genai.FinishReasonImageSafety, ai.FinishReasonBlocked},
		{"image prohibited content", genai.FinishReasonImageProhibitedContent, ai.FinishReasonBlocked},
		{"image recitation", genai.FinishReasonImageRecitation, ai.FinishReasonBlocked},
		{"malformed function call", genai.FinishReasonMalformedFunctionCall, ai.FinishReasonOther},
		{"unexpected tool call", genai.FinishReasonUnexpectedToolCall, ai.FinishReasonOther},
		{"no image", genai.FinishReasonNoImage, ai.FinishReasonOther},
		{"image other", genai.FinishReasonImageOther, ai.FinishReasonOther},
		{"other", genai.FinishReasonOther, ai.FinishReasonOther},
		{"missing thought signature", "MISSING_THOUGHT_SIGNATURE", ai.FinishReasonOther},
		{"unknown reason", "SOME_FUTURE_REASON", ai.FinishReasonUnknown},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			candidate := &genai.Candidate{
				FinishReason: tc.geminiReason,
				Content: &genai.Content{
					Role:  "model",
					Parts: []*genai.Part{{Text: "test"}},
				},
			}

			resp, err := translateCandidate(candidate)
			if err != nil {
				t.Fatalf("translateCandidate failed: %v", err)
			}

			if resp.FinishReason != tc.expectedReason {
				t.Errorf("finish reason mismatch: got %q, want %q", resp.FinishReason, tc.expectedReason)
			}
		})
	}
}
