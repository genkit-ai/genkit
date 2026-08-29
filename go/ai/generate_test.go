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

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/registry"
	test_utils "github.com/firebase/genkit/go/tests/utils"
	"github.com/google/go-cmp/cmp"
)

type StructuredResponse struct {
	Subject  string
	Location string
}

var r = registry.New()

func init() {
	// Set up default formats
	ConfigureFormats(r)
	// Register the generate action that Generate() function expects
	DefineGenerateAction(context.Background(), r)
}

// echoModel attributes
var (
	modelName = "echo"
	metadata  = ModelOptions{
		Label: modelName,
		Supports: &ModelSupports{
			Multiturn:   true,
			Tools:       true,
			SystemRole:  true,
			Media:       false,
			Constrained: ConstrainedSupportNone,
		},
		Versions: []string{"echo-001", "echo-002"},
		Stage:    ModelStageDeprecated,
	}

	echoModel = defineModel(r, "test/"+modelName, &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		if msc != nil {
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewTextPart("stream!")},
			})
		}
		textResponse := ""
		for _, m := range gr.Messages {
			if m.Role == RoleUser {
				textResponse = m.Text()
			}
		}
		return &ModelResponse{
			Request: gr,
			Message: NewModelTextMessage(textResponse),
		}, nil
	})
)

// with tools
var gablorkenTool = defineTool(r, "gablorken", "use when need to calculate a gablorken",
	func(ctx *ToolContext, input struct {
		Value float64
		Over  float64
	},
	) (float64, error) {
		return math.Pow(input.Value, input.Over), nil
	},
)

func TestStreamingChunksHaveRoleAndIndex(t *testing.T) {
	t.Parallel()

	r := childRegistry(t)
	ctx := context.Background()

	convertTempTool := defineTool(r, "convertTemp", "converts temperature",
		func(ctx *ToolContext, input struct {
			From        string
			To          string
			Temperature float64
		},
		) (float64, error) {
			if input.From == "celsius" && input.To == "fahrenheit" {
				return input.Temperature*9/5 + 32, nil
			}
			return input.Temperature, nil
		},
	)

	toolModel := defineModel(r, "test/toolModel", &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		hasToolResponse := false
		for _, msg := range gr.Messages {
			if msg.Role == RoleTool {
				hasToolResponse = true
				break
			}
		}

		if hasToolResponse {
			if msc != nil {
				msc(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("20 degrees Celsius is 68 degrees Fahrenheit.")},
				})
			}
			return &ModelResponse{
				Request: gr,
				Message: NewModelTextMessage("20 degrees Celsius is 68 degrees Fahrenheit."),
			}, nil
		}

		if msc != nil {
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewToolRequestPart(&ToolRequest{
					Name: "convertTemp",
					Input: map[string]any{
						"From":        "celsius",
						"To":          "fahrenheit",
						"Temperature": 20.0,
					},
					Ref: "0",
				})},
			})
		}
		return &ModelResponse{
			Request: gr,
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{NewToolRequestPart(&ToolRequest{
					Name: "convertTemp",
					Input: map[string]any{
						"From":        "celsius",
						"To":          "fahrenheit",
						"Temperature": 20.0,
					},
					Ref: "0",
				})},
			},
		}, nil
	})

	var chunks []*ModelResponseChunk
	_, err := Generate(ctx, r,
		WithModel(toolModel),
		WithMessages(NewUserTextMessage("convert 20 c to f")),
		WithTools(convertTempTool),
		WithStreaming(func(ctx context.Context, chunk *ModelResponseChunk) error {
			chunks = append(chunks, chunk)
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if len(chunks) < 2 {
		t.Fatalf("Expected at least 2 chunks, got %d", len(chunks))
	}

	for i, chunk := range chunks {
		if chunk.Role == "" {
			t.Errorf("Chunk %d: Role is empty", i)
		}
		t.Logf("Chunk %d: Role=%s, Index=%d", i, chunk.Role, chunk.Index)
	}

	if chunks[0].Role != RoleModel {
		t.Errorf("Expected first chunk to have role 'model', got %s", chunks[0].Role)
	}
	if chunks[0].Index != 0 {
		t.Errorf("Expected first chunk to have index 0, got %d", chunks[0].Index)
	}

	toolChunkFound := false
	for _, chunk := range chunks {
		if chunk.Role == RoleTool {
			toolChunkFound = true
			if chunk.Index != 1 {
				t.Errorf("Expected tool chunk to have index 1, got %d", chunk.Index)
			}
		}
	}
	if !toolChunkFound {
		t.Error("Expected to find at least one tool chunk")
	}
}

func TestValidMessage(t *testing.T) {
	t.Parallel()

	t.Run("Valid message with text format", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart("Hello, World!"),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatText,
		}
		_, err := validTestMessage(message, outputSchema)
		if err != nil {
			t.Fatal(err)
		}
	})

	t.Run("Valid message with JSON format and matching schema", func(t *testing.T) {
		json := `{
			"name": "John",
			"age": 30,
			"address": {
				"street": "123 Main St",
				"city": "New York",
				"country": "USA"
			}
		}`
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(json)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type":     "object",
				"required": []string{"name", "age", "address"},
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
					"address": map[string]any{
						"type":     "object",
						"required": []string{"street", "city", "country"},
						"properties": map[string]any{
							"street":  map[string]any{"type": "string"},
							"city":    map[string]any{"type": "string"},
							"country": map[string]any{"type": "string"},
						},
					},
					"phone": map[string]any{"type": "string"},
				},
			},
		}
		message, err := validTestMessage(message, outputSchema)
		if err != nil {
			t.Fatal(err)
		}
		text := message.Text()
		if strings.TrimSpace(text) != strings.TrimSpace(json) {
			t.Fatalf("got %q, want %q", json, text)
		}
	})

	t.Run("Invalid message with JSON format and non-matching schema", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "age": "30"}`)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
			},
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "data did not match expected schema")
	})

	t.Run("Message with invalid JSON", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "age": 30`)), // Missing trailing }.
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
		}
		_, err := validTestMessage(message, outputSchema)
		t.Log(err)
		errorContains(t, err, "not a valid JSON")
	})

	t.Run("No message", func(t *testing.T) {
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
		}
		_, err := validTestMessage(nil, outputSchema)
		errorContains(t, err, "message is empty")
	})

	t.Run("Empty message", func(t *testing.T) {
		message := &Message{}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "message has no content")
	})

	t.Run("Candidate contains unexpected field", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "height": 190}`)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "data did not match expected schema")
	})

	t.Run("Invalid expected schema", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "age": 30}`)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type": "invalid",
			},
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "failed to validate data against expected schema")
	})
}

func TestGenerate(t *testing.T) {
	r := childRegistry(t)
	JSON := "{\"subject\": \"bananas\", \"location\": \"tropics\"}"
	JSONmd := "```json" + JSON + "```"

	bananaModel := defineModel(r, "test/banana", &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		if msc != nil {
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewTextPart("stream!")},
			})
		}

		return &ModelResponse{
			Request: gr,
			Message: NewModelTextMessage(JSONmd),
		}, nil
	})

	t.Run("constructs request", func(t *testing.T) {
		wantText := JSON
		wantStreamText := "stream!"
		wantRequest := &ModelRequest{
			Messages: []*Message{
				{
					Role: RoleSystem,
					Content: []*Part{
						NewTextPart("You are a helpful assistant."),
						{
							ContentType: "plain/text",
							Text:        "ignored (conformance message)",
							Metadata:    map[string]any{"purpose": string("output")},
						},
					},
				},
				NewUserTextMessage("How many bananas are there?"),
				NewModelTextMessage("There are at least 10 bananas."),
				{
					Role: RoleUser,
					Content: []*Part{
						NewTextPart("Where can they be found?"),
						{
							Text: "\n\nUse the following information " +
								"to complete your task:\n\n- [0]: Bananas are plentiful in the tropics.\n\n",
							Metadata: map[string]any{"purpose": "context"},
						},
					},
				},
			},
			Config: &GenerationCommonConfig{Temperature: 1},
			Docs:   []*Document{DocumentFromText("Bananas are plentiful in the tropics.", nil)},
			Output: &ModelOutputConfig{
				Format:      OutputFormatJSON,
				ContentType: "application/json",
			},
			Tools: []*ToolDefinition{
				{
					Description: "use when need to calculate a gablorken",
					InputSchema: map[string]any{
						"additionalProperties": bool(false),
						"properties": map[string]any{
							"Over":  map[string]any{"type": string("number")},
							"Value": map[string]any{"type": string("number")},
						},
						"required": []any{string("Value"), string("Over")},
						"type":     string("object"),
					},
					Name:         "gablorken",
					OutputSchema: map[string]any{"type": string("number")},
					Metadata: map[string]any{
						"multipart": false,
					},
				},
			},
			ToolChoice: ToolChoiceAuto,
		}

		streamText := ""
		res, err := Generate(context.Background(), r,
			WithModel(bananaModel),
			WithSystem("You are a helpful assistant."),
			WithMessages(
				NewUserTextMessage("How many bananas are there?"),
				NewModelTextMessage("There are at least 10 bananas."),
			),
			WithPrompt("Where can they be found?"),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
			}),
			WithDocs(DocumentFromText("Bananas are plentiful in the tropics.", nil)),
			WithOutputType(struct {
				Subject  string `json:"subject"`
				Location string `json:"location"`
			}{}),
			WithTools(gablorkenTool),
			WithToolChoice(ToolChoiceAuto),
			WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		gotText := res.Text()
		if diff := cmp.Diff(gotText, wantText); diff != "" {
			t.Errorf("Text() diff (+got -want):\n%s", diff)
		}
		if diff := cmp.Diff(streamText, wantStreamText); diff != "" {
			t.Errorf("Text() diff (+got -want):\n%s", diff)
		}
		if diff := cmp.Diff(wantRequest, res.Request, test_utils.IgnoreNoisyParts([]string{
			"{*ai.ModelRequest}.Messages[0].Content[1].Text", "{*ai.ModelRequest}.Messages[0].Content[1].Metadata",
		})); diff != "" {
			t.Errorf("Request diff (+got -want):\n%s", diff)
		}
	})

	t.Run("handles tool interrupts", func(t *testing.T) {
		interruptTool := defineTool(r, "interruptor", "always interrupts",
			func(ctx *ToolContext, input any) (any, error) {
				return nil, ctx.Interrupt(&InterruptOptions{
					Metadata: map[string]any{
						"reason": "test interrupt",
					},
				})
			},
		)

		info := &ModelOptions{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		interruptModel := defineModel(r, "test/interrupt", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewToolRequestPart(&ToolRequest{
								Name:  "interruptor",
								Input: nil,
							}),
						},
					},
				}, nil
			})

		res, err := Generate(context.Background(), r,
			WithModel(interruptModel),
			WithPrompt("trigger interrupt"),
			WithTools(interruptTool),
		)
		if err != nil {
			t.Fatal(err)
		}
		if res.FinishReason != "interrupted" {
			t.Errorf("expected finish reason 'interrupted', got %q", res.FinishReason)
		}
		if res.FinishMessage != "One or more tool calls resulted in interrupts." {
			t.Errorf("unexpected finish message: %q", res.FinishMessage)
		}

		if len(res.Message.Content) != 1 {
			t.Fatalf("expected 1 content part, got %d", len(res.Message.Content))
		}

		metadata := res.Message.Content[0].Metadata
		if metadata == nil {
			t.Fatal("expected metadata in content part")
		}

		interrupt, ok := metadata["interrupt"].(map[string]any)
		if !ok {
			t.Fatal("expected interrupt metadata")
		}

		reason, ok := interrupt["reason"].(string)
		if !ok || reason != "test interrupt" {
			t.Errorf("expected interrupt reason 'test interrupt', got %v", reason)
		}
	})

	t.Run("handles multiple parallel tool calls", func(t *testing.T) {
		roundCount := 0
		info := &ModelOptions{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		parallelModel := defineModel(r, "test/parallel", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				roundCount++
				if roundCount == 1 {
					return &ModelResponse{
						Request: gr,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 2, "Over": 3},
								}),
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 3, "Over": 2},
								}),
							},
						},
					}, nil
				}
				var sum float64
				for _, msg := range gr.Messages {
					if msg.Role == RoleTool {
						for _, part := range msg.Content {
							if part.ToolResponse != nil {
								sum += part.ToolResponse.Output.(float64)
							}
						}
					}
				}
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewTextPart(fmt.Sprintf("Final result: %d", int(sum))),
						},
					},
				}, nil
			})

		res, err := Generate(context.Background(), r,
			WithModel(parallelModel),
			WithPrompt("trigger parallel tools"),
			WithTools(gablorkenTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		finalPart := res.Message.Content[0]
		if finalPart.Text != "Final result: 17" {
			t.Errorf("expected final result text to be 'Final result: 17', got %q", finalPart.Text)
		}
	})

	t.Run("handles multiple rounds of tool calls", func(t *testing.T) {
		roundCount := 0
		info := &ModelOptions{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		multiRoundModel := defineModel(r, "test/multiround", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				roundCount++
				if roundCount == 1 {
					return &ModelResponse{
						Request: gr,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 2, "Over": 3},
								}),
							},
						},
					}, nil
				}
				if roundCount == 2 {
					return &ModelResponse{
						Request: gr,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 3, "Over": 2},
								}),
							},
						},
					}, nil
				}
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewTextPart("Final result"),
						},
					},
				}, nil
			})

		res, err := Generate(context.Background(), r,
			WithModel(multiRoundModel),
			WithPrompt("trigger multiple rounds"),
			WithTools(gablorkenTool),
			WithMaxTurns(2),
		)
		if err != nil {
			t.Fatal(err)
		}

		if roundCount != 3 {
			t.Errorf("expected 3 rounds, got %d", roundCount)
		}

		if res.Text() != "Final result" {
			t.Errorf("expected final message 'Final result', got %q", res.Text())
		}
	})

	t.Run("exceeds maximum turns", func(t *testing.T) {
		info := &ModelOptions{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		infiniteModel := defineModel(r, "test/infinite", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewToolRequestPart(&ToolRequest{
								Name:  "gablorken",
								Input: map[string]any{"Value": 2, "Over": 2},
							}),
						},
					},
				}, nil
			})

		_, err := Generate(context.Background(), r,
			WithModel(infiniteModel),
			WithPrompt("trigger infinite loop"),
			WithTools(gablorkenTool),
			WithMaxTurns(2),
		)

		if err == nil {
			t.Fatal("expected error for exceeding maximum turns")
		}
		if !strings.Contains(err.Error(), "exceeded maximum tool call iterations (2)") {
			t.Errorf("unexpected error message: %v", err)
		}
	})

	t.Run("applies middleware", func(t *testing.T) {
		middlewareCalled := false
		testMiddleware := func(next ModelFunc) ModelFunc {
			return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				middlewareCalled = true
				req.Messages = append(req.Messages, NewUserTextMessage("middleware was here"))
				return next(ctx, req, cb)
			}
		}

		res, err := Generate(context.Background(), r,
			WithModel(echoModel),
			WithPrompt("test middleware"),
			WithMiddleware(testMiddleware),
		)
		if err != nil {
			t.Fatal(err)
		}

		if !middlewareCalled {
			t.Error("middleware was not called")
		}

		expectedText := "middleware was here"
		if res.Text() != expectedText {
			t.Errorf("got text %q, want %q", res.Text(), expectedText)
		}
	})

	t.Run("registers dynamic tools", func(t *testing.T) {
		// A root registry, not the enclosing child: Generate only quarantines
		// dynamic tools in a child of its own when the caller hands it a root,
		// which is the isolation this subtest asserts.
		r := newTestRegistry(t)

		// Create a tool that is NOT registered in the global registry
		dynamicTool := NewTool("dynamicTestTool", "a tool that is dynamically registered",
			func(ctx *ToolContext, input struct {
				Message string
			},
			) (string, error) {
				return "Dynamic: " + input.Message, nil
			},
		)

		// Verify the tool is not in the global registry
		if LookupTool(r, "dynamicTestTool") != nil {
			t.Fatal("dynamicTestTool should not be registered in global registry")
		}

		// Create a model that will call the dynamic tool then provide a final response
		roundCount := 0
		info := &ModelOptions{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		toolCallModel := defineModel(r, "test/toolcall", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				roundCount++
				if roundCount == 1 {
					// First response: call the dynamic tool
					return &ModelResponse{
						Request: gr,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{
								NewToolRequestPart(&ToolRequest{
									Name:  "dynamicTestTool",
									Input: map[string]any{"Message": "Hello from dynamic tool"},
								}),
							},
						},
					}, nil
				}
				// Second response: provide final answer based on tool response
				var toolResult string
				for _, msg := range gr.Messages {
					if msg.Role == RoleTool {
						for _, part := range msg.Content {
							if part.ToolResponse != nil {
								toolResult = part.ToolResponse.Output.(string)
							}
						}
					}
				}
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewTextPart(toolResult),
						},
					},
				}, nil
			})

		// Use Generate with the dynamic tool - this should trigger the dynamic registration
		res, err := Generate(context.Background(), r,
			WithModel(toolCallModel),
			WithPrompt("call the dynamic tool"),
			WithTools(dynamicTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		// The tool should have been called and returned a response
		expectedText := "Dynamic: Hello from dynamic tool"
		if res.Text() != expectedText {
			t.Errorf("expected text %q, got %q", expectedText, res.Text())
		}

		// Verify two rounds were executed: tool call + final response
		if roundCount != 2 {
			t.Errorf("expected 2 rounds, got %d", roundCount)
		}

		// Verify the tool is still not in the global registry (it was registered in a child)
		if LookupTool(r, "dynamicTestTool") != nil {
			t.Error("dynamicTestTool should not be registered in global registry after generation")
		}
	})

	t.Run("handles duplicate dynamic tools", func(t *testing.T) {
		// Create two tools with the same name
		dynamicTool1 := NewTool("duplicateTool", "first tool",
			func(ctx *ToolContext, input any) (string, error) {
				return "tool1", nil
			},
		)
		dynamicTool2 := NewTool("duplicateTool", "second tool",
			func(ctx *ToolContext, input any) (string, error) {
				return "tool2", nil
			},
		)

		// Using both tools should result in an error
		_, err := Generate(context.Background(), r,
			WithModel(echoModel),
			WithPrompt("test duplicate tools"),
			WithTools(dynamicTool1, dynamicTool2),
		)

		if err == nil {
			t.Fatal("expected error for duplicate tool names")
		}
		if !strings.Contains(err.Error(), "duplicate tool \"duplicateTool\"") {
			t.Errorf("unexpected error message: %v", err)
		}
	})
}

func TestGenerateWithOutputSchemaName(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)

	// Define a model that supports constrained output
	model := defineModel(r, "test/constrained", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		// Mock response
		return &ModelResponse{
			Message: NewModelTextMessage(`{"foo": "bar"}`),
			Request: req,
		}, nil
	})

	r.RegisterSchema("FooSchema", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"foo": map[string]any{"type": "string"},
		},
	})

	t.Run("Valid Schema", func(t *testing.T) {
		resp, err := Generate(context.Background(), r,
			WithModel(model),
			WithPrompt("test"),
			WithOutputSchemaName("FooSchema"),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		if resp.Request.Output.Schema == nil {
			t.Fatal("Expected output schema to be set")
		}

		// Verify schema is resolved
		if props, ok := resp.Request.Output.Schema["properties"].(map[string]any); ok {
			if _, ok := props["foo"]; !ok {
				t.Error("Expected schema to have 'foo' property")
			}
		} else {
			t.Fatalf("Expected properties map in schema, got: %+v", resp.Request.Output.Schema)
		}
	})

	t.Run("Missing Schema", func(t *testing.T) {
		_, err := Generate(context.Background(), r,
			WithModel(model),
			WithPrompt("test"),
			WithOutputSchemaName("MissingSchema"),
		)
		if err == nil {
			t.Fatal("Expected error when executing generate with missing schema")
		}
		if !strings.Contains(err.Error(), "schema \"MissingSchema\" not found") {
			t.Errorf("Expected error 'schema \"MissingSchema\" not found', got: %v", err)
		}
	})
}

func TestModelVersion(t *testing.T) {
	t.Run("valid version", func(t *testing.T) {
		_, err := Generate(context.Background(), r,
			WithModel(echoModel),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
				Version:     "echo-001",
			}),
			WithPrompt("tell a joke about batman"))
		if err != nil {
			t.Errorf("model version should be valid")
		}
	})
	t.Run("invalid version", func(t *testing.T) {
		_, err := Generate(context.Background(), r,
			WithModel(echoModel),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
				Version:     "echo-im-not-a-version",
			}),
			WithPrompt("tell a joke about batman"))
		if err == nil {
			t.Errorf("model version should be invalid: %v", err)
		}
	})
}

func TestLookupModel(t *testing.T) {
	t.Run("should return model", func(t *testing.T) {
		if LookupModel(r, "test/"+modelName) == nil {
			t.Errorf("LookupModel did not return model")
		}
	})
	t.Run("should return nil", func(t *testing.T) {
		if LookupModel(r, "foo/bar") != nil {
			t.Errorf("LookupModel did not return nil")
		}
	})
}

func JSONMarkdown(text string) string {
	return "```json\n" + text + "\n```"
}

func errorContains(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Error("got nil, want error")
	} else if !strings.Contains(err.Error(), want) {
		t.Errorf("got error message %q, want it to contain %q", err, want)
	}
}

func validTestMessage(m *Message, output *ModelOutputConfig) (*Message, error) {
	resolvedFormat, err := resolveFormat(r, output.Schema, output.Format)
	if err != nil {
		return nil, err
	}

	handler, err := resolvedFormat.Handler(output.Schema)
	if err != nil {
		return nil, err
	}

	return handler.ParseMessage(m)
}

type conditionalToolInput struct {
	Value     string
	Interrupt bool
}

type resumableToolInput struct {
	Action string
	Data   string
}

func TestToolInterruptsAndResume(t *testing.T) {
	r := childRegistry(t)
	conditionalTool := defineTool(r, "conditional", "tool that may interrupt based on input",
		func(ctx *ToolContext, input conditionalToolInput) (string, error) {
			if input.Interrupt {
				return "", ctx.Interrupt(&InterruptOptions{
					Metadata: map[string]any{
						"reason":      "user_intervention_required",
						"value":       input.Value,
						"interrupted": true,
					},
				})
			}
			return fmt.Sprintf("processed: %s", input.Value), nil
		},
	)

	resumableTool := defineTool(r, "resumable", "tool that can be resumed",
		func(ctx *ToolContext, input resumableToolInput) (string, error) {
			if ctx.Resumed != nil {
				resumedData, ok := ctx.Resumed["data"].(string)
				if ok {
					return fmt.Sprintf("resumed with: %s, original: %s", resumedData, input.Data), nil
				}
				return fmt.Sprintf("resumed: %s", input.Data), nil
			}
			return fmt.Sprintf("first run: %s", input.Data), nil
		},
	)

	info := &ModelOptions{
		Supports: &ModelSupports{
			Multiturn: true,
			Tools:     true,
		},
	}

	toolModel := defineModel(r, "test/toolmodel", info,
		func(ctx context.Context, mr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Request: mr,
				Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("I need to use some tools."),
						NewToolRequestPart(&ToolRequest{
							Name: "conditional",
							Ref:  "tool1",
							Input: map[string]any{
								"Value":     "test_data",
								"Interrupt": true,
							},
						}),
						NewToolRequestPart(&ToolRequest{
							Name: "resumable",
							Ref:  "tool2",
							Input: map[string]any{
								"Action": "process",
								"Data":   "initial_data",
							},
						}),
					},
				},
			}, nil
		})

	t.Run("basic interrupt flow", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(toolModel),
			WithPrompt("use tools"),
			WithTools(conditionalTool, resumableTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		if res.FinishReason != "interrupted" {
			t.Errorf("expected finish reason 'interrupted', got %q", res.FinishReason)
		}

		if len(res.Message.Content) != 3 {
			t.Fatalf("expected 3 content parts, got %d", len(res.Message.Content))
		}

		interruptedPart := res.Message.Content[1]
		if !interruptedPart.IsToolRequest() {
			t.Fatal("expected second part to be a tool request")
		}

		interruptMeta, ok := interruptedPart.Metadata["interrupt"].(map[string]any)
		if !ok {
			t.Fatal("expected interrupt metadata in tool request")
		}

		if reason, ok := interruptMeta["reason"].(string); !ok || reason != "user_intervention_required" {
			t.Errorf("expected interrupt reason 'user_intervention_required', got %v", reason)
		}
	})

	t.Run("tool.Respond functionality", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(toolModel),
			WithPrompt("use tools"),
			WithTools(conditionalTool, resumableTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		interruptedPart := res.Message.Content[1]

		responsePart := conditionalTool.Respond(interruptedPart, "manual_response_data", &RespondOptions{
			Metadata: map[string]any{
				"manual": true,
				"source": "user",
			},
		})

		if !responsePart.IsToolResponse() {
			t.Fatal("expected response part to be a tool response")
		}

		if responsePart.ToolResponse.Name != "conditional" {
			t.Errorf("expected tool response name 'conditional', got %q", responsePart.ToolResponse.Name)
		}

		if responsePart.ToolResponse.Ref != "tool1" {
			t.Errorf("expected tool response ref 'tool1', got %q", responsePart.ToolResponse.Ref)
		}

		if responsePart.ToolResponse.Output != "manual_response_data" {
			t.Errorf("expected output 'manual_response_data', got %v", responsePart.ToolResponse.Output)
		}

		interruptResponseMeta, ok := responsePart.Metadata["interruptResponse"].(map[string]any)
		if !ok {
			t.Fatal("expected interruptResponse metadata")
		}

		if manual, ok := interruptResponseMeta["manual"].(bool); !ok || !manual {
			t.Errorf("expected manual metadata to be true")
		}
	})

	t.Run("tool.Restart functionality", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(toolModel),
			WithPrompt("use tools"),
			WithTools(conditionalTool, resumableTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		interruptedPart := res.Message.Content[1]

		newInput := conditionalToolInput{
			Value:     "new_test_data",
			Interrupt: false,
		}
		restartPart := conditionalTool.Restart(interruptedPart, &RestartOptions{
			ReplaceInput: newInput,
			ResumedMetadata: map[string]any{
				"data":   "resumed_data",
				"source": "restart",
			},
		})

		if !restartPart.IsToolRequest() {
			t.Fatal("expected restart part to be a tool request")
		}

		if restartPart.ToolRequest.Name != "conditional" {
			t.Errorf("expected tool request name 'conditional', got %q", restartPart.ToolRequest.Name)
		}

		replacedInput, ok := restartPart.ToolRequest.Input.(conditionalToolInput)
		if !ok {
			t.Fatalf("expected input to be conditionalInput, got %T", restartPart.ToolRequest.Input)
		}

		if replacedInput.Value != "new_test_data" {
			t.Errorf("expected new input value 'new_test_data', got %v", replacedInput.Value)
		}

		if replacedInput.Interrupt != false {
			t.Errorf("expected interrupt to be false, got %v", replacedInput.Interrupt)
		}

		if _, hasInterrupt := restartPart.Metadata["interrupt"]; hasInterrupt {
			t.Error("expected interrupt metadata to be removed")
		}

		resumedMeta, ok := restartPart.Metadata["resumed"].(map[string]any)
		if !ok {
			t.Fatal("expected resumed metadata")
		}

		if resumedMeta["data"] != "resumed_data" {
			t.Errorf("expected resumed data 'resumed_data', got %v", resumedMeta["data"])
		}
	})

	t.Run("resume with respond directive", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(toolModel),
			WithPrompt("use tools"),
			WithTools(conditionalTool, resumableTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		interruptedPart := res.Message.Content[1]
		responsePart := conditionalTool.Respond(interruptedPart, "user_provided_response", nil)

		history := res.History()
		resumeRes, err := Generate(context.Background(), r,
			WithModel(NewModelRef("test/echo", nil)),
			WithMessages(history...),
			WithTools(conditionalTool, resumableTool),
			WithToolResponses(responsePart),
		)
		if err != nil {
			t.Fatal(err)
		}

		if resumeRes.FinishReason == "interrupted" {
			t.Error("expected generation to not be interrupted after responding")
		}
	})

	t.Run("resume with restart directive", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(toolModel),
			WithPrompt("use tools"),
			WithTools(conditionalTool, resumableTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		interruptedPart := res.Message.Content[1]

		newInput := conditionalToolInput{
			Value:     "restarted_data",
			Interrupt: false,
		}
		restartPart := conditionalTool.Restart(interruptedPart, &RestartOptions{
			ReplaceInput: newInput,
			ResumedMetadata: map[string]any{
				"data": "restart_context",
			},
		})

		history := res.History()
		resumeRes, err := Generate(context.Background(), r,
			WithModel(NewModelRef("test/echo", nil)),
			WithMessages(history...),
			WithTools(conditionalTool, resumableTool),
			WithToolRestarts(restartPart),
		)
		if err != nil {
			t.Fatal(err)
		}

		if resumeRes.FinishReason == "interrupted" {
			t.Error("expected generation to not be interrupted after restarting")
		}
	})
}

func TestResourceProcessing(t *testing.T) {
	r := registry.New()

	// Create test resources using DefineResource
	defineResource(r, "test-file", &ResourceOptions{
		URI:         "file:///test.txt",
		Description: "Test file resource",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{Content: []*Part{NewTextPart("FILE CONTENT")}}, nil
	})

	defineResource(r, "test-api", &ResourceOptions{
		URI:         "api://data/123",
		Description: "Test API resource",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{Content: []*Part{NewTextPart("API DATA")}}, nil
	})

	// Test message with resources
	messages := []*Message{
		NewUserMessage(
			NewTextPart("Read this:"),
			NewResourcePart("file:///test.txt"),
			NewTextPart("And this:"),
			NewResourcePart("api://data/123"),
			NewTextPart("Done."),
		),
	}

	// Process resources
	processed, err := processResources(context.Background(), r, messages)
	if err != nil {
		t.Fatalf("resource processing failed: %v", err)
	}

	// Verify content
	content := processed[0].Content
	expected := []string{"Read this:", "FILE CONTENT", "And this:", "API DATA", "Done."}

	if len(content) != len(expected) {
		t.Fatalf("expected %d parts, got %d", len(expected), len(content))
	}

	for i, want := range expected {
		if content[i].Text != want {
			t.Fatalf("part %d: got %q, want %q", i, content[i].Text, want)
		}
	}
}

func TestResourceProcessingError(t *testing.T) {
	r := registry.New()

	// No resources registered
	messages := []*Message{
		NewUserMessage(NewResourcePart("missing://resource")),
	}

	_, err := processResources(context.Background(), r, messages)
	if err == nil {
		t.Fatal("expected error when no resources available")
	}

	if !strings.Contains(err.Error(), "no resource found for URI") {
		t.Fatalf("wrong error: %v", err)
	}
}

func TestModelResponseOutput(t *testing.T) {
	t.Run("single JSON part (json format)", func(t *testing.T) {
		mr := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart(`{"name":"Alice","age":30}`),
				},
			},
		}

		var result struct {
			Name string `json:"name"`
			Age  int    `json:"age"`
		}
		err := mr.Output(&result)
		if err != nil {
			t.Fatalf("Output() error = %v", err)
		}
		if result.Name != "Alice" || result.Age != 30 {
			t.Errorf("Output() = %+v, want {Alice 30}", result)
		}
	})

	t.Run("JSON array without format handler", func(t *testing.T) {
		mr := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(`[{"id":1},{"id":2},{"id":3}]`),
				},
			},
		}

		var result []struct {
			ID int `json:"id"`
		}
		err := mr.Output(&result)
		if err != nil {
			t.Fatalf("Output() error = %v", err)
		}
		if len(result) != 3 {
			t.Fatalf("Output() got %d items, want 3", len(result))
		}
		for i, item := range result {
			if item.ID != i+1 {
				t.Errorf("Output()[%d].ID = %d, want %d", i, item.ID, i+1)
			}
		}
	})

	t.Run("plain JSON text without format handler", func(t *testing.T) {
		mr := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(`{"value":42}`),
				},
			},
		}

		var result struct {
			Value int `json:"value"`
		}
		err := mr.Output(&result)
		if err != nil {
			t.Fatalf("Output() error = %v", err)
		}
		if result.Value != 42 {
			t.Errorf("Output().Value = %d, want 42", result.Value)
		}
	})

	t.Run("no content error", func(t *testing.T) {
		mr := &ModelResponse{
			Message: &Message{
				Role:    RoleModel,
				Content: []*Part{},
			},
		}

		var result any
		err := mr.Output(&result)
		if err == nil {
			t.Error("Output() expected error for empty content")
		}
	})

	t.Run("nil message error", func(t *testing.T) {
		mr := &ModelResponse{
			Message: nil,
		}

		var result any
		err := mr.Output(&result)
		if err == nil {
			t.Error("Output() expected error for nil message")
		}
	})

	t.Run("no JSON found error", func(t *testing.T) {
		mr := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Just plain text with no JSON"),
				},
			},
		}

		var result any
		err := mr.Output(&result)
		if err == nil {
			t.Error("Output() expected error when no JSON found")
		}
	})

	t.Run("format-aware: jsonl format with handler", func(t *testing.T) {
		schema := map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"line": map[string]any{"type": "integer"},
				},
			},
		}
		formatter := jsonlFormatter{}
		handler, err := formatter.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}
		streamingHandler := handler.(StreamingFormatHandler)

		mr := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("{\"line\":1}\n{\"line\":2}"),
				},
			},
			formatHandler: streamingHandler,
		}

		var result []struct {
			Line int `json:"line"`
		}
		err = mr.Output(&result)
		if err != nil {
			t.Fatalf("Output() error = %v", err)
		}
		if len(result) != 2 || result[0].Line != 1 || result[1].Line != 2 {
			t.Errorf("Output() = %+v, want [{1} {2}]", result)
		}
	})

	t.Run("format-aware: array format with handler", func(t *testing.T) {
		schema := map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"item": map[string]any{"type": "string"},
				},
			},
		}
		formatter := arrayFormatter{}
		handler, err := formatter.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}
		streamingHandler := handler.(StreamingFormatHandler)

		mr := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(`[{"item":"a"},{"item":"b"}]`),
				},
			},
			formatHandler: streamingHandler,
		}

		var result []struct {
			Item string `json:"item"`
		}
		err = mr.Output(&result)
		if err != nil {
			t.Fatalf("Output() error = %v", err)
		}
		if len(result) != 2 || result[0].Item != "a" || result[1].Item != "b" {
			t.Errorf("Output() = %+v, want [{a} {b}]", result)
		}
	})

	t.Run("format-aware: json format with handler", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"key": map[string]any{"type": "string"},
			},
		}
		formatter := jsonFormatter{}
		handler, err := formatter.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}
		streamingHandler := handler.(StreamingFormatHandler)

		mr := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(`{"key":"value"}`),
				},
			},
			formatHandler: streamingHandler,
		}

		var result struct {
			Key string `json:"key"`
		}
		err = mr.Output(&result)
		if err != nil {
			t.Fatalf("Output() error = %v", err)
		}
		if result.Key != "value" {
			t.Errorf("Output().Key = %q, want %q", result.Key, "value")
		}
	})
}

func TestMultipartTools(t *testing.T) {
	t.Run("define multipart tool registers as tool.v2 only", func(t *testing.T) {
		r := registry.New()

		defineMultipartTool(r, "multipartTest", "a multipart tool",
			func(ctx *ToolContext, input struct{ Query string }) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output:  "main output",
					Content: []*Part{NewTextPart("content part 1")},
				}, nil
			},
		)

		// Should be found via LookupTool
		tool := LookupTool(r, "multipartTest")
		if tool == nil {
			t.Fatal("expected multipart tool to be found via LookupTool")
		}

		// Should be able to produce response with content
		resp, err := tool.RunRawMultipart(context.Background(), struct{ Query string }{Query: "Q"})
		if err != nil {
			t.Fatalf("failed running multipart tool: %v", err)
		}
		if len(resp.Content) == 0 {
			t.Error("expected tool response to have content")
		}
	})

	t.Run("regular tool registers as both tool and tool.v2", func(t *testing.T) {
		r := registry.New()

		defineTool(r, "regularTestTool", "a regular tool",
			func(ctx *ToolContext, input struct{ Value int }) (int, error) {
				return input.Value * 2, nil
			},
		)

		// Should be found via LookupTool
		tool := LookupTool(r, "regularTestTool")
		if tool == nil {
			t.Fatal("expected regular tool to be found via LookupTool")
		}

		// Should produce response without content by default
		resp, err := tool.RunRawMultipart(context.Background(), struct{ Value int }{Value: 21})
		if err != nil {
			t.Fatalf("failed running regular tool: %v", err)
		}
		if len(resp.Content) > 0 {
			t.Error("expected regular tool response to have no content")
		}
	})

	t.Run("multipart tool returns metadata and content in response", func(t *testing.T) {
		r := registry.New()
		ConfigureFormats(r)
		DefineGenerateAction(context.Background(), r)

		multipartTool := defineMultipartTool(r, "imageGenerator", "generates images",
			func(ctx *ToolContext, input struct{ Prompt string }) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output:   map[string]any{"description": "generated image"},
					Metadata: map[string]any{"size": 1},
					Content: []*Part{
						NewMediaPart("image/png", "data:image/png;base64,iVBORw0..."),
					},
				}, nil
			},
		)

		// Create a model that requests the tool
		multipartToolModel := defineModel(r, "test/multipartToolModel", &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
			// Check if we already have a tool response
			for _, msg := range gr.Messages {
				if msg.Role == RoleTool {
					for _, part := range msg.Content {
						if part.IsToolResponse() {
							// Verify the metadata and content are present
							if len(part.Metadata) == 0 {
								return nil, fmt.Errorf("expected tool response to have metadata")
							}
							if len(part.ToolResponse.Content) == 0 {
								return nil, fmt.Errorf("expected tool response to have content")
							}
							return &ModelResponse{
								Request: gr,
								Message: NewModelTextMessage("Image generated successfully"),
							}, nil
						}
					}
				}
			}

			// First call: request the tool
			return &ModelResponse{
				Request: gr,
				Message: &Message{
					Role: RoleModel,
					Content: []*Part{NewToolRequestPart(&ToolRequest{
						Name:  "imageGenerator",
						Input: map[string]any{"Prompt": "a cat"},
						Ref:   "img1",
					})},
				},
			}, nil
		})

		resp, err := Generate(context.Background(), r,
			WithModel(multipartToolModel),
			WithPrompt("Generate an image of a cat"),
			WithTools(multipartTool),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		if resp.Text() != "Image generated successfully" {
			t.Errorf("expected 'Image generated successfully', got %q", resp.Text())
		}
	})

	t.Run("RunRawMultipart returns MultipartToolResponse for regular tool", func(t *testing.T) {
		r := registry.New()

		tool := defineTool(r, "multipartWrapperTest", "test multipart wrapper",
			func(ctx *ToolContext, input struct{ Value int }) (int, error) {
				return input.Value * 3, nil
			},
		)

		resp, err := tool.RunRawMultipart(context.Background(), map[string]any{"Value": 5})
		if err != nil {
			t.Fatalf("RunRawMultipart failed: %v", err)
		}

		// Output should be wrapped in MultipartToolResponse
		output, ok := resp.Output.(float64) // JSON unmarshals numbers as float64
		if !ok {
			t.Fatalf("expected output to be float64, got %T", resp.Output)
		}
		if output != 15 {
			t.Errorf("expected output 15, got %v", output)
		}

		// Content should be nil for regular tools
		if resp.Content != nil {
			t.Errorf("expected nil content for regular tool, got %v", resp.Content)
		}
	})

	t.Run("RunRawMultipart returns full response for multipart tool", func(t *testing.T) {
		r := registry.New()

		tool := defineMultipartTool(r, "multipartFullTest", "test multipart",
			func(ctx *ToolContext, input struct{ Query string }) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output:  "result",
					Content: []*Part{NewTextPart("additional content")},
				}, nil
			},
		)

		resp, err := tool.RunRawMultipart(context.Background(), map[string]any{"Query": "test"})
		if err != nil {
			t.Fatalf("RunRawMultipart failed: %v", err)
		}

		if resp.Output != "result" {
			t.Errorf("expected output 'result', got %v", resp.Output)
		}

		if len(resp.Content) != 1 {
			t.Fatalf("expected 1 content part, got %d", len(resp.Content))
		}

		if resp.Content[0].Text != "additional content" {
			t.Errorf("expected content 'additional content', got %q", resp.Content[0].Text)
		}
	})
}

// streamingTestData holds test output structures
type streamingTestData struct {
	Name  string `json:"name"`
	Value int    `json:"value"`
}

func TestGenerateStream(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(context.Background(), r)

	t.Run("yields chunks then final response", func(t *testing.T) {
		chunkTexts := []string{"Hello", " ", "World"}
		chunkIndex := 0

		streamModel := defineModel(r, "test/streamModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				for _, text := range chunkTexts {
					cb(ctx, &ModelResponseChunk{
						Content: []*Part{NewTextPart(text)},
					})
				}
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("Hello World"),
			}, nil
		})

		var receivedChunks []*ModelResponseChunk
		var finalResponse *ModelResponse

		for val, err := range GenerateStream(context.Background(), r,
			WithModel(streamModel),
			WithPrompt("test streaming"),
		) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalResponse = val.Response
			} else {
				receivedChunks = append(receivedChunks, val.Chunk)
				chunkIndex++
			}
		}

		if len(receivedChunks) != len(chunkTexts) {
			t.Errorf("expected %d chunks, got %d", len(chunkTexts), len(receivedChunks))
		}

		for i, chunk := range receivedChunks {
			if chunk.Text() != chunkTexts[i] {
				t.Errorf("chunk %d: expected %q, got %q", i, chunkTexts[i], chunk.Text())
			}
		}

		if finalResponse == nil {
			t.Fatal("expected final response")
		}
		if finalResponse.Text() != "Hello World" {
			t.Errorf("expected final text %q, got %q", "Hello World", finalResponse.Text())
		}
	})

	t.Run("handles no streaming callback gracefully", func(t *testing.T) {
		noStreamModel := defineModel(r, "test/noStreamModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("response without streaming"),
			}, nil
		})

		var finalResponse *ModelResponse
		chunkCount := 0

		for val, err := range GenerateStream(context.Background(), r,
			WithModel(noStreamModel),
			WithPrompt("test no stream"),
		) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalResponse = val.Response
			} else {
				chunkCount++
			}
		}

		if chunkCount != 0 {
			t.Errorf("expected 0 chunks when model doesn't stream, got %d", chunkCount)
		}
		if finalResponse == nil {
			t.Fatal("expected final response")
		}
		if finalResponse.Text() != "response without streaming" {
			t.Errorf("expected text %q, got %q", "response without streaming", finalResponse.Text())
		}
	})

	t.Run("propagates generation errors", func(t *testing.T) {
		expectedErr := errors.New("generation failed")

		errorModel := defineModel(r, "test/errorModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return nil, expectedErr
		})

		var receivedErr error
		for _, err := range GenerateStream(context.Background(), r,
			WithModel(errorModel),
			WithPrompt("test error"),
		) {
			if err != nil {
				receivedErr = err
				break
			}
		}

		if receivedErr == nil {
			t.Fatal("expected error to be propagated")
		}
		if !errors.Is(receivedErr, expectedErr) {
			t.Errorf("expected error %v, got %v", expectedErr, receivedErr)
		}
	})

	t.Run("context cancellation stops iteration", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		streamModel := defineModel(r, "test/cancelModel", &ModelOptions{
			Supports: &ModelSupports{Multiturn: true},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				for i := 0; i < 100; i++ {
					err := cb(ctx, &ModelResponseChunk{
						Content: []*Part{NewTextPart("chunk")},
					})
					if err != nil {
						return nil, err
					}
				}
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("done"),
			}, nil
		})

		chunksReceived := 0
		var receivedErr error
		for val, err := range GenerateStream(ctx, r,
			WithModel(streamModel),
			WithPrompt("test cancel"),
		) {
			if err != nil {
				receivedErr = err
				break
			}
			if !val.Done {
				chunksReceived++
				if chunksReceived == 2 {
					cancel()
				}
			}
		}

		if chunksReceived < 2 {
			t.Errorf("expected at least 2 chunks before cancellation, got %d", chunksReceived)
		}
		if receivedErr == nil {
			t.Error("expected error from cancelled context")
		}
	})

	t.Run("should not yield after stop", func(t *testing.T) {
		streamModel := defineModel(r, "test/breakStreamModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn: true,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			for range 3 {
				if err := cb(ctx, &ModelResponseChunk{Content: []*Part{NewTextPart("chunk")}}); err != nil {
					return nil, err
				}
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("done"),
			}, nil
		})

		for _, err := range GenerateStream(t.Context(), r, WithModel(streamModel)) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			break
		}
	})
}

func TestGenerateDataStream(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(context.Background(), r)

	t.Run("yields typed chunks and final output", func(t *testing.T) {
		streamModel := defineModel(r, "test/typedStreamModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewJSONPart(`{"name":"partial","value":1}`)},
				})
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewJSONPart(`{"name":"complete","value":42}`)},
				})
			}
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"name":"final","value":42}`)},
				},
			}, nil
		})

		var chunks []streamingTestData
		var finalOutput streamingTestData
		var finalResponse *ModelResponse

		for val, err := range GenerateDataStream[streamingTestData](context.Background(), r,
			WithModel(streamModel),
			WithPrompt("test typed streaming"),
		) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalOutput = val.Output
				finalResponse = val.Response
			} else {
				chunks = append(chunks, val.Chunk)
			}
		}

		if len(chunks) < 1 {
			t.Errorf("expected at least 1 chunk, got %d", len(chunks))
		}

		if finalOutput.Name != "final" || finalOutput.Value != 42 {
			t.Errorf("expected final output {final, 42}, got %+v", finalOutput)
		}
		if finalResponse == nil {
			t.Fatal("expected final response")
		}
	})

	t.Run("final output is correctly typed", func(t *testing.T) {
		streamModel := defineModel(r, "test/finalTypedModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"name":"result","value":123}`)},
				},
			}, nil
		})

		var finalOutput streamingTestData
		var gotFinal bool

		for val, err := range GenerateDataStream[streamingTestData](context.Background(), r,
			WithModel(streamModel),
			WithPrompt("test final typed"),
		) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalOutput = val.Output
				gotFinal = true
			}
		}

		if !gotFinal {
			t.Fatal("expected to receive final output")
		}
		if finalOutput.Name != "result" || finalOutput.Value != 123 {
			t.Errorf("expected final output {result, 123}, got %+v", finalOutput)
		}
	})

	t.Run("automatically sets output type", func(t *testing.T) {
		var capturedRequest *ModelRequest

		streamModel := defineModel(r, "test/autoOutputModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			capturedRequest = req
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"name":"test","value":1}`)},
				},
			}, nil
		})

		for range GenerateDataStream[streamingTestData](context.Background(), r,
			WithModel(streamModel),
			WithPrompt("test auto output type"),
		) {
		}

		if capturedRequest == nil {
			t.Fatal("expected request to be captured")
		}
		if capturedRequest.Output == nil || capturedRequest.Output.Schema == nil {
			t.Error("expected output schema to be set automatically")
		}
	})

	t.Run("handles tool interrupts", func(t *testing.T) {
		interruptTool := defineTool(r, "streamInterruptor", "always interrupts",
			func(ctx *ToolContext, input any) (any, error) {
				return nil, ctx.Interrupt(&InterruptOptions{
					Metadata: map[string]any{
						"reason": "needs confirmation",
					},
				})
			},
		)

		streamModel := defineModel(r, "test/streamInterruptModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Tools:       true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("thinking...")},
				})
			}
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{
							Name:  "streamInterruptor",
							Input: nil,
						}),
					},
				},
			}, nil
		})

		var finalResponse *ModelResponse
		var gotError error

		for val, err := range GenerateDataStream[streamingTestData](context.Background(), r,
			WithModel(streamModel),
			WithPrompt("trigger interrupt"),
			WithTools(interruptTool),
		) {
			if err != nil {
				gotError = err
				break
			}
			if val.Done {
				finalResponse = val.Response
			}
		}

		if gotError != nil {
			t.Fatalf("unexpected error: %v", gotError)
		}
		if finalResponse == nil {
			t.Fatal("expected final response")
		}
		if finalResponse.FinishReason != "interrupted" {
			t.Errorf("expected finish reason 'interrupted', got %q", finalResponse.FinishReason)
		}
		if len(finalResponse.Interrupts()) != 1 {
			t.Errorf("expected 1 interrupt, got %d", len(finalResponse.Interrupts()))
		}
	})

	t.Run("handles returnToolRequests", func(t *testing.T) {
		greetTool := defineTool(r, "streamGreeter", "greets",
			func(ctx *ToolContext, input any) (any, error) {
				return "hello", nil
			},
		)

		streamModel := defineModel(r, "test/streamReturnToolModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Tools:       true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{
							Name:  "streamGreeter",
							Input: map[string]any{"name": "world"},
						}),
					},
				},
			}, nil
		})

		var finalResponse *ModelResponse
		var gotError error

		for val, err := range GenerateDataStream[streamingTestData](context.Background(), r,
			WithModel(streamModel),
			WithPrompt("greet"),
			WithTools(greetTool),
			WithReturnToolRequests(true),
		) {
			if err != nil {
				gotError = err
				break
			}
			if val.Done {
				finalResponse = val.Response
			}
		}

		if gotError != nil {
			t.Fatalf("unexpected error: %v", gotError)
		}
		if finalResponse == nil {
			t.Fatal("expected final response")
		}
		if len(finalResponse.ToolRequests()) != 1 {
			t.Errorf("expected 1 tool request, got %d", len(finalResponse.ToolRequests()))
		}
	})

	t.Run("propagates chunk parsing errors", func(t *testing.T) {
		streamModel := defineModel(r, "test/parseErrorModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				cb(ctx, &ModelResponseChunk{
					Content: []*Part{NewTextPart("not valid json")},
				})
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("done"),
			}, nil
		})

		var receivedErr error
		for _, err := range GenerateDataStream[streamingTestData](context.Background(), r,
			WithModel(streamModel),
			WithPrompt("test parse error"),
		) {
			if err != nil {
				receivedErr = err
				break
			}
		}

		if receivedErr == nil {
			t.Error("expected parsing error to be propagated")
		}
	})

	t.Run("should not yield after stop", func(t *testing.T) {
		streamModel := defineModel(r, "test/breakDataStreamModel", &ModelOptions{
			Supports: &ModelSupports{
				Multiturn:   true,
				Constrained: ConstrainedSupportAll,
			},
		}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			for i := range 3 {
				if err := cb(ctx, &ModelResponseChunk{Content: []*Part{NewJSONPart(fmt.Sprintf(`{"name":"chunk","value":%d}`, i))}}); err != nil {
					return nil, err
				}
			}
			return &ModelResponse{
				Request: req,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewJSONPart(`{"name":"done","value":4}`)},
				},
			}, nil
		})

		for _, err := range GenerateDataStream[streamingTestData](t.Context(), r, WithModel(streamModel)) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			break
		}
	})
}

func TestGenerateText(t *testing.T) {
	r := newTestRegistry(t)

	echoModel := defineModel(r, "test/echoTextModel", nil, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage("echo: " + req.Messages[0].Content[0].Text),
		}, nil
	})

	t.Run("returns text from model", func(t *testing.T) {
		text, err := GenerateText(context.Background(), r,
			WithModel(echoModel),
			WithPrompt("hello"),
		)
		if err != nil {
			t.Fatalf("GenerateText error: %v", err)
		}
		if text != "echo: hello" {
			t.Errorf("text = %q, want %q", text, "echo: hello")
		}
	})
}

func TestGenerateData(t *testing.T) {
	r := newTestRegistry(t)

	type TestOutput struct {
		Value int `json:"value"`
	}

	jsonModel := defineModel(r, "test/jsonDataModel", &ModelOptions{
		Supports: &ModelSupports{
			Constrained: ConstrainedSupportAll,
		},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage(`{"value": 42}`),
		}, nil
	})

	t.Run("returns typed data from model", func(t *testing.T) {
		output, _, err := GenerateData[TestOutput](context.Background(), r,
			WithModel(jsonModel),
			WithPrompt("get value"),
		)
		if err != nil {
			t.Fatalf("GenerateData error: %v", err)
		}
		if output.Value != 42 {
			t.Errorf("output.Value = %d, want 42", output.Value)
		}
	})
}

// TestGenerateDataCallerSchemaOverride verifies that GenerateData injects the
// output type inferred from Out but still lets a caller-supplied
// WithOutputSchema win the schema slot, while typed extraction keeps working.
func TestGenerateDataCallerSchemaOverride(t *testing.T) {
	r := newTestRegistry(t)

	type TestOutput struct {
		Value int `json:"value"`
	}

	var capturedSchema map[string]any
	model := defineFakeModel(t, r, fakeModelConfig{
		name: "test/captureSchema",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if req.Output != nil {
				capturedSchema = req.Output.Schema
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage(`{"value": 42}`),
			}, nil
		},
	})

	// A distinctive schema the inferred one would never produce.
	customSchema := map[string]any{
		"type":  "object",
		"title": "CallerProvided",
		"properties": map[string]any{
			"value": map[string]any{"type": "integer"},
		},
	}

	output, _, err := GenerateData[TestOutput](context.Background(), r,
		WithModel(model),
		WithPrompt("get value"),
		WithOutputSchema(customSchema),
	)
	if err != nil {
		t.Fatalf("GenerateData error: %v", err)
	}

	// The caller's schema reaches the model, overriding the type-inferred one.
	if capturedSchema["title"] != "CallerProvided" {
		t.Errorf("request output schema = %v, want caller-provided schema (title CallerProvided)", capturedSchema)
	}
	// Typed extraction from Out still works.
	if output.Value != 42 {
		t.Errorf("output.Value = %d, want 42", output.Value)
	}
}

// TestGenerateStreamChainsUserCallback verifies that the stream-returning
// wrappers chain a caller-supplied WithStreaming callback with their internal
// iterator callback instead of displacing it: both must see every chunk.
func TestGenerateStreamChainsUserCallback(t *testing.T) {
	r := newTestRegistry(t)

	model := defineFakeModel(t, r, fakeModelConfig{
		name: "test/chunkedModel",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if cb != nil {
				if err := cb(ctx, &ModelResponseChunk{Content: []*Part{NewTextPart("one")}}); err != nil {
					return nil, err
				}
				if err := cb(ctx, &ModelResponseChunk{Content: []*Part{NewTextPart("two")}}); err != nil {
					return nil, err
				}
			}
			return &ModelResponse{
				Request: req,
				Message: NewModelTextMessage("onetwo"),
			}, nil
		},
	})

	var userChunks []string
	userCB := func(ctx context.Context, chunk *ModelResponseChunk) error {
		userChunks = append(userChunks, chunk.Text())
		return nil
	}

	var iterChunks []string
	for v, err := range GenerateStream(context.Background(), r,
		WithModel(model),
		WithPrompt("count"),
		WithStreaming(userCB),
	) {
		if err != nil {
			t.Fatalf("GenerateStream error: %v", err)
		}
		if v.Done {
			break
		}
		iterChunks = append(iterChunks, v.Chunk.Text())
	}

	want := []string{"one", "two"}
	assertEqual(t, userChunks, want)
	assertEqual(t, iterChunks, want)
}

func TestModelResponseReasoning(t *testing.T) {
	t.Run("returns reasoning from response", func(t *testing.T) {
		resp := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewReasoningPart("thinking about this...", nil),
					NewTextPart("final answer"),
				},
			},
		}

		reasoning := resp.Reasoning()

		if reasoning != "thinking about this..." {
			t.Errorf("Reasoning() = %q, want %q", reasoning, "thinking about this...")
		}
	})

	t.Run("returns empty string when no reasoning", func(t *testing.T) {
		resp := &ModelResponse{
			Message: NewModelTextMessage("just text"),
		}

		reasoning := resp.Reasoning()

		if reasoning != "" {
			t.Errorf("Reasoning() = %q, want empty string", reasoning)
		}
	})
}

func TestModelResponseHistory(t *testing.T) {
	userMsg := NewUserTextMessage("question")
	modelMsg := NewModelTextMessage("answer")

	t.Run("combines request messages with the response message", func(t *testing.T) {
		resp := &ModelResponse{
			Request: &ModelRequest{Messages: []*Message{userMsg}},
			Message: modelMsg,
		}

		history := resp.History()

		if len(history) != 2 || history[0] != userMsg || history[1] != modelMsg {
			t.Errorf("History() = %v, want [userMsg modelMsg]", history)
		}
	})

	t.Run("returns nil for a nil response", func(t *testing.T) {
		var resp *ModelResponse

		if history := resp.History(); history != nil {
			t.Errorf("History() = %v, want nil", history)
		}
	})

	t.Run("returns the response message when request is nil", func(t *testing.T) {
		resp := &ModelResponse{Message: modelMsg}

		history := resp.History()

		if len(history) != 1 || history[0] != modelMsg {
			t.Errorf("History() = %v, want [modelMsg]", history)
		}
	})

	t.Run("returns request messages when response message is nil", func(t *testing.T) {
		resp := &ModelResponse{Request: &ModelRequest{Messages: []*Message{userMsg}}}

		history := resp.History()

		if len(history) != 1 || history[0] != userMsg {
			t.Errorf("History() = %v, want [userMsg]", history)
		}
	})

	t.Run("returns nil when request and response message are both nil", func(t *testing.T) {
		resp := &ModelResponse{}

		if history := resp.History(); history != nil {
			t.Errorf("History() = %v, want nil", history)
		}
	})

	// Request.Messages with spare capacity used to let History() append into the
	// caller's backing array, so the result aliased whatever else pointed at it.
	t.Run("does not write into spare capacity of request messages", func(t *testing.T) {
		backing := make([]*Message, 3, 8)
		backing[0], backing[1], backing[2] = userMsg, modelMsg, userMsg
		sentinel := NewUserTextMessage("caller owns this slot")
		extended := backing[:4]
		extended[3] = sentinel

		resp := &ModelResponse{
			Request: &ModelRequest{Messages: backing},
			Message: modelMsg,
		}

		resp.History()

		if extended[3] != sentinel {
			t.Errorf("History() overwrote the caller's backing array at index 3: got %v, want the sentinel", extended[3])
		}
	})

	t.Run("successive calls return independent slices", func(t *testing.T) {
		backing := make([]*Message, 3, 8)
		backing[0], backing[1], backing[2] = userMsg, modelMsg, userMsg
		resp := &ModelResponse{
			Request: &ModelRequest{Messages: backing},
			Message: modelMsg,
		}

		first := resp.History()
		replacement := NewModelTextMessage("second answer")
		resp.Message = replacement
		second := resp.History()

		if first[3] != modelMsg {
			t.Errorf("first History() result was mutated by the second call: got %v, want the original model message", first[3])
		}
		if second[3] != replacement {
			t.Errorf("second History() = %v at index 3, want the replacement message", second[3])
		}
	})
}

func TestModelResponseInterrupts(t *testing.T) {
	t.Run("returns interrupt tool requests", func(t *testing.T) {
		interruptPart := NewToolRequestPart(&ToolRequest{
			Name:  "confirmAction",
			Input: map[string]any{},
		})
		interruptPart.Metadata = map[string]any{"interrupt": true}

		resp := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Please confirm"),
					interruptPart,
				},
			},
		}

		interrupts := resp.Interrupts()

		if len(interrupts) != 1 {
			t.Fatalf("len(Interrupts()) = %d, want 1", len(interrupts))
		}
		if interrupts[0].ToolRequest.Name != "confirmAction" {
			t.Errorf("interrupt name = %q, want %q", interrupts[0].ToolRequest.Name, "confirmAction")
		}
	})

	t.Run("returns empty slice when no interrupts", func(t *testing.T) {
		resp := &ModelResponse{
			Message: NewModelTextMessage("no interrupts here"),
		}

		interrupts := resp.Interrupts()

		if len(interrupts) != 0 {
			t.Errorf("len(Interrupts()) = %d, want 0", len(interrupts))
		}
	})
}

func TestModelResponseMedia(t *testing.T) {
	t.Run("returns media URL from response", func(t *testing.T) {
		resp := &ModelResponse{
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Here's an image"),
					NewMediaPart("image/png", "data:image/png;base64,abc123"),
				},
			},
		}

		media := resp.Media()

		if media == "" {
			t.Error("Media() returned empty string")
		}
		if media != "data:image/png;base64,abc123" {
			t.Errorf("Media() = %q, want %q", media, "data:image/png;base64,abc123")
		}
	})

	t.Run("returns empty string when no media", func(t *testing.T) {
		resp := &ModelResponse{
			Message: NewModelTextMessage("just text"),
		}

		media := resp.Media()

		if media != "" {
			t.Errorf("Media() = %q, want empty string", media)
		}
	})
}

func TestOutputFrom(t *testing.T) {
	type TestData struct {
		Name  string `json:"name"`
		Count int    `json:"count"`
	}

	t.Run("extracts typed output from response", func(t *testing.T) {
		resp := &ModelResponse{
			Message: NewModelTextMessage(`{"name": "test", "count": 5}`),
		}

		output := OutputFrom[TestData](resp)

		if output.Name != "test" {
			t.Errorf("output.Name = %q, want %q", output.Name, "test")
		}
		if output.Count != 5 {
			t.Errorf("output.Count = %d, want 5", output.Count)
		}
	})
}

func TestGenerateWithMarkdownJSON(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(context.Background(), r)

	// A model that returns JSON wrapped in markdown
	markdownModel := defineModel(r, "test/markdownJson", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		jsonContent := "{\"name\": \"test\", \"value\": 123}"
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage("```json\n" + jsonContent + "\n```"),
		}, nil
	})

	// A model that returns JSON wrapped in markdown with loose formatting (spaces)
	looseMarkdownModel := defineModel(r, "test/looseMarkdownJson", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		jsonContent := "{\"name\": \"test\", \"value\": 123}"
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage("Here is your JSON:\n ``` json \n" + jsonContent + "\n```"),
		}, nil
	})

	type OutputData struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	t.Run("Standard Markdown JSON", func(t *testing.T) {
		resp, err := Generate(context.Background(), r,
			WithModel(markdownModel),
			WithPrompt("get data"),
			WithOutputType(OutputData{}),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		var out OutputData
		if err := resp.Output(&out); err != nil {
			t.Fatalf("Output unmarshal failed: %v", err)
		}

		if out.Name != "test" || out.Value != 123 {
			t.Errorf("Unexpected output: %+v", out)
		}
	})

	t.Run("Loose Markdown JSON", func(t *testing.T) {
		resp, err := Generate(context.Background(), r,
			WithModel(looseMarkdownModel),
			WithPrompt("get data"),
			WithOutputType(OutputData{}),
		)
		if err != nil {
			t.Fatalf("Generate failed: %v", err)
		}

		var out OutputData
		if err := resp.Output(&out); err != nil {
			t.Fatalf("Output unmarshal failed: %v", err)
		}

		if out.Name != "test" || out.Value != 123 {
			t.Errorf("Unexpected output: %+v", out)
		}
	})
}

// TestGenerateAbnormalFinishSkipsOutputParsing verifies that a response that
// did not run to a normal completion (e.g. safety-blocked) is returned as-is
// when structured output is requested, instead of failing output parsing and
// masking the finish reason with a schema error.
func TestGenerateAbnormalFinishSkipsOutputParsing(t *testing.T) {
	r := childRegistry(t)

	type OutputData struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	tests := []struct {
		name     string
		response *ModelResponse
		opts     []GenerateOption
		wantErr  error
	}{
		{
			name: "blocked contentless message with output type",
			response: &ModelResponse{
				FinishReason:  FinishReasonBlocked,
				FinishMessage: "blocked by safety settings",
				Message:       &Message{Role: RoleModel},
			},
			opts: []GenerateOption{WithOutputType(OutputData{})},
		},
		{
			name: "blocked nil message with output type",
			response: &ModelResponse{
				FinishReason:  FinishReasonBlocked,
				FinishMessage: "blocked by safety settings",
			},
			opts: []GenerateOption{WithOutputType(OutputData{})},
		},
		{
			name: "blocked contentless message with enum output",
			response: &ModelResponse{
				FinishReason:  FinishReasonBlocked,
				FinishMessage: "blocked by safety settings",
				Message:       &Message{Role: RoleModel},
			},
			opts: []GenerateOption{WithOutputEnums("YES", "NO")},
		},
		{
			name: "other finish reason keeps unparsed text",
			response: &ModelResponse{
				FinishReason:  FinishReasonOther,
				FinishMessage: "malformed function call",
				Message:       NewModelTextMessage("filter details, not JSON"),
			},
			opts: []GenerateOption{WithOutputType(OutputData{})},
		},
		{
			name: "stop with non-conforming text still fails parsing",
			response: &ModelResponse{
				FinishReason: FinishReasonStop,
				Message:      NewModelTextMessage("not json at all"),
			},
			opts:    []GenerateOption{WithOutputType(OutputData{})},
			wantErr: status.ErrInvalidOutput,
		},
		{
			// Plugins map unrecognized provider finish reasons to unknown, so
			// it keeps the parse path: only reasons known to be abnormal skip
			// output validation.
			name: "unknown with non-conforming text still fails parsing",
			response: &ModelResponse{
				FinishReason: FinishReasonUnknown,
				Message:      NewModelTextMessage("not json at all"),
			},
			opts:    []GenerateOption{WithOutputType(OutputData{})},
			wantErr: status.ErrInvalidOutput,
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := defineModel(r, fmt.Sprintf("test/abnormal-finish-%d", i), &ModelOptions{Supports: defaultModelSupports()},
				func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
					resp := *tt.response
					resp.Request = req
					return &resp, nil
				})
			wantText := tt.response.Text()

			resp, err := Generate(context.Background(), r, append([]GenerateOption{
				WithModel(model),
				WithPrompt("please respond"),
			}, tt.opts...)...)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("Generate() err = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Generate() returned error for %q response: %v", tt.response.FinishReason, err)
			}
			if resp.FinishReason != tt.response.FinishReason {
				t.Errorf("FinishReason = %q, want %q", resp.FinishReason, tt.response.FinishReason)
			}
			if resp.FinishMessage != tt.response.FinishMessage {
				t.Errorf("FinishMessage = %q, want %q", resp.FinishMessage, tt.response.FinishMessage)
			}
			if got := resp.Text(); got != wantText {
				t.Errorf("Text() = %q, want %q", got, wantText)
			}
		})
	}
}

// TestGenerateDataAbnormalFinish verifies that GenerateData and
// GenerateDataStream apply the same abnormal-finish rule as Generate: a
// response that ended blocked, aborted, interrupted, or other is handed back
// unparsed so the caller reads the finish reason, instead of being reported as
// a schema mismatch that names the wrong cause.
func TestGenerateDataAbnormalFinish(t *testing.T) {
	r := childRegistry(t)

	type Report struct {
		Title string `json:"title"`
		Score int    `json:"score"`
	}

	tests := []struct {
		name     string
		response *ModelResponse
		wantData *Report
		wantErr  error
	}{
		{
			// The common path: a provider reports a safety block and returns
			// prose explaining it, with no middleware involved. A refusal
			// cannot produce a Report, so it is an error rather than a nil
			// value the caller would read as success.
			name: "blocked with explanatory text",
			response: &ModelResponse{
				FinishReason:  FinishReasonBlocked,
				FinishMessage: "blocked by safety settings",
				Message:       NewModelTextMessage("Response was blocked for safety reasons."),
			},
			wantErr: ErrGenerationBlocked,
		},
		{
			name: "blocked with no content",
			response: &ModelResponse{
				FinishReason:  FinishReasonBlocked,
				FinishMessage: "blocked by safety settings",
				Message:       &Message{Role: RoleModel},
			},
			wantErr: ErrGenerationBlocked,
		},
		{
			// What a soft-failing middleware produces when the provider is
			// unreachable: an aborted response carrying the failure text.
			name: "aborted with failure text",
			response: &ModelResponse{
				FinishReason:  FinishReasonAborted,
				FinishMessage: "provider down",
				Message:       NewModelTextMessage("Error: provider down"),
			},
		},
		{
			// What the loop's failure partial reports; parsing must skip it
			// the same way.
			name: "failed with failure text",
			response: &ModelResponse{
				FinishReason:  FinishReasonFailed,
				FinishMessage: "provider down",
				Message:       NewModelTextMessage("Error: provider down"),
			},
		},
		{
			name: "other with filter details",
			response: &ModelResponse{
				FinishReason:  FinishReasonOther,
				FinishMessage: "malformed function call",
				Message:       NewModelTextMessage("filter details, not JSON"),
			},
		},
		{
			// A normal completion still validates: the fix must not turn every
			// schema mismatch into a silent nil.
			name: "stop with non-conforming text still fails parsing",
			response: &ModelResponse{
				FinishReason: FinishReasonStop,
				Message:      NewModelTextMessage("not json at all"),
			},
			wantErr: status.ErrInvalidOutput,
		},
		{
			name: "stop with conforming text parses",
			response: &ModelResponse{
				FinishReason: FinishReasonStop,
				Message:      NewModelTextMessage(`{"title":"ok","score":7}`),
			},
			wantData: &Report{Title: "ok", Score: 7},
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := defineModel(r, fmt.Sprintf("test/data-abnormal-finish-%d", i), &ModelOptions{Supports: defaultModelSupports()},
				func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
					resp := *tt.response
					resp.Request = req
					return &resp, nil
				})
			opts := []GenerateOption{WithModel(model), WithPrompt("please respond")}

			t.Run("GenerateData", func(t *testing.T) {
				data, resp, err := GenerateData[Report](context.Background(), r, opts...)
				if tt.wantErr != nil {
					if !errors.Is(err, tt.wantErr) {
						t.Fatalf("GenerateData() err = %v, want %v", err, tt.wantErr)
					}
					if errors.Is(err, ErrGenerationBlocked) {
						if !strings.Contains(err.Error(), tt.response.FinishMessage) {
							t.Errorf("GenerateData() err = %v, want it to carry %q", err, tt.response.FinishMessage)
						}
						// The response rides along so the caller can still
						// inspect the turn that was refused.
						checkResponse(t, resp, tt.response)
					}
					return
				}
				if err != nil {
					t.Fatalf("GenerateData() returned error for %q response: %v", tt.response.FinishReason, err)
				}
				checkResponse(t, resp, tt.response)
				if tt.wantData == nil {
					if data != nil {
						t.Errorf("GenerateData() data = %+v, want nil", *data)
					}
					return
				}
				if data == nil {
					t.Fatalf("GenerateData() data = nil, want %+v", *tt.wantData)
				}
				if *data != *tt.wantData {
					t.Errorf("GenerateData() data = %+v, want %+v", *data, *tt.wantData)
				}
			})

			t.Run("GenerateDataStream", func(t *testing.T) {
				var (
					final *StreamValue[Report, Report]
					err   error
				)
				for v, streamErr := range GenerateDataStream[Report](context.Background(), r, opts...) {
					if streamErr != nil {
						err = streamErr
						break
					}
					if v.Done {
						final = v
					}
				}
				if tt.wantErr != nil {
					if !errors.Is(err, tt.wantErr) {
						t.Fatalf("GenerateDataStream() err = %v, want %v", err, tt.wantErr)
					}
					return
				}
				if err != nil {
					t.Fatalf("GenerateDataStream() returned error for %q response: %v", tt.response.FinishReason, err)
				}
				if final == nil {
					t.Fatal("GenerateDataStream() never yielded a done value")
				}
				checkResponse(t, final.Response, tt.response)
				want := Report{}
				if tt.wantData != nil {
					want = *tt.wantData
				}
				if final.Output != want {
					t.Errorf("GenerateDataStream() output = %+v, want %+v", final.Output, want)
				}
			})
		})
	}
}

// TestGenerateDataStreamBlockedAfterChunks covers the path the one-shot tests
// miss: a model that streams output and only then reports a block. Chunks are
// parsed as they arrive, before any finish reason exists, so the caller can see
// a populated chunk for a generation that is ultimately refused. The contract
// is that the terminal value settles it, and here that means the refusal
// surfaces as an error rather than as a zeroed Output the caller reads as an
// empty answer.
func TestGenerateDataStreamBlockedAfterChunks(t *testing.T) {
	r := childRegistry(t)

	type Report struct {
		Title string `json:"title"`
	}

	tests := []struct {
		name      string
		streamed  string
		wantChunk *Report
	}{
		{
			// Partial structured output arrives, then the block lands.
			name:      "partial json then blocked",
			streamed:  `{"title":"partial`,
			wantChunk: &Report{Title: "partial"},
		},
		{
			// A refusal streamed as prose parses to nothing useful, which must
			// not be mistaken for a stream failure: the finish reason still has
			// to reach the caller.
			name:     "prose then blocked",
			streamed: "I cannot help with that request.",
		},
	}

	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := defineModel(r, fmt.Sprintf("test/streamBlocked-%d", i), &ModelOptions{Supports: defaultModelSupports()},
				func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
					if cb != nil {
						if err := cb(ctx, &ModelResponseChunk{Content: []*Part{NewTextPart(tt.streamed)}}); err != nil {
							return nil, err
						}
					}
					return &ModelResponse{
						Request:       req,
						FinishReason:  FinishReasonBlocked,
						FinishMessage: "blocked by safety settings",
						Message:       NewModelTextMessage(tt.streamed),
					}, nil
				})

			var (
				chunks []Report
				gotErr error
				done   bool
			)
			for v, err := range GenerateDataStream[Report](context.Background(), r,
				WithModel(model), WithPrompt("please respond")) {
				if err != nil {
					gotErr = err
					break
				}
				if v.Done {
					done = true
					continue
				}
				chunks = append(chunks, v.Chunk)
			}

			if !errors.Is(gotErr, ErrGenerationBlocked) {
				t.Fatalf("stream err = %v, want %v", gotErr, ErrGenerationBlocked)
			}
			if !strings.Contains(gotErr.Error(), "blocked by safety settings") {
				t.Errorf("stream err = %v, want it to carry the finish message", gotErr)
			}
			if done {
				t.Error("stream yielded a done value for a refused generation")
			}
			if tt.wantChunk != nil {
				// Documenting the provisional chunk rather than asserting it
				// away: it is why the terminal value has to be authoritative.
				if len(chunks) == 0 || chunks[len(chunks)-1] != *tt.wantChunk {
					t.Errorf("chunks = %+v, want the last to be %+v", chunks, *tt.wantChunk)
				}
			}
		})
	}
}

// checkResponse asserts that the caller was handed the model's own finish
// reason, message, and text rather than a rewritten or parsed stand-in.
func checkResponse(t *testing.T, got, want *ModelResponse) {
	t.Helper()
	if got == nil {
		t.Fatal("response = nil, want the model response")
	}
	if got.FinishReason != want.FinishReason {
		t.Errorf("FinishReason = %q, want %q", got.FinishReason, want.FinishReason)
	}
	if got.FinishMessage != want.FinishMessage {
		t.Errorf("FinishMessage = %q, want %q", got.FinishMessage, want.FinishMessage)
	}
	if got.Text() != want.Text() {
		t.Errorf("Text() = %q, want %q", got.Text(), want.Text())
	}
}

func TestGenerateNoGoroutineLeak(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(t.Context(), r)

	done := make(chan struct{})

	slowTool := defineTool(r, "slow", "slow",
		func(*ToolContext, any) (any, error) {
			<-done
			return nil, nil
		},
	)

	failTool := defineTool(r, "fail", "fail",
		func(*ToolContext, any) (any, error) {
			return nil, errors.New("boom")
		},
	)

	testModel := defineModel(r, "test/testModel", &ModelOptions{
		Supports: &ModelSupports{
			Multiturn: true,
			Tools:     true,
		},
	}, func(_ context.Context, req *ModelRequest, _ ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Request: req,
			Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewToolRequestPart(&ToolRequest{Name: "slow", Input: map[string]any{}, Ref: "a"}),
					NewToolRequestPart(&ToolRequest{Name: "slow", Input: map[string]any{}, Ref: "b"}),
					NewToolRequestPart(&ToolRequest{Name: "slow", Input: map[string]any{}, Ref: "c"}),
					NewToolRequestPart(&ToolRequest{Name: "fail", Input: map[string]any{}, Ref: "d"}),
				},
			},
		}, nil
	})

	before := runtime.NumGoroutine()

	if _, err := Generate(t.Context(), r,
		WithModel(testModel),
		WithTools(slowTool, failTool),
	); err == nil {
		t.Fatal("expected tool error")
	}

	close(done)

	for i := 0; i < 100; i++ {
		if runtime.NumGoroutine() <= before {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}

	t.Fatalf("goroutines leaked: %d", runtime.NumGoroutine()-before)
}

func TestMessageText(t *testing.T) {
	tests := []struct {
		name    string
		message *Message
		want    string
	}{
		{
			name:    "nil message",
			message: nil,
			want:    "",
		},
		{
			name:    "no content",
			message: &Message{Role: RoleModel},
			want:    "",
		},
		{
			name:    "lone text part",
			message: NewModelTextMessage("hello"),
			want:    "hello",
		},
		{
			// A model allowed to answer with an image often answers with only
			// an image. Its data is not the message's text.
			name:    "lone media part",
			message: NewMessage(RoleModel, nil, NewMediaPart("image/png", "iVBORw0KGgo=")),
			want:    "",
		},
		{
			// A data part carries a blob, which the plugins send as bytes.
			// Concatenating it would splice base64 into the message's prose.
			name:    "lone data part",
			message: NewMessage(RoleModel, nil, NewDataPart("data:application/octet-stream;base64,AAAA")),
			want:    "",
		},
		{
			name: "data part alongside text",
			message: NewMessage(RoleModel, nil,
				NewTextPart("here is the blob"),
				NewDataPart("data:application/octet-stream;base64,AAAA"),
			),
			want: "here is the blob",
		},
		{
			name:    "lone reasoning part",
			message: NewMessage(RoleModel, nil, NewReasoningPart("thinking", nil)),
			want:    "",
		},
		{
			name: "text alongside media",
			message: NewMessage(RoleModel, nil,
				NewTextPart("a drawing of a cat"),
				NewMediaPart("image/png", "iVBORw0KGgo="),
			),
			want: "a drawing of a cat",
		},
		{
			name: "text parts concatenate",
			message: NewMessage(RoleModel, nil,
				NewTextPart("one "),
				NewTextPart("two"),
			),
			want: "one two",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.message.Text(); got != tt.want {
				t.Errorf("Text() = %q, want %q", got, tt.want)
			}
			// ModelResponse.Text() reads through to the message, so the two
			// must not disagree.
			resp := &ModelResponse{Message: tt.message}
			if got := resp.Text(); got != tt.want {
				t.Errorf("ModelResponse.Text() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMediaParts(t *testing.T) {
	t.Run("returns every media part with its content type", func(t *testing.T) {
		resp := &ModelResponse{
			Message: NewMessage(RoleModel, nil,
				NewTextPart("two drawings"),
				NewMediaPart("image/png", "first"),
				NewMediaPart("image/jpeg", "second"),
			),
		}

		parts := resp.MediaParts()

		if len(parts) != 2 {
			t.Fatalf("MediaParts() returned %d parts, want 2", len(parts))
		}
		for i, want := range []struct{ contentType, data string }{
			{"image/png", "first"},
			{"image/jpeg", "second"},
		} {
			if parts[i].ContentType != want.contentType || parts[i].Text != want.data {
				t.Errorf("part %d = (%q, %q), want (%q, %q)",
					i, parts[i].ContentType, parts[i].Text, want.contentType, want.data)
			}
		}
		// Media returns only the first, which is why MediaParts exists.
		if got := resp.Media(); got != "first" {
			t.Errorf("Media() = %q, want %q", got, "first")
		}
	})

	t.Run("returns nil when there is no media", func(t *testing.T) {
		resp := &ModelResponse{Message: NewModelTextMessage("just text")}

		if parts := resp.MediaParts(); parts != nil {
			t.Errorf("MediaParts() = %v, want nil", parts)
		}
	})

	t.Run("nil safe", func(t *testing.T) {
		var resp *ModelResponse
		if parts := resp.MediaParts(); parts != nil {
			t.Errorf("MediaParts() on nil response = %v, want nil", parts)
		}
		var msg *Message
		if parts := msg.MediaParts(); parts != nil {
			t.Errorf("MediaParts() on nil message = %v, want nil", parts)
		}
	})
}

// generateSpanMessageCounts returns, sorted, the number of messages each
// collected span named "generate" recorded as its input.
func generateSpanMessageCounts(t *testing.T, c *spanCollector) []int {
	t.Helper()
	var counts []int
	for _, s := range c.allByName("generate") {
		input, ok := spanAttr(s, "genkit:input")
		if !ok {
			t.Errorf("generate span recorded no input")
			continue
		}
		var decoded struct {
			Messages []*Message `json:"messages"`
		}
		if err := json.Unmarshal([]byte(input), &decoded); err != nil {
			t.Errorf("generate span input is not a request: %v", err)
			continue
		}
		counts = append(counts, len(decoded.Messages))
	}
	slices.Sort(counts)
	return counts
}

// TestGenerateSpanRecordsAccumulatedMessages checks that every tool-loop turn
// opens a generate span recording the conversation as of that turn, not the
// messages the call started with.
func TestGenerateSpanRecordsAccumulatedMessages(t *testing.T) {
	// Two tool calls and a final answer: three model calls in all, whose
	// requests hold 1, 3, and 5 messages as each turn appends a model and a
	// tool message. Through the action the counts are the same, since the
	// action's own span stands in for the first turn's: a fourth count here
	// would mean the two are duplicating each other.
	const turns = 3
	want := []int{1, 3, 5}

	setup := func(t *testing.T) (api.Registry, *spanCollector) {
		t.Helper()
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name:    "test/loopModel",
			handler: loopingToolModel("myTool", turns-1),
		})
		defineTool(r, "myTool", "A test tool",
			func(ctx *ToolContext, in map[string]any) (string, error) { return "ok", nil })
		return r, collectSpans(t)
	}

	t.Run("through Generate", func(t *testing.T) {
		r, spans := setup(t)

		_, err := Generate(testCtx, r,
			WithModelName("test/loopModel"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
		)
		assertNoError(t, err)

		if got := generateSpanMessageCounts(t, spans); !slices.Equal(got, want) {
			t.Errorf("generate spans recorded %v messages, want %v", got, want)
		}
		// The loop's spans are annotated by type alone. A subtype would show
		// up in their trace paths, and in every path built under them.
		for _, s := range spans.allByName("generate") {
			assertSpanAttr(t, s, "genkit:type", "util")
			if got, ok := spanAttr(s, "genkit:metadata:subtype"); ok {
				t.Errorf("generate span carries subtype %q, want none", got)
			}
		}
	})

	t.Run("through the generate action", func(t *testing.T) {
		r, spans := setup(t)
		action := DefineGenerateAction(testCtx, r)

		_, err := action.Run(testCtx, &GenerateActionOptions{
			Model:    "test/loopModel",
			Messages: []*Message{NewUserTextMessage("start")},
			Tools:    []string{"myTool"},
		}, nil)
		assertNoError(t, err)

		if got := generateSpanMessageCounts(t, spans); !slices.Equal(got, want) {
			t.Errorf("generate spans recorded %v messages, want %v", got, want)
		}
	})
}

// TestToolLoopLeavesEarlierRequestsAlone checks that each turn of the tool loop
// works from its own request: middleware holds these, so reusing one would
// retroactively add the next turn's messages to a request a hook read.
func TestToolLoopLeavesEarlierRequestsAlone(t *testing.T) {
	r := newTestRegistry(t)
	var seen []*ModelRequest
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/loopModel",
		handler: loopingToolModel("myTool", 2),
	})
	defineTool(r, "myTool", "A test tool",
		func(ctx *ToolContext, in map[string]any) (string, error) { return "ok", nil })

	recorder := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
		return &Hooks{
			WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
				seen = append(seen, p.Request)
				return next(ctx, p)
			},
		}, nil
	})

	_, err := Generate(testCtx, r,
		WithModelName("test/loopModel"),
		WithPrompt("start"),
		WithTools(LookupTool(r, "myTool")),
		WithUse(recorder),
	)
	assertNoError(t, err)

	// One request per turn, each holding what the turn before it appended,
	// still true once the whole loop has run.
	want := []int{1, 3, 5}
	if len(seen) != len(want) {
		t.Fatalf("middleware saw %d requests, want %d", len(seen), len(want))
	}
	for i, n := range want {
		if got := len(seen[i].Messages); got != n {
			t.Errorf("request %d holds %d messages after the loop finished, want %d", i, got, n)
		}
	}
}

// interruptedForResume runs a generate that stops on a tool interrupt and
// returns the registry, the tool, and the interrupted response, ready for a
// resuming call.
func interruptedForResume(t *testing.T) (api.Registry, Tool, *ModelResponse) {
	t.Helper()
	r := childRegistry(t)
	tool := defineTool(r, "conditional", "A tool that interrupts on request",
		func(ctx *ToolContext, in conditionalToolInput) (string, error) {
			if in.Interrupt {
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"reason": "test"}})
			}
			return "processed: " + in.Value, nil
		})
	defineFakeModel(t, r, fakeModelConfig{
		name:    "test/resumeModel",
		handler: toolCallingModelHandler("conditional", map[string]any{"Value": "v", "Interrupt": true}, "done"),
	})

	res, err := Generate(testCtx, r, WithModelName("test/resumeModel"),
		WithPrompt("go"), WithTools(tool))
	assertNoError(t, err)
	if res.FinishReason != "interrupted" {
		t.Fatalf("setup: finish reason = %q, want %q", res.FinishReason, "interrupted")
	}
	return r, tool, res
}

// TestResumeCarriesOptionsForward checks the options the loop switches to once
// a resumed call has replayed its tools. It reads them for the rest of the
// call, so whatever the revised copy drops is lost from that point on.
func TestResumeCarriesOptionsForward(t *testing.T) {
	t.Run("keeps the caller's step name", func(t *testing.T) {
		r, tool, res := interruptedForResume(t)
		respond := tool.Respond(res.Message.Content[0], "answer", nil)
		spans := collectSpans(t)

		_, err := Generate(testCtx, r, WithModelName("test/resumeModel"),
			WithMessages(res.History()...), WithTools(tool),
			WithToolResponses(respond), WithStepName("myStep"))
		assertNoError(t, err)

		// Both turns are the caller's step, so neither may fall back to the default.
		if got := len(spans.allByName("generate")); got != 0 {
			t.Errorf("got %d spans named %q, want 0: every iteration is the named step", got, "generate")
		}
		if got := len(spans.allByName("myStep")); got != 2 {
			t.Errorf("got %d spans named %q, want 2 (one per iteration)", got, "myStep")
		}
	})

	t.Run("resume survives a hook writing to its options", func(t *testing.T) {
		r, tool, res := interruptedForResume(t)
		respond := tool.Respond(res.Message.Content[0], "answer", nil)

		clobber := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
			return &Hooks{
				WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
					if p.Options.Resume != nil {
						p.Options.Resume.Respond = nil
						p.Options.Resume.Restart = nil
					}
					return next(ctx, p)
				},
			}, nil
		})

		out, err := Generate(testCtx, r, WithModelName("test/resumeModel"),
			WithMessages(res.History()...), WithTools(tool),
			WithToolResponses(respond), WithUse(clobber))
		assertNoError(t, err)

		// The hook's writes land on its own copy, so the call resumes as usual.
		if out.FinishReason == "interrupted" {
			t.Error("call stayed interrupted: a hook's write to Options reached the loop")
		}
		if got := out.Text(); got != "done" {
			t.Errorf("resumed response = %q, want %q", got, "done")
		}
	})
}

// TestGeneratePartialResponseOnFailure covers the partial-response contract:
// when the generate loop fails after the request has resolved, the classified
// error is returned together with a partial [ModelResponse] that preserves
// the loop's progress.
func TestGeneratePartialResponseOnFailure(t *testing.T) {
	t.Parallel()

	// loopSetup defines a model that always answers with a tool request and a
	// tool that succeeds, so only the configured turn limit ends the loop.
	loopSetup := func(t *testing.T) api.Registry {
		t.Helper()
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name:    "test/loopModel",
			handler: loopingToolModel("myTool", 100),
		})
		defineTool(r, "myTool", "A test tool",
			func(ctx *ToolContext, in map[string]any) (string, error) { return "ok", nil })
		return r
	}

	// badJSONSetup defines a model whose final text does not parse against a
	// requested JSON output schema.
	badJSONSetup := func(t *testing.T) api.Registry {
		t.Helper()
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/badJSON",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request:      req,
					FinishReason: FinishReasonStop,
					Message:      NewModelTextMessage("this is not json"),
				}, nil
			},
		})
		return r
	}

	t.Run("max turns drops the round it will not run", func(t *testing.T) {
		r := loopSetup(t)

		resp, err := Generate(testCtx, r,
			WithModelName("test/loopModel"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
			WithMaxTurns(2),
		)
		if !errors.Is(err, ErrMaxTurnsExceeded) {
			t.Fatalf("err = %v, want ErrMaxTurnsExceeded", err)
		}
		if resp == nil {
			t.Fatal("response is nil, want the loop's partial response")
		}
		// The caller set the limit, so reaching it stopped the loop rather
		// than breaking it.
		if resp.FinishReason != FinishReasonAborted {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, FinishReasonAborted)
		}
		if !strings.Contains(resp.FinishMessage, "exceeded maximum tool call iterations (2)") {
			t.Errorf("FinishMessage = %q, want the max-turns cause", resp.FinishMessage)
		}
		// The same cause classified, so a consumer reading the response as
		// data branches on a status rather than matching the message.
		if resp.Error == nil {
			t.Fatal("Error is nil, want the classified cause")
		}
		if resp.Error.Status != status.Aborted {
			t.Errorf("Error.Status = %q, want %q", resp.Error.Status, status.Aborted)
		}
		if resp.Error.Message != resp.FinishMessage {
			t.Errorf("Error.Message = %q, want the FinishMessage %q", resp.Error.Message, resp.FinishMessage)
		}
		// Two completed rounds and nothing else: the model message whose
		// tools the limit refused to run goes with the round it opened.
		history := resp.History()
		if len(history) != 5 {
			t.Fatalf("History() has %d messages, want 5 (user, model, tool, model, tool)", len(history))
		}
		if history[4].Role != RoleTool {
			t.Errorf("history ends with a %s message, want the tool message closing the last round", history[4].Role)
		}
		if resp.Message != nil {
			t.Errorf("Message = %v, want nil: the refused round is not handed back", resp.Message)
		}
	})

	t.Run("model failure keeps completed turns", func(t *testing.T) {
		r := newTestRegistry(t)
		calls := 0
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/failsSecond",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				calls++
				if calls == 1 {
					return &ModelResponse{Request: req, Message: &Message{
						Role:    RoleModel,
						Content: []*Part{NewToolRequestPart(&ToolRequest{Name: "myTool", Input: map[string]any{}})},
					}}, nil
				}
				return nil, errors.New("model exploded")
			},
		})
		defineTool(r, "myTool", "A test tool",
			func(ctx *ToolContext, in map[string]any) (string, error) { return "ok", nil })

		resp, err := Generate(testCtx, r,
			WithModelName("test/failsSecond"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
		)
		errorContains(t, err, "model exploded")
		if resp == nil {
			t.Fatal("response is nil, want the loop's partial response")
		}
		if resp.FinishReason != FinishReasonFailed {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, FinishReasonFailed)
		}
		if !strings.Contains(resp.FinishMessage, "model exploded") {
			t.Errorf("FinishMessage = %q, want the model error", resp.FinishMessage)
		}
		if resp.Message != nil {
			t.Errorf("Message = %v, want nil: the failing call produced nothing", resp.Message)
		}
		history := resp.History()
		if len(history) != 3 {
			t.Fatalf("History() has %d messages, want 3 (user, model, tool)", len(history))
		}
		if history[1].Role != RoleModel || history[2].Role != RoleTool {
			t.Errorf("history roles = %s, %s, want model, tool", history[1].Role, history[2].Role)
		}
	})

	t.Run("cancellation reports aborted, not failed", func(t *testing.T) {
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/cancelled",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return nil, fmt.Errorf("call gave up: %w", context.Canceled)
			},
		})

		resp, err := Generate(testCtx, r,
			WithModelName("test/cancelled"),
			WithPrompt("start"),
		)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		if resp == nil {
			t.Fatal("response is nil, want the loop's partial response")
		}
		if resp.FinishReason != FinishReasonAborted {
			t.Errorf("FinishReason = %q, want %q: the caller stopped the loop", resp.FinishReason, FinishReasonAborted)
		}
		if resp.Error == nil || resp.Error.Status != status.Cancelled {
			t.Errorf("Error = %+v, want a CANCELLED classification", resp.Error)
		}
	})

	t.Run("a service that answers ABORTED reports failed", func(t *testing.T) {
		// The status map turns HTTP 409 into ABORTED and 504 into
		// DEADLINE_EXCEEDED, so a provider stamping the service's own status
		// reaches the same names a stopped caller does. Only the context and
		// the limits the caller set say aborted: a service dropping the
		// request broke the run, which is what a retry client reads.
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/serviceAborted",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return nil, status.Errorf(status.ErrAborted, "409 from the service")
			},
		})

		resp, err := Generate(testCtx, r,
			WithModelName("test/serviceAborted"),
			WithPrompt("start"),
		)
		if err == nil {
			t.Fatal("expected the service's error")
		}
		if resp == nil {
			t.Fatal("response is nil, want the loop's partial response")
		}
		if resp.FinishReason != FinishReasonFailed {
			t.Errorf("FinishReason = %q, want %q: the service stopped the request, not the caller", resp.FinishReason, FinishReasonFailed)
		}
		// The classification is untouched; only who ended the run is decided
		// here.
		if resp.Error == nil || resp.Error.Status != status.Aborted {
			t.Errorf("Error = %+v, want the service's ABORTED preserved", resp.Error)
		}
	})

	t.Run("cancellation inside a tool reports aborted", func(t *testing.T) {
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/toolCancel",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{Request: req, Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewToolRequestPart(&ToolRequest{Name: "blockingTool", Input: map[string]any{}})},
				}}, nil
			},
		})
		running := make(chan struct{})
		defineTool(r, "blockingTool", "blocks until the call is cancelled",
			func(tc *ToolContext, in map[string]any) (string, error) {
				close(running)
				<-tc.Context.Done()
				return "", tc.Context.Err()
			})

		ctx, cancel := context.WithCancel(testCtx)
		go func() {
			<-running
			cancel()
		}()
		resp, err := Generate(ctx, r,
			WithModelName("test/toolCancel"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "blockingTool")),
		)
		// The tool stopped because the caller did, so the loop reports the
		// cancellation rather than blaming the tool for it.
		if errors.Is(err, ErrToolFailed) {
			t.Errorf("err = %v, want a cancellation rather than ErrToolFailed", err)
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		if resp == nil {
			t.Fatal("response is nil, want the loop's partial response")
		}
		if resp.FinishReason != FinishReasonAborted {
			t.Errorf("FinishReason = %q, want %q: the caller stopped the loop", resp.FinishReason, FinishReasonAborted)
		}
		if resp.Error == nil || resp.Error.Status != status.Cancelled {
			t.Errorf("Error = %+v, want a CANCELLED classification", resp.Error)
		}
		history := resp.History()
		if len(history) != 1 || history[0].Role != RoleUser {
			t.Errorf("History() = %d messages, want the user message alone: the unfinished round goes", len(history))
		}
	})

	t.Run("mid-stream failure drops the unfinished message", func(t *testing.T) {
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/failsMidStream",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				cb(ctx, &ModelResponseChunk{Content: []*Part{NewTextPart("Hello, ")}})
				cb(ctx, &ModelResponseChunk{Content: []*Part{NewTextPart("wor")}})
				return nil, errors.New("stream died")
			},
		})

		var streamed strings.Builder
		resp, err := Generate(testCtx, r,
			WithModelName("test/failsMidStream"),
			WithPrompt("start"),
			WithStreaming(func(ctx context.Context, c *ModelResponseChunk) error {
				streamed.WriteString(c.Text())
				return nil
			}),
		)
		errorContains(t, err, "stream died")
		if resp == nil {
			t.Fatal("response is nil, want the loop's partial response")
		}
		// The prefix reached the caller through the stream; it does not also
		// ride back on a conversation the caller would send again.
		if got := streamed.String(); got != "Hello, wor" {
			t.Errorf("streamed = %q, want the prefix %q", got, "Hello, wor")
		}
		if resp.Message != nil {
			t.Errorf("Message = %v, want nil: the model never finished it", resp.Message)
		}
		if got := len(resp.History()); got != 1 {
			t.Errorf("History() has %d messages, want 1 (the user message alone)", got)
		}
		if resp.FinishReason != FinishReasonFailed {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, FinishReasonFailed)
		}
	})

	t.Run("tool failure drops the whole round", func(t *testing.T) {
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/twoTools",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{Request: req, Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{Name: "boomTool", Input: map[string]any{}}),
						NewToolRequestPart(&ToolRequest{Name: "okTool", Input: map[string]any{}}),
					},
				}}, nil
			},
		})
		// The handshake orders the failure after okTool completes, so the
		// round demonstrably held a finished sibling before it was dropped.
		okDone := make(chan struct{})
		defineTool(r, "boomTool", "always fails",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				<-okDone
				return "", errors.New("boom")
			})
		defineTool(r, "okTool", "succeeds",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				defer close(okDone)
				return "fine", nil
			})

		resp, err := Generate(testCtx, r,
			WithModelName("test/twoTools"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "boomTool"), LookupTool(r, "okTool")),
		)
		if !errors.Is(err, ErrToolFailed) {
			t.Fatalf("err = %v, want ErrToolFailed", err)
		}
		if resp == nil {
			t.Fatal("response is nil, want the loop's partial response")
		}
		if resp.FinishReason != FinishReasonFailed {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, FinishReasonFailed)
		}
		// okTool finished, but its output cannot be handed back on its own:
		// the round's other request has no response, and a conversation that
		// ends there is one no provider accepts.
		if resp.Message != nil {
			t.Errorf("Message = %v, want nil: the round goes with the tool that failed", resp.Message)
		}
		if got := len(resp.ToolRequests()); got != 0 {
			t.Errorf("ToolRequests() = %d parts, want none", got)
		}
		history := resp.History()
		if len(history) != 1 || history[0].Role != RoleUser {
			t.Errorf("History() = %d messages ending in %v, want the user message alone", len(history), history[len(history)-1].Role)
		}
	})

	t.Run("invalid structured output returns the full response", func(t *testing.T) {
		r := badJSONSetup(t)

		resp, err := Generate(testCtx, r,
			WithModelName("test/badJSON"),
			WithPrompt("start"),
			WithOutputType(StructuredResponse{}),
		)
		if !errors.Is(err, status.ErrInvalidOutput) {
			t.Fatalf("err = %v, want ErrInvalidOutput", err)
		}
		if resp == nil {
			t.Fatal("response is nil, want the model's full response")
		}
		// The model finished; only post-processing failed. The response keeps
		// the model's own finish reason and raw output.
		if resp.FinishReason != FinishReasonStop {
			t.Errorf("FinishReason = %q, want the model's own %q", resp.FinishReason, FinishReasonStop)
		}
		if got := resp.Text(); got != "this is not json" {
			t.Errorf("Text() = %q, want the raw model output", got)
		}
	})

	t.Run("GenerateData returns the partial response with the error", func(t *testing.T) {
		r := badJSONSetup(t)

		out, resp, err := GenerateData[StructuredResponse](testCtx, r,
			WithModelName("test/badJSON"),
			WithPrompt("start"),
		)
		if !errors.Is(err, status.ErrInvalidOutput) {
			t.Fatalf("err = %v, want ErrInvalidOutput", err)
		}
		if out != nil {
			t.Errorf("output = %v, want nil", out)
		}
		if resp == nil || resp.Text() != "this is not json" {
			t.Errorf("response = %v, want the raw model output", resp)
		}
	})

	t.Run("GenerateText returns the partial text", func(t *testing.T) {
		r := badJSONSetup(t)

		text, err := GenerateText(testCtx, r,
			WithModelName("test/badJSON"),
			WithPrompt("start"),
			WithOutputType(StructuredResponse{}),
		)
		if !errors.Is(err, status.ErrInvalidOutput) {
			t.Fatalf("err = %v, want ErrInvalidOutput", err)
		}
		if text != "this is not json" {
			t.Errorf("text = %q, want the raw model output", text)
		}
	})

	t.Run("a hook dropping the response still yields the partial", func(t *testing.T) {
		r := loopSetup(t)
		drop := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
			return &Hooks{
				WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
					resp, err := next(ctx, p)
					if err != nil {
						return nil, err // The common reflex that loses the response.
					}
					return resp, nil
				},
			}, nil
		})

		resp, err := Generate(testCtx, r,
			WithModelName("test/loopModel"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
			WithMaxTurns(1),
			WithUse(drop),
		)
		if !errors.Is(err, ErrMaxTurnsExceeded) {
			t.Fatalf("err = %v, want ErrMaxTurnsExceeded", err)
		}
		if resp == nil {
			t.Fatal("response is nil, want the loop's recorded partial restored")
		}
		// The caller set the limit, so reaching it stopped the loop rather
		// than breaking it.
		if resp.FinishReason != FinishReasonAborted {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, FinishReasonAborted)
		}
		// One completed round: the model message the limit refused goes with
		// the round it opened, restored partial or not.
		if got := len(resp.History()); got != 3 {
			t.Errorf("History() has %d messages, want 3 (user, model, tool)", got)
		}
	})

	t.Run("an error outside a turn synthesizes a partial", func(t *testing.T) {
		r := loopSetup(t)
		deny := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
			return &Hooks{
				WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
					if p.Iteration == 1 {
						return nil, errors.New("hook denied")
					}
					return next(ctx, p)
				},
			}, nil
		})

		resp, err := Generate(testCtx, r,
			WithModelName("test/loopModel"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
			WithUse(deny),
		)
		errorContains(t, err, "hook denied")
		if resp == nil {
			t.Fatal("response is nil, want a synthesized partial")
		}
		if resp.FinishReason != FinishReasonFailed {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, FinishReasonFailed)
		}
		if resp.Message != nil {
			t.Errorf("Message = %v, want nil: no turn produced a message for this failure", resp.Message)
		}
		// The conversation entering the denied turn: user, model, tool.
		if got := len(resp.History()); got != 3 {
			t.Errorf("History() has %d messages, want 3", got)
		}
	})

	t.Run("the stream helpers yield the partial with the error", func(t *testing.T) {
		r := loopSetup(t)

		var last *ModelStreamValue
		var streamErr error
		for v, err := range GenerateStream(testCtx, r,
			WithModelName("test/loopModel"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
			WithMaxTurns(1),
		) {
			if err != nil {
				last, streamErr = v, err
			}
		}
		if !errors.Is(streamErr, ErrMaxTurnsExceeded) {
			t.Fatalf("err = %v, want ErrMaxTurnsExceeded", streamErr)
		}
		if last == nil || last.Response == nil {
			t.Fatal("stream yielded no response with its error, want the partial")
		}
		if !last.Done {
			t.Error("the failing value is not marked Done")
		}
		if got := len(last.Response.History()); got != 3 {
			t.Errorf("History() has %d messages, want 3 (user, model, tool)", got)
		}

		var typed *StreamValue[StructuredResponse, StructuredResponse]
		for v, err := range GenerateDataStream[StructuredResponse](testCtx, r,
			WithModelName("test/loopModel"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
			WithMaxTurns(1),
		) {
			if err != nil {
				typed, streamErr = v, err
			}
		}
		if !errors.Is(streamErr, ErrMaxTurnsExceeded) {
			t.Fatalf("typed err = %v, want ErrMaxTurnsExceeded", streamErr)
		}
		if typed == nil || typed.Response == nil {
			t.Fatal("typed stream yielded no response with its error, want the partial")
		}
		var zero StructuredResponse
		if typed.Output != zero {
			t.Errorf("Output = %v, want the zero value: the call produced none", typed.Output)
		}
	})
}

// TestGenerateLoopFailureHardening covers edge cases of the partial-response
// machinery: stale partials do not survive recovered failures, sibling
// outcomes (interrupts, resolved restarts) survive into partials and
// resumes, and the model layer's own partial responses are adopted.
func TestGenerateLoopFailureHardening(t *testing.T) {
	t.Parallel()

	t.Run("a recovered failure does not leak a stale partial", func(t *testing.T) {
		r := newTestRegistry(t)
		calls := 0
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/flakyThenTool",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				calls++
				if calls == 1 {
					return nil, errors.New("transient model error")
				}
				return &ModelResponse{Request: req, Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewToolRequestPart(&ToolRequest{Name: "myTool", Input: map[string]any{}})},
				}}, nil
			},
		})
		defineTool(r, "myTool", "A test tool",
			func(ctx *ToolContext, in map[string]any) (string, error) { return "ok", nil })

		retryThenDeny := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
			return &Hooks{
				WrapGenerate: func(ctx context.Context, p *GenerateParams, next GenerateNext) (*ModelResponse, error) {
					if p.Iteration == 1 {
						return nil, errors.New("budget denied")
					}
					resp, err := next(ctx, p)
					if err != nil {
						return next(ctx, p) // Retry the turn once.
					}
					return resp, nil
				},
			}, nil
		})

		resp, err := Generate(testCtx, r,
			WithModelName("test/flakyThenTool"),
			WithPrompt("start"),
			WithTools(LookupTool(r, "myTool")),
			WithUse(retryThenDeny),
		)
		errorContains(t, err, "budget denied")
		if resp == nil {
			t.Fatal("response is nil, want a synthesized partial")
		}
		if !strings.Contains(resp.FinishMessage, "budget denied") {
			t.Errorf("FinishMessage = %q, want the current failure, not the recovered one", resp.FinishMessage)
		}
		// The conversation entering the denied turn: user, model, tool. The
		// recovered turn-0 failure must not regress it to just the user
		// message.
		if got := len(resp.History()); got != 3 {
			t.Errorf("History() has %d messages, want 3", got)
		}
	})

	t.Run("a sibling interrupt goes with the failed round", func(t *testing.T) {
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/interruptAndBoom",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{Request: req, Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{Name: "pauser", Input: map[string]any{}}),
						NewToolRequestPart(&ToolRequest{Name: "boomTool", Input: map[string]any{}}),
					},
				}}, nil
			},
		})
		pauseDone := make(chan struct{})
		pauser := defineTool(r, "pauser", "interrupts",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				defer close(pauseDone)
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"reason": "approval"}})
			})
		boom := defineTool(r, "boomTool", "fails after the interrupt arrived",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				<-pauseDone
				return "", errors.New("boom")
			})

		resp, err := Generate(testCtx, r,
			WithModelName("test/interruptAndBoom"),
			WithPrompt("start"),
			WithTools(pauser, boom),
		)
		if !errors.Is(err, ErrToolFailed) {
			t.Fatalf("err = %v, want ErrToolFailed", err)
		}
		// An interrupt is a resume point only while its round can still be
		// answered. This one cannot, so it is dropped rather than handed
		// back as an interrupt the caller could never resolve.
		if got := len(resp.Interrupts()); got != 0 {
			t.Errorf("partial has %d interrupt parts, want none", got)
		}
		if resp.Message != nil {
			t.Errorf("Message = %v, want nil", resp.Message)
		}
		if got := len(resp.History()); got != 1 {
			t.Errorf("History() has %d messages, want 1 (the user message alone)", got)
		}
	})

	t.Run("response metadata survives a resume replay", func(t *testing.T) {
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/auditPlusInterrupt",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				for _, m := range req.Messages {
					if m.Role == RoleTool {
						return &ModelResponse{Request: req, Message: NewModelTextMessage("done")}, nil
					}
				}
				return &ModelResponse{Request: req, Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{Name: "pauser", Input: map[string]any{}}),
						NewToolRequestPart(&ToolRequest{Name: "auditTool", Input: map[string]any{}}),
					},
				}}, nil
			},
		})
		pauser := defineTool(r, "pauser", "interrupts",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"reason": "approval"}})
			})
		audit := defineMultipartTool(r, "auditTool", "annotates its response",
			func(ctx *ToolContext, in map[string]any) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output:   "ok",
					Metadata: map[string]any{"audit": "checked"},
				}, nil
			})

		res, err := Generate(testCtx, r,
			WithModelName("test/auditPlusInterrupt"),
			WithPrompt("start"),
			WithTools(pauser, audit),
		)
		assertNoError(t, err)
		if res.FinishReason != FinishReasonInterrupted {
			t.Fatalf("FinishReason = %q, want interrupted", res.FinishReason)
		}

		respond := pauser.Respond(res.Interrupts()[0], "approved", nil)
		resumed, err := Generate(testCtx, r,
			WithModelName("test/auditPlusInterrupt"),
			WithMessages(res.History()...),
			WithTools(pauser, audit),
			WithToolResponses(respond),
		)
		assertNoError(t, err)

		var auditResp *Part
		for _, m := range resumed.History() {
			if m.Role != RoleTool {
				continue
			}
			for _, p := range m.Content {
				if p.IsToolResponse() && p.ToolResponse.Name == "auditTool" {
					auditResp = p
				}
			}
		}
		if auditResp == nil {
			t.Fatal("resumed history has no tool response for auditTool")
		}
		if auditResp.Metadata["audit"] != "checked" {
			t.Errorf("replayed metadata = %v, want the tool's own audit key restored", auditResp.Metadata)
		}
		if auditResp.Metadata["source"] != "pending" {
			t.Errorf("replayed metadata = %v, want source pending kept", auditResp.Metadata)
		}
	})

	t.Run("the model layer's accounting outlives its dropped message", func(t *testing.T) {
		r := newTestRegistry(t)
		defineFakeModel(t, r, fakeModelConfig{name: "test/plainModel"})

		partialModel := MiddlewareFunc(func(ctx context.Context) (*Hooks, error) {
			return &Hooks{
				WrapModel: func(ctx context.Context, p *ModelParams, next ModelNext) (*ModelResponse, error) {
					return &ModelResponse{
						Message: NewModelTextMessage("half an answer"),
						Usage:   &GenerationUsage{OutputTokens: 7},
					}, errors.New("model died")
				},
			}, nil
		})

		resp, err := Generate(testCtx, r,
			WithModelName("test/plainModel"),
			WithPrompt("start"),
			WithUse(partialModel),
		)
		errorContains(t, err, "model died")
		// The tokens were spent, so the accounting rides back; the message
		// they produced does not, since the call did not finish.
		if resp.Message != nil {
			t.Errorf("Message = %v, want nil", resp.Message)
		}
		if resp.Usage == nil || resp.Usage.OutputTokens != 7 {
			t.Errorf("Usage = %+v, want the model layer's accounting kept", resp.Usage)
		}
		if resp.FinishReason != FinishReasonFailed {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, FinishReasonFailed)
		}
	})

	t.Run("restarted interrupt with no metadata is still an interrupt", func(t *testing.T) {
		r := newTestRegistry(t)
		calls := 0
		fragile := defineTool(r, "fragile", "always interrupts",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				calls++
				return "", ctx.Interrupt(&InterruptOptions{})
			})
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/reinterrupt",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{Request: req, Message: &Message{
					Role:    RoleModel,
					Content: []*Part{NewToolRequestPart(&ToolRequest{Name: "fragile", Input: map[string]any{}})},
				}}, nil
			},
		})

		res, err := Generate(testCtx, r,
			WithModelName("test/reinterrupt"),
			WithPrompt("start"),
			WithTools(fragile),
		)
		assertNoError(t, err)

		restart := fragile.Restart(res.Interrupts()[0], nil)
		resumed, err := Generate(testCtx, r,
			WithModelName("test/reinterrupt"),
			WithMessages(res.History()...),
			WithTools(fragile),
			WithToolRestarts(restart),
		)
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("err = %v, want FAILED_PRECONDITION", err)
		}
		if resumed == nil {
			t.Fatal("response is nil, want the re-interrupted partial")
		}
		if resumed.FinishReason != FinishReasonInterrupted {
			t.Errorf("FinishReason = %q, want interrupted", resumed.FinishReason)
		}
		if got := len(resumed.Interrupts()); got != 1 {
			t.Errorf("Interrupts() = %d parts, want the metadata-free interrupt detectable", got)
		}
		if got := len(resumed.History()); got != len(res.History()) {
			t.Errorf("History() has %d messages, want %d", got, len(res.History()))
		}
		if calls != 2 {
			t.Errorf("tool ran %d times, want 2", calls)
		}
	})

	t.Run("re-interrupt preserves resolved siblings", func(t *testing.T) {
		r := newTestRegistry(t)
		aCalls, bCalls := 0, 0
		toolA := defineTool(r, "toolA", "interrupts once, then succeeds",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				aCalls++
				if aCalls == 1 {
					return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"which": "A"}})
				}
				return "A done", nil
			})
		toolB := defineTool(r, "toolB", "always interrupts",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				bCalls++
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"which": "B"}})
			})
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/twoInterrupts",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{Request: req, Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{Name: "toolA", Input: map[string]any{}}),
						NewToolRequestPart(&ToolRequest{Name: "toolB", Input: map[string]any{}}),
					},
				}}, nil
			},
		})

		res, err := Generate(testCtx, r,
			WithModelName("test/twoInterrupts"),
			WithPrompt("start"),
			WithTools(toolA, toolB),
		)
		assertNoError(t, err)

		var partA, partB *Part
		for _, p := range res.Interrupts() {
			switch p.ToolRequest.Name {
			case "toolA":
				partA = p
			case "toolB":
				partB = p
			}
		}
		resumed, err := Generate(testCtx, r,
			WithModelName("test/twoInterrupts"),
			WithMessages(res.History()...),
			WithTools(toolA, toolB),
			WithToolRestarts(toolA.Restart(partA, nil), toolB.Restart(partB, nil)),
		)
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("err = %v, want FAILED_PRECONDITION from toolB's re-interrupt", err)
		}
		if resumed == nil {
			t.Fatal("response is nil, want the re-interrupted partial")
		}
		var resolvedA *Part
		for _, p := range resumed.Message.Content {
			if p.IsToolRequest() && p.ToolRequest.Name == "toolA" {
				resolvedA = p
			}
		}
		if resolvedA == nil {
			t.Fatal("toolA's request part missing from the partial")
		}
		if resolvedA.Metadata["pendingOutput"] != "A done" {
			t.Errorf("toolA pendingOutput = %v, want its completed output preserved", resolvedA.Metadata["pendingOutput"])
		}
	})
}

// TestGenerateResumePreservesMultipartContent covers pendingContent: a
// multipart tool's content parts survive an interrupted turn and a wire
// round-trip, and reappear on the replayed tool response when the
// generation resumes.
func TestGenerateResumePreservesMultipartContent(t *testing.T) {
	t.Parallel()

	t.Run("replay restores multipart content", func(t *testing.T) {
		r := newTestRegistry(t)
		pauser := defineTool(r, "pauser", "interrupts",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"reason": "approval"}})
			})
		media := NewMultipartTool("mediaTool", "returns a chart",
			func(ctx *ToolContext, in map[string]any) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output:  "described",
					Content: []*Part{NewMediaPart("image/png", "data:image/png;base64,AAA")},
				}, nil
			})
		media.Register(r)

		defineFakeModel(t, r, fakeModelConfig{
			name: "test/mediaAndPause",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				for _, m := range req.Messages {
					if m.Role == RoleTool {
						return &ModelResponse{Request: req, Message: NewModelTextMessage("done")}, nil
					}
				}
				return &ModelResponse{Request: req, Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{Name: "pauser", Input: map[string]any{}}),
						NewToolRequestPart(&ToolRequest{Name: "mediaTool", Input: map[string]any{}}),
					},
				}}, nil
			},
		})

		res, err := Generate(testCtx, r,
			WithModelName("test/mediaAndPause"),
			WithPrompt("start"),
			WithTools(pauser, media),
		)
		assertNoError(t, err)
		if res.FinishReason != FinishReasonInterrupted {
			t.Fatalf("FinishReason = %q, want interrupted", res.FinishReason)
		}

		// A persistence round-trip degrades metadata values to generic JSON;
		// the replay must decode pendingContent from that form too.
		raw, err := json.Marshal(res.History())
		if err != nil {
			t.Fatal(err)
		}
		var msgs []*Message
		if err := json.Unmarshal(raw, &msgs); err != nil {
			t.Fatal(err)
		}

		var interruptPart *Part
		for _, p := range msgs[len(msgs)-1].Content {
			if p.IsInterrupt() {
				interruptPart = p
			}
		}
		if interruptPart == nil {
			t.Fatal("no interrupt part survived the round-trip")
		}

		resumed, err := Generate(testCtx, r,
			WithModelName("test/mediaAndPause"),
			WithMessages(msgs...),
			WithTools(pauser, media),
			WithToolResponses(pauser.Respond(interruptPart, "approved", nil)),
		)
		assertNoError(t, err)

		var mediaResp *Part
		for _, m := range resumed.History() {
			if m.Role != RoleTool {
				continue
			}
			for _, p := range m.Content {
				if p.IsToolResponse() && p.ToolResponse.Name == "mediaTool" {
					mediaResp = p
				}
			}
		}
		if mediaResp == nil {
			t.Fatal("resumed history has no tool response for mediaTool")
		}
		if mediaResp.ToolResponse.Output != "described" {
			t.Errorf("replayed output = %v, want %q", mediaResp.ToolResponse.Output, "described")
		}
		content := mediaResp.ToolResponse.Content
		if len(content) != 1 || !content[0].IsMedia() || content[0].ContentType != "image/png" {
			t.Fatalf("replayed content = %+v, want the tool's media part restored", content)
		}
		if _, ok := mediaResp.Metadata["pendingContent"]; ok {
			t.Error("pendingContent metadata leaked onto the replayed response")
		}
	})

	t.Run("re-interrupt stamps content for resolved siblings", func(t *testing.T) {
		r := newTestRegistry(t)
		aCalls := 0
		multiA := NewMultipartTool("multiA", "interrupts once, then returns media",
			func(ctx *ToolContext, in map[string]any) (*MultipartToolResponse, error) {
				aCalls++
				if aCalls == 1 {
					return nil, ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"which": "A"}})
				}
				return &MultipartToolResponse{
					Output:  "A done",
					Content: []*Part{NewMediaPart("image/png", "data:image/png;base64,BBB")},
				}, nil
			})
		multiA.Register(r)
		toolB := defineTool(r, "toolB", "always interrupts",
			func(ctx *ToolContext, in map[string]any) (string, error) {
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"which": "B"}})
			})
		defineFakeModel(t, r, fakeModelConfig{
			name: "test/multiInterrupts",
			handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{Request: req, Message: &Message{
					Role: RoleModel,
					Content: []*Part{
						NewToolRequestPart(&ToolRequest{Name: "multiA", Input: map[string]any{}}),
						NewToolRequestPart(&ToolRequest{Name: "toolB", Input: map[string]any{}}),
					},
				}}, nil
			},
		})

		res, err := Generate(testCtx, r,
			WithModelName("test/multiInterrupts"),
			WithPrompt("start"),
			WithTools(multiA, toolB),
		)
		assertNoError(t, err)

		var partA, partB *Part
		for _, p := range res.Interrupts() {
			switch p.ToolRequest.Name {
			case "multiA":
				partA = p
			case "toolB":
				partB = p
			}
		}
		resumed, err := Generate(testCtx, r,
			WithModelName("test/multiInterrupts"),
			WithMessages(res.History()...),
			WithTools(multiA, toolB),
			WithToolRestarts(multiA.Restart(partA, nil), toolB.Restart(partB, nil)),
		)
		if !errors.Is(err, status.ErrFailedPrecondition) {
			t.Fatalf("err = %v, want FAILED_PRECONDITION from toolB's re-interrupt", err)
		}
		var resolvedA *Part
		for _, p := range resumed.Message.Content {
			if p.IsToolRequest() && p.ToolRequest.Name == "multiA" {
				resolvedA = p
			}
		}
		if resolvedA == nil {
			t.Fatal("multiA's request part missing from the partial")
		}
		content, ok := resolvedA.Metadata["pendingContent"].([]*Part)
		if !ok || len(content) != 1 || !content[0].IsMedia() {
			t.Errorf("multiA pendingContent = %v, want its media part preserved", resolvedA.Metadata["pendingContent"])
		}
	})
}

// TestResumedToolMessageOrder pins that a resumed tool message carries its
// responses in the order their requests appear in the model message, not the
// order the concurrent resolutions finished, the same guarantee
// handleToolRequests gives the first run.
func TestResumedToolMessageOrder(t *testing.T) {
	t.Parallel()
	r := newTestRegistry(t)

	// On restart, beta finishes first and alpha waits for it, so completion
	// order inverts request order. The sleep gives beta's result time to
	// reach the collection loop ahead of alpha's.
	betaDone := make(chan struct{})
	aCalls, bCalls := 0, 0
	alpha := defineTool(r, "alpha", "interrupts once, then finishes last",
		func(ctx *ToolContext, in map[string]any) (string, error) {
			aCalls++
			if aCalls == 1 {
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"which": "A"}})
			}
			<-betaDone
			time.Sleep(10 * time.Millisecond)
			return "A", nil
		})
	beta := defineTool(r, "beta", "interrupts once, then finishes first",
		func(ctx *ToolContext, in map[string]any) (string, error) {
			bCalls++
			if bCalls == 1 {
				return "", ctx.Interrupt(&InterruptOptions{Metadata: map[string]any{"which": "B"}})
			}
			defer close(betaDone)
			return "B", nil
		})

	var resumedToolMsg *Message
	defineFakeModel(t, r, fakeModelConfig{
		name: "test/orderedResume",
		handler: func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			if last := req.Messages[len(req.Messages)-1]; last.Role == RoleTool {
				resumedToolMsg = last
				return &ModelResponse{Request: req, Message: NewModelTextMessage("done")}, nil
			}
			return &ModelResponse{Request: req, Message: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewToolRequestPart(&ToolRequest{Name: "alpha", Input: map[string]any{}}),
					NewToolRequestPart(&ToolRequest{Name: "beta", Input: map[string]any{}}),
				},
			}}, nil
		},
	})

	res, err := Generate(testCtx, r,
		WithModelName("test/orderedResume"),
		WithPrompt("start"),
		WithTools(alpha, beta),
	)
	assertNoError(t, err)

	var partA, partB *Part
	for _, p := range res.Interrupts() {
		switch p.ToolRequest.Name {
		case "alpha":
			partA = p
		case "beta":
			partB = p
		}
	}
	resumed, err := Generate(testCtx, r,
		WithModelName("test/orderedResume"),
		WithMessages(res.History()...),
		WithTools(alpha, beta),
		WithToolRestarts(alpha.Restart(partA, nil), beta.Restart(partB, nil)),
	)
	assertNoError(t, err)
	if got := resumed.Text(); got != "done" {
		t.Errorf("Text() = %q, want %q", got, "done")
	}

	if resumedToolMsg == nil {
		t.Fatal("the model never received a tool message")
	}
	var names []string
	for _, p := range resumedToolMsg.Content {
		if p.IsToolResponse() {
			names = append(names, p.ToolResponse.Name)
		}
	}
	if want := []string{"alpha", "beta"}; !slices.Equal(names, want) {
		t.Errorf("resumed tool message order = %v, want %v", names, want)
	}
}
