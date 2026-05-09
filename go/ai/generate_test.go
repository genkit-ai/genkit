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
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core"
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

	echoModel = DefineModel(r, "test/"+modelName, &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
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
var gablorkenTool = DefineTool(r, "gablorken", "use when need to calculate a gablorken",
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

	ctx := context.Background()

	convertTempTool := DefineTool(r, "convertTemp", "converts temperature",
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

	toolModel := DefineModel(r, "test/toolModel", &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
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
	JSON := "{\"subject\": \"bananas\", \"location\": \"tropics\"}"
	JSONmd := "```json" + JSON + "```"

	bananaModel := DefineModel(r, "test/banana", &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
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
		interruptTool := DefineTool(r, "interruptor", "always interrupts",
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
		interruptModel := DefineModel(r, "test/interrupt", info,
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
		parallelModel := DefineModel(r, "test/parallel", info,
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
		multiRoundModel := DefineModel(r, "test/multiround", info,
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
		infiniteModel := DefineModel(r, "test/infinite", info,
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
		toolCallModel := DefineModel(r, "test/toolcall", info,
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
	model := DefineModel(r, "test/constrained", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		// Mock response
		return &ModelResponse{
			Message: NewModelTextMessage(`{"foo": "bar"}`),
			Request: req,
		}, nil
	})

	core.DefineSchema(r, "FooSchema", map[string]any{
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
	conditionalTool := DefineTool(r, "conditional", "tool that may interrupt based on input",
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

	resumableTool := DefineTool(r, "resumable", "tool that can be resumed",
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

	toolModel := DefineModel(r, "test/toolmodel", info,
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
	DefineResource(r, "test-file", &ResourceOptions{
		URI:         "file:///test.txt",
		Description: "Test file resource",
	}, func(ctx context.Context, input *ResourceInput) (*ResourceOutput, error) {
		return &ResourceOutput{Content: []*Part{NewTextPart("FILE CONTENT")}}, nil
	})

	DefineResource(r, "test-api", &ResourceOptions{
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

		DefineMultipartTool(r, "multipartTest", "a multipart tool",
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

		DefineTool(r, "regularTestTool", "a regular tool",
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

	t.Run("multipart tool returns content in response", func(t *testing.T) {
		r := registry.New()
		ConfigureFormats(r)
		DefineGenerateAction(context.Background(), r)

		multipartTool := DefineMultipartTool(r, "imageGenerator", "generates images",
			func(ctx *ToolContext, input struct{ Prompt string }) (*MultipartToolResponse, error) {
				return &MultipartToolResponse{
					Output: map[string]any{"description": "generated image"},
					Content: []*Part{
						NewMediaPart("image/png", "data:image/png;base64,iVBORw0..."),
					},
				}, nil
			},
		)

		// Create a model that requests the tool
		multipartToolModel := DefineModel(r, "test/multipartToolModel", &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
			// Check if we already have a tool response
			for _, msg := range gr.Messages {
				if msg.Role == RoleTool {
					for _, part := range msg.Content {
						if part.IsToolResponse() {
							// Verify the content is present
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

		tool := DefineTool(r, "multipartWrapperTest", "test multipart wrapper",
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

		tool := DefineMultipartTool(r, "multipartFullTest", "test multipart",
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

		streamModel := DefineModel(r, "test/streamModel", &ModelOptions{
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
		noStreamModel := DefineModel(r, "test/noStreamModel", &ModelOptions{
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

		errorModel := DefineModel(r, "test/errorModel", &ModelOptions{
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

		streamModel := DefineModel(r, "test/cancelModel", &ModelOptions{
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
}

func TestGenerateDataStream(t *testing.T) {
	r := registry.New()
	ConfigureFormats(r)
	DefineGenerateAction(context.Background(), r)

	t.Run("yields typed chunks and final output", func(t *testing.T) {
		streamModel := DefineModel(r, "test/typedStreamModel", &ModelOptions{
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
		streamModel := DefineModel(r, "test/finalTypedModel", &ModelOptions{
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

		streamModel := DefineModel(r, "test/autoOutputModel", &ModelOptions{
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
		interruptTool := DefineTool(r, "streamInterruptor", "always interrupts",
			func(ctx *ToolContext, input any) (any, error) {
				return nil, ctx.Interrupt(&InterruptOptions{
					Metadata: map[string]any{
						"reason": "needs confirmation",
					},
				})
			},
		)

		streamModel := DefineModel(r, "test/streamInterruptModel", &ModelOptions{
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
		greetTool := DefineTool(r, "streamGreeter", "greets",
			func(ctx *ToolContext, input any) (any, error) {
				return "hello", nil
			},
		)

		streamModel := DefineModel(r, "test/streamReturnToolModel", &ModelOptions{
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
		streamModel := DefineModel(r, "test/parseErrorModel", &ModelOptions{
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
}

func TestGenerateText(t *testing.T) {
	r := newTestRegistry(t)

	echoModel := DefineModel(r, "test/echoTextModel", nil, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
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

	jsonModel := DefineModel(r, "test/jsonDataModel", &ModelOptions{
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
	markdownModel := DefineModel(r, "test/markdownJson", &ModelOptions{
		Supports: &ModelSupports{Constrained: ConstrainedSupportAll},
	}, func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
		jsonContent := "{\"name\": \"test\", \"value\": 123}"
		return &ModelResponse{
			Request: req,
			Message: NewModelTextMessage("```json\n" + jsonContent + "\n```"),
		}, nil
	})

	// A model that returns JSON wrapped in markdown with loose formatting (spaces)
	looseMarkdownModel := DefineModel(r, "test/looseMarkdownJson", &ModelOptions{
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
