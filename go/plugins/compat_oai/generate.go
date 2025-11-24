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

package compat_oai

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/adler32"
	"strconv"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"

	"github.com/firebase/genkit/go/ai"
)

// mapToStruct unmarshals a map[string]any to the expected config api.
func mapToStruct(m map[string]any, v any) error {
	jsonData, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return json.Unmarshal(jsonData, v)
}

// ModelGenerator handles OpenAI generation requests
type ModelGenerator struct {
	client       *openai.Client
	modelName    string
	request      *openai.ChatCompletionNewParams
	messages     []openai.ChatCompletionMessageParamUnion
	tools        []openai.ChatCompletionToolParam
	toolChoice   openai.ChatCompletionToolChoiceOptionUnionParam
	outputFormat openai.ChatCompletionNewParamsResponseFormatUnion
	// Store any errors that occur during building
	err error
}

func (g *ModelGenerator) GetRequest() *openai.ChatCompletionNewParams {
	return g.request
}

// NewModelGenerator creates a new ModelGenerator instance
func NewModelGenerator(client *openai.Client, modelName string) *ModelGenerator {
	return &ModelGenerator{
		client:    client,
		modelName: modelName,
		request: &openai.ChatCompletionNewParams{
			Model: modelName,
		},
	}
}

// WithMessages adds messages to the request
func (g *ModelGenerator) WithMessages(messages []*ai.Message) *ModelGenerator {
	// Return early if we already have an error
	if g.err != nil {
		return g
	}

	if messages == nil {
		return g
	}

	oaiMessages := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		switch msg.Role {
		case ai.RoleSystem:
			oaiMessages = append(oaiMessages, openai.SystemMessage(concatenateTextParts(msg.Content)))
		case ai.RoleModel:
			am := openai.ChatCompletionAssistantMessageParam{}
			am.Content.OfString = param.NewOpt(concatenateTextParts(msg.Content))
			toolCalls, err := convertToolCalls(msg.Content)
			if err != nil {
				g.err = err
				return g
			}
			if len(toolCalls) > 0 {
				am.ToolCalls = toolCalls
			}
			oaiMessages = append(oaiMessages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &am,
			})
		case ai.RoleTool:
			for _, p := range msg.Content {
				if !p.IsToolResponse() {
					continue
				}
				// Use the captured tool call ID (Ref) if available, otherwise fall back to tool name
				toolCallID := p.ToolResponse.Ref
				if toolCallID == "" {
					toolCallID = p.ToolResponse.Name
				}

				toolOutput, err := anyToJSONString(p.ToolResponse.Output)
				if err != nil {
					g.err = err
					return g
				}
				tm := openai.ToolMessage(toolOutput, toolCallID)
				oaiMessages = append(oaiMessages, tm)
			}
		case ai.RoleUser:
			parts := []openai.ChatCompletionContentPartUnionParam{}
			for _, p := range msg.Content {
				if p.IsText() {
					parts = append(parts, openai.TextContentPart(p.Text))
				}
				if p.IsMedia() {
					part := openai.ImageContentPart(
						openai.ChatCompletionContentPartImageImageURLParam{
							URL: p.Text,
						})
					parts = append(parts, part)
				}
			}

			if len(parts) > 0 {
				oaiMessages = append(oaiMessages, openai.UserMessage(parts))
			}
		default:
			// ignore parts from not supported roles
			continue
		}

	}
	g.messages = oaiMessages
	return g
}

// WithConfig adds configuration parameters from the model request
// see https://platform.openai.com/docs/api-reference/responses/create
// for more details on openai's request fields
func (g *ModelGenerator) WithConfig(config any) *ModelGenerator {
	// Return early if we already have an error
	if g.err != nil {
		return g
	}

	if config == nil {
		return g
	}

	var openaiConfig openai.ChatCompletionNewParams
	switch cfg := config.(type) {
	case openai.ChatCompletionNewParams:
		openaiConfig = cfg
	case *openai.ChatCompletionNewParams:
		openaiConfig = *cfg
	case map[string]any:
		if err := mapToStruct(cfg, &openaiConfig); err != nil {
			g.err = fmt.Errorf("failed to convert config to openai.ChatCompletionNewParams: %w", err)
			return g
		}
	default:
		g.err = fmt.Errorf("unexpected config type: %T", config)
		return g
	}

	// keep the original model in the updated config structure
	openaiConfig.Model = g.request.Model
	g.request = &openaiConfig
	return g
}

// WithTools adds tools to the request
func (g *ModelGenerator) WithTools(tools []*ai.ToolDefinition) *ModelGenerator {
	if g.err != nil {
		return g
	}

	if tools == nil {
		return g
	}

	toolParams := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, tool := range tools {
		if tool == nil || tool.Name == "" {
			continue
		}

		toolParams = append(toolParams, openai.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters:  tool.InputSchema,
				Strict:      openai.Bool(false), // TODO: implement strict mode
			},
		})
	}

	// Set the tools in the request
	// If no tools are provided, set it to nil
	// This is important to avoid sending an empty array in the request
	// which is not supported by some vendor APIs
	if len(toolParams) > 0 {
		g.tools = toolParams
	}

	return g
}

func schemaHash(schema map[string]any) string {
	if schema == nil {
		return ""
	}

	jsonBytes, err := json.Marshal(schema)
	if err != nil {
		return "" // should never happen
	}

	return strconv.FormatInt(int64(adler32.Checksum(jsonBytes)), 10) // adler32 entropy should be sufficient
}

func (g *ModelGenerator) WithOutput(output *ai.ModelOutputConfig) *ModelGenerator {
	if g.err != nil {
		return g
	}

	if output == nil {
		return g
	}

	if !output.Constrained || output.Format != "json" && output.Schema == nil {
		return g
	}

	g.outputFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   schemaHash(output.Schema),
				Strict: openai.Bool(true),
				Schema: output.Schema,
			},
		},
	}

	return g
}

// Generate executes the generation request
func (g *ModelGenerator) Generate(ctx context.Context, req *ai.ModelRequest, handleChunk func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Check for any errors that occurred during building
	if g.err != nil {
		return nil, g.err
	}

	if len(g.messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}
	g.request.Messages = g.messages

	if len(g.tools) > 0 {
		g.request.Tools = g.tools
	}

	g.request.ResponseFormat = g.outputFormat

	if handleChunk != nil {
		return g.generateStream(ctx, handleChunk)
	}
	return g.generateComplete(ctx, req)
}

// generateStream generates a streaming model response
func (g *ModelGenerator) generateStream(ctx context.Context, handleChunk func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	stream := g.client.Chat.Completions.NewStreaming(ctx, *g.request)
	defer stream.Close()

	var fullResponse ai.ModelResponse
	fullResponse.Message = &ai.Message{
		Role:    ai.RoleModel,
		Content: make([]*ai.Part, 0),
	}

	// Initialize request and usage
	fullResponse.Request = &ai.ModelRequest{}
	fullResponse.Usage = &ai.GenerationUsage{
		InputTokens:  0,
		OutputTokens: 0,
		TotalTokens:  0,
	}

	var currentToolCall *ai.ToolRequest
	var currentArguments string
	var toolCallCollects []struct {
		toolCall *ai.ToolRequest
		args     string
	}

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			modelChunk := &ai.ModelResponseChunk{}

			switch choice.FinishReason {
			case "tool_calls", "stop":
				fullResponse.FinishReason = ai.FinishReasonStop
			case "length":
				fullResponse.FinishReason = ai.FinishReasonLength
			case "content_filter":
				fullResponse.FinishReason = ai.FinishReasonBlocked
			case "function_call":
				fullResponse.FinishReason = ai.FinishReasonOther
			default:
				fullResponse.FinishReason = ai.FinishReasonUnknown
			}

			// handle tool calls
			for _, toolCall := range choice.Delta.ToolCalls {
				// first tool call (= current tool call is nil) contains the tool call name
				if currentToolCall != nil && toolCall.ID != "" && currentToolCall.Ref != toolCall.ID {
					toolCallCollects = append(toolCallCollects, struct {
						toolCall *ai.ToolRequest
						args     string
					}{
						toolCall: currentToolCall,
						args:     currentArguments,
					})
					currentToolCall = nil
					currentArguments = ""
				}

				if currentToolCall == nil {
					currentToolCall = &ai.ToolRequest{
						Name: toolCall.Function.Name,
						Ref:  toolCall.ID,
					}
				}

				if toolCall.Function.Arguments != "" {
					currentArguments += toolCall.Function.Arguments
				}

				modelChunk.Content = append(modelChunk.Content, ai.NewToolRequestPart(&ai.ToolRequest{
					Name:  currentToolCall.Name,
					Input: toolCall.Function.Arguments,
					Ref:   currentToolCall.Ref,
				}))
			}

			// when tool call is complete
			if choice.FinishReason == "tool_calls" && currentToolCall != nil {
				// parse accumulated arguments string
				for _, toolcall := range toolCallCollects {
					args, err := jsonStringToMap(toolcall.args)
					if err != nil {
						return nil, fmt.Errorf("could not parse tool args: %w", err)
					}
					toolcall.toolCall.Input = args
					fullResponse.Message.Content = append(fullResponse.Message.Content, ai.NewToolRequestPart(toolcall.toolCall))
				}
				if currentArguments != "" {
					args, err := jsonStringToMap(currentArguments)
					if err != nil {
						return nil, fmt.Errorf("could not parse tool args: %w", err)
					}
					currentToolCall.Input = args
				}
				fullResponse.Message.Content = append(fullResponse.Message.Content, ai.NewToolRequestPart(currentToolCall))
			}

			content := chunk.Choices[0].Delta.Content
			// when starting a tool call, the content is empty
			if content != "" {
				modelChunk.Content = append(modelChunk.Content, ai.NewTextPart(content))
				fullResponse.Message.Content = append(fullResponse.Message.Content, modelChunk.Content...)
			}

			if err := handleChunk(ctx, modelChunk); err != nil {
				return nil, fmt.Errorf("callback error: %w", err)
			}

			fullResponse.Usage.InputTokens += int(chunk.Usage.PromptTokens)
			fullResponse.Usage.OutputTokens += int(chunk.Usage.CompletionTokens)
			fullResponse.Usage.TotalTokens += int(chunk.Usage.TotalTokens)
		}
	}

	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}

	return &fullResponse, nil
}

// generateComplete generates a complete model response
func (g *ModelGenerator) generateComplete(ctx context.Context, req *ai.ModelRequest) (*ai.ModelResponse, error) {
	completion, err := g.client.Chat.Completions.New(ctx, *g.request)
	if err != nil {
		return nil, fmt.Errorf("failed to create completion: %w", err)
	}

	resp := &ai.ModelResponse{
		Request: req,
		Usage: &ai.GenerationUsage{
			InputTokens:  int(completion.Usage.PromptTokens),
			OutputTokens: int(completion.Usage.CompletionTokens),
			TotalTokens:  int(completion.Usage.TotalTokens),
		},
		Message: &ai.Message{
			Role: ai.RoleModel,
		},
	}

	choice := completion.Choices[0]

	switch choice.FinishReason {
	case "stop", "tool_calls":
		resp.FinishReason = ai.FinishReasonStop
	case "length":
		resp.FinishReason = ai.FinishReasonLength
	case "content_filter":
		resp.FinishReason = ai.FinishReasonBlocked
	case "function_call":
		resp.FinishReason = ai.FinishReasonOther
	default:
		resp.FinishReason = ai.FinishReasonUnknown
	}

	// handle tool calls
	var toolRequestParts []*ai.Part
	for _, toolCall := range choice.Message.ToolCalls {
		args, err := jsonStringToMap(toolCall.Function.Arguments)
		if err != nil {
			return nil, err
		}
		toolRequestParts = append(toolRequestParts, ai.NewToolRequestPart(&ai.ToolRequest{
			Ref:   toolCall.ID,
			Name:  toolCall.Function.Name,
			Input: args,
		}))
	}

	// content and tool call may exist simultaneously
	if completion.Choices[0].Message.Content != "" {
		resp.Message.Content = append(resp.Message.Content, ai.NewTextPart(completion.Choices[0].Message.Content))
	}

	if len(toolRequestParts) > 0 {
		resp.Message.Content = append(resp.Message.Content, toolRequestParts...)
		return resp, nil
	}

	return resp, nil
}

// concatenateTextParts concatenates text parts into a single string
func concatenateTextParts(parts []*ai.Part) string {
	text := ""
	for _, part := range parts {
		if part.Kind != ai.PartText {
			continue
		}

		text += part.Text
	}
	return text
}

func convertToolCalls(content []*ai.Part) ([]openai.ChatCompletionMessageToolCallParam, error) {
	var toolCalls []openai.ChatCompletionMessageToolCallParam
	for _, p := range content {
		if !p.IsToolRequest() {
			continue
		}
		toolCall, err := convertToolCall(p)
		if err != nil {
			return nil, err
		}
		toolCalls = append(toolCalls, *toolCall)
	}
	return toolCalls, nil
}

func convertToolCall(part *ai.Part) (*openai.ChatCompletionMessageToolCallParam, error) {
	toolCallID := part.ToolRequest.Ref
	if toolCallID == "" {
		toolCallID = part.ToolRequest.Name
	}

	param := &openai.ChatCompletionMessageToolCallParam{
		ID: toolCallID,
		Function: openai.ChatCompletionMessageToolCallFunctionParam{
			Name: part.ToolRequest.Name,
		},
	}

	args, err := anyToJSONString(part.ToolRequest.Input)
	if err != nil {
		return nil, err
	}
	if part.ToolRequest.Input != nil {
		param.Function.Arguments = args
	}

	return param, nil
}

func jsonStringToMap(jsonString string) (map[string]any, error) {
	var result map[string]any
	if err := json.Unmarshal([]byte(jsonString), &result); err != nil {
		return nil, fmt.Errorf("unmarshal failed to parse json string %s: %w", jsonString, err)
	}
	return result, nil
}

func anyToJSONString(data any) (string, error) {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to marshal any to JSON string: data, %#v %w", data, err)
	}
	return string(jsonBytes), nil
}
