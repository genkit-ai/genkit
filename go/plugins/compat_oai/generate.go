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
	"log/slog"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/packages/respjson"
	"github.com/openai/openai-go/shared"
)

// ModelGenerator handles OpenAI generation requests
type ModelGenerator struct {
	client     *openai.Client
	modelName  string
	request    *openai.ChatCompletionNewParams
	messages   []openai.ChatCompletionMessageParamUnion
	tools      []openai.ChatCompletionToolParam
	toolChoice openai.ChatCompletionToolChoiceOptionUnionParam
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
			Model: (modelName),
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

	filterAll := func(p *ai.Part) bool { return true }
	filterReasoning := func(p *ai.Part) bool { return p.IsReasoning() }
	filterExceptReasoning := func(p *ai.Part) bool { return !p.IsReasoning() }

	for _, msg := range messages {
		switch msg.Role {
		case ai.RoleSystem:
			content := g.concatenateContent(msg.Content, filterAll)
			oaiMessages = append(oaiMessages, openai.SystemMessage(content))
		case ai.RoleModel:
			am := openai.ChatCompletionAssistantMessageParam{}
			toolCalls, err := convertToolCalls(msg.Content)
			if err != nil {
				g.err = err
				return g
			}
			if len(toolCalls) > 0 {
				am.ToolCalls = (toolCalls)
			}

			// Store reasoning in extra fields if present
			reasoningKey, ok := g.firstReasoningMetadataKey(msg.Content)
			reasoning := g.concatenateContent(msg.Content, filterReasoning)
			var content string
			if ok && reasoning != "" {
				am.SetExtraFields(map[string]any{
					reasoningKey: reasoning,
				})
				content = g.concatenateContent(msg.Content, filterExceptReasoning)
			} else {
				content = g.concatenateContent(msg.Content, filterAll)
			}

			am.Content.OfString = param.NewOpt(content)
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
					continue
				}
			}
			if len(parts) > 0 {
				oaiMessages = append(oaiMessages, openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: openai.ChatCompletionUserMessageParamContentUnion{OfArrayOfContentParts: parts},
					},
				})
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
		var err error
		openaiConfig, err = base.MapToStruct[openai.ChatCompletionNewParams](cfg)
		if err != nil {
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
			Function: (shared.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters:  openai.FunctionParameters(tool.InputSchema),
				Strict:      openai.Bool(false), // TODO: implement strict mode
			}),
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

	if req.Output != nil {
		g.request.ResponseFormat = getResponseFormat(req.Output)
	}

	if handleChunk != nil {
		return g.generateStream(ctx, handleChunk)
	}
	return g.generateComplete(ctx, req)
}

// getResponseFormat determines the appropriate response format based on the output configuration
func getResponseFormat(output *ai.ModelOutputConfig) openai.ChatCompletionNewParamsResponseFormatUnion {
	var format openai.ChatCompletionNewParamsResponseFormatUnion

	if output == nil {
		return format
	}

	switch output.Format {
	case "json":
		if output.Schema != nil {
			jsonSchemaParam := shared.ResponseFormatJSONSchemaParam{
				JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "output",
					Schema: output.Schema,
					Strict: openai.Bool(true),
				},
			}
			format.OfJSONSchema = &jsonSchemaParam
		} else {
			jsonObjectParam := shared.NewResponseFormatJSONObjectParam()
			format.OfJSONObject = &jsonObjectParam
		}
	case "text":
		textParam := shared.NewResponseFormatTextParam()
		format.OfText = &textParam
	}

	return format
}

// concatenateContent concatenates text content into a single string.
// Only parts matching the filter are included.
func (g *ModelGenerator) concatenateContent(parts []*ai.Part, filter func(*ai.Part) bool) string {
	var sb strings.Builder
	for _, part := range parts {
		if filter(part) {
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}

// firstReasoningMetadataKey returns the key which is used for the reasoning values
func (g *ModelGenerator) firstReasoningMetadataKey(parts []*ai.Part) (string, bool) {
	for _, part := range parts {
		if !part.IsReasoning() || part.Metadata == nil {
			continue
		}
		if key, ok := part.Metadata["reasoningKey"].(string); ok && key != "" {
			return key, true
		}
	}
	return "", false
}

// generateStream generates a streaming model response
func (g *ModelGenerator) generateStream(ctx context.Context, handleChunk func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	stream := g.client.Chat.Completions.NewStreaming(ctx, *g.request)
	defer stream.Close()

	collector := streamResponseCollector{}

	for stream.Next() {
		chunk := stream.Current()
		modelChunk, ok := collector.AddChunk(chunk)
		if !ok {
			continue
		}

		// Call the chunk handler with incremental data
		if len(modelChunk.Content) > 0 {
			if err := handleChunk(ctx, modelChunk); err != nil {
				return nil, fmt.Errorf("callback error: %w", err)
			}
		}
	}

	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}

	return collector.ToModelResponse()
}

// convertChatCompletionToModelResponse converts openai.ChatCompletion to ai.ModelResponse
func convertChatCompletionToModelResponse(completion *openai.ChatCompletion) (*ai.ModelResponse, error) {
	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("no choices in completion")
	}

	choice := completion.Choices[0]

	// Build usage information with detailed token breakdown
	usage := &ai.GenerationUsage{
		InputTokens:  int(completion.Usage.PromptTokens),
		OutputTokens: int(completion.Usage.CompletionTokens),
		TotalTokens:  int(completion.Usage.TotalTokens),
	}

	// Add reasoning tokens (thoughts tokens) if available
	if completion.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
		usage.ThoughtsTokens = int(completion.Usage.CompletionTokensDetails.ReasoningTokens)
	}

	// Add cached tokens if available
	if completion.Usage.PromptTokensDetails.CachedTokens > 0 {
		usage.CachedContentTokens = int(completion.Usage.PromptTokensDetails.CachedTokens)
	}

	// Add audio tokens to custom field if available
	if completion.Usage.CompletionTokensDetails.AudioTokens > 0 {
		if usage.Custom == nil {
			usage.Custom = make(map[string]float64)
		}
		usage.Custom["audioTokens"] = float64(completion.Usage.CompletionTokensDetails.AudioTokens)
	}

	// Add prediction tokens to custom field if available
	if completion.Usage.CompletionTokensDetails.AcceptedPredictionTokens > 0 {
		if usage.Custom == nil {
			usage.Custom = make(map[string]float64)
		}
		usage.Custom["acceptedPredictionTokens"] = float64(completion.Usage.CompletionTokensDetails.AcceptedPredictionTokens)
	}
	if completion.Usage.CompletionTokensDetails.RejectedPredictionTokens > 0 {
		if usage.Custom == nil {
			usage.Custom = make(map[string]float64)
		}
		usage.Custom["rejectedPredictionTokens"] = float64(completion.Usage.CompletionTokensDetails.RejectedPredictionTokens)
	}

	resp := &ai.ModelResponse{
		Request: &ai.ModelRequest{},
		Usage:   usage,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: make([]*ai.Part, 0),
		},
	}

	// Map finish reason
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

	// Set finish message if there's a refusal
	if choice.Message.Refusal != "" {
		resp.FinishMessage = choice.Message.Refusal
		resp.FinishReason = ai.FinishReasonBlocked
	}

	// Add text content
	if choice.Message.Content != "" {
		resp.Message.Content = append(resp.Message.Content, ai.NewTextPart(choice.Message.Content))
	}

	// Add tool calls
	for _, toolCall := range choice.Message.ToolCalls {
		args, err := jsonStringToMap(toolCall.Function.Arguments)
		if err != nil {
			return nil, fmt.Errorf("could not parse tool args: %w", err)
		}
		resp.Message.Content = append(resp.Message.Content, ai.NewToolRequestPart(&ai.ToolRequest{
			Ref:   toolCall.ID,
			Name:  toolCall.Function.Name,
			Input: args,
		}))
	}

	// Add reasoning content if present (for Kimi/Moonshot compatibility)
	// Check ExtraFields for reasoning fields when tool calls are present
	reasoning, ok := firstReasoningField(choice.Message.JSON.ExtraFields)
	if ok {
		resp.Message.Content = append([]*ai.Part{reasoning.ToPart()}, resp.Message.Content...)
	}

	// Store additional metadata in custom field if needed
	if completion.SystemFingerprint != "" {
		resp.Custom = map[string]any{
			"systemFingerprint": completion.SystemFingerprint,
			"model":             completion.Model,
			"id":                completion.ID,
		}
	}

	return resp, nil
}

// generateComplete generates a complete model response
func (g *ModelGenerator) generateComplete(ctx context.Context, req *ai.ModelRequest) (*ai.ModelResponse, error) {
	completion, err := g.client.Chat.Completions.New(ctx, *g.request)
	if err != nil {
		return nil, fmt.Errorf("failed to create completion: %w", err)
	}

	resp, err := convertChatCompletionToModelResponse(completion)
	if err != nil {
		return nil, err
	}

	// Set the original request
	resp.Request = req

	return resp, nil
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
		ID: (toolCallID),
		Function: (openai.ChatCompletionMessageToolCallFunctionParam{
			Name: (part.ToolRequest.Name),
		}),
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

var reasoningFieldNames = []string{
	"reasoning_content",
	"reasoning",
	"reasoning_text",
}

type reasoningField struct {
	key   string
	value string
}

// firstReasoningField returns the first non-empty reasoning field from an
// OpenAI-compatible response using the compatibility order in reasoningFieldNames.
func firstReasoningField(fields map[string]respjson.Field) (reasoningField, bool) {
	for _, name := range reasoningFieldNames {
		raw := fields[name].Raw()
		if raw == "" || raw == "null" {
			continue
		}

		var val string
		if err := json.Unmarshal([]byte(raw), &val); err != nil {
			slog.Debug("could not unmarshal reasoning field", "err", err)
			return reasoningField{}, false
		}
		if val != "" {
			return reasoningField{key: name, value: val}, true
		}
	}
	return reasoningField{}, false
}

// ToPart converts the reasoning field to an ai.Part with appropriate metadata.
func (rf reasoningField) ToPart() *ai.Part {
	part := ai.NewReasoningPart(rf.value, nil)
	if rf.key == "" {
		return part
	}
	if part.Metadata == nil {
		part.Metadata = make(map[string]any)
	}
	part.Metadata["reasoningKey"] = rf.key
	return part
}

type streamResponseCollector struct {
	accumulator openai.ChatCompletionAccumulator
	reasoning   reasoningAccumulator
}

func (c *streamResponseCollector) AddChunk(chunk openai.ChatCompletionChunk) (*ai.ModelResponseChunk, bool) {
	c.accumulator.AddChunk(chunk)

	if len(chunk.Choices) == 0 {
		return nil, false
	}

	// Create chunk for callback
	modelChunk := &ai.ModelResponseChunk{}

	if chunk.Choices[0].Delta.Content != "" {
		modelChunk.Content = append(modelChunk.Content, ai.NewTextPart(chunk.Choices[0].Delta.Content))
	}

	for _, toolCall := range chunk.Choices[0].Delta.ToolCalls {
		// Send the incremental tool call part in the chunk
		if toolCall.Function.Name != "" || toolCall.Function.Arguments != "" {
			modelChunk.Content = append(modelChunk.Content, ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  toolCall.Function.Name,
				Input: toolCall.Function.Arguments,
				Ref:   toolCall.ID,
			}))
		}
	}

	reasoningPart, ok := c.reasoning.Add(chunk.Choices[0].Delta.JSON.ExtraFields)
	if ok {
		modelChunk.Content = append(modelChunk.Content, reasoningPart)
	}

	return modelChunk, true
}

func (c *streamResponseCollector) ToModelResponse() (*ai.ModelResponse, error) {
	c.reasoning.MergeInto(&c.accumulator.ChatCompletion)
	return convertChatCompletionToModelResponse(&c.accumulator.ChatCompletion)
}

type reasoningAccumulator struct {
	// key should be present if content not empty
	key     string
	content strings.Builder
}

// Add accumulates reasoning content if present and returns added part
func (ra *reasoningAccumulator) Add(fields map[string]respjson.Field) (*ai.Part, bool) {
	reasoning, ok := firstReasoningField(fields)
	if !ok {
		return nil, false
	}
	ra.content.WriteString(reasoning.value)
	ra.key = reasoning.key
	return reasoning.ToPart(), true
}

// MergeInto adds accumulated reasoning to the givem completion
func (ra *reasoningAccumulator) MergeInto(completion *openai.ChatCompletion) {
	if completion == nil || len(completion.Choices) == 0 || ra.content.Len() == 0 {
		return
	}

	fields := completion.Choices[0].Message.JSON.ExtraFields
	if fields == nil {
		fields = make(map[string]respjson.Field)
		completion.Choices[0].Message.JSON.ExtraFields = fields
	}

	raw, err := json.Marshal(ra.content.String())
	if err != nil {
		slog.Debug("could not marshal reasoning", "err", err)
		return
	}
	fields[ra.key] = respjson.NewField(string(raw))
}
