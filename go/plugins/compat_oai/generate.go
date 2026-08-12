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
	"maps"
	"reflect"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	pluginjsonschema "github.com/firebase/genkit/go/plugins/internal/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
)

// ModelGenerator handles OpenAI generation requests
type ModelGenerator struct {
	client    *openai.Client
	modelName string
	request   *openai.ChatCompletionNewParams
	messages  []openai.ChatCompletionMessageParamUnion
	tools     []openai.ChatCompletionToolParam
	// outputFormats is what the model declares it serves natively on the
	// wire; empty declares nothing and keeps every format eligible.
	outputFormats []string
	// Store any errors that occur during building
	err error
}

// GetRequest returns the request built so far, for tests and for plugins
// that need to inspect what will be sent.
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
	for _, msg := range messages {
		if msg == nil {
			continue
		}
		content := concatenateTextContent(msg.Content)
		switch msg.Role {
		case ai.RoleSystem:
			oaiMessages = append(oaiMessages, openai.SystemMessage(content))
		case ai.RoleModel:
			am := openai.ChatCompletionAssistantMessageParam{}
			am.Content.OfString = param.NewOpt(content)
			if reasoning := concatenateReasoningContent(msg.Content); reasoning != "" {
				am.SetExtraFields(map[string]any{
					"reasoning_content": reasoning,
				})
			}
			toolCalls, err := convertToolCalls(msg.Content)
			if err != nil {
				g.err = err
				return g
			}
			if len(toolCalls) > 0 {
				am.ToolCalls = (toolCalls)
			}
			oaiMessages = append(oaiMessages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &am,
			})
		case ai.RoleTool:
			for _, p := range msg.Content {
				if p == nil || !p.IsToolResponse() {
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
				if p == nil {
					continue
				}
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

// chatCompletionParamFields is the set of wire names the SDK's request params
// model, used by the deprecated [ModelGenerator.WithConfig] to tell a
// provider-specific key apart from one the SDK already carries.
var chatCompletionParamFields = func() map[string]struct{} {
	paramsType := reflect.TypeOf(openai.ChatCompletionNewParams{})
	fields := make(map[string]struct{}, paramsType.NumField())
	for i := range paramsType.NumField() {
		name, _, _ := strings.Cut(paramsType.Field(i).Tag.Get("json"), ",")
		if name != "" && name != "-" {
			fields[name] = struct{}{}
		}
	}
	return fields
}()

// WithConfig adds configuration parameters from the model request
// see https://platform.openai.com/docs/api-reference/responses/create
// for more details on openai's request fields
//
// Deprecated: use [ModelGenerator.WithParams], which takes the SDK request
// params directly. A plugin with a config type of its own converts it once, in
// its ApplyToChatCompletion, instead of leaving every request to a runtime
// type switch that silently drops the keys it does not recognize.
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
		if cfg == nil {
			return g
		}
		openaiConfig = *cfg
	case map[string]any:
		normalizedConfig := make(map[string]any, len(cfg))
		maps.Copy(normalizedConfig, cfg)
		for source, target := range map[string]string{
			"frequencyPenalty": "frequency_penalty",
			"logProbs":         "logprobs",
			"presencePenalty":  "presence_penalty",
			"stopSequences":    "stop",
			"topLogProbs":      "top_logprobs",
			"topP":             "top_p",
		} {
			if value, ok := normalizedConfig[source]; ok {
				normalizedConfig[target] = value
				delete(normalizedConfig, source)
			}
		}
		// These Genkit common fields are handled outside the provider request
		// or are not supported by the chat-completions adapter.
		for _, key := range []string{
			"apiKey", "maxOutputTokens", "topK", "version", "visualDetailLevel",
		} {
			delete(normalizedConfig, key)
		}

		var err error
		openaiConfig, err = base.MapToStruct[openai.ChatCompletionNewParams](normalizedConfig)
		if err != nil {
			g.err = fmt.Errorf("failed to convert config to openai.ChatCompletionNewParams: %w", err)
			return g
		}
		// Match the JS compat-oai adapter's config passthrough behavior. The
		// OpenAI SDK silently ignores provider-specific fields while
		// unmarshaling, so retain them as JSON extras. Fields managed by Genkit
		// are excluded to prevent config from overriding request construction.
		extraFields := make(map[string]any, len(normalizedConfig))
		for key, value := range normalizedConfig {
			if _, standard := chatCompletionParamFields[key]; standard {
				continue
			}
			extraFields[key] = value
		}
		openaiConfig.SetExtraFields(extraFields)
	default:
		g.err = fmt.Errorf("unexpected config type: %T", config)
		return g
	}

	// keep the original model in the updated config structure
	openaiConfig.Model = g.request.Model
	clearManagedFields(&openaiConfig)
	g.request = &openaiConfig
	return g
}

// WithParams uses params as the base the request is built on, carrying the
// request's config onto the wire. A model the params carry wins over the
// generator's: that is how a config pins the exact version the request is
// served by, matching the JS plugin's version handling. The generator's model
// fills in otherwise.
//
// The fields Genkit manages are cleared out of params and refilled from the
// Genkit request by the other builders (see [clearManagedFields]). Clearing
// them is what keeps a config from smuggling in a tool: the builders only
// assign when the Genkit request carries something, so a tool set here and
// nowhere else would otherwise survive onto the wire and the model could
// answer with a call the framework has no handler for.
func (g *ModelGenerator) WithParams(params openai.ChatCompletionNewParams) *ModelGenerator {
	if g.err != nil {
		return g
	}

	if params.Model == "" {
		params.Model = g.request.Model
	}
	clearManagedFields(&params)
	g.request = &params
	return g
}

// clearManagedFields zeroes the request fields Genkit owns, so that whatever
// the caller's config carried in them cannot reach the provider. Messages,
// tools, and the tool choice are rebuilt from the Genkit request; functions
// and function_call are the same surface under the names OpenAI used before
// tools, and have no Genkit counterpart to rebuild them from. The params'
// extra fields are swept for the same names, since the SDK marshals an extra
// field over the struct field it collides with, which would resurrect what
// the zeroing removed.
func clearManagedFields(params *openai.ChatCompletionNewParams) {
	params.Messages = nil
	params.Tools = nil
	params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{}
	params.Functions = nil
	params.FunctionCall = openai.ChatCompletionNewParamsFunctionCallUnion{}
	if extras := params.ExtraFields(); len(extras) > 0 {
		kept := maps.Clone(extras)
		for _, field := range managedRequestFields {
			delete(kept, field)
		}
		if len(kept) != len(extras) {
			params.SetExtraFields(kept)
		}
	}
}

// WithOutputFormats declares the output formats the model serves natively on
// the wire. When the declaration leaves "json" out, a schema-less JSON
// request sends no response_format and rides on the injected format
// instructions instead. Nil declares nothing and keeps every format eligible.
func (g *ModelGenerator) WithOutputFormats(formats []string) *ModelGenerator {
	if g.err != nil {
		return g
	}
	g.outputFormats = formats
	return g
}

// WithToolChoice adds Genkit's tool choice setting to the OpenAI-compatible
// request.
func (g *ModelGenerator) WithToolChoice(toolChoice ai.ToolChoice) *ModelGenerator {
	if g.err != nil || toolChoice == "" {
		return g
	}

	switch toolChoice {
	case ai.ToolChoiceAuto, ai.ToolChoiceNone, ai.ToolChoiceRequired:
		g.request.ToolChoice.OfAuto = param.NewOpt(string(toolChoice))
	default:
		g.err = fmt.Errorf("unsupported tool choice: %q", toolChoice)
	}

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

		// Strict mode is opt-in. When enabled, recursively set
		// additionalProperties: false on every object subschema; the caller
		// is responsible for OpenAI's other strict requirements (e.g. every
		// property must be listed in "required").
		strict := false
		if v, ok := tool.Metadata["strict"].(bool); ok {
			strict = v
		}
		var params openai.FunctionParameters
		if strict {
			params = openai.FunctionParameters(pluginjsonschema.EnforceStrict(tool.InputSchema))
		} else {
			params = openai.FunctionParameters(tool.InputSchema)
		}

		toolParams = append(toolParams, openai.ChatCompletionToolParam{
			Function: (shared.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: openai.String(tool.Description),
				Parameters:  params,
				Strict:      openai.Bool(strict),
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
		g.applyResponseFormat(req.Output)
	}

	if handleChunk != nil {
		return g.generateStream(ctx, req, handleChunk)
	}
	return g.generateComplete(ctx, req)
}

// applyResponseFormat sets the request's response format from the output
// configuration. A model that declares its native output formats and leaves
// "json" out has no schema-less JSON mode on the wire (Anthropic's compatible
// endpoint rejects the json_object type), so such a request sends no
// response_format and the format instructions the framework injects carry it
// instead.
func (g *ModelGenerator) applyResponseFormat(output *ai.ModelOutputConfig) {
	format := getResponseFormat(output)
	if format.OfJSONObject != nil && len(g.outputFormats) > 0 && !slices.Contains(g.outputFormats, "json") {
		format = openai.ChatCompletionNewParamsResponseFormatUnion{}
	}
	g.request.ResponseFormat = format
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
	}
	// The text format sends nothing: an explicit response_format of type
	// text only restates the default, and some compatible endpoints
	// (Anthropic's among them) reject the parameter outright.

	return format
}

func concatenateTextContent(parts []*ai.Part) string {
	var content strings.Builder
	for _, part := range parts {
		if part == nil {
			continue
		}
		if part.IsText() || part.IsData() {
			content.WriteString(part.Text)
		}
	}
	return content.String()
}

func concatenateReasoningContent(parts []*ai.Part) string {
	var reasoning strings.Builder
	for _, part := range parts {
		if part == nil {
			continue
		}
		if part.IsReasoning() {
			reasoning.WriteString(part.Text)
		}
	}
	return reasoning.String()
}

// generateStream generates a streaming model response
func (g *ModelGenerator) generateStream(ctx context.Context, req *ai.ModelRequest, handleChunk func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// A streamed request carries no token usage at all unless it asks for it,
	// so it is requested here rather than left to each plugin: a response
	// without usage would otherwise report zero tokens for every stream.
	g.request.StreamOptions.IncludeUsage = openai.Bool(true)

	stream := g.client.Chat.Completions.NewStreaming(ctx, *g.request)
	defer func() {
		_ = stream.Close()
	}()

	// Use openai-go's accumulator to collect the complete response
	acc := &openai.ChatCompletionAccumulator{}
	var reasoning strings.Builder
	// The accumulator adds up the three top-level token counts and drops
	// everything else the usage carries, including the reasoning and cache
	// breakdowns, so the usage chunk is kept whole and used in its place.
	var usage openai.CompletionUsage
	var usageSeen bool
	// The accumulator knows nothing of the citations xAI answers a live search
	// with either, so they are carried out of the chunk that has them.
	var citations any

	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)

		// Usage rides on a final chunk of its own and is null on the rest.
		if chunk.JSON.Usage.Valid() {
			usage = chunk.Usage
			usageSeen = true
		}
		if value := extractJSONValue(chunk.JSON.ExtraFields["citations"].Raw()); value != nil {
			citations = value
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		// Create chunk for callback
		modelChunk := &ai.ModelResponseChunk{}

		// Some OpenAI-compatible reasoning models, including Kimi, return
		// reasoning in the non-standard reasoning_content field.
		if reasoningDelta := extractReasoningContent(
			chunk.Choices[0].Delta.JSON.ExtraFields["reasoning_content"].Raw(),
		); reasoningDelta != "" {
			reasoning.WriteString(reasoningDelta)
			modelChunk.Content = append(modelChunk.Content, ai.NewReasoningPart(reasoningDelta, nil))
		}

		// Handle content delta
		if chunk.Choices[0].Delta.Content != "" {
			modelChunk.Content = append(modelChunk.Content, ai.NewTextPart(chunk.Choices[0].Delta.Content))
		}

		// Handle tool call deltas
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

	if usageSeen {
		acc.Usage = usage
	}

	// Convert accumulated ChatCompletion to ai.ModelResponse.
	resp, err := convertChatCompletionToModelResponse(&acc.ChatCompletion)
	if err != nil {
		return nil, err
	}
	if reasoning.Len() > 0 && resp.Reasoning() == "" {
		resp.Message.Content = append(
			[]*ai.Part{ai.NewReasoningPart(reasoning.String(), nil)},
			resp.Message.Content...,
		)
	}
	if citations != nil {
		custom, ok := resp.Custom.(map[string]any)
		if !ok {
			custom = map[string]any{}
		}
		custom["citations"] = citations
		resp.Custom = custom
		resp.Raw = custom
	}
	resp.Request = req
	return resp, nil
}

// extractTokenCount reads a token count a provider reports as a usage field the
// SDK does not model, returning 0 when it is absent or is not a number.
func extractTokenCount(raw string) int {
	if raw == "" || raw == "null" {
		return 0
	}
	var count int
	if err := json.Unmarshal([]byte(raw), &count); err != nil {
		return 0
	}
	return count
}

// extractJSONValue decodes a response field a provider returns that the SDK
// does not model, returning nil when it is absent or does not parse. The value
// is kept in whatever shape it arrived in, so a provider adding detail to it
// does not turn into a decode that silently drops the field.
func extractJSONValue(raw string) any {
	if raw == "" || raw == "null" {
		return nil
	}
	var value any
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return nil
	}
	return value
}

func extractReasoningContent(raw string) string {
	if raw == "" || raw == "null" {
		return ""
	}
	var reasoning string
	if err := json.Unmarshal([]byte(raw), &reasoning); err != nil {
		return ""
	}
	return reasoning
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

	// Add cached tokens if available. DeepSeek reports its cache hits as a
	// usage field of its own and returns no prompt_tokens_details at all, so
	// that field stands in when OpenAI's breakdown is absent.
	if completion.Usage.PromptTokensDetails.CachedTokens > 0 {
		usage.CachedContentTokens = int(completion.Usage.PromptTokensDetails.CachedTokens)
	} else if cached := extractTokenCount(
		completion.Usage.JSON.ExtraFields["prompt_cache_hit_tokens"].Raw(),
	); cached > 0 {
		usage.CachedContentTokens = cached
	}

	// Add the token counts Genkit has no field of its own for.
	addCustomTokens(usage, "audioTokens", int(completion.Usage.CompletionTokensDetails.AudioTokens))
	addCustomTokens(usage, "acceptedPredictionTokens", int(completion.Usage.CompletionTokensDetails.AcceptedPredictionTokens))
	addCustomTokens(usage, "rejectedPredictionTokens", int(completion.Usage.CompletionTokensDetails.RejectedPredictionTokens))
	// xAI counts the live-search sources it consulted and breaks image tokens
	// out of the prompt, neither of which is in OpenAI's usage shape.
	addCustomTokens(usage, "numSourcesUsed", extractTokenCount(
		completion.Usage.JSON.ExtraFields["num_sources_used"].Raw()))
	addCustomTokens(usage, "imageTokens", extractTokenCount(
		completion.Usage.PromptTokensDetails.JSON.ExtraFields["image_tokens"].Raw()))

	resp := &ai.ModelResponse{
		Usage: usage,
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: make([]*ai.Part, 0),
		},
	}

	// Map finish reason. end_turn is xAI's name for an answer the model chose
	// to end, one of the three reasons it documents, so leaving it out reports
	// an ordinary completion as unknown.
	switch choice.FinishReason {
	case "stop", "tool_calls", "end_turn":
		resp.FinishReason = ai.FinishReasonStop
	case "length", "model_context_window_exceeded":
		resp.FinishReason = ai.FinishReasonLength
	case "content_filter", "sensitive":
		resp.FinishReason = ai.FinishReasonBlocked
	case "function_call", "network_error", "insufficient_system_resource":
		resp.FinishReason = ai.FinishReasonOther
	default:
		resp.FinishReason = ai.FinishReasonUnknown
	}

	// Set finish message if there's a refusal
	if choice.Message.Refusal != "" {
		resp.FinishMessage = choice.Message.Refusal
		resp.FinishReason = ai.FinishReasonBlocked
	}

	// Keep parity with the JS compat-oai plugin by surfacing the non-standard
	// reasoning_content field as a Genkit reasoning part.
	if reasoning := extractReasoningContent(
		choice.Message.JSON.ExtraFields["reasoning_content"].Raw(),
	); reasoning != "" {
		resp.Message.Content = append(resp.Message.Content, ai.NewReasoningPart(reasoning, nil))
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

	// Collect the response metadata that is not part of the generated message.
	custom := map[string]any{}
	if completion.SystemFingerprint != "" {
		custom["systemFingerprint"] = completion.SystemFingerprint
		custom["model"] = completion.Model
		custom["id"] = completion.ID
	}
	// xAI answers a live search with the sources behind it, in a citations
	// field the SDK does not model. It is the only way a caller that asked for
	// citations gets them, so it rides on the response's custom metadata. The
	// entries pass through as xAI returns them, like the search sources the
	// request carries.
	if citations := extractJSONValue(completion.JSON.ExtraFields["citations"].Raw()); citations != nil {
		custom["citations"] = citations
	}
	// Raw carries the same metadata as Custom, which is deprecated in favor of
	// it: new fields are documented against Raw, and the readers of the older
	// one keep working.
	if len(custom) > 0 {
		resp.Custom = custom
		resp.Raw = custom
	}

	return resp, nil
}

// addCustomTokens records a token count Genkit has no [ai.GenerationUsage]
// field of its own for, allocating the map on first use. A count of zero is
// dropped: every caller reads it from a usage field that is absent, and
// reporting a zero would not tell that apart from a genuine zero.
func addCustomTokens(usage *ai.GenerationUsage, name string, count int) {
	if count <= 0 {
		return
	}
	if usage.Custom == nil {
		usage.Custom = make(map[string]float64)
	}
	usage.Custom[name] = float64(count)
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
		if p == nil || !p.IsToolRequest() {
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
