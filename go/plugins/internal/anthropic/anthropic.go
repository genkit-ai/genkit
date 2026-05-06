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

package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"regexp"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"github.com/invopop/jsonschema"

	"github.com/anthropics/anthropic-sdk-go"
)

const (
	ToolNameRegex = `^[a-zA-Z0-9_-]{1,64}$`
)

func DefineModel(client anthropic.Client, provider, name string, info ai.ModelOptions) ai.Model {
	label := "Anthropic"

	if provider == "vertexai" {
		label = "Vertex AI"
	}

	configSchema := info.ConfigSchema
	if configSchema == nil {
		configSchema = ConfigSchema(anthropic.MessageNewParams{})
	}

	meta := &ai.ModelOptions{
		Label:        label + "-" + name,
		Supports:     info.Supports,
		Versions:     info.Versions,
		ConfigSchema: configSchema,
	}

	return ai.NewModel(api.NewName(provider, name), meta, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return Generate(ctx, client, provider, name, input, cb)
	})
}

// ConfigSchema converts a config struct to a map[string]any.
func ConfigSchema(config any) map[string]any {
	r := jsonschema.Reflector{
		DoNotReference:             true, // Prevent $ref usage
		AllowAdditionalProperties:  false,
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: true,
	}
	// The anthropic SDK uses a number of wrapper types for float, int, etc.
	// By default, jsonschema will treat these as objects, but we want to
	// treat them as their underlying primitive types.
	r.Mapper = func(r reflect.Type) *jsonschema.Schema {
		if r.Name() == "Opt[float64]" {
			return &jsonschema.Schema{
				Type: "number",
			}
		}
		if r.Name() == "Opt[int64]" {
			return &jsonschema.Schema{
				Type: "integer",
			}
		}
		if r.Name() == "Opt[string]" {
			return &jsonschema.Schema{
				Type: "string",
			}
		}
		if r.Name() == "Opt[bool]" {
			return &jsonschema.Schema{
				Type: "boolean",
			}
		}
		return nil
	}
	schema := r.Reflect(config)
	result := base.SchemaAsMap(schema)

	return result
}

// Generate function defines how a generate request is done in Anthropic models
func Generate(
	ctx context.Context,
	client anthropic.Client,
	provider string,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	req, err := toAnthropicRequest(provider, input)
	if err != nil {
		return nil, fmt.Errorf("unable to generate anthropic request: %w", err)
	}

	req.Model = anthropic.Model(model)

	// no streaming
	if cb == nil {
		msg, err := client.Messages.New(ctx, *req)
		if err != nil {
			return nil, err
		}

		r, err := toGenkitResponse(msg)
		if err != nil {
			return nil, err
		}

		r.Request = input
		return r, nil
	} else {
		stream := client.Messages.NewStreaming(ctx, *req)
		message := anthropic.Message{}
		for stream.Next() {
			event := stream.Current()
			err := message.Accumulate(event)
			if err != nil {
				return nil, err
			}

			content := []*ai.Part{}
			switch event := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				if event.Delta.Type == "thinking_delta" {
					content = append(content, ai.NewReasoningPart(event.Delta.Thinking, []byte(event.Delta.Signature)))
				} else {
					content = append(content, ai.NewTextPart(event.Delta.Text))
				}
				err := cb(ctx, &ai.ModelResponseChunk{
					Content: content,
				})
				if err != nil {
					return nil, err
				}
			case anthropic.ContentBlockStopEvent:
				if int(event.Index) < len(message.Content) {
					block := message.Content[event.Index]
					if toolBlock, ok := block.AsAny().(anthropic.ToolUseBlock); ok {
						p := ai.NewToolRequestPart(&ai.ToolRequest{
							Ref:   toolBlock.ID,
							Input: toolBlock.Input,
							Name:  toolBlock.Name,
						})
						err := cb(ctx, &ai.ModelResponseChunk{
							Content: []*ai.Part{p},
						})
						if err != nil {
							return nil, err
						}
					}
				}
			case anthropic.MessageStopEvent:
				r, err := toGenkitResponse(&message)
				if err != nil {
					return nil, err
				}
				r.Request = input
				return r, nil
			}
		}
		if stream.Err() != nil {
			return nil, stream.Err()
		}
	}

	return nil, nil
}

func toAnthropicRole(role ai.Role) (anthropic.MessageParamRole, error) {
	switch role {
	case ai.RoleUser:
		return anthropic.MessageParamRoleUser, nil
	case ai.RoleModel:
		return anthropic.MessageParamRoleAssistant, nil
	case ai.RoleTool:
		return anthropic.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("unknown role given: %q", role)
	}
}

// toAnthropicRequest translates [ai.ModelRequest] to an Anthropic request
func toAnthropicRequest(provider string, i *ai.ModelRequest) (*anthropic.MessageNewParams, error) {
	messages := make([]anthropic.MessageParam, 0)

	req, err := configFromRequest(i)
	if err != nil {
		return nil, err
	}

	if req.MaxTokens == 0 {
		return nil, errors.New("maxTokens not set")
	}

	// configure system prompt (if given)
	sysBlocks := []anthropic.TextBlockParam{}
	for _, message := range i.Messages {
		if message.Role == ai.RoleSystem {
			block := anthropic.TextBlockParam{Text: message.Text()}
			// Enable prompt caching if the message has cache metadata.
			// Supports both explicit {"type": "ephemeral"} and the generic
			// {"ttlSeconds": N} used by ai.WithCacheTTL for cross-provider consistency.
			// Anthropic only supports ephemeral caching regardless of TTL value.
			if cache, ok := message.Metadata["cache"].(map[string]any); ok {
				t, _ := cache["type"].(string)
				_, hasTTL := cache["ttlSeconds"]
				if t == "ephemeral" || hasTTL {
					block.CacheControl = anthropic.CacheControlEphemeralParam{Type: "ephemeral"}
				}
			}
			sysBlocks = append(sysBlocks, block)
		} else if message.Content[len(message.Content)-1].IsToolResponse() {
			// if the last message is a ToolResponse, the conversation must continue
			// and the ToolResponse message must be sent as a user
			// see: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-result-content-blocks
			parts, err := toAnthropicParts(message.Content)
			if err != nil {
				return nil, err
			}
			messages = append(messages, anthropic.NewUserMessage(parts...))
		} else {
			parts, err := toAnthropicParts(message.Content)
			if err != nil {
				return nil, err
			}
			role, err := toAnthropicRole(message.Role)
			if err != nil {
				return nil, err
			}
			messages = append(messages, anthropic.MessageParam{
				Role:    role,
				Content: parts,
			})
		}
	}

	req.System = sysBlocks
	req.Messages = messages

	tools, err := toAnthropicTools(provider, i.Tools)
	if err != nil {
		return nil, err
	}
	req.Tools = tools

	if i.Output != nil && i.Output.Format == "json" && i.Output.Schema != nil && i.Output.Constrained {
		// Native structured output via OutputConfig
		var outputSchema map[string]any
		if b, err := json.Marshal(i.Output.Schema); err != nil {
			return nil, fmt.Errorf("failed to clone output schema: %w", err)
		} else if err := json.Unmarshal(b, &outputSchema); err != nil {
			return nil, fmt.Errorf("failed to clone output schema: %w", err)
		}
		enforceStrictSchema(outputSchema)

		req.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{
				Schema: outputSchema,
				// Type is elided, defaults to "json_schema"
			},
		}
	}

	return req, nil
}

// configFromRequest converts any supported config type to [anthropic.MessageNewParams]
func configFromRequest(input *ai.ModelRequest) (*anthropic.MessageNewParams, error) {
	var result anthropic.MessageNewParams

	switch config := input.Config.(type) {
	case anthropic.MessageNewParams:
		result = config
	case *anthropic.MessageNewParams:
		result = *config
	case map[string]any:
		var err error
		result, err = base.MapToStruct[anthropic.MessageNewParams](config)
		if err != nil {
			return nil, err
		}
	case nil:
		// Empty configuration is considered valid
	default:
		return nil, fmt.Errorf("unexpected config type: %T", input.Config)
	}
	return &result, nil
}

// toAnthropicTools translates [ai.ToolDefinition] to an anthropic.ToolParam type
func toAnthropicTools(provider string, tools []*ai.ToolDefinition) ([]anthropic.ToolUnionParam, error) {
	resp := make([]anthropic.ToolUnionParam, 0)
	regex := regexp.MustCompile(ToolNameRegex)

	for _, t := range tools {
		if t.Name == "" {
			return nil, fmt.Errorf("tool name is required")
		}
		if !regex.MatchString(t.Name) {
			return nil, fmt.Errorf("tool name must match regex: %s", ToolNameRegex)
		}

		var schema anthropic.ToolInputSchemaParam
		inputSchema := t.InputSchema
		if len(inputSchema) == 0 {
			inputSchema = map[string]any{"type": "object", "properties": map[string]any{}}
		}

		// Strict tools require additionalProperties: false recursively
		var strictInputSchema map[string]any
		if b, err := json.Marshal(inputSchema); err != nil {
			return nil, fmt.Errorf("failed to clone tool input schema: %w", err)
		} else if err := json.Unmarshal(b, &strictInputSchema); err != nil {
			return nil, fmt.Errorf("failed to clone tool input schema: %w", err)
		}
		enforceStrictSchema(strictInputSchema)
		inputSchema = strictInputSchema

		var err error
		schema, err = base.MapToStruct[anthropic.ToolInputSchemaParam](inputSchema)
		if err != nil {
			return nil, fmt.Errorf("unable to parse tool input schema: %w", err)
		}

		// ToolInputSchemaParam struct doesn't have AdditionalProperties field,
		// so we must add it to ExtraFields manually for the top-level schema.
		if schema.ExtraFields == nil {
			schema.ExtraFields = make(map[string]any)
		}
		if t, ok := inputSchema["type"].(string); ok && t == "object" {
			schema.ExtraFields["additionalProperties"] = false
		}

		tool := &anthropic.ToolParam{
			Name:        t.Name,
			Description: anthropic.String(t.Description),
			InputSchema: schema,
		}
		// Vertex AI's Anthropic endpoint does not support the strict field.
		if provider != "vertexai" {
			tool.Strict = anthropic.Bool(true)
		}
		resp = append(resp, anthropic.ToolUnionParam{OfTool: tool})
	}

	return resp, nil
}

// enforceStrictSchema is a helper function that sets additionalProperties to false
// for every object in a schema
func enforceStrictSchema(schema map[string]any) {
	if t, ok := schema["type"].(string); ok && t == "object" {
		schema["additionalProperties"] = false
		if props, ok := schema["properties"].(map[string]any); ok {
			for _, v := range props {
				if subSchema, ok := v.(map[string]any); ok {
					enforceStrictSchema(subSchema)
				}
			}
		}
	}
	// recursively enforce additionalProperties to false
	if t, ok := schema["type"].(string); ok && t == "array" {
		if items, ok := schema["items"].(map[string]any); ok {
			enforceStrictSchema(items)
		}
	}
}

// toAnthropicParts translates [ai.Part] to an anthropic.ContentBlockParamUnion type
func toAnthropicParts(parts []*ai.Part) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := []anthropic.ContentBlockParamUnion{}

	for _, p := range parts {
		switch {
		case p.IsText():
			blocks = append(blocks, anthropic.NewTextBlock(p.Text))
		case p.IsMedia():
			contentType, data, err := uri.Data(p)
			if err != nil {
				return nil, fmt.Errorf("unable to parse media part: %w", err)
			}
			blocks = append(blocks, anthropic.NewImageBlockBase64(contentType, base64.StdEncoding.EncodeToString(data)))
		case p.IsData():
			contentType, data, err := uri.Data(p)
			if err != nil {
				return nil, fmt.Errorf("unable to parse data part: %w", err)
			}
			blocks = append(blocks, anthropic.NewImageBlockBase64(contentType, base64.RawStdEncoding.EncodeToString(data)))
		case p.IsToolRequest():
			toolReq := p.ToolRequest
			blocks = append(blocks, anthropic.NewToolUseBlock(toolReq.Ref, toolReq.Input, toolReq.Name))
		case p.IsToolResponse():
			toolResp := p.ToolResponse
			output, err := json.Marshal(toolResp.Output)
			if err != nil {
				return nil, fmt.Errorf("unable to parse tool response, err: %w", err)
			}
			blocks = append(blocks, anthropic.NewToolResultBlock(toolResp.Ref, string(output), false))
		case p.IsReasoning():
			signature := []byte{}
			if p.Metadata != nil {
				if sig, ok := p.Metadata["signature"].([]byte); ok {
					signature = sig
				}
			}
			blocks = append(blocks, anthropic.NewThinkingBlock(string(signature), p.Text))
		default:
			return nil, errors.New("unknown part type in the request")
		}
	}

	return blocks, nil
}

// toGenkitResponse translates an Anthropic Message to [ai.ModelResponse]
func toGenkitResponse(m *anthropic.Message) (*ai.ModelResponse, error) {
	r := ai.ModelResponse{}

	switch m.StopReason {
	case anthropic.StopReasonMaxTokens:
		r.FinishReason = ai.FinishReasonLength
	case anthropic.StopReasonStopSequence:
		r.FinishReason = ai.FinishReasonStop
	case anthropic.StopReasonEndTurn:
		r.FinishReason = ai.FinishReasonStop
	case anthropic.StopReasonToolUse:
		r.FinishReason = ai.FinishReasonStop
	default:
		r.FinishReason = ai.FinishReasonUnknown
	}

	msg := &ai.Message{}
	msg.Role = ai.RoleModel
	for _, part := range m.Content {
		var p *ai.Part
		switch part.AsAny().(type) {
		case anthropic.ThinkingBlock:
			p = ai.NewReasoningPart(part.Thinking, []byte(part.Signature))
		case anthropic.TextBlock:
			p = ai.NewTextPart(string(part.Text))
		case anthropic.ToolUseBlock:
			p = ai.NewToolRequestPart(&ai.ToolRequest{
				Ref:   part.ID,
				Input: part.Input,
				Name:  part.Name,
			})
		default:
			return nil, fmt.Errorf("unknown part: %#v", part)
		}
		msg.Content = append(msg.Content, p)
	}

	r.Message = msg
	r.Raw = m.JSON
	r.Usage = &ai.GenerationUsage{
		InputTokens:         int(m.Usage.InputTokens),
		OutputTokens:        int(m.Usage.OutputTokens),
		CachedContentTokens: int(m.Usage.CacheReadInputTokens),
	}
	// Track cache creation tokens in Custom metrics (no dedicated field in GenerationUsage)
	if m.Usage.CacheCreationInputTokens > 0 {
		if r.Usage.Custom == nil {
			r.Usage.Custom = make(map[string]float64)
		}
		r.Usage.Custom["cacheCreationInputTokens"] = float64(m.Usage.CacheCreationInputTokens)
	}
	return &r, nil
}
