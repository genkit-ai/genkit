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
	"fmt"
	"reflect"
	"regexp"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal"
	pluginjsonschema "github.com/firebase/genkit/go/plugins/internal/jsonschema"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"github.com/invopop/jsonschema"

	"github.com/anthropics/anthropic-sdk-go"
)

const (
	ToolNameRegex = `^[a-zA-Z0-9_-]{1,64}$`

	// DefaultMaxOutputTokens is used when the request config does not set
	// MaxTokens, which the Anthropic API requires. It matches the JS plugin's
	// DEFAULT_MAX_OUTPUT_TOKENS and is low enough for every Claude model.
	DefaultMaxOutputTokens = 4096
)

// defaultConfigSchema is the schema every Claude model advertises for its
// config. Reflecting the SDK params struct is expensive and the result is
// read-only, so it is built once and shared by every model of both plugins.
var defaultConfigSchema = reflectConfigSchema(anthropic.MessageNewParams{})

// metadataSignature extracts a reasoning signature from part metadata. It
// handles both []byte (the value [ai.NewReasoningPart] stores) and string
// (base64-encoded, after the part has been through a JSON roundtrip such as
// persisted session history). Without the string case the signature is dropped
// and Anthropic rejects the replayed thinking block.
func metadataSignature(metadata map[string]any) []byte {
	switch sig := metadata["signature"].(type) {
	case []byte:
		return sig
	case string:
		decoded, err := base64.StdEncoding.DecodeString(sig)
		if err != nil {
			return nil
		}
		return decoded
	}
	return nil
}

// toAnthropicMediaBlock converts a media or data [ai.Part] to the content block
// its media type calls for. Anthropic only accepts images in an image block;
// PDFs and plain text must be sent as document blocks.
func toAnthropicMediaBlock(p *ai.Part, kind string) (anthropic.ContentBlockParamUnion, error) {
	contentType, data, err := uri.Data(p)
	if err != nil {
		return anthropic.ContentBlockParamUnion{}, status.Errorf(ai.ErrInvalidPart, "unable to parse %s part: %w", kind, err)
	}

	switch {
	case strings.HasPrefix(contentType, "image/"):
		return anthropic.NewImageBlockBase64(contentType, base64.StdEncoding.EncodeToString(data)), nil
	case contentType == "application/pdf":
		return anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{Data: base64.StdEncoding.EncodeToString(data)}), nil
	case contentType == "text/plain":
		return anthropic.NewDocumentBlock(anthropic.PlainTextSourceParam{Data: string(data)}), nil
	default:
		return anthropic.ContentBlockParamUnion{}, status.Errorf(ai.ErrUnsupportedByModel,
			"unsupported %s content type %q: Anthropic accepts image/*, application/pdf, and text/plain", kind, contentType)
	}
}

// NewModel creates a Claude model action without registering it. name is both
// the Genkit action name and the model ID requests are sent to: Anthropic
// resolves an alias like claude-opus-4-5 to its current dated release itself.
//
// opts is used as given, except that a nil ConfigSchema defaults to the
// reflected [anthropic.MessageNewParams] schema and an empty Label is derived
// from the provider and the name. The framework validates the request's config
// against that schema and deserializes it into [anthropic.MessageNewParams]
// before the model function runs.
func NewModel(client anthropic.Client, provider, name string, opts ai.ModelOptions) *ai.ModelAction {
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = defaultConfigSchema
	}
	if opts.Label == "" {
		opts.Label = internal.ProviderLabel(DisplayName(provider), name)
	}

	return ai.NewModelAction(api.NewName(provider, name), &opts, func(
		ctx context.Context,
		input *ai.ModelRequest,
		config anthropic.MessageNewParams,
		cb ai.ModelStreamCallback,
	) (*ai.ModelResponse, error) {
		return Generate(ctx, client, provider, name, input, config, cb)
	})
}

// DisplayName is how a provider is spelled in a model's dev UI label. Plugins
// that curate their own labels join it with [internal.ProviderLabel], so every Claude
// model names its provider the same way whichever plugin serves it.
func DisplayName(provider string) string {
	if provider == "vertexai" {
		return "Vertex AI"
	}
	return "Anthropic"
}

// reflectConfigSchema converts a config struct to a map[string]any.
func reflectConfigSchema(config any) map[string]any {
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
	schema := base.SchemaAsMap(r.Reflect(config))
	stripParamObjArtifact(schema)
	internal.ApplySchemaOverrides(schema, mncOverrides)
	return schema
}

// rejectManagedConfig reports a config field that a Genkit primitive owns.
//
// Each one is overwritten while the request is built, so accepting it would
// drop the caller's value on the floor: messages and the model unconditionally,
// the system prompt whenever the request carries system messages, and the
// output format whenever constrained generation is on. Failing with the option
// to use instead beats a request that silently ignores half of what it was
// given.
//
// These are hidden from the advertised schema (see mncOverrides), so this is
// what a caller setting one in code sees. They are hidden by being replaced
// with a permissive schema rather than deleted, so the value reaches here
// rather than failing validation as an unknown property.
//
// Classified ErrInvalidArgument: the request is the caller's to fix, so this
// reaches the dev UI and any HTTP transport as a 400 rather than a 500. Not
// ErrInvalidInput, which means a value failed the action's input schema; these
// pass the schema and are refused on what they mean.
func rejectManagedConfig(config *anthropic.MessageNewParams) error {
	switch {
	case len(config.Messages) > 0:
		return status.Errorf(status.ErrInvalidArgument, "messages must be set using Genkit feature: ai.WithMessages() or ai.WithPrompt()")
	case len(config.System) > 0:
		return status.Errorf(status.ErrInvalidArgument, "system prompt must be set using Genkit feature: ai.WithSystem()")
	case config.Model != "":
		return status.Errorf(status.ErrInvalidArgument, "the model is chosen by the action; set it using Genkit feature: ai.WithModel() or ai.WithModelName()")
	case config.OutputConfig.Format.Schema != nil || config.OutputConfig.Format.Type != "":
		return status.Errorf(status.ErrInvalidArgument, "output format must be set using Genkit feature: ai.WithOutputType() or ai.WithOutputSchema(); the config-level output_config.effort field is unaffected")
	}
	for _, t := range config.Tools {
		if t.OfTool != nil {
			return status.Errorf(status.ErrInvalidArgument, "custom function tools must be set using Genkit feature: ai.WithTools(); the config-level tools field is reserved for server-side tools (web search, code execution, etc.)")
		}
	}
	return nil
}

// Generate function defines how a generate request is done in Anthropic models.
// config is the request's config, already deserialized by the framework, and is
// the base the request is built on.
func Generate(
	ctx context.Context,
	client anthropic.Client,
	provider string,
	model string,
	input *ai.ModelRequest,
	config anthropic.MessageNewParams,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	req, err := toAnthropicRequest(provider, input, config)
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
		if err := stream.Err(); err != nil {
			return nil, err
		}
		// The loop only returns from the message_stop case. Falling out of it
		// means the stream ended early without one, and the SDK reports no
		// error for a body that simply stops, so say so rather than returning
		// a nil response the caller would dereference.
		return nil, fmt.Errorf("anthropic stream ended without a message_stop event")
	}
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

// toAnthropicRequest folds an [ai.ModelRequest] into the config the framework
// deserialized for the request, and returns the result to send to the API.
// config is taken by value: the request's own copy is what gets amended, never
// the caller's.
func toAnthropicRequest(provider string, i *ai.ModelRequest, config anthropic.MessageNewParams) (*anthropic.MessageNewParams, error) {
	messages := make([]anthropic.MessageParam, 0)

	req := &config
	if err := rejectManagedConfig(req); err != nil {
		return nil, err
	}

	// max_tokens is required by the Anthropic API. Fall back to a conservative
	// default that every Claude model accepts, mirroring the JS plugin's
	// DEFAULT_MAX_OUTPUT_TOKENS, so a bare Generate call works without config.
	if req.MaxTokens == 0 {
		req.MaxTokens = DefaultMaxOutputTokens
	}

	// configure system prompt (if given)
	sysBlocks := []anthropic.TextBlockParam{}
	for _, message := range i.Messages {
		if message.Role == ai.RoleSystem {
			// only text is supported for system messages
			sysBlocks = append(sysBlocks, anthropic.TextBlockParam{Text: message.Text()})
			continue
		}

		parts, err := toAnthropicParts(message.Content)
		if err != nil {
			return nil, err
		}
		// Anthropic rejects messages with an empty content array.
		if len(parts) == 0 {
			continue
		}

		if lastPart := message.Content[len(message.Content)-1]; lastPart.IsToolResponse() {
			// if the last message is a ToolResponse, the conversation must continue
			// and the ToolResponse message must be sent as a user
			// see: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-result-content-blocks
			messages = append(messages, anthropic.NewUserMessage(parts...))
			continue
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

	// The config cannot carry a system prompt (rejectManagedConfig refuses
	// one), so this only guards against sending an empty array.
	if len(sysBlocks) > 0 {
		req.System = sysBlocks
	}
	req.Messages = messages

	tools, err := toAnthropicTools(provider, i.Tools)
	if err != nil {
		return nil, err
	}
	// Append rather than assign: server-side tools (web search, code execution,
	// ...) can only be expressed through the config, and assigning here would
	// silently drop them.
	//
	// Clip first so the append always allocates. config is only a shallow copy,
	// so its slice header still points at the caller's backing array, and a
	// config hoisted into a package-level var or a ModelRef is shared by every
	// request made with it. Appending in place would write into that array's
	// spare capacity, which two concurrent requests then race over, and one
	// request's tools would surface in another's.
	req.Tools = append(slices.Clip(req.Tools), tools...)

	if toolChoice, ok := toAnthropicToolChoice(i.ToolChoice); ok {
		req.ToolChoice = toolChoice
	}

	if i.Output != nil && i.Output.Format == "json" && i.Output.Schema != nil && i.Output.Constrained {
		// Native structured output via OutputConfig. Set only the format so a
		// config-provided OutputConfig.Effort survives.
		req.OutputConfig.Format = anthropic.JSONOutputFormatParam{
			Schema: pluginjsonschema.EnforceStrict(i.Output.Schema),
			// Type is elided, defaults to "json_schema"
		}
	}

	return req, nil
}

// toAnthropicToolChoice translates [ai.ToolChoice] to the Anthropic tool_choice
// union. The second return value reports whether a choice was set; when false
// the caller leaves any config-provided tool_choice untouched.
func toAnthropicToolChoice(choice ai.ToolChoice) (anthropic.ToolChoiceUnionParam, bool) {
	switch choice {
	case ai.ToolChoiceAuto:
		return anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}, true
	case ai.ToolChoiceRequired:
		return anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}, true
	case ai.ToolChoiceNone:
		return anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}, true
	default:
		return anthropic.ToolChoiceUnionParam{}, false
	}
}

// toAnthropicTools translates [ai.ToolDefinition] to an anthropic.ToolParam type
func toAnthropicTools(provider string, tools []*ai.ToolDefinition) ([]anthropic.ToolUnionParam, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	resp := make([]anthropic.ToolUnionParam, 0, len(tools))
	regex := regexp.MustCompile(ToolNameRegex)

	for _, t := range tools {
		if t.Name == "" {
			return nil, fmt.Errorf("tool name is required")
		}
		if !regex.MatchString(t.Name) {
			return nil, fmt.Errorf("tool name must match regex: %s", ToolNameRegex)
		}

		inputSchema := t.InputSchema
		if len(inputSchema) == 0 {
			inputSchema = map[string]any{"type": "object", "properties": map[string]any{}}
		}

		// Vertex AI's Anthropic endpoint does not support the strict field;
		// elsewhere, strict is the default unless the tool opts out.
		strictSupported := provider != "vertexai"
		strictRequested := true
		if v, ok := t.Metadata["strict"].(bool); ok {
			strictRequested = v
		}
		strict := strictSupported && strictRequested

		if strict {
			inputSchema = pluginjsonschema.EnforceStrict(inputSchema)
		}

		schema, err := base.MapToStruct[anthropic.ToolInputSchemaParam](inputSchema)
		if err != nil {
			return nil, fmt.Errorf("unable to parse tool input schema: %w", err)
		}

		// ToolInputSchemaParam struct doesn't have AdditionalProperties field,
		// so we must add it to ExtraFields manually for the top-level schema.
		if strict {
			if schema.ExtraFields == nil {
				schema.ExtraFields = make(map[string]any)
			}
			if typ, ok := inputSchema["type"].(string); ok && typ == "object" {
				schema.ExtraFields["additionalProperties"] = false
			}
		}

		tool := &anthropic.ToolParam{
			Name:        t.Name,
			Description: anthropic.String(t.Description),
			InputSchema: schema,
		}
		// Only set strict when true. Sending strict: false still triggers
		// Anthropic's supported-keywords validator (which rejects e.g.
		// maxItems/minItems); omitting the field skips validation entirely.
		if strict {
			tool.Strict = anthropic.Bool(true)
		}
		resp = append(resp, anthropic.ToolUnionParam{OfTool: tool})
	}

	return resp, nil
}

// toAnthropicParts translates [ai.Part] to an anthropic.ContentBlockParamUnion type
func toAnthropicParts(parts []*ai.Part) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := []anthropic.ContentBlockParamUnion{}

	for _, p := range parts {
		switch {
		case p.IsText():
			blocks = append(blocks, anthropic.NewTextBlock(p.Text))
		case p.IsMedia():
			block, err := toAnthropicMediaBlock(p, "media")
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case p.IsData():
			block, err := toAnthropicMediaBlock(p, "data")
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case p.IsToolRequest():
			toolReq := p.ToolRequest
			blocks = append(blocks, anthropic.NewToolUseBlock(toolReq.Ref, toolReq.Input, toolReq.Name))
		case p.IsToolResponse():
			block, err := toAnthropicToolResultBlock(p.ToolResponse)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case p.IsReasoning():
			blocks = append(blocks, anthropic.NewThinkingBlock(string(metadataSignature(p.Metadata)), p.Text))
		default:
			return nil, status.Errorf(ai.ErrInvalidPart, "unknown part type in the request")
		}
	}

	return blocks, nil
}

// toAnthropicToolResultBlock translates an [ai.ToolResponse] to an Anthropic
// tool_result block.
//
// Multipart tools return rich content parts alongside their structured output.
// Anthropic exposes a single content array per tool_result rather than separate
// fields, so the structured output is emitted as a leading text block and the
// content parts follow as text, image, or document blocks.
func toAnthropicToolResultBlock(toolResp *ai.ToolResponse) (anthropic.ContentBlockParamUnion, error) {
	if toolResp == nil {
		return anthropic.ContentBlockParamUnion{}, status.Errorf(ai.ErrInvalidPart, "tool response part carries no tool response")
	}

	content := make([]anthropic.ToolResultBlockParamContentUnion, 0, len(toolResp.Content)+1)

	// Only send the structured output when the tool produced one. A multipart
	// tool may return content parts alone, and a literal "null" text block
	// would be noise to the model. Tool responses with neither output nor
	// content still send "null" so the block is never empty, which the API
	// rejects.
	if toolResp.Output != nil || len(toolResp.Content) == 0 {
		output, err := json.Marshal(toolResp.Output)
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, fmt.Errorf("unable to parse tool response, err: %w", err)
		}
		content = append(content, anthropic.ToolResultBlockParamContentUnion{
			OfText: &anthropic.TextBlockParam{Text: string(output)},
		})
	}

	for _, p := range toolResp.Content {
		c, err := toAnthropicToolResultContent(p)
		if err != nil {
			return anthropic.ContentBlockParamUnion{}, err
		}
		content = append(content, c)
	}

	return anthropic.ContentBlockParamUnion{
		OfToolResult: &anthropic.ToolResultBlockParam{
			ToolUseID: toolResp.Ref,
			Content:   content,
			IsError:   anthropic.Bool(false),
		},
	}, nil
}

// toAnthropicToolResultContent translates a part of a multipart tool response
// to a block accepted inside an Anthropic tool_result.
func toAnthropicToolResultContent(p *ai.Part) (anthropic.ToolResultBlockParamContentUnion, error) {
	switch {
	case p.IsText():
		return anthropic.ToolResultBlockParamContentUnion{
			OfText: &anthropic.TextBlockParam{Text: p.Text},
		}, nil
	case p.IsMedia(), p.IsData():
		kind := "media"
		if p.IsData() {
			kind = "data"
		}
		block, err := toAnthropicMediaBlock(p, kind)
		if err != nil {
			return anthropic.ToolResultBlockParamContentUnion{}, err
		}
		switch {
		case block.OfImage != nil:
			return anthropic.ToolResultBlockParamContentUnion{OfImage: block.OfImage}, nil
		case block.OfDocument != nil:
			return anthropic.ToolResultBlockParamContentUnion{OfDocument: block.OfDocument}, nil
		}
	}

	return anthropic.ToolResultBlockParamContentUnion{}, status.Errorf(ai.ErrInvalidPart,
		"unsupported part in tool response content: Anthropic tool results accept text, image, and document parts")
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
			return nil, status.Errorf(ai.ErrInvalidPart, "unknown part: %#v", part)
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
	return &r, nil
}
