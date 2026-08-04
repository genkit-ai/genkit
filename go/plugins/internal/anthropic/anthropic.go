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
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
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

func DefineModel(client anthropic.Client, provider, name string, info ai.ModelOptions, defaultAPIVersion string) ai.Model {
	label := "Anthropic"

	if provider == "vertexai" {
		label = "Vertex AI"
	}

	configSchema := info.ConfigSchema
	if configSchema == nil {
		configSchema = ConfigSchemaWithRouting(anthropic.MessageNewParams{})
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
		return Generate(ctx, client, provider, name, input, cb, defaultAPIVersion)
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

// ConfigSchemaWithRouting extends [ConfigSchema] with Genkit-only routing
// fields (apiVersion, betas) used to select the Anthropic beta Messages surface.
func ConfigSchemaWithRouting(config any) map[string]any {
	schema := ConfigSchema(config)
	props, _ := schema["properties"].(map[string]any)
	if props == nil {
		props = map[string]any{}
		schema["properties"] = props
	}
	props["apiVersion"] = map[string]any{
		"type":        "string",
		"enum":        []any{APIVersionStable, APIVersionBeta},
		"description": "Anthropic API surface to use for this request (stable or beta).",
	}
	props["betas"] = map[string]any{
		"type":        "array",
		"items":       map[string]any{"type": "string"},
		"description": "Beta feature headers when apiVersion is beta. Defaults to the plugin's standard beta set when omitted.",
	}
	return schema
}

// Generate function defines how a generate request is done in Anthropic models.
// defaultAPIVersion is the plugin-wide default ("stable" or "beta"); per-request
// config.apiVersion overrides it when present.
func Generate(
	ctx context.Context,
	client anthropic.Client,
	provider string,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
	defaultAPIVersion string,
) (*ai.ModelResponse, error) {
	apiVersion, betas := resolveAPIVersion(input, defaultAPIVersion)

	req, err := toAnthropicRequest(provider, input)
	if err != nil {
		return nil, fmt.Errorf("unable to generate anthropic request: %w", err)
	}

	req.Model = anthropic.Model(model)

	if apiVersion == APIVersionBeta {
		return generateBeta(ctx, client, req, betas, input, cb)
	}

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
	}

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
			if event.Index >= 0 && int(event.Index) < len(message.Content) {
				p, err := contentBlockToPart(message.Content[event.Index])
				if err != nil {
					return nil, err
				}
				if shouldEmitOnContentBlockStop(p) {
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

	return nil, nil
}

// resolveAPIVersion picks the Anthropic API surface for a request.
// Priority: request config.apiVersion > plugin default > stable.
// Betas: nil means unset (use plugin defaults on the beta path); a non-nil
// empty slice means the request explicitly set betas: [].
func resolveAPIVersion(input *ai.ModelRequest, pluginDefault string) (string, []anthropic.AnthropicBeta) {
	version := pluginDefault
	if version == "" {
		version = APIVersionStable
	}
	var betas []anthropic.AnthropicBeta

	if cfg, ok := input.Config.(map[string]any); ok {
		if v, ok := cfg["apiVersion"].(string); ok && v != "" {
			version = v
		}
		// Match JS: only honor betas when it is an array. null/other types leave
		// betas unset so the beta path applies defaultBetas.
		switch b := cfg["betas"].(type) {
		case []string:
			betas = make([]anthropic.AnthropicBeta, 0, len(b))
			for _, s := range b {
				betas = append(betas, anthropic.AnthropicBeta(s))
			}
		case []any:
			betas = make([]anthropic.AnthropicBeta, 0, len(b))
			for _, item := range b {
				if s, ok := item.(string); ok {
					betas = append(betas, anthropic.AnthropicBeta(s))
				}
			}
		}
	}

	return version, betas
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

	// Only overwrite the config-provided system prompt when the request
	// actually carries system messages, and never send an empty array.
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
	req.Tools = append(req.Tools, tools...)

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

// configFromRequest converts any supported config type to [anthropic.MessageNewParams]
func configFromRequest(input *ai.ModelRequest) (*anthropic.MessageNewParams, error) {
	var result anthropic.MessageNewParams

	switch config := input.Config.(type) {
	case anthropic.MessageNewParams:
		result = config
	case *anthropic.MessageNewParams:
		result = *config
	case map[string]any:
		cleaned := make(map[string]any, len(config))
		for k, v := range config {
			// Genkit-only routing fields — not part of MessageNewParams.
			if k == "apiVersion" || k == "betas" {
				continue
			}
			cleaned[k] = v
		}
		var err error
		result, err = base.MapToStruct[anthropic.MessageNewParams](cleaned)
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
			toolResp := p.ToolResponse
			output, err := json.Marshal(toolResp.Output)
			if err != nil {
				return nil, fmt.Errorf("unable to parse tool response, err: %w", err)
			}
			blocks = append(blocks, anthropic.NewToolResultBlock(toolResp.Ref, string(output), false))
		case p.IsReasoning():
			blocks = append(blocks, anthropic.NewThinkingBlock(string(metadataSignature(p.Metadata)), p.Text))
		default:
			return nil, status.Errorf(ai.ErrInvalidPart, "unknown part type in the request")
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
		p, err := contentBlockToPart(part)
		if err != nil {
			return nil, err
		}
		if p != nil {
			msg.Content = append(msg.Content, p)
		}
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

func contentBlockToPart(part anthropic.ContentBlockUnion) (*ai.Part, error) {
	switch b := part.AsAny().(type) {
	case anthropic.ThinkingBlock:
		return ai.NewReasoningPart(b.Thinking, []byte(b.Signature)), nil
	case anthropic.TextBlock:
		return ai.NewTextPart(string(b.Text)), nil
	case anthropic.ToolUseBlock:
		return ai.NewToolRequestPart(&ai.ToolRequest{
			Ref:   b.ID,
			Input: b.Input,
			Name:  b.Name,
		}), nil
	case anthropic.RedactedThinkingBlock:
		return ai.NewCustomPart(map[string]any{"redactedThinking": b.Data}), nil
	case anthropic.ServerToolUseBlock:
		return serverToolUseToPart(b.ID, string(b.Name), b.Input), nil
	case anthropic.WebSearchToolResultBlock:
		return webSearchToolResultToPart(b.ToolUseID, parseJSONAny(b.Content.RawJSON())), nil
	case anthropic.WebFetchToolResultBlock,
		anthropic.CodeExecutionToolResultBlock,
		anthropic.BashCodeExecutionToolResultBlock,
		anthropic.TextEditorCodeExecutionToolResultBlock,
		anthropic.ToolSearchToolResultBlock,
		anthropic.ContainerUploadBlock:
		return nil, unsupportedServerToolError(part.Type)
	default:
		return nil, status.Errorf(ai.ErrInvalidPart, "unknown part: %#v", part)
	}
}
