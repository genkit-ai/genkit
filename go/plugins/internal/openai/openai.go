// Copyright 2026 Google LLC
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

package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strings"

	oa "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
	pluginjsonschema "github.com/firebase/genkit/go/plugins/internal/jsonschema"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"github.com/invopop/jsonschema"
)

const toolNameRegex = `^[a-zA-Z0-9_-]{1,64}$`

type streamState struct {
	functionCallNames map[string]string
	functionCallIDs   map[string]string
	emittedToolCalls  map[string]bool
	emittedText       map[string]bool
	emittedRefusals   map[string]bool
	emittedReasoning  map[string]bool
}

func newStreamState() *streamState {
	return &streamState{
		functionCallNames: make(map[string]string),
		functionCallIDs:   make(map[string]string),
		emittedToolCalls:  make(map[string]bool),
		emittedText:       make(map[string]bool),
		emittedRefusals:   make(map[string]bool),
		emittedReasoning:  make(map[string]bool),
	}
}

func DefineModel(client oa.Client, provider, name string, info ai.ModelOptions) ai.Model {
	configSchema := info.ConfigSchema
	if configSchema == nil {
		configSchema = ConfigSchema(responses.ResponseNewParams{})
	}

	meta := &ai.ModelOptions{
		Label:        info.Label,
		Supports:     info.Supports,
		Versions:     info.Versions,
		ConfigSchema: configSchema,
		Stage:        info.Stage,
	}

	return ai.NewModel(api.NewName(provider, name), meta, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return Generate(ctx, client, name, input, cb)
	})
}

// ConfigSchema converts a config struct to a map[string]any.
func ConfigSchema(config any) map[string]any {
	r := jsonschema.Reflector{
		DoNotReference:             true,
		AllowAdditionalProperties:  false,
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: true,
	}
	r.Mapper = func(rt reflect.Type) *jsonschema.Schema {
		switch rt.Name() {
		case "Opt[string]":
			return &jsonschema.Schema{Type: "string"}
		case "Opt[int64]":
			return &jsonschema.Schema{Type: "integer"}
		case "Opt[float64]":
			return &jsonschema.Schema{Type: "number"}
		case "Opt[bool]":
			return &jsonschema.Schema{Type: "boolean"}
		}
		return nil
	}
	return base.SchemaAsMap(r.Reflect(config))
}

func Generate(
	ctx context.Context,
	client oa.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	req, err := toOpenAIRequest(input)
	if err != nil {
		return nil, fmt.Errorf("unable to generate openai request: %w", err)
	}
	req.Model = model

	if cb == nil {
		resp, err := client.Responses.New(ctx, *req)
		if err != nil {
			return nil, err
		}
		out, err := toGenkitResponse(resp)
		if err != nil {
			return nil, err
		}
		out.Request = input
		return out, nil
	}

	stream := client.Responses.NewStreaming(ctx, *req)
	var final *responses.Response
	state := newStreamState()

	for stream.Next() {
		event := stream.Current()
		if err := handleStreamEvent(ctx, state, event, cb); err != nil {
			return nil, err
		}
		switch v := event.AsAny().(type) {
		case responses.ResponseCompletedEvent:
			final = &v.Response
			out, err := toGenkitResponse(final)
			if err != nil {
				return nil, err
			}
			out.Request = input
			return out, nil
		case responses.ResponseIncompleteEvent:
			final = &v.Response
			out, err := toGenkitResponse(final)
			if err != nil {
				return nil, err
			}
			out.Request = input
			return out, nil
		case responses.ResponseFailedEvent:
			return nil, fmt.Errorf("openai response failed: %s", v.Response.RawJSON())
		case responses.ResponseErrorEvent:
			return nil, fmt.Errorf("openai stream error: %s", v.Message)
		}
	}
	if stream.Err() != nil {
		return nil, stream.Err()
	}
	if final != nil {
		out, err := toGenkitResponse(final)
		if err != nil {
			return nil, err
		}
		out.Request = input
		return out, nil
	}
	return nil, errors.New("openai stream completed without final response")
}

func handleStreamEvent(
	ctx context.Context,
	state *streamState,
	event responses.ResponseStreamEventUnion,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) error {
	switch v := event.AsAny().(type) {
	case responses.ResponseTextDeltaEvent:
		if v.Delta == "" {
			return nil
		}
		state.emittedText[contentKey(v.ItemID, v.ContentIndex)] = true
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(v.Delta)},
		})
	case responses.ResponseTextDoneEvent:
		if v.Text == "" {
			return nil
		}
		key := contentKey(v.ItemID, v.ContentIndex)
		if state.emittedText[key] {
			return nil
		}
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(v.Text)},
		})
	case responses.ResponseRefusalDeltaEvent:
		if v.Delta == "" {
			return nil
		}
		state.emittedRefusals[contentKey(v.ItemID, v.ContentIndex)] = true
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(v.Delta)},
		})
	case responses.ResponseRefusalDoneEvent:
		if v.Refusal == "" {
			return nil
		}
		key := contentKey(v.ItemID, v.ContentIndex)
		if state.emittedRefusals[key] {
			return nil
		}
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(v.Refusal)},
		})
	case responses.ResponseReasoningDeltaEvent:
		text := normalizeReasoningDelta(v.Delta)
		if text == "" {
			return nil
		}
		state.emittedReasoning[contentKey(v.ItemID, v.ContentIndex)] = true
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewReasoningPart(text, nil)},
		})
	case responses.ResponseReasoningDoneEvent:
		if v.Text == "" {
			return nil
		}
		key := contentKey(v.ItemID, v.ContentIndex)
		if state.emittedReasoning[key] {
			return nil
		}
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewReasoningPart(v.Text, nil)},
		})
	case responses.ResponseReasoningSummaryTextDeltaEvent:
		if v.Delta == "" {
			return nil
		}
		state.emittedReasoning[summaryKey(v.ItemID, v.SummaryIndex)] = true
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewReasoningPart(v.Delta, nil)},
		})
	case responses.ResponseReasoningSummaryTextDoneEvent:
		if v.Text == "" {
			return nil
		}
		key := summaryKey(v.ItemID, v.SummaryIndex)
		if state.emittedReasoning[key] {
			return nil
		}
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewReasoningPart(v.Text, nil)},
		})
	case responses.ResponseReasoningSummaryDoneEvent:
		if v.Text == "" {
			return nil
		}
		key := summaryKey(v.ItemID, v.SummaryIndex)
		if state.emittedReasoning[key] {
			return nil
		}
		return cb(ctx, &ai.ModelResponseChunk{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewReasoningPart(v.Text, nil)},
		})
	case responses.ResponseOutputItemAddedEvent:
		if tool, ok := v.Item.AsAny().(responses.ResponseFunctionToolCall); ok {
			ref := firstNonEmpty(tool.ID, tool.CallID)
			state.functionCallNames[ref] = tool.Name
			state.functionCallIDs[ref] = firstNonEmpty(tool.CallID, tool.ID)
		}
	case responses.ResponseFunctionCallArgumentsDeltaEvent:
		if v.ItemID == "" {
			return nil
		}
	case responses.ResponseFunctionCallArgumentsDoneEvent:
		if v.ItemID == "" {
			return nil
		}
		name := state.functionCallNames[v.ItemID]
		if name == "" {
			return nil
		}
		if state.emittedToolCalls[v.ItemID] {
			return nil
		}
		args := decodeJSONArg(v.Arguments)
		state.emittedToolCalls[v.ItemID] = true
		return cb(ctx, &ai.ModelResponseChunk{
			Role: ai.RoleModel,
			Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
				Ref:   firstNonEmpty(state.functionCallIDs[v.ItemID], v.ItemID),
				Name:  name,
				Input: args,
			})},
		})
	case responses.ResponseOutputItemDoneEvent:
		if tool, ok := v.Item.AsAny().(responses.ResponseFunctionToolCall); ok {
			ref := firstNonEmpty(tool.ID, tool.CallID)
			if ref != "" && tool.Name != "" {
				state.functionCallNames[ref] = tool.Name
				state.functionCallIDs[ref] = firstNonEmpty(tool.CallID, tool.ID)
			}
			if ref != "" && state.emittedToolCalls[ref] {
				return nil
			}
			args := decodeJSONArg(tool.Arguments)
			if ref != "" {
				state.emittedToolCalls[ref] = true
			}
			return cb(ctx, &ai.ModelResponseChunk{
				Role: ai.RoleModel,
				Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
					Ref:   firstNonEmpty(tool.CallID, tool.ID),
					Name:  tool.Name,
					Input: args,
				})},
			})
		}
	case responses.ResponseErrorEvent:
		return fmt.Errorf("openai stream error: %s", v.Message)
	}
	return nil
}

func toOpenAIRequest(input *ai.ModelRequest) (*responses.ResponseNewParams, error) {
	req, err := configFromRequest(input)
	if err != nil {
		return nil, err
	}

	items, err := toOpenAIInput(input.Messages)
	if err != nil {
		return nil, err
	}
	if len(items) > 0 {
		req.Input = responses.ResponseNewParamsInputUnion{
			OfInputItemList: items,
		}
	}

	tools, err := toOpenAITools(input.Tools)
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		req.Tools = tools
		req.ToolChoice = toOpenAIToolChoice(input.ToolChoice)
	}

	if input.Output != nil && input.Output.Format == "json" && input.Output.Schema != nil {
		req.Text = responses.ResponseTextConfigParam{
			Format: responses.ResponseFormatTextConfigUnionParam{
				OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
					Name:        "structured_output",
					Schema:      pluginjsonschema.EnforceStrict(input.Output.Schema),
					Strict:      oa.Bool(input.Output.Constrained),
					Description: oa.String("Structured output generated by Genkit."),
				},
			},
		}
	}

	return req, nil
}

func configFromRequest(input *ai.ModelRequest) (*responses.ResponseNewParams, error) {
	var result responses.ResponseNewParams

	switch config := input.Config.(type) {
	case responses.ResponseNewParams:
		result = config
	case *responses.ResponseNewParams:
		result = *config
	case map[string]any:
		var err error
		result, err = base.MapToStruct[responses.ResponseNewParams](config)
		if err != nil {
			return nil, err
		}
	case nil:
	default:
		return nil, fmt.Errorf("unexpected config type: %T", input.Config)
	}
	return &result, nil
}

func toOpenAIInput(messages []*ai.Message) (responses.ResponseInputParam, error) {
	items := responses.ResponseInputParam{}

	for _, message := range messages {
		if message == nil {
			continue
		}

		switch message.Role {
		case ai.RoleSystem:
			parts, err := toOpenAIMessageContent(message.Content)
			if err != nil {
				return nil, err
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfInputMessage: &responses.ResponseInputItemMessageParam{
					Content: parts,
					Role:    "system",
				},
			})
		case ai.RoleUser:
			parts, err := toOpenAIMessageContent(message.Content)
			if err != nil {
				return nil, err
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfInputMessage: &responses.ResponseInputItemMessageParam{
					Content: parts,
					Role:    "user",
				},
			})
		case ai.RoleModel:
			outputItems, err := toOpenAIModelHistoryItems(message.Content)
			if err != nil {
				return nil, err
			}
			items = append(items, outputItems...)
		case ai.RoleTool:
			for _, part := range message.Content {
				if !part.IsToolResponse() || part.ToolResponse == nil {
					return nil, fmt.Errorf("unsupported tool message part: %+v", part)
				}
				output, err := json.Marshal(part.ToolResponse.Output)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal tool response output for %s: %w", part.ToolResponse.Ref, err)
				}
				items = append(items, responses.ResponseInputItemUnionParam{
					OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
						CallID: part.ToolResponse.Ref,
						Output: string(output),
						Status: "completed",
					},
				})
			}
		default:
			return nil, fmt.Errorf("unknown role given: %q", message.Role)
		}
	}

	return items, nil
}

func toOpenAIMessageContent(parts []*ai.Part) (responses.ResponseInputMessageContentListParam, error) {
	out := responses.ResponseInputMessageContentListParam{}

	for _, part := range parts {
		switch {
		case part.IsText():
			out = append(out, responses.ResponseInputContentUnionParam{
				OfInputText: &responses.ResponseInputTextParam{
					Text: part.Text,
				},
			})
		case part.IsMedia():
			contentType, data, err := uri.Data(part)
			if err != nil {
				return nil, fmt.Errorf("unable to parse media part: %w", err)
			}
			imageURL := part.Text
			if imageURL == "" && len(data) > 0 {
				imageURL = fmt.Sprintf("data:%s;base64,%s", contentType, base64.StdEncoding.EncodeToString(data))
			}
			out = append(out, responses.ResponseInputContentUnionParam{
				OfInputImage: &responses.ResponseInputImageParam{
					Detail:   responses.ResponseInputImageDetailAuto,
					ImageURL: oa.String(imageURL),
				},
			})
		case part.IsData():
			out = append(out, responses.ResponseInputContentUnionParam{
				OfInputFile: &responses.ResponseInputFileParam{
					FileData: oa.String(part.Text),
					Filename: oa.String("input.txt"),
				},
			})
		default:
			return nil, fmt.Errorf("unsupported part type in OpenAI input message: %+v", part)
		}
	}

	return out, nil
}

func toOpenAIModelHistoryItems(parts []*ai.Part) ([]responses.ResponseInputItemUnionParam, error) {
	items := make([]responses.ResponseInputItemUnionParam, 0)
	textContent := make([]responses.ResponseOutputMessageContentUnionParam, 0)
	flushText := func() {
		if len(textContent) == 0 {
			return
		}
		items = append(items, responses.ResponseInputItemUnionParam{
			OfOutputMessage: &responses.ResponseOutputMessageParam{
				Status:  responses.ResponseOutputMessageStatusCompleted,
				Content: textContent,
			},
		})
		textContent = nil
	}

	for _, part := range parts {
		switch {
		case part.IsText():
			textContent = append(textContent, responses.ResponseOutputMessageContentUnionParam{
				OfOutputText: &responses.ResponseOutputTextParam{
					Text: part.Text,
				},
			})
		case part.IsData():
			textContent = append(textContent, responses.ResponseOutputMessageContentUnionParam{
				OfOutputText: &responses.ResponseOutputTextParam{
					Text: part.Text,
				},
			})
		case part.IsToolRequest():
			flushText()
			payload, err := json.Marshal(part.ToolRequest.Input)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal tool request input for %s: %w", part.ToolRequest.Ref, err)
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCall: &responses.ResponseFunctionToolCallParam{
					CallID:    part.ToolRequest.Ref,
					Name:      part.ToolRequest.Name,
					Arguments: string(payload),
					Status:    responses.ResponseFunctionToolCallStatusCompleted,
				},
			})
		case part.IsToolResponse():
			flushText()
			output, err := json.Marshal(part.ToolResponse.Output)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal tool response output for %s: %w", part.ToolResponse.Ref, err)
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
					CallID: part.ToolResponse.Ref,
					Output: string(output),
					Status: "completed",
				},
			})
		case part.IsReasoning():
			flushText()
			reasoning := &responses.ResponseReasoningItemParam{
				Summary: []responses.ResponseReasoningItemSummaryParam{{
					Text: part.Text,
				}},
				Status: responses.ResponseReasoningItemStatusCompleted,
			}
			if part.Metadata != nil {
				if encrypted, ok := part.Metadata["encrypted_content"].(string); ok && encrypted != "" {
					reasoning.EncryptedContent = oa.String(encrypted)
				}
				if id, ok := part.Metadata["id"].(string); ok && id != "" {
					reasoning.ID = id
				}
			}
			items = append(items, responses.ResponseInputItemUnionParam{
				OfReasoning: reasoning,
			})
		default:
			return nil, fmt.Errorf("unsupported part type in OpenAI model history: %+v", part)
		}
	}

	flushText()
	return items, nil
}

func toOpenAITools(tools []*ai.ToolDefinition) ([]responses.ToolUnionParam, error) {
	resp := make([]responses.ToolUnionParam, 0, len(tools))
	regex := regexp.MustCompile(toolNameRegex)
	for _, t := range tools {
		if t.Name == "" {
			return nil, errors.New("tool name is required")
		}
		if !regex.MatchString(t.Name) {
			return nil, fmt.Errorf("tool name must match regex: %s", toolNameRegex)
		}
		inputSchema := t.InputSchema
		if len(inputSchema) == 0 {
			inputSchema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		strict := true
		if v, ok := t.Metadata["strict"].(bool); ok {
			strict = v
		}
		if strict {
			inputSchema = pluginjsonschema.EnforceStrict(inputSchema)
		}
		resp = append(resp, responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        t.Name,
				Description: oa.String(t.Description),
				Parameters:  inputSchema,
				Strict:      oa.Bool(strict),
			},
		})
	}
	return resp, nil
}

func toOpenAIToolChoice(choice ai.ToolChoice) responses.ResponseNewParamsToolChoiceUnion {
	switch choice {
	case ai.ToolChoiceRequired:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
		}
	case ai.ToolChoiceNone:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
		}
	default:
		return responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
		}
	}
}

func toGenkitResponse(resp *responses.Response) (*ai.ModelResponse, error) {
	r := &ai.ModelResponse{
		Raw: resp.RawJSON(),
		Usage: &ai.GenerationUsage{
			InputTokens:         int(resp.Usage.InputTokens),
			OutputTokens:        int(resp.Usage.OutputTokens),
			TotalTokens:         int(resp.Usage.TotalTokens),
			CachedContentTokens: int(resp.Usage.InputTokensDetails.CachedTokens),
			ThoughtsTokens:      int(resp.Usage.OutputTokensDetails.ReasoningTokens),
		},
	}

	switch resp.Status {
	case responses.ResponseStatusCompleted:
		r.FinishReason = ai.FinishReasonStop
	case responses.ResponseStatusIncomplete:
		switch resp.IncompleteDetails.Reason {
		case "max_output_tokens":
			r.FinishReason = ai.FinishReasonLength
		case "content_filter":
			r.FinishReason = ai.FinishReasonBlocked
		default:
			r.FinishReason = ai.FinishReasonOther
		}
	case responses.ResponseStatusFailed:
		r.FinishReason = ai.FinishReasonOther
	case responses.ResponseStatusCancelled:
		r.FinishReason = ai.FinishReasonInterrupted
	default:
		r.FinishReason = ai.FinishReasonUnknown
	}

	msg := &ai.Message{Role: ai.RoleModel}
	for _, item := range resp.Output {
		switch v := item.AsAny().(type) {
		case responses.ResponseOutputMessage:
			for _, content := range v.Content {
				switch c := content.AsAny().(type) {
				case responses.ResponseOutputText:
					msg.Content = append(msg.Content, ai.NewTextPart(c.Text))
				case responses.ResponseOutputRefusal:
					msg.Content = append(msg.Content, ai.NewTextPart(c.Refusal))
				}
			}
		case responses.ResponseFunctionToolCall:
			msg.Content = append(msg.Content, ai.NewToolRequestPart(&ai.ToolRequest{
				Ref:   firstNonEmpty(v.CallID, v.ID),
				Name:  v.Name,
				Input: decodeJSONArg(v.Arguments),
			}))
		case responses.ResponseReasoningItem:
			text := make([]string, 0, len(v.Summary))
			for _, s := range v.Summary {
				text = append(text, s.Text)
			}
			if len(text) > 0 {
				part := ai.NewReasoningPart(strings.Join(text, "\n"), nil)
				if v.ID != "" {
					part.Metadata["id"] = v.ID
				}
				if v.EncryptedContent != "" {
					part.Metadata["encrypted_content"] = v.EncryptedContent
				}
				msg.Content = append(msg.Content, part)
			}
		}
	}

	r.Message = msg
	return r, nil
}

func normalizeReasoningDelta(delta any) string {
	switch v := delta.(type) {
	case string:
		return v
	case map[string]any:
		if text, ok := v["text"].(string); ok {
			return text
		}
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			if text := normalizeReasoningDelta(item); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "")
	}
	return ""
}

func decodeJSONArg(raw string) any {
	var args any
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return map[string]any{"raw": raw}
	}
	return args
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func contentKey(itemID string, index int64) string {
	return fmt.Sprintf("%s:%d", itemID, index)
}

func summaryKey(itemID string, index int64) string {
	return fmt.Sprintf("%s:summary:%d", itemID, index)
}
