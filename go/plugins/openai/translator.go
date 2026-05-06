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

package openai

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
)

// toOpenAIResponseParams translates an [ai.ModelRequest] into [responses.ResponseNewParams]
func toOpenAIResponseParams(model string, input *ai.ModelRequest) (*responses.ResponseNewParams, error) {
	params, err := configFromRequest(input.Config)
	if err != nil {
		return nil, err
	}
	if params == nil {
		params = &responses.ResponseNewParams{}
	}

	params.Model = shared.ResponsesModel(model)

	// Handle output format
	params.Text = handleOutputFormat(params.Text, input.Output)

	// Handle tools
	if len(input.Tools) > 0 {

		tools, err := toOpenAITools(input.Tools)
		if err != nil {
			return nil, err
		}
		// Append user tools to any existing tools (e.g. built-in tools provided in config)
		params.Tools = append(params.Tools, tools...)

		switch input.ToolChoice {
		case ai.ToolChoiceAuto, "":
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions("auto")),
			}
		case ai.ToolChoiceRequired:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions("required")),
			}
		case ai.ToolChoiceNone:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions("none")),
			}
		default:
			params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
				OfFunctionTool: &responses.ToolChoiceFunctionParam{
					Name: string(input.ToolChoice),
				},
			}
		}
	}

	// messages to input items
	var inputItems []responses.ResponseInputItemUnionParam

	for _, m := range input.Messages {
		items, err := toOpenAIInputItems(m)
		if err != nil {
			return nil, err
		}
		inputItems = append(inputItems, items...)
	}

	if len(inputItems) > 0 {
		params.Input = responses.ResponseNewParamsInputUnion{
			OfInputItemList: inputItems,
		}
	}
	return params, nil
}

// translateResponse translates an [responses.Response] into an [ai.ModelResponse]
func translateResponse(r *responses.Response) (*ai.ModelResponse, error) {
	resp := &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: make([]*ai.Part, 0),
		},
	}

	resp.Usage = &ai.GenerationUsage{
		InputTokens:         int(r.Usage.InputTokens),
		OutputTokens:        int(r.Usage.OutputTokens),
		CachedContentTokens: int(r.Usage.InputTokensDetails.CachedTokens),
		ThoughtsTokens:      int(r.Usage.OutputTokensDetails.ReasoningTokens),
		TotalTokens:         int(r.Usage.TotalTokens),
	}

	switch r.Status {
	case responses.ResponseStatusCompleted:
		resp.FinishReason = ai.FinishReasonStop
	case responses.ResponseStatusIncomplete:
		resp.FinishReason = ai.FinishReasonLength
	case responses.ResponseStatusFailed, responses.ResponseStatusCancelled:
		resp.FinishReason = ai.FinishReasonOther
	default:
		resp.FinishReason = ai.FinishReasonUnknown
	}

	resp.Raw = r.JSON

	for _, item := range r.Output {
		if err := handleResponseItem(item, resp); err != nil {
			return nil, err
		}
	}

	return resp, nil
}

// handleResponseItem is the entry point to translate response items
func handleResponseItem(item responses.ResponseOutputItemUnion, resp *ai.ModelResponse) error {
	switch v := item.AsAny().(type) {
	case responses.ResponseOutputMessage:
		return handleOutputMessage(v, resp)
	case responses.ResponseReasoningItem:
		return handleReasoningItem(v, resp)
	case responses.ResponseFunctionToolCall:
		return handleFunctionToolCall(v, resp)
	case responses.ResponseFunctionWebSearch:
		return handleWebSearchResponse(v, resp)
	default:
		return fmt.Errorf("unsupported response item type: %T", v)
	}
}

// handleOutputMessage translates a [responses.ResponseOutputMessage] into an [ai.ModelResponse]
// and appends the content into the provided response message.
func handleOutputMessage(msg responses.ResponseOutputMessage, resp *ai.ModelResponse) error {
	for _, content := range msg.Content {
		switch c := content.AsAny().(type) {
		case responses.ResponseOutputText:
			resp.Message.Content = append(resp.Message.Content, ai.NewTextPart(c.Text))
		case responses.ResponseOutputRefusal:
			resp.FinishMessage = c.Refusal
			resp.FinishReason = ai.FinishReasonBlocked
		}
	}
	return nil
}

// handleReasoningItem translates a [responses.ResponseReasoningItem] into an [ai.ModelResponse]
// and appends the content into the provided response message.
func handleReasoningItem(item responses.ResponseReasoningItem, resp *ai.ModelResponse) error {
	for _, content := range item.Content {
		resp.Message.Content = append(resp.Message.Content, ai.NewReasoningPart(content.Text, nil))
	}
	return nil
}

// handleFunctionToolCall translates a [responses.ResponseFunctionToolCall] into an [ai.ModelResponse]
// and appends the content into the provided response message.
func handleFunctionToolCall(call responses.ResponseFunctionToolCall, resp *ai.ModelResponse) error {
	args, err := jsonStringToMap(call.Arguments)
	if err != nil {
		return fmt.Errorf("could not parse tool args: %w", err)
	}
	resp.Message.Content = append(resp.Message.Content, ai.NewToolRequestPart(&ai.ToolRequest{
		Ref:   call.CallID,
		Name:  call.Name,
		Input: args,
	}))
	return nil
}

// handleWebSearchResponse translates a [responses.ResponseFunctionWebSearch] into an [ai.ModelResponse]
// and appends the content into the provided response message.
func handleWebSearchResponse(webSearch responses.ResponseFunctionWebSearch, resp *ai.ModelResponse) error {
	resp.Message.Content = append(resp.Message.Content, ai.NewToolResponsePart(&ai.ToolResponse{
		Ref:  webSearch.ID,
		Name: string(webSearch.Type),
		Output: map[string]any{
			"query":   webSearch.Action.Query,
			"type":    webSearch.Action.Type,
			"sources": webSearch.Action.Sources,
		},
	}))
	return nil
}

// toOpenAIInputItems converts a Genkit message to OpenAI Input Items
func toOpenAIInputItems(m *ai.Message) ([]responses.ResponseInputItemUnionParam, error) {
	var items []responses.ResponseInputItemUnionParam
	var partsBuffer []*ai.Part

	// flush() converts a sequence of text and media parts into a single OpenAI Input Item.
	// Message roles taken in consideration:
	// Model (or Assistant): converted to [responses.ResponseOutputMessageContentUnionParam]
	// User/System: converted to [responses.ResponseInputContentUnionParam]
	//
	// This is needed for the Responses API since it forbids to use Input Items for assistant role messages
	flush := func() error {
		if len(partsBuffer) == 0 {
			return nil
		}

		if m.Role == ai.RoleModel {
			// conversation-history text messages that the model previously generated
			var content []responses.ResponseOutputMessageContentUnionParam
			for _, p := range partsBuffer {
				if p.IsText() {
					content = append(content, responses.ResponseOutputMessageContentUnionParam{
						OfOutputText: &responses.ResponseOutputTextParam{
							Text:        p.Text,
							Annotations: []responses.ResponseOutputTextAnnotationUnionParam{},
							Type:        constant.OutputText("output_text"),
						},
					})
				}
			}
			if len(content) > 0 {
				// we need a unique ID for the output message
				id := fmt.Sprintf("msg_%p", m)
				items = append(items, responses.ResponseInputItemParamOfOutputMessage(
					content,
					id,
					responses.ResponseOutputMessageStatusCompleted,
				))
			}
		} else {
			var content []responses.ResponseInputContentUnionParam
			for _, p := range partsBuffer {
				if p.IsText() {
					content = append(content, responses.ResponseInputContentParamOfInputText(p.Text))
				} else if p.IsImage() || p.IsMedia() {
					content = append(content, responses.ResponseInputContentUnionParam{
						OfInputImage: &responses.ResponseInputImageParam{
							ImageURL: param.NewOpt(p.Text),
						},
					})
				}
			}
			if len(content) > 0 {
				role := responses.EasyInputMessageRoleUser
				// prevent unexpected system messages being sent as User, use Developer role to
				// provide new "system" instructions during the conversation
				if m.Role == ai.RoleSystem {
					role = responses.EasyInputMessageRole("developer")
				}
				items = append(items, responses.ResponseInputItemParamOfMessage(
					responses.ResponseInputMessageContentListParam(content), role),
				)
			}
		}

		partsBuffer = nil
		return nil
	}

	for _, p := range m.Content {
		if p.IsText() || p.IsImage() || p.IsMedia() {
			partsBuffer = append(partsBuffer, p)
		} else if p.IsToolRequest() {
			if err := flush(); err != nil {
				return nil, err
			}
			args, err := anyToJSONString(p.ToolRequest.Input)
			if err != nil {
				return nil, err
			}
			ref := p.ToolRequest.Ref
			if ref == "" {
				ref = p.ToolRequest.Name
			}
			items = append(items, responses.ResponseInputItemParamOfFunctionCall(args, ref, p.ToolRequest.Name))
		} else if p.IsReasoning() {
			if err := flush(); err != nil {
				return nil, err
			}
			id := fmt.Sprintf("reasoning_%p", p)
			summary := []responses.ResponseReasoningItemSummaryParam{
				{
					Text: p.Text,
					Type: constant.SummaryText("summary_text"),
				},
			}
			items = append(items, responses.ResponseInputItemParamOfReasoning(id, summary))
		} else if p.IsToolResponse() {
			if err := flush(); err != nil {
				return nil, err
			}

			// handle built-in tools
			// TODO: consider adding support for more built-in tools
			if p.ToolResponse.Name == "web_search_call" {
				item, err := handleWebSearchCall(p.ToolResponse, p.ToolResponse.Ref)
				if err != nil {
					return nil, err
				}
				items = append(items, item)
				continue
			}

			output, err := anyToJSONString(p.ToolResponse.Output)
			if err != nil {
				return nil, err
			}
			ref := p.ToolResponse.Ref
			items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(ref, output))
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}

	return items, nil
}

// handleWebSearchCall handles built-in tool responses for the web_search tool
func handleWebSearchCall(toolResponse *ai.ToolResponse, ref string) (responses.ResponseInputItemUnionParam, error) {
	output, ok := toolResponse.Output.(map[string]any)
	if !ok {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("invalid output format for web_search_call: expected map[string]any")
	}

	actionType, _ := output["type"].(string)
	jsonBytes, err := json.Marshal(output)
	if err != nil {
		return responses.ResponseInputItemUnionParam{}, err
	}

	var item responses.ResponseInputItemUnionParam

	switch actionType {
	case "open_page":
		var openPageAction responses.ResponseFunctionWebSearchActionOpenPageParam
		if err := json.Unmarshal(jsonBytes, &openPageAction); err != nil {
			return responses.ResponseInputItemUnionParam{}, err
		}
		item = responses.ResponseInputItemParamOfWebSearchCall(
			openPageAction,
			ref,
			responses.ResponseFunctionWebSearchStatusCompleted,
		)
	case "find":
		var findAction responses.ResponseFunctionWebSearchActionFindParam
		if err := json.Unmarshal(jsonBytes, &findAction); err != nil {
			return responses.ResponseInputItemUnionParam{}, err
		}
		item = responses.ResponseInputItemParamOfWebSearchCall(
			findAction,
			ref,
			responses.ResponseFunctionWebSearchStatusCompleted,
		)
	default:
		var searchAction responses.ResponseFunctionWebSearchActionSearchParam
		if err := json.Unmarshal(jsonBytes, &searchAction); err != nil {
			return responses.ResponseInputItemUnionParam{}, err
		}
		item = responses.ResponseInputItemParamOfWebSearchCall(
			searchAction,
			ref,
			responses.ResponseFunctionWebSearchStatusCompleted,
		)
	}
	return item, nil
}

// toOpenAITools converts a slice of [ai.ToolDefinition] to [responses.ToolUnionParam]
func toOpenAITools(tools []*ai.ToolDefinition) ([]responses.ToolUnionParam, error) {
	var result []responses.ToolUnionParam
	for _, t := range tools {
		if t == nil || t.Name == "" {
			continue
		}
		result = append(result, responses.ToolParamOfFunction(t.Name, t.InputSchema, false))
	}
	return result, nil
}

// configFromRequest casts the given configuration into [responses.ResponseNewParams]
func configFromRequest(config any) (*responses.ResponseNewParams, error) {
	var openaiConfig responses.ResponseNewParams

	switch cfg := config.(type) {
	case responses.ResponseNewParams:
		openaiConfig = cfg
	case *responses.ResponseNewParams:
		openaiConfig = *cfg
	case map[string]any:
		if err := mapToStruct(cfg, &openaiConfig); err != nil {
			return nil, fmt.Errorf("failed to convert config to responses.ResponseNewParams: %w", err)
		}
	case nil:
		// empty but valid config
	default:
		return nil, fmt.Errorf("unexpected config type: %T", config)
	}
	return &openaiConfig, nil
}

// anyToJSONString converts a stream of bytes to a JSON string
func anyToJSONString(data any) (string, error) {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to marshal any to JSON string: %w", err)
	}
	return string(jsonBytes), nil
}

// mapToStruct converts the provided map into a given struct
func mapToStruct(m map[string]any, v any) error {
	jsonData, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return json.Unmarshal(jsonData, v)
}

// jsonStringToMap translates a JSON string into a map
func jsonStringToMap(jsonString string) (map[string]any, error) {
	var result map[string]any
	if err := json.Unmarshal([]byte(jsonString), &result); err != nil {
		return nil, fmt.Errorf("unmarshal failed to parse json string %s: %w", jsonString, err)
	}
	return result, nil
}

// configToMap converts a config struct to a map[string]any
func configToMap(config any) map[string]any {
	r := jsonschema.Reflector{
		DoNotReference:             true,
		AllowAdditionalProperties:  false,
		ExpandedStruct:             true,
		RequiredFromJSONSchemaTags: true,
	}

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

// handleOutputFormat determines whether to enable structured output or json_mode in the request
func handleOutputFormat(textConfig responses.ResponseTextConfigParam, output *ai.ModelOutputConfig) responses.ResponseTextConfigParam {
	if output == nil || output.Format != ai.OutputFormatJSON {
		return textConfig
	}

	if !output.Constrained || output.Schema == nil {
		return textConfig
	}

	// strict mode is used for latest gpt models
	name := "output_schema"
	// openai schemas require a name to be provided
	if title, ok := output.Schema["title"].(string); ok {
		name = title
	}

	textConfig.Format = responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Type:   constant.JSONSchema("json_schema"),
			Name:   sanitizeSchemaName(name),
			Strict: param.NewOpt(true),
			Schema: output.Schema,
		},
	}
	return textConfig
}

// sanitizeSchemaName ensures the schema name contains only alphanumeric characters, underscores, or dashes,
// replaces invalid characters with underscores (_) and makes sure is no longer than 64 characters.
func sanitizeSchemaName(name string) string {
	schemaNameRegex := regexp.MustCompile(`[^a-zA-Z0-9_-]+`)
	newName := schemaNameRegex.ReplaceAllString(name, "_")

	// do not return error, cut the string instead
	if len(newName) > 64 {
		return newName[:64]
	}
	if newName == "" {
		// schema name is a mandatory field
		return "output_schema"
	}
	return newName
}
