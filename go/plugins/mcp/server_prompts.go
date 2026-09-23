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

package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/mcp"
)

func (s *GenkitMCPServer) registerPromptWithMCP(desc api.ActionDesc) {
	description := genkitPromptDescription(desc)
	arguments := genkitPromptArguments(desc)
	s.mcpServer.AddPrompt(mcp.Prompt{
		Name:        desc.Name,
		Description: description,
		Arguments:   arguments,
	}, func(ctx context.Context, request mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
		prompt := genkit.LookupPrompt(s.genkit, desc.Name)
		if prompt == nil {
			return nil, fmt.Errorf("genkit prompt %q is no longer registered", desc.Name)
		}

		for _, arg := range arguments {
			if _, ok := request.Params.Arguments[arg.Name]; arg.Required && !ok {
				return nil, fmt.Errorf("prompt %q requires argument %q", desc.Name, arg.Name)
			}
		}

		input, err := mcpPromptInput(desc.InputSchema, request.Params.Arguments)
		if err != nil {
			return nil, fmt.Errorf("invalid arguments for Genkit prompt %q: %w", desc.Name, err)
		}
		if defaults := genkitPromptDefaults(desc); isObjectPromptSchema(desc.InputSchema) && len(defaults) > 0 && len(request.Params.Arguments) > 0 {
			// MCP arguments are individual fields. Keep omitted fields from the
			// Genkit default while allowing callers to replace supplied fields.
			merged := make(map[string]any, len(defaults)+len(request.Params.Arguments))
			for name, value := range defaults {
				merged[name] = value
			}
			for name, value := range input.(map[string]any) {
				merged[name] = value
			}
			input = merged
		}
		rendered, err := prompt.Render(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("render Genkit prompt %q: %w", desc.Name, err)
		}
		if rendered == nil {
			return nil, fmt.Errorf("genkit prompt %q rendered no request", desc.Name)
		}

		promptMessages := rendered.Messages
		if !genkitHasOutputPart(promptMessages) {
			instructions, err := genkitOutputInstructions(s.genkit, rendered.Output)
			if err != nil {
				return nil, fmt.Errorf("convert Genkit prompt %q output: %w", desc.Name, err)
			}
			promptMessages = withGenkitOutputInstructions(promptMessages, instructions)
		}
		messages, err := genkitMessagesToMCP(promptMessages)
		if err != nil {
			return nil, fmt.Errorf("convert Genkit prompt %q: %w", desc.Name, err)
		}
		docs, err := genkitDocsToMCP(rendered.Docs)
		if err != nil {
			return nil, fmt.Errorf("convert Genkit prompt %q documents: %w", desc.Name, err)
		}
		messages = append(docs, messages...)
		if len(messages) == 0 {
			return nil, fmt.Errorf("genkit prompt %q rendered no messages", desc.Name)
		}
		return &mcp.GetPromptResult{Description: description, Messages: messages}, nil
	})
}

// MCP prompt arguments are strings. Convert them using the Genkit input schema
// before rendering typed prompt functions, while leaving unknown fields as text.
func mcpPromptInput(schema map[string]any, arguments map[string]string) (any, error) {
	if len(arguments) == 0 {
		return nil, nil
	}
	if !isObjectPromptSchema(schema) {
		value, ok := arguments["input"]
		if !ok || len(arguments) != 1 {
			return nil, fmt.Errorf("scalar prompts accept only the input argument")
		}
		return convertMCPPromptArgument(schema, value)
	}
	properties, _ := schema["properties"].(map[string]any)
	input := make(map[string]any, len(arguments))
	genericInput := hasGenericPromptInput(schema, properties)
	if genericInput {
		if raw, ok := arguments["inputJson"]; ok {
			objectSchema := map[string]any{
				"type":                 "object",
				"properties":           properties,
				"additionalProperties": schema["additionalProperties"],
			}
			parsed, err := convertMCPPromptArgument(objectSchema, raw)
			if err != nil {
				return nil, fmt.Errorf("argument %q: %w", "inputJson", err)
			}
			for name, value := range parsed.(map[string]any) {
				input[name] = value
			}
		}
	}
	for name, value := range arguments {
		if name == "inputJson" && genericInput {
			continue
		}
		property := promptPropertySchema(schema, properties, name)
		converted, err := convertMCPPromptArgument(property, value)
		if err != nil {
			return nil, fmt.Errorf("argument %q: %w", name, err)
		}
		input[name] = converted
	}
	return input, nil
}

func hasGenericPromptInput(schema, properties map[string]any) bool {
	_, named := properties["inputJson"]
	return !named && (len(properties) == 0 || schema["additionalProperties"] != nil)
}

func promptPropertySchema(schema, properties map[string]any, name string) map[string]any {
	if property, ok := properties[name].(map[string]any); ok {
		return property
	}
	additional, _ := schema["additionalProperties"].(map[string]any)
	return additional
}

func isObjectPromptSchema(schema map[string]any) bool {
	kind, _ := schema["type"].(string)
	return kind == "object" || kind == ""
}

func convertMCPPromptArgument(schema map[string]any, value string) (any, error) {
	kind, _ := schema["type"].(string)
	switch kind {
	case "integer":
		return parseMCPPromptInteger(value)
	case "number":
		converted, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return nil, fmt.Errorf("must be a number: %w", err)
		}
		return converted, nil
	case "boolean":
		converted, err := strconv.ParseBool(value)
		if err != nil {
			return nil, fmt.Errorf("must be a boolean: %w", err)
		}
		return converted, nil
	case "object", "array":
		decoder := json.NewDecoder(strings.NewReader(value))
		decoder.UseNumber()
		var converted any
		if err := decoder.Decode(&converted); err != nil {
			return nil, fmt.Errorf("must be JSON: %w", err)
		}
		if err := decoder.Decode(new(any)); err != io.EOF {
			return nil, fmt.Errorf("must contain exactly one JSON value")
		}
		if kind == "object" {
			if _, ok := converted.(map[string]any); !ok {
				return nil, fmt.Errorf("must be a JSON object")
			}
		} else if _, ok := converted.([]any); !ok {
			return nil, fmt.Errorf("must be a JSON array")
		}
		return normalizeMCPPromptJSON(converted, schema)
	default:
		return value, nil
	}
}

func parseMCPPromptInteger(value string) (any, error) {
	if !json.Valid([]byte(value)) {
		return nil, fmt.Errorf("%q is not a JSON integer", value)
	}
	rational, ok := new(big.Rat).SetString(value)
	if !ok || !rational.IsInt() {
		return nil, fmt.Errorf("%q is not a whole number", value)
	}
	integer := rational.Num()
	if integer.IsInt64() {
		return integer.Int64(), nil
	}
	if integer.IsUint64() {
		return integer.Uint64(), nil
	}
	return nil, fmt.Errorf("integer %q is outside the int64/uint64 range", value)
}

// Decode with UseNumber, then use the schema to retain integer precision in
// nested objects and arrays before the input reaches a typed prompt function.
func normalizeMCPPromptJSON(value any, schema map[string]any) (any, error) {
	switch value := value.(type) {
	case json.Number:
		if schema["type"] == "number" {
			return value.Float64()
		}
		if integer, err := parseMCPPromptInteger(string(value)); err == nil {
			return integer, nil
		} else if schema["type"] == "integer" {
			return nil, err
		}
		return value.Float64()
	case map[string]any:
		properties, _ := schema["properties"].(map[string]any)
		for name, item := range value {
			property := promptPropertySchema(schema, properties, name)
			converted, err := normalizeMCPPromptJSON(item, property)
			if err != nil {
				return nil, fmt.Errorf("field %q: %w", name, err)
			}
			value[name] = converted
		}
	case []any:
		itemSchema, _ := schema["items"].(map[string]any)
		for i, item := range value {
			converted, err := normalizeMCPPromptJSON(item, itemSchema)
			if err != nil {
				return nil, fmt.Errorf("item %d: %w", i, err)
			}
			value[i] = converted
		}
	}
	return value, nil
}

func genkitPromptDescription(desc api.ActionDesc) string {
	if prompt, ok := desc.Metadata["prompt"].(map[string]any); ok {
		if description, ok := prompt["description"].(string); ok && description != "" {
			return description
		}
	}
	return desc.Description
}

func genkitPromptDefaults(desc api.ActionDesc) map[string]any {
	if prompt, ok := desc.Metadata["prompt"].(map[string]any); ok {
		defaults, _ := prompt["defaultInput"].(map[string]any)
		return defaults
	}
	return nil
}

func genkitPromptArguments(desc api.ActionDesc) []mcp.PromptArgument {
	schema := desc.InputSchema
	if !isObjectPromptSchema(schema) {
		description, _ := schema["description"].(string)
		return []mcp.PromptArgument{{Name: "input", Description: description, Required: true}}
	}
	properties, _ := schema["properties"].(map[string]any)
	required := make(map[string]bool)
	switch names := schema["required"].(type) {
	case []string:
		for _, name := range names {
			required[name] = true
		}
	case []any:
		for _, raw := range names {
			if name, ok := raw.(string); ok {
				required[name] = true
			}
		}
	}

	arguments := make([]mcp.PromptArgument, 0, len(properties)+1)
	defaults := genkitPromptDefaults(desc)
	for name, raw := range properties {
		hasDefault := defaults[name] != nil
		arg := mcp.PromptArgument{Name: name, Required: required[name] && !hasDefault}
		if property, ok := raw.(map[string]any); ok {
			arg.Description, _ = property["description"].(string)
		}
		arguments = append(arguments, arg)
	}
	if hasGenericPromptInput(schema, properties) {
		arguments = append(arguments, mcp.PromptArgument{
			Name:        "inputJson",
			Description: "JSON object of prompt input fields",
		})
	}
	sort.Slice(arguments, func(i, j int) bool { return arguments[i].Name < arguments[j].Name })
	return arguments
}

func genkitMessagesToMCP(messages []*ai.Message) ([]mcp.PromptMessage, error) {
	var result []mcp.PromptMessage
	for _, message := range messages {
		if message == nil {
			continue
		}
		var role mcp.Role
		switch message.Role {
		case ai.RoleUser, ai.RoleSystem, ai.RoleTool:
			role = mcp.RoleUser
		case ai.RoleModel:
			role = mcp.RoleAssistant
		default:
			return nil, fmt.Errorf("unsupported message role %q", message.Role)
		}
		for _, part := range message.Content {
			if part == nil {
				continue
			}
			content, err := genkitPartToMCP(part)
			if err != nil {
				return nil, err
			}
			if message.Role == ai.RoleSystem || message.Role == ai.RoleTool {
				if text, ok := content.(mcp.TextContent); ok {
					if message.Role == ai.RoleSystem {
						text.Text = "System instructions:\n" + text.Text
					} else {
						text.Text = "Tool output:\n" + text.Text
					}
					content = text
				}
			}
			result = append(result, mcp.PromptMessage{Role: role, Content: content})
		}
	}
	return result, nil
}

func genkitDocsToMCP(docs []*ai.Document) ([]mcp.PromptMessage, error) {
	var messages []mcp.PromptMessage
	for index, doc := range docs {
		if doc == nil {
			continue
		}
		header := fmt.Sprintf("Context document %d:", index+1)
		if len(doc.Metadata) > 0 {
			metadata, err := json.Marshal(doc.Metadata)
			if err != nil {
				return nil, fmt.Errorf("document %d metadata: %w", index+1, err)
			}
			header += "\nMetadata: " + string(metadata)
		}
		messages = append(messages, mcp.PromptMessage{Role: mcp.RoleUser, Content: mcp.NewTextContent(header)})
		for _, part := range doc.Content {
			if part == nil {
				continue
			}
			content, err := genkitPartToMCP(part)
			if err != nil {
				return nil, fmt.Errorf("document %d: %w", index+1, err)
			}
			messages = append(messages, mcp.PromptMessage{Role: mcp.RoleUser, Content: content})
		}
	}
	return messages, nil
}

func genkitHasOutputPart(messages []*ai.Message) bool {
	for _, message := range messages {
		if message == nil {
			continue
		}
		for _, part := range message.Content {
			if part != nil && part.Metadata["purpose"] == "output" {
				return true
			}
		}
	}
	return false
}

// Put formatter guidance in an existing system or user turn, matching Genkit's
// generation path. Copy the message and part so Render's output is untouched.
func withGenkitOutputInstructions(messages []*ai.Message, instructions string) []*ai.Message {
	if instructions == "" {
		return messages
	}
	targetIndex := -1
	for i, message := range messages {
		if message != nil && message.Role == ai.RoleSystem {
			targetIndex = i
			break
		}
	}
	if targetIndex == -1 {
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i] != nil && messages[i].Role == ai.RoleUser {
				targetIndex = i
				break
			}
		}
	}
	if targetIndex == -1 {
		return messages
	}
	out := slices.Clone(messages)
	target := messages[targetIndex].Clone()
	guidance := "Output instructions:\n" + instructions
	for i := len(target.Content) - 1; i >= 0; i-- {
		if target.Content[i] != nil && target.Content[i].IsText() {
			part := target.Content[i].Clone()
			part.Text += "\n\n" + guidance
			target.Content[i] = part
			out[targetIndex] = target
			return out
		}
	}
	target.Content = append(target.Content, ai.NewTextPart(guidance))
	out[targetIndex] = target
	return out
}

func genkitOutputInstructions(g *genkit.Genkit, output *ai.GenerateActionOutputConfig) (string, error) {
	if output == nil {
		return "", nil
	}
	if output.Instructions != nil {
		return *output.Instructions, nil
	}
	format := output.Format
	if format == "" {
		if output.JsonSchema != nil {
			format = ai.OutputFormatJSON
		} else {
			format = ai.OutputFormatText
		}
	}
	formatter, ok := genkit.LookupValue(g, "/format/"+format).(ai.Formatter)
	if !ok {
		return "", fmt.Errorf("output format %q is not registered", format)
	}
	handler, err := formatter.Handler(output.JsonSchema)
	if err != nil {
		return "", err
	}
	return handler.Instructions(), nil
}

func genkitPartToMCP(part *ai.Part) (mcp.Content, error) {
	switch part.Kind {
	case ai.PartText:
		return mcp.NewTextContent(part.Text), nil
	case ai.PartData:
		data, err := json.Marshal(part.Data)
		if err != nil {
			return nil, fmt.Errorf("encode structured prompt part: %w", err)
		}
		return mcp.NewTextContent(string(data)), nil
	case ai.PartToolRequest, ai.PartToolResponse, ai.PartCustom, ai.PartReasoning:
		data, err := json.Marshal(part)
		if err != nil {
			return nil, fmt.Errorf("encode prompt part: %w", err)
		}
		return mcp.NewTextContent(string(data)), nil
	case ai.PartResource:
		if part.Resource == nil {
			return nil, fmt.Errorf("prompt resource part has no URI")
		}
		return mcp.NewTextContent(part.Resource.Uri), nil
	case ai.PartMedia:
		if scheme, _, found := strings.Cut(part.Text, "://"); found &&
			(strings.EqualFold(scheme, "http") || strings.EqualFold(scheme, "https") || strings.EqualFold(scheme, "gs")) {
			return nil, fmt.Errorf("remote prompt media %q cannot be embedded in an MCP prompt", part.Text)
		}
		mimeType, data := part.ContentType, part.Text
		if encoded, ok := strings.CutPrefix(data, "data:"); ok {
			header, payload, found := strings.Cut(encoded, ",")
			if !found || !strings.HasSuffix(header, ";base64") {
				return nil, fmt.Errorf("prompt media must use a base64 data URI")
			}
			if mimeType == "" {
				mimeType = strings.TrimSuffix(header, ";base64")
			}
			data = payload
		}
		if _, err := base64.StdEncoding.DecodeString(data); err != nil {
			return nil, fmt.Errorf("decode prompt media: %w", err)
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			return mcp.NewImageContent(data, mimeType), nil
		case strings.HasPrefix(mimeType, "audio/"):
			return mcp.NewAudioContent(data, mimeType), nil
		default:
			return nil, fmt.Errorf("unsupported prompt media type %q", mimeType)
		}
	default:
		return nil, fmt.Errorf("unsupported prompt part kind %d", part.Kind)
	}
}
