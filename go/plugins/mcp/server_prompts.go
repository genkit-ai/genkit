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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/mark3labs/mcp-go/mcp"
)

// registerPromptWithMCP exposes a Genkit prompt as an MCP prompt that clients
// render with prompts/get. Rendering does not call a model. As in the
// JavaScript server, the prompt's input schema must describe an object, whose
// properties become the MCP prompt arguments.
func (s *GenkitMCPServer) registerPromptWithMCP(desc api.ActionDesc) error {
	schema, err := normalizeJSONSchema(desc.InputSchema)
	if err != nil {
		return fmt.Errorf("encode input schema: %w", err)
	}
	if kind, ok := schema["type"]; ok && kind != "object" {
		return fmt.Errorf("MCP prompts need an object input schema, not type %v", kind)
	}
	properties, _ := schema["properties"].(map[string]any)
	defaults := genkitPromptDefaults(desc)
	arguments := genkitPromptArguments(schema, defaults)

	s.mcpServer.AddPrompt(mcp.Prompt{
		Name:        desc.Name,
		Description: desc.Description,
		Arguments:   arguments,
	}, func(ctx context.Context, request mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
		for _, arg := range arguments {
			if _, ok := request.Params.Arguments[arg.Name]; arg.Required && !ok {
				return nil, fmt.Errorf("prompt %q requires argument %q", desc.Name, arg.Name)
			}
		}
		// Each MCP argument is one input field, so supplied arguments replace
		// the matching fields of the default input and keep the others.
		input := maps.Clone(defaults)
		if input == nil {
			input = map[string]any{}
		}
		for name, value := range request.Params.Arguments {
			property, _ := properties[name].(map[string]any)
			converted, err := convertPromptArgument(property, value)
			if err != nil {
				return nil, fmt.Errorf("prompt %q argument %q: %w", desc.Name, name, err)
			}
			input[name] = converted
		}

		prompt := genkit.LookupPrompt(s.genkit, desc.Name)
		if prompt == nil {
			return nil, fmt.Errorf("prompt %q is no longer registered", desc.Name)
		}
		rendered, err := prompt.Render(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("render prompt %q: %w", desc.Name, err)
		}
		messages, err := genkitMessagesToMCP(rendered.Messages)
		if err != nil {
			return nil, fmt.Errorf("convert prompt %q: %w", desc.Name, err)
		}
		return &mcp.GetPromptResult{Description: desc.Description, Messages: messages}, nil
	})
	return nil
}

// normalizeJSONSchema returns schema in its JSON form. A schema built in Go
// can hold typed values, such as map[string]map[string]any properties or a
// []string required list, that plain JSON type assertions would miss.
func normalizeJSONSchema(schema map[string]any) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	var normalized map[string]any
	if err := json.Unmarshal(data, &normalized); err != nil {
		return nil, err
	}
	return normalized, nil
}

func genkitPromptDefaults(desc api.ActionDesc) map[string]any {
	prompt, _ := desc.Metadata["prompt"].(map[string]any)
	defaults, _ := prompt["defaultInput"].(map[string]any)
	return defaults
}

// genkitPromptArguments lists the properties of a normalized object schema as
// MCP prompt arguments, sorted by name. A required property is optional only
// when its default is not a zero value: ai.WithInputType stores its whole
// value as the default, so the zero struct T{} sets no real defaults.
func genkitPromptArguments(schema, defaults map[string]any) []mcp.PromptArgument {
	properties, _ := schema["properties"].(map[string]any)
	required, _ := schema["required"].([]any)
	arguments := make([]mcp.PromptArgument, 0, len(properties))
	for name, raw := range properties {
		arg := mcp.PromptArgument{Name: name, Required: slices.Contains(required, any(name)) && isZeroValue(defaults[name])}
		if property, ok := raw.(map[string]any); ok {
			arg.Description, _ = property["description"].(string)
		}
		arguments = append(arguments, arg)
	}
	slices.SortFunc(arguments, func(a, b mcp.PromptArgument) int { return strings.Compare(a.Name, b.Name) })
	return arguments
}

func isZeroValue(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	if v.Kind() == reflect.Map || v.Kind() == reflect.Slice {
		return v.Len() == 0
	}
	return v.IsZero()
}

// convertPromptArgument converts an MCP prompt argument, which is always a
// string, to the type its property schema names. A property without exactly
// one type keeps the string; prompt.Render validates the input either way.
func convertPromptArgument(property map[string]any, value string) (any, error) {
	switch kind, _ := property["type"].(string); kind {
	case "integer":
		return strconv.ParseInt(value, 10, 64)
	case "number":
		return strconv.ParseFloat(value, 64)
	case "boolean":
		return strconv.ParseBool(value)
	case "object", "array":
		// Keep numbers as json.Number so large integers keep their digits.
		decoder := json.NewDecoder(strings.NewReader(value))
		decoder.UseNumber()
		var converted any
		if err := decoder.Decode(&converted); err != nil {
			return nil, fmt.Errorf("not a JSON %s: %w", kind, err)
		}
		if decoder.Decode(new(any)) != io.EOF {
			return nil, fmt.Errorf("not a single JSON %s", kind)
		}
		return converted, nil
	}
	return value, nil
}

// genkitMessagesToMCP converts rendered prompt messages to MCP prompt
// messages, one per part. MCP prompts have only user and assistant roles, so
// the parts of a system message become user messages, the first of them
// starting with "System instructions:". When that first part is not text, a
// text message holding only the prefix comes before it.
func genkitMessagesToMCP(messages []*ai.Message) ([]mcp.PromptMessage, error) {
	const systemPrefix = "System instructions:"
	result := []mcp.PromptMessage{} // MCP requires an array, even when empty.
	for _, message := range messages {
		if message == nil {
			continue
		}
		var role mcp.Role
		switch message.Role {
		case ai.RoleUser, ai.RoleSystem:
			role = mcp.RoleUser
		case ai.RoleModel:
			role = mcp.RoleAssistant
		default:
			return nil, fmt.Errorf("MCP prompts do not support role %q", message.Role)
		}
		prefix := message.Role == ai.RoleSystem
		for _, part := range message.Content {
			if part == nil {
				continue
			}
			content, err := genkitPartToMCP(part)
			if err != nil {
				return nil, err
			}
			if prefix {
				prefix = false
				if text, ok := content.(mcp.TextContent); ok {
					text.Text = systemPrefix + "\n" + text.Text
					content = text
				} else {
					result = append(result, mcp.PromptMessage{Role: role, Content: mcp.NewTextContent(systemPrefix)})
				}
			}
			result = append(result, mcp.PromptMessage{Role: role, Content: content})
		}
	}
	return result, nil
}

// genkitPartToMCP converts a rendered prompt part with partToMCP, which also
// converts tool results. Unlike a tool result, a prompt fails, as in the
// JavaScript server, on a part MCP cannot carry: tool requests and responses,
// custom, reasoning and resource parts, media given by URL or by a data URI
// that is not base64-encoded, media other than images and audio, and media
// without data.
func genkitPartToMCP(part *ai.Part) (mcp.Content, error) {
	if rest, ok := strings.CutPrefix(part.Text, "data:"); ok && part.IsMedia() {
		if header, _, _ := strings.Cut(rest, ","); !strings.HasSuffix(header, ";base64") {
			return nil, errors.New("media data URI must be base64-encoded")
		}
	}
	content, err := partToMCP(part)
	empty := part.IsMedia() && part.Text == ""
	switch c := content.(type) {
	case mcp.ImageContent:
		empty = c.Data == ""
	case mcp.AudioContent:
		empty = c.Data == ""
	}
	if empty {
		return nil, errors.New("media part has no data")
	}
	if err != nil {
		return nil, fmt.Errorf("%s part: %w", part.Kind, err)
	}
	return content, nil
}
