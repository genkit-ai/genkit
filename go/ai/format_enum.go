// Copyright 2024 Google LLC
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

package ai

import (
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core"
)

type enumFormatter struct{}

// Name returns the name of the formatter.
func (e enumFormatter) Name() string {
	return OutputFormatEnum
}

// Handler returns a new formatter handler for the given schema.
func (e enumFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	enums := objectEnums(schema)
	if schema == nil || len(enums) == 0 {
		return nil, core.NewError(core.INVALID_ARGUMENT, "schema must be an object with an 'enum' property for enum format")
	}

	instructions := fmt.Sprintf("Output should be ONLY one of the following enum values. Do not output any additional information or add quotes.\n\n```%s```", strings.Join(enums, "\n"))

	handler := &enumHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Constrained: true,
			Format:      OutputFormatEnum,
			Schema:      schema,
			ContentType: "text/enum",
		},
		enums: enums,
	}

	return handler, nil
}

type enumHandler struct {
	instructions    string
	config          ModelOutputConfig
	enums           []string
	accumulatedText string
	currentIndex    int
}

// Instructions returns the instructions for the formatter.
func (e *enumHandler) Instructions() string {
	return e.instructions
}

// Config returns the output config for the formatter.
func (e *enumHandler) Config() ModelOutputConfig {
	return e.config
}

// ParseOutput parses the final message and returns the enum value.
func (e *enumHandler) ParseOutput(m *Message) (any, error) {
	return e.parseEnum(m.Text())
}

// ParseChunk processes a streaming chunk and returns parsed output.
func (e *enumHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	if chunk.Index != e.currentIndex {
		e.accumulatedText = ""
		e.currentIndex = chunk.Index
	}

	for _, part := range chunk.Content {
		if part.IsText() {
			e.accumulatedText += part.Text
		}
	}

	// Ignore error since we are doing best effort parsing.
	enum, _ := e.parseEnum(e.accumulatedText)

	return enum, nil
}

// ParseMessage parses the message and returns the formatted message.
func (e *enumHandler) ParseMessage(m *Message) (*Message, error) {
	if e.config.Format == OutputFormatEnum {
		if m == nil {
			return nil, errors.New("message is empty")
		}
		if len(m.Content) == 0 {
			return nil, errors.New("message has no content")
		}

		var nonTextParts []*Part
		accumulatedText := strings.Builder{}
		for _, part := range m.Content {
			if !part.IsText() {
				nonTextParts = append(nonTextParts, part)
			} else {
				accumulatedText.WriteString(part.Text)
			}
		}

		// replace single and double quotes
		re := regexp.MustCompile(`['"]`)
		clean := re.ReplaceAllString(accumulatedText.String(), "")

		// trim whitespace
		trimmed := strings.TrimSpace(clean)

		if !slices.Contains(e.enums, trimmed) {
			return nil, fmt.Errorf("message %s not in list of valid enums: %s", trimmed, strings.Join(e.enums, ", "))
		}

		newParts := []*Part{NewTextPart(trimmed)}
		newParts = append(newParts, nonTextParts...)

		m.Content = newParts
	}

	return m, nil
}

// Get enum strings from json schema.
// Supports both top-level enum (e.g. {"type": "string", "enum": ["a", "b"]})
// and nested property enum (e.g. {"properties": {"value": {"enum": ["a", "b"]}}}).
func objectEnums(schema map[string]any) []string {
	if enums := extractEnumStrings(schema["enum"]); len(enums) > 0 {
		return enums
	}

	if properties, ok := schema["properties"].(map[string]any); ok {
		for _, propValue := range properties {
			if propMap, ok := propValue.(map[string]any); ok {
				if enums := extractEnumStrings(propMap["enum"]); len(enums) > 0 {
					return enums
				}
			}
		}
	}

	return nil
}

// Extracts string values from an enum field, supporting both []any (from JSON) and []string (from Go code).
func extractEnumStrings(v any) []string {
	if v == nil {
		return nil
	}

	if strs, ok := v.([]string); ok {
		return strs
	}

	if slice, ok := v.([]any); ok {
		enums := make([]string, 0, len(slice))
		for _, val := range slice {
			if s, ok := val.(string); ok {
				enums = append(enums, s)
			}
		}
		return enums
	}

	return nil
}

// parseEnum is the shared parsing logic used by both ParseOutput and ParseChunk.
func (e *enumHandler) parseEnum(text string) (string, error) {
	if text == "" {
		return "", nil
	}

	re := regexp.MustCompile(`['"]`)
	clean := re.ReplaceAllString(text, "")
	trimmed := strings.TrimSpace(clean)

	if !slices.Contains(e.enums, trimmed) {
		return "", fmt.Errorf("message %s not in list of valid enums: %s", trimmed, strings.Join(e.enums, ", "))
	}

	return trimmed, nil
}
