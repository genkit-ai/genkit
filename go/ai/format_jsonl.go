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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/base"
)

type jsonlFormatter struct{}

// Name returns the name of the formatter.
func (j jsonlFormatter) Name() string {
	return OutputFormatJSONL
}

// Handler returns a new formatter handler for the given schema.
func (j jsonlFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	if schema == nil || !base.ValidateIsJSONArray(schema) {
		return nil, core.NewError(core.INVALID_ARGUMENT, "schema must be an array of objects for JSONL format")
	}

	jsonBytes, err := json.Marshal(schema["items"])
	if err != nil {
		return nil, fmt.Errorf("error marshalling schema to JSONL: %w", err)
	}

	instructions := fmt.Sprintf("Output should be JSONL format, a sequence of JSON objects (one per line) separated by a newline '\\n' character. Each line should be a JSON object conforming to the following schema:\n\n```%s```", string(jsonBytes))

	handler := &jsonlHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Format:      OutputFormatJSONL,
			Schema:      schema,
			ContentType: "application/jsonl",
		},
	}

	return handler, nil
}

type jsonlHandler struct {
	instructions    string
	config          ModelOutputConfig
	accumulatedText string
	currentIndex    int
	cursor          int
}

// Instructions returns the instructions for the formatter.
func (j *jsonlHandler) Instructions() string {
	return j.instructions
}

// Config returns the output config for the formatter.
func (j *jsonlHandler) Config() ModelOutputConfig {
	return j.config
}

// ParseOutput parses the final message and returns the parsed array of objects.
func (j *jsonlHandler) ParseOutput(m *Message) (any, error) {
	var sb strings.Builder
	for _, part := range m.Content {
		if part.IsText() {
			sb.WriteString(part.Text)
		}
	}

	result, _, err := j.parseJSONL(sb.String(), 0, false)
	if err != nil {
		return nil, err
	}

	if j.config.Schema != nil {
		if err := base.ValidateValue(result, j.config.Schema); err != nil {
			return nil, err
		}
	}

	return result, nil
}

// ParseChunk processes a streaming chunk and returns parsed output.
func (j *jsonlHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	if chunk.Index != j.currentIndex {
		j.accumulatedText = ""
		j.currentIndex = chunk.Index
		j.cursor = 0
	}

	for _, part := range chunk.Content {
		if part.IsText() {
			j.accumulatedText += part.Text
		}
	}

	items, newCursor, err := j.parseJSONL(j.accumulatedText, j.cursor, true)
	if err != nil {
		return nil, err
	}
	j.cursor = newCursor
	return items, nil
}

// ParseMessage parses the message and returns the formatted message.
func (j *jsonlHandler) ParseMessage(m *Message) (*Message, error) {
	return m, nil
}

// parseJSONL parses JSONL starting from the cursor position.
// Returns the parsed items, the new cursor position, and any error.
func (j *jsonlHandler) parseJSONL(text string, cursor int, allowPartial bool) ([]any, int, error) {
	if text == "" || cursor >= len(text) {
		return nil, cursor, nil
	}

	results := []any{}
	remaining := text[cursor:]
	lines := strings.Split(remaining, "\n")
	currentPos := cursor

	for i, line := range lines {
		isLastLine := i == len(lines)-1
		lineLen := len(line)
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, "{") {
			var result any
			err := json.Unmarshal([]byte(trimmed), &result)
			if err != nil {
				if allowPartial && isLastLine {
					partialResult, partialErr := base.ParsePartialJSON(trimmed)
					if partialErr == nil && partialResult != nil {
						results = append(results, partialResult)
					}
					// Don't advance cursor for partial line.
					break
				}
				return nil, cursor, fmt.Errorf("invalid JSON on line %d: %w", i+1, err)
			}
			if result != nil {
				results = append(results, result)
			}
		}

		if !isLastLine {
			currentPos += lineLen + 1 // +1 for newline
		}
	}

	return results, currentPos, nil
}
