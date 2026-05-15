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
	"errors"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/internal/base"
)

type jsonFormatter struct{}

// Name returns the name of the formatter.
func (j jsonFormatter) Name() string {
	return OutputFormatJSON
}

// Handler returns a new formatter handler for the given schema.
func (j jsonFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	var instructions string
	if schema != nil {
		jsonBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("error marshalling schema to JSON: %w", err)
		}

		instructions = fmt.Sprintf("Output should be in JSON format and conform to the following schema:\n\n```%s```", string(jsonBytes))
	}

	handler := &jsonHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Constrained: true,
			Format:      OutputFormatJSON,
			Schema:      schema,
			ContentType: "application/json",
		},
	}

	return handler, nil
}

// jsonHandler is a handler for the JSON formatter.
type jsonHandler struct {
	instructions    string
	config          ModelOutputConfig
	accumulatedText string
	currentIndex    int
}

// Instructions returns the instructions for the formatter.
func (j *jsonHandler) Instructions() string {
	return j.instructions
}

// Config returns the output config for the formatter.
func (j *jsonHandler) Config() ModelOutputConfig {
	return j.config
}

// ParseOutput parses the final message and returns the parsed JSON value.
func (j *jsonHandler) ParseOutput(m *Message) (any, error) {
	result, err := j.parseJSON(m.Text())
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
func (j *jsonHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	if chunk.Index != j.currentIndex {
		j.accumulatedText = ""
		j.currentIndex = chunk.Index
	}

	for _, part := range chunk.Content {
		if part.IsText() {
			j.accumulatedText += part.Text
		}
	}

	return j.parseJSON(j.accumulatedText)
}

// ParseMessage parses the message and returns the formatted message.
func (j *jsonHandler) ParseMessage(m *Message) (*Message, error) {
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

	newParts := []*Part{}
	text := base.ExtractJSONFromMarkdown(accumulatedText.String())
	if text != "" {
		if j.config.Schema != nil {
			schemaBytes, err := json.Marshal(j.config.Schema)
			if err != nil {
				return nil, fmt.Errorf("expected schema is not valid: %w", err)
			}
			if err = base.ValidateRaw([]byte(text), schemaBytes); err != nil {
				return nil, err
			}
		} else {
			if !base.ValidJSON(text) {
				return nil, errors.New("message is not a valid JSON")
			}
		}
		newParts = append(newParts, NewJSONPart(text))
	}

	newParts = append(newParts, nonTextParts...)

	m.Content = newParts

	return m, nil
}

// parseJSON is the shared parsing logic used by both ParseOutput and ParseChunk.
func (j *jsonHandler) parseJSON(text string) (any, error) {
	if text == "" {
		return nil, nil
	}

	extracted := base.ExtractJSONFromMarkdown(text)
	if extracted == "" {
		return nil, nil
	}

	result, err := base.ExtractJSON(extracted)
	if err != nil {
		return nil, nil
	}

	return result, nil
}
