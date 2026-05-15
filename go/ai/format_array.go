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

	"github.com/firebase/genkit/go/internal/base"
)

type arrayFormatter struct{}

// Name returns the name of the formatter.
func (a arrayFormatter) Name() string {
	return OutputFormatArray
}

// Handler returns a new formatter handler for the given schema.
func (a arrayFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	if schema == nil || !base.ValidateIsJSONArray(schema) {
		return nil, fmt.Errorf("schema is not valid JSON array")
	}

	jsonBytes, err := json.Marshal(schema["items"])
	if err != nil {
		return nil, fmt.Errorf("error marshalling schema to JSON, must supply an 'array' schema type when using the 'array' parser format.: %w", err)
	}
	instructions := fmt.Sprintf("Output should be a JSON array conforming to the following schema:\n\n```%s```", string(jsonBytes))

	handler := &arrayHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Constrained: true,
			Format:      OutputFormatArray,
			Schema:      schema,
			ContentType: "application/json",
		},
	}

	return handler, nil
}

type arrayHandler struct {
	instructions    string
	config          ModelOutputConfig
	accumulatedText string
	currentIndex    int
	cursor          int
}

// Instructions returns the instructions for the formatter.
func (a *arrayHandler) Instructions() string {
	return a.instructions
}

// Config returns the output config for the formatter.
func (a *arrayHandler) Config() ModelOutputConfig {
	return a.config
}

// ParseOutput parses the final message and returns the parsed array.
func (a *arrayHandler) ParseOutput(m *Message) (any, error) {
	result := base.ExtractItems(m.Text(), 0)
	return result.Items, nil
}

// ParseChunk processes a streaming chunk and returns parsed output.
func (a *arrayHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	if chunk.Index != a.currentIndex {
		a.accumulatedText = ""
		a.currentIndex = chunk.Index
		a.cursor = 0
	}

	for _, part := range chunk.Content {
		if part.IsText() {
			a.accumulatedText += part.Text
		}
	}

	result := base.ExtractItems(a.accumulatedText, a.cursor)
	a.cursor = result.Cursor
	return result.Items, nil
}

// ParseMessage parses the message and returns the formatted message.
func (a *arrayHandler) ParseMessage(m *Message) (*Message, error) {
	return m, nil
}
