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

type textFormatter struct{}

// Name returns the name of the formatter.
func (t textFormatter) Name() string {
	return OutputFormatText
}

// Handler returns a new formatter handler for the given schema.
func (t textFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	handler := &textHandler{
		config: ModelOutputConfig{
			ContentType: "text/plain",
		},
	}

	return handler, nil
}

type textHandler struct {
	instructions    string
	config          ModelOutputConfig
	accumulatedText string
	currentIndex    int
}

// Config returns the output config for the formatter.
func (t *textHandler) Config() ModelOutputConfig {
	return t.config
}

// Instructions returns the instructions for the formatter.
func (t *textHandler) Instructions() string {
	return t.instructions
}

// ParseOutput parses the final message and returns the text content.
func (t *textHandler) ParseOutput(m *Message) (any, error) {
	return m.Text(), nil
}

// ParseChunk processes a streaming chunk and returns parsed output.
func (t *textHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	if chunk.Index != t.currentIndex {
		t.accumulatedText = ""
		t.currentIndex = chunk.Index
	}

	for _, part := range chunk.Content {
		if part.IsText() {
			t.accumulatedText += part.Text
		}
	}

	return t.accumulatedText, nil
}

// ParseMessage parses the message and returns the formatted message.
func (t *textHandler) ParseMessage(m *Message) (*Message, error) {
	return m, nil
}
