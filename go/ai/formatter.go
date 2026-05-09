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
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
)

const (
	// OutputFormatText is the default format.
	OutputFormatText string = "text"
	// OutputFormatJSON is the legacy format for JSON content.
	// For streaming, each chunk represents the full object received up to that point.
	OutputFormatJSON string = "json"
	// OutputFormatJSONL is the format for JSONL content.
	// For streaming, each chunk represents new items since the last chunk.
	OutputFormatJSONL string = "jsonl"
	// OutputFormatMedia is the format for media content.
	OutputFormatMedia string = "media"
	// OutputFormatArray is the format for array content.
	// For streaming, each chunk represents new items since the last chunk.
	OutputFormatArray string = "array"
	// OutputFormatEnum is the format for enum content.
	// The value must be a string.
	OutputFormatEnum string = "enum"
)

// Default formats get automatically registered on registry init
var DEFAULT_FORMATS = []Formatter{
	textFormatter{},
	jsonFormatter{},
	jsonlFormatter{},
	arrayFormatter{},
	enumFormatter{},
}

// Formatter represents the Formatter interface.
type Formatter interface {
	// Name returns the name of the formatter.
	Name() string
	// Handler returns the handler for the formatter.
	Handler(schema map[string]any) (FormatHandler, error)
}

// FormatHandler is a handler for formatting messages.
// A new instance is created via [Formatter.Handler] for each request.
type FormatHandler interface {
	// ParseMessage parses the message and returns a new formatted message.
	//
	// Legacy: New format handlers should implement this as a no-op passthrough and implement [StreamingFormatHandler] instead.
	ParseMessage(message *Message) (*Message, error)
	// Instructions returns the formatter instructions to embed in the prompt.
	Instructions() string
	// Config returns the output config for the model request.
	Config() ModelOutputConfig
}

// StreamingFormatHandler is a handler for formatting messages that supports streaming.
// This interface must be implemented to be able to use [ModelResponse.Output] and [ModelResponseChunk.Output].
type StreamingFormatHandler interface {
	// ParseOutput parses the final output and returns parsed output.
	ParseOutput(message *Message) (any, error)
	// ParseChunk processes a streaming chunk and returns parsed output.
	// The handler maintains its own internal state. When the chunk's index changes, the state is reset for the new turn.
	// Returns parsed output, or nil if nothing can be parsed yet.
	ParseChunk(chunk *ModelResponseChunk) (any, error)
}

// ConfigureFormats registers default formats in the registry
func ConfigureFormats(reg api.Registry) {
	for _, format := range DEFAULT_FORMATS {
		DefineFormat(reg, "/format/"+format.Name(), format)
	}
}

// DefineFormat defines and registers a new [Formatter].
func DefineFormat(r api.Registry, name string, formatter Formatter) {
	r.RegisterValue(name, formatter)
}

// resolveFormat returns a [Formatter], either a default one or one from the registry.
func resolveFormat(reg api.Registry, schema map[string]any, format string) (Formatter, error) {
	var formatter any
	if format == "" {
		if schema != nil {
			format = OutputFormatJSON
		} else {
			format = OutputFormatText
		}
	}
	formatter = reg.LookupValue("/format/" + format)
	if f, ok := formatter.(Formatter); ok {
		return f, nil
	}
	return nil, core.NewError(core.INVALID_ARGUMENT, "output format %q is invalid", format)
}

// injectInstructions looks through the messages and injects formatting directives
func injectInstructions(messages []*Message, instructions string) []*Message {
	if instructions == "" {
		return messages
	}

	// bail out if an output part is already present
	for _, m := range messages {
		for _, p := range m.Content {
			if p.Metadata != nil && p.Metadata["purpose"] == "output" {
				return messages
			}
		}
	}

	part := NewTextPart(instructions)
	part.Metadata = map[string]any{"purpose": "output"}

	targetIndex := -1

	// First try to find a system message
	for i, m := range messages {
		if m.Role == RoleSystem {
			targetIndex = i
			break
		}
	}

	// If no system message, find the last user message
	if targetIndex == -1 {
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == RoleUser {
				targetIndex = i
				break
			}
		}
	}

	if targetIndex != -1 {
		messages[targetIndex].Content = append(messages[targetIndex].Content, part)
	}

	return messages
}
