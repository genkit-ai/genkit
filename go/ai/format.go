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
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
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

// quoteStripper removes the single and double quotes models sometimes wrap enum
// values in. It runs on every streamed chunk, so it is built once here rather
// than per call.
var quoteStripper = strings.NewReplacer("'", "", `"`, "")

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
	DefineFormats(reg, DEFAULT_FORMATS...)
}

// DefineFormats registers each [Formatter] under the name returned by its Name
// method. It panics if a format with that name is already registered, which
// includes the built-in formats registered by [ConfigureFormats], so a format
// cannot be replaced once defined.
func DefineFormats(r api.Registry, formatters ...Formatter) {
	for _, f := range formatters {
		r.RegisterValue("/format/"+f.Name(), f)
	}
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
	return nil, status.Errorf(status.ErrInvalidArgument, "output format %q is invalid", format)
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

	if targetIndex == -1 {
		return messages
	}

	// Copy the target message and both slices so the caller's messages are never
	// mutated in place. Callers reuse their message slice across turns, so an
	// in-place append would leak the instructions into stored history.
	target := *messages[targetIndex]
	target.Content = append(slices.Clone(target.Content), part)
	out := slices.Clone(messages)
	out[targetIndex] = &target

	return out
}

// Every handler carries the same three things: the instructions it embeds in
// the prompt, the output config it puts on the request, and the text streamed
// so far for the turn it is parsing. baseHandler holds all three so a handler
// is only its own parsing.
type baseHandler struct {
	instructions string
	config       ModelOutputConfig
	text         string // Text accumulated for the current turn.
	index        int    // Index of that turn.
	cursor       int    // How much of text has already been emitted.
}

// Instructions returns the instructions for the formatter.
func (h *baseHandler) Instructions() string { return h.instructions }

// Config returns the output config for the formatter.
func (h *baseHandler) Config() ModelOutputConfig { return h.config }

// ParseMessage returns the message unchanged. Handlers that rewrite a message
// into parsed parts override it.
func (h *baseHandler) ParseMessage(m *Message) (*Message, error) { return m, nil }

// accumulate appends chunk's text to the turn in progress and returns the text
// so far. A chunk from a new turn resets the state first: a handler instance
// spans the whole request, and a tool loop starts a fresh message each turn.
func (h *baseHandler) accumulate(chunk *ModelResponseChunk) string {
	if chunk.Index != h.index {
		h.text, h.index, h.cursor = "", chunk.Index, 0
	}
	for _, part := range chunk.Content {
		if part.IsText() {
			h.text += part.Text
		}
	}
	return h.text
}

// splitTextParts separates a message's text, concatenated, from its other
// parts, which the handlers that rewrite a message carry through untouched.
func splitTextParts(m *Message) (text string, others []*Part) {
	var b strings.Builder
	for _, part := range m.Content {
		if part.IsText() {
			b.WriteString(part.Text)
		} else {
			others = append(others, part)
		}
	}
	return b.String(), others
}

// requireContent rejects the messages a handler cannot parse at all.
func requireContent(m *Message) error {
	if m == nil {
		return status.Errorf(status.ErrInvalidOutput, "message is empty")
	}
	if len(m.Content) == 0 {
		return status.Errorf(status.ErrInvalidOutput, "message has no content")
	}
	return nil
}

type textFormatter struct{}

// Name returns the name of the formatter.
func (t textFormatter) Name() string {
	return OutputFormatText
}

// Handler returns a new formatter handler for the given schema.
func (t textFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	return &textHandler{baseHandler{config: ModelOutputConfig{ContentType: "text/plain"}}}, nil
}

type textHandler struct{ baseHandler }

// ParseOutput parses the final message and returns the text content.
func (t *textHandler) ParseOutput(m *Message) (any, error) {
	return m.Text(), nil
}

// ParseChunk processes a streaming chunk and returns parsed output.
func (t *textHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	return t.accumulate(chunk), nil
}

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
			return nil, status.Errorf(status.ErrInvalidSchema, "marshalling schema to JSON: %w", err)
		}

		instructions = fmt.Sprintf("Output should be in JSON format and conform to the following schema:\n\n```%s```", string(jsonBytes))
	}

	return &jsonHandler{baseHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Constrained: true,
			Format:      OutputFormatJSON,
			Schema:      schema,
			ContentType: "application/json",
		},
	}}, nil
}

// jsonHandler is a handler for the JSON formatter.
type jsonHandler struct{ baseHandler }

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
	return j.parseJSON(j.accumulate(chunk))
}

// ParseMessage rewrites the message's text as a single JSON part, validated
// against the schema when there is one, and carries any non-text parts through.
func (j *jsonHandler) ParseMessage(m *Message) (*Message, error) {
	if err := requireContent(m); err != nil {
		return nil, err
	}
	rawText, nonTextParts := splitTextParts(m)

	newParts := []*Part{}
	text := base.ExtractJSONFromMarkdown(rawText)
	if text == "" && len(nonTextParts) == 0 {
		// Every part was empty text. Falling through would hand back a message
		// with no content at all, which is the one input this method rejects.
		return nil, status.Errorf(status.ErrInvalidOutput, "message has no content to parse as JSON")
	}
	if text != "" {
		if j.config.Schema != nil {
			schemaBytes, err := json.Marshal(j.config.Schema)
			if err != nil {
				return nil, status.Errorf(status.ErrInvalidSchema, "expected schema is not valid: %w", err)
			}
			if err = base.ValidateRaw([]byte(text), schemaBytes); err != nil {
				return nil, err
			}
		} else {
			if !base.ValidJSON(text) {
				return nil, status.Errorf(status.ErrInvalidOutput, "message is not a valid JSON")
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

type jsonlFormatter struct{}

// Name returns the name of the formatter.
func (j jsonlFormatter) Name() string {
	return OutputFormatJSONL
}

// Handler returns a new formatter handler for the given schema.
func (j jsonlFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	if schema == nil || !base.ValidateIsJSONArray(schema) {
		return nil, status.Errorf(status.ErrInvalidSchema, "schema must be an array of objects for JSONL format")
	}

	jsonBytes, err := json.Marshal(schema["items"])
	if err != nil {
		return nil, status.Errorf(status.ErrInvalidSchema, "marshalling schema to JSONL: %w", err)
	}

	instructions := fmt.Sprintf("Output should be JSONL format, a sequence of JSON objects (one per line) separated by a newline '\\n' character. Each line should be a JSON object conforming to the following schema:\n\n```%s```", string(jsonBytes))

	return &jsonlHandler{baseHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Format:      OutputFormatJSONL,
			Schema:      schema,
			ContentType: "application/jsonl",
		},
	}}, nil
}

type jsonlHandler struct{ baseHandler }

// ParseOutput parses the final message and returns the parsed array of objects.
func (j *jsonlHandler) ParseOutput(m *Message) (any, error) {
	text, _ := splitTextParts(m)
	result, _, err := j.parseJSONL(text, 0, false)
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
	items, cursor, err := j.parseJSONL(j.accumulate(chunk), j.cursor, true)
	if err != nil {
		return nil, err
	}
	j.cursor = cursor
	return items, nil
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
				return nil, cursor, status.Errorf(status.ErrInvalidOutput, "invalid JSON on line %d: %w", i+1, err)
			}
			if result != nil {
				results = append(results, result)
			}
			if isLastLine {
				// The line parsed as a complete object even though its
				// terminating newline has not arrived yet. Consume it now, or
				// the next chunk re-parses and re-emits the same object.
				currentPos += lineLen
			}
		}

		if !isLastLine {
			currentPos += lineLen + 1 // +1 for newline
		}
	}

	return results, currentPos, nil
}

type arrayFormatter struct{}

// Name returns the name of the formatter.
func (a arrayFormatter) Name() string {
	return OutputFormatArray
}

// Handler returns a new formatter handler for the given schema.
func (a arrayFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	if schema == nil || !base.ValidateIsJSONArray(schema) {
		return nil, status.Errorf(status.ErrInvalidSchema, "schema must be an array of objects for array format")
	}

	jsonBytes, err := json.Marshal(schema["items"])
	if err != nil {
		return nil, status.Errorf(status.ErrInvalidSchema, "marshalling schema to JSON: %w", err)
	}
	instructions := fmt.Sprintf("Output should be a JSON array conforming to the following schema:\n\n```%s```", string(jsonBytes))

	return &arrayHandler{baseHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Constrained: true,
			Format:      OutputFormatArray,
			Schema:      schema,
			ContentType: "application/json",
		},
	}}, nil
}

type arrayHandler struct{ baseHandler }

// ParseOutput parses the final message and returns the parsed array.
func (a *arrayHandler) ParseOutput(m *Message) (any, error) {
	result := base.ExtractItems(m.Text(), 0)
	return result.Items, nil
}

// ParseChunk processes a streaming chunk and returns parsed output.
func (a *arrayHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	result := base.ExtractItems(a.accumulate(chunk), a.cursor)
	a.cursor = result.Cursor
	return result.Items, nil
}

type enumFormatter struct{}

// Name returns the name of the formatter.
func (e enumFormatter) Name() string {
	return OutputFormatEnum
}

// Handler returns a new formatter handler for the given schema.
func (e enumFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	enums := objectEnums(schema)
	if schema == nil || len(enums) == 0 {
		return nil, status.Errorf(status.ErrInvalidSchema, "schema must be an object with an 'enum' property for enum format")
	}

	instructions := fmt.Sprintf("Output should be ONLY one of the following enum values. Do not output any additional information or add quotes.\n\n```%s```", strings.Join(enums, "\n"))

	return &enumHandler{
		baseHandler: baseHandler{
			instructions: instructions,
			config: ModelOutputConfig{
				Constrained: true,
				Format:      OutputFormatEnum,
				Schema:      schema,
				ContentType: "text/enum",
			},
		},
		enums: enums,
	}, nil
}

type enumHandler struct {
	baseHandler
	enums []string
}

// ParseOutput parses the final message and returns the enum value.
func (e *enumHandler) ParseOutput(m *Message) (any, error) {
	return e.parseEnum(m.Text())
}

// ParseChunk processes a streaming chunk and returns parsed output.
func (e *enumHandler) ParseChunk(chunk *ModelResponseChunk) (any, error) {
	// Ignore the error: a partial stream is not yet a whole enum value.
	enum, _ := e.parseEnum(e.accumulate(chunk))
	return enum, nil
}

// ParseMessage rewrites the message's text as the single enum value it names,
// and carries any non-text parts through.
func (e *enumHandler) ParseMessage(m *Message) (*Message, error) {
	if err := requireContent(m); err != nil {
		return nil, err
	}
	text, nonTextParts := splitTextParts(m)
	value, err := e.parseEnum(text)
	if err != nil {
		return nil, err
	}
	m.Content = append([]*Part{NewTextPart(value)}, nonTextParts...)
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
		return "", status.Errorf(status.ErrInvalidOutput, "message has no enum value, want one of: %s", strings.Join(e.enums, ", "))
	}

	trimmed := strings.TrimSpace(quoteStripper.Replace(text))

	if !slices.Contains(e.enums, trimmed) {
		return "", status.Errorf(status.ErrInvalidOutput, "message %s not in list of valid enums: %s", trimmed, strings.Join(e.enums, ", "))
	}

	return trimmed, nil
}
