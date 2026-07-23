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
	"regexp"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

const (
	// OutputFormatText is the default format.
	OutputFormatText string = "text"
	// OutputFormatJSON is the format for JSON content.
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

// defaultFormats are automatically registered on registry init.
var defaultFormats = []Formatter{
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

// FormatHandler parses model output for a single request.
// A new instance is created via [Formatter.Handler] for each request.
// Handlers must not modify the messages or chunks they are given; they only
// read them and return the parsed output.
type FormatHandler interface {
	// Instructions returns the formatter instructions to embed in the prompt.
	Instructions() string
	// Config returns the output config for the model request.
	Config() ModelOutputConfig
	// ParseOutput parses the final output and returns parsed output.
	ParseOutput(message *Message) (any, error)
	// ParseChunk processes a streaming chunk and returns parsed output.
	// The handler maintains its own internal state. When the chunk's index changes, the state is reset for the new turn.
	// Returns parsed output, or nil if nothing can be parsed yet.
	ParseChunk(chunk *ModelResponseChunk) (any, error)
}

// ConfigureFormats registers default formats in the registry
func ConfigureFormats(reg api.Registry) {
	for _, format := range defaultFormats {
		DefineFormat(reg, format)
	}
}

// DefineFormat registers a [Formatter] under the name returned by [Formatter.Name].
func DefineFormat(r api.Registry, formatter Formatter) {
	r.RegisterValue("/format/"+formatter.Name(), formatter)
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

// injectInstructions returns the messages with formatting instructions added.
// The input messages are not modified; the message that receives the
// instructions is shallow-copied with a new content slice.
func injectInstructions(messages []*Message, instructions string) []*Message {
	if instructions == "" {
		return messages
	}

	// bail out if an output part is already present
	for _, m := range messages {
		for _, p := range m.Content {
			if p.Metadata != nil && p.Metadata[PartMetaPurpose] == PartPurposeOutput {
				return messages
			}
		}
	}

	part := NewTextPart(instructions)
	part.Metadata = map[string]any{PartMetaPurpose: PartPurposeOutput}

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

	result := make([]*Message, len(messages))
	copy(result, messages)

	target := *messages[targetIndex]
	target.Content = append(slices.Clone(target.Content), part)
	result[targetIndex] = &target

	return result
}

// textFormatter is the default formatter and returns the raw text.
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

// jsonFormatter parses a single JSON object from the output.
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

// jsonlFormatter parses a sequence of newline-separated JSON objects from the output.
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

// arrayFormatter parses a JSON array from the output.
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

// enumFormatter parses a single enum value from the output.
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
