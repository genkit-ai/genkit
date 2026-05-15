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
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestTextFormatter(t *testing.T) {
	t.Run("handler config", func(t *testing.T) {
		handler, err := textFormatter{}.Handler(nil)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		config := handler.Config()
		if config.ContentType != "text/plain" {
			t.Errorf("ContentType = %q, want %q", config.ContentType, "text/plain")
		}
		if handler.Instructions() != "" {
			t.Errorf("Instructions() = %q, want empty", handler.Instructions())
		}
	})

	t.Run("ParseOutput returns text content", func(t *testing.T) {
		handler, _ := textFormatter{}.Handler(nil)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("Hello, world!")},
		}

		got, err := sfh.ParseOutput(msg)
		if err != nil {
			t.Fatalf("ParseOutput() error = %v", err)
		}
		if got != "Hello, world!" {
			t.Errorf("ParseOutput() = %v, want %q", got, "Hello, world!")
		}
	})

	t.Run("ParseChunk accumulates text across chunks", func(t *testing.T) {
		handler, _ := textFormatter{}.Handler(nil)
		sfh := handler.(StreamingFormatHandler)

		chunks := []string{"Hello", ", ", "world!"}
		var lastResult any

		for _, text := range chunks {
			chunk := &ModelResponseChunk{
				Content: []*Part{NewTextPart(text)},
				Index:   0,
			}
			got, err := sfh.ParseChunk(chunk)
			if err != nil {
				t.Fatalf("ParseChunk() error = %v", err)
			}
			lastResult = got
		}

		if lastResult != "Hello, world!" {
			t.Errorf("final ParseChunk() = %v, want %q", lastResult, "Hello, world!")
		}
	})

	t.Run("ParseChunk resets on index change", func(t *testing.T) {
		handler, _ := textFormatter{}.Handler(nil)
		sfh := handler.(StreamingFormatHandler)

		sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart("first turn")},
			Index:   0,
		})

		got, _ := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart("second turn")},
			Index:   1,
		})

		if got != "second turn" {
			t.Errorf("ParseChunk() after index change = %v, want %q", got, "second turn")
		}
	})
}

func TestJSONFormatter(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		"additionalProperties": false,
	}

	t.Run("handler config with schema", func(t *testing.T) {
		handler, err := jsonFormatter{}.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		config := handler.Config()
		if config.Format != OutputFormatJSON {
			t.Errorf("Format = %q, want %q", config.Format, OutputFormatJSON)
		}
		if config.ContentType != "application/json" {
			t.Errorf("ContentType = %q, want %q", config.ContentType, "application/json")
		}
		if !config.Constrained {
			t.Errorf("Constrained = false, want true")
		}
		if config.Schema == nil {
			t.Errorf("Schema = nil, want schema")
		}

		instructions := handler.Instructions()
		if !strings.Contains(instructions, "JSON format") {
			t.Errorf("Instructions() should mention JSON format")
		}
		if !strings.Contains(instructions, `"name"`) {
			t.Errorf("Instructions() should contain schema")
		}
	})

	t.Run("handler config without schema", func(t *testing.T) {
		handler, err := jsonFormatter{}.Handler(nil)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		if handler.Instructions() != "" {
			t.Errorf("Instructions() = %q, want empty when no schema", handler.Instructions())
		}
	})

	t.Run("ParseOutput extracts and validates JSON", func(t *testing.T) {
		handler, _ := jsonFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart(`{"name": "Alice", "age": 30}`)},
		}

		got, err := sfh.ParseOutput(msg)
		if err != nil {
			t.Fatalf("ParseOutput() error = %v", err)
		}

		want := map[string]any{"name": "Alice", "age": float64(30)}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ParseOutput() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ParseOutput extracts JSON from markdown", func(t *testing.T) {
		handler, _ := jsonFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("Here is the result:\n\n```json\n{\"name\": \"Bob\", \"age\": 25}\n```")},
		}

		got, err := sfh.ParseOutput(msg)
		if err != nil {
			t.Fatalf("ParseOutput() error = %v", err)
		}

		want := map[string]any{"name": "Bob", "age": float64(25)}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ParseOutput() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ParseOutput validates against schema", func(t *testing.T) {
		handler, _ := jsonFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart(`{"name": "Alice", "unknown_field": "bad"}`)},
		}

		_, err := sfh.ParseOutput(msg)
		if err == nil {
			t.Error("ParseOutput() should fail validation for unknown field")
		}
	})

	t.Run("ParseChunk handles partial JSON", func(t *testing.T) {
		handler, _ := jsonFormatter{}.Handler(nil)
		sfh := handler.(StreamingFormatHandler)

		got, err := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart(`{"name": "Jo`)},
			Index:   0,
		})
		if err != nil {
			t.Fatalf("ParseChunk() error = %v", err)
		}

		gotMap, ok := got.(map[string]any)
		if !ok {
			t.Fatalf("ParseChunk() returned %T, want map[string]any", got)
		}
		if gotMap["name"] != "Jo" {
			t.Errorf("ParseChunk() name = %v, want %q", gotMap["name"], "Jo")
		}
	})

	t.Run("ParseChunk accumulates across chunks", func(t *testing.T) {
		handler, _ := jsonFormatter{}.Handler(nil)
		sfh := handler.(StreamingFormatHandler)

		sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart(`{"name"`)},
			Index:   0,
		})

		got, err := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart(`: "Alice", "age": 30}`)},
			Index:   0,
		})
		if err != nil {
			t.Fatalf("ParseChunk() error = %v", err)
		}

		want := map[string]any{"name": "Alice", "age": float64(30)}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ParseChunk() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ParseMessage validates JSON and creates JSON part", func(t *testing.T) {
		handler, _ := jsonFormatter{}.Handler(schema)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart(`{"name": "Alice", "age": 30}`)},
		}

		got, err := handler.ParseMessage(msg)
		if err != nil {
			t.Fatalf("ParseMessage() error = %v", err)
		}

		if len(got.Content) != 1 {
			t.Fatalf("ParseMessage() returned %d parts, want 1", len(got.Content))
		}
		if got.Content[0].ContentType != "application/json" {
			t.Errorf("Part ContentType = %q, want %q", got.Content[0].ContentType, "application/json")
		}
	})

	t.Run("ParseMessage fails on invalid JSON", func(t *testing.T) {
		handler, _ := jsonFormatter{}.Handler(nil)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("not valid json")},
		}

		_, err := handler.ParseMessage(msg)
		if err == nil {
			t.Error("ParseMessage() should fail for invalid JSON")
		}
	})
}

func TestJSONLFormatter(t *testing.T) {
	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id":   map[string]any{"type": "integer"},
				"name": map[string]any{"type": "string"},
			},
		},
	}

	t.Run("handler requires array schema", func(t *testing.T) {
		_, err := jsonlFormatter{}.Handler(nil)
		if err == nil {
			t.Error("Handler() should fail without schema")
		}

		_, err = jsonlFormatter{}.Handler(map[string]any{"type": "object"})
		if err == nil {
			t.Error("Handler() should fail with non-array schema")
		}
	})

	t.Run("handler config", func(t *testing.T) {
		handler, err := jsonlFormatter{}.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		config := handler.Config()
		if config.Format != OutputFormatJSONL {
			t.Errorf("Format = %q, want %q", config.Format, OutputFormatJSONL)
		}
		if config.ContentType != "application/jsonl" {
			t.Errorf("ContentType = %q, want %q", config.ContentType, "application/jsonl")
		}

		instructions := handler.Instructions()
		if !strings.Contains(instructions, "JSONL format") {
			t.Errorf("Instructions() should mention JSONL format")
		}
	})

	t.Run("ParseOutput parses multiple lines", func(t *testing.T) {
		handler, _ := jsonlFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("{\"id\": 1, \"name\": \"Alice\"}\n{\"id\": 2, \"name\": \"Bob\"}")},
		}

		got, err := sfh.ParseOutput(msg)
		if err != nil {
			t.Fatalf("ParseOutput() error = %v", err)
		}

		want := []any{
			map[string]any{"id": float64(1), "name": "Alice"},
			map[string]any{"id": float64(2), "name": "Bob"},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ParseOutput() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ParseChunk handles streaming JSONL", func(t *testing.T) {
		handler, _ := jsonlFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		got1, _ := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart("{\"id\": 1}\n")},
			Index:   0,
		})

		items1, ok := got1.([]any)
		if !ok || len(items1) != 1 {
			t.Fatalf("first ParseChunk() should return 1 item, got %v", got1)
		}

		got2, _ := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart("{\"id\": 2}")},
			Index:   0,
		})

		items2, ok := got2.([]any)
		if !ok || len(items2) != 1 {
			t.Fatalf("second ParseChunk() should return 1 new item (partial), got %v", got2)
		}
	})
}

func TestArrayFormatter(t *testing.T) {
	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{"type": "integer"},
			},
		},
	}

	t.Run("handler requires array schema", func(t *testing.T) {
		_, err := arrayFormatter{}.Handler(nil)
		if err == nil {
			t.Error("Handler() should fail without schema")
		}

		_, err = arrayFormatter{}.Handler(map[string]any{"type": "string"})
		if err == nil {
			t.Error("Handler() should fail with non-array schema")
		}
	})

	t.Run("handler config", func(t *testing.T) {
		handler, err := arrayFormatter{}.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		config := handler.Config()
		if config.Format != OutputFormatArray {
			t.Errorf("Format = %q, want %q", config.Format, OutputFormatArray)
		}
		if config.ContentType != "application/json" {
			t.Errorf("ContentType = %q, want %q", config.ContentType, "application/json")
		}
		if !config.Constrained {
			t.Error("Constrained = false, want true")
		}

		instructions := handler.Instructions()
		if !strings.Contains(instructions, "JSON array") {
			t.Errorf("Instructions() should mention JSON array")
		}
	})

	t.Run("ParseOutput extracts array items", func(t *testing.T) {
		handler, _ := arrayFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart(`[{"id": 1}, {"id": 2}]`)},
		}

		got, err := sfh.ParseOutput(msg)
		if err != nil {
			t.Fatalf("ParseOutput() error = %v", err)
		}

		want := []any{
			map[string]any{"id": float64(1)},
			map[string]any{"id": float64(2)},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ParseOutput() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ParseChunk handles streaming array", func(t *testing.T) {
		handler, _ := arrayFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart(`[{"id": 1},`)},
			Index:   0,
		})

		got, _ := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart(` {"id": 2}]`)},
			Index:   0,
		})

		items, ok := got.([]any)
		if !ok {
			t.Fatalf("ParseChunk() returned %T, want []any", got)
		}
		if len(items) != 1 {
			t.Errorf("ParseChunk() returned %d new items, want 1", len(items))
		}
	})
}

func TestEnumFormatter(t *testing.T) {
	schema := map[string]any{
		"type": "string",
		"enum": []any{"red", "green", "blue"},
	}

	nestedSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"color": map[string]any{
				"type": "string",
				"enum": []any{"red", "green", "blue"},
			},
		},
	}

	t.Run("handler requires schema with enum", func(t *testing.T) {
		_, err := enumFormatter{}.Handler(nil)
		if err == nil {
			t.Error("Handler() should fail without schema")
		}

		_, err = enumFormatter{}.Handler(map[string]any{"type": "string"})
		if err == nil {
			t.Error("Handler() should fail without enum property")
		}
	})

	t.Run("handler config with top-level enum", func(t *testing.T) {
		handler, err := enumFormatter{}.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		config := handler.Config()
		if config.Format != OutputFormatEnum {
			t.Errorf("Format = %q, want %q", config.Format, OutputFormatEnum)
		}
		if config.ContentType != "text/enum" {
			t.Errorf("ContentType = %q, want %q", config.ContentType, "text/enum")
		}
		if !config.Constrained {
			t.Error("Constrained = false, want true")
		}

		instructions := handler.Instructions()
		if !strings.Contains(instructions, "red") || !strings.Contains(instructions, "green") {
			t.Errorf("Instructions() should list enum values")
		}
	})

	t.Run("handler supports nested enum in properties", func(t *testing.T) {
		handler, err := enumFormatter{}.Handler(nestedSchema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		instructions := handler.Instructions()
		if !strings.Contains(instructions, "red") {
			t.Errorf("Instructions() should list enum values from nested property")
		}
	})

	t.Run("ParseOutput validates enum value", func(t *testing.T) {
		handler, _ := enumFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("green")},
		}

		got, err := sfh.ParseOutput(msg)
		if err != nil {
			t.Fatalf("ParseOutput() error = %v", err)
		}
		if got != "green" {
			t.Errorf("ParseOutput() = %v, want %q", got, "green")
		}
	})

	t.Run("ParseOutput strips quotes and whitespace", func(t *testing.T) {
		handler, _ := enumFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("  \"blue\"  \n")},
		}

		got, err := sfh.ParseOutput(msg)
		if err != nil {
			t.Fatalf("ParseOutput() error = %v", err)
		}
		if got != "blue" {
			t.Errorf("ParseOutput() = %v, want %q", got, "blue")
		}
	})

	t.Run("ParseOutput fails on invalid enum", func(t *testing.T) {
		handler, _ := enumFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("yellow")},
		}

		_, err := sfh.ParseOutput(msg)
		if err == nil {
			t.Error("ParseOutput() should fail for invalid enum value")
		}
	})

	t.Run("ParseChunk returns valid enum or empty", func(t *testing.T) {
		handler, _ := enumFormatter{}.Handler(schema)
		sfh := handler.(StreamingFormatHandler)

		got1, err := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart("re")},
			Index:   0,
		})
		if err != nil {
			t.Fatalf("ParseChunk() error = %v", err)
		}
		if got1 != "" {
			t.Errorf("ParseChunk() with partial = %v, want empty", got1)
		}

		got2, err := sfh.ParseChunk(&ModelResponseChunk{
			Content: []*Part{NewTextPart("d")},
			Index:   0,
		})
		if err != nil {
			t.Fatalf("ParseChunk() error = %v", err)
		}
		if got2 != "red" {
			t.Errorf("ParseChunk() with complete enum = %v, want %q", got2, "red")
		}
	})

	t.Run("ParseMessage validates and cleans enum", func(t *testing.T) {
		handler, _ := enumFormatter{}.Handler(schema)

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("  'green'\n")},
		}

		got, err := handler.ParseMessage(msg)
		if err != nil {
			t.Fatalf("ParseMessage() error = %v", err)
		}

		if got.Content[0].Text != "green" {
			t.Errorf("ParseMessage() text = %q, want %q", got.Content[0].Text, "green")
		}
	})

	t.Run("ParseMessage preserves non-text parts", func(t *testing.T) {
		handler, _ := enumFormatter{}.Handler(schema)

		toolPart := NewToolRequestPart(&ToolRequest{Name: "test"})
		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("red"), toolPart},
		}

		got, err := handler.ParseMessage(msg)
		if err != nil {
			t.Fatalf("ParseMessage() error = %v", err)
		}

		if len(got.Content) != 2 {
			t.Fatalf("ParseMessage() returned %d parts, want 2", len(got.Content))
		}
	})
}

func TestResolveFormat(t *testing.T) {
	t.Run("defaults to text when no format or schema", func(t *testing.T) {
		formatter, err := resolveFormat(r, nil, "")
		if err != nil {
			t.Fatalf("resolveFormat() error = %v", err)
		}
		if formatter.Name() != OutputFormatText {
			t.Errorf("resolveFormat() = %q, want %q", formatter.Name(), OutputFormatText)
		}
	})

	t.Run("defaults to json when schema present but no format", func(t *testing.T) {
		schema := map[string]any{"type": "object"}
		formatter, err := resolveFormat(r, schema, "")
		if err != nil {
			t.Fatalf("resolveFormat() error = %v", err)
		}
		if formatter.Name() != OutputFormatJSON {
			t.Errorf("resolveFormat() = %q, want %q", formatter.Name(), OutputFormatJSON)
		}
	})

	t.Run("uses explicit format", func(t *testing.T) {
		formatter, err := resolveFormat(r, nil, OutputFormatJSON)
		if err != nil {
			t.Fatalf("resolveFormat() error = %v", err)
		}
		if formatter.Name() != OutputFormatJSON {
			t.Errorf("resolveFormat() = %q, want %q", formatter.Name(), OutputFormatJSON)
		}
	})

	t.Run("returns error for unknown format", func(t *testing.T) {
		_, err := resolveFormat(r, nil, "unknown_format")
		if err == nil {
			t.Error("resolveFormat() should fail for unknown format")
		}
	})
}

func TestInjectInstructions(t *testing.T) {
	t.Run("empty instructions returns unchanged messages", func(t *testing.T) {
		msgs := []*Message{{Role: RoleUser, Content: []*Part{NewTextPart("hello")}}}
		result := injectInstructions(msgs, "")
		if len(result[0].Content) != 1 {
			t.Errorf("should not modify messages with empty instructions")
		}
	})

	t.Run("adds to system message if present", func(t *testing.T) {
		msgs := []*Message{
			{Role: RoleSystem, Content: []*Part{NewTextPart("system prompt")}},
			{Role: RoleUser, Content: []*Part{NewTextPart("user message")}},
		}

		result := injectInstructions(msgs, "output instructions")

		if len(result[0].Content) != 2 {
			t.Fatalf("system message should have 2 parts, got %d", len(result[0].Content))
		}
		if result[0].Content[1].Text != "output instructions" {
			t.Errorf("injected text = %q, want %q", result[0].Content[1].Text, "output instructions")
		}
		if result[0].Content[1].Metadata["purpose"] != "output" {
			t.Errorf("injected part should have purpose=output metadata")
		}
	})

	t.Run("adds to last user message if no system message", func(t *testing.T) {
		msgs := []*Message{
			{Role: RoleUser, Content: []*Part{NewTextPart("first user")}},
			{Role: RoleModel, Content: []*Part{NewTextPart("model response")}},
			{Role: RoleUser, Content: []*Part{NewTextPart("second user")}},
		}

		result := injectInstructions(msgs, "output instructions")

		if len(result[2].Content) != 2 {
			t.Fatalf("last user message should have 2 parts, got %d", len(result[2].Content))
		}
		if result[2].Content[1].Text != "output instructions" {
			t.Errorf("should inject into last user message")
		}
	})

	t.Run("does not inject if output part already exists", func(t *testing.T) {
		existingOutputPart := NewTextPart("existing output")
		existingOutputPart.Metadata = map[string]any{"purpose": "output"}

		msgs := []*Message{
			{Role: RoleUser, Content: []*Part{NewTextPart("user"), existingOutputPart}},
		}

		result := injectInstructions(msgs, "new instructions")

		if len(result[0].Content) != 2 {
			t.Errorf("should not add additional output part when one exists")
		}
	})
}

func TestObjectEnums(t *testing.T) {
	t.Run("extracts top-level enum", func(t *testing.T) {
		schema := map[string]any{
			"type": "string",
			"enum": []any{"a", "b", "c"},
		}

		enums := objectEnums(schema)
		want := []string{"a", "b", "c"}
		if diff := cmp.Diff(want, enums); diff != "" {
			t.Errorf("objectEnums() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("extracts enum from nested property", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"status": map[string]any{
					"type": "string",
					"enum": []any{"active", "inactive"},
				},
			},
		}

		enums := objectEnums(schema)
		want := []string{"active", "inactive"}
		if diff := cmp.Diff(want, enums); diff != "" {
			t.Errorf("objectEnums() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("handles []string enum values", func(t *testing.T) {
		schema := map[string]any{
			"type": "string",
			"enum": []string{"x", "y", "z"},
		}

		enums := objectEnums(schema)
		want := []string{"x", "y", "z"}
		if diff := cmp.Diff(want, enums); diff != "" {
			t.Errorf("objectEnums() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns nil for schema without enum", func(t *testing.T) {
		schema := map[string]any{"type": "string"}
		enums := objectEnums(schema)
		if enums != nil {
			t.Errorf("objectEnums() = %v, want nil", enums)
		}
	})
}

func TestStreamingFormatHandlerInterface(t *testing.T) {
	formatters := []Formatter{
		textFormatter{},
		jsonFormatter{},
		jsonlFormatter{},
		arrayFormatter{},
		enumFormatter{},
	}

	arraySchema := map[string]any{
		"type":  "array",
		"items": map[string]any{"type": "object"},
	}
	enumSchema := map[string]any{
		"type": "string",
		"enum": []any{"a", "b"},
	}

	for _, f := range formatters {
		t.Run(f.Name()+" implements StreamingFormatHandler", func(t *testing.T) {
			var schema map[string]any
			switch f.Name() {
			case OutputFormatJSONL, OutputFormatArray:
				schema = arraySchema
			case OutputFormatEnum:
				schema = enumSchema
			}

			handler, err := f.Handler(schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}

			if _, ok := handler.(StreamingFormatHandler); !ok {
				t.Errorf("%s handler does not implement StreamingFormatHandler", f.Name())
			}
		})
	}
}

func TestConstrainedGenerateWithFormats(t *testing.T) {
	type FooBar struct {
		Foo string `json:"foo"`
	}

	JSON := `{"foo": "bar"}`
	JSONmd := "```json" + JSON + "```"

	modelOpts := ModelOptions{
		Label: "formatModel",
		Supports: &ModelSupports{
			Multiturn:   true,
			Tools:       true,
			SystemRole:  true,
			Media:       false,
			Constrained: ConstrainedSupportAll,
		},
	}

	formatModel := DefineModel(r, "test/format", &modelOpts, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		return &ModelResponse{
			Request: gr,
			Message: NewModelTextMessage(JSONmd),
		}, nil
	})

	t.Run("native constrained mode sends schema to model", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithOutputType(FooBar{}),
		)
		if err != nil {
			t.Fatal(err)
		}

		if res.Request.Output == nil {
			t.Fatal("Output config should be set")
		}
		if !res.Request.Output.Constrained {
			t.Error("Constrained should be true for native mode")
		}
		if res.Request.Output.Schema == nil {
			t.Error("Schema should be set in output config")
		}
	})

	t.Run("simulated constrained mode injects instructions", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithOutputType(FooBar{}),
			WithCustomConstrainedOutput(),
		)
		if err != nil {
			t.Fatal(err)
		}

		if res.Request.Output.Constrained {
			t.Error("Constrained should be false for simulated mode")
		}

		var foundOutputPart bool
		for _, msg := range res.Request.Messages {
			for _, part := range msg.Content {
				if part.Metadata != nil && part.Metadata["purpose"] == "output" {
					foundOutputPart = true
					if !strings.Contains(part.Text, "JSON format") {
						t.Error("Output instructions should mention JSON format")
					}
				}
			}
		}
		if !foundOutputPart {
			t.Error("Should inject output instructions in simulated mode")
		}
	})

	t.Run("empty output instructions disables injection", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithOutputType(FooBar{}),
			WithOutputInstructions(""),
		)
		if err != nil {
			t.Fatal(err)
		}

		for _, msg := range res.Request.Messages {
			for _, part := range msg.Content {
				if part.Metadata != nil && part.Metadata["purpose"] == "output" {
					t.Error("Should not inject output part when instructions are empty string")
				}
			}
		}
		if res.Request.Output.Schema != nil {
			t.Error("Schema should not be set when using empty output instructions")
		}
	})

	t.Run("custom output instructions are used", func(t *testing.T) {
		customInstructions := "Please respond with valid JSON matching the schema"

		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithOutputType(FooBar{}),
			WithOutputInstructions(customInstructions),
		)
		if err != nil {
			t.Fatal(err)
		}

		var foundCustomInstructions bool
		for _, msg := range res.Request.Messages {
			for _, part := range msg.Content {
				if part.Text == customInstructions {
					foundCustomInstructions = true
				}
			}
		}
		if !foundCustomInstructions {
			t.Error("Custom output instructions should be injected")
		}
	})

	t.Run("parsed output is available", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithOutputType(FooBar{}),
		)
		if err != nil {
			t.Fatal(err)
		}

		var output FooBar
		err = res.Output(&output)
		if err != nil {
			t.Fatalf("Output() error = %v", err)
		}

		if output.Foo != "bar" {
			t.Errorf("Output().Foo = %v, want %q", output.Foo, "bar")
		}
	})

	t.Run("text response is extracted from markdown", func(t *testing.T) {
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithOutputType(FooBar{}),
		)
		if err != nil {
			t.Fatal(err)
		}

		text := res.Text()
		if text != JSON {
			t.Errorf("Text() = %q, want %q", text, JSON)
		}
	})
}

func TestDefaultFormats(t *testing.T) {
	expectedFormats := []string{
		OutputFormatText,
		OutputFormatJSON,
		OutputFormatJSONL,
		OutputFormatArray,
		OutputFormatEnum,
	}

	for _, format := range expectedFormats {
		t.Run(format+" is registered", func(t *testing.T) {
			formatter := r.LookupValue("/format/" + format)
			if formatter == nil {
				t.Errorf("format %q not registered", format)
			}
			if f, ok := formatter.(Formatter); !ok || f.Name() != format {
				t.Errorf("registered format has wrong name")
			}
		})
	}
}

func TestArrayFormatterParseMessage(t *testing.T) {
	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{"type": "integer"},
			},
		},
	}

	t.Run("returns message unchanged", func(t *testing.T) {
		handler, err := arrayFormatter{}.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart(`[{"id": 1}, {"id": 2}]`)},
		}

		got, err := handler.ParseMessage(msg)
		if err != nil {
			t.Fatalf("ParseMessage() error = %v", err)
		}

		// Array formatter's ParseMessage returns the message unchanged
		if got != msg {
			t.Error("ParseMessage() should return the same message object")
		}
	})
}

func TestJSONLFormatterParseMessage(t *testing.T) {
	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id":   map[string]any{"type": "integer"},
				"name": map[string]any{"type": "string"},
			},
		},
	}

	t.Run("returns message unchanged", func(t *testing.T) {
		handler, err := jsonlFormatter{}.Handler(schema)
		if err != nil {
			t.Fatalf("Handler() error = %v", err)
		}

		msg := &Message{
			Role:    RoleModel,
			Content: []*Part{NewTextPart("{\"id\": 1, \"name\": \"Alice\"}\n{\"id\": 2, \"name\": \"Bob\"}")},
		}

		got, err := handler.ParseMessage(msg)
		if err != nil {
			t.Fatalf("ParseMessage() error = %v", err)
		}

		// JSONL formatter's ParseMessage returns the message unchanged
		if got != msg {
			t.Error("ParseMessage() should return the same message object")
		}
	})
}
