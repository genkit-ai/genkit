// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import "testing"

// TestToGeminiSchemaRejectsNonObjectSubschemas covers shapes a caller-supplied
// schema can legitimately contain but [genai.Schema] cannot express. They used
// to panic on an unchecked type assertion, taking the process down rather than
// failing the request.
func TestToGeminiSchemaRejectsNonObjectSubschemas(t *testing.T) {
	tests := []struct {
		name   string
		schema map[string]any
	}{
		{"items is a boolean schema", map[string]any{"type": "array", "items": true}},
		{"items is a draft-07 tuple", map[string]any{"type": "array", "items": []any{map[string]any{"type": "string"}}}},
		{"properties is not an object", map[string]any{"type": "object", "properties": "nope"}},
		{"property is a boolean schema", map[string]any{"type": "object", "properties": map[string]any{"x": false}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panicked instead of returning an error: %v", r)
				}
			}()
			if _, err := toGeminiSchema(map[string]any{}, tt.schema); err == nil {
				t.Error("expected an error, got nil")
			}
		})
	}
}

// TestToGeminiSchemaIgnoresNonStringAnnotations checks that a malformed
// annotation is skipped rather than panicking; annotations are advisory, so
// dropping one is preferable to failing the whole request.
func TestToGeminiSchemaIgnoresNonStringAnnotations(t *testing.T) {
	for _, key := range []string{"description", "format", "title"} {
		t.Run(key, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("panicked on non-string %q: %v", key, r)
				}
			}()
			if _, err := toGeminiSchema(map[string]any{}, map[string]any{"type": "string", key: 123}); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
