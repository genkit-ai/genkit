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

package base

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestExtractJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    any
		wantErr bool
	}{
		{
			name:  "complete object",
			input: `{"name": "John", "age": 30}`,
			want:  map[string]any{"name": "John", "age": float64(30)},
		},
		{
			name:  "complete array",
			input: `[1, 2, 3]`,
			want:  []any{float64(1), float64(2), float64(3)},
		},
		{
			name:  "object with prefix text",
			input: `Some text before {"name": "Jane"}`,
			want:  map[string]any{"name": "Jane"},
		},
		{
			name:  "incomplete object",
			input: `{"name": "John", "age": 3`,
			want:  map[string]any{"name": "John", "age": float64(3)},
		},
		{
			name:  "incomplete object with partial string",
			input: `{"name": "Jo`,
			want:  map[string]any{"name": "Jo"},
		},
		{
			name:  "incomplete nested object",
			input: `{"person": {"name": "John"`,
			want:  map[string]any{"person": map[string]any{"name": "John"}},
		},
		{
			name:  "object with trailing comma",
			input: `{"name": "John",`,
			want:  map[string]any{"name": "John"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractJSON(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ExtractJSON() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractItems(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		cursor     int
		wantItems  []any
		wantCursor int
	}{
		{
			name:       "complete array",
			input:      `[{"name": "John"}, {"name": "Jane"}]`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}, map[string]any{"name": "Jane"}},
			wantCursor: 35,
		},
		{
			name:       "partial array - first item",
			input:      `[{"name": "John"}`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}},
			wantCursor: 17,
		},
		{
			name:       "partial array - incomplete second item",
			input:      `[{"name": "John"}, {"name": "J`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}},
			wantCursor: 17,
		},
		{
			name:       "incremental parsing from cursor",
			input:      `[{"name": "John"}, {"name": "Jane"}]`,
			cursor:     18,
			wantItems:  []any{map[string]any{"name": "Jane"}},
			wantCursor: 35,
		},
		{
			name:       "array with prefix text",
			input:      `Some text [{"name": "John"}]`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}},
			wantCursor: 27,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractItems(tt.input, tt.cursor)
			if diff := cmp.Diff(tt.wantItems, result.Items); diff != "" {
				t.Errorf("ExtractItems() items mismatch (-want +got):\n%s", diff)
			}
			if result.Cursor != tt.wantCursor {
				t.Errorf("ExtractItems() cursor = %v, want %v", result.Cursor, tt.wantCursor)
			}
		})
	}
}

func TestCompleteJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "unclosed object",
			input: `{"name": "John"`,
			want:  `{"name": "John"}`,
		},
		{
			name:  "unclosed array",
			input: `[1, 2, 3`,
			want:  `[1, 2, 3]`,
		},
		{
			name:  "unclosed string",
			input: `{"name": "John`,
			want:  `{"name": "John"}`,
		},
		{
			name:  "nested unclosed",
			input: `{"person": {"name": "John"`,
			want:  `{"person": {"name": "John"}}`,
		},
		{
			name:  "trailing comma",
			input: `{"name": "John",`,
			want:  `{"name": "John"}`,
		},
		{
			name:  "empty string",
			input: "",
			want:  "{}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CompleteJSON(tt.input)
			if got != tt.want {
				t.Errorf("CompleteJSON() = %v, want %v", got, tt.want)
			}

			// Verify result is valid JSON
			var result any
			err := json.Unmarshal([]byte(got), &result)
			if err != nil {
				t.Errorf("CompleteJSON() produced invalid JSON: %v", err)
			}
		})
	}
}
