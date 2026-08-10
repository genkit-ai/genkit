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
//
// SPDX-License-Identifier: Apache-2.0

package base

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestNormalizeInput(t *testing.T) {
	tests := []struct {
		name     string
		data     any
		schema   map[string]any
		expected any
	}{
		{
			name: "removes null fields from object",
			data: map[string]any{
				"name":        "test",
				"nullField":   nil,
				"emptyString": "",
				"number":      42.0,
			},
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name":        map[string]any{"type": "string"},
					"nullField":   map[string]any{"type": "string"},
					"emptyString": map[string]any{"type": "string"},
					"number":      map[string]any{"type": "integer"},
				},
			},
			expected: map[string]any{
				"name":        "test",
				"emptyString": "",
				"number":      int64(42),
			},
		},
		{
			name: "removes null fields without schema",
			data: map[string]any{
				"name":      "test",
				"nullField": nil,
				"value":     123,
			},
			schema: nil,
			expected: map[string]any{
				"name":  "test",
				"value": 123,
			},
		},
		{
			name: "handles nested objects with null fields",
			data: map[string]any{
				"outer": map[string]any{
					"inner":     "value",
					"nullInner": nil,
				},
				"nullOuter": nil,
			},
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"outer": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"inner":     map[string]any{"type": "string"},
							"nullInner": map[string]any{"type": "string"},
						},
					},
					"nullOuter": map[string]any{"type": "string"},
				},
			},
			expected: map[string]any{
				"outer": map[string]any{
					"inner": "value",
				},
			},
		},
		{
			name: "converts numbers correctly",
			data: map[string]any{
				"intField":   42.0,
				"floatField": 3.14,
			},
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"intField":   map[string]any{"type": "integer"},
					"floatField": map[string]any{"type": "number"},
				},
			},
			expected: map[string]any{
				"intField":   int64(42),
				"floatField": 3.14,
			},
		},
		{
			name: "handles arrays with null elements",
			data: []any{"item1", nil, "item2"},
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "string",
				},
			},
			expected: []any{"item1", nil, "item2"}, // Arrays preserve null elements
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := NormalizeInput(test.data, test.schema)
			if err != nil {
				t.Fatalf("NormalizeInput returned error: %v", err)
			}

			if !reflect.DeepEqual(result, test.expected) {
				t.Errorf("NormalizeInput result mismatch:\nExpected: %+v\nGot: %+v", test.expected, result)
			}
		})
	}
}

func TestUnmarshalAndNormalize(t *testing.T) {
	type TestStruct struct {
		Name   string `json:"name"`
		Age    int64  `json:"age"`
		Active bool   `json:"active"`
	}

	t.Run("unmarshal into structured type", func(t *testing.T) {
		input := json.RawMessage(`{"name":"John","age":30,"active":true}`)
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":   map[string]any{"type": "string"},
				"age":    map[string]any{"type": "integer"},
				"active": map[string]any{"type": "boolean"},
			},
		}
		expected := TestStruct{Name: "John", Age: 30, Active: true}

		result, err := UnmarshalAndNormalize[TestStruct](input, schema)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !reflect.DeepEqual(result, expected) {
			t.Errorf("result mismatch:\nExpected: %+v\nGot: %+v", expected, result)
		}
	})

	t.Run("unmarshal into any type preserves types", func(t *testing.T) {
		input := json.RawMessage(`{"name":"John","age":30,"count":42.0}`)
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":  map[string]any{"type": "string"},
				"age":   map[string]any{"type": "integer"},
				"count": map[string]any{"type": "integer"},
			},
		}
		expected := map[string]any{"name": "John", "age": int64(30), "count": int64(42)}

		result, err := UnmarshalAndNormalize[any](input, schema)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !reflect.DeepEqual(result, expected) {
			t.Errorf("result mismatch:\nExpected: %+v (%T)\nGot: %+v (%T)", expected, expected, result, result)
		}
	})

	t.Run("empty input returns zero value", func(t *testing.T) {
		input := json.RawMessage(``)
		expected := TestStruct{}

		result, err := UnmarshalAndNormalize[TestStruct](input, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !reflect.DeepEqual(result, expected) {
			t.Errorf("result mismatch:\nExpected: %+v\nGot: %+v", expected, result)
		}
	})

	t.Run("removes null fields", func(t *testing.T) {
		input := json.RawMessage(`{"name":"John","age":30,"active":null}`)
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":   map[string]any{"type": "string"},
				"age":    map[string]any{"type": "integer"},
				"active": map[string]any{"type": "boolean"},
			},
		}
		expected := TestStruct{Name: "John", Age: 30, Active: false}

		result, err := UnmarshalAndNormalize[TestStruct](input, schema)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if !reflect.DeepEqual(result, expected) {
			t.Errorf("result mismatch:\nExpected: %+v\nGot: %+v", expected, result)
		}
	})

	t.Run("invalid JSON returns error", func(t *testing.T) {
		input := json.RawMessage(`{invalid json}`)

		_, err := UnmarshalAndNormalize[TestStruct](input, nil)
		if err == nil {
			t.Fatal("expected error but got none")
		}
	})

	t.Run("null input with any type returns nil", func(t *testing.T) {
		input := json.RawMessage(`null`)

		result, err := UnmarshalAndNormalize[any](input, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if result != nil {
			t.Errorf("expected nil, got %v", result)
		}
	})
}
