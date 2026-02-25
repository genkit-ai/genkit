// Copyright 2025 Google LLC
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
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestExtractJSONFromMarkdown(t *testing.T) {
	tests := []struct {
		desc string
		in   string
		want string
	}{
		{
			desc: "no markdown",
			in:   "abcdefg",
			want: "abcdefg",
		},
		{
			desc: "no markdown (with line breaks)",
			in:   "ab\ncd\nfg",
			want: "ab\ncd\nfg",
		},
		{
			desc: "simple markdown",
			in:   "```foo bar```",
			want: "```foo bar```",
		},
		{
			desc: "empty markdown",
			in:   "``` ```",
			want: "``` ```",
		},
		{
			desc: "json markdown",
			in:   "```json{\"a\":1}```",
			want: "{\"a\":1}",
		},
		{
			desc: "json multiple line markdown",
			in:   "```json\n{\"a\": 1}\n```",
			want: "{\"a\": 1}",
		},
		{
			desc: "returns first of multiple blocks",
			in:   "```json{\"a\":\n1}```\n```json\n{\"b\":\n1}```",
			want: "{\"a\":\n1}",
		},
		{
			desc: "yaml markdown",
			in:   "```yaml\nkey: 1\nanother-key: 2```",
			want: "```yaml\nkey: 1\nanother-key: 2```",
		},
		{
			desc: "yaml + json markdown",
			in:   "```yaml\nkey: 1\nanother-key: 2``` ```json\n{\"a\": 1}\n```",
			want: "{\"a\": 1}",
		},
		{
			desc: "json + yaml markdown",
			in:   "```json\n{\"a\": 1}\n``` ```yaml\nkey: 1\nanother-key: 2```",
			want: "{\"a\": 1}",
		},
		{
			desc: "uppercase JSON identifier",
			in:   "```JSON\n{\"a\": 1}\n```",
			want: "{\"a\": 1}",
		},
		{
			desc: "mixed case Json identifier",
			in:   "```Json\n{\"a\": 1}\n```",
			want: "{\"a\": 1}",
		},
		{
			desc: "plain code block without identifier",
			in:   "```\n{\"a\": 1}\n```",
			want: "{\"a\": 1}",
		},
		{
			desc: "plain code block with text before",
			in:   "Here is the result:\n\n```\n{\"title\": \"Pizza\"}\n```",
			want: "{\"title\": \"Pizza\"}",
		},
		{
			desc: "json block preferred over plain block",
			in:   "```\n{\"plain\": true}\n``` then ```json\n{\"json\": true}\n```",
			want: "{\"json\": true}",
		},
		{
			desc: "json block with spaces",
			in:   "``` json\n{\"a\": 1}\n```",
			want: "{\"a\": 1}",
		},
		{
			desc: "implicit json block",
			in:   "```{\"a\": 1}```",
			want: "{\"a\": 1}",
		},
		{
			desc: "implicit json block array",
			in:   "```[1, 2]```",
			want: "[1, 2]",
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			if diff := cmp.Diff(ExtractJSONFromMarkdown(tc.in), tc.want); diff != "" {
				t.Errorf("ExtractJSONFromMarkdown diff (+got -want):\n%s", diff)
			}
		})
	}
}

func TestSchemaAsMap(t *testing.T) {
	type Bar struct {
		Bar string
	}
	type Foo struct {
		BarField Bar
		Str      string
	}

	want := map[string]any{
		"additionalProperties": bool(false),
		"properties": map[string]any{
			"BarField": map[string]any{
				"additionalProperties": bool(false),
				"properties": map[string]any{
					"Bar": map[string]any{"type": string("string")},
				},
				"required": []any{string("Bar")},
				"type":     string("object"),
			},
			"Str": map[string]any{"type": string("string")},
		},
		"required": []any{string("BarField"), string("Str")},
		"type":     string("object"),
	}

	got := SchemaAsMap(InferJSONSchema(Foo{}))
	if diff := cmp.Diff(got, want); diff != "" {
		t.Errorf("SchemaAsMap diff (+got -want):\n%s", diff)
	}
}

func TestSchemaAsMapRecursive(t *testing.T) {
	type Node struct {
		Value    string  `json:"value,omitempty"`
		Children []*Node `json:"children,omitempty"`
	}

	schema := SchemaAsMap(InferJSONSchema(Node{}))

	// With DoNotReference and recursion limiting, the schema should be flat
	// and recursive references should become "any" schema.
	if _, ok := schema["$defs"]; ok {
		t.Error("expected no $defs with DoNotReference: true")
	}

	if _, ok := schema["$ref"]; ok {
		t.Error("expected no $ref with DoNotReference: true")
	}

	// Check top-level structure
	if schema["type"] != "object" {
		t.Errorf("expected type to be object, got %v", schema["type"])
	}

	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}

	// Check value field
	valueField, ok := props["value"].(map[string]any)
	if !ok {
		t.Fatal("expected value field in properties")
	}
	if valueField["type"] != "string" {
		t.Errorf("expected value.type to be string, got %v", valueField["type"])
	}

	// Check children field - recursive reference should be "any" schema
	childrenField, ok := props["children"].(map[string]any)
	if !ok {
		t.Fatal("expected children field in properties")
	}
	if childrenField["type"] != "array" {
		t.Errorf("expected children.type to be array, got %v", childrenField["type"])
	}

	items, ok := childrenField["items"].(map[string]any)
	if !ok {
		t.Fatal("expected children to have items")
	}
	// The recursive Node reference should have become an "any" schema
	// including "type" property to prevent recursion errors for schemas for the same type
	if items["type"] != "object" {
		t.Errorf("expected children.items.type to be 'object', got %v", items["type"])
	}
	if items["additionalProperties"] != true {
		t.Errorf("expected children.items to be 'any' schema (additionalProperties: true), got %v", items)
	}
}

func TestInferJSONSchema_SharedType(t *testing.T) {
	type Shared struct {
		Amount float64 `json:"amount"`
	}
	type Prizes struct {
		First  Shared `json:"first"`
		Second Shared `json:"second"`
	}

	schema := SchemaAsMap(InferJSONSchema(Prizes{}))
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}

	second, ok := properties["second"].(map[string]any)
	if !ok {
		t.Fatal("expected 'second' property in schema")
	}

	if second["type"] != "object" {
		t.Errorf("expected type: object for shared type occurrence, got %v", second["type"])
	}
	if second["additionalProperties"] != true {
		t.Errorf("expected additionalProperties: true for shared type occurrence, got %v", second["additionalProperties"])
	}
}

type testStringer struct {
	Value string
}

func (s testStringer) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.Value)
}

func TestInferJSONSchema_SharedTypeWithMarshaler(t *testing.T) {
	type Container struct {
		A testStringer
		B testStringer
	}
	schema := SchemaAsMap(InferJSONSchema(Container{}))
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}

	second, ok := properties["B"].(map[string]any)
	if !ok {
		t.Fatal("expected 'B' property in schema")
	}

	// type should be non-object because it implements json.Marshaler
	if typ, hasType := second["type"]; hasType {
		if typ == "object" {
			t.Errorf("expected type to NOT be object for shared type with marshaler, got %v", typ)
		}
	}

	if val, ok := second["additionalProperties"]; !ok || val != true {
		t.Errorf("expected additionalProperties: true, got %v", val)
	}
}
