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
	"time"

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

	got := InferJSONSchemaMap(Foo{})
	if diff := cmp.Diff(got, want); diff != "" {
		t.Errorf("SchemaAsMap diff (+got -want):\n%s", diff)
	}
}

func TestSchemaAsMapRecursive(t *testing.T) {
	type Node struct {
		Value    string  `json:"value,omitempty"`
		Children []*Node `json:"children,omitempty"`
	}

	schema := InferJSONSchemaMap(Node{})

	// A recursive type must express recursion via $ref/$defs rather than
	// collapsing the self-reference to an "any" schema.
	defs, ok := schema["$defs"].(map[string]any)
	if !ok {
		t.Fatalf("expected $defs for recursive type, got %v", schema)
	}
	node, ok := defs["Node"].(map[string]any)
	if !ok {
		t.Fatalf("expected $defs.Node, got %v", defs)
	}

	// The root references the recursive definition.
	if ref, _ := schema["$ref"].(string); ref != "#/$defs/Node" {
		t.Errorf("expected root $ref '#/$defs/Node', got %v", schema["$ref"])
	}

	props, ok := node["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in Node definition")
	}

	// Check value field is inlined.
	valueField, ok := props["value"].(map[string]any)
	if !ok {
		t.Fatal("expected value field in properties")
	}
	if valueField["type"] != "string" {
		t.Errorf("expected value.type to be string, got %v", valueField["type"])
	}

	// The recursive children field references Node via $ref, not an "any" schema.
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
	if ref, _ := items["$ref"].(string); ref != "#/$defs/Node" {
		t.Errorf("expected children.items.$ref '#/$defs/Node', got %v", items)
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

	schema := InferJSONSchemaMap(Prizes{})
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}

	// A non-recursive shared struct type should produce the same full schema
	// for every occurrence rather than collapsing to {additionalProperties: true}.
	want := map[string]any{
		"additionalProperties": false,
		"type":                 "object",
		"required":             []any{"amount"},
		"properties": map[string]any{
			"amount": map[string]any{"type": "number"},
		},
	}
	for _, name := range []string{"first", "second"} {
		got, ok := properties[name].(map[string]any)
		if !ok {
			t.Fatalf("expected %q property in schema", name)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("%q schema mismatch (-want +got):\n%s", name, diff)
		}
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
	schema := InferJSONSchemaMap(Container{})
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}

	a, ok := properties["A"].(map[string]any)
	if !ok {
		t.Fatal("expected 'A' property in schema")
	}
	b, ok := properties["B"].(map[string]any)
	if !ok {
		t.Fatal("expected 'B' property in schema")
	}
	if diff := cmp.Diff(a, b); diff != "" {
		t.Errorf("expected A and B to have identical schemas, diff:\n%s", diff)
	}
}

// TestInferJSONSchema_SharedTimeFields is a regression test for issue #5200:
// `time.Time` used in two fields of the same struct must produce the correct
// `{type: string, format: date-time}` schema for both fields.
func TestInferJSONSchema_SharedTimeFields(t *testing.T) {
	type Input struct {
		StartsAfter  *time.Time `json:"starts_after,omitempty"`
		StartsBefore *time.Time `json:"starts_before,omitempty"`
	}

	schema := InferJSONSchemaMap(Input{})
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}

	want := map[string]any{
		"type":   "string",
		"format": "date-time",
	}
	for _, name := range []string{"starts_after", "starts_before"} {
		got, ok := properties[name].(map[string]any)
		if !ok {
			t.Fatalf("expected %q property in schema", name)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("%q schema mismatch (-want +got):\n%s", name, diff)
		}
	}
}

// TestInferJSONSchema_RecursiveSharedType verifies that a recursive type used
// in multiple fields of the same struct references the same recursive
// definition. Both fields reference Node via $ref, and the recursive Node
// definition is retained in $defs.
func TestInferJSONSchema_RecursiveSharedType(t *testing.T) {
	type Node struct {
		Value    string  `json:"value,omitempty"`
		Children []*Node `json:"children,omitempty"`
	}
	type Pair struct {
		Left  Node `json:"left"`
		Right Node `json:"right"`
	}

	schema := InferJSONSchemaMap(Pair{})
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}

	left, ok := properties["left"].(map[string]any)
	if !ok {
		t.Fatal("expected 'left' property in schema")
	}
	right, ok := properties["right"].(map[string]any)
	if !ok {
		t.Fatal("expected 'right' property in schema")
	}
	if diff := cmp.Diff(left, right); diff != "" {
		t.Errorf("expected 'left' and 'right' schemas to match, diff:\n%s", diff)
	}
	if ref, _ := left["$ref"].(string); ref != "#/$defs/Node" {
		t.Errorf("expected left.$ref '#/$defs/Node', got %v", left)
	}

	// The recursive Node definition is retained and exposes its fields.
	defs, ok := schema["$defs"].(map[string]any)
	if !ok {
		t.Fatal("expected $defs for recursive Node type")
	}
	node, ok := defs["Node"].(map[string]any)
	if !ok {
		t.Fatalf("expected $defs.Node, got %v", defs)
	}
	nodeProps, ok := node["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected Node definition to have properties")
	}
	if _, ok := nodeProps["value"]; !ok {
		t.Errorf("expected Node definition to expose 'value' field, got %v", nodeProps)
	}
}

// genericBox is instantiated below with a type argument from another package,
// which the reflector names after that argument's full import path. The
// definition name therefore contains "/" characters.
type genericBox[T any] struct {
	Item T `json:"item"`
}

// TestInferJSONSchemaMap_GenericTypeName covers a definition name that
// contains "/", as an instantiated generic's does: it is named after its type
// argument, import path and all. Reading the name off the last "/" segment of
// a "#/$defs/..." reference misidentifies it, leaving the reference behind
// while its definition is dropped as acyclic — a dangling $ref that no
// validator can resolve.
func TestInferJSONSchemaMap_GenericTypeName(t *testing.T) {
	schema := InferJSONSchemaMap(genericBox[json.RawMessage]{})

	if _, ok := schema["$defs"]; ok {
		t.Errorf("acyclic generic type should inline fully, got $defs: %v", schema["$defs"])
	}
	assertNoRefs(t, schema)

	if schema["type"] != "object" {
		t.Errorf("expected the definition to be inlined at the root, got %v", schema)
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties, got %v", schema)
	}
	if _, ok := props["item"]; !ok {
		t.Errorf("expected inlined 'item' property, got %v", props)
	}
}

// assertNoRefs fails if any $ref survives anywhere in node.
func assertNoRefs(t *testing.T, node any) {
	t.Helper()
	switch n := node.(type) {
	case map[string]any:
		if ref, ok := n["$ref"]; ok {
			t.Errorf("unresolved $ref left in schema: %v", ref)
		}
		for _, v := range n {
			assertNoRefs(t, v)
		}
	case []any:
		for _, v := range n {
			assertNoRefs(t, v)
		}
	}
}
