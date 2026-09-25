// Copyright 2026 Google LLC
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

package core

import (
	"reflect"
	"testing"
)

func TestCloseSchemaObjects(t *testing.T) {
	input := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"choice": map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{
						"type":       "object",
						"properties": map[string]any{"name": map[string]any{"type": "string"}},
					},
				},
			},
		},
		"$defs": map[string]any{
			"Address": map[string]any{
				"type":       "object",
				"properties": map[string]any{"street": map[string]any{"type": "string"}},
			},
		},
	}

	got := CloseSchemaObjects(input)
	if got["additionalProperties"] != false {
		t.Errorf("root additionalProperties = %v, want false", got["additionalProperties"])
	}
	choice := got["properties"].(map[string]any)["choice"].(map[string]any)
	branch := choice["anyOf"].([]any)[1].(map[string]any)
	if branch["additionalProperties"] != false {
		t.Errorf("anyOf object additionalProperties = %v, want false", branch["additionalProperties"])
	}
	address := got["$defs"].(map[string]any)["Address"].(map[string]any)
	if address["additionalProperties"] != false {
		t.Errorf("$defs object additionalProperties = %v, want false", address["additionalProperties"])
	}
	if _, ok := input["additionalProperties"]; ok {
		t.Fatal("CloseSchemaObjects mutated its input")
	}
}

func TestRequireSchemaProperties(t *testing.T) {
	input := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"zeta":  map[string]any{"type": "string"},
			"alpha": map[string]any{"type": "string"},
			"kept":  map[string]any{"type": "string"},
		},
		"required": []any{"kept"},
	}

	got := RequireSchemaProperties(input)
	want := []string{"kept", "alpha", "zeta"}
	if !reflect.DeepEqual(got["required"], want) {
		t.Errorf("required = %#v, want %#v", got["required"], want)
	}
	if !reflect.DeepEqual(input["required"], []any{"kept"}) {
		t.Fatal("RequireSchemaProperties mutated its input")
	}
}

func TestResolveLocalSchemaRef(t *testing.T) {
	schema := map[string]any{
		"$defs": map[string]any{
			"a/b~c": map[string]any{"type": "string"},
		},
	}
	got, err := ResolveLocalSchemaRef(schema, "#/$defs/a~1b~0c")
	if err != nil {
		t.Fatalf("ResolveLocalSchemaRef() error = %v", err)
	}
	if got["type"] != "string" {
		t.Errorf("resolved type = %v, want string", got["type"])
	}
}
