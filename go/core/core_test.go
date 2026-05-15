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

package core

import (
	"testing"

	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/go-cmp/cmp"
)

func TestDefineSchema(t *testing.T) {
	t.Run("registers schema in registry", func(t *testing.T) {
		r := registry.New()
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
				"age":  map[string]any{"type": "integer"},
			},
			"required": []any{"name"},
		}

		DefineSchema(r, "Person", schema)

		found := r.LookupSchema("Person")
		if found == nil {
			t.Fatal("schema not found in registry")
		}
		if diff := cmp.Diff(schema, found); diff != "" {
			t.Errorf("schema mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestDefineSchemaFor(t *testing.T) {
	t.Run("registers schema derived from Go type", func(t *testing.T) {
		r := registry.New()

		type User struct {
			Name  string `json:"name"`
			Email string `json:"email"`
		}

		DefineSchemaFor[User](r)

		found := r.LookupSchema("User")
		if found == nil {
			t.Fatal("schema not found in registry")
		}
		// Check that the schema has expected properties
		props, ok := found["properties"].(map[string]any)
		if !ok {
			t.Fatal("expected properties in schema")
		}
		if props["name"] == nil {
			t.Error("expected 'name' property in schema")
		}
		if props["email"] == nil {
			t.Error("expected 'email' property in schema")
		}
	})

	t.Run("handles pointer types", func(t *testing.T) {
		r := registry.New()

		type Config struct {
			Debug bool `json:"debug"`
		}

		DefineSchemaFor[*Config](r)

		found := r.LookupSchema("Config")
		if found == nil {
			t.Fatal("schema not found in registry for pointer type")
		}
	})
}

func TestSchemaRef(t *testing.T) {
	t.Run("returns schema reference map", func(t *testing.T) {
		ref := SchemaRef("MyType")

		want := map[string]any{
			"$ref": "genkit:MyType",
		}
		if diff := cmp.Diff(want, ref); diff != "" {
			t.Errorf("SchemaRef mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("handles various names", func(t *testing.T) {
		tests := []struct {
			name string
			want string
		}{
			{"Simple", "genkit:Simple"},
			{"Package.Type", "genkit:Package.Type"},
			{"my-schema", "genkit:my-schema"},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				ref := SchemaRef(tt.name)
				if ref["$ref"] != tt.want {
					t.Errorf("$ref = %q, want %q", ref["$ref"], tt.want)
				}
			})
		}
	})
}

func TestResolveSchema(t *testing.T) {
	t.Run("returns nil for nil schema", func(t *testing.T) {
		r := registry.New()

		resolved, err := ResolveSchema(r, nil)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resolved != nil {
			t.Errorf("expected nil, got %v", resolved)
		}
	})

	t.Run("returns original schema without ref", func(t *testing.T) {
		r := registry.New()
		schema := map[string]any{
			"type": "string",
		}

		resolved, err := ResolveSchema(r, schema)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if diff := cmp.Diff(schema, resolved); diff != "" {
			t.Errorf("schema mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("resolves genkit ref", func(t *testing.T) {
		r := registry.New()
		originalSchema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{"type": "integer"},
			},
		}
		r.RegisterSchema("Entity", originalSchema)

		refSchema := map[string]any{
			"$ref": "genkit:Entity",
		}

		resolved, err := ResolveSchema(r, refSchema)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if diff := cmp.Diff(originalSchema, resolved); diff != "" {
			t.Errorf("resolved schema mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns original schema for non-genkit ref", func(t *testing.T) {
		r := registry.New()
		schema := map[string]any{
			"$ref": "#/definitions/Other",
		}

		resolved, err := ResolveSchema(r, schema)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if diff := cmp.Diff(schema, resolved); diff != "" {
			t.Errorf("schema mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns error for missing schema", func(t *testing.T) {
		r := registry.New()
		refSchema := map[string]any{
			"$ref": "genkit:NonExistent",
		}

		_, err := ResolveSchema(r, refSchema)

		if err == nil {
			t.Error("expected error for missing schema, got nil")
		}
	})
}

func TestInferSchemaMap(t *testing.T) {
	t.Run("infers schema from struct", func(t *testing.T) {
		type TestStruct struct {
			Name    string `json:"name"`
			Count   int    `json:"count"`
			Enabled bool   `json:"enabled"`
		}

		schema := InferSchemaMap(TestStruct{})

		if schema["type"] != "object" {
			t.Errorf("type = %v, want %q", schema["type"], "object")
		}
		props, ok := schema["properties"].(map[string]any)
		if !ok {
			t.Fatal("expected properties map")
		}
		if props["name"] == nil {
			t.Error("expected 'name' property")
		}
		if props["count"] == nil {
			t.Error("expected 'count' property")
		}
		if props["enabled"] == nil {
			t.Error("expected 'enabled' property")
		}
	})

	t.Run("infers schema from primitive types", func(t *testing.T) {
		tests := []struct {
			value    any
			wantType string
		}{
			{"hello", "string"},
			{42, "integer"},
			{3.14, "number"},
			{true, "boolean"},
		}

		for _, tt := range tests {
			t.Run(tt.wantType, func(t *testing.T) {
				schema := InferSchemaMap(tt.value)
				if schema["type"] != tt.wantType {
					t.Errorf("type = %v, want %q", schema["type"], tt.wantType)
				}
			})
		}
	})

	t.Run("infers schema from slice", func(t *testing.T) {
		schema := InferSchemaMap([]string{})

		if schema["type"] != "array" {
			t.Errorf("type = %v, want %q", schema["type"], "array")
		}
	})

	t.Run("infers schema from nested struct", func(t *testing.T) {
		type Inner struct {
			Value string `json:"value"`
		}
		type Outer struct {
			Inner Inner `json:"inner"`
		}

		schema := InferSchemaMap(Outer{})

		props := schema["properties"].(map[string]any)
		innerProp := props["inner"].(map[string]any)
		if innerProp["type"] != "object" {
			t.Errorf("inner type = %v, want %q", innerProp["type"], "object")
		}
	})
}
