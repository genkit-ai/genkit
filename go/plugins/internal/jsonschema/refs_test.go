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

package jsonschema

import "testing"

func TestResolveRefUsesReferencedPool(t *testing.T) {
	schema := map[string]any{
		"$defs": map[string]any{
			"Tag": map[string]any{"type": "integer"},
		},
		"definitions": map[string]any{
			"Tag": map[string]any{"type": "string"},
		},
	}

	got, err := ResolveRef(schema, "#/definitions/Tag")
	if err != nil {
		t.Fatalf("ResolveRef() error = %v", err)
	}
	if got["type"] != "string" {
		t.Errorf("ResolveRef() type = %v, want string", got["type"])
	}
}

func TestResolveRefRejectsUnrelatedPointer(t *testing.T) {
	schema := map[string]any{
		"$defs": map[string]any{
			"Tag": map[string]any{"type": "string"},
		},
	}
	if _, err := ResolveRef(schema, "#/properties/Tag"); err == nil {
		t.Fatal("ResolveRef() succeeded for a pointer outside definition maps")
	}
}

func TestResolveRefsDoesNotMutateInput(t *testing.T) {
	schema := map[string]any{
		"$ref": "https://example.com/external-schema",
		"$defs": map[string]any{
			"Unused": map[string]any{"type": "string"},
		},
	}

	got := ResolveRefs(schema)
	if _, ok := got["$defs"]; ok {
		t.Error("ResolveRefs() retained unused definitions")
	}
	if _, ok := schema["$defs"]; !ok {
		t.Error("ResolveRefs() mutated its input")
	}
}
