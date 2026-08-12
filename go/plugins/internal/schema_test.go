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

package internal

import (
	"reflect"
	"slices"
	"testing"

	"github.com/firebase/genkit/go/internal/base"
)

// testSchema is shaped like a reflected SDK params struct: nested objects, an
// array of objects, and a branch schema.
func testSchema() map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []any{"model", "temperature"},
		"properties": map[string]any{
			"model":       map[string]any{"type": "string"},
			"temperature": map[string]any{"type": "number"},
			"thinking": map[string]any{
				"type":     "object",
				"required": []any{"budget"},
				"properties": map[string]any{
					"budget": map[string]any{"type": "integer"},
				},
			},
			"tools": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
					},
				},
			},
		},
	}
}

func TestApplySchemaOverrides(t *testing.T) {
	schema := testSchema()
	ApplySchemaOverrides(schema, SchemaOverrides{
		Descriptions: map[string]string{
			"temperature":     "how random",
			"thinking.budget": "how much thinking",
			"tools[].name":    "what to call it",
		},
		Hidden: []string{"model", "thinking.budget"},
	})

	if got := mustNavigate(t, schema, "temperature")["description"]; got != "how random" {
		t.Errorf("top-level description = %v", got)
	}
	if got := mustNavigate(t, schema, "tools", "[]", "name")["description"]; got != "what to call it" {
		t.Errorf("description inside an array's items = %v", got)
	}

	props := schema["properties"].(map[string]any)
	if props["model"] != true {
		t.Errorf("hidden property = %v, want the permissive true schema", props["model"])
	}
	nested := props["thinking"].(map[string]any)["properties"].(map[string]any)
	if nested["budget"] != true {
		t.Errorf("hidden nested property = %v, want true", nested["budget"])
	}

	// Hiding must not open the object: an unknown property is still a typo
	// worth rejecting, and a hidden one is present rather than absent.
	if schema["additionalProperties"] != false {
		t.Errorf("hiding opened the object: additionalProperties = %v", schema["additionalProperties"])
	}
	// A hidden property the SDK marked required would demand a field the dev UI
	// cannot show and the plugin refuses.
	if required := schema["required"].([]any); slices.Contains(required, any("model")) {
		t.Errorf("required still demands the hidden property: %v", required)
	} else if !slices.Contains(required, any("temperature")) {
		t.Errorf("required lost a property that is not hidden: %v", required)
	}
	// The same holds at depth: hiding a nested property prunes it from its own
	// parent's required list, not just the root's. "budget" was the only entry,
	// so the key goes rather than being left empty.
	if required, ok := props["thinking"].(map[string]any)["required"]; ok {
		t.Errorf("nested required survived hiding its only entry: %v", required)
	}
}

// A path that no longer resolves must no-op: plugins build these schemas during
// init, so a renamed SDK field would otherwise stop the plugin from loading.
func TestApplySchemaOverridesIsBestEffort(t *testing.T) {
	schema := testSchema()
	before := base.CloneSchema(schema)
	ApplySchemaOverrides(schema, SchemaOverrides{
		Descriptions: map[string]string{
			"doesNotExist":             "x",
			"alsoMissing.deeper":       "x",
			"tools[].notReal":          "x",
			"completely[].fake[].path": "x",
			"temperature.notAnObject":  "x",
		},
		Hidden: []string{"doesNotExist", "missing[].alsoMissing", "tools[].notReal", "[]"},
	})
	if !reflect.DeepEqual(before, schema) {
		t.Error("bogus paths changed the schema")
	}
}

func TestUnresolvedPaths(t *testing.T) {
	schema := testSchema()
	o := SchemaOverrides{
		Descriptions: map[string]string{"temperature": "x", "tools[].name": "x", "renamedAway": "x"},
		Hidden:       []string{"model", "alsoRenamed"},
	}
	got := o.UnresolvedPaths(schema)
	if want := []string{"alsoRenamed", "renamedAway"}; !reflect.DeepEqual(got, want) {
		t.Errorf("UnresolvedPaths() = %v, want %v", got, want)
	}

	// Applying the overrides must not change the answer: a hidden path resolves
	// to the `true` schema afterward, and that still counts as present.
	ApplySchemaOverrides(schema, o)
	if got := o.UnresolvedPaths(schema); slices.Contains(got, "model") {
		t.Errorf("an applied hidden path was reported stale: %v", got)
	}
}

func mustNavigate(t *testing.T, schema map[string]any, steps ...string) map[string]any {
	t.Helper()
	got, ok := schemaAt(schema, steps).(map[string]any)
	if !ok {
		t.Fatalf("path %v does not resolve to a schema", steps)
	}
	return got
}
