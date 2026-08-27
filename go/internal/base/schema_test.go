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

package base

import (
	"reflect"
	"testing"
)

// walkTestSchema is shaped like a reflected SDK params struct: nested objects,
// an array of objects, and a branch schema, each of which the walker must reach.
func walkTestSchema() map[string]any {
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
			"effort": map[string]any{
				"oneOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
	}
}

func TestWalkSubschemasReachesEveryBranch(t *testing.T) {
	schema := walkTestSchema()
	var seen []string
	WalkSubschemas(schema, func(sub map[string]any) map[string]any {
		if t, ok := sub["type"].(string); ok {
			seen = append(seen, t)
		}
		sub["visited"] = true
		return sub
	})

	// One per named property, the array's items, the object inside it, and both
	// oneOf branches. The root is deliberately not visited.
	for _, want := range []string{"string", "number", "object", "array", "integer"} {
		if !contains(seen, want) {
			t.Errorf("walk never reached a %q subschema; saw %v", want, seen)
		}
	}
	if _, ok := schema["visited"]; ok {
		t.Error("walk visited the root, which callers handle themselves")
	}
	props := schema["properties"].(map[string]any)
	items := props["tools"].(map[string]any)["items"].(map[string]any)
	if got := items["properties"].(map[string]any)["name"].(map[string]any)["visited"]; got != true {
		t.Error("walk did not reach a property inside an array's items")
	}
	branch := props["effort"].(map[string]any)["oneOf"].([]any)[0]
	if got := branch.(map[string]any)["visited"]; got != true {
		t.Error("walk did not reach a oneOf branch")
	}
}

// A visit that returns a different map must replace the original, which is what
// lets a transform widen a subschema rather than only annotate it.
func TestWalkSubschemasHonorsReplacement(t *testing.T) {
	schema := walkTestSchema()
	WalkSubschemas(schema, func(sub map[string]any) map[string]any {
		return map[string]any{"replaced": true}
	})
	props := schema["properties"].(map[string]any)
	if got := props["model"].(map[string]any)["replaced"]; got != true {
		t.Error("a replaced property schema was not written back")
	}
}

// Reflected schemas carry non-schema values in the same slots (a boolean
// subschema, or additionalProperties: false), and the walker must step over
// them rather than panicking.
func TestWalkSubschemasSkipsNonSchemas(t *testing.T) {
	schema := map[string]any{
		"additionalProperties": false,
		"properties": map[string]any{
			"hidden": true,
			"real":   map[string]any{"type": "string"},
		},
	}
	WalkSubschemas(schema, func(sub map[string]any) map[string]any {
		sub["visited"] = true
		return sub
	})
	props := schema["properties"].(map[string]any)
	if props["hidden"] != true {
		t.Errorf("boolean subschema was rewritten to %v", props["hidden"])
	}
	if schema["additionalProperties"] != false {
		t.Errorf("additionalProperties: false was rewritten to %v", schema["additionalProperties"])
	}
	if props["real"].(map[string]any)["visited"] != true {
		t.Error("walk skipped the real subschema alongside the boolean one")
	}
}

func TestCloneSchemaIsDeep(t *testing.T) {
	original := walkTestSchema()
	clone := CloneSchema(original)
	if !reflect.DeepEqual(original, clone) {
		t.Fatal("clone does not equal the original")
	}
	clone["properties"].(map[string]any)["thinking"].(map[string]any)["properties"].(map[string]any)["budget"].(map[string]any)["type"] = "mutated"
	clone["required"].([]any)[0] = "mutated"

	origBudget := original["properties"].(map[string]any)["thinking"].(map[string]any)["properties"].(map[string]any)["budget"].(map[string]any)
	if got := origBudget["type"]; got != "integer" {
		t.Errorf("mutating the clone reached the original: type = %v", got)
	}
	if got := original["required"].([]any)[0]; got != "model" {
		t.Errorf("mutating the clone's array reached the original: %v", got)
	}
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
