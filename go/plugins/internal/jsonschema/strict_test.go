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

package jsonschema

import (
	"reflect"
	"testing"
)

func TestEnforceStrict_NilReturnsNil(t *testing.T) {
	if got := EnforceStrict(nil); got != nil {
		t.Errorf("EnforceStrict(nil) = %v, want nil", got)
	}
}

func TestEnforceStrict_TopLevelObject(t *testing.T) {
	got := EnforceStrict(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	})
	if got["additionalProperties"] != false {
		t.Errorf("missing additionalProperties on top-level object: %v", got)
	}
}

func TestEnforceStrict_RecursesIntoNestedObjects(t *testing.T) {
	got := EnforceStrict(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"outer": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"inner": map[string]any{
						"type":       "object",
						"properties": map[string]any{},
					},
				},
			},
		},
	})

	outer := got["properties"].(map[string]any)["outer"].(map[string]any)
	if outer["additionalProperties"] != false {
		t.Errorf("missing additionalProperties on nested object: %v", outer)
	}
	inner := outer["properties"].(map[string]any)["inner"].(map[string]any)
	if inner["additionalProperties"] != false {
		t.Errorf("missing additionalProperties on doubly-nested object: %v", inner)
	}
}

func TestEnforceStrict_RecursesIntoArrayItems(t *testing.T) {
	got := EnforceStrict(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"list": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				},
			},
		},
	})
	list := got["properties"].(map[string]any)["list"].(map[string]any)
	items := list["items"].(map[string]any)
	if items["additionalProperties"] != false {
		t.Errorf("missing additionalProperties on array items: %v", items)
	}
}

func TestEnforceStrict_DoesNotMutateInput(t *testing.T) {
	input := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}
	// Snapshot a deep-ish copy for comparison.
	snapshot := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
	}
	_ = EnforceStrict(input)
	if !reflect.DeepEqual(input, snapshot) {
		t.Errorf("input was mutated: got %v, want %v", input, snapshot)
	}
}

func TestEnforceStrict_LeavesNonObjectsAlone(t *testing.T) {
	got := EnforceStrict(map[string]any{
		"type": "string",
	})
	if _, present := got["additionalProperties"]; present {
		t.Errorf("did not expect additionalProperties on non-object schema, got %v", got)
	}
}
