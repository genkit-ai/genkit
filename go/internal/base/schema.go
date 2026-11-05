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

// WalkSubschemas replaces every subschema reachable from schema with visit's
// result: the members of properties, the items schema, a schema-valued
// additionalProperties, and the branches of anyOf, oneOf, and allOf, at every
// depth. Non-schema values (a boolean subschema, say) are left alone.
//
// schema itself is not visited, since a transform that widens or annotates
// fields usually means something different for the root. Apply visit to it
// directly when it should be included.
func WalkSubschemas(schema map[string]any, visit func(map[string]any) map[string]any) {
	if schema == nil {
		return
	}
	replace := func(sub any) (any, bool) {
		m, ok := sub.(map[string]any)
		if !ok {
			return sub, false
		}
		WalkSubschemas(m, visit)
		return visit(m), true
	}
	if props, ok := schema["properties"].(map[string]any); ok {
		for name, sub := range props {
			if next, ok := replace(sub); ok {
				props[name] = next
			}
		}
	}
	for _, key := range []string{"items", "additionalProperties"} {
		if next, ok := replace(schema[key]); ok {
			schema[key] = next
		}
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		branches, ok := schema[key].([]any)
		if !ok {
			continue
		}
		for i, sub := range branches {
			if next, ok := replace(sub); ok {
				branches[i] = next
			}
		}
	}
}

// CloneSchema returns a deep copy of schema, so a transform applied to the
// result cannot reach the caller's map. Values other than nested schemas and
// their arrays are the scalars JSON decodes to, which are immutable.
func CloneSchema(schema map[string]any) map[string]any {
	if schema == nil {
		return nil
	}
	out := make(map[string]any, len(schema))
	for k, v := range schema {
		out[k] = cloneSchemaValue(v)
	}
	return out
}

func cloneSchemaValue(v any) any {
	switch v := v.(type) {
	case map[string]any:
		return CloneSchema(v)
	case []any:
		out := make([]any, len(v))
		for i, e := range v {
			out[i] = cloneSchemaValue(e)
		}
		return out
	}
	return v
}
