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

// Package jsonschema contains JSON Schema helpers shared by Genkit provider plugins.
package jsonschema

import "encoding/json"

// EnforceStrict returns a deep copy of schema with additionalProperties: false
// set on every object subschema, recursing through "properties" and "items".
// This is the shape OpenAI and Anthropic require for strict tool schemas.
//
// The input is not mutated. If the schema is not JSON-marshalable, the
// original map is returned unchanged.
func EnforceStrict(schema map[string]any) map[string]any {
	if schema == nil {
		return nil
	}
	var cloned map[string]any
	b, err := json.Marshal(schema)
	if err != nil {
		return schema
	}
	if err := json.Unmarshal(b, &cloned); err != nil {
		return schema
	}
	enforceStrict(cloned)
	return cloned
}

// enforceStrict mutates schema in place. Callers must clone first.
func enforceStrict(schema map[string]any) {
	if t, ok := schema["type"].(string); ok && t == "object" {
		schema["additionalProperties"] = false
		if props, ok := schema["properties"].(map[string]any); ok {
			for _, v := range props {
				if subSchema, ok := v.(map[string]any); ok {
					enforceStrict(subSchema)
				}
			}
		}
	}
	if t, ok := schema["type"].(string); ok && t == "array" {
		if items, ok := schema["items"].(map[string]any); ok {
			enforceStrict(items)
		}
	}
}
