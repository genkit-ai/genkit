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

import (
	"fmt"
	"strings"
)

// ResolveRefs returns a copy of schema with direct $ref pointers into the
// top-level $defs and definitions maps inlined where possible. Annotation
// siblings are preserved. References with structural siblings, unknown
// references, non-object definitions, and cycles are left intact.
func ResolveRefs(schema map[string]any) map[string]any {
	defs, _ := schema["$defs"].(map[string]any)
	definitions, _ := schema["definitions"].(map[string]any)
	if len(defs) == 0 && len(definitions) == 0 {
		return schema
	}
	visited := make(map[string]bool)
	result, _ := inlineRefs(schema, defs, definitions, visited).(map[string]any)
	if result == nil {
		return schema
	}
	if !hasLocalRefsOutsideDefs(result) {
		cloned := make(map[string]any, len(result))
		for k, v := range result {
			cloned[k] = v
		}
		result = cloned
		delete(result, "$defs")
		delete(result, "definitions")
	}
	return result
}

// ResolveRef resolves a direct JSON Pointer into a top-level $defs or
// definitions map. Other local pointers and external references are rejected.
func ResolveRef(schema map[string]any, ref string) (map[string]any, error) {
	defs, _ := schema["$defs"].(map[string]any)
	definitions, _ := schema["definitions"].(map[string]any)
	value, ok := refTarget(ref, defs, definitions)
	if !ok {
		return nil, fmt.Errorf("unable to resolve schema reference %q", ref)
	}
	resolved, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("schema reference %q does not resolve to an object", ref)
	}
	return resolved, nil
}

func inlineRefs(v any, defs, definitions map[string]any, visited map[string]bool) any {
	switch node := v.(type) {
	case map[string]any:
		if ref, ok := node["$ref"].(string); ok {
			def, found := refTarget(ref, defs, definitions)
			defMap, isMap := def.(map[string]any)
			if found && isMap && !visited[ref] && hasOnlyAnnotationSiblings(node) {
				visited[ref] = true
				resolved, _ := inlineRefs(defMap, defs, definitions, visited).(map[string]any)
				delete(visited, ref)
				if resolved == nil {
					return node
				}
				if len(node) > 1 {
					merged := make(map[string]any, len(resolved)+len(node))
					for k, val := range resolved {
						merged[k] = val
					}
					for k, val := range node {
						if k != "$ref" {
							merged[k] = inlineRefs(val, defs, definitions, visited)
						}
					}
					return merged
				}
				return resolved
			}
			return node
		}
		result := make(map[string]any, len(node))
		for k, val := range node {
			result[k] = inlineRefs(val, defs, definitions, visited)
		}
		return result
	case []any:
		result := make([]any, len(node))
		for i, item := range node {
			result[i] = inlineRefs(item, defs, definitions, visited)
		}
		return result
	case []map[string]any:
		result := make([]any, len(node))
		for i, item := range node {
			result[i] = inlineRefs(item, defs, definitions, visited)
		}
		return result
	default:
		return v
	}
}

func refTarget(ref string, defs, definitions map[string]any) (any, bool) {
	var encodedName string
	var pool map[string]any
	switch {
	case strings.HasPrefix(ref, "#/$defs/"):
		encodedName = strings.TrimPrefix(ref, "#/$defs/")
		pool = defs
	case strings.HasPrefix(ref, "#/definitions/"):
		encodedName = strings.TrimPrefix(ref, "#/definitions/")
		pool = definitions
	default:
		return nil, false
	}
	if encodedName == "" || strings.Contains(encodedName, "/") {
		return nil, false
	}
	name := strings.ReplaceAll(encodedName, "~1", "/")
	name = strings.ReplaceAll(name, "~0", "~")
	def, ok := pool[name]
	return def, ok
}

func hasOnlyAnnotationSiblings(node map[string]any) bool {
	for key := range node {
		if key != "$ref" && key != "$defs" && key != "definitions" && !isAnnotation(key) {
			return false
		}
	}
	return true
}

func isAnnotation(key string) bool {
	switch key {
	case "description", "title", "default", "examples", "deprecated", "readOnly", "writeOnly":
		return true
	default:
		return false
	}
}

func hasLocalRefsOutsideDefs(v any) bool {
	switch node := v.(type) {
	case map[string]any:
		if ref, ok := node["$ref"].(string); ok && strings.HasPrefix(ref, "#/") {
			return true
		}
		for key, val := range node {
			if key == "$defs" || key == "definitions" {
				continue
			}
			if hasLocalRefsOutsideDefs(val) {
				return true
			}
		}
	case []any:
		for _, item := range node {
			if hasLocalRefsOutsideDefs(item) {
				return true
			}
		}
	case []map[string]any:
		for _, item := range node {
			if hasLocalRefsOutsideDefs(item) {
				return true
			}
		}
	}
	return false
}
