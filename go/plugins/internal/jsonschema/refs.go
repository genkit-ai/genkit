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

package schemautil

import (
	"fmt"
	"strings"
)

// ResolveRefs returns schema with direct $ref pointers into the top-level
// $defs and definitions maps inlined where possible. Annotation siblings are
// preserved. References with structural siblings, unknown references,
// non-object definitions, and cycles are left intact.
//
// The result may share structure with schema: a subtree with no $ref to
// inline is returned as-is rather than copied, and when schema itself has no
// $defs or definitions to resolve, schema is returned unchanged. Treat the
// result as read-only, the same as schema itself.
func ResolveRefs(schema map[string]any) map[string]any {
	defs, _ := schema["$defs"].(map[string]any)
	definitions, _ := schema["definitions"].(map[string]any)
	if len(defs) == 0 && len(definitions) == 0 {
		return schema
	}
	visited := make(map[string]bool)
	// resolved caches each def's own inlining, keyed by ref, so a def
	// referenced from multiple places is expanded once rather than once per
	// occurrence: without it, a schema whose defs form a diamond (several
	// defs sharing the same descendant) re-expands that descendant from
	// scratch at every occurrence, and the work doubles with each layer of
	// the diamond.
	resolved := make(map[string]map[string]any)
	result, _ := inlineRefs(schema, defs, definitions, visited, resolved).(map[string]any)
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

func inlineRefs(v any, defs, definitions map[string]any, visited map[string]bool, resolved map[string]map[string]any) any {
	switch node := v.(type) {
	case map[string]any:
		if ref, ok := node["$ref"].(string); ok {
			def, found := refTarget(ref, defs, definitions)
			defMap, isMap := def.(map[string]any)
			if found && isMap && !visited[ref] && hasOnlyAnnotationSiblings(node) {
				inlined, cached := resolved[ref]
				if !cached {
					visited[ref] = true
					inlined, _ = inlineRefs(defMap, defs, definitions, visited, resolved).(map[string]any)
					delete(visited, ref)
					resolved[ref] = inlined
				}
				if inlined == nil {
					return node
				}
				if len(node) > 1 {
					merged := make(map[string]any, len(inlined)+len(node))
					for k, val := range inlined {
						merged[k] = val
					}
					for k, val := range node {
						if k != "$ref" {
							merged[k] = inlineRefs(val, defs, definitions, visited, resolved)
						}
					}
					return merged
				}
				return inlined
			}
			return node
		}
		result := make(map[string]any, len(node))
		for k, val := range node {
			result[k] = inlineRefs(val, defs, definitions, visited, resolved)
		}
		return result
	case []any:
		result := make([]any, len(node))
		for i, item := range node {
			result[i] = inlineRefs(item, defs, definitions, visited, resolved)
		}
		return result
	case []map[string]any:
		result := make([]any, len(node))
		for i, item := range node {
			result[i] = inlineRefs(item, defs, definitions, visited, resolved)
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

// isAnnotation reports whether key is safe to keep alongside an inlined $ref:
// annotation keywords proper (description, title, ...), plus $comment, $id,
// and $anchor. Those three are not annotations by the spec's own taxonomy,
// but this resolver never does identifier- or comment-aware resolution, so
// they carry no meaning it acts on either; treating them as structural would
// leave a reflected schema's stray $comment shipping an unresolved $ref to
// Ollama, which is the exact failure this package exists to prevent.
func isAnnotation(key string) bool {
	switch key {
	case "description", "title", "default", "examples", "deprecated", "readOnly", "writeOnly",
		"$comment", "$id", "$anchor":
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
