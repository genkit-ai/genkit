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
	"fmt"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/internal/base"
)

// InlineLocalSchemaRefs returns schema with direct $ref pointers into the
// top-level $defs and definitions maps inlined where possible. Annotation
// siblings are preserved. References with structural siblings, unknown
// references, non-object definitions, and cycles are left intact.
//
// The result may share structure with schema: a subtree with no $ref to inline
// is returned as-is rather than copied. Treat the result as read-only, the same
// as schema itself.
func InlineLocalSchemaRefs(schema map[string]any) map[string]any {
	defs, _ := schema["$defs"].(map[string]any)
	definitions, _ := schema["definitions"].(map[string]any)
	if len(defs) == 0 && len(definitions) == 0 {
		return schema
	}
	visited := make(map[string]bool)
	// Cache each definition's expansion so a diamond-shaped definition graph
	// does not expand the shared descendants again for every reference.
	resolved := make(map[string]map[string]any)
	result, _ := inlineLocalSchemaRefs(schema, defs, definitions, visited, resolved).(map[string]any)
	if result == nil {
		return schema
	}
	if !hasLocalSchemaRefsOutsideDefs(result) {
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

// ResolveLocalSchemaRef resolves a direct JSON Pointer into a top-level $defs
// or definitions map. Other local pointers and external references are
// rejected.
func ResolveLocalSchemaRef(schema map[string]any, ref string) (map[string]any, error) {
	defs, _ := schema["$defs"].(map[string]any)
	definitions, _ := schema["definitions"].(map[string]any)
	value, ok := localSchemaRefTarget(ref, defs, definitions)
	if !ok {
		return nil, fmt.Errorf("unable to resolve schema reference %q", ref)
	}
	resolved, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("schema reference %q does not resolve to an object", ref)
	}
	return resolved, nil
}

// CloseSchemaObjects returns a deep copy of schema with
// additionalProperties:false on every object schema. This is one component of
// the strict JSON Schema dialects used by providers such as OpenAI and
// Anthropic; it deliberately does not make optional properties required.
func CloseSchemaObjects(schema map[string]any) map[string]any {
	return transformSchemaObjects(schema, func(object map[string]any) {
		object["additionalProperties"] = false
	})
}

// RequireSchemaProperties returns a deep copy of schema with every declared
// object property included in required. Existing required order is retained;
// newly required properties are appended in lexical order. This is a separate
// transform from CloseSchemaObjects because provider strictness dialects differ.
func RequireSchemaProperties(schema map[string]any) map[string]any {
	return transformSchemaObjects(schema, func(object map[string]any) {
		properties, _ := object["properties"].(map[string]any)
		if len(properties) == 0 {
			return
		}

		required := schemaStringSlice(object["required"])
		seen := make(map[string]bool, len(required))
		for _, name := range required {
			seen[name] = true
		}
		missing := make([]string, 0, len(properties)-len(required))
		for name := range properties {
			if !seen[name] {
				missing = append(missing, name)
			}
		}
		slices.Sort(missing)
		object["required"] = append(required, missing...)
	})
}

func transformSchemaObjects(schema map[string]any, transform func(map[string]any)) map[string]any {
	if schema == nil {
		return nil
	}
	result := base.CloneSchema(schema)
	walkJSONSubschemas(result, func(subschema map[string]any) {
		if schemaHasType(subschema, "object") {
			transform(subschema)
		}
	})
	return result
}

func walkJSONSubschemas(schema map[string]any, visit func(map[string]any)) {
	if schema == nil {
		return
	}

	for _, key := range []string{"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"} {
		children, _ := schema[key].(map[string]any)
		for _, child := range children {
			if subschema, ok := child.(map[string]any); ok {
				walkJSONSubschemas(subschema, visit)
			}
		}
	}
	for _, key := range []string{
		"items", "additionalProperties", "contains", "propertyNames",
		"unevaluatedItems", "unevaluatedProperties", "not", "if", "then", "else",
	} {
		if subschema, ok := schema[key].(map[string]any); ok {
			walkJSONSubschemas(subschema, visit)
		}
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf", "prefixItems"} {
		switch branches := schema[key].(type) {
		case []any:
			for _, branch := range branches {
				if subschema, ok := branch.(map[string]any); ok {
					walkJSONSubschemas(subschema, visit)
				}
			}
		case []map[string]any:
			for _, subschema := range branches {
				walkJSONSubschemas(subschema, visit)
			}
		}
	}

	visit(schema)
}

func schemaHasType(schema map[string]any, want string) bool {
	switch types := schema["type"].(type) {
	case string:
		return types == want
	case []string:
		return slices.Contains(types, want)
	case []any:
		return slices.Contains(types, any(want))
	default:
		return false
	}
}

func schemaStringSlice(value any) []string {
	switch values := value.(type) {
	case []string:
		return slices.Clone(values)
	case []any:
		result := make([]string, 0, len(values))
		for _, value := range values {
			if text, ok := value.(string); ok {
				result = append(result, text)
			}
		}
		return result
	default:
		return nil
	}
}

func inlineLocalSchemaRefs(v any, defs, definitions map[string]any, visited map[string]bool, resolved map[string]map[string]any) any {
	switch node := v.(type) {
	case map[string]any:
		if ref, ok := node["$ref"].(string); ok {
			def, found := localSchemaRefTarget(ref, defs, definitions)
			defMap, isMap := def.(map[string]any)
			if found && isMap && !visited[ref] && hasOnlySchemaAnnotationSiblings(node) {
				inlined, cached := resolved[ref]
				if !cached {
					visited[ref] = true
					inlined, _ = inlineLocalSchemaRefs(defMap, defs, definitions, visited, resolved).(map[string]any)
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
							merged[k] = inlineLocalSchemaRefs(val, defs, definitions, visited, resolved)
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
			result[k] = inlineLocalSchemaRefs(val, defs, definitions, visited, resolved)
		}
		return result
	case []any:
		result := make([]any, len(node))
		for i, item := range node {
			result[i] = inlineLocalSchemaRefs(item, defs, definitions, visited, resolved)
		}
		return result
	case []map[string]any:
		result := make([]any, len(node))
		for i, item := range node {
			result[i] = inlineLocalSchemaRefs(item, defs, definitions, visited, resolved)
		}
		return result
	default:
		return v
	}
}

func localSchemaRefTarget(ref string, defs, definitions map[string]any) (any, bool) {
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

func hasOnlySchemaAnnotationSiblings(node map[string]any) bool {
	for key := range node {
		if key != "$ref" && key != "$defs" && key != "definitions" && !isSchemaAnnotation(key) {
			return false
		}
	}
	return true
}

func isSchemaAnnotation(key string) bool {
	switch key {
	case "description", "title", "default", "examples", "deprecated", "readOnly", "writeOnly",
		"$comment", "$id", "$anchor":
		return true
	default:
		return false
	}
}

func hasLocalSchemaRefsOutsideDefs(v any) bool {
	return hasLocalSchemaRefs(v, true)
}

func hasLocalSchemaRefs(v any, root bool) bool {
	switch node := v.(type) {
	case map[string]any:
		if ref, ok := node["$ref"].(string); ok && strings.HasPrefix(ref, "#/") {
			return true
		}
		for key, val := range node {
			if root && (key == "$defs" || key == "definitions") {
				continue
			}
			if hasLocalSchemaRefs(val, false) {
				return true
			}
		}
	case []any:
		for _, item := range node {
			if hasLocalSchemaRefs(item, false) {
				return true
			}
		}
	case []map[string]any:
		for _, item := range node {
			if hasLocalSchemaRefs(item, false) {
				return true
			}
		}
	}
	return false
}
