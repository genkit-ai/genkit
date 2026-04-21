// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"google.golang.org/genai"
)

// toGeminiSchema translates a map representing a standard JSON schema to a more
// limited [genai.Schema].
func toGeminiSchema(originalSchema map[string]any, genkitSchema map[string]any) (*genai.Schema, error) {
	// this covers genkitSchema == nil and {}
	// genkitSchema will be {} if it's any
	if len(genkitSchema) == 0 {
		return nil, nil
	}
	if v, ok := genkitSchema["$ref"]; ok {
		ref, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("invalid $ref value: not a string")
		}
		s, err := resolveRef(originalSchema, ref)
		if err != nil {
			return nil, err
		}
		return toGeminiSchema(originalSchema, s)
	}

	// Handle "anyOf" subschemas by finding the first valid schema definition
	if v, ok := genkitSchema["anyOf"]; ok {
		if anyOfList, isList := v.([]map[string]any); isList {
			for _, subSchema := range anyOfList {
				if subSchemaType, hasType := subSchema["type"]; hasType {
					if typeStr, isString := subSchemaType.(string); isString && typeStr != "null" {
						if title, ok := genkitSchema["title"]; ok {
							subSchema["title"] = title
						}
						if description, ok := genkitSchema["description"]; ok {
							subSchema["description"] = description
						}
						// Found a schema like: {"type": "string"}
						return toGeminiSchema(originalSchema, subSchema)
					}
				}
			}
		}
	}

	schema := &genai.Schema{}
	typeVal, ok := genkitSchema["type"]
	if !ok {
		return nil, fmt.Errorf("schema is missing the 'type' field: %#v", genkitSchema)
	}

	typeStr, ok := typeVal.(string)
	if !ok {
		return nil, fmt.Errorf("schema 'type' field is not a string, but %T", typeVal)
	}

	switch typeStr {
	case "string":
		schema.Type = genai.TypeString
	case "float64", "number":
		schema.Type = genai.TypeNumber
	case "integer":
		schema.Type = genai.TypeInteger
	case "boolean":
		schema.Type = genai.TypeBoolean
	case "object":
		schema.Type = genai.TypeObject
	case "array":
		schema.Type = genai.TypeArray
	default:
		return nil, fmt.Errorf("schema type %q not allowed", genkitSchema["type"])
	}
	if v, ok := genkitSchema["required"]; ok {
		schema.Required = castToStringArray(v)
	}
	if v, ok := genkitSchema["propertyOrdering"]; ok {
		schema.PropertyOrdering = castToStringArray(v)
	}
	if v, ok := genkitSchema["description"]; ok {
		schema.Description = v.(string)
	}
	if v, ok := genkitSchema["format"]; ok {
		schema.Format = v.(string)
	}
	if v, ok := genkitSchema["title"]; ok {
		schema.Title = v.(string)
	}
	if v, ok := genkitSchema["minItems"]; ok {
		if i64, ok := castToInt64(v); ok {
			schema.MinItems = genai.Ptr(i64)
		}
	}
	if v, ok := genkitSchema["maxItems"]; ok {
		if i64, ok := castToInt64(v); ok {
			schema.MaxItems = genai.Ptr(i64)
		}
	}
	if v, ok := genkitSchema["maximum"]; ok {
		if f64, ok := castToFloat64(v); ok {
			schema.Maximum = genai.Ptr(f64)
		}
	}
	if v, ok := genkitSchema["minimum"]; ok {
		if f64, ok := castToFloat64(v); ok {
			schema.Minimum = genai.Ptr(f64)
		}
	}
	if v, ok := genkitSchema["enum"]; ok {
		schema.Enum = castToStringArray(v)
	}
	if v, ok := genkitSchema["items"]; ok {
		items, err := toGeminiSchema(originalSchema, v.(map[string]any))
		if err != nil {
			return nil, err
		}
		schema.Items = items
	}
	if val, ok := genkitSchema["properties"]; ok {
		props := map[string]*genai.Schema{}
		for k, v := range val.(map[string]any) {
			p, err := toGeminiSchema(originalSchema, v.(map[string]any))
			if err != nil {
				return nil, err
			}
			props[k] = p
		}
		schema.Properties = props
	}
	// Nullable -- not supported in jsonschema.Schema

	return schema, nil
}

// resolveRef resolves a $ref reference in a JSON schema.
func resolveRef(originalSchema map[string]any, ref string) (map[string]any, error) {
	tkns := strings.Split(ref, "/")
	// refs look like: $/ref/foo -- we need the foo part
	name := tkns[len(tkns)-1]
	if defs, ok := originalSchema["$defs"].(map[string]any); ok {
		if def, ok := defs[name].(map[string]any); ok {
			return def, nil
		}
	}
	// definitions (legacy)
	if defs, ok := originalSchema["definitions"].(map[string]any); ok {
		if def, ok := defs[name].(map[string]any); ok {
			return def, nil
		}
	}
	return nil, fmt.Errorf("unable to resolve schema reference")
}

// castToStringArray converts either []any or []string to []string, filtering non-strings.
// This handles enum values from JSON Schema which may come as either type depending on unmarshaling.
// Filter out non-string types from if v is []any type.
func castToStringArray(v any) []string {
	switch a := v.(type) {
	case []string:
		// Return a shallow copy to avoid aliasing
		out := make([]string, 0, len(a))
		for _, s := range a {
			if s != "" {
				out = append(out, s)
			}
		}
		return out
	case []any:
		var out []string
		for _, it := range a {
			if s, ok := it.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

// castToInt64 converts v to int64 when possible.
func castToInt64(v any) (int64, bool) {
	switch t := v.(type) {
	case int:
		return int64(t), true
	case int64:
		return t, true
	case float64:
		return int64(t), true
	case string:
		if i, err := strconv.ParseInt(t, 10, 64); err == nil {
			return i, true
		}
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return i, true
		}
	}
	return 0, false
}

// castToFloat64 converts v to float64 when possible.
func castToFloat64(v any) (float64, bool) {
	switch t := v.(type) {
	case float64:
		return t, true
	case int:
		return float64(t), true
	case int64:
		return float64(t), true
	case string:
		if f, err := strconv.ParseFloat(t, 64); err == nil {
			return f, true
		}
	case json.Number:
		if f, err := t.Float64(); err == nil {
			return f, true
		}
	}
	return 0, false
}
