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

// Run the npm script that generates JSON Schemas from the zod types
// in the *.ts files. It writes the result to genkit-tools/genkit-schema.json
//go:generate npm --prefix ../../genkit-tools run export:schemas

// Run the Go code generator on the file just created.
//go:generate go run ../internal/cmd/jsonschemagen -outdir .. -config schemas.config ../../genkit-tools/genkit-schema.json ai

// Package core implements Genkit actions and other essential machinery.
// This package is primarily intended for Genkit internals and for plugins.
// Genkit applications should use the genkit package.
package core

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

// DefineSchema defines a named JSON schema and registers it in the registry.
// The `schema` argument must be a JSON schema definition represented as a map.
// It panics if a schema with the same name is already registered.
func DefineSchema(r api.Registry, name string, schema map[string]any) {
	r.RegisterSchema(name, schema)
}

// DefineSchemaFor defines a named JSON schema derived from a Go type
// and registers it in the registry using the type's name.
func DefineSchemaFor[T any](r api.Registry) {
	var v T
	t := reflect.TypeOf(v)
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	name := t.Name()
	r.RegisterSchema(name, InferSchemaMap(v))
}

// SchemaRef returns a JSON schema reference map for the given name.
func SchemaRef(name string) map[string]any {
	return map[string]any{
		"$ref": fmt.Sprintf("genkit:%s", name),
	}
}

// ResolveSchema resolves a schema that may contain a $ref to a registered schema.
// If the schema contains a $ref with the "genkit:" prefix, it looks up the schema by name.
// Returns the original schema if no $ref is present, or the resolved schema if found.
// Returns an error if the schema reference cannot be resolved.
func ResolveSchema(r api.Registry, schema map[string]any) (map[string]any, error) {
	return resolveSchema(r, schema, nil, 0)
}

const maxSchemaDepth = 50

func resolveSchema(r api.Registry, schema map[string]any, seen map[uintptr]bool, depth int) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	if depth > maxSchemaDepth {
		return nil, fmt.Errorf("schema reference too deep (possible cycle)")
	}

	if ref, ok := schema["$ref"].(string); ok {
		if schemaName, found := strings.CutPrefix(ref, "genkit:"); found {
			resolved := r.LookupSchema(schemaName)
			if resolved == nil {
				return nil, fmt.Errorf("schema %q not found", schemaName)
			}

			if seen[reflect.ValueOf(resolved).Pointer()] {
				return schema, nil
			}

			newSeen := make(map[uintptr]bool, len(seen)+1)
			for k, v := range seen {
				newSeen[k] = v
			}
			newSeen[reflect.ValueOf(schema).Pointer()] = true

			// Recursive call to resolve any refs within the looked-up schema.
			return resolveSchema(r, resolved, newSeen, depth+1)
		}
		// If it's a non-genkit $ref, we return the schema as-is to allow
		// the underlying validator to handle it.
		return schema, nil
	}

	ptr := reflect.ValueOf(schema).Pointer()
	if seen[ptr] {
		return schema, nil
	}

	newSeen := make(map[uintptr]bool, len(seen)+1)
	for k, v := range seen {
		newSeen[k] = v
	}
	newSeen[ptr] = true
	seen = newSeen

	// Iterate and recursively resolve any nested maps or arrays.
	// We only clone the schema if a change is actually made.
	var newSchema map[string]any
	for k, v := range schema {
		resolved, err := resolveValue(r, v, seen, depth+1)
		if err != nil {
			return nil, err
		}

		if newSchema == nil && !isSame(v, resolved) {
			newSchema = make(map[string]any, len(schema))
			for k2, v2 := range schema {
				newSchema[k2] = v2
			}
		}
		if newSchema != nil {
			newSchema[k] = resolved
		}
	}

	if newSchema == nil {
		return schema, nil
	}
	return newSchema, nil
}

func resolveValue(r api.Registry, v any, seen map[uintptr]bool, depth int) (any, error) {
	switch val := v.(type) {
	case map[string]any:
		return resolveSchema(r, val, seen, depth)
	case []any:
		return resolveArray(r, val, seen, depth)
	default:
		return v, nil
	}
}

func resolveArray(r api.Registry, arr []any, seen map[uintptr]bool, depth int) ([]any, error) {
	if depth > maxSchemaDepth {
		return nil, fmt.Errorf("schema reference too deep (possible cycle)")
	}
	var newArray []any
	for i, v := range arr {
		resolved, err := resolveValue(r, v, seen, depth+1)
		if err != nil {
			return nil, err
		}

		if newArray == nil && !isSame(v, resolved) {
			newArray = make([]any, len(arr))
			copy(newArray, arr[:i])
		}
		if newArray != nil {
			newArray[i] = resolved
		}
	}
	if newArray == nil {
		return arr, nil
	}
	return newArray, nil
}

func isSame(a, b any) bool {
	if reflect.TypeOf(a) != reflect.TypeOf(b) {
		return false
	}
	va := reflect.ValueOf(a)
	vb := reflect.ValueOf(b)
	switch va.Kind() {
	case reflect.Map, reflect.Slice:
		return va.Pointer() == vb.Pointer()
	default:
		return a == b
	}
}

// InferSchemaMap infers a JSON schema from a Go value and converts it to a map.
func InferSchemaMap(value any) map[string]any {
	schema := base.InferJSONSchema(value)
	return base.SchemaAsMap(schema)
}
