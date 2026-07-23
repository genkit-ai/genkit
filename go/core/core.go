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

// schemaRefName returns the referenced schema name if schema is a named-schema
// reference ({"$ref": "genkit:Name"}, as produced by [SchemaRef]), or "" if the
// schema is nil or inline.
func schemaRefName(schema map[string]any) string {
	ref, ok := schema["$ref"].(string)
	if !ok {
		return ""
	}
	name, found := strings.CutPrefix(ref, "genkit:")
	if !found {
		return ""
	}
	return name
}

// ResolveSchema resolves a schema that may contain a $ref to a registered schema.
// If the schema contains a $ref with the "genkit:" prefix, it looks up the schema by name.
// Returns the original schema if no $ref is present, or the resolved schema if found.
// Returns an error if the schema reference cannot be resolved, including when
// r is nil (e.g. the schema belongs to an action that was never registered).
func ResolveSchema(r api.Registry, schema map[string]any) (map[string]any, error) {
	name := schemaRefName(schema)
	if name == "" {
		return schema, nil
	}
	if r == nil {
		return nil, fmt.Errorf("schema %q cannot be resolved without a registry; register the action first", name)
	}
	resolved := r.LookupSchema(name)
	if resolved == nil {
		return nil, fmt.Errorf("schema %q not found", name)
	}
	return resolved, nil
}

// UnresolvedSchemaRefs returns the names of the named-schema references
// ({"$ref": "genkit:Name"}) that remain unresolved in desc's schema slots.
// Serving boundaries (e.g. the reflection servers) use it to reject
// descriptors that still carry references to schemas that were never defined,
// instead of silently passing the reference through to consumers.
func UnresolvedSchemaRefs(desc *api.ActionDesc) []string {
	var names []string
	for _, schema := range []map[string]any{desc.InputSchema, desc.OutputSchema, desc.StreamSchema, desc.InitSchema} {
		if name := schemaRefName(schema); name != "" {
			names = append(names, name)
		}
	}
	return names
}

// InferSchemaMap infers a JSON schema from a Go value and converts it to a map.
func InferSchemaMap(value any) map[string]any {
	schema := base.InferJSONSchema(value)
	return base.SchemaAsMap(schema)
}
