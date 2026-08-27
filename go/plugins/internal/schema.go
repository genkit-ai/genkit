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

package internal

import (
	"slices"
	"sort"
	"strings"
)

// This file holds the config-schema curation plugins share. A provider plugin
// reflects its SDK's params struct into a config schema, then curates the
// result: SDK structs carry no schema descriptions, and some of their fields
// are owned by a Genkit option and must not be offered. Doing that by hand per
// plugin is how two plugins end up with the same field spelled two ways.

// SchemaOverrides is the presentation a plugin layers onto a reflected config
// schema: help text for the fields a caller may set, and hiding for the ones a
// Genkit option owns.
//
// Paths are property names, dotted for nesting, with "[]" to enter an array's
// items: "temperature", "thinking.budget_tokens", "tools[].name".
type SchemaOverrides struct {
	// Descriptions maps a path to the field's dev UI help text.
	Descriptions map[string]string
	// Hidden lists paths to keep out of the dev UI.
	Hidden []string
}

// ApplySchemaOverrides applies o to schema in place.
//
// A hidden property is replaced by the permissive `true` schema rather than
// deleted. Both keep it out of the dev UI, which renders only properties whose
// type it recognizes, but the config schema is also enforced on every request:
// a hidden field must still reach the plugin's own check, which names the
// Genkit option to use instead. Deleting it would fail validation as an unknown
// property, and forcing additionalProperties open to let it back through would
// give up rejecting real typos anywhere in that object.
//
// A path that does not resolve is a no-op, so an entry left stale by a renamed
// SDK field is not fatal. Plugins assert [SchemaOverrides.UnresolvedPaths] is
// empty in a test to catch those.
func ApplySchemaOverrides(schema map[string]any, o SchemaOverrides) {
	if schema == nil {
		return
	}
	for _, path := range o.Hidden {
		hideSchemaAt(schema, parseSchemaPath(path))
	}
	for path, desc := range o.Descriptions {
		if target, ok := schemaAt(schema, parseSchemaPath(path)).(map[string]any); ok {
			target["description"] = desc
		}
	}
}

// UnresolvedPaths returns the paths in o that schema has no property for,
// sorted. Applying one is a silent no-op, so it is dead weight that stopped
// describing anything when the SDK it was written against changed.
func (o SchemaOverrides) UnresolvedPaths(schema map[string]any) []string {
	var stale []string
	for path := range o.Descriptions {
		if _, ok := schemaAt(schema, parseSchemaPath(path)).(map[string]any); !ok {
			stale = append(stale, path)
		}
	}
	for _, path := range o.Hidden {
		// A hidden path resolves to the `true` schema once applied, so accept
		// any present value rather than requiring a map.
		if schemaAt(schema, parseSchemaPath(path)) == nil {
			stale = append(stale, path)
		}
	}
	sort.Strings(stale)
	return stale
}

// SchemaAt returns what path addresses in schema: a subschema, the `true` of a
// hidden property, or nil when a step does not resolve. Paths use the notation
// in [SchemaOverrides].
func SchemaAt(schema map[string]any, path string) any {
	return schemaAt(schema, parseSchemaPath(path))
}

// parseSchemaPath splits a path into navigation steps, each a property name or
// the literal "[]" meaning "descend into an array's item schema".
//
//	"temperature"          -> ["temperature"]
//	"output_config.format" -> ["output_config", "format"]
//	"tools[].name"         -> ["tools", "[]", "name"]
func parseSchemaPath(path string) []string {
	var steps []string
	for _, tok := range strings.Split(path, ".") {
		if name := strings.TrimSuffix(tok, "[]"); name != tok {
			steps = append(steps, name, "[]")
		} else {
			steps = append(steps, tok)
		}
	}
	return steps
}

// schemaAt descends a schema, walking items for "[]" steps and properties for
// named ones, and returns what the path addresses: a subschema, a boolean
// schema, or nil when a step does not resolve.
func schemaAt(schema map[string]any, steps []string) any {
	var cur any = schema
	for _, step := range steps {
		m, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		if step == "[]" {
			cur = m["items"]
			continue
		}
		props, ok := m["properties"].(map[string]any)
		if !ok {
			return nil
		}
		if cur, ok = props[step]; !ok {
			return nil
		}
	}
	return cur
}

// hideSchemaAt replaces the property at the given path with the permissive
// `true` schema, and drops it from its parent's "required" list: a hidden
// property the SDK marks required would otherwise leave the schema demanding a
// field the dev UI cannot show and the plugin refuses. A path that does not
// resolve is a no-op.
func hideSchemaAt(schema map[string]any, steps []string) {
	if len(steps) == 0 || steps[len(steps)-1] == "[]" {
		return
	}
	parent, ok := schemaAt(schema, steps[:len(steps)-1]).(map[string]any)
	if !ok {
		return
	}
	props, ok := parent["properties"].(map[string]any)
	if !ok {
		return
	}
	leaf := steps[len(steps)-1]
	if _, ok := props[leaf]; !ok {
		return
	}
	props[leaf] = true
	if required, ok := parent["required"].([]any); ok {
		required = slices.DeleteFunc(required, func(r any) bool { return r == any(leaf) })
		// Hiding the last required property drops the key rather than leaving
		// an empty list: it constrains nothing either way, and draft-04
		// validators reject the empty one (its meta-schema puts minItems 1 on
		// required, a constraint draft-06 removed).
		if len(required) == 0 {
			delete(parent, "required")
		} else {
			parent["required"] = required
		}
	}
}
