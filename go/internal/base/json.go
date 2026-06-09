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

package base

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"reflect"
	"regexp"
	"strings"

	"github.com/invopop/jsonschema"
)

// JSONString returns json.Marshal(x) as a string. If json.Marshal returns
// an error, jsonString returns the error text as a JSON string beginning "ERROR:".
func JSONString(x any) string {
	bytes, err := json.Marshal(x)
	if err != nil {
		bytes, _ = json.Marshal(fmt.Sprintf("ERROR: %v", err))
	}
	return string(bytes)
}

// PrettyJSONString returns json.MarshalIndent(x, "", "  ") as a string.
// If json.MarshalIndent returns an error, jsonString returns the error text as
// a JSON string beginning "ERROR:".
func PrettyJSONString(x any) string {
	bytes, err := json.MarshalIndent(x, "", "  ")
	if err != nil {
		bytes, _ = json.MarshalIndent(fmt.Sprintf("ERROR: %v", err), "", "  ")
	}
	return string(bytes)
}

// WriteJSONFile writes value to filename as JSON.
func WriteJSONFile(filename string, value any) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, f.Close())
	}()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "    ") // make the value easy to read for debugging
	return enc.Encode(value)
}

// ReadJSONFile JSON-decodes the contents of filename into pvalue,
// which must be a pointer.
func ReadJSONFile(filename string, pvalue any) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewDecoder(f).Decode(pvalue)
}

// InferJSONSchema infers a JSON schema from a Go value.
//
// Named struct types are reflected with references enabled, so the raw result
// uses `$ref`/`$defs`. Call [InferJSONSchemaMap] (or [InlineAcyclicDefs] on the
// marshaled map) to inline the acyclic definitions; only genuinely recursive
// types retain `$ref`/`$defs`, which lets self-referential Go types round-trip
// as proper recursive JSON Schema rather than collapsing to an "any" schema.
func InferJSONSchema(x any) *jsonschema.Schema {
	r := jsonschema.Reflector{
		// References are required so recursive types can express recursion via
		// $ref; acyclic definitions are inlined afterwards by InlineAcyclicDefs.
		DoNotReference: false,
		Anonymous:      true, // suppress $id
		Mapper: func(t reflect.Type) *jsonschema.Schema {
			// []any reflects to `{ type: "array", items: true }` which is not valid JSON schema.
			if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Interface {
				return &jsonschema.Schema{
					Type:  "array",
					Items: &jsonschema.Schema{AdditionalProperties: jsonschema.TrueSchema},
				}
			}
			return nil
		},
	}
	s := r.Reflect(x)
	s.Version = "" // suppress $schema
	return s
}

// InferJSONSchemaMap infers a JSON schema from a Go value as a map, inlining
// all non-recursive `$defs` so that only genuinely recursive types keep
// `$ref`/`$defs`.
func InferJSONSchemaMap(x any) map[string]any {
	return InlineAcyclicDefs(SchemaAsMap(InferJSONSchema(x)))
}

// InlineAcyclicDefs rewrites a JSON schema document so that every non-recursive
// definition in `$defs` is inlined into the sites that reference it, leaving
// only definitions that participate in a reference cycle. Acyclic schemas come
// out fully inlined with no `$defs`/`$ref`; recursive types keep a `$defs`
// entry referenced via `$ref`, which is the recursion the Gemini
// `responseJsonSchema` field unrolls server-side.
func InlineAcyclicDefs(root map[string]any) map[string]any {
	defsAny, ok := root["$defs"].(map[string]any)
	if !ok || len(defsAny) == 0 {
		return root
	}

	defs := make(map[string]map[string]any, len(defsAny))
	for name, v := range defsAny {
		if m, ok := v.(map[string]any); ok {
			defs[name] = m
		}
	}

	// Build the def-reference graph and determine which defs are recursive
	// (reachable from themselves).
	edges := make(map[string]map[string]bool, len(defs))
	for name, body := range defs {
		refs := map[string]bool{}
		collectRefs(body, refs)
		edges[name] = refs
	}
	recursive := map[string]bool{}
	for name := range defs {
		if canReach(name, name, edges, map[string]bool{}) {
			recursive[name] = true
		}
	}

	// Only non-recursive defs are inlined; recursive refs are preserved.
	inlineable := map[string]map[string]any{}
	for name, body := range defs {
		if !recursive[name] {
			inlineable[name] = body
		}
	}

	// Resolve the root schema (root minus its $defs container).
	rootSchema := map[string]any{}
	for k, v := range root {
		if k == "$defs" {
			continue
		}
		rootSchema[k] = v
	}
	result, _ := inlineRefs(rootSchema, inlineable).(map[string]any)
	if result == nil {
		result = map[string]any{}
	}

	// Keep recursive defs, with their own non-recursive refs inlined.
	newDefs := map[string]any{}
	for name := range recursive {
		newDefs[name] = inlineRefs(defs[name], inlineable)
	}
	if len(newDefs) > 0 {
		result["$defs"] = newDefs
	}
	return result
}

// refName extracts the definition name from a "#/$defs/Name" reference.
func refName(ref string) string {
	tkns := strings.Split(ref, "/")
	return tkns[len(tkns)-1]
}

// collectRefs records the name of every $ref found anywhere within node.
func collectRefs(node any, out map[string]bool) {
	switch n := node.(type) {
	case map[string]any:
		if ref, ok := n["$ref"].(string); ok {
			out[refName(ref)] = true
		}
		for _, v := range n {
			collectRefs(v, out)
		}
	case []any:
		for _, v := range n {
			collectRefs(v, out)
		}
	}
}

// canReach reports whether target is reachable from start in the ref graph.
func canReach(start, target string, edges map[string]map[string]bool, seen map[string]bool) bool {
	for next := range edges[start] {
		if next == target {
			return true
		}
		if !seen[next] {
			seen[next] = true
			if canReach(next, target, edges, seen) {
				return true
			}
		}
	}
	return false
}

// inlineRefs returns a copy of node with every $ref to an inlineable
// definition replaced by that definition's (recursively inlined) body. Refs to
// definitions not in inlineable (i.e. recursive ones) are left untouched, which
// terminates the recursion.
func inlineRefs(node any, inlineable map[string]map[string]any) any {
	switch n := node.(type) {
	case map[string]any:
		if ref, ok := n["$ref"].(string); ok {
			if body, ok := inlineable[refName(ref)]; ok {
				return inlineRefs(body, inlineable)
			}
			return n
		}
		out := make(map[string]any, len(n))
		for k, v := range n {
			out[k] = inlineRefs(v, inlineable)
		}
		return out
	case []any:
		out := make([]any, len(n))
		for i, v := range n {
			out[i] = inlineRefs(v, inlineable)
		}
		return out
	default:
		return node
	}
}

// MapToStruct converts a map[string]any to a struct of type T via JSON round-trip.
func MapToStruct[T any](m map[string]any) (T, error) {
	var result T
	data, err := json.Marshal(m)
	if err != nil {
		return result, err
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return result, err
	}
	return result, nil
}

// StructToMap converts a struct to map[string]any via JSON round-trip.
func StructToMap[T any](v T) (map[string]any, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// SchemaAsMap converts json schema struct to a map (JSON representation).
func SchemaAsMap(s *jsonschema.Schema) map[string]any {
	jsb, err := s.MarshalJSON()
	if err != nil {
		log.Panicf("failed to marshal schema: %v", err)
	}

	// Check if the marshaled JSON is "true" (indicates an empty schema)
	if string(jsb) == "true" {
		return make(map[string]any)
	}

	var m map[string]any
	err = json.Unmarshal(jsb, &m)
	if err != nil {
		log.Panicf("failed to unmarshal schema: %v", err)
	}
	return m
}

// jsonMarkdownRegex matches fenced code blocks with "json" language identifier (case-insensitive).
var jsonMarkdownRegex = regexp.MustCompile("(?si)```\\s*json\\s*(.*?)```")

// plainMarkdownRegex matches fenced code blocks without any language identifier.
var plainMarkdownRegex = regexp.MustCompile("(?s)```\\s*\\n(.*?)```")

// implicitJSONRegex matches fenced code blocks with no language identifier that start with { or [
var implicitJSONRegex = regexp.MustCompile("(?si)```\\s*([{\\[].*?)```")

// ExtractJSONFromMarkdown returns the contents of the first fenced code block in
// the markdown text md. It matches code blocks with "json" identifier (case-insensitive)
// or code blocks without any language identifier. If there is no matching block, it returns md.
func ExtractJSONFromMarkdown(md string) string {
	// First try to match explicit json code blocks
	matches := jsonMarkdownRegex.FindStringSubmatch(md)
	if len(matches) >= 2 {
		return strings.TrimSpace(matches[1])
	}

	// Fall back to plain code blocks (no language identifier)
	matches = plainMarkdownRegex.FindStringSubmatch(md)
	if len(matches) >= 2 {
		return strings.TrimSpace(matches[1])
	}

	// Fall back to implicit JSON blocks (no language identifier, starts with { or [)
	matches = implicitJSONRegex.FindStringSubmatch(md)
	if len(matches) >= 2 {
		return strings.TrimSpace(matches[1])
	}

	return md
}

// GetJSONObjectLines splits a string by newlines, trims whitespace from each line,
// and returns a slice containing only the lines that start with '{'.
func GetJSONObjectLines(text string) []string {
	jsonText := ExtractJSONFromMarkdown(text)

	// Handle both actual "\n" newline strings, as well as newline bytes
	jsonText = strings.ReplaceAll(jsonText, "\n", `\n`)

	// Split the input string into lines based on the newline character.
	lines := strings.Split(jsonText, `\n`)

	var result []string
	for _, line := range lines {
		if line == "" {
			continue
		}

		// Trim leading and trailing whitespace from the current line.
		trimmedLine := strings.TrimSpace(line)
		// Check if the trimmed line starts with the character '{'.
		if strings.HasPrefix(trimmedLine, "{") {
			// If it does, append the trimmed line to our result slice.
			result = append(result, trimmedLine)
		}
	}

	// Return the slice containing the filtered and trimmed lines.
	return result
}
