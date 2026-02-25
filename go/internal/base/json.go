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
func InferJSONSchema(x any) (s *jsonschema.Schema) {
	seen := make(map[reflect.Type]bool)

	r := jsonschema.Reflector{
		DoNotReference: true,
		Mapper: func(t reflect.Type) *jsonschema.Schema {
			// []any generates `{ type: "array", items: true }` which is not valid JSON schema.
			if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Interface {
				return &jsonschema.Schema{
					Type: "array",
					Items: &jsonschema.Schema{
						// This field is not necessary but it's the most benign way for the object to not be empty.
						AdditionalProperties: jsonschema.TrueSchema,
					},
				}
			}

			// Handle recursive types: track struct types we've seen.
			// The first encounter is reflected normally; subsequent encounters
			// (including self-references) return an "any" schema to break recursion.
			baseType := t
			if t.Kind() == reflect.Ptr {
				baseType = t.Elem()
			}
			if baseType.Kind() == reflect.Struct {
				if seen[baseType] {
					// check if the type implements json.Marshaler since it might serialize to a non-object
					marshalerType := reflect.TypeOf((*json.Marshaler)(nil)).Elem()
					if baseType.Implements(marshalerType) || reflect.PointerTo(baseType).Implements(marshalerType) {
						return &jsonschema.Schema{
							AdditionalProperties: jsonschema.TrueSchema,
						}
					}
					return &jsonschema.Schema{
						Type:                 "object",
						AdditionalProperties: jsonschema.TrueSchema,
					}
				}
				seen[baseType] = true
			}

			return nil // Return nil to use default schema generation for other types
		},
	}
	s = r.Reflect(x)
	s.Version = ""
	s.ID = ""
	return s
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
