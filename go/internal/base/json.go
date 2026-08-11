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
	"bytes"
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

// HasJSONValue reports whether raw carries an actual JSON value: it is
// non-empty and not the JSON null literal, ignoring surrounding whitespace.
func HasJSONValue(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

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

var jsonMarshalerType = reflect.TypeOf((*json.Marshaler)(nil)).Elem()

// InferJSONSchema infers a JSON schema from a Go value.
//
// Recursion is detected by stack: while a struct type T is being reflected, T
// is marked in-progress. Any nested encounter of T (a self-reference) returns
// an "any" schema; T is unmarked when its reflection completes. Each top-level
// occurrence of T (siblings, repeats) gets its own full reflection — so a
// struct used in multiple fields produces the correct schema each time.
//
// We can't observe reflection completion through the library's Mapper hook
// alone, so each struct type is reflected via a sub-Reflector. The Mapper's
// defer fires when the sub-Reflector returns, which is the exit point.
func InferJSONSchema(x any) *jsonschema.Schema {
	inProgress := make(map[reflect.Type]bool)
	var mapper func(reflect.Type) *jsonschema.Schema
	mapper = func(t reflect.Type) *jsonschema.Schema {
		// []any reflects to `{ type: "array", items: true }` which is not valid JSON schema.
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Interface {
			return &jsonschema.Schema{
				Type:  "array",
				Items: &jsonschema.Schema{AdditionalProperties: jsonschema.TrueSchema},
			}
		}
		baseType := t
		if t.Kind() == reflect.Ptr {
			baseType = t.Elem()
		}
		if baseType.Kind() != reflect.Struct {
			return nil
		}
		if inProgress[baseType] {
			return anyStructSchema(baseType)
		}

		inProgress[baseType] = true
		defer delete(inProgress, baseType)

		// The sub-Reflector's first Mapper call is for baseType itself: return
		// nil so the library reflects it. All nested calls (fields, including
		// recursive self-references) delegate back to the outer mapper, where
		// inProgress[baseType] is set and recursion is broken.
		firstCall := true
		sub := jsonschema.Reflector{
			DoNotReference: true,
			Anonymous:      true, // suppress $id on this nested schema
			Mapper: func(st reflect.Type) *jsonschema.Schema {
				if firstCall && st == baseType {
					firstCall = false
					return nil
				}
				return mapper(st)
			},
		}
		s := sub.ReflectFromType(baseType)
		s.Version = "" // suppress $schema on this nested schema
		return s
	}

	r := jsonschema.Reflector{DoNotReference: true, Anonymous: true, Mapper: mapper}
	s := r.Reflect(x)
	s.Version = ""
	return s
}

// ErrTypeMismatch reports that a value's Go type cannot be reinterpreted as
// the requested type in [ConvertToExact].
//
// This package cannot classify it, since core/status depends on this one. A
// caller that surfaces the failure past the framework boundary must attach a
// sentinel users can match, as ai.ErrInputTypeMismatch does.
var ErrTypeMismatch = errors.New("type mismatch")

// ConvertToExact converts a dynamically typed value to T.
//
// It accepts a value that is already a T, a *T (a nil pointer yields the zero
// value), or a value in the JSON wire form the framework's transports produce
// (map[string]any, []any, a scalar), which it decodes into T. A nil value
// yields the zero value.
//
// Reinterpreting one struct as an unrelated struct is refused with an error
// wrapping [ErrTypeMismatch]: a JSON round-trip between two unrelated structs
// succeeds while leaving every field zero, so the caller would otherwise get a
// blank value instead of a diagnosis.
//
// When T cannot carry JSON's types on its own, meaning T is an interface or a
// map, slice or array whose elements are, the decoded value is normalized
// against the schema inferred from v (see [NormalizeInput]) so that an integer
// stays an int64 instead of widening to float64. That is what the reflection
// API does to action input, so a value reaches T with the same Go types
// whichever way it arrived. It also drops null-valued keys, as the wire does.
func ConvertToExact[T any](v any) (T, error) {
	var zero T
	if v == nil {
		return zero, nil
	}
	// Covers an exact match and T being an interface that v satisfies, in
	// which case the value is handed over untouched.
	if typed, ok := v.(T); ok {
		return typed, nil
	}
	if p, ok := v.(*T); ok {
		if p == nil {
			return zero, nil
		}
		return *p, nil
	}

	dstType := reflect.TypeFor[T]()
	if src, ok := structType(reflect.TypeOf(v)); ok {
		if dst, ok := structType(dstType); ok && src != dst {
			return zero, fmt.Errorf("%w: got %T, want %T", ErrTypeMismatch, v, zero)
		}
	}

	data, err := json.Marshal(v)
	if err != nil {
		return zero, fmt.Errorf("cannot convert %T to %T: %w", v, zero, err)
	}

	if !dynamicType(dstType) {
		var result T
		if err := json.Unmarshal(data, &result); err != nil {
			return zero, fmt.Errorf("cannot convert %T to %T: %w", v, zero, err)
		}
		return result, nil
	}

	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		return zero, fmt.Errorf("cannot convert %T to %T: %w", v, zero, err)
	}
	normalized, err := NormalizeInput(decoded, SchemaAsMap(InferJSONSchema(v)))
	if err != nil {
		return zero, fmt.Errorf("cannot convert %T to %T: %w", v, zero, err)
	}
	if normalized == nil {
		return zero, nil
	}
	result, ok := asTarget[T](normalized)
	if !ok {
		return zero, fmt.Errorf("%w: got %T, want %T", ErrTypeMismatch, v, zero)
	}
	return result, nil
}

// asTarget converts an already-normalized dynamic value into T.
//
// A type assertion alone only matches the value form, so a pointer target such
// as *map[string]any would miss and lose the normalization. The pointer chain
// is allocated around the value instead.
func asTarget[T any](v any) (T, bool) {
	var zero T
	if typed, ok := v.(T); ok {
		return typed, true
	}

	dst := reflect.TypeFor[T]()
	if dst.Kind() != reflect.Pointer {
		return zero, false
	}
	val := reflect.ValueOf(v)
	if !val.IsValid() {
		return zero, false
	}

	depth := 0
	elem := dst
	for elem.Kind() == reflect.Pointer {
		elem = elem.Elem()
		depth++
	}
	if !val.Type().AssignableTo(elem) {
		return zero, false
	}

	// Build from the element type rather than the value's own type, so that a
	// *any target holds an interface rather than the concrete value.
	cur := reflect.New(elem)
	cur.Elem().Set(val)
	for range depth - 1 {
		p := reflect.New(cur.Type())
		p.Elem().Set(cur)
		cur = p
	}

	typed, ok := cur.Interface().(T)
	return typed, ok
}

// structType reports whether t is a struct, or a pointer chain ending in one,
// and returns the struct type itself so that T and *T compare equal.
func structType(t reflect.Type) (reflect.Type, bool) {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t != nil && t.Kind() == reflect.Struct {
		return t, true
	}
	return nil, false
}

// dynamicType reports whether decoding JSON into t yields dynamically typed
// values, so that t cannot on its own restore the Go types the source had.
//
// Pointers are followed, since a *map[string]any restores no more than the
// map it points at.
func dynamicType(t reflect.Type) bool {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == nil {
		return false
	}
	switch t.Kind() {
	case reflect.Interface:
		return true
	case reflect.Map, reflect.Slice, reflect.Array:
		return t.Elem().Kind() == reflect.Interface
	}
	return false
}

// ConvertTo is [ConvertToExact] for callers that only need to know whether the
// conversion was possible.
func ConvertTo[T any](v any) (T, bool) {
	result, err := ConvertToExact[T](v)
	return result, err == nil
}

// anyStructSchema returns the "any" schema used to break recursion. Types
// that implement json.Marshaler may serialize to a non-object, so we omit
// `type: object` for them.
func anyStructSchema(t reflect.Type) *jsonschema.Schema {
	if t.Implements(jsonMarshalerType) || reflect.PointerTo(t).Implements(jsonMarshalerType) {
		return &jsonschema.Schema{AdditionalProperties: jsonschema.TrueSchema}
	}
	return &jsonschema.Schema{
		Type:                 "object",
		AdditionalProperties: jsonschema.TrueSchema,
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

// SchemaMapFor returns the JSON schema inferred from type T as a map, or nil
// for interface types (e.g. `any`), whose zero value carries no type
// information to infer from. Like [SchemaAsMap], the returned map is freshly
// built on every call and belongs to the caller.
func SchemaMapFor[T any]() map[string]any {
	var v T
	if reflect.ValueOf(v).Kind() == reflect.Invalid {
		return nil
	}
	return SchemaAsMap(InferJSONSchema(v))
}

// SchemaAsMap converts json schema struct to a map (JSON representation). The
// map is rebuilt from JSON on every call, so the caller owns it and may mutate
// it in place; memoizing the result here would break callers that do.
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
