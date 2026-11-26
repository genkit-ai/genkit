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

package compat_oai

import (
	"encoding/json"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"github.com/firebase/genkit/go/plugins/compat_oai/internal/hooks"
)

func init() {
	hooks.SetObjectSchemaRoot = func(o any) {
		o.(*OpenAICompatible).objectSchemaRoot = true
	}
}

// wrapperKey is the property that holds the caller's value inside the
// wrapper object.
const wrapperKey = "value"

// hasObjectRoot reports whether the schema's root is explicitly an object,
// either directly or through a bare root $ref to a top-level definition,
// which OpenAI accepts unwrapped. Anything else, including properties without
// a type (which also allows non-object values), is wrapped.
//
// A root $ref with sibling keywords (description, title, ...) is wrapped,
// because strict mode rejects those siblings; see [wrapSchemaRoot]. So is a
// $ref to an escaped name (~0, ~1), which OpenAI does not resolve wrapped or
// not.
func hasObjectRoot(schema map[string]any) bool {
	seen := map[string]bool{}
	node := schema
	allowed := []string{"$ref", "$schema", "$defs", "definitions"}
	for node != nil {
		if t, ok := node["type"]; ok {
			return t == "object"
		}
		ref, ok := node["$ref"].(string)
		if !ok || seen[ref] {
			return false
		}
		for k := range node {
			if !slices.Contains(allowed, k) {
				return false
			}
		}
		allowed = []string{"$ref"}
		seen[ref] = true
		keyword, name, ok := topLevelDef(ref)
		if !ok {
			return false
		}
		defs, _ := schema[keyword].(map[string]any)
		node, _ = defs[name].(map[string]any)
	}
	return false
}

// topLevelDef splits a reference to a whole top-level definition
// ("#/$defs/Name" or "#/definitions/Name") into its keyword and name. Names
// with escapes are not matched.
func topLevelDef(ref string) (keyword, name string, ok bool) {
	for _, kw := range []string{"$defs", "definitions"} {
		if rest, found := strings.CutPrefix(ref, "#/"+kw+"/"); found && rest != "" && !strings.ContainsAny(rest, "/~") {
			return kw, rest, true
		}
	}
	return "", "", false
}

// wrapSchemaRoot wraps schema as the single required property of an object
// schema. $schema, $defs and definitions stay at the root, so references
// into the definitions still resolve.
//
// Any other local reference ("#", "#/items/...") pointed into the caller's
// schema, which is no longer the root. Strict hosts accept references only to
// top-level definitions, so each referenced subschema moves into $defs and
// both its old place and every reference to it point there instead.
//
// Strict mode also rejects a $ref with sibling keywords, so a root $ref that
// has them moves into a one-branch anyOf, which takes siblings.
func wrapSchemaRoot(schema map[string]any) map[string]any {
	h := newRefHoister(schema)
	inner := make(map[string]any, len(schema))
	wrapped := map[string]any{
		"type":                 "object",
		"required":             []any{wrapperKey},
		"additionalProperties": false,
	}
	for k, v := range schema {
		switch k {
		case "$schema":
			wrapped[k] = v
		case "definitions":
			wrapped[k] = h.walkNames(v, "#/definitions")
		case "$defs":
			if !h.walkDefs(v) {
				wrapped[k] = v
			}
		default:
			inner[k] = v
		}
	}
	value := h.walk(inner, "#")
	if v, ok := value.(map[string]any); ok {
		if ref, isRef := v["$ref"]; isRef && len(v) > 1 && v["anyOf"] == nil {
			delete(v, "$ref")
			v["anyOf"] = []any{map[string]any{"$ref": ref}}
		}
	}
	wrapped["properties"] = map[string]any{wrapperKey: value}
	if len(h.defs) > 0 {
		wrapped["$defs"] = h.defs
	}
	return wrapped
}

// dataKeywords hold data rather than subschemas, so a $ref inside them is not
// a reference.
var dataKeywords = map[string]bool{"const": true, "enum": true, "default": true, "examples": true}

// nameMaps are the keywords whose values map names to subschemas. Keys there
// are names, so a property called "default" or "$ref" is a schema like any
// other.
var nameMaps = map[string]bool{
	"properties":        true,
	"patternProperties": true,
	"$defs":             true,
	"definitions":       true,
	"dependentSchemas":  true,
}

// refHoister moves the subschemas that local references point at into $defs
// as it copies a schema.
type refHoister struct {
	// targets maps each referenced JSON pointer to its name in defs.
	targets map[string]string
	defs    map[string]any
}

// newRefHoister finds the local references in schema that strict hosts
// can't follow once it is wrapped: all but those to a whole top-level
// definition.
func newRefHoister(schema map[string]any) *refHoister {
	h := &refHoister{targets: map[string]string{}, defs: map[string]any{}}
	taken := map[string]bool{}
	if defs, ok := schema["$defs"].(map[string]any); ok {
		for name := range defs {
			taken[name] = true
		}
	}
	var refs []string
	collectRefs(schema, func(ref string) {
		if !hoistable(ref) {
			return
		}
		if _, seen := h.targets[ref]; !seen && resolvePointer(schema, ref) {
			h.targets[ref] = ""
			refs = append(refs, ref)
		}
	}, false)
	slices.Sort(refs)
	for _, ref := range refs {
		h.targets[ref] = uniqueDefName(ref, taken)
	}
	return h
}

// walkDefs copies the $defs value v into defs, and reports false when v isn't
// a map of definitions.
func (h *refHoister) walkDefs(v any) bool {
	defs, ok := v.(map[string]any)
	if !ok {
		return false
	}
	for name, def := range defs {
		h.defs[name] = h.walk(def, "#/$defs/"+escapePointer(name))
	}
	return true
}

// walk copies the schema node, found at JSON pointer ptr in the caller's
// schema. A node that a reference points at goes into defs, and a reference
// to it is left in its place.
func (h *refHoister) walk(node any, ptr string) any {
	return h.hoist(node, ptr, false)
}

// walkNames is walk for a map of names to subschemas, such as the value of
// properties.
func (h *refHoister) walkNames(node any, ptr string) any {
	return h.hoist(node, ptr, true)
}

func (h *refHoister) hoist(node any, ptr string, names bool) any {
	if name, ok := h.targets[ptr]; ok {
		h.defs[name] = h.copy(node, ptr, names)
		return map[string]any{"$ref": "#/$defs/" + name}
	}
	return h.copy(node, ptr, names)
}

func (h *refHoister) copy(node any, ptr string, names bool) any {
	switch n := node.(type) {
	case map[string]any:
		out := make(map[string]any, len(n))
		for k, v := range n {
			at := ptr + "/" + escapePointer(k)
			switch ref, isString := v.(string); {
			case names:
				out[k] = h.walk(v, at)
			case k == "$ref" && isString:
				if name, ok := h.targets[ref]; ok {
					ref = "#/$defs/" + name
				}
				out[k] = ref
			case dataKeywords[k]:
				out[k] = v
			default:
				out[k] = h.hoist(v, at, nameMaps[k])
			}
		}
		return out
	case []any:
		out := make([]any, len(n))
		for i, v := range n {
			out[i] = h.walk(v, ptr+"/"+strconv.Itoa(i))
		}
		return out
	default:
		return node
	}
}

// collectRefs calls found with every $ref in node. names says node is a map
// of names to subschemas (the value of properties, say), not a schema.
func collectRefs(node any, found func(string), names bool) {
	switch n := node.(type) {
	case map[string]any:
		for k, v := range n {
			switch ref, isString := v.(string); {
			case names:
				collectRefs(v, found, false)
			case k == "$ref" && isString:
				found(ref)
			case dataKeywords[k]:
			default:
				collectRefs(v, found, nameMaps[k])
			}
		}
	case []any:
		for _, v := range n {
			collectRefs(v, found, false)
		}
	}
}

// hoistable reports whether ref is a local reference that strict hosts can't
// follow once the schema is wrapped. A reference to a whole top-level
// definition is fine; one into the middle of a definition or anywhere else in
// the schema is not.
func hoistable(ref string) bool {
	if ref != "#" && !strings.HasPrefix(ref, "#/") {
		return false
	}
	if rest, ok := defsPath(ref); ok {
		return strings.Contains(rest, "/")
	}
	return ref != "#/$defs" && ref != "#/definitions"
}

// defsPath returns what follows "#/$defs/" or "#/definitions/" in ref.
func defsPath(ref string) (string, bool) {
	for _, prefix := range []string{"#/$defs/", "#/definitions/"} {
		if rest, ok := strings.CutPrefix(ref, prefix); ok {
			return rest, true
		}
	}
	return "", false
}

// resolvePointer reports whether the JSON pointer ref names a schema inside
// schema.
func resolvePointer(schema map[string]any, ref string) bool {
	var node any = schema
	for _, seg := range strings.Split(strings.TrimPrefix(ref, "#"), "/")[1:] {
		seg = strings.ReplaceAll(strings.ReplaceAll(seg, "~1", "/"), "~0", "~")
		switch n := node.(type) {
		case map[string]any:
			next, ok := n[seg]
			if !ok {
				return false
			}
			node = next
		case []any:
			i, err := strconv.Atoi(seg)
			if err != nil || i < 0 || i >= len(n) {
				return false
			}
			node = n[i]
		default:
			return false
		}
	}
	_, isSchema := node.(map[string]any)
	return isSchema
}

func escapePointer(key string) string {
	return strings.ReplaceAll(strings.ReplaceAll(key, "~", "~0"), "/", "~1")
}

// uniqueDefName names the definition a referenced pointer moves to, avoiding
// the names already taken. A pointer into a definition is named after its
// path there ("Person_properties_name"), and any other after its path under
// the wrapper ("value_items").
func uniqueDefName(ref string, taken map[string]bool) string {
	path, inDefs := defsPath(ref)
	if !inDefs {
		path = wrapperKey + strings.TrimPrefix(ref, "#")
	}
	base := strings.Map(func(r rune) rune {
		if r == '_' || r == '-' || unicode.IsLetter(r) || unicode.IsDigit(r) {
			return r
		}
		return '_'
	}, path)
	name := base
	for i := 2; taken[name]; i++ {
		name = base + "_" + strconv.Itoa(i)
	}
	taken[name] = true
	return name
}

// fenced matches a reply in a markdown code fence; the closing fence may be
// cut off.
var fenced = regexp.MustCompile("(?s)^\\s*```(?:json)?\\s*(.*?)\\s*(?:```\\s*)?$")

// unwrapText returns the wrapped value's JSON from text, or text unchanged
// when it isn't a wrapper object (a host that ignored the schema may reply
// with the bare value, or with an object that merely has a value key).
//
// The wrapper is stripped as text, as [streamUnwrapper] does, so a reply cut
// off mid-value is unwrapped as far as it goes and the value keeps its key
// order and number formatting.
func unwrapText(text string) string {
	body := text
	if m := fenced.FindStringSubmatch(text); m != nil {
		body = m[1]
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &obj); err == nil {
		if _, ok := obj[wrapperKey]; !ok || len(obj) != 1 {
			return text
		}
	} else if _, invalid := err.(*json.SyntaxError); !invalid {
		// Valid JSON, but not an object.
		return text
	}
	var u streamUnwrapper
	unwrapped := u.push(body) + u.flush()
	if unwrapped == body {
		return text
	}
	return unwrapped
}

// prefixTokens is the wrapper's opening, as the tokens it is made of.
// Whitespace may sit before and between them.
var prefixTokens = []string{"{", `"` + wrapperKey + `"`, ":"}

// prefixMatch is how far text matches the wrapper's opening.
type prefixMatch int

const (
	prefixNo      prefixMatch = iota // text cannot become the opening
	prefixPartial                    // text could still become the opening
	prefixFull                       // the opening is all there
)

// matchPrefix matches the start of text against the wrapper's opening. On a
// full match it also returns the opening's length, trailing whitespace
// included.
func matchPrefix(text string) (prefixMatch, int) {
	i := 0
	skipSpace := func() {
		for i < len(text) && unicode.IsSpace(rune(text[i])) {
			i++
		}
	}
	for _, token := range prefixTokens {
		skipSpace()
		rest := text[i:min(i+len(token), len(text))]
		if rest != token {
			if strings.HasPrefix(token, rest) && i+len(rest) == len(text) {
				return prefixPartial, 0
			}
			return prefixNo, 0
		}
		i += len(token)
	}
	skipSpace()
	return prefixFull, i
}

// streamUnwrapper strips the {"value": prefix and the closing brace from
// streamed text, so chunks read as the caller's value (such as [{...}, for
// an array) and the format's chunk parser sees what it expects.
//
// Text is held back for as long as it could still be the start of the
// wrapper, so the result does not depend on how the stream is split.
// Trailing whitespace and braces are then held until more text arrives,
// because the last brace belongs to the wrapper only if nothing follows it;
// flush drops it. A reply that is not a wrapper passes through unchanged.
type streamUnwrapper struct {
	started      bool
	wrapped      bool
	valueStarted bool
	head         string
	held         string
}

// push takes a streamed delta and returns the text to emit for it, which
// may be empty.
func (u *streamUnwrapper) push(delta string) string {
	if u.started {
		return u.release(delta)
	}
	u.head += delta
	match, n := matchPrefix(u.head)
	if match == prefixPartial {
		return ""
	}
	u.started = true
	out := u.head
	u.head = ""
	if match == prefixFull {
		u.wrapped = true
		out = out[n:]
	}
	return u.release(out)
}

func (u *streamUnwrapper) release(text string) string {
	all := u.held + text
	// Whitespace after the colon belongs to the wrapper too, however much of
	// it arrives after the opening matched.
	if u.wrapped && !u.valueStarted {
		all = strings.TrimLeftFunc(all, unicode.IsSpace)
		u.valueStarted = all != ""
	}
	cut := len(strings.TrimRight(all, " \t\r\n}"))
	u.held = all[cut:]
	return all[:cut]
}

// passThrough gives up unwrapping, for a reply that turns out not to be the
// output (text before a tool call), and returns the text held back so far.
// A preamble that never matched the wrapper prefix comes back exactly as it
// streamed; one that did has already lost the prefix, which a preamble
// shaped like the wrapper object is not expected to need.
func (u *streamUnwrapper) passThrough() string {
	if !u.started {
		out := u.head
		u.head = ""
		return out
	}
	out := u.held
	u.held = ""
	return out
}

// flush returns what is left at the end of the stream: anything held except
// the wrapper's closing brace and the whitespace around it.
func (u *streamUnwrapper) flush() string {
	if !u.started {
		out := u.head
		u.head = ""
		return out
	}
	text := u.held
	u.held = ""
	if !u.wrapped {
		return text
	}
	if i := strings.LastIndex(text, "}"); i >= 0 {
		text = text[:i]
	}
	return strings.TrimRight(text, " \t\r\n")
}
