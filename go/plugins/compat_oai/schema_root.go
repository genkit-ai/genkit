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

// Some OpenAI-compatible hosts (OpenAI itself, and gateways that forward to
// it) only accept a json_schema response format whose root is type "object".
// For those hosts, a schema with any other root (array, enum, scalar, anyOf)
// is sent wrapped in an object, and the reply is unwrapped before the
// framework parses it. Only the openai plugin turns this on, through
// [hooks.SetObjectSchemaRoot].

// wrapperKey is the property that holds the caller's value inside the
// wrapper object.
const wrapperKey = "value"

// hasObjectRoot reports whether the schema's root is explicitly an object.
// Anything else, including properties without a type (which also allows
// non-object values), is wrapped.
func hasObjectRoot(schema map[string]any) bool {
	return schema["type"] == "object"
}

// wrapSchemaRoot wraps schema as the single required property of an object
// schema. $schema, $defs and definitions stay at the root, so references
// into the definitions still resolve.
//
// Any other local reference ("#", "#/items/...") pointed into the caller's
// schema, which is no longer the root. Strict hosts accept references only to
// top-level definitions, so each referenced subschema moves into $defs and
// both its old place and every reference to it point there instead.
func wrapSchemaRoot(schema map[string]any) map[string]any {
	h := newRefHoister(schema, true)
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
			wrapped[k] = h.walk(v, "#/definitions")
		case "$defs":
			if !h.walkDefs(v) {
				wrapped[k] = v
			}
		default:
			inner[k] = v
		}
	}
	wrapped["properties"] = map[string]any{wrapperKey: h.walk(inner, "#")}
	if len(h.defs) > 0 {
		wrapped["$defs"] = h.defs
	}
	return wrapped
}

// hoistDefRefs returns schema with every reference into the middle of a
// definition ("#/$defs/Person/properties/name") pointed at a new top-level
// definition instead, since strict hosts resolve only whole top-level
// definitions. A schema without such references comes back unchanged.
func hoistDefRefs(schema map[string]any) map[string]any {
	h := newRefHoister(schema, false)
	if len(h.targets) == 0 {
		return schema
	}
	out := make(map[string]any, len(schema)+1)
	for k, v := range schema {
		if k == "$defs" && h.walkDefs(v) {
			continue
		}
		out[k] = h.walk(v, "#/"+escapePointer(k))
	}
	out["$defs"] = h.defs
	return out
}

// dataKeywords hold data rather than subschemas, so a $ref inside them is not
// a reference.
var dataKeywords = map[string]bool{"const": true, "enum": true, "default": true, "examples": true}

// refHoister moves the subschemas that local references point at into $defs
// as it copies a schema.
type refHoister struct {
	// targets maps each referenced JSON pointer to its name in defs.
	targets map[string]string
	defs    map[string]any
}

// newRefHoister finds the references in schema that strict hosts can't
// follow: those into the middle of a definition and, when local is set, those
// anywhere else outside $defs and definitions.
func newRefHoister(schema map[string]any, local bool) *refHoister {
	h := &refHoister{targets: map[string]string{}, defs: map[string]any{}}
	taken := map[string]bool{}
	if defs, ok := schema["$defs"].(map[string]any); ok {
		for name := range defs {
			taken[name] = true
		}
	}
	var refs []string
	collectRefs(schema, func(ref string) {
		if !hoistable(ref, local) {
			return
		}
		if _, seen := h.targets[ref]; !seen && resolvePointer(schema, ref) {
			h.targets[ref] = ""
			refs = append(refs, ref)
		}
	})
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

// walk copies node, found at JSON pointer ptr in the caller's schema. A node
// that a reference points at goes into defs, and a reference to it is left in
// its place.
func (h *refHoister) walk(node any, ptr string) any {
	if name, ok := h.targets[ptr]; ok {
		h.defs[name] = h.copy(node, ptr)
		return map[string]any{"$ref": "#/$defs/" + name}
	}
	return h.copy(node, ptr)
}

func (h *refHoister) copy(node any, ptr string) any {
	switch n := node.(type) {
	case map[string]any:
		out := make(map[string]any, len(n))
		for k, v := range n {
			switch ref, isString := v.(string); {
			case k == "$ref" && isString:
				if name, ok := h.targets[ref]; ok {
					ref = "#/$defs/" + name
				}
				out[k] = ref
			case dataKeywords[k]:
				out[k] = v
			default:
				out[k] = h.walk(v, ptr+"/"+escapePointer(k))
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

// collectRefs calls found with every $ref in node.
func collectRefs(node any, found func(string)) {
	switch n := node.(type) {
	case map[string]any:
		for k, v := range n {
			switch ref, isString := v.(string); {
			case k == "$ref" && isString:
				found(ref)
			case dataKeywords[k]:
			default:
				collectRefs(v, found)
			}
		}
	case []any:
		for _, v := range n {
			collectRefs(v, found)
		}
	}
}

// hoistable reports whether ref is a local reference that strict hosts can't
// follow. A reference to a whole top-level definition is fine. One into the
// middle of a definition is not, and nor, when local is set, is any other
// local reference.
func hoistable(ref string, local bool) bool {
	if ref != "#" && !strings.HasPrefix(ref, "#/") {
		return false
	}
	if rest, ok := defsPath(ref); ok {
		return strings.Contains(rest, "/")
	}
	return local && ref != "#/$defs" && ref != "#/definitions"
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

// unwrapText returns the wrapped value's JSON from text, or text unchanged
// when it isn't a wrapper object. The value's raw JSON is kept as sent, so
// key order and number formatting survive.
func unwrapText(text string) string {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(text), &obj); err != nil {
		return text
	}
	value, ok := obj[wrapperKey]
	if !ok {
		return text
	}
	return string(value)
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
	started bool
	wrapped bool
	head    string
	held    string
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
