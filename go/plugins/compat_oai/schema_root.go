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
	"strings"
)

// Some OpenAI-compatible hosts (OpenAI itself, and gateways that forward to
// it) only accept a json_schema response format whose root is type "object".
// For those hosts, a schema with any other root (array, enum, scalar, anyOf)
// is sent wrapped in an object, and the reply is unwrapped before the
// framework parses it. See [OpenAICompatible.ObjectSchemaRoot].

// wrapperKey is the property that holds the caller's value inside the
// wrapper object.
const wrapperKey = "value"

// hasObjectRoot reports whether the schema's root is an object, which every
// host accepts.
func hasObjectRoot(schema map[string]any) bool {
	t, hasType := schema["type"]
	if t == "object" {
		return true
	}
	_, hasProperties := schema["properties"]
	return !hasType && hasProperties
}

// wrapSchemaRoot wraps schema as the single required property of an object
// schema. $schema, $defs and definitions stay at the root, so references
// into the definitions still resolve.
func wrapSchemaRoot(schema map[string]any) map[string]any {
	inner := make(map[string]any, len(schema))
	wrapped := map[string]any{
		"type":                 "object",
		"required":             []any{wrapperKey},
		"additionalProperties": false,
	}
	for k, v := range schema {
		switch k {
		case "$schema", "$defs", "definitions":
			wrapped[k] = v
		default:
			inner[k] = v
		}
	}
	wrapped["properties"] = map[string]any{wrapperKey: inner}
	return wrapped
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

var wrapperPrefix = regexp.MustCompile(`^\s*\{\s*"` + wrapperKey + `"\s*:\s*`)

// maxPrefixWait is how much text the stream unwrapper buffers, waiting to
// see the wrapper prefix, before it gives up and passes text through.
const maxPrefixWait = 40

// streamUnwrapper strips the {"value": prefix and the closing brace from
// streamed text, so chunks read as the caller's value (such as [{...}, for
// an array) and the format's chunk parser sees what it expects.
//
// Text is held back until the prefix is seen. Trailing whitespace and braces
// are then held until more text arrives, because the last brace belongs to
// the wrapper only if nothing follows it; flush drops it.
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
	if loc := wrapperPrefix.FindStringIndex(u.head); loc != nil {
		u.started, u.wrapped = true, true
		rest := u.head[loc[1]:]
		u.head = ""
		return u.release(rest)
	}
	// Not a wrapper, or not enough text yet to tell: keep buffering unless it
	// can no longer become one.
	lead := strings.TrimLeft(u.head, " \t\r\n")
	if lead != "" && (lead[0] != '{' || len(u.head) > maxPrefixWait) {
		u.started = true
		out := u.head
		u.head = ""
		return u.release(out)
	}
	return ""
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
