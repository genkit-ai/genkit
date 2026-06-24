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

package exp

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

// A small, dependency-free RFC 6902 (JSON Patch) / RFC 6901 (JSON Pointer)
// implementation. Genkit uses it to stream incremental changes to a session's
// custom state (see [AgentStreamChunk.CustomPatch]): the runtime [Diff]s the
// client-facing custom value against the last sent one and emits the result,
// and a client (or [AgentConnection]) [ApplyPatch]es each delta to keep a live
// local copy.
//
// [Diff] emits a valid RFC 6902 subset (only add / remove / replace; move /
// copy are optional optimizations we skip). [ApplyPatch] understands the full
// operation set for interoperability and is deliberately lenient so a stream of
// deltas stays robust.
//
// Both operate on JSON-shaped values: the map[string]any / []any / float64 /
// string / bool / nil tree produced by unmarshaling into an any. Inputs are
// normalized (round-tripped through JSON) on the way in, so any
// JSON-serializable Go value may be passed.

// Diff computes an RFC 6902 JSON Patch that transforms from into to.
//
// The diff is rooted at the document, so pointers are bare (e.g.
// "/agentStatus", "/items/0"). Only add, remove, and replace operations are
// emitted. Object members recurse (add for new keys, remove for deleted keys);
// arrays diff by index, appending with the end-of-array token "/-" and removing
// from the tail backwards so indices stay valid. A root-level change that
// cannot be expressed member-by-member (object↔array↔primitive) collapses to a
// single whole-document replace at path "".
//
// Object keys are visited in sorted order, so the patch is deterministic.
func Diff(from, to any) JSONPatch {
	return diffValues(normalizeJSON(from), normalizeJSON(to))
}

// diffValues diffs two already-normalized JSON values. The runtime uses it
// directly with a cached, already-normalized baseline to avoid re-normalizing
// it every turn.
func diffValues(from, to any) JSONPatch {
	var patch JSONPatch
	diffWalk(from, to, "", &patch)
	return patch
}

func diffWalk(from, to any, pointer string, patch *JSONPatch) {
	if jsonEqual(from, to) {
		return
	}

	// Both objects: recurse member by member.
	fromObj, fromIsObj := from.(map[string]any)
	toObj, toIsObj := to.(map[string]any)
	if fromIsObj && toIsObj {
		for _, key := range unionKeys(fromObj, toObj) {
			child := pointer + "/" + escapeToken(key)
			fv, inFrom := fromObj[key]
			tv, inTo := toObj[key]
			switch {
			case inFrom && !inTo:
				*patch = append(*patch, &JSONPatchOperation{Op: JSONPatchOpRemove, Path: child})
			case !inFrom && inTo:
				*patch = append(*patch, &JSONPatchOperation{Op: JSONPatchOpAdd, Path: child, Value: cloneJSON(tv)})
			default:
				diffWalk(fv, tv, child, patch)
			}
		}
		return
	}

	// Both arrays: recurse by index, then add/remove the tail difference.
	fromArr, fromIsArr := from.([]any)
	toArr, toIsArr := to.([]any)
	if fromIsArr && toIsArr {
		common := min(len(fromArr), len(toArr))
		for i := 0; i < common; i++ {
			diffWalk(fromArr[i], toArr[i], pointer+"/"+strconv.Itoa(i), patch)
		}
		// Appended elements use the "-" end-of-array token.
		for i := len(fromArr); i < len(toArr); i++ {
			*patch = append(*patch, &JSONPatchOperation{Op: JSONPatchOpAdd, Path: pointer + "/-", Value: cloneJSON(toArr[i])})
		}
		// Removals from the tail backwards so earlier indices stay valid.
		for i := len(fromArr) - 1; i >= len(toArr); i-- {
			*patch = append(*patch, &JSONPatchOperation{Op: JSONPatchOpRemove, Path: pointer + "/" + strconv.Itoa(i)})
		}
		return
	}

	// Type mismatch or differing primitives: replace at this location.
	*patch = append(*patch, &JSONPatchOperation{Op: JSONPatchOpReplace, Path: pointer, Value: cloneJSON(to)})
}

// ApplyPatch applies an RFC 6902 JSON Patch to document and returns the new
// value. The input is not mutated; a normalized clone is patched and returned.
// Operating on the root pointer ("") replaces or removes the whole document.
//
// It is lenient to keep streaming robust: an add or replace whose parent
// container is missing initializes it as an object, and a remove or replace of
// a missing member is a no-op. A test operation is honored and returns an error
// on mismatch. Other unknown operations also return an error.
func ApplyPatch(document any, patch JSONPatch) (any, error) {
	return applyOps(cloneJSON(normalizeJSON(document)), patch)
}

// applyOps applies patch to an already-normalized doc, mutating it in place
// where possible and returning the result (which may be a fresh value for
// root-level operations). The runtime/client pass a clone they own.
func applyOps(doc any, patch JSONPatch) (any, error) {
	for _, op := range patch {
		if op == nil {
			continue
		}
		var err error
		doc, err = applyOp(doc, op)
		if err != nil {
			return nil, err
		}
	}
	return doc, nil
}

func applyOp(doc any, op *JSONPatchOperation) (any, error) {
	tokens, err := parsePointer(op.Path)
	if err != nil {
		return nil, err
	}
	switch op.Op {
	case JSONPatchOpAdd:
		return setPath(doc, tokens, normalizeJSON(op.Value), true), nil
	case JSONPatchOpReplace:
		return setPath(doc, tokens, normalizeJSON(op.Value), false), nil
	case JSONPatchOpRemove:
		return removePath(doc, tokens), nil
	case JSONPatchOpTest:
		if !jsonEqual(getPath(doc, tokens), normalizeJSON(op.Value)) {
			return nil, fmt.Errorf("jsonpatch: test failed at %q", op.Path)
		}
		return doc, nil
	case JSONPatchOpMove:
		fromTokens, err := parsePointer(op.From)
		if err != nil {
			return nil, err
		}
		v := cloneJSON(getPath(doc, fromTokens))
		doc = removePath(doc, fromTokens)
		return setPath(doc, tokens, v, true), nil
	case JSONPatchOpCopy:
		fromTokens, err := parsePointer(op.From)
		if err != nil {
			return nil, err
		}
		return setPath(doc, tokens, cloneJSON(getPath(doc, fromTokens)), true), nil
	default:
		return nil, fmt.Errorf("jsonpatch: unsupported op %q", op.Op)
	}
}

// setPath sets value at tokens within node, creating missing intermediate
// objects, and returns the (possibly new) node. isAdd inserts into arrays
// (vs. replacing an element) and appends on the "-" token.
func setPath(node any, tokens []string, value any, isAdd bool) any {
	if len(tokens) == 0 {
		return value // root add/replace
	}
	// Lenient: initialize a missing/null container so member sets still land.
	if node == nil {
		node = map[string]any{}
	}
	token := tokens[0]
	if len(tokens) == 1 {
		return setMember(node, token, value, isAdd)
	}
	switch n := node.(type) {
	case map[string]any:
		child, ok := n[token]
		if !ok || !isContainer(child) {
			child = map[string]any{}
		}
		n[token] = setPath(child, tokens[1:], value, isAdd)
		return n
	case []any:
		idx, ok := arrayIndex(token, len(n), false)
		if !ok || idx >= len(n) {
			return n // lenient: nothing to descend into
		}
		if !isContainer(n[idx]) {
			n[idx] = map[string]any{}
		}
		n[idx] = setPath(n[idx], tokens[1:], value, isAdd)
		return n
	default:
		// Primitive where a container was expected: replace it with one.
		return setPath(map[string]any{}, tokens, value, isAdd)
	}
}

// setMember sets the leaf member token on node, returning the (possibly new)
// node.
func setMember(node any, token string, value any, isAdd bool) any {
	switch n := node.(type) {
	case map[string]any:
		n[token] = value
		return n
	case []any:
		if token == "-" {
			return append(n, value)
		}
		idx, ok := arrayIndex(token, len(n), isAdd)
		if !ok {
			return n
		}
		if isAdd {
			if idx > len(n) {
				return n
			}
			n = append(n, nil)
			copy(n[idx+1:], n[idx:])
			n[idx] = value
			return n
		}
		if idx >= 0 && idx < len(n) {
			n[idx] = value
		}
		return n
	default:
		return map[string]any{token: value}
	}
}

// removePath deletes the member at tokens within node and returns the
// (possibly new) node. Missing members are a no-op.
func removePath(node any, tokens []string) any {
	if len(tokens) == 0 {
		return nil // remove whole document
	}
	if node == nil {
		return nil
	}
	token := tokens[0]
	if len(tokens) == 1 {
		switch n := node.(type) {
		case map[string]any:
			delete(n, token)
			return n
		case []any:
			idx, ok := arrayIndex(token, len(n), false)
			if ok && idx >= 0 && idx < len(n) {
				return append(n[:idx], n[idx+1:]...)
			}
			return n
		default:
			return n
		}
	}
	switch n := node.(type) {
	case map[string]any:
		if child, ok := n[token]; ok {
			n[token] = removePath(child, tokens[1:])
		}
		return n
	case []any:
		idx, ok := arrayIndex(token, len(n), false)
		if ok && idx >= 0 && idx < len(n) {
			n[idx] = removePath(n[idx], tokens[1:])
		}
		return n
	default:
		return n
	}
}

// getPath reads the value at tokens, returning nil for any missing segment.
func getPath(node any, tokens []string) any {
	cur := node
	for _, t := range tokens {
		switch n := cur.(type) {
		case map[string]any:
			cur = n[t]
		case []any:
			idx, ok := arrayIndex(t, len(n), false)
			if !ok || idx < 0 || idx >= len(n) {
				return nil
			}
			cur = n[idx]
		default:
			return nil
		}
	}
	return cur
}

// arrayIndex parses an array reference token. The "-" end-of-array token
// resolves to length only when allowEnd is set (an insert/append position);
// otherwise it is rejected. Non-numeric or negative tokens are rejected.
func arrayIndex(token string, length int, allowEnd bool) (int, bool) {
	if token == "-" {
		if allowEnd {
			return length, true
		}
		return 0, false
	}
	idx, err := strconv.Atoi(token)
	if err != nil || idx < 0 {
		return 0, false
	}
	return idx, true
}

// --- JSON Pointer (RFC 6901) ---

// parsePointer splits a JSON Pointer into its reference tokens. The root
// pointer "" yields an empty slice.
func parsePointer(pointer string) ([]string, error) {
	if pointer == "" {
		return nil, nil
	}
	if pointer[0] != '/' {
		return nil, fmt.Errorf("jsonpatch: invalid JSON Pointer %q: must start with %q", pointer, "/")
	}
	parts := strings.Split(pointer[1:], "/")
	for i, p := range parts {
		parts[i] = unescapeToken(p)
	}
	return parts, nil
}

// escapeToken escapes a single reference token per RFC 6901 ("~" → "~0",
// "/" → "~1"). The "~" replacement runs first so a literal "/" does not become
// "~01".
func escapeToken(token string) string {
	return strings.ReplaceAll(strings.ReplaceAll(token, "~", "~0"), "/", "~1")
}

// unescapeToken reverses escapeToken ("~1" → "/", "~0" → "~"), with "~1" first
// so "~01" decodes to "~1" rather than "/".
func unescapeToken(token string) string {
	return strings.ReplaceAll(strings.ReplaceAll(token, "~1", "/"), "~0", "~")
}

// --- JSON value helpers ---

func isContainer(v any) bool {
	switch v.(type) {
	case map[string]any, []any:
		return true
	default:
		return false
	}
}

// unionKeys returns the sorted union of two objects' keys, for deterministic
// diff output.
func unionKeys(a, b map[string]any) []string {
	seen := make(map[string]struct{}, len(a)+len(b))
	keys := make([]string, 0, len(a)+len(b))
	for k := range a {
		if _, ok := seen[k]; !ok {
			seen[k] = struct{}{}
			keys = append(keys, k)
		}
	}
	for k := range b {
		if _, ok := seen[k]; !ok {
			seen[k] = struct{}{}
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return keys
}

// jsonEqual reports deep equality of two normalized JSON values.
func jsonEqual(a, b any) bool {
	return reflect.DeepEqual(a, b)
}

// normalizeJSON round-trips v through JSON so it becomes the canonical
// map[string]any / []any / float64 / string / bool / nil shape the diff and
// apply logic operate on. A value that cannot be marshaled is returned
// unchanged (best effort); JSON-shaped inputs never hit that path.
func normalizeJSON(v any) any {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		return v
	}
	return out
}

// cloneJSON deep-copies a normalized JSON value. Primitives are immutable and
// returned as-is; maps and slices are copied recursively so callers cannot
// alias each other's state.
func cloneJSON(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = cloneJSON(val)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = cloneJSON(val)
		}
		return out
	default:
		return v
	}
}
