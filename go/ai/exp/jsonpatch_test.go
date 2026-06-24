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
	"strings"
	"testing"
)

func TestDiff_Shapes(t *testing.T) {
	tests := []struct {
		name string
		from any
		to   any
		want JSONPatch
	}{
		{
			name: "equal yields empty patch",
			from: map[string]any{"a": 1},
			to:   map[string]any{"a": 1},
			want: nil,
		},
		{
			name: "add member",
			from: map[string]any{"a": 1},
			to:   map[string]any{"a": 1, "b": 2},
			want: JSONPatch{{Op: JSONPatchOpAdd, Path: "/b", Value: float64(2)}},
		},
		{
			name: "remove member",
			from: map[string]any{"a": 1, "b": 2},
			to:   map[string]any{"a": 1},
			want: JSONPatch{{Op: JSONPatchOpRemove, Path: "/b"}},
		},
		{
			name: "replace member",
			from: map[string]any{"a": 1},
			to:   map[string]any{"a": 2},
			want: JSONPatch{{Op: JSONPatchOpReplace, Path: "/a", Value: float64(2)}},
		},
		{
			name: "keys are sorted for determinism",
			from: map[string]any{},
			to:   map[string]any{"b": 1, "a": 2},
			want: JSONPatch{
				{Op: JSONPatchOpAdd, Path: "/a", Value: float64(2)},
				{Op: JSONPatchOpAdd, Path: "/b", Value: float64(1)},
			},
		},
		{
			name: "nested member recurses",
			from: map[string]any{"o": map[string]any{"x": 1}},
			to:   map[string]any{"o": map[string]any{"x": 2}},
			want: JSONPatch{{Op: JSONPatchOpReplace, Path: "/o/x", Value: float64(2)}},
		},
		{
			name: "array append uses end-of-array token",
			from: []any{"a"},
			to:   []any{"a", "b"},
			want: JSONPatch{{Op: JSONPatchOpAdd, Path: "/-", Value: "b"}},
		},
		{
			name: "array tail removal is backwards",
			from: []any{"a", "b", "c"},
			to:   []any{"a"},
			want: JSONPatch{
				{Op: JSONPatchOpRemove, Path: "/2"},
				{Op: JSONPatchOpRemove, Path: "/1"},
			},
		},
		{
			name: "array element change replaces by index",
			from: []any{"a", "b"},
			to:   []any{"a", "z"},
			want: JSONPatch{{Op: JSONPatchOpReplace, Path: "/1", Value: "z"}},
		},
		{
			name: "root type change collapses to whole-doc replace",
			from: map[string]any{"a": 1},
			to:   []any{1, 2},
			want: JSONPatch{{Op: JSONPatchOpReplace, Path: "", Value: []any{float64(1), float64(2)}}},
		},
		{
			name: "escapes pointer tokens",
			from: map[string]any{},
			to:   map[string]any{"a/b": 1, "m~n": 2},
			want: JSONPatch{
				{Op: JSONPatchOpAdd, Path: "/a~1b", Value: float64(1)},
				{Op: JSONPatchOpAdd, Path: "/m~0n", Value: float64(2)},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Diff(tc.from, tc.to)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Diff()\n  got  %s\n  want %s", patchString(got), patchString(tc.want))
			}
		})
	}
}

// TestDiffApply_RoundTrip is the core property: applying Diff(a, b) to a always
// yields b (normalized).
func TestDiffApply_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		a, b any
	}{
		{"both nil", nil, nil},
		{"nil to object", nil, map[string]any{"a": 1}},
		{"object to nil", map[string]any{"a": 1}, nil},
		{"add and remove and change", map[string]any{"a": 1, "b": 2}, map[string]any{"a": 9, "c": 3}},
		{"deep nesting", map[string]any{"o": map[string]any{"p": map[string]any{"q": 1}}}, map[string]any{"o": map[string]any{"p": map[string]any{"q": 2, "r": 3}}}},
		{"array grow", []any{1, 2}, []any{1, 2, 3, 4}},
		{"array shrink", []any{1, 2, 3, 4}, []any{1}},
		{"array of objects", []any{map[string]any{"k": 1}}, []any{map[string]any{"k": 2}, map[string]any{"k": 3}}},
		{"object holding array", map[string]any{"xs": []any{1, 2}}, map[string]any{"xs": []any{1, 9, 3}}},
		{"primitive change", "hello", "world"},
		{"type flip primitive to object", "x", map[string]any{"a": 1}},
		{"agentStatus style", map[string]any{"agentStatus": "step 1"}, map[string]any{"agentStatus": "step 3 of 12", "done": false}},
		{"null value", map[string]any{"a": 1, "b": nil}, map[string]any{"a": nil, "b": 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			patch := Diff(tc.a, tc.b)
			got, err := ApplyPatch(tc.a, patch)
			if err != nil {
				t.Fatalf("ApplyPatch: %v", err)
			}
			want := normalizeJSON(tc.b)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("round trip mismatch\n  got  %#v\n  want %#v\n  patch %s", got, want, patchString(patch))
			}
		})
	}
}

// TestDiff_NullOperandSurvivesWire guards against json omitempty dropping a
// null operand: a replace (or add) to JSON null must serialize with an explicit
// "value":null, otherwise a peer applier (e.g. the JS client) reads it as absent
// and removes the member instead of setting it to null. The in-memory round-trip
// test cannot catch this because a missing value decodes back to nil Go-side.
func TestDiff_NullOperandSurvivesWire(t *testing.T) {
	patch := Diff(
		map[string]any{"a": 1, "b": 2},
		map[string]any{"a": nil, "b": 2},
	)
	wire, err := json.Marshal(patch)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(wire), `"value":null`) {
		t.Fatalf("null operand dropped from wire: %s", wire)
	}

	// Decode as a peer would and apply: the member must be present and null.
	var decoded JSONPatch
	if err := json.Unmarshal(wire, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	got, err := ApplyPatch(map[string]any{"a": 1, "b": 2}, decoded)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}
	gotMap, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("apply result is %T, want map", got)
	}
	if v, present := gotMap["a"]; !present || v != nil {
		t.Errorf("member \"a\" = %v (present %v), want present and null", v, present)
	}
}

func TestApplyPatch_RootOps(t *testing.T) {
	// Whole-document replace re-bases any prior value.
	got, err := ApplyPatch(map[string]any{"old": true}, JSONPatch{
		{Op: JSONPatchOpReplace, Path: "", Value: map[string]any{"new": 1}},
	})
	if err != nil {
		t.Fatalf("ApplyPatch: %v", err)
	}
	if want := map[string]any{"new": float64(1)}; !reflect.DeepEqual(got, want) {
		t.Errorf("root replace = %#v, want %#v", got, want)
	}

	// Replace at root onto a nil document initializes it.
	got, err = ApplyPatch(nil, JSONPatch{{Op: JSONPatchOpReplace, Path: "", Value: "v"}})
	if err != nil {
		t.Fatalf("ApplyPatch: %v", err)
	}
	if got != "v" {
		t.Errorf("root replace onto nil = %#v, want %q", got, "v")
	}
}

func TestApplyPatch_Lenient(t *testing.T) {
	// add onto a missing/nil document initializes the root container.
	got, err := ApplyPatch(nil, JSONPatch{{Op: JSONPatchOpAdd, Path: "/agentStatus", Value: "x"}})
	if err != nil {
		t.Fatalf("ApplyPatch: %v", err)
	}
	if want := map[string]any{"agentStatus": "x"}; !reflect.DeepEqual(got, want) {
		t.Errorf("add onto nil = %#v, want %#v", got, want)
	}

	// Missing intermediate parents are created as objects.
	got, err = ApplyPatch(map[string]any{}, JSONPatch{{Op: JSONPatchOpAdd, Path: "/a/b/c", Value: 1}})
	if err != nil {
		t.Fatalf("ApplyPatch: %v", err)
	}
	want := map[string]any{"a": map[string]any{"b": map[string]any{"c": float64(1)}}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("nested add = %#v, want %#v", got, want)
	}

	// remove of a missing member is a no-op.
	got, err = ApplyPatch(map[string]any{"a": 1}, JSONPatch{{Op: JSONPatchOpRemove, Path: "/missing"}})
	if err != nil {
		t.Fatalf("ApplyPatch: %v", err)
	}
	if w := map[string]any{"a": float64(1)}; !reflect.DeepEqual(got, w) {
		t.Errorf("remove missing = %#v, want %#v", got, w)
	}

	// replace of a missing member is a no-op (not an error).
	if _, err := ApplyPatch(map[string]any{"a": 1}, JSONPatch{{Op: JSONPatchOpReplace, Path: "/missing", Value: 2}}); err != nil {
		t.Errorf("replace missing should be a no-op, got error: %v", err)
	}
}

func TestApplyPatch_DoesNotMutateInput(t *testing.T) {
	in := map[string]any{"a": float64(1)}
	if _, err := ApplyPatch(in, JSONPatch{{Op: JSONPatchOpReplace, Path: "/a", Value: 2}}); err != nil {
		t.Fatalf("ApplyPatch: %v", err)
	}
	if in["a"] != float64(1) {
		t.Errorf("input was mutated: %#v", in)
	}
}

func TestApplyPatch_TestOp(t *testing.T) {
	doc := map[string]any{"a": 1}
	if _, err := ApplyPatch(doc, JSONPatch{{Op: JSONPatchOpTest, Path: "/a", Value: 1}}); err != nil {
		t.Errorf("matching test should pass, got: %v", err)
	}
	if _, err := ApplyPatch(doc, JSONPatch{{Op: JSONPatchOpTest, Path: "/a", Value: 2}}); err == nil {
		t.Error("mismatching test should error")
	}
}

func TestApplyPatch_MoveCopy(t *testing.T) {
	got, err := ApplyPatch(map[string]any{"a": 1}, JSONPatch{{Op: JSONPatchOpMove, From: "/a", Path: "/b"}})
	if err != nil {
		t.Fatalf("move: %v", err)
	}
	if want := map[string]any{"b": float64(1)}; !reflect.DeepEqual(got, want) {
		t.Errorf("move = %#v, want %#v", got, want)
	}

	got, err = ApplyPatch(map[string]any{"a": 1}, JSONPatch{{Op: JSONPatchOpCopy, From: "/a", Path: "/b"}})
	if err != nil {
		t.Fatalf("copy: %v", err)
	}
	if want := map[string]any{"a": float64(1), "b": float64(1)}; !reflect.DeepEqual(got, want) {
		t.Errorf("copy = %#v, want %#v", got, want)
	}
}

func TestApplyPatch_ArrayInsertAndAppend(t *testing.T) {
	// "-" appends.
	got, err := ApplyPatch([]any{"a"}, JSONPatch{{Op: JSONPatchOpAdd, Path: "/-", Value: "b"}})
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	if want := []any{"a", "b"}; !reflect.DeepEqual(got, want) {
		t.Errorf("append = %#v, want %#v", got, want)
	}

	// numeric index inserts.
	got, err = ApplyPatch([]any{"a", "c"}, JSONPatch{{Op: JSONPatchOpAdd, Path: "/1", Value: "b"}})
	if err != nil {
		t.Fatalf("insert: %v", err)
	}
	if want := []any{"a", "b", "c"}; !reflect.DeepEqual(got, want) {
		t.Errorf("insert = %#v, want %#v", got, want)
	}
}

func TestParsePointer(t *testing.T) {
	tests := []struct {
		in      string
		want    []string
		wantErr bool
	}{
		{"", nil, false},
		{"/a", []string{"a"}, false},
		{"/a/b", []string{"a", "b"}, false},
		{"/a~1b", []string{"a/b"}, false},
		{"/m~0n", []string{"m~n"}, false},
		{"/~01", []string{"~1"}, false}, // ~01 decodes to ~1, not /
		{"bad", nil, true},
	}
	for _, tc := range tests {
		got, err := parsePointer(tc.in)
		if (err != nil) != tc.wantErr {
			t.Errorf("parsePointer(%q) err = %v, wantErr %v", tc.in, err, tc.wantErr)
			continue
		}
		if !tc.wantErr && !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parsePointer(%q) = %#v, want %#v", tc.in, got, tc.want)
		}
	}
}

// patchString renders a patch compactly for test failure messages.
func patchString(p JSONPatch) string {
	if len(p) == 0 {
		return "[]"
	}
	out := "["
	for i, op := range p {
		if i > 0 {
			out += ", "
		}
		out += fmt.Sprintf("%s %s", op.Op, op.Path)
		if op.From != "" {
			out += " from=" + op.From
		}
		if op.Value != nil {
			out += fmt.Sprintf(" value=%v", op.Value)
		}
	}
	return out + "]"
}
