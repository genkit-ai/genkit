// Copyright 2024 Google LLC
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

package base

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp"
)

func TestExtractJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    any
		wantErr bool
	}{
		{
			name:  "complete object",
			input: `{"name": "John", "age": 30}`,
			want:  map[string]any{"name": "John", "age": float64(30)},
		},
		{
			name:  "complete array",
			input: `[1, 2, 3]`,
			want:  []any{float64(1), float64(2), float64(3)},
		},
		{
			name:  "object with prefix text",
			input: `Some text before {"name": "Jane"}`,
			want:  map[string]any{"name": "Jane"},
		},
		{
			name:  "incomplete object",
			input: `{"name": "John", "age": 3`,
			want:  map[string]any{"name": "John", "age": float64(3)},
		},
		{
			name:  "incomplete object with partial string",
			input: `{"name": "Jo`,
			want:  map[string]any{"name": "Jo"},
		},
		{
			name:  "incomplete nested object",
			input: `{"person": {"name": "John"`,
			want:  map[string]any{"person": map[string]any{"name": "John"}},
		},
		{
			name:  "object with trailing comma",
			input: `{"name": "John",`,
			want:  map[string]any{"name": "John"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractJSON(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ExtractJSON() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtractItems(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		cursor     int
		wantItems  []any
		wantCursor int
	}{
		{
			name:       "complete array",
			input:      `[{"name": "John"}, {"name": "Jane"}]`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}, map[string]any{"name": "Jane"}},
			wantCursor: 35,
		},
		{
			name:       "partial array - first item",
			input:      `[{"name": "John"}`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}},
			wantCursor: 17,
		},
		{
			name:       "partial array - incomplete second item",
			input:      `[{"name": "John"}, {"name": "J`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}},
			wantCursor: 17,
		},
		{
			name:       "incremental parsing from cursor",
			input:      `[{"name": "John"}, {"name": "Jane"}]`,
			cursor:     18,
			wantItems:  []any{map[string]any{"name": "Jane"}},
			wantCursor: 35,
		},
		{
			name:       "array with prefix text",
			input:      `Some text [{"name": "John"}]`,
			cursor:     0,
			wantItems:  []any{map[string]any{"name": "John"}},
			wantCursor: 27,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractItems(tt.input, tt.cursor)
			if diff := cmp.Diff(tt.wantItems, result.Items); diff != "" {
				t.Errorf("ExtractItems() items mismatch (-want +got):\n%s", diff)
			}
			if result.Cursor != tt.wantCursor {
				t.Errorf("ExtractItems() cursor = %v, want %v", result.Cursor, tt.wantCursor)
			}
		})
	}
}

func TestCompleteJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "unclosed object",
			input: `{"name": "John"`,
			want:  `{"name": "John"}`,
		},
		{
			name:  "unclosed array",
			input: `[1, 2, 3`,
			want:  `[1, 2, 3]`,
		},
		{
			name:  "unclosed string",
			input: `{"name": "John`,
			want:  `{"name": "John"}`,
		},
		{
			name:  "nested unclosed",
			input: `{"person": {"name": "John"`,
			want:  `{"person": {"name": "John"}}`,
		},
		{
			name:  "trailing comma",
			input: `{"name": "John",`,
			want:  `{"name": "John"}`,
		},
		{
			name:  "empty string",
			input: "",
			want:  "{}",
		},
		{
			// Containers must be closed innermost first. Closing by type
			// instead produced `[{"a": 1]}` and broke every streamed array of
			// objects, the most common structured output shape.
			name:  "object unclosed inside array",
			input: `[{"a": 1`,
			want:  `[{"a": 1}]`,
		},
		{
			name:  "second object unclosed inside array",
			input: `[{"a": 1}, {"b": 2`,
			want:  `[{"a": 1}, {"b": 2}]`,
		},
		{
			name:  "array unclosed inside object",
			input: `{"a": [1, 2`,
			want:  `{"a": [1, 2]}`,
		},
		{
			name:  "deeply alternating containers",
			input: `{"a": [{"b": [{"c": "d`,
			want:  `{"a": [{"b": [{"c": "d"}]}]}`,
		},
		{
			name:  "empty containers",
			input: `{"a": [{`,
			want:  `{"a": [{}]}`,
		},
		{
			// A trailing backslash would escape the quote used to close the
			// string, leaving it open.
			name:  "dangling escape",
			input: `{"a": "x\`,
			want:  `{"a": "x"}`,
		},
		{
			name:  "truncated unicode escape",
			input: `{"a": "x\u26`,
			want:  `{"a": "x"}`,
		},
		{
			name:  "complete escapes are kept",
			input: `{"a": "x\"y\\`,
			want:  `{"a": "x\"y\\"}`,
		},
		{
			name:  "escaped quote does not close the string",
			input: `{"a": "x\"`,
			want:  `{"a": "x\""}`,
		},
		{
			name:  "rune split by chunk boundary",
			input: "{\"a\": \"hi \xe4\xb8",
			want:  `{"a": "hi "}`,
		},
		{
			// A key with no value yet cannot be completed without inventing
			// one, so the whole member is dropped.
			name:  "key without colon",
			input: `{"a": 1, "b"`,
			want:  `{"a": 1}`,
		},
		{
			name:  "key without value",
			input: `{"a": 1, "b":`,
			want:  `{"a": 1}`,
		},
		{
			name:  "partial key",
			input: `{"a": 1, "b`,
			want:  `{"a": 1}`,
		},
		{
			name:  "only a key",
			input: `{"a"`,
			want:  `{}`,
		},
		{
			name:  "missing value in nested object",
			input: `{"a": {"b": 1, "c":`,
			want:  `{"a": {"b": 1}}`,
		},
		{
			// Unambiguous: nothing else in JSON starts with "tr".
			name:  "truncated keyword",
			input: `{"a": tr`,
			want:  `{"a": true}`,
		},
		{
			name:  "truncated keyword false",
			input: `[fals`,
			want:  `[false]`,
		},
		{
			name:  "truncated keyword null",
			input: `{"a": n`,
			want:  `{"a": null}`,
		},
		{
			name:  "complete keyword",
			input: `{"a": true`,
			want:  `{"a": true}`,
		},
		{
			name:  "keyword truncated by a delimiter is not a keyword",
			input: `{"a": tr,`,
			want:  `{}`,
		},
		{
			name:  "truncated number",
			input: `{"a": 1.`,
			want:  `{}`,
		},
		{
			name:  "lone minus sign",
			input: `[1, -`,
			want:  `[1]`,
		},
		{
			name:  "truncated exponent",
			input: `[1e`,
			want:  `[]`,
		},
		{
			name:  "float value",
			input: `{"a": 1.5`,
			want:  `{"a": 1.5}`,
		},
		{
			name:  "trailing comma in array",
			input: `[1, 2,`,
			want:  `[1, 2]`,
		},
		{
			name:  "root scalar",
			input: `123`,
			want:  `123`,
		},
		{
			name:  "root string",
			input: `"abc`,
			want:  `"abc"`,
		},
		{
			name:  "nothing salvageable",
			input: `xy`,
			want:  `{}`,
		},
		{
			name:  "trailing text after complete value",
			input: `{"a": 1} and then some prose`,
			want:  `{"a": 1}`,
		},
		{
			name:  "mismatched closer",
			input: `[1, 2}`,
			want:  `[1, 2]`,
		},
		{
			name:  "structural characters inside strings are ignored",
			input: `{"a": "}]{[", "b": [{"c": "\"[`,
			want:  `{"a": "}]{[", "b": [{"c": "\"["}]}`,
		},
		{
			// Malformed rather than truncated: a container cannot open where a
			// key belongs, so the scan ends and takes the rest with it.
			name:  "object opened where a key belongs",
			input: `{"a": 1, {`,
			want:  `{"a": 1}`,
		},
		{
			name:  "array opened where a key belongs",
			input: `{"a": 1, [`,
			want:  `{"a": 1}`,
		},
		{
			name:  "container opened immediately inside an object",
			input: `{{`,
			want:  `{}`,
		},
		{
			// The closer arrives, but a trailing comma means the container is
			// not closable at that point, so it is closed at the last value.
			name:  "trailing comma before an explicit closer",
			input: `[1, 2,]`,
			want:  `[1, 2]`,
		},
		{
			name:  "trailing comma before an explicit brace",
			input: `{"a": 1,}`,
			want:  `{"a": 1}`,
		},
		{
			name:  "colon where a key belongs",
			input: `{:}`,
			want:  `{}`,
		},
		{
			name:  "key with no value before the closer",
			input: `{"a"}`,
			want:  `{}`,
		},
		{
			name:  "comma where a key belongs",
			input: `{,}`,
			want:  `{}`,
		},
		{
			name:  "value with no separator",
			input: `[1 2]`,
			want:  `[1]`,
		},
		{
			// A raw newline is illegal inside a JSON string, and a closing
			// quote further on cannot make the bytes before it legal.
			name:  "raw control character inside a string",
			input: "{\"a\": \"line1\nline2\", \"b\": 2}",
			want:  `{"a": "line1"}`,
		},
		{
			name:  "unrecognized escape",
			input: `{"a": "x\q"}`,
			want:  `{"a": "x"}`,
		},
		{
			name:  "non-hex unicode escape",
			input: `{"a": "x\u12zz"}`,
			want:  `{"a": "x"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CompleteJSON(tt.input)
			if got != tt.want {
				t.Errorf("CompleteJSON() = %v, want %v", got, tt.want)
			}

			// Verify result is valid JSON
			var result any
			err := json.Unmarshal([]byte(got), &result)
			if err != nil {
				t.Errorf("CompleteJSON() produced invalid JSON: %v", err)
			}
		})
	}
}

// jsonPrefixCorpus holds documents whose every byte prefix models a point at
// which a stream could be cut.
var jsonPrefixCorpus = []string{
	`{"name": "John", "age": 30, "admin": true, "tags": null}`,
	`[{"id": 1, "name": "Jo\"hn"}, {"id": 2, "name": "Jane"}]`,
	`{"items": [{"n": [1.5, -2, 3e4]}, {"n": []}], "done": false}`,
	`[[[1], [2, [3]]], {"a": {"b": {"c": [{"d": "e"}]}}}]`,
	`{"text": "braces {} and brackets [] and a quote \" and é and 世界"}`,
	`[1, 2, 3]`,
	`{}`,
	`"just a string"`,
	`-12.5e-3`,
}

// TestCompleteJSONPrefixes checks the invariant the streaming path depends on:
// every prefix of a valid document completes to something that parses.
func TestCompleteJSONPrefixes(t *testing.T) {
	for _, doc := range jsonPrefixCorpus {
		t.Run(doc, func(t *testing.T) {
			for n := 0; n <= len(doc); n++ {
				prefix := doc[:n]
				completed := CompleteJSON(prefix)

				var result any
				if err := json.Unmarshal([]byte(completed), &result); err != nil {
					t.Errorf("CompleteJSON(%q) = %q, invalid JSON: %v", prefix, completed, err)
					continue
				}
				if strings.ContainsRune(completed, utf8.RuneError) && !strings.ContainsRune(doc, utf8.RuneError) {
					t.Errorf("CompleteJSON(%q) = %q, introduced a replacement character", prefix, completed)
				}
			}
		})
	}
}

// TestCompleteJSONPrefixesConverge checks that a completed prefix never
// contradicts the finished document, so a consumer that renders every chunk
// only ever sees values grow. Completion may not invent a member, reorder one,
// or report a string or number whose text the document later diverges from.
func TestCompleteJSONPrefixesConverge(t *testing.T) {
	for _, doc := range jsonPrefixCorpus {
		want, err := decodeJSON(doc)
		if err != nil {
			t.Fatalf("corpus document %q is not valid JSON: %v", doc, err)
		}

		t.Run(doc, func(t *testing.T) {
			for n := 1; n <= len(doc); n++ {
				prefix := doc[:n]
				completed := CompleteJSON(prefix)
				if completed == "{}" {
					// Nothing in the prefix could be completed, so CompleteJSON
					// fell back to reporting no value at all. That is not a
					// claim about the document, so there is nothing to check.
					continue
				}

				got, err := decodeJSON(completed)
				if err != nil {
					continue // Already reported by TestCompleteJSONPrefixes.
				}
				if !isJSONPrefixOf(got, want) {
					t.Errorf("CompleteJSON(%q) = %q, which is not a snapshot of %q", prefix, completed, doc)
				}
			}
		})
	}
}

// decodeJSON decodes into any, keeping numbers as their source text so a
// partially streamed number stays comparable to the finished one.
func decodeJSON(s string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()

	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	return v, nil
}

// isJSONPrefixOf reports whether got could be an earlier snapshot of want. A
// string or number that is still streaming shows up as a text prefix of its
// final value; containers may be missing trailing members.
func isJSONPrefixOf(got, want any) bool {
	switch g := got.(type) {
	case map[string]any:
		w, ok := want.(map[string]any)
		if !ok {
			return false
		}
		for k, gv := range g {
			wv, ok := w[k]
			if !ok || !isJSONPrefixOf(gv, wv) {
				return false
			}
		}
		return true
	case []any:
		w, ok := want.([]any)
		if !ok || len(g) > len(w) {
			return false
		}
		for i, gv := range g {
			if !isJSONPrefixOf(gv, w[i]) {
				return false
			}
		}
		return true
	case string:
		w, ok := want.(string)
		return ok && strings.HasPrefix(w, g)
	case json.Number:
		w, ok := want.(json.Number)
		return ok && strings.HasPrefix(w.String(), g.String())
	default:
		// Booleans and null arrive whole or not at all.
		return got == want
	}
}

// FuzzCompleteJSONArbitrary asserts the whole contract against unconstrained
// input: whatever it is handed, the result parses. Truncation is only half of
// what reaches this function, and a corpus of well-formed documents cannot
// reach the other half, so this target takes the input as given rather than
// deriving it from something already valid.
func FuzzCompleteJSONArbitrary(f *testing.F) {
	f.Add(`{"a": 1, {`)
	f.Add(`[1, 2,]`)
	f.Add(`{"a"}`)
	f.Add(`[1 2]`)
	f.Add("{\"a\": \"raw\ncontrol\"}")
	f.Add(`{"a": "x\q"}`)
	for _, doc := range jsonPrefixCorpus {
		f.Add(doc)
	}

	f.Fuzz(func(t *testing.T, s string) {
		completed := CompleteJSON(s)

		if !json.Valid([]byte(completed)) {
			t.Errorf("CompleteJSON(%q) = %q, which is not valid JSON", s, completed)
		}
	})
}

// FuzzCompleteJSON explores documents beyond the fixed corpus, cutting each one
// at an arbitrary point. Where FuzzCompleteJSONArbitrary only pins validity,
// this one holds the input to being a genuine prefix, which is what lets the
// corpus tests above check that the content is right and not merely parseable.
func FuzzCompleteJSON(f *testing.F) {
	for _, doc := range jsonPrefixCorpus {
		for _, n := range []uint{0, 1, 5, uint(len(doc))} {
			f.Add(doc, n)
		}
	}
	// Syntactically valid but too large for float64: the promise is that the
	// output parses, not that every document maps onto a Go value.
	f.Add("1E0007000", uint(128))

	f.Fuzz(func(t *testing.T, doc string, n uint) {
		if !json.Valid([]byte(doc)) {
			t.Skip()
		}

		prefix := doc[:int(n)%(len(doc)+1)]
		completed := CompleteJSON(prefix)

		if !json.Valid([]byte(completed)) {
			t.Errorf("CompleteJSON(%q) = %q, which is not valid JSON", prefix, completed)
		}
	})
}
