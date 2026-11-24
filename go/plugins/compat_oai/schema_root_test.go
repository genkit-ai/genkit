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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/hooks"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

var personArray = map[string]any{
	"type": "array",
	"items": map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"name": map[string]any{"type": "string"}},
		"required":             []any{"name"},
		"additionalProperties": false,
	},
}

func TestHasObjectRoot(t *testing.T) {
	for _, tc := range []struct {
		schema map[string]any
		want   bool
	}{
		{map[string]any{"type": "object", "properties": map[string]any{}}, true},
		{map[string]any{"properties": map[string]any{"a": map[string]any{"type": "string"}}}, false},
		{personArray, false},
		{map[string]any{"type": "string", "enum": []any{"a", "b"}}, false},
		{map[string]any{"type": "integer"}, false},
		{map[string]any{"anyOf": []any{map[string]any{"type": "object"}, map[string]any{"type": "string"}}}, false},
	} {
		if got := hasObjectRoot(tc.schema); got != tc.want {
			t.Errorf("hasObjectRoot(%v) = %v, want %v", tc.schema, got, tc.want)
		}
	}
}

func TestWrapSchemaRoot(t *testing.T) {
	got := wrapSchemaRoot(personArray)
	want := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"value": personArray},
		"required":             []any{"value"},
		"additionalProperties": false,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("wrapSchemaRoot() = %v, want %v", got, want)
	}
}

func TestWrapSchemaRootKeepsDefinitionsAtRoot(t *testing.T) {
	schema := map[string]any{
		"$schema":     "https://json-schema.org/draft/2020-12/schema",
		"type":        "array",
		"items":       map[string]any{"$ref": "#/$defs/Item"},
		"$defs":       map[string]any{"Item": map[string]any{"type": "string"}},
		"definitions": map[string]any{"Old": map[string]any{"type": "number"}},
	}
	got := wrapSchemaRoot(schema)
	for _, k := range []string{"$schema", "$defs", "definitions"} {
		if !reflect.DeepEqual(got[k], schema[k]) {
			t.Errorf("wrapped[%q] = %v, want %v at the root", k, got[k], schema[k])
		}
	}
	inner := got["properties"].(map[string]any)["value"]
	wantInner := map[string]any{"type": "array", "items": map[string]any{"$ref": "#/$defs/Item"}}
	if !reflect.DeepEqual(inner, wantInner) {
		t.Errorf("wrapped value schema = %v, want %v", inner, wantInner)
	}
	if _, ok := schema["properties"]; ok {
		t.Error("wrapSchemaRoot mutated its input")
	}
}

func TestWrapSchemaRootHoistsLocalRefs(t *testing.T) {
	first := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"n": map[string]any{"type": "number"}},
		"required":             []any{"n"},
		"additionalProperties": false,
	}
	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"first":  first,
				"second": map[string]any{"$ref": "#/items/properties/first"},
				"all":    map[string]any{"$ref": "#"},
				"a":      map[string]any{"$ref": "#/$defs/value"},
				"remote": map[string]any{"$ref": "https://example.com/schema"},
				"gone":   map[string]any{"$ref": "#/nowhere"},
			},
			"default": map[string]any{"$ref": "#/items"},
		},
		// A definition already named "value" pushes the hoisted root to
		// "value_2".
		"$defs": map[string]any{"value": map[string]any{"type": "string"}},
	}
	got := wrapSchemaRoot(schema)

	want := map[string]any{
		"type":                 "object",
		"required":             []any{"value"},
		"additionalProperties": false,
		"properties":           map[string]any{"value": map[string]any{"$ref": "#/$defs/value_2"}},
		"$defs": map[string]any{
			"value": map[string]any{"type": "string"},
			"value_2": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"first":  map[string]any{"$ref": "#/$defs/value_items_properties_first"},
						"second": map[string]any{"$ref": "#/$defs/value_items_properties_first"},
						"all":    map[string]any{"$ref": "#/$defs/value_2"},
						"a":      map[string]any{"$ref": "#/$defs/value"},
						"remote": map[string]any{"$ref": "https://example.com/schema"},
						"gone":   map[string]any{"$ref": "#/nowhere"},
					},
					"default": map[string]any{"$ref": "#/items"},
				},
			},
			"value_items_properties_first": first,
		},
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		t.Errorf("wrapSchemaRoot() =\n%s", gotJSON)
	}
	if ref := schema["items"].(map[string]any)["properties"].(map[string]any)["all"].(map[string]any)["$ref"]; ref != "#" {
		t.Errorf("wrapSchemaRoot mutated its input: $ref = %v", ref)
	}
}

func TestWrappedSchemaResolvesLocalRefs(t *testing.T) {
	// A recursive tree, as an array root: "#" refers back to the array and
	// "#/items" to its item, both of which move under the wrapper.
	schema := map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
				"kids": map[string]any{"type": "array", "items": map[string]any{"$ref": "#/items"}},
				"all":  map[string]any{"$ref": "#"},
			},
			"required": []any{"name"},
		},
	}
	wrapped := wrapSchemaRoot(schema)
	valid := map[string]any{"value": []any{map[string]any{
		"name": "a",
		"kids": []any{map[string]any{"name": "b"}},
		"all":  []any{map[string]any{"name": "c"}},
	}}}
	if err := base.ValidateValue(valid, wrapped); err != nil {
		t.Errorf("valid tree failed validation: %v", err)
	}
	invalid := map[string]any{"value": []any{map[string]any{
		"name": "a",
		"kids": []any{map[string]any{"name": 1}},
	}}}
	if err := base.ValidateValue(invalid, wrapped); err == nil {
		t.Error("a kid with a numeric name passed validation, want the rewritten $ref enforced")
	}
}

// personDefs is a $defs map whose Person has a name that other schemas refer
// into, plus a definition already holding the name that ref would hoist to.
func personDefs() map[string]any {
	return map[string]any{
		"Person": map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"name": map[string]any{"type": "string", "minLength": 1}},
			"required":             []any{"name"},
			"additionalProperties": false,
		},
		"Person_properties_name": map[string]any{"type": "number"},
	}
}

func TestHoistDefRefs(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"person":   map[string]any{"$ref": "#/$defs/Person"},
			"nickname": map[string]any{"$ref": "#/$defs/Person/properties/name"},
			"legacy":   map[string]any{"$ref": "#/definitions/Old/properties/id"},
			"self":     map[string]any{"$ref": "#/properties/person"},
		},
		"required":             []any{"person", "nickname", "legacy", "self"},
		"additionalProperties": false,
		"$defs":                personDefs(),
		"definitions": map[string]any{
			"Old": map[string]any{
				"type":       "object",
				"properties": map[string]any{"id": map[string]any{"type": "integer"}},
			},
		},
	}
	got := hoistDefRefs(schema)

	want := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"person":   map[string]any{"$ref": "#/$defs/Person"},
			"nickname": map[string]any{"$ref": "#/$defs/Person_properties_name_2"},
			"legacy":   map[string]any{"$ref": "#/$defs/Old_properties_id"},
			// Other local refs are left alone on an object root.
			"self": map[string]any{"$ref": "#/properties/person"},
		},
		"required":             []any{"person", "nickname", "legacy", "self"},
		"additionalProperties": false,
		"$defs": map[string]any{
			"Person": map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"name": map[string]any{"$ref": "#/$defs/Person_properties_name_2"}},
				"required":             []any{"name"},
				"additionalProperties": false,
			},
			"Person_properties_name":   map[string]any{"type": "number"},
			"Person_properties_name_2": map[string]any{"type": "string", "minLength": 1},
			"Old_properties_id":        map[string]any{"type": "integer"},
		},
		"definitions": map[string]any{
			"Old": map[string]any{
				"type":       "object",
				"properties": map[string]any{"id": map[string]any{"$ref": "#/$defs/Old_properties_id"}},
			},
		},
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		t.Errorf("hoistDefRefs() =\n%s", gotJSON)
	}
	if !reflect.DeepEqual(schema["$defs"], personDefs()) {
		t.Error("hoistDefRefs mutated its input")
	}

	valid := map[string]any{"person": map[string]any{"name": "Ann"}, "nickname": "Annie", "legacy": 1, "self": map[string]any{"name": "B"}}
	if err := base.ValidateValue(valid, got); err != nil {
		t.Errorf("valid value failed validation: %v", err)
	}
	for _, bad := range []map[string]any{
		{"person": map[string]any{"name": "Ann"}, "nickname": "", "legacy": 1, "self": map[string]any{"name": "B"}},
		{"person": map[string]any{"name": ""}, "nickname": "Annie", "legacy": 1, "self": map[string]any{"name": "B"}},
		{"person": map[string]any{"name": "Ann"}, "nickname": "Annie", "legacy": "x", "self": map[string]any{"name": "B"}},
	} {
		if err := base.ValidateValue(bad, got); err == nil {
			t.Errorf("%v passed validation, want the hoisted constraints enforced", bad)
		}
	}
}

func TestHoistDefRefsLeavesSchemaWithoutThemAlone(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"p": map[string]any{"$ref": "#/$defs/Person"}},
		"$defs":      personDefs(),
	}
	if got := hoistDefRefs(schema); reflect.ValueOf(got).Pointer() != reflect.ValueOf(schema).Pointer() {
		t.Errorf("hoistDefRefs() returned a copy, want the schema itself")
	}
}

func TestWrapSchemaRootHoistsRefsIntoDefinitions(t *testing.T) {
	schema := map[string]any{
		"type":  "array",
		"items": map[string]any{"$ref": "#/$defs/Person/properties/name"},
		"$defs": personDefs(),
	}
	got := wrapSchemaRoot(schema)

	want := map[string]any{
		"type":                 "object",
		"required":             []any{"value"},
		"additionalProperties": false,
		"properties": map[string]any{"value": map[string]any{
			"type":  "array",
			"items": map[string]any{"$ref": "#/$defs/Person_properties_name_2"},
		}},
		"$defs": map[string]any{
			"Person": map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"name": map[string]any{"$ref": "#/$defs/Person_properties_name_2"}},
				"required":             []any{"name"},
				"additionalProperties": false,
			},
			"Person_properties_name":   map[string]any{"type": "number"},
			"Person_properties_name_2": map[string]any{"type": "string", "minLength": 1},
		},
	}
	if !reflect.DeepEqual(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		t.Errorf("wrapSchemaRoot() =\n%s", gotJSON)
	}
	if err := base.ValidateValue(map[string]any{"value": []any{"Ann"}}, got); err != nil {
		t.Errorf("valid value failed validation: %v", err)
	}
	if err := base.ValidateValue(map[string]any{"value": []any{""}}, got); err == nil {
		t.Error("an empty name passed validation, want minLength enforced")
	}
}

func TestObjectSchemaRootHook(t *testing.T) {
	var o OpenAICompatible
	if o.objectSchemaRoot {
		t.Fatal("objectSchemaRoot is on by default, want it off")
	}
	hooks.SetObjectSchemaRoot(&o)
	if !o.objectSchemaRoot {
		t.Error("hooks.SetObjectSchemaRoot did not turn objectSchemaRoot on")
	}
}

func TestUnwrapText(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`{"value":[{"name":"a"}]}`, `[{"name":"a"}]`},
		{`{"value": {"b":1,"a":2}}`, `{"b":1,"a":2}`},
		{`{"value":0}`, `0`},
		{`{"value":false}`, `false`},
		{`{"value":""}`, `""`},
		{`[1,2]`, `[1,2]`},
		{`{"other":1}`, `{"other":1}`},
		{`not json`, `not json`},
	} {
		if got := unwrapText(tc.in); got != tc.want {
			t.Errorf("unwrapText(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func runUnwrapper(deltas []string) string {
	u := &streamUnwrapper{}
	var b strings.Builder
	for _, d := range deltas {
		b.WriteString(u.push(d))
	}
	b.WriteString(u.flush())
	return b.String()
}

func TestStreamUnwrapper(t *testing.T) {
	full := `{"value":[{"name":"a"},{"name":"b"}]}`
	want := `[{"name":"a"},{"name":"b"}]`
	for _, tc := range []struct {
		name   string
		deltas []string
		want   string
	}{
		{"whole", []string{full}, want},
		{"one character at a time", strings.Split(full, ""), want},
		{"token-sized", []string{`{"`, `value`, `":[`, `{"name":"a"}`, `,{"name":"b"}`, `]}`}, want},
		{"pretty-printed", []string{"{\n  \"value\": ", "[1, 2]\n", "}\n"}, "[1, 2]"},
		{"nested braces in the value", []string{`{"value":{"a":{"b":1}}}`}, `{"a":{"b":1}}`},
		{"object closing before the array", []string{`{"value":[{"a":1}`, `]}`}, `[{"a":1}]`},
		{"scalar", []string{`{"value":"he`, `llo"}`}, `"hello"`},
		{"spaced opening whole", []string{"{" + strings.Repeat(" ", 42) + `"value":1}`}, "1"},
		{"spaced opening one character at a time", strings.Split("{"+strings.Repeat(" ", 42)+`"value":1}`, ""), "1"},
		{"not a wrapper", []string{`[1,`, `2]`}, `[1,2]`},
		{"key that only starts like the wrapper's", strings.Split(`{"valuable":1}`, ""), `{"valuable":1}`},
		{"other object one character at a time", strings.Split(`{"other":1}`, ""), `{"other":1}`},
		{"long object that is not a wrapper", []string{`{"other":1,"more":"text that is long enough to decide"}`}, `{"other":1,"more":"text that is long enough to decide"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := runUnwrapper(tc.deltas); got != tc.want {
				t.Errorf("unwrapped stream = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestStreamUnwrapperNeverEmitsTheClosingBraceEarly(t *testing.T) {
	u := &streamUnwrapper{}
	emitted := u.push(`{"value":[1,2`) + u.push(`]}`)
	if emitted != "[1,2]" {
		t.Errorf("emitted before flush = %q, want %q", emitted, "[1,2]")
	}
	if rest := u.flush(); rest != "" {
		t.Errorf("flush() = %q, want nothing left", rest)
	}
}

func TestApplyResponseFormatWrapsNonObjectRoot(t *testing.T) {
	jsonOut := func(schema map[string]any) *ai.ModelOutputConfig {
		return &ai.ModelOutputConfig{Format: ai.OutputFormatJSON, Schema: schema, Constrained: true}
	}
	sentSchema := func(g *ModelGenerator) any {
		if g.request.ResponseFormat.OfJSONSchema == nil {
			return nil
		}
		return g.request.ResponseFormat.OfJSONSchema.JSONSchema.Schema
	}

	g := NewModelGenerator(nil, "m").withObjectSchemaRoot(true)
	g.applyResponseFormat(jsonOut(personArray))
	if got := sentSchema(g); !reflect.DeepEqual(got, wrapSchemaRoot(personArray)) {
		t.Errorf("object-root host sent %v, want the wrapped schema", got)
	}
	if !g.unwrapRoot {
		t.Error("unwrapRoot = false, want the reply unwrapped")
	}

	object := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}
	g = NewModelGenerator(nil, "m").withObjectSchemaRoot(true)
	g.applyResponseFormat(jsonOut(object))
	if got := sentSchema(g); !reflect.DeepEqual(got, object) || g.unwrapRoot {
		t.Errorf("object root sent %v (unwrap %v), want it unchanged and not unwrapped", got, g.unwrapRoot)
	}

	g = NewModelGenerator(nil, "m")
	g.applyResponseFormat(jsonOut(personArray))
	if got := sentSchema(g); !reflect.DeepEqual(got, personArray) || g.unwrapRoot {
		t.Errorf("host without ObjectSchemaRoot sent %v (unwrap %v), want the schema unchanged", got, g.unwrapRoot)
	}

	// An object root that refers into the middle of a definition has that
	// part hoisted to a definition of its own, but only on object-root hosts.
	intoDef := map[string]any{
		"type":       "object",
		"properties": map[string]any{"nick": map[string]any{"$ref": "#/$defs/Person/properties/name"}},
		"$defs":      personDefs(),
	}
	g = NewModelGenerator(nil, "m").withObjectSchemaRoot(true)
	g.applyResponseFormat(jsonOut(intoDef))
	if got := sentSchema(g); !reflect.DeepEqual(got, hoistDefRefs(intoDef)) || g.unwrapRoot {
		t.Errorf("object root with a ref into a definition sent %v (unwrap %v), want it hoisted and not unwrapped", got, g.unwrapRoot)
	}
	g = NewModelGenerator(nil, "m")
	g.applyResponseFormat(jsonOut(intoDef))
	if got := sentSchema(g); !reflect.DeepEqual(got, intoDef) {
		t.Errorf("host without ObjectSchemaRoot sent %v, want the schema unchanged", got)
	}
}

// newWrappedRootServer serves one chat completion reply, as JSON or as an
// event stream, and records the response_format schema the request sent.
func newWrappedRootServer(t *testing.T, events []string, completion string) (*openai.Client, *any) {
	t.Helper()
	var sent any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream         bool `json:"stream"`
			ResponseFormat struct {
				JSONSchema struct {
					Schema any `json:"schema"`
				} `json:"json_schema"`
			} `json:"response_format"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		sent = body.ResponseFormat.JSONSchema.Schema
		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, e := range events {
				_, _ = io.WriteString(w, "data: "+e+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completion)
	}))
	t.Cleanup(srv.Close)
	client := openai.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("stub"))
	return &client, &sent
}

func deltaEvent(content string) string {
	b, _ := json.Marshal(content)
	return `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":` + string(b) + `},"finish_reason":null}]}`
}

func completionBody(content string) string {
	b, _ := json.Marshal(content)
	return `{"id":"1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":` + string(b) + `},"finish_reason":"stop"}]}`
}

func wrappedRootRequest() *ai.ModelRequest {
	return &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("list people")},
		Output:   &ai.ModelOutputConfig{Format: ai.OutputFormatJSON, Schema: personArray, Constrained: true},
	}
}

func TestGenerateUnwrapsWrappedRoot(t *testing.T) {
	client, sent := newWrappedRootServer(t, nil, completionBody(`{"value":[{"name":"a"}]}`))
	req := wrappedRootRequest()
	resp, err := NewModelGenerator(client, "m").withObjectSchemaRoot(true).
		WithMessages(req.Messages).
		Generate(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if want := normalize(t, wrapSchemaRoot(personArray)); !reflect.DeepEqual(*sent, want) {
		t.Errorf("sent schema = %v, want %v", *sent, want)
	}
	if got := resp.Message.Text(); got != `[{"name":"a"}]` {
		t.Errorf("response text = %q, want the unwrapped array", got)
	}
}

func TestGenerateStreamUnwrapsWrappedRoot(t *testing.T) {
	pieces := []string{`{"`, `value`, `":[`, `{"name":"a"}`, `,{"name":"b"}`, `]}`}
	var events []string
	for _, p := range pieces {
		events = append(events, deltaEvent(p))
	}
	events = append(events, `{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
	client, _ := newWrappedRootServer(t, events, "")

	req := wrappedRootRequest()
	var streamed strings.Builder
	chunks := 0
	resp, err := NewModelGenerator(client, "m").withObjectSchemaRoot(true).
		WithMessages(req.Messages).
		Generate(context.Background(), req, func(_ context.Context, c *ai.ModelResponseChunk) error {
			chunks++
			for _, p := range c.Content {
				if p.IsText() {
					streamed.WriteString(p.Text)
				}
			}
			return nil
		})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	want := `[{"name":"a"},{"name":"b"}]`
	if got := streamed.String(); got != want {
		t.Errorf("streamed text = %q, want %q", got, want)
	}
	if chunks < 2 {
		t.Errorf("chunks = %d, want the value streamed in several", chunks)
	}
	if got := resp.Message.Text(); got != want {
		t.Errorf("final text = %q, want %q", got, want)
	}
}

func TestGenerateLeavesToolCallPreambleAlone(t *testing.T) {
	body := `{"id":"1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"{\"value\":1}","tool_calls":[{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`
	client, _ := newWrappedRootServer(t, nil, body)
	req := wrappedRootRequest()
	resp, err := NewModelGenerator(client, "m").withObjectSchemaRoot(true).
		WithMessages(req.Messages).
		Generate(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := resp.Message.Text(); got != `{"value":1}` {
		t.Errorf("text beside a tool call = %q, want it left as sent", got)
	}
}

// normalize round-trips v through JSON so it compares equal to a decoded
// request body.
func normalize(t *testing.T, v any) any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestStreamUnwrapperPassThrough(t *testing.T) {
	u := &streamUnwrapper{}
	emitted := u.push("Let me check. ")
	if got := emitted + u.passThrough(); got != "Let me check. " {
		t.Errorf("preamble = %q, want it exactly as streamed", got)
	}

	u = &streamUnwrapper{}
	if got := u.push(`{"val`) + u.passThrough(); got != `{"val` {
		t.Errorf("undecided prefix = %q, want the buffered text back", got)
	}
}

func TestGenerateStreamLeavesToolCallPreambleAlone(t *testing.T) {
	events := []string{
		deltaEvent("Let me "),
		deltaEvent("check. "),
		`{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}`,
		`{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	}
	client, _ := newWrappedRootServer(t, events, "")
	req := wrappedRootRequest()
	var streamed strings.Builder
	sawToolRequest := false
	resp, err := NewModelGenerator(client, "m").withObjectSchemaRoot(true).
		WithMessages(req.Messages).
		Generate(context.Background(), req, func(_ context.Context, c *ai.ModelResponseChunk) error {
			for _, p := range c.Content {
				switch {
				case p.IsText():
					streamed.WriteString(p.Text)
				case p.IsToolRequest():
					sawToolRequest = true
				}
			}
			return nil
		})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got := streamed.String(); got != "Let me check. " {
		t.Errorf("streamed preamble = %q, want it exactly as sent", got)
	}
	if !sawToolRequest {
		t.Error("no tool request was streamed")
	}
	if got := resp.Message.Text(); got != "Let me check. " {
		t.Errorf("final preamble = %q, want it exactly as sent", got)
	}
	if !slices.ContainsFunc(resp.Message.Content, (*ai.Part).IsToolRequest) {
		t.Error("final message has no tool request")
	}
}
