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

package ollama

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	pluginjsonschema "github.com/firebase/genkit/go/plugins/internal/jsonschema"
)

func TestOllamaChatRequest_MarshalJSON(t *testing.T) {
	req := &ollamaChatRequest{
		Model: "qwen3",
		Think: ThinkEnabled(true),
		Options: map[string]any{
			"temperature": 0.7,
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	jsonStr := string(data)
	if !strings.Contains(jsonStr, `"think":true`) {
		t.Errorf("expected json to contain \"think\":true, got: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"options":{"temperature":0.7}`) {
		t.Errorf("expected json to contain \"options\":{\"temperature\":0.7}, got: %s", jsonStr)
	}
}

func TestOllamaChatRequest_FormatField(t *testing.T) {
	t.Run("string json mode", func(t *testing.T) {
		req := &ollamaChatRequest{Model: "llama3", Format: "json"}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		got := string(data)
		if !strings.Contains(got, `"format":"json"`) {
			t.Errorf("expected \"format\":\"json\", got: %s", got)
		}
	})

	t.Run("schema object mode", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "string"},
			},
		}
		req := &ollamaChatRequest{Model: "llama3", Format: schema}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		got := string(data)
		// format must be a JSON object, not the string "json"
		if strings.Contains(got, `"format":"json"`) {
			t.Errorf("format should be a JSON object, not the string \"json\": %s", got)
		}
		if !strings.Contains(got, `"format":{"`) {
			t.Errorf("expected format to be a JSON object, got: %s", got)
		}
		if !strings.Contains(got, `"type":"object"`) {
			t.Errorf("expected schema type in format, got: %s", got)
		}
	})

	t.Run("nil omits field", func(t *testing.T) {
		req := &ollamaChatRequest{Model: "llama3"}
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatalf("marshal error: %v", err)
		}
		got := string(data)
		if strings.Contains(got, `"format"`) {
			t.Errorf("expected \"format\" key to be absent, got: %s", got)
		}
	})
}

func TestResolveSchemaRefs(t *testing.T) {
	t.Run("no defs — returned unchanged", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		if got["type"] != "object" {
			t.Errorf("expected type=object, got %v", got["type"])
		}
		if _, has := got["$defs"]; has {
			t.Error("expected no $defs in output")
		}
	})

	t.Run("$ref inlined and $defs removed", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Address": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"street": map[string]any{"type": "string"},
					},
				},
			},
			"type": "object",
			"properties": map[string]any{
				"addr": map[string]any{"$ref": "#/$defs/Address"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected $defs to be removed")
		}
		props, _ := got["properties"].(map[string]any)
		addr, _ := props["addr"].(map[string]any)
		if addr["type"] != "object" {
			t.Errorf("expected addr.type=object, got %v", addr["type"])
		}
		addrProps, _ := addr["properties"].(map[string]any)
		if addrProps["street"] == nil {
			t.Error("expected addr.properties.street to be present after inlining")
		}
	})

	t.Run("nested $ref resolved transitively", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Inner": map[string]any{"type": "string"},
				"Outer": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"value": map[string]any{"$ref": "#/$defs/Inner"},
					},
				},
			},
			"type": "object",
			"properties": map[string]any{
				"outer": map[string]any{"$ref": "#/$defs/Outer"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected $defs to be removed")
		}
		props, _ := got["properties"].(map[string]any)
		outer, _ := props["outer"].(map[string]any)
		outerProps, _ := outer["properties"].(map[string]any)
		value, _ := outerProps["value"].(map[string]any)
		if value["type"] != "string" {
			t.Errorf("expected nested value.type=string after transitive inlining, got %v", value)
		}
	})

	t.Run("anyOf with []any refs inlined", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Str": map[string]any{"type": "string"},
				"Num": map[string]any{"type": "number"},
			},
			"anyOf": []any{
				map[string]any{"$ref": "#/$defs/Str"},
				map[string]any{"$ref": "#/$defs/Num"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected $defs to be removed")
		}
		anyOf, _ := got["anyOf"].([]any)
		if len(anyOf) != 2 {
			t.Fatalf("expected 2 anyOf entries, got %d", len(anyOf))
		}
		first, _ := anyOf[0].(map[string]any)
		if first["type"] != "string" {
			t.Errorf("expected first anyOf to be inlined string type, got %v", first)
		}
	})

	t.Run("anyOf with []map[string]any refs inlined", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Foo": map[string]any{"type": "object"},
				"Bar": map[string]any{"type": "array"},
			},
			// Go-constructed schema uses []map[string]any (not []any)
			"anyOf": []map[string]any{
				{"$ref": "#/$defs/Foo"},
				{"$ref": "#/$defs/Bar"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		anyOf, _ := got["anyOf"].([]any)
		if len(anyOf) != 2 {
			t.Fatalf("expected 2 anyOf entries after []map[string]any walk, got %d", len(anyOf))
		}
		first, _ := anyOf[0].(map[string]any)
		if first["type"] != "object" {
			t.Errorf("expected first anyOf inlined as object type, got %v", first)
		}
	})

	t.Run("sibling keywords merged into resolved definition", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Addr": map[string]any{"type": "object"},
			},
			"properties": map[string]any{
				"addr": map[string]any{
					"$ref":        "#/$defs/Addr",
					"description": "shipping address",
				},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		addr, _ := props["addr"].(map[string]any)
		if addr["type"] != "object" {
			t.Errorf("expected addr.type=object after inlining, got %v", addr["type"])
		}
		if addr["description"] != "shipping address" {
			t.Errorf("expected sibling description to be preserved, got %v", addr["description"])
		}
	})

	t.Run("circular $ref terminates without panic", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"A": map[string]any{"$ref": "#/$defs/B"},
				"B": map[string]any{"$ref": "#/$defs/A"},
			},
			"properties": map[string]any{
				"root": map[string]any{"$ref": "#/$defs/A"},
			},
		}
		// Must not panic or infinitely recurse.
		got := pluginjsonschema.ResolveRefs(schema)
		if got == nil {
			t.Error("expected non-nil result for circular schema")
		}
		if _, has := got["$defs"]; !has {
			t.Error("expected $defs to be preserved when circular $refs remain")
		}
	})

	t.Run("boolean $defs entry leaves $ref in place", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Never": false, // boolean schema (draft 2019-09+)
			},
			"properties": map[string]any{
				"x": map[string]any{"$ref": "#/$defs/Never"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		x, _ := props["x"].(map[string]any)
		if x["$ref"] != "#/$defs/Never" {
			t.Errorf("expected $ref to boolean def to be left in place, got %v", x)
		}
		if _, has := got["$defs"]; !has {
			t.Error("expected $defs to be preserved when boolean-schema $ref remains")
		}
	})

	t.Run("$defs wins over definitions on name collision", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Tag": map[string]any{"type": "integer"}, // newer spec wins
			},
			"definitions": map[string]any{
				"Tag": map[string]any{"type": "string"}, // legacy — should lose
			},
			"properties": map[string]any{
				"tag": map[string]any{"$ref": "#/$defs/Tag"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		tag, _ := props["tag"].(map[string]any)
		if tag["type"] != "integer" {
			t.Errorf("expected $defs to win over definitions on collision, got type=%v", tag["type"])
		}
	})

	t.Run("definitions ref resolves from definitions on name collision", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Tag": map[string]any{"type": "integer"},
			},
			"definitions": map[string]any{
				"Tag": map[string]any{"type": "string"},
			},
			"properties": map[string]any{
				"tag": map[string]any{"$ref": "#/definitions/Tag"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		tag, _ := props["tag"].(map[string]any)
		if tag["type"] != "string" {
			t.Errorf("expected definitions ref to resolve as string, got type=%v", tag["type"])
		}
	})

	t.Run("unrelated local pointer is not resolved from defs", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"b": map[string]any{"type": "string"},
			},
			"properties": map[string]any{
				"a": map[string]any{"$ref": "#/properties/b"},
				"b": map[string]any{"type": "integer"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		a, _ := props["a"].(map[string]any)
		if a["$ref"] != "#/properties/b" {
			t.Errorf("expected unrelated local pointer to remain intact, got %v", a)
		}
		if _, ok := got["$defs"]; !ok {
			t.Error("expected $defs to remain while an unresolved local ref exists")
		}
	})

	t.Run("structural ref siblings are not merged", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Addr": map[string]any{
					"type":       "object",
					"properties": map[string]any{"s": map[string]any{"type": "string"}},
				},
			},
			"properties": map[string]any{
				"addr": map[string]any{"$ref": "#/$defs/Addr", "type": "string"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		addr, _ := props["addr"].(map[string]any)
		if addr["$ref"] != "#/$defs/Addr" || addr["type"] != "string" {
			t.Errorf("expected structural sibling ref to remain intact, got %v", addr)
		}
		if _, ok := got["$defs"]; !ok {
			t.Error("expected $defs to remain for an unflattened structural sibling ref")
		}
	})

	t.Run("unknown $ref left in place", func(t *testing.T) {
		schema := map[string]any{
			"properties": map[string]any{
				"x": map[string]any{"$ref": "#/$defs/Unknown"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		x, _ := props["x"].(map[string]any)
		if x["$ref"] != "#/$defs/Unknown" {
			t.Errorf("expected unknown $ref to be preserved, got %v", x)
		}
	})

	t.Run("unknown local $ref preserves existing defs", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Known": map[string]any{"type": "string"},
			},
			"properties": map[string]any{
				"x": map[string]any{"$ref": "#/$defs/Unknown"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		x, _ := props["x"].(map[string]any)
		if x["$ref"] != "#/$defs/Unknown" {
			t.Errorf("expected unknown $ref to be preserved, got %v", x)
		}
		if _, has := got["$defs"]; !has {
			t.Error("expected $defs to be preserved when unknown local $ref remains")
		}
	})

	t.Run("legacy definitions key", func(t *testing.T) {
		schema := map[string]any{
			"definitions": map[string]any{
				"Tag": map[string]any{"type": "string"},
			},
			"properties": map[string]any{
				"tag": map[string]any{"$ref": "#/definitions/Tag"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		if _, has := got["definitions"]; has {
			t.Error("expected definitions key to be removed")
		}
		props, _ := got["properties"].(map[string]any)
		tag, _ := props["tag"].(map[string]any)
		if tag["type"] != "string" {
			t.Errorf("expected tag to be inlined as string type, got %v", tag)
		}
	})

	t.Run("JSON Pointer escaped definition names", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Path/Name":  map[string]any{"type": "string"},
				"Tilde~Name": map[string]any{"type": "integer"},
			},
			"properties": map[string]any{
				"path":  map[string]any{"$ref": "#/$defs/Path~1Name"},
				"tilde": map[string]any{"$ref": "#/$defs/Tilde~0Name"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected $defs to be removed after escaped refs are inlined")
		}
		props, _ := got["properties"].(map[string]any)
		path, _ := props["path"].(map[string]any)
		if path["type"] != "string" {
			t.Errorf("expected escaped slash ref to inline string type, got %v", path)
		}
		tilde, _ := props["tilde"].(map[string]any)
		if tilde["type"] != "integer" {
			t.Errorf("expected escaped tilde ref to inline integer type, got %v", tilde)
		}
	})

	t.Run("unresolved top-level external ref does not mutate input schema", func(t *testing.T) {
		schema := map[string]any{
			"$ref": "https://example.com/schemas/External",
			"$defs": map[string]any{
				"Unused": map[string]any{"type": "string"},
			},
		}
		got := pluginjsonschema.ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected returned schema to omit unused $defs")
		}
		if _, has := schema["$defs"]; !has {
			t.Error("expected original schema to retain $defs")
		}
	})
}

func TestOllamaChatRequest_ApplyOptions(t *testing.T) {
	tests := []struct {
		name string
		cfg  GenerateContentConfig
		want *ollamaChatRequest
	}{
		{
			name: "configured values",
			cfg: GenerateContentConfig{
				Seed:        Ptr(42),
				Temperature: Ptr(0.7),
				Think:       ThinkEnabled(true),
				KeepAlive:   "10m",
			},
			want: &ollamaChatRequest{
				Think:     ThinkEnabled(true),
				KeepAlive: "10m",
				Options: map[string]any{
					"seed":        42,
					"temperature": 0.7,
				},
			},
		},
		{
			name: "explicit zero values",
			cfg: GenerateContentConfig{
				Seed:        Ptr(0),
				Temperature: Ptr(0.0),
			},
			want: &ollamaChatRequest{
				Options: map[string]any{
					"seed":        0,
					"temperature": 0.0,
				},
			},
		},
		{
			name: "thinking effort",
			cfg: GenerateContentConfig{
				Think: ThinkEffort("high"),
			},
			want: &ollamaChatRequest{
				Think: ThinkEffort("high"),
			},
		},
		{
			name: "zero config",
			cfg:  GenerateContentConfig{},
			want: &ollamaChatRequest{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &ollamaChatRequest{}
			req.ApplyOptions(tt.cfg)

			if !reflect.DeepEqual(req, tt.want) {
				t.Errorf(
					"unexpected result:\nwant: %#v\n got: %#v",
					tt.want,
					req,
				)
			}
		})
	}
}
