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

package schemautil

import "testing"

func TestResolveRefUsesReferencedPool(t *testing.T) {
	schema := map[string]any{
		"$defs": map[string]any{
			"Tag": map[string]any{"type": "integer"},
		},
		"definitions": map[string]any{
			"Tag": map[string]any{"type": "string"},
		},
	}

	got, err := ResolveRef(schema, "#/definitions/Tag")
	if err != nil {
		t.Fatalf("ResolveRef() error = %v", err)
	}
	if got["type"] != "string" {
		t.Errorf("ResolveRef() type = %v, want string", got["type"])
	}
}

func TestResolveRefRejectsUnrelatedPointer(t *testing.T) {
	schema := map[string]any{
		"$defs": map[string]any{
			"Tag": map[string]any{"type": "string"},
		},
	}
	if _, err := ResolveRef(schema, "#/properties/Tag"); err == nil {
		t.Fatal("ResolveRef() succeeded for a pointer outside definition maps")
	}
}

func TestResolveRefs(t *testing.T) {
	t.Run("no defs — returned unchanged", func(t *testing.T) {
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
		props, _ := got["properties"].(map[string]any)
		addr, _ := props["addr"].(map[string]any)
		if addr["type"] != "object" {
			t.Errorf("expected addr.type=object after inlining, got %v", addr["type"])
		}
		if addr["description"] != "shipping address" {
			t.Errorf("expected sibling description to be preserved, got %v", addr["description"])
		}
	})

	t.Run("$comment sibling does not block inlining", func(t *testing.T) {
		schema := map[string]any{
			"$defs": map[string]any{
				"Addr": map[string]any{"type": "object"},
			},
			"properties": map[string]any{
				"addr": map[string]any{
					"$ref":     "#/$defs/Addr",
					"$comment": "internal note",
				},
			},
		}
		got := ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected $defs to be removed once the only ref is inlined")
		}
		props, _ := got["properties"].(map[string]any)
		addr, _ := props["addr"].(map[string]any)
		if addr["type"] != "object" {
			t.Errorf("expected addr.type=object after inlining despite $comment sibling, got %v", addr)
		}
		if addr["$comment"] != "internal note" {
			t.Errorf("expected sibling $comment to be preserved, got %v", addr["$comment"])
		}
	})

	t.Run("shared def referenced from multiple places inlines identically", func(t *testing.T) {
		// Diamond-shaped: both "left" and "right" point at the same "Leaf" def.
		// Regression coverage for the memoized inlining path: a def visited more
		// than once must produce the same result at every occurrence.
		schema := map[string]any{
			"$defs": map[string]any{
				"Leaf": map[string]any{"type": "string"},
			},
			"properties": map[string]any{
				"left":  map[string]any{"$ref": "#/$defs/Leaf"},
				"right": map[string]any{"$ref": "#/$defs/Leaf"},
			},
		}
		got := ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected $defs to be removed")
		}
		props, _ := got["properties"].(map[string]any)
		left, _ := props["left"].(map[string]any)
		right, _ := props["right"].(map[string]any)
		if left["type"] != "string" || right["type"] != "string" {
			t.Errorf("expected both occurrences inlined to type=string, got left=%v right=%v", left, right)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
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
		got := ResolveRefs(schema)
		if _, has := got["$defs"]; has {
			t.Error("expected returned schema to omit unused $defs")
		}
		if _, has := schema["$defs"]; !has {
			t.Error("expected original schema to retain $defs")
		}
	})
}
