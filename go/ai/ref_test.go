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

package ai

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/invopop/jsonschema"
)

// TestArgContractsAreDisjoint pins the property the sealed marker methods
// exist for: a value of one kind must not satisfy another kind's argument
// contract. Without it, [WithTools] would accept a model, [WithModel] would
// accept an embedder ref, and the mistake would surface as a misleading
// "not found" error at generate time instead of a compile error.
//
// The positive direction is checked at compile time by the assertions next to
// each type. This checks the negative direction, which the compiler cannot
// express, so widening an interface or attaching a marker to the wrong type
// fails here.
func TestArgContractsAreDisjoint(t *testing.T) {
	kinds := []struct {
		name  string
		value any
		model bool
		embed bool
		eval  bool
		tool  bool
	}{
		{name: "*Model", value: (*Model)(nil), model: true},
		{name: "*BackgroundModel", value: (*BackgroundModel)(nil), model: true},
		{name: "ModelRef", value: ModelRef{}, model: true},
		{name: "*Embedder", value: (*Embedder)(nil), embed: true},
		{name: "EmbedderRef", value: EmbedderRef{}, embed: true},
		{name: "*Evaluator", value: (*Evaluator)(nil), eval: true},
		{name: "EvaluatorRef", value: EvaluatorRef{}, eval: true},
		{name: "*Tool", value: (*Tool[any, any])(nil), tool: true},
		{name: "*InterruptibleTool", value: (*InterruptibleTool[any, any, any])(nil), tool: true},
		{name: "ToolName", value: ToolName(""), tool: true},
	}

	for _, k := range kinds {
		t.Run(k.name, func(t *testing.T) {
			if _, ok := k.value.(Named); !ok {
				t.Errorf("%s does not satisfy Named; every action value and ref should", k.name)
			}
			if _, ok := k.value.(ModelArg); ok != k.model {
				t.Errorf("%s satisfies ModelArg = %v, want %v", k.name, ok, k.model)
			}
			if _, ok := k.value.(EmbedderArg); ok != k.embed {
				t.Errorf("%s satisfies EmbedderArg = %v, want %v", k.name, ok, k.embed)
			}
			if _, ok := k.value.(EvaluatorArg); ok != k.eval {
				t.Errorf("%s satisfies EvaluatorArg = %v, want %v", k.name, ok, k.eval)
			}
			if _, ok := k.value.(ToolArg); ok != k.tool {
				t.Errorf("%s satisfies ToolArg = %v, want %v", k.name, ok, k.tool)
			}
		})
	}
}

func TestRefJSON(t *testing.T) {
	t.Run("marshals as an object", func(t *testing.T) {
		got, err := json.Marshal(NewModelRef("googleai/gemini-flash-latest", map[string]any{"temperature": 0.7}))
		if err != nil {
			t.Fatal(err)
		}
		want := `{"name":"googleai/gemini-flash-latest","config":{"temperature":0.7}}`
		if diff := cmp.Diff(want, string(got)); diff != "" {
			t.Errorf("Marshal() diff (-want +got):\n%s", diff)
		}
	})

	t.Run("omits an absent config", func(t *testing.T) {
		got, err := json.Marshal(NewEmbedderRef("googleai/text-embedding-004", nil))
		if err != nil {
			t.Fatal(err)
		}
		want := `{"name":"googleai/text-embedding-004"}`
		if diff := cmp.Diff(want, string(got)); diff != "" {
			t.Errorf("Marshal() diff (-want +got):\n%s", diff)
		}
	})

	t.Run("unmarshals the object form", func(t *testing.T) {
		var ref ModelRef
		if err := json.Unmarshal([]byte(`{"name":"test/model","config":{"temperature":0.5}}`), &ref); err != nil {
			t.Fatal(err)
		}
		if ref.Name() != "test/model" {
			t.Errorf("Name() = %q, want %q", ref.Name(), "test/model")
		}
		if diff := cmp.Diff(map[string]any{"temperature": 0.5}, ref.Config()); diff != "" {
			t.Errorf("Config() diff (-want +got):\n%s", diff)
		}
	})

	t.Run("unmarshals the string shorthand", func(t *testing.T) {
		var ref EvaluatorRef
		if err := json.Unmarshal([]byte(`"genkitEval/faithfulness"`), &ref); err != nil {
			t.Fatal(err)
		}
		if ref.Name() != "genkitEval/faithfulness" {
			t.Errorf("Name() = %q, want %q", ref.Name(), "genkitEval/faithfulness")
		}
		if ref.Config() != nil {
			t.Errorf("Config() = %v, want nil", ref.Config())
		}
	})
}

// TestRefJSONSchema checks that each ref reflects to its object schema rather
// than the empty object its unexported fields would otherwise produce. The
// schema is embedded in middleware config (e.g. the fallback middleware's
// model list), so the Dev UI depends on it.
func TestRefJSONSchema(t *testing.T) {
	refs := []struct {
		name string
		got  *jsonschema.Schema
	}{
		{"ModelRef", ModelRef{}.JSONSchema()},
		{"EmbedderRef", EmbedderRef{}.JSONSchema()},
		{"EvaluatorRef", EvaluatorRef{}.JSONSchema()},
	}
	for _, r := range refs {
		t.Run(r.name, func(t *testing.T) {
			if r.got.Type != "object" {
				t.Errorf("Type = %q, want %q", r.got.Type, "object")
			}
			if diff := cmp.Diff([]string{"name"}, r.got.Required); diff != "" {
				t.Errorf("Required diff (-want +got):\n%s", diff)
			}
			if _, ok := r.got.Properties.Get("name"); !ok {
				t.Error("schema has no \"name\" property")
			}
			if _, ok := r.got.Properties.Get("config"); !ok {
				t.Error("schema has no \"config\" property")
			}
		})
	}

	// Reflection through a struct field must pick up the custom schema, which
	// only works if the outer type promotes or defines JSONSchema.
	type holder struct {
		Models []ModelRef `json:"models,omitempty"`
	}
	reflected := (&jsonschema.Reflector{DoNotReference: true}).Reflect(&holder{})
	models, ok := reflected.Properties.Get("models")
	if !ok {
		t.Fatal("reflected schema has no \"models\" property")
	}
	if models.Items == nil || models.Items.Type != "object" {
		t.Errorf("models item schema = %+v, want an object schema", models.Items)
	}
}
