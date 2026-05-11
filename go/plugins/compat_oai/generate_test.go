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

package compat_oai

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go"
)

// newGen returns a ModelGenerator with a nil client; only local tool-shaping
// logic is exercised, so no network call is made.
func newGen() *ModelGenerator {
	return NewModelGenerator((*openai.Client)(nil), "test-model")
}

// TestWithTools_StrictDefaultsOff verifies the default behavior of sending
// strict: false to OpenAI when the tool has no strict metadata.
func TestWithTools_StrictDefaultsOff(t *testing.T) {
	g := newGen().WithTools([]*ai.ToolDefinition{
		{
			Name:        "ping",
			Description: "ping",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"msg": map[string]any{"type": "string"}},
			},
		},
	})
	if len(g.tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(g.tools))
	}
	fn := g.tools[0].Function
	if !fn.Strict.Valid() || fn.Strict.Value {
		t.Errorf("expected Strict=false by default, got valid=%v value=%v", fn.Strict.Valid(), fn.Strict.Value)
	}
	if _, present := fn.Parameters["additionalProperties"]; present {
		t.Errorf("expected no additionalProperties when strict is off, got %v", fn.Parameters["additionalProperties"])
	}
}

// TestWithTools_StrictOptIn verifies a tool with Metadata["strict"]=true is
// sent with Strict=true and additionalProperties: false applied recursively.
func TestWithTools_StrictOptIn(t *testing.T) {
	g := newGen().WithTools([]*ai.ToolDefinition{
		{
			Name:        "weather",
			Description: "get weather",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city": map[string]any{"type": "string"},
						},
					},
				},
			},
			Metadata: map[string]any{"strict": true},
		},
	})
	if len(g.tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(g.tools))
	}
	fn := g.tools[0].Function
	if !fn.Strict.Valid() || !fn.Strict.Value {
		t.Errorf("expected Strict=true, got valid=%v value=%v", fn.Strict.Valid(), fn.Strict.Value)
	}
	if fn.Parameters["additionalProperties"] != false {
		t.Errorf("expected top-level additionalProperties: false, got %v", fn.Parameters["additionalProperties"])
	}
	props, _ := fn.Parameters["properties"].(map[string]any)
	loc, _ := props["location"].(map[string]any)
	if loc["additionalProperties"] != false {
		t.Errorf("expected nested additionalProperties: false on location, got %v", loc["additionalProperties"])
	}
}

// TestWithTools_StrictExplicitFalse verifies an explicit opt-out matches the
// default: Strict=false and the schema is not enforced.
func TestWithTools_StrictExplicitFalse(t *testing.T) {
	g := newGen().WithTools([]*ai.ToolDefinition{
		{
			Name:        "ping",
			Description: "ping",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"msg": map[string]any{"type": "string"}},
			},
			Metadata: map[string]any{"strict": false},
		},
	})
	fn := g.tools[0].Function
	if !fn.Strict.Valid() || fn.Strict.Value {
		t.Errorf("expected Strict=false, got valid=%v value=%v", fn.Strict.Valid(), fn.Strict.Value)
	}
	if _, present := fn.Parameters["additionalProperties"]; present {
		t.Errorf("expected no additionalProperties when strict is off, got %v", fn.Parameters["additionalProperties"])
	}
}

// TestWithTools_StrictDoesNotMutateCallerSchema guards against the caller's
// input schema being mutated in place by strict-mode enforcement.
func TestWithTools_StrictDoesNotMutateCallerSchema(t *testing.T) {
	original := map[string]any{
		"type":       "object",
		"properties": map[string]any{"msg": map[string]any{"type": "string"}},
	}
	def := &ai.ToolDefinition{
		Name:        "ping",
		Description: "ping",
		InputSchema: original,
		Metadata:    map[string]any{"strict": true},
	}
	newGen().WithTools([]*ai.ToolDefinition{def})
	if _, present := original["additionalProperties"]; present {
		t.Errorf("caller schema was mutated: additionalProperties leaked into original input schema")
	}
}
