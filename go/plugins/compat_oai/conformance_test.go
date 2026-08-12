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

package compat_oai_test

import (
	"context"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/anthropic"
	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
)

// canonicalTypes is the JSON type each shared config field must have wherever
// a plugin declares it. Plugins declare their own fields rather than inheriting
// them, so this is what keeps one config JSON meaning the same thing across
// providers and runtimes. A field a provider does not accept is simply absent.
var canonicalTypes = map[string]string{
	"version":          "string",
	"temperature":      "number",
	"topP":             "number",
	"maxOutputTokens":  "integer",
	"stopSequences":    "array",
	"frequencyPenalty": "number",
	"presencePenalty":  "number",
	"logProbs":         "boolean",
	"topLogProbs":      "integer",
	"seed":             "integer",
	"reasoningEffort":  "string",
}

// TestConfigSchemaConformance pins the contract every plugin config in this
// package shares: canonical camelCase names with canonical types, a version
// key, and no credential or wire-name key. The openai plugin is deliberately
// absent: its config is the OpenAI SDK request type, so it speaks the SDK's
// own snake_case names by design.
func TestConfigSchemaConformance(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("DASHSCOPE_API_KEY", "test-key")
	t.Setenv("KIMI_API_KEY", "test-key")

	g := genkit.Init(context.Background(), genkit.WithPlugins(
		&anthropic.Anthropic{},
		&dashscope.DashScope{},
		&kimi.Kimi{},
	))

	models := []string{
		"anthropic/claude-sonnet-4-5-20250929",
		"dashscope/qwen-plus",
		"kimi/kimi-k3",
	}

	for _, name := range models {
		t.Run(name, func(t *testing.T) {
			m := genkit.LookupModel(g, name)
			if m == nil {
				t.Fatalf("%s not registered by Init", name)
			}
			model, ok := m.(api.Action).Desc().Metadata["model"].(map[string]any)
			if !ok {
				t.Fatalf("model metadata missing for %s", name)
			}
			schema, ok := model["customOptions"].(map[string]any)
			if !ok {
				t.Fatalf("%s advertises no config schema", name)
			}
			props, ok := schema["properties"].(map[string]any)
			if !ok {
				t.Fatalf("%s config schema has no properties: %v", name, schema)
			}

			if props["version"] == nil {
				t.Error("config schema is missing the version property")
			}
			if props["apiKey"] != nil {
				t.Error("config schema advertises apiKey, want the credential kept out of serialized configs")
			}

			for key, prop := range props {
				if strings.ContainsAny(key, "_-") {
					t.Errorf("config schema advertises %q, want the camelCase contract", key)
				}
				want, canonical := canonicalTypes[key]
				if !canonical {
					continue // A provider-specific field defines its own shape.
				}
				field, ok := prop.(map[string]any)
				if !ok {
					t.Errorf("%s property is %#v, want an object schema", key, prop)
					continue
				}
				if got := field["type"]; got != want {
					t.Errorf("%s type = %v, want %q to match every other plugin", key, got, want)
				}
			}

			requireDescriptions(t, "", props)
		})
	}
}

// requireDescriptions walks a schema's properties, recursively through nested
// objects, and fails for any that carries no description. The description is
// the help text the Dev UI shows for the field, so a bare property is a knob
// users see and get no explanation of.
func requireDescriptions(t *testing.T, path string, props map[string]any) {
	t.Helper()
	for key, prop := range props {
		field, ok := prop.(map[string]any)
		if !ok {
			continue // Reported as a malformed property by the caller's checks.
		}
		name := path + key
		if desc, _ := field["description"].(string); desc == "" {
			t.Errorf("%s has no description, want Dev UI help text on every config field", name)
		}
		if nested, ok := field["properties"].(map[string]any); ok {
			requireDescriptions(t, name+".", nested)
		}
	}
}

// TestModelsOverrideConformance pins that a caller's Models entry reaches a
// curated model, on every plugin. This is the answer to a catalog that
// describes a model wrongly: Init registers the curated models and nothing can
// re-register them, so the override has to be read before Init registers, not
// applied afterwards.
func TestModelsOverrideConformance(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	t.Setenv("DASHSCOPE_API_KEY", "test-key")
	t.Setenv("KIMI_API_KEY", "test-key")

	// Keyed provider-prefixed on half of them, bare on the rest: both forms
	// name the same model.
	pinned := ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Media: false}}

	g := genkit.Init(context.Background(), genkit.WithPlugins(
		&anthropic.Anthropic{Models: map[string]ai.ModelOptions{
			"anthropic/claude-sonnet-4-5-20250929": pinned}},
		&dashscope.DashScope{Models: map[string]ai.ModelOptions{"qwen-plus": pinned}},
		&kimi.Kimi{Models: map[string]ai.ModelOptions{"kimi-k3": pinned}},
	))

	for _, name := range []string{
		"anthropic/claude-sonnet-4-5-20250929",
		"dashscope/qwen-plus",
		"kimi/kimi-k3",
	} {
		t.Run(name, func(t *testing.T) {
			m := genkit.LookupModel(g, name)
			if m == nil {
				t.Fatalf("%s not registered by Init", name)
			}
			model, ok := m.(api.Action).Desc().Metadata["model"].(map[string]any)
			if !ok {
				t.Fatalf("model metadata missing for %s", name)
			}
			supports, ok := model["supports"].(map[string]any)
			if !ok {
				t.Fatalf("%s advertises no supports", name)
			}
			if supports["media"] != false {
				t.Errorf("media = %v, want the override's false", supports["media"])
			}
			// Overlaid, not replaced: the curated label survives an entry that
			// says nothing about it.
			if label, _ := model["label"].(string); label == "" {
				t.Error("label is empty, want the curated one kept by the overlay")
			}
		})
	}
}
