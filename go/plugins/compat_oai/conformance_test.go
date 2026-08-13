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
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/anthropic"
	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
	"github.com/firebase/genkit/go/plugins/compat_oai/zai"
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
	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	t.Setenv("KIMI_API_KEY", "test-key")
	t.Setenv("XAI_API_KEY", "test-key")
	t.Setenv("ZAI_API_KEY", "test-key")

	g := genkit.Init(context.Background(), genkit.WithPlugins(
		&anthropic.Anthropic{},
		&dashscope.DashScope{},
		&deepseek.DeepSeek{},
		&kimi.Kimi{},
		&xai.XAI{},
		&zai.ZAI{},
	))

	models := []string{
		"anthropic/claude-sonnet-4-5-20250929",
		"dashscope/qwen-plus",
		"deepseek/deepseek-v4-pro",
		"kimi/kimi-k3",
		"xai/grok-4.5",
		"zai/glm-5.1",
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
	t.Setenv("DEEPSEEK_API_KEY", "test-key")
	t.Setenv("KIMI_API_KEY", "test-key")
	t.Setenv("XAI_API_KEY", "test-key")
	t.Setenv("ZAI_API_KEY", "test-key")

	// Keyed provider-prefixed on half of them, bare on the rest: both forms
	// name the same model.
	pinned := ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Media: false}}

	g := genkit.Init(context.Background(), genkit.WithPlugins(
		&anthropic.Anthropic{Models: map[string]ai.ModelOptions{
			"anthropic/claude-sonnet-4-5-20250929": pinned}},
		&dashscope.DashScope{Models: map[string]ai.ModelOptions{"qwen-plus": pinned}},
		&deepseek.DeepSeek{Models: map[string]ai.ModelOptions{
			"deepseek/deepseek-v4-pro": pinned}},
		&kimi.Kimi{Models: map[string]ai.ModelOptions{"kimi-k3": pinned}},
		&xai.XAI{Models: map[string]ai.ModelOptions{"xai/grok-4.5": pinned}},
		&zai.ZAI{Models: map[string]ai.ModelOptions{"glm-5.1": pinned}},
	))

	for _, name := range []string{
		"anthropic/claude-sonnet-4-5-20250929",
		"dashscope/qwen-plus",
		"deepseek/deepseek-v4-pro",
		"kimi/kimi-k3",
		"xai/grok-4.5",
		"zai/glm-5.1",
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

// TestClosedEnumsUseNamedTypes pins the family rule for closed sets: a config
// field whose schema declares an enum is typed as a named string type of its
// package, so the allowed values are discoverable as constants in code rather
// than only in the schema. Open sets carry no enum and stay bare strings,
// since constants would imply a closure the provider's docs refuse to give.
// Each new plugin adds its config here.
func TestClosedEnumsUseNamedTypes(t *testing.T) {
	configs := map[string]any{
		"anthropic": anthropic.ChatConfig{},
		"dashscope": dashscope.ChatConfig{},
		"deepseek":  deepseek.ChatConfig{},
		"kimi":      kimi.ChatConfig{},
		"xai":       xai.ChatConfig{},
		"zai":       zai.ChatConfig{},
	}
	for name, config := range configs {
		t.Run(name, func(t *testing.T) {
			checkClosedEnums(t, "", reflect.TypeOf(config))
		})
	}
}

// checkClosedEnums walks a config struct's fields, recursing through structs
// and pointers to structs, and fails for any enum-tagged field typed as a
// bare string.
func checkClosedEnums(t *testing.T, path string, typ reflect.Type) {
	t.Helper()
	for i := range typ.NumField() {
		field := typ.Field(i)
		name := path + field.Name
		ft := field.Type
		for ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		if ft.Kind() == reflect.Struct {
			checkClosedEnums(t, name+".", ft)
			continue
		}
		if !strings.Contains(field.Tag.Get("jsonschema"), "enum=") {
			continue
		}
		if ft.Kind() == reflect.String && ft.PkgPath() == "" {
			t.Errorf("%s declares a schema enum on a bare string, want a named string type with exported constants", name)
		}
	}
}
