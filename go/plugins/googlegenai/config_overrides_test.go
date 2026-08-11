// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"sort"
	"testing"

	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
)

// TestConfigToMap_GenerateContentConfig verifies that the schema exposed for
// the Gemini chat config drops fields the plugin manages on the user's behalf
// and adds the curated descriptions used by the Genkit Developer UI.
func TestConfigToMap_GenerateContentConfig(t *testing.T) {
	schema := configToMap(genai.GenerateContentConfig{})

	for _, hidden := range gccOverrides.hidden {
		assertHidden(t, "Gemini", schema, hidden)
	}

	// Sanity: built-in API tools still surface in tools[]'s item shape so the
	// dev UI can let users enable them. Only functionDeclarations should have
	// been removed from there.
	if toolItem := navigate(schema, "tools", "[]"); toolItem != nil {
		if itemProps, ok := toolItem["properties"].(map[string]any); ok {
			for _, expected := range []string{"googleSearch", "retrieval", "codeExecution"} {
				if _, ok := itemProps[expected]; !ok {
					t.Errorf("Gemini schema: tools[].%s should remain visible — got %v", expected, keys(itemProps))
				}
			}
			if got := itemProps["functionDeclarations"]; got != true {
				t.Errorf("Gemini schema: tools[].functionDeclarations = %v, want true (hidden)", got)
			}
		}
	}

	checkDescriptions(t, "Gemini", schema, gccOverrides.descriptions)
}

// TestConfigToMap_HidingKeepsObjectsClosed pins the reason a hidden property
// is replaced rather than deleted: every object stays strict, including the
// ones that hide something.
//
// Deleting the property would fail a caller's value as an unknown one, so it
// would have to be paid for by forcing additionalProperties open on the
// parent, and that gives up rejecting unknown fields for every other property
// of that object too. The root and tools[] are exactly the objects that hide
// something, so they are the ones that would have gone open.
func TestConfigToMap_HidingKeepsObjectsClosed(t *testing.T) {
	schema := configToMap(genai.GenerateContentConfig{})

	if schema["additionalProperties"] != false {
		t.Errorf("root additionalProperties = %v, want false (hiding must not open the object)", schema["additionalProperties"])
	}
	if toolItem := navigate(schema, "tools", "[]"); toolItem != nil {
		if toolItem["additionalProperties"] != false {
			t.Errorf("tools[] additionalProperties = %v, want false (hiding must not open the object)", toolItem["additionalProperties"])
		}
	}
	if thinking := navigate(schema, "thinkingConfig"); thinking != nil {
		if thinking["additionalProperties"] != false {
			t.Errorf("thinkingConfig additionalProperties = %v, want false", thinking["additionalProperties"])
		}
	}
	for _, name := range []string{"Imagen", "Veo"} {
		var s map[string]any
		if name == "Imagen" {
			s = configToMap(genai.GenerateImagesConfig{})
		} else {
			s = configToMap(genai.GenerateVideosConfig{})
		}
		if s["additionalProperties"] != false {
			t.Errorf("%s schema additionalProperties = %v, want false (it hides nothing)", name, s["additionalProperties"])
		}
	}
}

func TestConfigToMap_GenerateImagesConfig(t *testing.T) {
	checkDescriptions(t, "Imagen", configToMap(genai.GenerateImagesConfig{}), gicOverrides.descriptions)
}

func TestConfigToMap_GenerateVideosConfig(t *testing.T) {
	checkDescriptions(t, "Veo", configToMap(genai.GenerateVideosConfig{}), gvcOverrides.descriptions)
}

// TestConfigToMap_PointerVariant covers the &Config{} call sites (e.g.
// model_type.DefaultConfig) to make sure overrides apply for pointer values
// too, not just value receivers.
func TestConfigToMap_PointerVariant(t *testing.T) {
	schema := configToMap(&genai.GenerateContentConfig{})
	props, _ := schema["properties"].(map[string]any)
	if got := props["systemInstruction"]; got != true {
		t.Errorf("systemInstruction = %v, want true (hidden) for pointer config too", got)
	}
	if prop, ok := props["temperature"].(map[string]any); !ok || prop["description"] == "" {
		t.Errorf("temperature should carry a description for pointer config too: %#v", prop)
	}
}

// TestApplyConfigOverrides_BestEffort exercises bogus paths that don't
// resolve in the schema. They must silently no-op rather than panicking,
// since this code runs during package init and a panic would prevent the
// plugin from loading.
func TestApplyConfigOverrides_BestEffort(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("applyConfigOverrides panicked on bogus paths: %v", r)
		}
	}()
	r := jsonschema.Reflector{
		DoNotReference: true,
		ExpandedStruct: true,
		IgnoredTypes:   []any{genai.Schema{}},
	}
	schema := r.Reflect(genai.GenerateContentConfig{})
	applyConfigOverrides(schema, configOverrides{
		descriptions: map[string]string{
			"doesNotExist":              "x",
			"alsoMissing.deeplyMissing": "x",
			"tools[].notARealField":     "x",
			"completely[].fake[].path":  "x",
			"thinkingConfig.gone":       "x",
		},
		hidden: []string{
			"doesNotExist",
			"missing[].alsoMissing",
			"tools[].notARealField",
			"[]",
		},
	})
}

func checkDescriptions(t *testing.T, label string, schema map[string]any, want map[string]string) {
	t.Helper()
	for path, desc := range want {
		target := navigate(schema, parsePath(path)...)
		if target == nil {
			// Stale entry: either upstream renamed the field or we removed it.
			// Surface the mismatch loudly so the override map stays honest.
			t.Errorf("%s schema: described field %q missing — update %s overrides", label, path, label)
			continue
		}
		if got, _ := target["description"].(string); got != desc {
			t.Errorf("%s schema: description for %q\n got: %q\nwant: %q", label, path, got, desc)
		}
	}
}

// assertHidden checks that a top-level or nested property (per parsePath
// notation) resolves to the permissive `true` schema.
//
// Hidden means typeless, not absent. The dev UI renders only properties whose
// type it recognizes, so `true` is enough to keep the field out of the form,
// while leaving it in properties is what lets a caller's value survive input
// validation and reach the plugin check that names the primitive to use.
func assertHidden(t *testing.T, label string, schema map[string]any, path string) {
	t.Helper()
	steps := parsePath(path)
	leaf := steps[len(steps)-1]
	parent := schema
	if len(steps) > 1 {
		parent = navigate(schema, steps[:len(steps)-1]...)
	}
	if parent == nil {
		return // upstream removed the parent — nothing to assert
	}
	props, _ := parent["properties"].(map[string]any)
	if props == nil && len(steps) == 1 {
		t.Fatalf("%s schema missing top-level properties", label)
	}
	got, present := props[leaf]
	if !present {
		t.Errorf("%s schema: %q is absent, want the permissive true schema — properties %v", label, path, keys(props))
		return
	}
	if got != true {
		t.Errorf("%s schema: %q = %v, want true (typeless, so the dev UI skips it)", label, path, got)
	}
}

// navigate descends a JSON Schema map by walking `properties` for ordinary
// step names and `items` for "[]" steps. Returns nil if the path doesn't
// resolve.
func navigate(schema map[string]any, steps ...string) map[string]any {
	cur := schema
	for _, step := range steps {
		if cur == nil {
			return nil
		}
		if step == "[]" {
			next, _ := cur["items"].(map[string]any)
			cur = next
			continue
		}
		props, _ := cur["properties"].(map[string]any)
		if props == nil {
			return nil
		}
		next, _ := props[step].(map[string]any)
		cur = next
	}
	return cur
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
