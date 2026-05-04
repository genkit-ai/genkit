// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"sort"
	"testing"

	"google.golang.org/genai"
)

// TestConfigToMap_GenerateContentConfig verifies that the schema exposed for
// the Gemini chat config drops fields the plugin manages on the user's behalf
// and adds the curated descriptions used by the Genkit Developer UI.
func TestConfigToMap_GenerateContentConfig(t *testing.T) {
	schema := configToMap(genai.GenerateContentConfig{})
	props := schemaProps(t, schema)

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
			if _, ok := itemProps["functionDeclarations"]; ok {
				t.Error("Gemini schema: tools[].functionDeclarations should be hidden")
			}
		}
	}

	checkDescriptions(t, "Gemini", props, gccOverrides.descriptions)
}

func TestConfigToMap_GenerateImagesConfig(t *testing.T) {
	props := schemaProps(t, configToMap(genai.GenerateImagesConfig{}))
	checkDescriptions(t, "Imagen", props, gicOverrides.descriptions)
}

func TestConfigToMap_GenerateVideosConfig(t *testing.T) {
	props := schemaProps(t, configToMap(genai.GenerateVideosConfig{}))
	checkDescriptions(t, "Veo", props, gvcOverrides.descriptions)
}

// TestConfigToMap_PointerVariant covers the &Config{} call sites (e.g.
// model_type.DefaultConfig) to make sure overrides apply for pointer values
// too, not just value receivers.
func TestConfigToMap_PointerVariant(t *testing.T) {
	props := schemaProps(t, configToMap(&genai.GenerateContentConfig{}))
	if _, present := props["systemInstruction"]; present {
		t.Errorf("systemInstruction must be hidden for pointer config too")
	}
	if prop, ok := props["temperature"].(map[string]any); !ok || prop["description"] == "" {
		t.Errorf("temperature should carry a description for pointer config too: %#v", prop)
	}
}

func schemaProps(t *testing.T, schema map[string]any) map[string]any {
	t.Helper()
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema missing properties: %#v", schema)
	}
	return props
}

func checkDescriptions(t *testing.T, label string, props map[string]any, want map[string]string) {
	t.Helper()
	for name, desc := range want {
		prop, ok := props[name].(map[string]any)
		if !ok {
			// Stale entry: either upstream renamed the field or we removed it.
			// Surface the mismatch loudly so the override map stays honest.
			t.Errorf("%s schema: described field %q missing — update %s overrides", label, name, label)
			continue
		}
		if got, _ := prop["description"].(string); got != desc {
			t.Errorf("%s schema: description for %q\n got: %q\nwant: %q", label, name, got, desc)
		}
	}
}

// assertHidden checks that a top-level or nested property (per parseHidePath
// notation) is absent from the resolved schema map.
func assertHidden(t *testing.T, label string, schema map[string]any, path string) {
	t.Helper()
	steps := parseHidePath(path)
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
	if _, present := props[leaf]; present {
		t.Errorf("%s schema: %q must be hidden — found in properties %v", label, path, keys(props))
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
