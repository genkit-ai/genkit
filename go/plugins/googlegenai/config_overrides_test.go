// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"testing"

	"google.golang.org/genai"
)

// TestConfigToMap_GenerateContentConfig verifies that the schema exposed for
// the Gemini chat config drops fields the plugin manages on the user's behalf
// and adds the curated descriptions used by the Genkit Developer UI.
func TestConfigToMap_GenerateContentConfig(t *testing.T) {
	props := schemaProps(t, configToMap(genai.GenerateContentConfig{}))

	for _, hidden := range gccOverrides.hidden {
		if _, present := props[hidden]; present {
			t.Errorf("hidden field %q must not appear in the Gemini config schema", hidden)
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
