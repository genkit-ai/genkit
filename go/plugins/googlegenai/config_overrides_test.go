// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"sort"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/plugins/internal"
	"google.golang.org/genai"
)

// curated pairs each reflected config schema with the overrides written for it.
type curated struct {
	schema    map[string]any
	overrides internal.SchemaOverrides
}

func curatedConfigs() map[string]curated {
	return map[string]curated{
		"Gemini": {configToMap(genai.GenerateContentConfig{}), gccOverrides},
		"Imagen": {configToMap(genai.GenerateImagesConfig{}), gicOverrides},
		"Veo":    {configToMap(genai.GenerateVideosConfig{}), gvcOverrides},
	}
}

// TestConfigOverridePathsResolve is the guard that keeps the override maps
// honest. Applying a path the genai SDK no longer has is a silent no-op, so an
// entry left behind by a renamed field would quietly stop describing anything.
func TestConfigOverridePathsResolve(t *testing.T) {
	for name, c := range curatedConfigs() {
		if stale := c.overrides.UnresolvedPaths(c.schema); len(stale) > 0 {
			t.Errorf("%s: override paths no longer in the SDK schema: %v", name, stale)
		}
	}
}

// TestConfigDescriptionsApplied checks the curated text actually lands, at the
// top level and inside an array's items.
func TestConfigDescriptionsApplied(t *testing.T) {
	for name, c := range curatedConfigs() {
		for path, want := range c.overrides.Descriptions {
			target, ok := internal.SchemaAt(c.schema, path).(map[string]any)
			if !ok {
				continue // reported by TestConfigOverridePathsResolve
			}
			if got, _ := target["description"].(string); got != want {
				t.Errorf("%s: description for %q\n got: %q\nwant: %q", name, path, got, want)
			}
		}
	}
}

// TestManagedFieldsHiddenButAccepted pins what hiding means here: the property
// stays present and typeless, so the dev UI skips it while a caller's value
// still survives input validation and reaches the plugin check that names the
// Genkit option to use instead.
func TestManagedFieldsHiddenButAccepted(t *testing.T) {
	schema := configToMap(genai.GenerateContentConfig{})
	for _, path := range gccOverrides.Hidden {
		if got := internal.SchemaAt(schema, path); got != true {
			t.Errorf("%q = %v, want true (typeless, so the dev UI skips it)", path, got)
		}
	}

	// Built-in API tools stay visible so the dev UI can offer them; only the
	// function declarations Genkit owns are hidden from tools[].
	item, _ := internal.SchemaAt(schema, "tools[]").(map[string]any)
	itemProps, _ := item["properties"].(map[string]any)
	for _, want := range []string{"googleSearch", "retrieval", "codeExecution"} {
		if _, ok := itemProps[want]; !ok {
			t.Errorf("tools[].%s should remain visible, got %v", want, keys(itemProps))
		}
	}
}

// TestHidingKeepsObjectsClosed pins the reason a hidden property is replaced
// rather than deleted: every object stays strict, including the ones that hide
// something. Deleting would fail a caller's value as unknown, and reopening the
// parent to let it through gives up rejecting real typos in that object.
func TestHidingKeepsObjectsClosed(t *testing.T) {
	for name, c := range curatedConfigs() {
		if c.schema["additionalProperties"] != false {
			t.Errorf("%s root additionalProperties = %v, want false", name, c.schema["additionalProperties"])
		}
	}
	schema := configToMap(genai.GenerateContentConfig{})
	for _, path := range []string{"tools[]", "thinkingConfig"} {
		obj, ok := internal.SchemaAt(schema, path).(map[string]any)
		if ok && obj["additionalProperties"] != false {
			t.Errorf("%s additionalProperties = %v, want false", path, obj["additionalProperties"])
		}
	}
}

// TestConfigToMapPointerVariant covers the &Config{} call sites (e.g.
// model_type.DefaultConfig), where overrides must apply just the same.
func TestConfigToMapPointerVariant(t *testing.T) {
	schema := configToMap(&genai.GenerateContentConfig{})
	if got := internal.SchemaAt(schema, "systemInstruction"); got != true {
		t.Errorf("systemInstruction = %v, want true (hidden) for a pointer config too", got)
	}
	prop, ok := internal.SchemaAt(schema, "temperature").(map[string]any)
	if !ok || prop["description"] == "" {
		t.Errorf("temperature should carry a description for a pointer config too: %#v", prop)
	}
}

// TestModelSupportsNativeConstraints keeps the constrained-output claim honest:
// only models whose schema advertises responseJsonSchema handling may set it.
func TestGeminiHidesResponseSchemaFields(t *testing.T) {
	schema := configToMap(genai.GenerateContentConfig{})
	for _, path := range []string{"responseSchema", "responseMimeType", "responseJsonSchema"} {
		if got := internal.SchemaAt(schema, path); got != true {
			t.Errorf("%s = %v, want true; ai.WithOutputType owns it", path, got)
		}
	}
	if !strings.Contains(strings.Join(gccOverrides.Hidden, ","), "candidateCount") {
		t.Error("candidateCount should stay hidden; the plugin pins it to 1")
	}
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
