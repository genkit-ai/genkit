// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestIsGemmaModelName(t *testing.T) {
	cases := map[string]bool{
		"gemma-4-31b-it":            true,
		"gemma-4-26b-a4b-it":        true,
		"googleai/gemma-4-31b-it":   true,
		"gemini-2.5-flash":          false,
		"googleai/gemini-2.5-flash": false,
		"imagen-4.0-generate-001":   false,
	}
	for name, want := range cases {
		if got := isGemmaModelName(name); got != want {
			t.Errorf("isGemmaModelName(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestGemmaConfigSchemaClampsTemperature(t *testing.T) {
	schema := gemmaConfigSchema()
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties map: %v", schema)
	}
	temp, ok := props["temperature"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no temperature property: %v", props["temperature"])
	}
	if temp["maximum"] != 1.0 {
		t.Errorf("temperature maximum = %v, want 1.0", temp["maximum"])
	}
	if temp["minimum"] != 0.0 {
		t.Errorf("temperature minimum = %v, want 0.0", temp["minimum"])
	}
}

func TestUnregisteredGemmaModelGetsClamp(t *testing.T) {
	// A gemma model that is not individually registered should still get the
	// temperature clamp, matching the JS wildcard (gemma-${string}) behavior.
	opts := GetModelOptions("gemma-5-future-it", googleAIProvider)
	props, ok := opts.ConfigSchema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("no properties in schema for unregistered gemma model")
	}
	temp, ok := props["temperature"].(map[string]any)
	if !ok {
		t.Fatalf("no temperature property for unregistered gemma model")
	}
	if temp["maximum"] != 1.0 {
		t.Errorf("temperature maximum = %v, want 1.0 (clamp should apply to all gemma)", temp["maximum"])
	}
}

func TestStripGemmaThoughts(t *testing.T) {
	keep := ai.NewTextPart("keep me")

	withSig := ai.NewTextPart("has a thought signature")
	withSig.Metadata = map[string]any{"signature": []byte("sig")}

	parts := []*ai.Part{
		keep,
		ai.NewReasoningPart("secret reasoning", []byte("sig")),
		withSig,
		ai.NewTextPart("also keep"),
	}

	got := stripGemmaThoughts(parts)

	if len(got) != 2 {
		t.Fatalf("stripped to %d parts, want 2: %+v", len(got), got)
	}
	if got[0].Text != "keep me" || got[1].Text != "also keep" {
		t.Errorf("unexpected surviving parts: %q, %q", got[0].Text, got[1].Text)
	}
	for _, p := range got {
		if p.IsReasoning() {
			t.Error("reasoning part survived the strip")
		}
		if p.Metadata != nil {
			if _, ok := p.Metadata["signature"]; ok {
				t.Error("part carrying a thought signature survived the strip")
			}
		}
	}

	// An all-thoughts turn strips to empty; the generate path skips such turns
	// rather than send an empty content block.
	allThoughts := stripGemmaThoughts([]*ai.Part{
		ai.NewReasoningPart("only a thought", []byte("sig")),
	})
	if len(allThoughts) != 0 {
		t.Errorf("all-thoughts content stripped to %d parts, want 0", len(allThoughts))
	}
}
