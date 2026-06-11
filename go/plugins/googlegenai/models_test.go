// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestImagen4ModelOptions(t *testing.T) {
	tests := []struct {
		name  string
		label string
	}{
		{imagen40FastGenerate001, "Google AI - Imagen 4 Fast Generate 001"},
		{imagen40Generate001, "Google AI - Imagen 4 Generate 001"},
		{imagen40UltraGenerate001, "Google AI - Imagen 4 Ultra Generate 001"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ClassifyModel(tt.name); got != ModelTypeImagen {
				t.Fatalf("ClassifyModel(%q) = %v, want %v", tt.name, got, ModelTypeImagen)
			}

			opts := GetModelOptions(tt.name, googleAIProvider)
			if opts.Label != tt.label {
				t.Fatalf("label = %q, want %q", opts.Label, tt.label)
			}
			if opts.Stage != ai.ModelStageStable {
				t.Fatalf("stage = %v, want %v", opts.Stage, ai.ModelStageStable)
			}
			if opts.Supports != &Media {
				t.Fatalf("supports = %#v, want Media", opts.Supports)
			}
			if opts.ConfigSchema == nil {
				t.Fatal("ConfigSchema should be populated for Imagen 4")
			}
		})
	}
}

func TestListModelsIncludesImagen4ForGoogleAI(t *testing.T) {
	models, err := listModels(googleAIProvider)
	if err != nil {
		t.Fatalf("listModels(%q) error = %v", googleAIProvider, err)
	}

	for _, name := range []string{
		imagen40FastGenerate001,
		imagen40Generate001,
		imagen40UltraGenerate001,
	} {
		if _, ok := models[name]; !ok {
			t.Fatalf("Google AI models missing %q", name)
		}
	}
}
