// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"slices"
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

// newlyRegisteredGeminiModels are the P0 Gemini-family models added for
// Go<->JS registration parity.
var newlyRegisteredGeminiModels = []string{
	gemini35Flash,
	gemini31FlashLite,
	gemini31FlashImage,
	gemini3ProImage,
}

// TestNewGeminiModelsResolveToRegisteredEntries verifies each model resolves to
// its concrete map entry (Stable stage, multimodal supports, labelled) rather
// than the unknown-model fallback (defaultGeminiOpts, which is Unstable).
func TestNewGeminiModelsResolveToRegisteredEntries(t *testing.T) {
	for _, name := range newlyRegisteredGeminiModels {
		if got := ClassifyModel(name); got != ModelTypeGemini {
			t.Errorf("ClassifyModel(%q) = %v, want ModelTypeGemini", name, got)
		}

		opts := GetModelOptions(name, googleAIProvider)
		if opts.Stage != ai.ModelStageStable {
			t.Errorf("GetModelOptions(%q).Stage = %q, want Stable (likely hit the unknown-model fallback)", name, opts.Stage)
		}
		if opts.Supports == nil || !opts.Supports.Multiturn || !opts.Supports.Media {
			t.Errorf("GetModelOptions(%q): expected multimodal supports, got %+v", name, opts.Supports)
		}
		if opts.ConfigSchema == nil {
			t.Errorf("GetModelOptions(%q): ConfigSchema is nil", name)
		}
	}
}

// TestNewGeminiModelsProviderSplit pins the per-provider registration from the
// ticket: GoogleAI gets all except gemini-3.1-flash-lite; VertexAI gets all four.
func TestNewGeminiModelsProviderSplit(t *testing.T) {
	for _, name := range newlyRegisteredGeminiModels {
		if !slices.Contains(vertexAIModels, name) {
			t.Errorf("vertexAIModels missing %q", name)
		}
	}

	for _, name := range []string{gemini35Flash, gemini31FlashImage, gemini3ProImage} {
		if !slices.Contains(googleAIModels, name) {
			t.Errorf("googleAIModels missing %q", name)
		}
	}

	// gemini-3.1-flash-lite is VertexAI-only per the ticket.
	if slices.Contains(googleAIModels, gemini31FlashLite) {
		t.Errorf("googleAIModels should not contain %q (VertexAI only)", gemini31FlashLite)
	}
}

// TestGeminiEmbedding2Registered verifies the embedder resolves via the embedder
// path with the correct dimensionality and multimodal input.
func TestGeminiEmbedding2Registered(t *testing.T) {
	// gemini-embedding-2 starts with the "gemini" prefix but must classify as an
	// embedder, not a generative model (the "embedding" check precedes the
	// "gemini" prefix check in ClassifyModel).
	if got := ClassifyModel(geminiEmbedding2); got != ModelTypeEmbedder {
		t.Errorf("ClassifyModel(%q) = %v, want ModelTypeEmbedder", geminiEmbedding2, got)
	}

	opts := GetEmbedderOptions(geminiEmbedding2, googleAIProvider)

	if opts.Dimensions != 3072 {
		t.Errorf("GetEmbedderOptions(%q).Dimensions = %d, want 3072", geminiEmbedding2, opts.Dimensions)
	}
	if opts.Supports == nil {
		t.Fatalf("GetEmbedderOptions(%q): Supports is nil", geminiEmbedding2)
	}
	for _, want := range []string{"text", "image", "video"} {
		if !slices.Contains(opts.Supports.Input, want) {
			t.Errorf("GetEmbedderOptions(%q): Input missing %q, got %v", geminiEmbedding2, want, opts.Supports.Input)
		}
	}
}
