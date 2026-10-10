// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// hitUnknownModelFallback reports whether GetModelOptions fell through to the
// default options instead of finding a concrete map entry. Every model is
// stable now, so the stage no longer distinguishes them; the label does. A
// registered model carries its curated label ("Gemini 3.6 Flash") while the
// fallback is labelled with the raw model ID.
func hitUnknownModelFallback(opts ai.ModelOptions, name string) bool {
	return strings.HasSuffix(opts.Label, name)
}

// No Imagen model is curated now that every backend has retired them, but an
// Imagen ID still resolves through the generic Imagen path with its config
// schema, so code that names one keeps compiling and gets the provider's own
// answer.
func TestImagenModelOptions(t *testing.T) {
	const name = "imagen-4.0-generate-001"
	if got := ClassifyModel(name); got != ModelTypeImagen {
		t.Fatalf("ClassifyModel(%q) = %v, want %v", name, got, ModelTypeImagen)
	}
	opts := GetModelOptions(name, googleAIProvider)
	if opts.Supports != &Media {
		t.Errorf("supports = %#v, want Media", opts.Supports)
	}
	if opts.ConfigSchema == nil {
		t.Error("ConfigSchema should be populated for an Imagen model")
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
// its concrete map entry (curated label, multimodal supports) rather than the
// unknown-model fallback (defaultGeminiOpts).
func TestNewGeminiModelsResolveToRegisteredEntries(t *testing.T) {
	for _, name := range newlyRegisteredGeminiModels {
		if got := ClassifyModel(name); got != ModelTypeGemini {
			t.Errorf("ClassifyModel(%q) = %v, want ModelTypeGemini", name, got)
		}

		opts := GetModelOptions(name, googleAIProvider)
		if hitUnknownModelFallback(opts, name) {
			t.Errorf("GetModelOptions(%q).Label = %q, want a curated label (hit the unknown-model fallback)", name, opts.Label)
		}
		if opts.Supports == nil || !opts.Supports.Multiturn || !opts.Supports.Media {
			t.Errorf("GetModelOptions(%q): expected multimodal supports, got %+v", name, opts.Supports)
		}
		if opts.ConfigSchema == nil {
			t.Errorf("GetModelOptions(%q): ConfigSchema is nil", name)
		}
	}
}

// TestNewGeminiModelsProviderSplit pins that both backends register the same
// Gemini models. gemini-3.1-flash-lite used to be withheld from Google AI, but
// the Gemini API documents it as stable there, so the split is gone.
func TestNewGeminiModelsProviderSplit(t *testing.T) {
	for _, name := range newlyRegisteredGeminiModels {
		if !slices.Contains(vertexAIModels, name) {
			t.Errorf("vertexAIModels missing %q", name)
		}
		if !slices.Contains(googleAIModels, name) {
			t.Errorf("googleAIModels missing %q", name)
		}
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

// TestNewlyRegisteredModels pins that the models added from the backend audit
// are reachable on both backends. A model listed for a provider but missing
// from supportedGeminiModels still resolves, silently, to the unknown-model
// fallback, which is labelled with the raw model ID.
func TestNewlyRegisteredModels(t *testing.T) {
	for _, name := range []string{gemini37Flash, gemini36Flash, gemini35FlashLite, gemini31ProPreview, gemini31FlashLiteImage} {
		if got := ClassifyModel(name); got != ModelTypeGemini {
			t.Errorf("ClassifyModel(%q) = %v, want ModelTypeGemini", name, got)
		}

		for _, provider := range []string{googleAIProvider, vertexAIProvider} {
			opts := GetModelOptions(name, provider)
			if hitUnknownModelFallback(opts, name) {
				t.Errorf("GetModelOptions(%q, %q).Label = %q, want a curated label (hit the unknown-model fallback)", name, provider, opts.Label)
			}
			if opts.Supports == nil || !opts.Supports.Multiturn || !opts.Supports.Media {
				t.Errorf("GetModelOptions(%q, %q): expected multimodal supports, got %+v", name, provider, opts.Supports)
			}
			if opts.ConfigSchema == nil {
				t.Errorf("GetModelOptions(%q, %q): ConfigSchema is nil", name, provider)
			}
		}

		if !slices.Contains(vertexAIModels, name) {
			t.Errorf("vertexAIModels missing %q", name)
		}
		if !slices.Contains(googleAIModels, name) {
			t.Errorf("googleAIModels missing %q", name)
		}
	}
}
