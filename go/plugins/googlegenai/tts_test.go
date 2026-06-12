// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
)

// ttsModels are the dedicated text-to-speech models registered for Google AI.
var ttsModels = []string{
	gemini25FlashPreviewTTS,
	gemini25ProPreviewTTS,
	gemini31FlashTTSPreview,
}

func TestTTSModelClassification(t *testing.T) {
	t.Parallel()

	// TTS models keep the "gemini" prefix, so they classify as regular Gemini
	// models and reuse the standard content-generation action path.
	for _, name := range ttsModels {
		if mt := ClassifyModel(name); mt != ModelTypeGemini {
			t.Errorf("ClassifyModel(%q) = %v, want ModelTypeGemini", name, mt)
		}
	}
}

func TestTTSModelOptions(t *testing.T) {
	t.Parallel()

	for _, name := range ttsModels {
		t.Run(name, func(t *testing.T) {
			opts := GetModelOptions(name, googleAIProvider)
			if opts.Supports == nil {
				t.Fatal("Supports is nil")
			}
			if opts.Supports != &TTSSupports {
				t.Errorf("Supports = %p, want &TTSSupports (%p)", opts.Supports, &TTSSupports)
			}
			// TTS models emit audio, declared with the shared "media" output
			// token (matching Imagen/Veo), and do not converse or call tools.
			if got := opts.Supports.Output; len(got) != 1 || got[0] != "media" {
				t.Errorf("Output = %v, want [media]", got)
			}
			if opts.Supports.Tools {
				t.Error("Tools = true, want false")
			}
			if opts.Supports.Multiturn {
				t.Error("Multiturn = true, want false")
			}
			if opts.Supports.Media {
				t.Error("Media = true, want false")
			}
			if opts.Stage != ai.ModelStageUnstable {
				t.Errorf("Stage = %v, want Unstable", opts.Stage)
			}
			// ConfigSchema falls back to GenerateContentConfig, which carries the
			// speechConfig field used to select voices.
			if opts.ConfigSchema == nil {
				t.Error("ConfigSchema is nil, want GenerateContentConfig schema")
			}
		})
	}
}

func TestTTSModelsRegisteredForGoogleAIOnly(t *testing.T) {
	t.Parallel()

	googleModels, err := listModels(googleAIProvider)
	if err != nil {
		t.Fatalf("listModels(googleAI) error = %v", err)
	}
	vertexModels, err := listModels(vertexAIProvider)
	if err != nil {
		t.Fatalf("listModels(vertexAI) error = %v", err)
	}

	for _, name := range ttsModels {
		if _, ok := googleModels[name]; !ok {
			t.Errorf("Google AI model list is missing TTS model %q", name)
		}
		if _, ok := vertexModels[name]; ok {
			t.Errorf("Vertex AI model list unexpectedly includes TTS model %q", name)
		}
	}
}
