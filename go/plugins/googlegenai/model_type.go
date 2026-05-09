// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

// ModelType categorizes models by their generation modality.
type ModelType int

const (
	ModelTypeUnknown  ModelType = iota
	ModelTypeGemini             // Text/multimodal generation (gemini-*, gemma-*)
	ModelTypeImagen             // Image generation (imagen-*)
	ModelTypeVeo                // Video generation (veo-*), long-running
	ModelTypeEmbedder           // Embedding models (*embedding*)
)

// ClassifyModel determines the model type from its name.
// This is the single source of truth for model type classification.
func ClassifyModel(name string) ModelType {
	switch {
	case strings.HasPrefix(name, "veo"):
		return ModelTypeVeo
	case strings.HasPrefix(name, "imagen"), strings.HasPrefix(name, "image"):
		return ModelTypeImagen
	case strings.HasPrefix(name, "gemini"), strings.HasPrefix(name, "gemma"):
		return ModelTypeGemini
	case strings.Contains(name, "embedding"):
		// Covers: text-embedding-*, embedding-*, textembedding-*, multimodalembedding
		return ModelTypeEmbedder
	default:
		return ModelTypeUnknown
	}
}

// ActionType returns the appropriate API action type for this model type.
func (mt ModelType) ActionType() api.ActionType {
	switch mt {
	case ModelTypeVeo:
		return api.ActionTypeBackgroundModel
	case ModelTypeEmbedder:
		return api.ActionTypeEmbedder
	default:
		return api.ActionTypeModel
	}
}

// DefaultSupports returns the default ModelSupports for this model type.
func (mt ModelType) DefaultSupports() *ai.ModelSupports {
	switch mt {
	case ModelTypeGemini:
		return &Multimodal
	case ModelTypeImagen:
		return &Media
	case ModelTypeVeo:
		return &VeoSupports
	default:
		return nil
	}
}

// DefaultConfig returns the default config struct for this model type.
func (mt ModelType) DefaultConfig() any {
	switch mt {
	case ModelTypeGemini:
		return &genai.GenerateContentConfig{}
	case ModelTypeImagen:
		return &genai.GenerateImagesConfig{}
	case ModelTypeVeo:
		return &genai.GenerateVideosConfig{}
	case ModelTypeEmbedder:
		return &genai.EmbedContentConfig{}
	default:
		return nil
	}
}
