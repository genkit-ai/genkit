// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

// --- Gemini (text generation) ---

// ModelRef creates a ModelRef for a Gemini model.
// The name should include provider prefix (e.g., "googleai/gemini-2.0-flash").
func ModelRef(name string, config *genai.GenerateContentConfig) ai.ModelRef {
	return ai.NewModelRef(name, config)
}

// GoogleAIModelRef creates a ModelRef for a Google AI Gemini model.
//
// Deprecated: Use ModelRef with full name instead.
func GoogleAIModelRef(id string, config *genai.GenerateContentConfig) ai.ModelRef {
	return ai.NewModelRef(googleAIProvider+"/"+id, config)
}

// VertexAIModelRef creates a ModelRef for a Vertex AI Gemini model.
//
// Deprecated: Use ModelRef with full name instead.
func VertexAIModelRef(id string, config *genai.GenerateContentConfig) ai.ModelRef {
	return ai.NewModelRef(vertexAIProvider+"/"+id, config)
}

// --- Image generation (Imagen) ---

// ImageModelRef creates a ModelRef for an image generation model.
// The name should include provider prefix (e.g., "googleai/imagen-3.0-generate-001").
func ImageModelRef(name string, config *genai.GenerateImagesConfig) ai.ModelRef {
	return ai.NewModelRef(name, config)
}

// --- Video generation (Veo) ---

// VideoModelRef creates a ModelRef for a video generation model.
// The name should include provider prefix (e.g., "googleai/veo-2.0-generate-001").
func VideoModelRef(name string, config *genai.GenerateVideosConfig) ai.ModelRef {
	return ai.NewModelRef(name, config)
}

// --- Embedders ---

// EmbedderRef creates an EmbedderRef for an embedding model.
// The name should include provider prefix (e.g., "googleai/text-embedding-004").
func EmbedderRef(name string, config *genai.EmbedContentConfig) ai.EmbedderRef {
	return ai.NewEmbedderRef(name, config)
}
