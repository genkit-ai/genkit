// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

// Model capability definitions - these describe what different model types support.
var (
	// BasicText describes model capabilities for text-only Gemini models.
	BasicText = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		ToolChoice: true,
		SystemRole: true,
		Media:      false,
	}

	// Multimodal describes model capabilities for multimodal Gemini models.
	Multimodal = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		ToolChoice:  true,
		SystemRole:  true,
		Media:       true,
		Constrained: ai.ConstrainedSupportNoTools,
	}

	// Media describes model capabilities for image generation models (Imagen).
	Media = ai.ModelSupports{
		Multiturn:  false,
		Tools:      false,
		SystemRole: false,
		Media:      true,
		Output:     []string{"media"},
	}

	// VeoSupports describes model capabilities for video generation models (Veo).
	VeoSupports = ai.ModelSupports{
		Media:       true,
		Multiturn:   false,
		Tools:       false,
		SystemRole:  false,
		Output:      []string{"media"},
		LongRunning: true,
	}
)

// Default options for unknown models of each type.
var (
	defaultGeminiOpts = ai.ModelOptions{
		Supports:     &Multimodal,
		Stage:        ai.ModelStageUnstable,
		ConfigSchema: configToMap(genai.GenerateContentConfig{}),
	}

	defaultImagenOpts = ai.ModelOptions{
		Supports:     &Media,
		Stage:        ai.ModelStageUnstable,
		ConfigSchema: configToMap(genai.GenerateImagesConfig{}),
	}

	defaultVeoOpts = ai.ModelOptions{
		Supports:     &VeoSupports,
		Stage:        ai.ModelStageUnstable,
		ConfigSchema: configToMap(genai.GenerateVideosConfig{}),
	}

	defaultEmbedOpts = ai.EmbedderOptions{
		Supports:   &ai.EmbedderSupports{Input: []string{"text"}},
		Dimensions: 768,
	}
)

const (
	gemini20Flash     = "gemini-2.0-flash"
	gemini20FlashExp  = "gemini-2.0-flash-exp"
	gemini20FlashLite = "gemini-2.0-flash-lite"

	gemini25Flash     = "gemini-2.5-flash"
	gemini25FlashLite = "gemini-2.5-flash-lite"

	gemini25Pro = "gemini-2.5-pro"

	imagen3Generate001     = "imagen-3.0-generate-001"
	imagen3FastGenerate001 = "imagen-3.0-fast-generate-001"

	veo20Generate001         = "veo-2.0-generate-001"
	veo30Generate001         = "veo-3.0-generate-001"
	veo30FastGenerate001     = "veo-3.0-fast-generate-001"
	veo31Generate001         = "veo-3.1-generate-001"
	veo31FastGenerate001     = "veo-3.1-fast-generate-001"
	veo31GeneratePreview     = "veo-3.1-generate-preview"
	veo31FastGeneratePreview = "veo-3.1-fast-generate-preview"

	embedding001                      = "embedding-001"
	textembeddinggecko003             = "textembedding-gecko@003"
	textembeddinggecko002             = "textembedding-gecko@002"
	textembeddinggecko001             = "textembedding-gecko@001"
	textembeddinggeckomultilingual001 = "textembedding-gecko-multilingual@001"
	textmultilingualembedding002      = "text-multilingual-embedding-002"
	multimodalembedding               = "multimodalembedding"
)

var (
	// eventually, Vertex AI and Google AI models will match, in the meantime,
	// keep them sepparated
	vertexAIModels = []string{
		gemini20Flash,
		gemini20FlashLite,
		gemini25Flash,
		gemini25FlashLite,
		gemini25Pro,

		imagen3Generate001,
		imagen3FastGenerate001,

		veo20Generate001,
		veo30Generate001,
		veo30FastGenerate001,
		veo31Generate001,
		veo31FastGenerate001,
	}

	googleAIModels = []string{
		gemini20Flash,
		gemini20FlashExp,
		gemini25Flash,
		gemini25FlashLite,
		gemini25Pro,

		veo20Generate001,
		veo30Generate001,
		veo30FastGenerate001,
		veo31GeneratePreview,
		veo31FastGeneratePreview,
	}

	supportedGeminiModels = map[string]ai.ModelOptions{
		gemini20Flash: {
			Label: "Gemini 2.0 Flash",
			Versions: []string{
				"gemini-2.0-flash-001",
			},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini20FlashLite: {
			Label: "Gemini 2.0 Flash Lite",
			Versions: []string{
				"gemini-2.0-flash-lite-001",
			},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini25Flash: {
			Label:    "Gemini 2.5 Flash",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini25FlashLite: {
			Label:    "Gemini 2.5 Flash Lite",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini25Pro: {
			Label:    "Gemini 2.5 Pro",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
	}

	supportedImagenModels = map[string]ai.ModelOptions{
		imagen3Generate001: {
			Label:    "Imagen 3 Generate 001",
			Versions: []string{},
			Supports: &Media,
			Stage:    ai.ModelStageStable,
		},
		imagen3FastGenerate001: {
			Label:    "Imagen 3 Fast Generate 001",
			Versions: []string{},
			Supports: &Media,
			Stage:    ai.ModelStageStable,
		},
	}

	supportedVideoModels = map[string]ai.ModelOptions{
		veo20Generate001: {
			Label:    "Veo 2.0 Generate 001",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
		veo30Generate001: {
			Label:    "Veo 3.0 Generate 001",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
		veo30FastGenerate001: {
			Label:    "Veo 3.0 Fast Generate 001",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
		veo31Generate001: {
			Label:    "Veo 3.1 Generate 001",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
		veo31FastGenerate001: {
			Label:    "Veo 3.1 Fast Generate 001",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
		veo31GeneratePreview: {
			Label:    "Veo 3.1 Generate Preview",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageUnstable,
		},
		veo31FastGeneratePreview: {
			Label:    "Veo 3.1 Fast Generate Preview",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageUnstable,
		},
	}

	embedderConfig = map[string]ai.EmbedderOptions{
		embedding001: {
			Dimensions: 768,
			Label:      "Google Gen AI - Text Embedding Gecko (Legacy)",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		textembeddinggecko003: {
			Dimensions: 768,
			Label:      "Google Gen AI - Text Embedding Gecko 003",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		textembeddinggecko002: {
			Dimensions: 768,
			Label:      "Vertex AI - Text Embedding Gecko 002",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		textembeddinggecko001: {
			Dimensions: 768,
			Label:      "Vertex AI - Text Embedding Gecko 001",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		textembeddinggeckomultilingual001: {
			Dimensions: 768,
			Label:      "Vertex AI - Text Embedding Gecko Multilingual 001",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		textmultilingualembedding002: {
			Dimensions: 768,
			Label:      "Vertex AI - Text Multilingual Embedding 001",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		multimodalembedding: {
			Dimensions: 768,
			Label:      "Google Gen AI - Text Embedding Gecko (Legacy)",
			Supports: &ai.EmbedderSupports{
				Input: []string{
					"text",
					"image",
					"video",
				},
			},
		},
	}
)

// GetModelOptions returns ModelOptions for a model name with provider-prefixed label.
func GetModelOptions(name, provider string) ai.ModelOptions {
	mt := ClassifyModel(name)
	var opts ai.ModelOptions
	var ok bool

	switch mt {
	case ModelTypeGemini:
		opts, ok = supportedGeminiModels[name]
		if !ok {
			opts = defaultGeminiOpts
		}
	case ModelTypeImagen:
		opts, ok = supportedImagenModels[name]
		if !ok {
			opts = defaultImagenOpts
		}
	case ModelTypeVeo:
		opts, ok = supportedVideoModels[name]
		if !ok {
			opts = defaultVeoOpts
		}
	default:
		opts = defaultGeminiOpts
	}

	if opts.ConfigSchema == nil {
		if cfg := mt.DefaultConfig(); cfg != nil {
			opts.ConfigSchema = configToMap(cfg)
		}
	}

	// Set label with provider prefix
	prefix := googleAILabelPrefix
	if provider == vertexAIProvider {
		prefix = vertexAILabelPrefix
	}
	if opts.Label == "" {
		opts.Label = name
	}
	opts.Label = fmt.Sprintf("%s - %s", prefix, opts.Label)

	return opts
}

// GetEmbedderOptions returns EmbedderOptions for an embedder name with provider-prefixed label.
func GetEmbedderOptions(name, provider string) ai.EmbedderOptions {
	opts, ok := embedderConfig[name]
	if !ok {
		opts = defaultEmbedOpts
	}

	prefix := googleAILabelPrefix
	if provider == vertexAIProvider {
		prefix = vertexAILabelPrefix
	}
	if opts.Label == "" {
		opts.Label = name
	}
	opts.Label = fmt.Sprintf("%s - %s", prefix, opts.Label)

	return opts
}

// listModels returns a map of supported models and their capabilities
// based on the detected backend.
func listModels(provider string) (map[string]ai.ModelOptions, error) {
	var names []string

	switch provider {
	case googleAIProvider:
		names = googleAIModels
	case vertexAIProvider:
		names = vertexAIModels
	default:
		return nil, fmt.Errorf("unknown provider detected %s", provider)
	}

	models := make(map[string]ai.ModelOptions, len(names))
	for _, n := range names {
		opts := GetModelOptions(n, provider)
		models[n] = opts
	}

	return models, nil
}

// genaiModels collects all the available models in go-genai SDK
type genaiModels struct {
	gemini    []string
	imagen    []string
	embedders []string
	veo       []string
}

// listGenaiModels returns a list of supported models and embedders from the
// Go Genai SDK, categorized by model type.
func listGenaiModels(ctx context.Context, client *genai.Client) (genaiModels, error) {
	models := genaiModels{}

	for item, err := range client.Models.All(ctx) {
		if err != nil {
			return genaiModels{}, fmt.Errorf("failed to list models: %w", err)
		}

		name := strings.TrimPrefix(item.Name, "publishers/google/")
		name = strings.TrimPrefix(name, "models/")

		// The Vertex AI backend does not populate SupportedActions,
		// so we fall back to name-based categorization.
		if slices.Contains(item.SupportedActions, "embedContent") || strings.Contains(name, "embed") {
			models.embedders = append(models.embedders, name)
			continue
		}

		if strings.Contains(name, "imagen") {
			models.imagen = append(models.imagen, name)
			continue
		}

		if strings.Contains(name, "veo") {
			models.veo = append(models.veo, name)
			continue
		}

		// Only include models with known generative prefixes.
		if strings.HasPrefix(name, "gemini") || strings.HasPrefix(name, "gemma") {
			models.gemini = append(models.gemini, name)
		}
	}

	return models, nil
}
