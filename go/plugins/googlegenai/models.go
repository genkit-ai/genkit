// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal"
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
		Constrained: ai.ConstrainedSupportAll,
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

	// TTSSupports describes model capabilities for text-to-speech models
	// (gemini-*-tts). They emit audio and, unlike conversational Gemini models,
	// do not support tools, multi-turn history, or system roles. Output is
	// "media" to match the convention used by the other media producers
	// (Imagen, Veo) rather than a TTS-only token.
	TTSSupports = ai.ModelSupports{
		Multiturn:  false,
		Media:      false,
		Tools:      false,
		ToolChoice: false,
		SystemRole: false,
		Output:     []string{"media"},
	}
)

// Config schemas advertised for each generation modality. Reflecting the SDK
// config structs is expensive and every model of a modality advertises the
// same read-only schema, so they are built once and shared.
var (
	geminiConfigSchema = configToMap(genai.GenerateContentConfig{})
	imagenConfigSchema = configToMap(genai.GenerateImagesConfig{})
	veoConfigSchema    = configToMap(genai.GenerateVideosConfig{})
)

// Default options for unknown models of each type. Every catalog entry is
// stable, but these are a guess at the capabilities of an ID the plugin does
// not know, so they stay unstable: the stage is the only thing telling a
// curated model apart from a typo the plugin served anyway.
var (
	defaultGeminiOpts = ai.ModelOptions{
		Supports:     &Multimodal,
		Stage:        ai.ModelStageUnstable,
		ConfigSchema: geminiConfigSchema,
	}

	defaultImagenOpts = ai.ModelOptions{
		Supports:     &Media,
		Stage:        ai.ModelStageUnstable,
		ConfigSchema: imagenConfigSchema,
	}

	defaultVeoOpts = ai.ModelOptions{
		Supports:     &VeoSupports,
		Stage:        ai.ModelStageUnstable,
		ConfigSchema: veoConfigSchema,
	}

	defaultEmbedOpts = ai.EmbedderOptions{
		Supports:   &ai.EmbedderSupports{Input: []string{"text"}},
		Dimensions: 768,
	}
)

const (
	gemini25Flash      = "gemini-2.5-flash"
	gemini25FlashLite  = "gemini-2.5-flash-lite"
	gemini25FlashImage = "gemini-2.5-flash-image"

	gemini25Pro = "gemini-2.5-pro"

	// Google AI names the omni model gemini-omni-flash; Vertex AI serves the
	// same model as gemini-omni-flash-preview.
	geminiOmniFlash        = "gemini-omni-flash"
	geminiOmniFlashPreview = "gemini-omni-flash-preview"

	gemini3FlashPreview    = "gemini-3-flash-preview"
	gemini37Flash          = "gemini-3.7-flash"
	gemini36Flash          = "gemini-3.6-flash"
	gemini35Flash          = "gemini-3.5-flash"
	gemini35FlashLite      = "gemini-3.5-flash-lite"
	gemini31ProPreview     = "gemini-3.1-pro-preview"
	gemini31FlashLite      = "gemini-3.1-flash-lite"
	gemini31FlashImage     = "gemini-3.1-flash-image"
	gemini31FlashLiteImage = "gemini-3.1-flash-lite-image"
	gemini3ProImage        = "gemini-3-pro-image"

	// Google AI TTS names. Vertex AI serves the 2.5 pair without the
	// "-preview-" infix, so the two backends need separate IDs.
	gemini25FlashPreviewTTS = "gemini-2.5-flash-preview-tts"
	gemini25ProPreviewTTS   = "gemini-2.5-pro-preview-tts"

	// Vertex AI TTS names.
	gemini25FlashTTS            = "gemini-2.5-flash-tts"
	gemini25ProTTS              = "gemini-2.5-pro-tts"
	gemini25FlashLitePreviewTTS = "gemini-2.5-flash-lite-preview-tts"

	// Served under the same ID by both backends.
	gemini31FlashTTSPreview = "gemini-3.1-flash-tts-preview"

	imagen40FastGenerate001  = "imagen-4.0-fast-generate-001"
	imagen40Generate001      = "imagen-4.0-generate-001"
	imagen40UltraGenerate001 = "imagen-4.0-ultra-generate-001"

	// Vertex AI serves Veo 3.1 as GA "-001" IDs; Google AI serves it as
	// "-preview". Each backend has retired the other's spelling.
	veo31Generate001         = "veo-3.1-generate-001"
	veo31FastGenerate001     = "veo-3.1-fast-generate-001"
	veo31LiteGenerate001     = "veo-3.1-lite-generate-001"
	veo31GeneratePreview     = "veo-3.1-generate-preview"
	veo31FastGeneratePreview = "veo-3.1-fast-generate-preview"
	veo31LiteGeneratePreview = "veo-3.1-lite-generate-preview"

	textembedding005             = "text-embedding-005"
	textembedding004             = "text-embedding-004"
	textmultilingualembedding002 = "text-multilingual-embedding-002"
	multimodalembedding          = "multimodalembedding"
	geminiEmbedding2             = "gemini-embedding-2"
	geminiEmbedding001           = "gemini-embedding-001"
)

var (
	// eventually, Vertex AI and Google AI models will match, in the meantime,
	// keep them sepparated
	vertexAIModels = []string{
		gemini25Flash,
		gemini25FlashLite,
		gemini25FlashImage,
		gemini25Pro,
		geminiOmniFlashPreview,
		gemini3FlashPreview,
		gemini37Flash,
		gemini36Flash,
		gemini35Flash,
		gemini35FlashLite,
		gemini31ProPreview,
		gemini31FlashLite,
		gemini31FlashImage,
		gemini31FlashLiteImage,
		gemini3ProImage,

		gemini25FlashTTS,
		gemini25ProTTS,
		gemini25FlashLitePreviewTTS,
		gemini31FlashTTSPreview,

		veo31Generate001,
		veo31FastGenerate001,
		veo31LiteGenerate001,
	}

	googleAIModels = []string{
		gemini25Flash,
		gemini25FlashLite,
		gemini25FlashImage,
		gemini25Pro,
		geminiOmniFlash,
		gemini3FlashPreview,
		gemini37Flash,
		gemini36Flash,
		gemini35Flash,
		gemini35FlashLite,
		gemini31ProPreview,
		gemini31FlashLite,
		gemini31FlashImage,
		gemini31FlashLiteImage,
		gemini3ProImage,

		// Imagen is retired on Vertex AI (June 30, 2026) and retires on Google
		// AI on August 17, 2026. Nano Banana (gemini-*-image, via
		// generateContent) is the replacement on both.
		imagen40FastGenerate001,
		imagen40Generate001,
		imagen40UltraGenerate001,

		gemini25FlashPreviewTTS,
		gemini25ProPreviewTTS,
		gemini31FlashTTSPreview,

		veo31GeneratePreview,
		veo31FastGeneratePreview,
		veo31LiteGeneratePreview,
	}

	supportedGeminiModels = map[string]ai.ModelOptions{
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
		gemini25FlashImage: {
			Label:    "Gemini 2.5 Flash Image",
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
		geminiOmniFlash: {
			Label:    "Gemini Omni Flash",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		geminiOmniFlashPreview: {
			Label:    "Gemini Omni Flash",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini3FlashPreview: {
			Label:    "Gemini 3 Flash Preview",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini37Flash: {
			Label:    "Gemini 3.7 Flash",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini36Flash: {
			Label:    "Gemini 3.6 Flash",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini35Flash: {
			Label:    "Gemini 3.5 Flash",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini35FlashLite: {
			Label:    "Gemini 3.5 Flash Lite",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini31ProPreview: {
			Label:    "Gemini 3.1 Pro Preview",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini31FlashLite: {
			Label:    "Gemini 3.1 Flash Lite",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini31FlashImage: {
			Label:    "Gemini 3.1 Flash Image",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini31FlashLiteImage: {
			Label:    "Gemini 3.1 Flash Lite Image",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini3ProImage: {
			Label:    "Gemini 3 Pro Image",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini25FlashPreviewTTS: {
			Label:    "Gemini 2.5 Flash Preview TTS",
			Versions: []string{},
			Supports: &TTSSupports,
			Stage:    ai.ModelStageStable,
		},
		gemini25ProPreviewTTS: {
			Label:    "Gemini 2.5 Pro Preview TTS",
			Versions: []string{},
			Supports: &TTSSupports,
			Stage:    ai.ModelStageStable,
		},
		gemini25FlashTTS: {
			Label:    "Gemini 2.5 Flash TTS",
			Versions: []string{},
			Supports: &TTSSupports,
			Stage:    ai.ModelStageStable,
		},
		gemini25ProTTS: {
			Label:    "Gemini 2.5 Pro TTS",
			Versions: []string{},
			Supports: &TTSSupports,
			Stage:    ai.ModelStageStable,
		},
		gemini25FlashLitePreviewTTS: {
			Label:    "Gemini 2.5 Flash Lite Preview TTS",
			Versions: []string{},
			Supports: &TTSSupports,
			Stage:    ai.ModelStageStable,
		},
		gemini31FlashTTSPreview: {
			Label:    "Gemini 3.1 Flash TTS Preview",
			Versions: []string{},
			Supports: &TTSSupports,
			Stage:    ai.ModelStageStable,
		},
	}

	supportedImagenModels = map[string]ai.ModelOptions{
		imagen40FastGenerate001: {
			Label:    "Imagen 4 Fast Generate 001",
			Versions: []string{},
			Supports: &Media,
			Stage:    ai.ModelStageStable,
		},
		imagen40Generate001: {
			Label:    "Imagen 4 Generate 001",
			Versions: []string{},
			Supports: &Media,
			Stage:    ai.ModelStageStable,
		},
		imagen40UltraGenerate001: {
			Label:    "Imagen 4 Ultra Generate 001",
			Versions: []string{},
			Supports: &Media,
			Stage:    ai.ModelStageStable,
		},
	}

	supportedVideoModels = map[string]ai.ModelOptions{
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
			Stage:    ai.ModelStageStable,
		},
		veo31FastGeneratePreview: {
			Label:    "Veo 3.1 Fast Generate Preview",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
		veo31LiteGenerate001: {
			Label:    "Veo 3.1 Lite Generate 001",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
		veo31LiteGeneratePreview: {
			Label:    "Veo 3.1 Lite Generate Preview",
			Versions: []string{},
			Supports: &VeoSupports,
			Stage:    ai.ModelStageStable,
		},
	}

	embedderConfig = map[string]ai.EmbedderOptions{
		textembedding005: {
			Dimensions: 768,
			Label:      "Text Embedding 005",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		textembedding004: {
			Dimensions: 768,
			Label:      "Text Embedding 004",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		textmultilingualembedding002: {
			Dimensions: 768,
			Label:      "Text Multilingual Embedding 002",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		multimodalembedding: {
			Dimensions: 768,
			Label:      "Multimodal Embedding",
			Supports: &ai.EmbedderSupports{
				Input: []string{
					"text",
					"image",
					"video",
				},
			},
		},
		geminiEmbedding001: {
			Dimensions: 3072,
			Label:      "Gemini Embedding 001",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		geminiEmbedding2: {
			Dimensions: 3072,
			Label:      "Gemini Embedding 2",
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
// The returned options share the package's schema maps and supports structs;
// they are read-only and must not be mutated.
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
		opts.ConfigSchema = mt.configSchema()
	}

	if opts.Label == "" {
		opts.Label = name
	}
	opts.Label = internal.ProviderLabel(displayName(provider), opts.Label)

	return opts
}

// GetEmbedderOptions returns EmbedderOptions for an embedder name with provider-prefixed label.
func GetEmbedderOptions(name, provider string) ai.EmbedderOptions {
	opts, ok := embedderConfig[name]
	if !ok {
		opts = defaultEmbedOpts
	}

	if opts.Label == "" {
		opts.Label = name
	}
	opts.Label = internal.ProviderLabel(displayName(provider), opts.Label)

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
