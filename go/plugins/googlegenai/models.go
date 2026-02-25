// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"fmt"
	"log"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

const (
	gemini15Flash   = "gemini-1.5-flash"
	gemini15Pro     = "gemini-1.5-pro"
	gemini15Flash8b = "gemini-1.5-flash-8b"

	gemini20Flash                = "gemini-2.0-flash"
	gemini20FlashExp             = "gemini-2.0-flash-exp"
	gemini20FlashLite            = "gemini-2.0-flash-lite"
	gemini20FlashLitePrev        = "gemini-2.0-flash-lite-preview"
	gemini20ProExp0205           = "gemini-2.0-pro-exp-02-05"
	gemini20FlashThinkingExp0121 = "gemini-2.0-flash-thinking-exp-01-21"
	gemini20FlashPrevImageGen    = "gemini-2.0-flash-preview-image-generation"

	gemini25Flash             = "gemini-2.5-flash"
	gemini25FlashLite         = "gemini-2.5-flash-lite"
	gemini25FlashLitePrev0617 = "gemini-2.5-flash-lite-preview-06-17"

	gemini25Pro            = "gemini-2.5-pro"
	gemini25ProExp0325     = "gemini-2.5-pro-exp-03-25"
	gemini25ProPreview0325 = "gemini-2.5-pro-preview-03-25"
	gemini25ProPreview0506 = "gemini-2.5-pro-preview-05-06"

	imagen3Generate001     = "imagen-3.0-generate-001"
	imagen3Generate002     = "imagen-3.0-generate-002"
	imagen3FastGenerate001 = "imagen-3.0-fast-generate-001"

	textembedding004                  = "text-embedding-004"
	embedding001                      = "embedding-001"
	textembeddinggecko003             = "textembedding-gecko@003"
	textembeddinggecko002             = "textembedding-gecko@002"
	textembeddinggecko001             = "textembedding-gecko@001"
	textembeddinggeckomultilingual001 = "textembedding-gecko-multilingual@001"
	textmultilingualembedding002      = "text-multilingual-embedding-002"
	multimodalembedding               = "multimodalembedding"
	veo20Generate001                  = "veo-2.0-generate-001"
	veo30Generate001                  = "veo-3.0-generate-001"
	veo30FastGenerate001              = "veo-3.0-fast-generate-001"
)

var (
	// eventually, Vertex AI and Google AI models will match, in the meantime,
	// keep them sepparated
	vertexAIModels = []string{
		gemini15Flash,
		gemini15Pro,
		gemini20Flash,
		gemini20FlashLite,
		gemini20FlashLitePrev,
		gemini20ProExp0205,
		gemini20FlashThinkingExp0121,
		gemini20FlashPrevImageGen,
		gemini25Flash,
		gemini25FlashLite,
		gemini25Pro,
		gemini25FlashLitePrev0617,
		gemini25ProExp0325,
		gemini25ProPreview0325,
		gemini25ProPreview0506,

		imagen3Generate001,
		imagen3Generate002,
		imagen3FastGenerate001,

		veo20Generate001,
		veo30Generate001,
		veo30FastGenerate001,
	}

	googleAIModels = []string{
		gemini15Flash,
		gemini15Pro,
		gemini15Flash8b,
		gemini20Flash,
		gemini20FlashExp,
		gemini20FlashLitePrev,
		gemini20ProExp0205,
		gemini20FlashThinkingExp0121,
		gemini20FlashPrevImageGen,
		gemini25Flash,
		gemini25FlashLite,
		gemini25Pro,
		gemini25FlashLitePrev0617,
		gemini25ProExp0325,
		gemini25ProPreview0325,
		gemini25ProPreview0506,

		imagen3Generate002,

		veo20Generate001,
		veo30Generate001,
		veo30FastGenerate001,
	}

	supportedGeminiModels = map[string]ai.ModelOptions{
		gemini15Flash: {
			Label: "Gemini 1.5 Flash",
			Versions: []string{
				"gemini-1.5-flash-latest",
				"gemini-1.5-flash-001",
				"gemini-1.5-flash-002",
			},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini15Pro: {
			Label: "Gemini 1.5 Pro",
			Versions: []string{
				"gemini-1.5-pro-latest",
				"gemini-1.5-pro-001",
				"gemini-1.5-pro-002",
			},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini15Flash8b: {
			Label: "Gemini 1.5 Flash 8B",
			Versions: []string{
				"gemini-1.5-flash-8b-latest",
				"gemini-1.5-flash-8b-001",
			},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini20Flash: {
			Label: "Gemini 2.0 Flash",
			Versions: []string{
				"gemini-2.0-flash-001",
			},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini20FlashExp: {
			Label:    "Gemini 2.0 Flash Exp",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini20FlashLite: {
			Label: "Gemini 2.0 Flash Lite",
			Versions: []string{
				"gemini-2.0-flash-lite-001",
			},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini20FlashLitePrev: {
			Label:    "Gemini 2.0 Flash Lite Preview 02-05",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini20ProExp0205: {
			Label:    "Gemini 2.0 Pro Exp 02-05",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini20FlashThinkingExp0121: {
			Label:    "Gemini 2.0 Flash Thinking Exp 01-21",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini20FlashPrevImageGen: {
			Label:    "Gemini 2.0 Flash Preview Image Generation",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini25Flash: {
			Label:    "Gemini 2.5 Flash",
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
		gemini25ProExp0325: {
			Label:    "Gemini 2.5 Pro Exp 03-25",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini25ProPreview0325: {
			Label:    "Gemini 2.5 Pro Preview 03-25",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini25ProPreview0506: {
			Label:    "Gemini 2.5 Pro Preview 05-06",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
		gemini25FlashLite: {
			Label:    "Gemini 2.5 Flash Lite",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageStable,
		},
		gemini25FlashLitePrev0617: {
			Label:    "Gemini 2.5 Flash Lite Preview 06-17",
			Versions: []string{},
			Supports: &Multimodal,
			Stage:    ai.ModelStageUnstable,
		},
	}

	supportedImagenModels = map[string]ai.ModelOptions{
		imagen3Generate001: {
			Label:    "Imagen 3 Generate 001",
			Versions: []string{},
			Supports: &Media,
			Stage:    ai.ModelStageStable,
		},
		imagen3Generate002: {
			Label:    "Imagen 3 Generate 002",
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
			Label:    "Google AI - Veo 2.0 Generate 001",
			Versions: []string{},
			Supports: &ai.ModelSupports{
				Media:       true,
				Multiturn:   false,
				Tools:       false,
				SystemRole:  false,
				Output:      []string{"media"},
				LongRunning: true,
			},
			Stage: ai.ModelStageStable,
		},
		veo30Generate001: {
			Label:    "Google AI - Veo 3.0 Generate 001",
			Versions: []string{},
			Supports: &ai.ModelSupports{
				Media:       true,
				Multiturn:   false,
				Tools:       false,
				SystemRole:  false,
				Output:      []string{"media"},
				LongRunning: true,
			},
			Stage: ai.ModelStageStable,
		},
		veo30FastGenerate001: {
			Label:    "Google AI - Veo 3.0 Fast Generate 001",
			Versions: []string{},
			Supports: &ai.ModelSupports{
				Media:       true,
				Multiturn:   false,
				Tools:       false,
				SystemRole:  false,
				Output:      []string{"media"},
				LongRunning: true,
			},
			Stage: ai.ModelStageStable,
		},
	}

	googleAIEmbedderConfig = map[string]ai.EmbedderOptions{
		textembedding004: {
			Dimensions: 768,
			Label:      "Google Gen AI - Text Embedding 001",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
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

// listModels returns a map of supported models and their capabilities
// based on the detected backend
func listModels(provider string) (map[string]ai.ModelOptions, error) {
	var names []string
	var prefix string

	switch provider {
	case googleAIProvider:
		names = googleAIModels
		prefix = googleAILabelPrefix
	case vertexAIProvider:
		names = vertexAIModels
		prefix = vertexAILabelPrefix
	default:
		return nil, fmt.Errorf("unknown provider detected %s", provider)
	}

	models := make(map[string]ai.ModelOptions, 0)
	for _, n := range names {
		var m ai.ModelOptions
		var ok bool
		if strings.HasPrefix(n, "image") {
			m, ok = supportedImagenModels[n]
		} else if strings.HasPrefix(n, "veo") {
			m, ok = supportedVideoModels[n]
		} else {
			m, ok = supportedGeminiModels[n]
		}
		if !ok {
			return nil, fmt.Errorf("model %s not found for provider %s", n, provider)
		}
		models[n] = ai.ModelOptions{
			Label:        prefix + " - " + m.Label,
			Versions:     m.Versions,
			Supports:     m.Supports,
			ConfigSchema: m.ConfigSchema,
			Stage:        m.Stage,
		}
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
// Go Genai SDK
func listGenaiModels(ctx context.Context, client *genai.Client) (genaiModels, error) {
	models := genaiModels{}
	allowedModels := []string{"gemini", "gemma"}

	for item, err := range client.Models.All(ctx) {
		var name string
		var description string
		if err != nil {
			log.Fatal(err)
		}
		switch {
		case strings.HasPrefix(item.Name, "publishers/google/models/"):
			name = strings.TrimPrefix(item.Name, "publishers/google/models/")
		case strings.HasPrefix(item.Name, "models/"):
			name = strings.TrimPrefix(item.Name, "models/")
		default:
			continue
		}
		description = strings.ToLower(item.Description)
		if strings.Contains(description, "deprecated") {
			continue
		}
		// The Vertex AI backend does not populate SupportedActions,
		// so fall back to name-based categorization when it's empty.
		if len(item.SupportedActions) > 0 {
			if slices.Contains(item.SupportedActions, "embedContent") {
				models.embedders = append(models.embedders, name)
				continue
			}
			if slices.Contains(item.SupportedActions, "predict") && strings.Contains(name, "imagen") {
				models.imagen = append(models.imagen, name)
				continue
			}
			if slices.Contains(item.SupportedActions, "generateContent") {
				found := slices.ContainsFunc(allowedModels, func(s string) bool {
					return strings.Contains(name, s)
				})
				if found {
					models.gemini = append(models.gemini, name)
				}
			}
		} else {
			switch {
			case strings.Contains(name, "embedding") || strings.Contains(name, "embed"):
				models.embedders = append(models.embedders, name)
			case strings.Contains(name, "imagen"):
				models.imagen = append(models.imagen, name)
			default:
				found := slices.ContainsFunc(allowedModels, func(s string) bool {
					return strings.Contains(name, s)
				})
				if found {
					models.gemini = append(models.gemini, name)
				}
			}
		}
	}

	return models, nil
}
