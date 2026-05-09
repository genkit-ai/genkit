// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package openai

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const provider = "openai"

type TextEmbeddingConfig struct {
	Dimensions     int                                       `json:"dimensions,omitempty"`
	EncodingFormat openaiGo.EmbeddingNewParamsEncodingFormat `json:"encodingFormat,omitempty"`
}

// EmbedderRef represents the main structure for an embedding model's definition.
type EmbedderRef struct {
	Name         string
	ConfigSchema TextEmbeddingConfig // Represents the schema, can be used for default config
	Label        string
	Supports     *ai.EmbedderSupports
	Dimensions   int
}

var (
	// Supported models: https://platform.openai.com/docs/models
	supportedModels = map[string]ai.ModelOptions{
		"gpt-4.1": {
			Label:    "OpenAI GPT-4.1",
			Supports: &compat_oai.Multimodal,
			Versions: []string{"gpt-4.1", "gpt-4.1-2025-04-14"},
		},
		"gpt-4.1-mini": {
			Label:    "OpenAI GPT-4.1-mini",
			Supports: &compat_oai.Multimodal,
			Versions: []string{"gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"},
		},
		"gpt-4.1-nano": {
			Label:    "OpenAI GPT-4.1-nano",
			Supports: &compat_oai.Multimodal,
			Versions: []string{"gpt-4.1-nano", "gpt-4.1-nano-2025-04-14"},
		},
		openaiGo.ChatModelO3Mini: {
			Label:    "OpenAI o3-mini",
			Supports: &compat_oai.BasicText,
			Versions: []string{"o3-mini", "o3-mini-2025-01-31"},
		},
		openaiGo.ChatModelO1: {
			Label:    "OpenAI o1",
			Supports: &compat_oai.BasicText,
			Versions: []string{"o1", "o1-2024-12-17"},
		},
		openaiGo.ChatModelO1Preview: {
			Label: "OpenAI o1-preview",
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      false,
				SystemRole: false,
				Media:      false,
			},
			Versions: []string{"o1-preview", "o1-preview-2024-09-12"},
		},
		openaiGo.ChatModelO1Mini: {
			Label: "OpenAI o1-mini",
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      false,
				SystemRole: false,
				Media:      false,
			},
			Versions: []string{"o1-mini", "o1-mini-2024-09-12"},
		},
		openaiGo.ChatModelGPT4o: {
			Label:    "OpenAI GPT-4o",
			Supports: &compat_oai.Multimodal,
			Versions: []string{"gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13"},
		},
		openaiGo.ChatModelGPT4oMini: {
			Label:    "OpenAI GPT-4o-mini",
			Supports: &compat_oai.Multimodal,
			Versions: []string{"gpt-4o-mini", "gpt-4o-mini-2024-07-18"},
		},
		openaiGo.ChatModelGPT4Turbo: {
			Label:    "OpenAI GPT-4-turbo",
			Supports: &compat_oai.Multimodal,
			Versions: []string{"gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview", "gpt-4-0125-preview"},
		},
		openaiGo.ChatModelGPT4: {
			Label: "OpenAI GPT-4",
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      false,
				SystemRole: true,
				Media:      false,
			},
			Versions: []string{"gpt-4", "gpt-4-0613", "gpt-4-0314"},
		},
		openaiGo.ChatModelGPT3_5Turbo: {
			Label: "OpenAI GPT-3.5-turbo",
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      false,
				SystemRole: true,
				Media:      false,
			},
			Versions: []string{"gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct"},
		},
	}

	supportedEmbeddingModels = map[string]EmbedderRef{
		openaiGo.EmbeddingModelTextEmbeddingAda002: {
			Name:         "text-embedding-ada-002",
			ConfigSchema: TextEmbeddingConfig{},
			Dimensions:   1536,
			Label:        "Open AI - Text Embedding ADA 002",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		openaiGo.EmbeddingModelTextEmbedding3Large: {
			Name:         "text-embedding-3-large",
			ConfigSchema: TextEmbeddingConfig{},
			Dimensions:   3072,
			Label:        "Open AI - Text Embedding 3 Large",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
		openaiGo.EmbeddingModelTextEmbedding3Small: {
			Name:         "text-embedding-3-small",
			ConfigSchema: TextEmbeddingConfig{}, // Represents the configurable options
			Dimensions:   1536,
			Label:        "Open AI - Text Embedding 3 Small",
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
		},
	}
)

type OpenAI struct {
	// APIKey is the API key for the OpenAI API. If empty, the values of the environment variable "OPENAI_API_KEY" will be consulted.
	// Request a key at https://platform.openai.com/api-keys
	APIKey string
	// Optional: Opts are additional options for the OpenAI client.
	// Can include other options like WithOrganization, WithBaseURL, etc.
	Opts []option.RequestOption

	openAICompatible *compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (o *OpenAI) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (o *OpenAI) Init(ctx context.Context) []api.Action {
	apiKey := o.APIKey

	// if api key is not set, get it from environment variable
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	if apiKey == "" {
		panic("openai plugin initialization failed: apiKey is required")
	}

	if o.openAICompatible == nil {
		o.openAICompatible = &compat_oai.OpenAICompatible{}
	}

	// set the options
	o.openAICompatible.Opts = []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if len(o.Opts) > 0 {
		o.openAICompatible.Opts = append(o.openAICompatible.Opts, o.Opts...)
	}

	o.openAICompatible.Provider = provider
	compatActions := o.openAICompatible.Init(ctx)

	var actions []api.Action
	actions = append(actions, compatActions...)

	// define default models
	for model, opts := range supportedModels {
		actions = append(actions, o.DefineModel(model, opts).(api.Action))
	}

	// define default embedders
	for _, embedder := range supportedEmbeddingModels {
		opts := &ai.EmbedderOptions{
			ConfigSchema: core.InferSchemaMap(embedder.ConfigSchema),
			Label:        embedder.Label,
			Supports:     embedder.Supports,
			Dimensions:   embedder.Dimensions,
		}
		actions = append(actions, o.DefineEmbedder(embedder.Name, opts).(api.Action))
	}

	return actions
}

func (o *OpenAI) Model(g *genkit.Genkit, name string) ai.Model {
	return o.openAICompatible.Model(g, api.NewName(provider, name))
}

func (o *OpenAI) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return o.openAICompatible.DefineModel(provider, id, opts)
}

func (o *OpenAI) DefineEmbedder(id string, opts *ai.EmbedderOptions) ai.Embedder {
	return o.openAICompatible.DefineEmbedder(provider, id, opts)
}

func (o *OpenAI) Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return o.openAICompatible.Embedder(g, api.NewName(provider, name))
}

func (o *OpenAI) ListActions(ctx context.Context) []api.ActionDesc {
	return o.openAICompatible.ListActions(ctx)
}

func (o *OpenAI) ResolveAction(atype api.ActionType, name string) api.Action {
	return o.openAICompatible.ResolveAction(atype, name)
}
