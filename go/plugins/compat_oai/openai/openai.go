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
	"fmt"
	"maps"
	"os"
	"strings"

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

	supportedSpeechModels = map[string]ai.ModelOptions{
		openaiGo.SpeechModelTTS1: {
			Label:        "OpenAI TTS 1",
			Supports:     &compat_oai.SpeechSupports,
			ConfigSchema: openAISpeechConfigSchema(openaiGo.SpeechModelTTS1),
		},
		openaiGo.SpeechModelTTS1HD: {
			Label:        "OpenAI TTS 1 HD",
			Supports:     &compat_oai.SpeechSupports,
			ConfigSchema: openAISpeechConfigSchema(openaiGo.SpeechModelTTS1HD),
		},
		openaiGo.SpeechModelGPT4oMiniTTS: {
			Label:        "OpenAI GPT-4o Mini TTS",
			Supports:     &compat_oai.SpeechSupports,
			ConfigSchema: openAISpeechConfigSchema(openaiGo.SpeechModelGPT4oMiniTTS),
		},
	}

	supportedTranscriptionModels = map[string]ai.ModelOptions{
		openaiGo.AudioModelGPT4oTranscribe: {
			Label: "OpenAI GPT-4o Transcribe",
		},
		openaiGo.AudioModelGPT4oMiniTranscribe: {
			Label: "OpenAI GPT-4o Mini Transcribe",
		},
	}

	supportedWhisperModels = map[string]ai.ModelOptions{
		openaiGo.AudioModelWhisper1: {
			Label:        "OpenAI Whisper 1",
			Supports:     &compat_oai.TranscriptionSupports,
			ConfigSchema: core.InferSchemaMap(compat_oai.WhisperConfig{}),
		},
	}
)

func openAISpeechConfigSchema(model string) map[string]any {
	schema := core.InferSchemaMap(compat_oai.SpeechConfig{})
	if isGPT4oMiniTTSModel(model) {
		return schemaWithoutProperty(schema, "speed")
	}
	return schemaWithoutProperty(schema, "instructions")
}

func isGPT4oMiniTTSModel(model string) bool {
	base := string(openaiGo.SpeechModelGPT4oMiniTTS)
	return model == base || strings.HasPrefix(model, base+"-")
}

func schemaWithoutProperty(schema map[string]any, property string) map[string]any {
	result := maps.Clone(schema)
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		return result
	}
	properties = maps.Clone(properties)
	delete(properties, property)
	result["properties"] = properties
	return result
}

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
	for model, opts := range supportedSpeechModels {
		actions = append(actions, o.DefineSpeechModel(model, opts).(api.Action))
	}
	for model, opts := range supportedTranscriptionModels {
		actions = append(actions, o.DefineTranscriptionModel(model, opts).(api.Action))
	}
	for model, opts := range supportedWhisperModels {
		actions = append(actions, o.DefineWhisperModel(model, opts).(api.Action))
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

// DefineSpeechModel defines an OpenAI text-to-speech model.
func (o *OpenAI) DefineSpeechModel(id string, opts ai.ModelOptions) ai.Model {
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = openAISpeechConfigSchema(id)
	}
	return o.openAICompatible.DefineSpeechModel(provider, id, opts)
}

// DefineTranscriptionModel defines an OpenAI speech-to-text model.
func (o *OpenAI) DefineTranscriptionModel(id string, opts ai.ModelOptions) ai.Model {
	return o.openAICompatible.DefineTranscriptionModel(provider, id, opts)
}

// DefineWhisperModel defines an OpenAI Whisper transcription and translation model.
func (o *OpenAI) DefineWhisperModel(id string, opts ai.ModelOptions) ai.Model {
	return o.openAICompatible.DefineWhisperModel(provider, id, opts)
}

func (o *OpenAI) DefineEmbedder(id string, opts *ai.EmbedderOptions) ai.Embedder {
	return o.openAICompatible.DefineEmbedder(provider, id, opts)
}

func (o *OpenAI) Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return o.openAICompatible.Embedder(g, api.NewName(provider, name))
}

func (o *OpenAI) ListActions(ctx context.Context) []api.ActionDesc {
	actions := o.openAICompatible.ListActions(ctx)
	for i := range actions {
		name := strings.TrimPrefix(actions[i].Name, provider+"/")
		opts, kind, ok := audioModelOptions(name)
		if !ok {
			continue
		}
		var model ai.Model
		switch kind {
		case "speech":
			model = o.DefineSpeechModel(name, opts)
		case "whisper":
			model = o.DefineWhisperModel(name, opts)
		default:
			model = o.DefineTranscriptionModel(name, opts)
		}
		actions[i] = model.(api.Action).Desc()
	}
	return actions
}

func (o *OpenAI) ResolveAction(atype api.ActionType, name string) api.Action {
	if atype == api.ActionTypeModel {
		if opts, kind, ok := audioModelOptions(name); ok {
			switch kind {
			case "speech":
				return o.DefineSpeechModel(name, opts).(api.Action)
			case "whisper":
				return o.DefineWhisperModel(name, opts).(api.Action)
			default:
				return o.DefineTranscriptionModel(name, opts).(api.Action)
			}
		}
	}
	return o.openAICompatible.ResolveAction(atype, name)
}

func audioModelOptions(name string) (ai.ModelOptions, string, bool) {
	if opts, ok := supportedSpeechModels[name]; ok {
		return opts, "speech", true
	}
	if opts, ok := supportedTranscriptionModels[name]; ok {
		return opts, "transcription", true
	}
	if opts, ok := supportedWhisperModels[name]; ok {
		return opts, "whisper", true
	}
	if strings.Contains(name, "tts") {
		return ai.ModelOptions{
			Label:        fmt.Sprintf("OpenAI %s", name),
			Supports:     &compat_oai.SpeechSupports,
			ConfigSchema: openAISpeechConfigSchema(name),
		}, "speech", true
	}
	if strings.Contains(name, "whisper") {
		return ai.ModelOptions{
			Label:        fmt.Sprintf("OpenAI %s", name),
			Supports:     &compat_oai.TranscriptionSupports,
			ConfigSchema: core.InferSchemaMap(compat_oai.WhisperConfig{}),
		}, "whisper", true
	}
	if strings.Contains(name, "transcribe") {
		return ai.ModelOptions{
			Label: fmt.Sprintf("OpenAI %s", name),
		}, "transcription", true
	}
	return ai.ModelOptions{}, "", false
}
