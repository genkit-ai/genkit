// Copyright 2026 Google LLC
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

// Package zai provides a Genkit plugin for Z.ai's GLM models.
package zai

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

const (
	provider       = "zai"
	defaultBaseURL = "https://api.z.ai/api/paas/v4"

	ModelGLM51           = "glm-5.1"
	ModelGLM5Turbo       = "glm-5-turbo"
	ModelGLM5            = "glm-5"
	ModelGLM47           = "glm-4.7"
	ModelGLM47Flash      = "glm-4.7-flash"
	ModelGLM47FlashX     = "glm-4.7-flashx"
	ModelGLM46           = "glm-4.6"
	ModelGLM45           = "glm-4.5"
	ModelGLM45Air        = "glm-4.5-air"
	ModelGLM45X          = "glm-4.5-x"
	ModelGLM45AirX       = "glm-4.5-airx"
	ModelGLM45Flash      = "glm-4.5-flash"
	ModelGLM432B0414128K = "glm-4-32b-0414-128k"
	ModelGLM5VTurbo      = "glm-5v-turbo"
	ModelGLM46V          = "glm-4.6v"
	ModelGLM46VFlash     = "glm-4.6v-flash"
	ModelGLM46VFlashX    = "glm-4.6v-flashx"
	ModelGLM45V          = "glm-4.5v"
)

var supportedModels = map[string]ai.ModelOptions{
	ModelGLM51:           modelOptions(ModelGLM51, "Z.ai GLM 5.1", false),
	ModelGLM5Turbo:       modelOptions(ModelGLM5Turbo, "Z.ai GLM 5 Turbo", false),
	ModelGLM5:            modelOptions(ModelGLM5, "Z.ai GLM 5", false),
	ModelGLM47:           modelOptions(ModelGLM47, "Z.ai GLM 4.7", false),
	ModelGLM47Flash:      modelOptions(ModelGLM47Flash, "Z.ai GLM 4.7 Flash", false),
	ModelGLM47FlashX:     modelOptions(ModelGLM47FlashX, "Z.ai GLM 4.7 FlashX", false),
	ModelGLM46:           modelOptions(ModelGLM46, "Z.ai GLM 4.6", false),
	ModelGLM45:           modelOptions(ModelGLM45, "Z.ai GLM 4.5", false),
	ModelGLM45Air:        modelOptions(ModelGLM45Air, "Z.ai GLM 4.5 Air", false),
	ModelGLM45X:          modelOptions(ModelGLM45X, "Z.ai GLM 4.5 X", false),
	ModelGLM45AirX:       modelOptions(ModelGLM45AirX, "Z.ai GLM 4.5 AirX", false),
	ModelGLM45Flash:      modelOptions(ModelGLM45Flash, "Z.ai GLM 4.5 Flash", false),
	ModelGLM432B0414128K: modelOptions(ModelGLM432B0414128K, "Z.ai GLM 4 32B 128K", false),
	ModelGLM5VTurbo:      modelOptions(ModelGLM5VTurbo, "Z.ai GLM 5V Turbo", true),
	ModelGLM46V:          modelOptions(ModelGLM46V, "Z.ai GLM 4.6V", true),
	ModelGLM46VFlash:     modelOptions(ModelGLM46VFlash, "Z.ai GLM 4.6V Flash", true),
	ModelGLM46VFlashX:    modelOptions(ModelGLM46VFlashX, "Z.ai GLM 4.6V FlashX", true),
	ModelGLM45V:          modelOptions(ModelGLM45V, "Z.ai GLM 4.5V", true),
}

func modelOptions(id, label string, media bool) ai.ModelOptions {
	return ai.ModelOptions{
		Label: label,
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      media,
			ToolChoice: false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{id},
	}
}

// ZAI configures the Z.ai GLM plugin.
type ZAI struct {
	// APIKey is the Z.ai API key. If empty, ZAI_API_KEY is consulted.
	APIKey string
	// BaseURL overrides the Z.ai API endpoint. If empty, ZAI_BASE_URL and then
	// the default international endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (z *ZAI) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (z *ZAI) Init(ctx context.Context) []api.Action {
	baseURL := z.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("ZAI_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := z.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ZAI_API_KEY")
	}
	if apiKey == "" {
		panic("zai plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, z.Opts...)

	z.openAICompatible.Provider = provider
	z.openAICompatible.Opts = opts
	actions := z.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, z.DefineModel(model, modelOpts).(api.Action))
	}
	return actions
}

// Model returns a registered Z.ai model.
func (z *ZAI) Model(g *genkit.Genkit, id string) ai.Model {
	return z.openAICompatible.Model(g, api.NewName(provider, id))
}

// DefineModel registers a Z.ai model, including models not in the built-in list.
func (z *ZAI) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return z.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions lists models exposed by the configured Z.ai endpoint.
func (z *ZAI) ListActions(ctx context.Context) []api.ActionDesc {
	return z.openAICompatible.ListActions(ctx)
}

// ResolveAction dynamically registers a model exposed by the Z.ai endpoint.
func (z *ZAI) ResolveAction(atype api.ActionType, name string) api.Action {
	return z.openAICompatible.ResolveAction(atype, name)
}
