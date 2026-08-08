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

// Package deepseek provides a Genkit plugin for DeepSeek models.
package deepseek

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

// chatCompletionConfig describes the schema accepted by DeepSeek chat
// completions.
type chatCompletionConfig struct {
	ai.GenerationCommonConfig
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty" jsonschema:"minimum=-2,maximum=2"`
	LogProbs         *bool    `json:"logProbs,omitempty"`
	PresencePenalty  *float64 `json:"presencePenalty,omitempty" jsonschema:"minimum=-2,maximum=2"`
	TopLogProbs      *int     `json:"topLogProbs,omitempty" jsonschema:"minimum=0,maximum=20"`
}

const (
	provider       = "deepseek"
	defaultBaseURL = "https://api.deepseek.com"

	// ModelDeepSeekChat is DeepSeek's general-purpose chat model.
	ModelDeepSeekChat = "deepseek-chat"
	// ModelDeepSeekReasoner is DeepSeek's reasoning model.
	ModelDeepSeekReasoner = "deepseek-reasoner"
)

var supportedModels = map[string]ai.ModelOptions{
	ModelDeepSeekChat:     modelOptions(ModelDeepSeekChat, "DeepSeek Chat"),
	ModelDeepSeekReasoner: modelOptions(ModelDeepSeekReasoner, "DeepSeek Reasoner"),
}

func modelOptions(id, label string) ai.ModelOptions {
	return ai.ModelOptions{
		Label:        label,
		ConfigSchema: core.InferSchemaMap(chatCompletionConfig{}),
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{id},
	}
}

// DeepSeek configures the DeepSeek plugin.
type DeepSeek struct {
	// APIKey is the DeepSeek API key. If empty, DEEPSEEK_API_KEY is consulted.
	APIKey string
	// BaseURL overrides the DeepSeek API endpoint. If empty,
	// DEEPSEEK_BASE_URL and then the default endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (d *DeepSeek) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (d *DeepSeek) Init(ctx context.Context) []api.Action {
	baseURL := d.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("DEEPSEEK_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := d.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	if apiKey == "" {
		panic("deepseek plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, d.Opts...)

	d.openAICompatible.Provider = provider
	d.openAICompatible.ConfigAliases = map[string]string{
		"maxOutputTokens": "max_tokens",
	}
	d.openAICompatible.Opts = opts
	actions := d.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, d.DefineModel(model, modelOpts).(api.Action))
	}
	return actions
}

// Model returns a registered DeepSeek model.
func (d *DeepSeek) Model(g *genkit.Genkit, id string) ai.Model {
	return d.openAICompatible.Model(g, api.NewName(provider, id))
}

// DefineModel registers a DeepSeek model, including models not in the built-in list.
func (d *DeepSeek) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return d.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions lists models exposed by the configured DeepSeek endpoint.
func (d *DeepSeek) ListActions(ctx context.Context) []api.ActionDesc {
	actions := d.openAICompatible.ListActions(ctx)
	for i := range actions {
		name := strings.TrimPrefix(actions[i].Name, provider+"/")
		actions[i] = d.DefineModel(name, optionsForModel(name)).(api.Action).Desc()
	}
	return actions
}

// ResolveAction dynamically registers a model exposed by the DeepSeek endpoint.
func (d *DeepSeek) ResolveAction(atype api.ActionType, name string) api.Action {
	if atype != api.ActionTypeModel {
		return nil
	}
	return d.DefineModel(name, optionsForModel(name)).(api.Action)
}

func optionsForModel(name string) ai.ModelOptions {
	if opts, ok := supportedModels[name]; ok {
		return opts
	}
	return modelOptions(name, fmt.Sprintf("DeepSeek - %s", name))
}
