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

// Package xai provides a Genkit plugin for xAI's Grok language models.
package xai

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

// chatCompletionConfig describes the schema accepted by xAI chat
// completions. WebSearchOptions permits xAI-specific web search fields.
type chatCompletionConfig struct {
	ai.GenerationCommonConfig
	Deferred         *bool          `json:"deferred,omitempty"`
	FrequencyPenalty *float64       `json:"frequencyPenalty,omitempty" jsonschema:"minimum=-2,maximum=2"`
	LogProbs         *bool          `json:"logProbs,omitempty"`
	PresencePenalty  *float64       `json:"presencePenalty,omitempty" jsonschema:"minimum=-2,maximum=2"`
	ReasoningEffort  *string        `json:"reasoningEffort,omitempty" jsonschema:"enum=low,enum=medium,enum=high"`
	TopLogProbs      *int           `json:"topLogProbs,omitempty" jsonschema:"minimum=0,maximum=20"`
	WebSearchOptions map[string]any `json:"webSearchOptions,omitempty"`
}

const (
	provider       = "xai"
	defaultBaseURL = "https://api.x.ai/v1"

	// ModelGrok3 is xAI's Grok 3 language model.
	ModelGrok3 = "grok-3"
	// ModelGrok3Fast is the low-latency Grok 3 language model.
	ModelGrok3Fast = "grok-3-fast"
	// ModelGrok3Mini is xAI's compact Grok 3 reasoning model.
	ModelGrok3Mini = "grok-3-mini"
	// ModelGrok3MiniFast is the low-latency Grok 3 Mini model.
	ModelGrok3MiniFast = "grok-3-mini-fast"
	// ModelGrok2Vision1212 is xAI's Grok 2 vision model.
	ModelGrok2Vision1212 = "grok-2-vision-1212"
)

var (
	textModelSupports = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
		Output:     []string{"text", "json"},
	}
	visionModelSupports = ai.ModelSupports{
		Multiturn:  false,
		Tools:      true,
		SystemRole: false,
		Media:      true,
		Output:     []string{"text", "json"},
	}
	supportedModels = map[string]ai.ModelOptions{
		ModelGrok3:           modelOptions(ModelGrok3, "xAI Grok 3", textModelSupports),
		ModelGrok3Fast:       modelOptions(ModelGrok3Fast, "xAI Grok 3 Fast", textModelSupports),
		ModelGrok3Mini:       modelOptions(ModelGrok3Mini, "xAI Grok 3 Mini", textModelSupports),
		ModelGrok3MiniFast:   modelOptions(ModelGrok3MiniFast, "xAI Grok 3 Mini Fast", textModelSupports),
		ModelGrok2Vision1212: modelOptions(ModelGrok2Vision1212, "xAI Grok 2 Vision 1212", visionModelSupports),
	}
)

func modelOptions(id, label string, supports ai.ModelSupports) ai.ModelOptions {
	return ai.ModelOptions{
		Label:        label,
		ConfigSchema: core.InferSchemaMap(chatCompletionConfig{}),
		Supports:     &supports,
		Versions:     []string{id},
	}
}

// XAI configures the xAI Grok plugin.
type XAI struct {
	// APIKey is the xAI API key. If empty, XAI_API_KEY is consulted.
	APIKey string
	// BaseURL overrides the xAI API endpoint. If empty, XAI_BASE_URL and then
	// the default endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (x *XAI) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (x *XAI) Init(ctx context.Context) []api.Action {
	baseURL := x.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("XAI_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := x.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("XAI_API_KEY")
	}
	if apiKey == "" {
		panic("xai plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, x.Opts...)

	x.openAICompatible.Provider = provider
	x.openAICompatible.ConfigAliases = map[string]string{
		"reasoningEffort":  "reasoning_effort",
		"webSearchOptions": "web_search_options",
	}
	x.openAICompatible.Opts = opts
	actions := x.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, x.DefineModel(model, modelOpts).(api.Action))
	}
	return actions
}

// Model returns a registered xAI model.
func (x *XAI) Model(g *genkit.Genkit, id string) ai.Model {
	return x.openAICompatible.Model(g, api.NewName(provider, id))
}

// DefineModel registers an xAI model, including models not in the built-in list.
func (x *XAI) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return x.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions lists models exposed by the configured xAI endpoint.
func (x *XAI) ListActions(ctx context.Context) []api.ActionDesc {
	return x.openAICompatible.ListActions(ctx)
}

// ResolveAction dynamically registers a model exposed by the xAI endpoint.
func (x *XAI) ResolveAction(atype api.ActionType, name string) api.Action {
	return x.openAICompatible.ResolveAction(atype, name)
}
