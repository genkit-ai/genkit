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
	"github.com/openai/openai-go"
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

// ChatConfig is the per-request config for GLM models: the generation fields
// Z.ai accepts plus the Z.ai-specific controls. See
// https://docs.z.ai/api-reference/llm/chat-completion.
//
// Z.ai documents no penalties, log probabilities, or seed, so those are
// deliberately absent, and its temperature range stops at 1 rather than the 2
// OpenAI allows.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 1.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP is the nucleus sampling threshold, from 0.01 to 1.
	TopP *float64 `json:"topP,omitempty"`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
	// StopSequences stop generation when produced by the model.
	StopSequences []string `json:"stopSequences,omitempty"`
	// Thinking controls the chain-of-thought mode of GLM 4.5 and later
	// models, sent as the API's thinking field.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
	// DoSample turns sampling off when set to false, making temperature and
	// TopP inert; sent as the API's do_sample.
	DoSample *bool `json:"doSample,omitempty"`
}

// ThinkingConfig configures the chain-of-thought mode of GLM models.
type ThinkingConfig struct {
	// Type turns thinking "enabled" or "disabled".
	Type string `json:"type,omitempty"`
	// ClearThinking controls whether the reasoning content is cleared from
	// the response, sent as the API's clear_thinking.
	ClearThinking *bool `json:"clearThinking,omitempty"`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts and the Z.ai controls ride
// as extra request fields.
func (c ChatConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
	c.ApplyVersion(params)

	if c.Temperature != nil {
		params.Temperature = openai.Float(*c.Temperature)
	}
	if c.TopP != nil {
		params.TopP = openai.Float(*c.TopP)
	}
	if c.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(c.MaxOutputTokens))
	}
	if len(c.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: c.StopSequences}
	}

	if c.Thinking != nil {
		thinking := map[string]any{}
		if c.Thinking.Type != "" {
			thinking["type"] = c.Thinking.Type
		}
		if c.Thinking.ClearThinking != nil {
			thinking["clear_thinking"] = *c.Thinking.ClearThinking
		}
		// An all-zero ThinkingConfig adds nothing rather than sending an
		// empty thinking object the API could reject.
		if len(thinking) > 0 {
			compat_oai.AddExtraFields(params, map[string]any{"thinking": thinking})
		}
	}
	if c.DoSample != nil {
		compat_oai.AddExtraFields(params, map[string]any{"do_sample": *c.DoSample})
	}
}

var supportedModels = map[string]ai.ModelOptions{
	ModelGLM51:           newModelOptions(ModelGLM51, "Z.ai GLM 5.1", false),
	ModelGLM5Turbo:       newModelOptions(ModelGLM5Turbo, "Z.ai GLM 5 Turbo", false),
	ModelGLM5:            newModelOptions(ModelGLM5, "Z.ai GLM 5", false),
	ModelGLM47:           newModelOptions(ModelGLM47, "Z.ai GLM 4.7", false),
	ModelGLM47Flash:      newModelOptions(ModelGLM47Flash, "Z.ai GLM 4.7 Flash", false),
	ModelGLM47FlashX:     newModelOptions(ModelGLM47FlashX, "Z.ai GLM 4.7 FlashX", false),
	ModelGLM46:           newModelOptions(ModelGLM46, "Z.ai GLM 4.6", false),
	ModelGLM45:           newModelOptions(ModelGLM45, "Z.ai GLM 4.5", false),
	ModelGLM45Air:        newModelOptions(ModelGLM45Air, "Z.ai GLM 4.5 Air", false),
	ModelGLM45X:          newModelOptions(ModelGLM45X, "Z.ai GLM 4.5 X", false),
	ModelGLM45AirX:       newModelOptions(ModelGLM45AirX, "Z.ai GLM 4.5 AirX", false),
	ModelGLM45Flash:      newModelOptions(ModelGLM45Flash, "Z.ai GLM 4.5 Flash", false),
	ModelGLM432B0414128K: newModelOptions(ModelGLM432B0414128K, "Z.ai GLM 4 32B 128K", false),
	ModelGLM5VTurbo:      newModelOptions(ModelGLM5VTurbo, "Z.ai GLM 5V Turbo", true),
	ModelGLM46V:          newModelOptions(ModelGLM46V, "Z.ai GLM 4.6V", true),
	ModelGLM46VFlash:     newModelOptions(ModelGLM46VFlash, "Z.ai GLM 4.6V Flash", true),
	ModelGLM46VFlashX:    newModelOptions(ModelGLM46VFlashX, "Z.ai GLM 4.6V FlashX", true),
	ModelGLM45V:          newModelOptions(ModelGLM45V, "Z.ai GLM 4.5V", true),
}

// newModelOptions builds the curated options entry for a GLM model. No
// versions are declared: Z.ai serves dated snapshots the plugin cannot
// enumerate, and an undeclared list leaves config version pinning
// unconstrained.
func newModelOptions(id, label string, media bool) ai.ModelOptions {
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
		actions = append(actions, z.newModel(model, modelOpts))
	}
	return actions
}

// newModel creates a GLM model without registering it.
func (z *ZAI) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&z.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a GLM model ID: curated
// capabilities for a known model and the GLM defaults for the rest.
func modelOptions(id string) ai.ModelOptions {
	if opts, ok := supportedModels[id]; ok {
		return opts
	}
	return ai.ModelOptions{
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			ToolChoice: false,
			Output:     []string{"text", "json"},
		},
		Versions: []string{},
		Stage:    ai.ModelStageStable,
	}
}

// ModelRef names a GLM model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(zai.ModelRef(zai.ModelGLM5, &zai.ChatConfig{
//		Thinking: &zai.ThinkingConfig{Type: "enabled"},
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// RegisterModel registers a GLM model with g and returns it. The plugin
// supplies the implementation; opts describes
// what the model supports, and a nil opts takes the capabilities the plugin
// resolves for that ID, curated for a known model and the GLM defaults for
// the rest.
//
// Registering an ID that is already registered panics; Init registers every
// curated model and generating with an ID registers it on demand, so define
// a model before its first use or guard with [IsDefinedModel]. name is the
// model ID, bare or provider-prefixed.
func (z *ZAI) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return compat_oai.RegisterChatModel[ChatConfig](g, &z.openAICompatible, id, opts, modelOptions)
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [ZAI.RegisterModel]).
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return compat_oai.IsDefinedModel(g, provider, id)
}

// Model returns a previously registered model.
//
// Deprecated: Generation resolves a model from its name, so looking one up
// first is rarely necessary: pass ai.WithModelName("zai/glm-5") or, to carry
// config with it, [ModelRef]. Use [genkit.LookupModel] when the action itself
// is what you need.
func (z *ZAI) Model(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, compat_oai.ActionName(provider, id))
}

// ListActions lists the models the configured Z.ai endpoint exposes,
// described by the plugin's config schema and capabilities.
func (z *ZAI) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &z.openAICompatible, modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Z.ai endpoint,
// described by the plugin's config schema and capabilities.
func (z *ZAI) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&z.openAICompatible, atype, id, modelOptions)
}
