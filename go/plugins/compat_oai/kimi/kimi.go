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

// Package kimi provides a Genkit plugin for Moonshot AI's Kimi models.
package kimi

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

const (
	provider       = "kimi"
	defaultBaseURL = "https://api.moonshot.ai/v1"
)

// ChatConfig is the per-request config for Kimi models: the generation fields
// the K-series accepts plus the Moonshot-specific controls. See
// https://platform.kimi.ai/docs/api/chat.
//
// Moonshot documents temperature, topP, and the frequency and presence
// penalties for the legacy moonshot-v1 family only, so the K-series models
// this plugin serves do not take them and they are deliberately absent.
type ChatConfig struct {
	compat_oai.RequestConfig

	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_completion_tokens; Moonshot deprecated max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
	// StopSequences stop generation when produced by the model, up to five.
	StopSequences []string `json:"stopSequences,omitempty"`
	// LogProbs requests log probabilities for the output tokens.
	LogProbs *bool `json:"logProbs,omitempty"`
	// TopLogProbs is how many of the most likely tokens to return log
	// probabilities for at each position, from 0 to 20; it requires LogProbs.
	TopLogProbs *int `json:"topLogProbs,omitempty"`
	// Thinking controls the reasoning mode of thinking-capable Kimi models,
	// sent as the API's thinking field.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
	// ReasoningEffort adjusts how hard the Kimi K3 generation thinks: "low",
	// "high", or "max".
	ReasoningEffort string `json:"reasoningEffort,omitempty"`
}

// ThinkingConfig configures the reasoning of thinking-capable Kimi models.
type ThinkingConfig struct {
	// Type turns thinking "enabled" or "disabled".
	Type string `json:"type,omitempty"`
	// Keep controls how much reasoning is preserved across turns, e.g. "all".
	Keep string `json:"keep,omitempty"`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts, reasoning effort on the
// SDK's reasoning_effort, and thinking rides as Moonshot's extra request
// field.
func (c ChatConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
	c.ApplyVersion(params)

	if c.MaxOutputTokens > 0 {
		params.MaxCompletionTokens = openai.Int(int64(c.MaxOutputTokens))
	}
	if len(c.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: c.StopSequences}
	}
	if c.LogProbs != nil {
		params.Logprobs = openai.Bool(*c.LogProbs)
	}
	if c.TopLogProbs != nil {
		params.TopLogprobs = openai.Int(int64(*c.TopLogProbs))
	}
	if c.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(c.ReasoningEffort)
	}
	if c.Thinking != nil {
		thinking := map[string]any{}
		if c.Thinking.Type != "" {
			thinking["type"] = c.Thinking.Type
		}
		if c.Thinking.Keep != "" {
			thinking["keep"] = c.Thinking.Keep
		}
		// An all-zero ThinkingConfig adds nothing rather than sending an
		// empty thinking object the API could reject.
		if len(thinking) > 0 {
			compat_oai.AddExtraFields(params, map[string]any{"thinking": thinking})
		}
	}
}

// multimodal is the capability set every K-series Kimi model shares: text and
// images in, text or JSON out, and tools. Moonshot's chat API takes
// response_format json_schema, so structured output is generated natively
// rather than coaxed through prompt instructions. See
// https://platform.kimi.ai/docs/api/chat.
var multimodal = ai.ModelSupports{
	Multiturn:   true,
	Tools:       true,
	SystemRole:  true,
	Media:       true,
	ToolChoice:  true,
	Output:      []string{"text", "json"},
	Constrained: ai.ConstrainedSupportAll,
}

// supportedModels curates capabilities for well-known Kimi models. It is not
// the set of usable models: any Kimi model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works. No versions are
// declared, since Moonshot serves each model under one ID whose snapshot moves
// underneath it, and an undeclared list leaves config version pinning
// unconstrained.
//
// Catalog: https://platform.kimi.ai/docs/api/chat
var supportedModels = map[string]ai.ModelOptions{
	"kimi-k3":                  {Label: "Kimi K3", Supports: &multimodal},
	"kimi-k2.5":                {Label: "Kimi K2.5 (Deprecated)", Supports: &multimodal, Stage: ai.ModelStageDeprecated},
	"kimi-k2.6":                {Label: "Kimi K2.6", Supports: &multimodal},
	"kimi-k2.7-code":           {Label: "Kimi K2.7 Code", Supports: &multimodal},
	"kimi-k2.7-code-highspeed": {Label: "Kimi K2.7 Code Highspeed", Supports: &multimodal},
}

// dynamicModelOptions is advertised for Kimi models that resolve dynamically
// rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &multimodal,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// Kimi configures the Moonshot AI Kimi plugin.
type Kimi struct {
	// APIKey is the Moonshot API key. If empty, KIMI_API_KEY and then
	// MOONSHOT_API_KEY are consulted.
	APIKey string
	// BaseURL overrides the Moonshot API endpoint. If empty, KIMI_BASE_URL,
	// MOONSHOT_BASE_URL, and then the default international endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (k *Kimi) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (k *Kimi) Init(ctx context.Context) []api.Action {
	baseURL := k.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("KIMI_BASE_URL")
	}
	if baseURL == "" {
		baseURL = os.Getenv("MOONSHOT_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := k.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("KIMI_API_KEY")
	}
	if apiKey == "" {
		apiKey = os.Getenv("MOONSHOT_API_KEY")
	}
	if apiKey == "" {
		panic("kimi plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, k.Opts...)

	k.openAICompatible.Provider = provider
	k.openAICompatible.Opts = opts
	actions := k.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, k.newModel(model, modelOpts))
	}
	return actions
}

// newModel creates a Kimi model without registering it.
func (k *Kimi) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&k.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Kimi model ID: curated
// capabilities for a known model and the Kimi defaults for the rest.
func modelOptions(id string) ai.ModelOptions {
	if opts, ok := supportedModels[id]; ok {
		return opts
	}
	return dynamicModelOptions
}

// ModelRef names a Kimi model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(kimi.ModelRef("kimi-k3", &kimi.ChatConfig{
//		ReasoningEffort: "high",
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// RegisterModel registers a Kimi model with g and returns it. The plugin
// supplies the implementation; opts describes
// what the model supports, and a nil opts takes the capabilities the plugin
// resolves for that ID, curated for a known model and the Kimi defaults for
// the rest.
//
// Registering an ID that is already registered panics; Init registers every
// curated model and generating with an ID registers it on demand, so define
// a model before its first use or guard with [IsDefinedModel]. name is the
// model ID, bare or provider-prefixed.
func (k *Kimi) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return compat_oai.RegisterChatModel[ChatConfig](g, &k.openAICompatible, id, opts, modelOptions)
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [Kimi.RegisterModel]).
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return compat_oai.IsDefinedModel(g, provider, id)
}

// ListActions lists the models the configured Kimi endpoint exposes,
// described by the plugin's config schema and capabilities.
func (k *Kimi) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &k.openAICompatible, modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Kimi endpoint,
// described by the plugin's config schema and capabilities.
func (k *Kimi) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&k.openAICompatible, atype, id, modelOptions)
}
