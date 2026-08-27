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
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

const (
	provider       = "kimi"
	defaultBaseURL = "https://api.moonshot.ai/v1"
)

// ReasoningEffort is how hard the Kimi K3 generation thinks before it
// answers. Moonshot documents three levels, with [ReasoningEffortMax] the
// default.
type ReasoningEffort string

const (
	// ReasoningEffortLow is the fastest, shallowest reasoning.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortHigh is deeper reasoning, below the default.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortMax is the deepest reasoning, and the default.
	ReasoningEffortMax ReasoningEffort = "max"
)

// ThinkingType turns the reasoning of thinking-capable Kimi models on or off.
type ThinkingType string

const (
	// ThinkingTypeEnabled turns thinking on.
	ThinkingTypeEnabled ThinkingType = "enabled"
	// ThinkingTypeDisabled turns thinking off.
	ThinkingTypeDisabled ThinkingType = "disabled"
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
	// API's max_completion_tokens; Moonshot deprecated max_tokens. The default
	// and the ceiling vary by model.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_completion_tokens. The default and the ceiling vary by model."`
	// StopSequences stop generation when produced by the model, up to five of
	// at most 32 bytes each. The schema enforces the per-entry limit in
	// characters; Moonshot counts bytes.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema:"maxItems=5,maxLength=32" jsonschema_description:"Stop generation when produced by the model, up to five sequences of at most 32 bytes each."`
	// LogProbs requests log probabilities for the output tokens.
	LogProbs *bool `json:"logProbs,omitempty" jsonschema_description:"Requests log probabilities for the output tokens."`
	// TopLogProbs is how many of the most likely tokens to return log
	// probabilities for at each position, from 0 to 20; it requires LogProbs.
	TopLogProbs *int `json:"topLogProbs,omitempty" jsonschema:"minimum=0,maximum=20" jsonschema_description:"How many of the most likely tokens to return log probabilities for at each position, from 0 to 20; requires logProbs."`
	// Thinking controls the reasoning mode of thinking-capable Kimi models,
	// sent as the API's thinking field.
	Thinking *ThinkingConfig `json:"thinking,omitempty" jsonschema_description:"Reasoning mode controls for thinking-capable Kimi models, sent as the API's thinking field."`
	// ReasoningEffort adjusts how hard the Kimi K3 generation thinks, from
	// [ReasoningEffortLow] to [ReasoningEffortMax], the default.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty" jsonschema:"enum=low,enum=high,enum=max" jsonschema_description:"How hard the Kimi K3 generation thinks: low, high, or max (the default)."`
}

// ThinkingConfig configures the reasoning of thinking-capable Kimi models.
type ThinkingConfig struct {
	// Type turns thinking [ThinkingTypeEnabled] or [ThinkingTypeDisabled].
	Type ThinkingType `json:"type,omitempty" jsonschema:"enum=enabled,enum=disabled" jsonschema_description:"Turns thinking enabled or disabled."`
	// Keep controls how much reasoning is preserved across turns, "all" or
	// unset. It is not an enum in the schema: Moonshot documents one value
	// today, and a list of one would reject whatever it adds next.
	Keep string `json:"keep,omitempty" jsonschema_description:"How much reasoning is preserved across turns: all, or unset for the default."`
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
			thinking["type"] = string(c.Thinking.Type)
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

// Capability sets shared by the entries below: text and images in, text or
// JSON out, and tools. Moonshot's chat API takes response_format json_schema,
// so structured output is generated natively rather than coaxed through
// prompt instructions. See https://platform.kimi.ai/docs/api/chat.
var (
	// multimodal is the Kimi K3 set, with free tool choice.
	multimodal = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       true,
		ToolChoice:  true,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportAll,
	}
	// multimodalNoToolChoice is multimodal minus tool-choice steering: the K2
	// generation rejects tool_choice required as incompatible with thinking,
	// which is on by default (K2.7 Code cannot even turn it off), so only the
	// auto default is dependable and the framework waves auto through while
	// rejecting the rest. A caller who always disables thinking can restore
	// the claim through [Kimi.Models].
	multimodalNoToolChoice = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       true,
		ToolChoice:  false,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportAll,
	}
)

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
	"kimi-k2.5":                {Label: "Kimi K2.5 (Deprecated)", Supports: &multimodalNoToolChoice, Stage: ai.ModelStageDeprecated},
	"kimi-k2.6":                {Label: "Kimi K2.6", Supports: &multimodalNoToolChoice},
	"kimi-k2.7-code":           {Label: "Kimi K2.7 Code", Supports: &multimodalNoToolChoice},
	"kimi-k2.7-code-highspeed": {Label: "Kimi K2.7 Code Highspeed", Supports: &multimodalNoToolChoice},
}

// dynamicModelOptions is advertised for Kimi models that resolve dynamically
// rather than appearing in supportedModels. A model Moonshot adds later is
// assumed K3-shaped, with free tool choice.
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
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (KIMI_BASE_URL and
	// MOONSHOT_BASE_URL work too). Options supplied here are applied after
	// the plugin defaults, so they win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a Kimi model, keyed by
	// model ID, bare or provider-prefixed. Every Kimi model already works
	// without an entry: known IDs carry curated capabilities and the rest take
	// the Kimi defaults. Supply an entry only to correct or extend what the
	// plugin resolves, most often for a model released after this version of
	// the plugin.
	//
	//	&kimi.Kimi{Models: map[string]ai.ModelOptions{
	//		"kimi-k3": {Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [Kimi.ListActions] advertises and [Kimi.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (k *Kimi) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (k *Kimi) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("KIMI_BASE_URL")
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

	for model := range supportedModels {
		actions = append(actions, k.newModel(model, k.modelOptions(model)))
	}
	return actions
}

// newModel creates a Kimi model without registering it.
func (k *Kimi) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&k.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Kimi model ID: curated
// capabilities for a known model and the Kimi defaults for the rest, with
// an entry from [Kimi.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (k *Kimi) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, k.Models)
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

// ListActions lists the models the configured Kimi endpoint exposes,
// described by the plugin's config schema and capabilities.
func (k *Kimi) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &k.openAICompatible, k.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Kimi endpoint,
// described by the plugin's config schema and capabilities.
func (k *Kimi) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&k.openAICompatible, atype, id, k.modelOptions)
}
