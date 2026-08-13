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

// Package deepseek provides a Genkit plugin for DeepSeek's models.
package deepseek

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
	provider       = "deepseek"
	defaultBaseURL = "https://api.deepseek.com"
)

// ThinkingType turns the thinking mode of DeepSeek models on or off.
type ThinkingType string

const (
	// ThinkingTypeEnabled turns thinking on, which is DeepSeek's default.
	ThinkingTypeEnabled ThinkingType = "enabled"
	// ThinkingTypeDisabled turns thinking off.
	ThinkingTypeDisabled ThinkingType = "disabled"
)

// ReasoningEffort is how hard a DeepSeek model thinks before it answers.
type ReasoningEffort string

const (
	// ReasoningEffortLow is the fastest, shallowest reasoning.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortHigh is deeper reasoning.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortMax is the deepest reasoning.
	ReasoningEffortMax ReasoningEffort = "max"
)

// ChatConfig is the per-request config for DeepSeek models: the generation
// fields DeepSeek accepts plus its thinking controls. See
// https://api-docs.deepseek.com/api/create-chat-completion.
//
// DeepSeek no longer supports the frequency and presence penalties, so those
// are deliberately absent.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2; DeepSeek's default is 1.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 2. DeepSeek's default is 1."`
	// TopP is the nucleus sampling threshold, up to 1.
	TopP *float64 `json:"topP,omitempty" jsonschema:"maximum=1" jsonschema_description:"Nucleus sampling threshold, up to 1."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_tokens."`
	// StopSequences stop generation when produced by the model, up to sixteen.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema:"maxItems=16" jsonschema_description:"Stop generation when produced by the model, up to sixteen sequences."`
	// LogProbs requests log probabilities for the output tokens.
	LogProbs *bool `json:"logProbs,omitempty" jsonschema_description:"Requests log probabilities for the output tokens."`
	// TopLogProbs is how many of the most likely tokens to return log
	// probabilities for at each position, from 0 to 20; it requires LogProbs.
	TopLogProbs *int `json:"topLogProbs,omitempty" jsonschema:"minimum=0,maximum=20" jsonschema_description:"How many of the most likely tokens to return log probabilities for at each position, from 0 to 20; requires logProbs."`
	// UserID identifies the end user a request is made on behalf of, up to 512
	// characters of [a-zA-Z0-9_-]. DeepSeek partitions its context cache by
	// this ID, so end users neither read nor evict each other's cached
	// prefixes; sent as the API's user_id, not OpenAI's user.
	UserID string `json:"userId,omitempty" jsonschema:"maxLength=512,pattern=^[a-zA-Z0-9_-]*$" jsonschema_description:"Identifies the end user a request is made on behalf of, up to 512 characters of letters, digits, hyphen and underscore. DeepSeek partitions its context cache by this ID; sent as the API's user_id."`
	// ReasoningEffort adjusts how hard the model thinks, [ReasoningEffortLow]
	// to [ReasoningEffortMax]; DeepSeek's default is high. Sent as the API's
	// top-level reasoning_effort: the create-chat-completion reference also
	// documents it nested inside thinking, but the service silently ignores it
	// there and reads only the top-level field the thinking-mode guide's
	// examples use.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty" jsonschema:"enum=low,enum=high,enum=max" jsonschema_description:"How hard the model thinks: low, high, or max. DeepSeek's default is high."`
	// Thinking controls the thinking mode of DeepSeek models, which is on by
	// default; sent as the API's thinking field.
	Thinking *ThinkingConfig `json:"thinking,omitempty" jsonschema_description:"Thinking mode controls, on by default; sent as the API's thinking field."`
}

// ThinkingConfig configures the thinking mode of DeepSeek models.
type ThinkingConfig struct {
	// Type turns thinking [ThinkingTypeEnabled] or [ThinkingTypeDisabled].
	Type ThinkingType `json:"type,omitempty" jsonschema:"enum=enabled,enum=disabled" jsonschema_description:"Turns thinking enabled or disabled."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts, MaxOutputTokens on the
// max_tokens DeepSeek reads, ReasoningEffort on the SDK's reasoning_effort,
// and the fields DeepSeek names differently than OpenAI, thinking and
// user_id, ride as extra request fields.
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
	if c.LogProbs != nil {
		params.Logprobs = openai.Bool(*c.LogProbs)
	}
	if c.TopLogProbs != nil {
		params.TopLogprobs = openai.Int(int64(*c.TopLogProbs))
	}
	if c.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(c.ReasoningEffort)
	}
	if c.UserID != "" {
		compat_oai.AddExtraFields(params, map[string]any{"user_id": c.UserID})
	}
	if c.Thinking != nil && c.Thinking.Type != "" {
		compat_oai.AddExtraFields(params, map[string]any{
			"thinking": map[string]any{"type": string(c.Thinking.Type)},
		})
	}
}

// textOnly is the capability set every DeepSeek model shares: text in, text
// or JSON out, tools, and thinking. No constrained generation is advertised:
// DeepSeek's response_format takes json_object only, not json_schema, so a
// schema reaches the model as prompt instructions. See
// https://api-docs.deepseek.com/guides/json_mode.
var textOnly = ai.ModelSupports{
	Multiturn:  true,
	Tools:      true,
	SystemRole: true,
	Media:      false,
	ToolChoice: true,
	Output:     []string{"text", "json"},
}

// supportedModels curates capabilities for well-known DeepSeek models. It is
// not the set of usable models: any DeepSeek model resolves on demand and
// takes [dynamicModelOptions], so an ID absent here still works. No versions
// are declared, since DeepSeek serves each model under one ID whose snapshot
// moves underneath it, and an undeclared list leaves config version pinning
// unconstrained.
//
// Catalog: https://api-docs.deepseek.com/quick_start/pricing
var supportedModels = map[string]ai.ModelOptions{
	"deepseek-v4-flash": {Label: "DeepSeek V4 Flash", Supports: &textOnly},
	"deepseek-v4-pro":   {Label: "DeepSeek V4 Pro", Supports: &textOnly},
}

// dynamicModelOptions is advertised for DeepSeek models that resolve
// dynamically rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &textOnly,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// DeepSeek configures the DeepSeek plugin.
type DeepSeek struct {
	// APIKey is the DeepSeek API key. If empty, DEEPSEEK_API_KEY is consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint, e.g. DeepSeek's beta one
	// (DEEPSEEK_BASE_URL works too). Options supplied here are applied after
	// the plugin defaults, so they win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a DeepSeek model, keyed by
	// model ID, bare or provider-prefixed. Every DeepSeek model already works
	// without an entry: known IDs carry curated capabilities and the rest take
	// the DeepSeek defaults. Supply an entry only to correct or extend what the
	// plugin resolves, most often for a model released after this version of
	// the plugin.
	//
	//	&deepseek.DeepSeek{Models: map[string]ai.ModelOptions{
	//		"deepseek-v4-pro": {Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [DeepSeek.ListActions] advertises and [DeepSeek.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (d *DeepSeek) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (d *DeepSeek) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("DEEPSEEK_BASE_URL")
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
	d.openAICompatible.Opts = opts
	actions := d.openAICompatible.Init(ctx)

	for model := range supportedModels {
		actions = append(actions, d.newModel(model, d.modelOptions(model)))
	}
	return actions
}

// newModel creates a DeepSeek model without registering it.
func (d *DeepSeek) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&d.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a DeepSeek model ID: curated
// capabilities for a known model and the DeepSeek defaults for the rest, with
// an entry from [DeepSeek.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (d *DeepSeek) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, d.Models)
}

// ModelRef names a DeepSeek model and carries the config to generate with, so
// the config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(deepseek.ModelRef("deepseek-v4-pro", &deepseek.ChatConfig{
//		Thinking: &deepseek.ThinkingConfig{Type: "disabled"},
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions lists the models the configured DeepSeek endpoint exposes,
// described by the plugin's config schema and capabilities.
func (d *DeepSeek) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &d.openAICompatible, d.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the DeepSeek endpoint,
// described by the plugin's config schema and capabilities.
func (d *DeepSeek) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&d.openAICompatible, atype, id, d.modelOptions)
}
