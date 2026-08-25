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

// Package xai provides a Genkit plugin for xAI's Grok models.
package xai

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
	provider       = "xai"
	defaultBaseURL = "https://api.x.ai/v1"
)

// ReasoningEffort is how hard a reasoning-capable Grok model thinks before it
// answers.
//
// xAI documents two sets of levels, because it documents two APIs. The chat
// completions reference this plugin serves lists [ReasoningEffortNone] through
// [ReasoningEffortHigh] with low the default; the model capability guide, which
// covers the Responses API, lists low through [ReasoningEffortXHigh] with high
// the default and no way to turn reasoning off. The schema enumerates the
// union of both sets; which subset a model takes is the model's to decide, so
// a declared level a model does not take is an error from xAI rather than one
// from here.
type ReasoningEffort string

const (
	// ReasoningEffortNone disables reasoning. Models documented only by the
	// capability guide cannot turn it off and reject this.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortLow is fast reasoning, for latency-sensitive work and
	// simple tool calling.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium adds thinking for complex analysis and
	// long-context tasks.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh is deeper thinking, for hard problems and multi-step
	// logic.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh is the maximum depth, which xAI documents for
	// grok-4.6 alone.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
)

// ServiceTier selects how a request is scheduled and billed.
//
// xAI takes two of the tiers the OpenAI SDK models, so it is declared here
// rather than reused from [openai.ChatCompletionNewParamsServiceTier]: a
// config advertising that type would offer "auto", "flex", and "scale", which
// xAI rejects.
type ServiceTier string

const (
	// ServiceTierDefault is standard scheduling and billing.
	ServiceTierDefault ServiceTier = "default"
	// ServiceTierPriority buys faster scheduling at a higher rate.
	ServiceTierPriority ServiceTier = "priority"
)

// ChatConfig is the per-request config for Grok models: the common generation
// fields plus the xAI-specific controls. See
// https://docs.x.ai/docs/api-reference.
//
// Two documented request fields are deliberately absent. n asks for several
// completion choices and bills for all of them, while Genkit reads only the
// first, so declaring it would only sell tokens that are then thrown away.
// deferred answers with a request ID to poll rather than a completion, which
// is not a shape this model action can return.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 2."`
	// TopP is the nucleus sampling threshold. xAI documents no range for it,
	// so the schema declares none.
	TopP *float64 `json:"topP,omitempty" jsonschema_description:"Nucleus sampling threshold: only the tokens comprising the top P probability mass are considered."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_completion_tokens; xAI deprecated max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_completion_tokens."`
	// StopSequences stop generation when produced by the model, up to four.
	// Reasoning models do not support them.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema:"maxItems=4" jsonschema_description:"Stop generation when produced by the model, up to four sequences. Reasoning models do not support them."`
	// FrequencyPenalty penalizes tokens by their frequency so far, from -2.0
	// to 2.0. Reasoning models do not support it.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens by their frequency so far, from -2.0 to 2.0. Reasoning models do not support it."`
	// PresencePenalty penalizes tokens that have appeared at all, from -2.0 to
	// 2.0. Reasoning models do not support it.
	PresencePenalty *float64 `json:"presencePenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens that have appeared at all, from -2.0 to 2.0. Reasoning models do not support it."`
	// LogProbs requests log probabilities for the output tokens.
	LogProbs *bool `json:"logProbs,omitempty" jsonschema_description:"Requests log probabilities for the output tokens."`
	// TopLogProbs is how many of the most likely tokens to return log
	// probabilities for at each position, from 0 to 8; it requires LogProbs.
	TopLogProbs *int `json:"topLogProbs,omitempty" jsonschema:"minimum=0,maximum=8" jsonschema_description:"How many of the most likely tokens to return log probabilities for at each position, from 0 to 8; requires logProbs."`
	// Seed makes generation reproducible across calls when set.
	Seed *int `json:"seed,omitempty" jsonschema_description:"Makes generation reproducible across calls when set, on a best-effort basis."`
	// ReasoningEffort adjusts how hard a reasoning-capable Grok model thinks.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty" jsonschema:"enum=none,enum=low,enum=medium,enum=high,enum=xhigh" jsonschema_description:"How hard a reasoning-capable Grok model thinks, from none to xhigh. Which levels a model takes is the model's to decide."`
	// ParallelToolCalls lets the model request several tool calls in one
	// response, which it may do by default. Setting it false caps the model at
	// one call per response. It applies to a request that carries tools.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty" jsonschema_description:"Lets the model request several tool calls in one response, which it may do by default; false caps it at one call per response."`
	// User identifies the end user a request is made for, which xAI uses to
	// monitor and detect abuse.
	User string `json:"user,omitempty" jsonschema_description:"Identifies the end user a request is made for, which xAI uses to monitor and detect abuse."`
	// PromptCacheKey routes requests sharing a prompt prefix to the same
	// backend, for best-effort prompt cache hits, and is sent as the API's
	// prompt_cache_key. Hits come back as the response usage's
	// CachedContentTokens.
	PromptCacheKey string `json:"promptCacheKey,omitempty" jsonschema_description:"Routes requests sharing a prompt prefix to the same backend for best-effort prompt cache hits, sent as the API's prompt_cache_key."`
	// ServiceTier selects how the request is scheduled and billed.
	ServiceTier ServiceTier `json:"serviceTier,omitempty" jsonschema:"enum=default,enum=priority" jsonschema_description:"How the request is scheduled and billed: default, or priority for faster scheduling at a higher rate."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts, reasoning effort on the
// SDK's reasoning_effort, and the xAI controls ride as extra request fields.
func (c ChatConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
	c.ApplyVersion(params)

	if c.Temperature != nil {
		params.Temperature = openai.Float(*c.Temperature)
	}
	if c.TopP != nil {
		params.TopP = openai.Float(*c.TopP)
	}
	if c.MaxOutputTokens > 0 {
		params.MaxCompletionTokens = openai.Int(int64(c.MaxOutputTokens))
	}
	if len(c.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: c.StopSequences}
	}
	if c.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(*c.FrequencyPenalty)
	}
	if c.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*c.PresencePenalty)
	}
	if c.LogProbs != nil {
		params.Logprobs = openai.Bool(*c.LogProbs)
	}
	if c.TopLogProbs != nil {
		params.TopLogprobs = openai.Int(int64(*c.TopLogProbs))
	}
	if c.Seed != nil {
		params.Seed = openai.Int(int64(*c.Seed))
	}
	if c.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(c.ReasoningEffort)
	}
	if c.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*c.ParallelToolCalls)
	}
	if c.User != "" {
		params.User = openai.String(c.User)
	}
	if c.ServiceTier != "" {
		params.ServiceTier = openai.ChatCompletionNewParamsServiceTier(c.ServiceTier)
	}
	if c.PromptCacheKey != "" {
		compat_oai.AddExtraFields(params, map[string]any{"prompt_cache_key": c.PromptCacheKey})
	}
}

// Capability sets shared by the entries below. Every Grok model xAI serves
// through chat completions takes text and images and answers with text or
// JSON, and calls tools. They differ on constrained generation: xAI documents
// that "structured outputs with tools is only available for supported Grok 4
// family models", so anything outside that family advertises no-tools and
// falls back to schema instructions in the prompt once a request carries
// tools. See https://docs.x.ai/developers/model-capabilities/text/structured-outputs.
var (
	multimodal = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       true,
		ToolChoice:  true,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportAll,
	}
	multimodalNoToolConstraint = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       true,
		ToolChoice:  true,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportNoTools,
	}
)

// supportedModels curates capabilities for well-known Grok models. It is not
// the set of usable models: any Grok model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works. No versions are
// declared, since xAI serves each model under floating aliases and dated
// snapshots the plugin cannot enumerate, and an undeclared list leaves config
// version pinning unconstrained.
//
// Catalog: https://docs.x.ai/docs/models
var supportedModels = map[string]ai.ModelOptions{
	// The flagship, and the one model xAI documents ReasoningEffortXHigh for.
	"grok-4.6": {Label: "Grok 4.6", Supports: &multimodal},
	"grok-4.5": {Label: "Grok 4.5", Supports: &multimodal},
	// The long-context model.
	"grok-4.3":                     {Label: "Grok 4.3", Supports: &multimodal},
	"grok-4.20-0309-reasoning":     {Label: "Grok 4.20 Reasoning", Supports: &multimodal},
	"grok-4.20-0309-non-reasoning": {Label: "Grok 4.20 Non-Reasoning", Supports: &multimodal},
	// The agentic coding model, also served as "grok-code-fast-1". Outside the
	// Grok 4 family, so structured output and tools cannot be combined.
	"grok-build-0.1": {Label: "Grok Build 0.1", Supports: &multimodalNoToolConstraint},

	// grok-4.20-multi-agent-0309 is deliberately absent: xAI serves the
	// multi-agent model only through the Responses API and its own SDK, and
	// chat completions rejects it ("Multi Agent requests are not allowed on
	// chat completions").
}

// dynamicModelOptions is advertised for Grok models that resolve dynamically
// rather than appearing in supportedModels. A model xAI adds later may sit
// outside the Grok 4 family, so it takes the narrower constrained support.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &multimodalNoToolConstraint,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// XAI configures the xAI Grok plugin.
type XAI struct {
	// APIKey is the xAI API key. If empty, XAI_API_KEY is consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (XAI_BASE_URL works too).
	// Options supplied here are applied after the plugin defaults, so they
	// win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a Grok model, keyed by
	// model ID, bare or provider-prefixed. Every Grok model already works
	// without an entry: known IDs carry curated capabilities and the rest take
	// the Grok defaults. Supply an entry only to correct or extend what the
	// plugin resolves, most often for a model released after this version of
	// the plugin.
	//
	//	&xai.XAI{Models: map[string]ai.ModelOptions{
	//		"grok-4.5": {Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [XAI.ListActions] advertises and [XAI.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (x *XAI) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (x *XAI) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("XAI_BASE_URL")
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
	x.openAICompatible.Opts = opts
	actions := x.openAICompatible.Init(ctx)

	for model := range supportedModels {
		actions = append(actions, x.newModel(model, x.modelOptions(model)))
	}
	return actions
}

// newModel creates a Grok model without registering it.
func (x *XAI) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&x.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Grok model ID: curated
// capabilities for a known model and the Grok defaults for the rest, with
// an entry from [XAI.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (x *XAI) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, x.Models)
}

// ModelRef names a Grok model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(xai.ModelRef("grok-4.3", &xai.ChatConfig{
//		ReasoningEffort: "high",
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions lists the models the configured xAI endpoint exposes, described
// by the plugin's config schema and capabilities.
func (x *XAI) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &x.openAICompatible, x.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the xAI endpoint,
// described by the plugin's config schema and capabilities.
func (x *XAI) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&x.openAICompatible, atype, id, x.modelOptions)
}
