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

// Package meta provides a Genkit plugin for Meta's Muse models, served by the
// Meta Model API.
package meta

import (
	"context"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

const (
	provider       = "meta"
	defaultBaseURL = "https://api.meta.ai/v1"
)

// ReasoningEffort is how hard a Muse model thinks before it answers. Muse
// models always reason; the effort only sets the depth, and an unset effort
// leaves the level to the model.
//
// Meta also documents "none", which every Muse chat model rejects with a 400,
// so it is not a declared level. A model that takes it can still be sent it
// through the config's extra map.
type ReasoningEffort string

const (
	// ReasoningEffortMinimal is the shallowest reasoning.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow is light reasoning.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium is moderate reasoning.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh is deep reasoning.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh is deeper reasoning than high.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
	// ReasoningEffortMax is the deepest reasoning. muse-spark-1.3-contributor
	// rejects it.
	ReasoningEffortMax ReasoningEffort = "max"
)

// PromptCacheRetention is how long Meta keeps a cached prompt prefix.
type PromptCacheRetention string

const (
	// PromptCacheRetentionInMemory keeps the prefix in memory only.
	PromptCacheRetentionInMemory PromptCacheRetention = "in_memory"
	// PromptCacheRetention24h keeps the prefix for up to 24 hours.
	PromptCacheRetention24h PromptCacheRetention = "24h"
)

// ChatConfig is the per-request config for Muse models: the generation fields
// Meta's chat completions accept plus the Meta-specific controls. See
// https://dev.meta.ai/docs/protocols/chat-completions.
//
// Several OpenAI fields are deliberately absent because Meta rejects them
// with a 400: stop (Muse models reason), logprobs, n above 1, and the
// Responses-only max_output_tokens. user is absent too, since Meta deprecated
// it in favor of SafetyIdentifier and PromptCacheKey.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2. Muse models are tuned for the default of 1.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 2. Muse models are tuned for the default of 1."`
	// TopP is the nucleus sampling threshold, above 0 and up to 1. Meta
	// advises setting it or Temperature, not both.
	TopP *float64 `json:"topP,omitempty" jsonschema:"exclusiveMinimum=0,maximum=1" jsonschema_description:"Nucleus sampling threshold, above 0 and up to 1 inclusive. Set this or temperature, not both."`
	// MaxOutputTokens is the maximum number of tokens to generate, reasoning
	// included, sent as the API's max_completion_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, reasoning included, sent as the API's max_completion_tokens."`
	// FrequencyPenalty penalizes tokens by their frequency so far, from -2.0
	// to 2.0.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens by their frequency so far, from -2.0 to 2.0."`
	// PresencePenalty penalizes tokens that have appeared at all, from -2.0 to
	// 2.0.
	PresencePenalty *float64 `json:"presencePenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens that have appeared at all, from -2.0 to 2.0."`
	// Seed makes generation reproducible across calls when set, on a
	// best-effort basis.
	Seed *int `json:"seed,omitempty" jsonschema_description:"Makes generation reproducible across calls when set, on a best-effort basis."`
	// ReasoningEffort sets how hard the model thinks, from
	// [ReasoningEffortMinimal] to [ReasoningEffortMax]. Unset leaves the level
	// to the model.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty" jsonschema:"enum=minimal,enum=low,enum=medium,enum=high,enum=xhigh,enum=max" jsonschema_description:"How hard the model thinks, from minimal to max. Unset leaves the level to the model. muse-spark-1.3-contributor rejects max."`
	// ParallelToolCalls lets the model request several tool calls in one
	// response, which it does by default. Setting it false caps the model at
	// one call per response. It applies to a request that carries tools.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty" jsonschema_description:"Lets the model request several tool calls in one response, which it does by default; false caps it at one call per response."`
	// PromptCacheKey groups requests that share a prompt prefix, for better
	// cache hits, and is sent as the API's prompt_cache_key. Hits come back as
	// the response usage's CachedContentTokens.
	PromptCacheKey string `json:"promptCacheKey,omitempty" jsonschema_description:"Groups requests that share a prompt prefix for better cache hits, sent as the API's prompt_cache_key."`
	// PromptCacheRetention sets how long a cached prefix is kept, sent as the
	// API's prompt_cache_retention.
	PromptCacheRetention PromptCacheRetention `json:"promptCacheRetention,omitempty" jsonschema:"enum=in_memory,enum=24h" jsonschema_description:"How long a cached prompt prefix is kept: in_memory, or 24h. Sent as the API's prompt_cache_retention."`
	// SafetyIdentifier identifies the end user a request is made for, up to 64
	// characters, which Meta uses to detect abuse. Send a hash of a username
	// or email, not the value itself.
	SafetyIdentifier string `json:"safetyIdentifier,omitempty" jsonschema:"maxLength=64" jsonschema_description:"Identifies the end user a request is made for, up to 64 characters, which Meta uses to detect abuse. Send a hash, not a raw username or email."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts, reasoning effort on the
// SDK's reasoning_effort, and the Meta controls ride as extra request fields.
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
	if c.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(*c.FrequencyPenalty)
	}
	if c.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*c.PresencePenalty)
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
	extra := map[string]any{}
	if c.SafetyIdentifier != "" {
		extra["safety_identifier"] = c.SafetyIdentifier
	}
	if c.PromptCacheKey != "" {
		extra["prompt_cache_key"] = c.PromptCacheKey
	}
	if c.PromptCacheRetention != "" {
		extra["prompt_cache_retention"] = string(c.PromptCacheRetention)
	}
	if len(extra) > 0 {
		compat_oai.AddExtraFields(params, extra)
	}
}

// multimodal is the capability set of every Muse chat model: text, images,
// audio and PDFs in, text or JSON out, and tools. Meta's chat completions
// take tool_choice auto only and answer none, required, or a named tool with
// a 400, so tool-choice steering is not advertised and the framework rejects
// it before the wire. Structured output is native, through response_format
// json_schema; Meta documents no combination of a response format with
// tools, so the constraint is claimed only for requests without them. See
// https://dev.meta.ai/docs/tool-calling and
// https://dev.meta.ai/docs/structured-output.
var multimodal = ai.ModelSupports{
	Multiturn:   true,
	Tools:       true,
	SystemRole:  true,
	Media:       true,
	ToolChoice:  false,
	Output:      []string{"text", "json", "array", "enum"},
	Constrained: ai.ConstrainedSupportNoTools,
}

// mediaTypes are the media Meta's chat completions read that the plugin
// sends: images as image_url, audio as input_audio (wav and mp3 only), and
// PDFs as file. Meta also reads video_url parts, which the compat_oai base
// does not send. See https://dev.meta.ai/docs/file-handling.
var mediaTypes = []string{"image/*", "audio/wav", "audio/mpeg", "application/pdf"}

// supportedModels curates capabilities for the Muse models Meta's chat
// completions serve. It is not the set of usable models: any model the Model
// API serves resolves on demand and takes [dynamicModelOptions], so an ID
// absent here still works. No versions are declared, since Meta versions its
// models in the ID itself.
//
// The -contributor variants are discounted in exchange for Meta training on
// the prompts, files and responses sent to them; calling one is the opt-in.
// Muse Wonder Full is absent: Meta serves it in Muse Code only.
//
// Catalog: https://dev.meta.ai/docs/models
var supportedModels = map[string]ai.ModelOptions{
	"muse-wonder-1.0":            {Label: "Muse Wonder 1.0 (Preview)", Supports: &multimodal, Stage: ai.ModelStageUnstable},
	"muse-spark-1.3":             {Label: "Muse Spark 1.3", Supports: &multimodal},
	"muse-spark-1.3-contributor": {Label: "Muse Spark 1.3 (Contributor, trains on your data)", Supports: &multimodal},
	"muse-spark-1.2":             {Label: "Muse Spark 1.2", Supports: &multimodal},
	"muse-spark-1.2-contributor": {Label: "Muse Spark 1.2 (Contributor, trains on your data)", Supports: &multimodal},
	"muse-spark-1.1":             {Label: "Muse Spark 1.1", Supports: &multimodal},
}

// dynamicModelOptions is advertised for Muse models that resolve dynamically
// rather than appearing in supportedModels. A model Meta adds later is
// assumed to share the Muse chat capabilities.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &multimodal,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// nonChatPrefixes name the model families the Model API lists that chat
// completions do not serve: Muse Image generates images, Muse Voice
// Transcribe transcribes audio, and SAM segments media. Listing them as chat
// models would offer the Dev UI models that fail on every request. A family
// Meta adds later is listed, since a new chat family is the likelier case and
// a stray listing costs less than a hidden model.
var nonChatPrefixes = []string{"muse-image", "muse-voice", "sam-"}

// listChatModels lists the Model API's models minus the families in
// [nonChatPrefixes].
func listChatModels(ctx context.Context, client *openai.Client) ([]string, error) {
	var models []string
	iter := client.Models.ListAutoPaging(ctx)
	for iter.Next() {
		if id := iter.Current().ID; !isNonChat(id) {
			models = append(models, id)
		}
	}
	if err := iter.Err(); err != nil {
		return nil, compat_oai.WrapAPIError(err)
	}
	return models, nil
}

// isNonChat reports whether id belongs to a family in [nonChatPrefixes].
func isNonChat(id string) bool {
	for _, prefix := range nonChatPrefixes {
		if strings.HasPrefix(id, prefix) {
			return true
		}
	}
	return false
}

// Meta configures the Meta Model API plugin.
type Meta struct {
	// APIKey is the Meta Model API key. If empty, META_API_KEY is consulted.
	// Meta's own examples read MODEL_API_KEY, which the plugin does not: the
	// name is generic enough to hold another provider's key, which would then
	// be sent to Meta.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (META_BASE_URL works
	// too). Options supplied here are applied after the plugin defaults, so
	// they win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a Muse model, keyed by
	// model ID, bare or provider-prefixed. Every model the Model API serves
	// already works without an entry: known IDs carry curated capabilities
	// and the rest take the Muse defaults. Supply an entry only to correct or
	// extend what the plugin resolves, most often for a model released after
	// this version of the plugin.
	//
	//	&meta.Meta{Models: map[string]ai.ModelOptions{
	//		"muse-wonder-1.0": {Stage: ai.ModelStageStable},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [Meta.ListActions] advertises and [Meta.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (m *Meta) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (m *Meta) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("META_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := m.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("META_API_KEY")
	}
	if apiKey == "" {
		panic("meta plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, m.Opts...)

	m.openAICompatible.Provider = provider
	m.openAICompatible.MediaTypes = mediaTypes
	m.openAICompatible.Opts = opts
	m.openAICompatible.ListModels = listChatModels
	actions := m.openAICompatible.Init(ctx)

	for model := range supportedModels {
		actions = append(actions, m.newModel(model, m.modelOptions(model)))
	}
	return actions
}

// newModel creates a Muse model without registering it.
func (m *Meta) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&m.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Muse model ID: curated
// capabilities for a known model and the Muse defaults for the rest, with an
// entry from [Meta.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (m *Meta) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, m.Models)
}

// ModelRef names a Muse model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(meta.ModelRef("muse-wonder-1.0", &meta.ChatConfig{
//		ReasoningEffort: meta.ReasoningEffortHigh,
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions lists the models the configured Meta endpoint exposes,
// described by the plugin's config schema and capabilities.
func (m *Meta) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &m.openAICompatible, m.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Meta endpoint,
// described by the plugin's config schema and capabilities.
func (m *Meta) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&m.openAICompatible, atype, id, m.modelOptions)
}
