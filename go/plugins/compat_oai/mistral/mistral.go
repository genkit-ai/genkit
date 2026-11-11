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

// Package mistral provides a Genkit plugin for Mistral AI's models.
package mistral

import (
	"context"
	"encoding/json"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/internal"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

const (
	provider       = "mistral"
	defaultBaseURL = "https://api.mistral.ai/v1"
)

// ReasoningEffort is how hard a reasoning-capable Mistral model thinks before
// it answers.
//
// The API reference enumerates every level below, but each model takes a
// subset of them: Mistral Small and Medium take none and high, Mistral Large 4
// low, medium and high. The schema enumerates the full set; which subset a
// model takes is the model's to decide, so a level a model does not take is an
// error from Mistral rather than one from here.
type ReasoningEffort string

const (
	// ReasoningEffortNone turns reasoning off.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortMinimal is the least reasoning short of none.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow is light reasoning.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium is moderate reasoning.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh is deep reasoning.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh is deeper reasoning than high.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
	// ReasoningEffortMax is the deepest reasoning.
	ReasoningEffortMax ReasoningEffort = "max"
)

// ServiceTier selects the capacity a request is served from.
//
// Mistral's tiers are not OpenAI's, so it is declared here rather than reused
// from [openai.ChatCompletionNewParamsServiceTier]: a config advertising that
// type would offer values Mistral rejects.
type ServiceTier string

const (
	// ServiceTierAuto lets Mistral choose, using priority capacity where the
	// account has it.
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierStandardOnly serves the request from standard capacity only.
	ServiceTierStandardOnly ServiceTier = "standard_only"
)

// ChatConfig is the per-request config for Mistral models: the generation
// fields Mistral's chat completions accept, under the names other plugins use
// for the same settings. See https://docs.mistral.ai/api/endpoint/chat.
//
// Mistral's request schema is closed, so a field it does not declare fails
// the request. That is why the config differs from OpenAI's in three places:
// maxOutputTokens is sent as max_tokens (Mistral has no
// max_completion_tokens), seed as random_seed, and there is no user field.
// n is deliberately absent: it bills for several choices while Genkit reads
// only the first.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection.
	// Mistral recommends values between 0.0 and 0.7 and documents no hard
	// range, so the schema declares none; the default varies by model.
	Temperature *float64 `json:"temperature,omitempty" jsonschema_description:"Controls the degree of randomness in token selection. Mistral recommends 0.0 to 0.7; the default varies by model."`
	// TopP is the nucleus sampling threshold: only the tokens comprising the
	// top P probability mass are considered.
	TopP *float64 `json:"topP,omitempty" jsonschema_description:"Nucleus sampling threshold: only the tokens comprising the top P probability mass are considered."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_tokens."`
	// StopSequences stop generation when produced by the model.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema_description:"Stop generation when produced by the model."`
	// FrequencyPenalty penalizes tokens by how often they have appeared so
	// far, discouraging verbatim repetition.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty" jsonschema_description:"Penalizes tokens by how often they have appeared so far, discouraging verbatim repetition."`
	// PresencePenalty penalizes tokens that have appeared at all, encouraging
	// a wider variety of words and phrases.
	PresencePenalty *float64 `json:"presencePenalty,omitempty" jsonschema_description:"Penalizes tokens that have appeared at all, encouraging a wider variety of words and phrases."`
	// Seed makes sampling deterministic across calls when set, sent as the
	// API's random_seed.
	Seed *int `json:"seed,omitempty" jsonschema_description:"Makes sampling deterministic across calls when set, sent as the API's random_seed."`
	// ReasoningEffort adjusts how hard a reasoning-capable model thinks.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty" jsonschema:"enum=none,enum=minimal,enum=low,enum=medium,enum=high,enum=xhigh,enum=max" jsonschema_description:"How hard a reasoning-capable model thinks, from none to max. Which levels a model takes is the model's to decide."`
	// ParallelToolCalls lets the model request several tool calls in one
	// response. It applies to a request that carries tools.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty" jsonschema_description:"Lets the model request several tool calls in one response; false caps it at one call per response."`
	// SafePrompt injects Mistral's safety prompt before the conversation,
	// sent as the API's safe_prompt.
	SafePrompt *bool `json:"safePrompt,omitempty" jsonschema_description:"Injects Mistral's safety prompt before the conversation, sent as the API's safe_prompt."`
	// PromptCacheKey groups requests that share a prompt prefix for prompt
	// caching, sent as the API's prompt_cache_key. Hits come back as the
	// response usage's CachedContentTokens.
	PromptCacheKey string `json:"promptCacheKey,omitempty" jsonschema_description:"Groups requests that share a prompt prefix for prompt caching, sent as the API's prompt_cache_key."`
	// ServiceTier selects the capacity the request is served from, sent as
	// the API's service_tier.
	ServiceTier ServiceTier `json:"serviceTier,omitempty" jsonschema:"enum=auto,enum=standard_only" jsonschema_description:"Capacity the request is served from: auto, or standard_only to stay on standard capacity."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts, MaxOutputTokens on the
// max_tokens Mistral reads, and the fields Mistral names differently than
// OpenAI, or that OpenAI does not have, ride as extra request fields.
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
	if c.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(*c.FrequencyPenalty)
	}
	if c.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*c.PresencePenalty)
	}
	if c.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(c.ReasoningEffort)
	}
	if c.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*c.ParallelToolCalls)
	}

	extra := map[string]any{}
	if c.Seed != nil {
		extra["random_seed"] = *c.Seed
	}
	if c.SafePrompt != nil {
		extra["safe_prompt"] = *c.SafePrompt
	}
	if c.PromptCacheKey != "" {
		extra["prompt_cache_key"] = c.PromptCacheKey
	}
	if c.ServiceTier != "" {
		extra["service_tier"] = string(c.ServiceTier)
	}
	compat_oai.AddExtraFields(params, extra)
}

// EmbedConfig is the per-request config for Mistral embedders. Mistral's
// embeddings request differs from OpenAI's: the output size is
// output_dimension rather than dimensions, and there is no user field. See
// https://docs.mistral.ai/api/endpoint/embeddings.
type EmbedConfig struct {
	// Dimensions is the number of dimensions the output embeddings should
	// have, for models that support choosing it, such as codestral-embed;
	// sent as the API's output_dimension.
	Dimensions int `json:"dimensions,omitempty" jsonschema:"minimum=1" jsonschema_description:"Number of dimensions the output embeddings should have, for models that support choosing it; sent as the API's output_dimension."`
	// EncodingFormat selects the encoding of the returned embeddings, "float"
	// (the default) or "base64".
	EncodingFormat openai.EmbeddingNewParamsEncodingFormat `json:"encodingFormat,omitempty" jsonschema:"enum=float,enum=base64" jsonschema_description:"Encoding of the returned embeddings, float (the default) or base64."`
	// APIKey overrides the plugin's API key for this request alone. Like
	// [compat_oai.RequestConfig.APIKey], it never serializes and can only be
	// set from a typed config in code.
	APIKey string `json:"-"`
	// Extra carries request body fields this config does not declare, sent
	// verbatim at the top level of the outgoing request and keyed by
	// Mistral's wire names; the contract is [compat_oai.RequestConfig.Extra]'s,
	// with the input the only field Genkit protects.
	Extra map[string]any `json:"extra,omitempty" jsonschema_description:"Request body fields to send verbatim beyond the declared ones, keyed by Mistral's wire names. A colliding key wins over the declared field. The input Genkit builds from the request is rejected."`
}

// ApplyToEmbedding implements [compat_oai.EmbedderConfig].
func (c EmbedConfig) ApplyToEmbedding(params *openai.EmbeddingNewParams) {
	if c.Dimensions > 0 {
		params.SetExtraFields(map[string]any{"output_dimension": c.Dimensions})
	}
	if c.EncodingFormat != "" {
		params.EncodingFormat = c.EncodingFormat
	}
}

// RequestAPIKey implements [compat_oai.EmbedderConfig].
func (c EmbedConfig) RequestAPIKey() string {
	return c.APIKey
}

// RequestExtra implements [compat_oai.EmbedderConfig].
func (c EmbedConfig) RequestExtra() map[string]any {
	return c.Extra
}

// Capability sets shared by the entries below. Mistral's current chat models
// call tools, take a tool choice, and constrain output to a JSON schema
// through response_format json_schema; they differ on whether they read
// images. Only object roots are advertised as constrained: whether Mistral
// takes an array or enum schema root is not documented, so those formats
// reach the model as prompt instructions.
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
	textOnly = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       false,
		ToolChoice:  true,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportAll,
	}
	// dynamicSupports is for a model the plugin knows nothing about, which
	// may predate json_schema support: a schema reaches it as prompt
	// instructions, with JSON mode on the wire.
	dynamicSupports = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       false,
		ToolChoice:  true,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportNone,
	}
)

// mediaTypes are the media Mistral's chat completions read through the
// content parts the plugin sends. Mistral reads PDFs as document_url chunks,
// which the plugin does not send, rather than as OpenAI file parts.
var mediaTypes = []string{"image/*"}

// supportedModels curates capabilities for Mistral's current chat models,
// by the aliases Mistral moves to each new snapshot. It is not the set of
// usable models: any chat model Mistral serves resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works, and a dated
// snapshot can be pinned with a config version. No versions are declared,
// since Mistral serves dated snapshots the plugin cannot enumerate, and an
// undeclared list leaves version pinning unconstrained.
//
// Catalog: https://docs.mistral.ai/getting-started/models
var supportedModels = map[string]ai.ModelOptions{
	// Mistral Large 3.
	"mistral-large-latest": {Label: "Mistral Large", Supports: &multimodal},
	// Mistral Large 4, in public preview; it always reasons.
	"mistral-large-4": {Label: "Mistral Large 4", Supports: &multimodal, Stage: ai.ModelStageUnstable},
	// Mistral Medium 3.5.
	"mistral-medium-latest": {Label: "Mistral Medium", Supports: &multimodal},
	// Mistral Small 4.
	"mistral-small-latest": {Label: "Mistral Small", Supports: &multimodal},
	"ministral-8b-latest":  {Label: "Ministral 8B", Supports: &multimodal},
	"ministral-3b-latest":  {Label: "Ministral 3B", Supports: &multimodal},
	"codestral-latest":     {Label: "Codestral", Supports: &textOnly},
}

// dynamicModelOptions is advertised for Mistral models that resolve
// dynamically rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &dynamicSupports,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// supportedEmbedders curates Mistral's embedders.
var supportedEmbedders = map[string]ai.EmbedderOptions{
	"mistral-embed": {
		Label:    "Mistral Embed",
		Supports: &ai.EmbedderSupports{Input: []string{"text"}},
	},
	// Codestral Embed defaults to 1536 dimensions and takes up to 3072
	// through EmbedConfig.Dimensions.
	"codestral-embed": {
		Label:      "Codestral Embed",
		Dimensions: 1536,
		Supports:   &ai.EmbedderSupports{Input: []string{"text"}},
	},
}

// Mistral configures the Mistral plugin.
type Mistral struct {
	// APIKey is the Mistral API key. If empty, MISTRAL_API_KEY is consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (MISTRAL_BASE_URL works
	// too). Options supplied here are applied after the plugin defaults, so
	// they win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a Mistral model, keyed by
	// model ID, bare or provider-prefixed. Every Mistral chat model already
	// works without an entry: known IDs carry curated capabilities and the
	// rest take the Mistral defaults. Supply an entry only to correct or
	// extend what the plugin resolves, most often for a model released after
	// this version of the plugin.
	//
	//	&mistral.Mistral{Models: map[string]ai.ModelOptions{
	//		"mistral-small-latest": {Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [Mistral.ListActions] advertises and [Mistral.ResolveAction]
	// builds, which is the way to describe a curated model differently: Init
	// has already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions
	// Embedders is [Mistral.Models] for embedders, keyed by embedder ID.
	Embedders map[string]ai.EmbedderOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (m *Mistral) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (m *Mistral) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("MISTRAL_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := m.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("MISTRAL_API_KEY")
	}
	if apiKey == "" {
		panic("mistral plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, m.Opts...)

	m.openAICompatible.Provider = provider
	m.openAICompatible.MediaTypes = mediaTypes
	m.openAICompatible.ListModels = listChatModels
	// Mistral's request schema is closed: it rejects stream_options, and it
	// reports usage on the final chunk of every stream unasked.
	m.openAICompatible.OmitStreamOptions = true
	// It also has no reasoning_content field: reasoning goes back as the
	// thinking chunk it arrived in.
	m.openAICompatible.ThinkingChunks = true
	m.openAICompatible.Opts = opts
	actions := m.openAICompatible.Init(ctx)

	for model := range supportedModels {
		actions = append(actions, m.newModel(model, m.modelOptions(model)))
	}
	for id := range supportedEmbedders {
		actions = append(actions, m.newEmbedder(id, m.embedderOptions(id)))
	}
	return actions
}

// newModel creates a Mistral model without registering it.
func (m *Mistral) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&m.openAICompatible, id, opts)
}

// newEmbedder creates a Mistral embedder without registering it.
func (m *Mistral) newEmbedder(id string, opts ai.EmbedderOptions) *ai.EmbedderAction {
	return compat_oai.NewEmbedderWithConfig[EmbedConfig](&m.openAICompatible, id, &opts)
}

// modelOptions returns the ModelOptions for a Mistral model ID: curated
// capabilities for a known model and the Mistral defaults for the rest, with
// an entry from [Mistral.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (m *Mistral) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, m.Models)
}

// embedderOptions is [Mistral.modelOptions] for embedders. An unknown ID has
// no curated entry and takes the caller's alone.
func (m *Mistral) embedderOptions(id string) ai.EmbedderOptions {
	opts := supportedEmbedders[id]
	if override, ok := internal.LookupOverride(m.Embedders, provider, id); ok {
		return opts.Overlay(override)
	}
	return opts
}

// ModelRef names a Mistral model and carries the config to generate with, so
// the config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(mistral.ModelRef("mistral-small-latest", &mistral.ChatConfig{
//		ReasoningEffort: mistral.ReasoningEffortHigh,
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// EmbedderRef names a Mistral embedder and carries the config to embed with.
// A nil config leaves the request's config unset.
//
//	ai.WithEmbedder(mistral.EmbedderRef("codestral-embed", &mistral.EmbedConfig{
//		Dimensions: 256,
//	}))
//
// id is the embedder ID, with or without the provider prefix.
func EmbedderRef(id string, config *EmbedConfig) ai.EmbedderRef {
	return ai.NewEmbedderRef(compat_oai.ActionName(provider, id), config)
}

// ListActions lists the chat models the configured Mistral endpoint exposes,
// described by the plugin's config schema and capabilities.
func (m *Mistral) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &m.openAICompatible, m.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Mistral endpoint,
// described by the plugin's config schema and capabilities.
func (m *Mistral) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&m.openAICompatible, atype, id, m.modelOptions)
}

// listChatModels lists the models Mistral serves through chat completions.
// Mistral's models endpoint also lists its embedding, OCR, moderation and
// transcription models, which a chat model action would only fail on, so a
// model whose card does not report the completion_chat capability is left
// out.
func listChatModels(ctx context.Context, client *openai.Client) ([]string, error) {
	var ids []string
	iter := client.Models.ListAutoPaging(ctx)
	for iter.Next() {
		model := iter.Current()
		var capabilities struct {
			CompletionChat bool `json:"completion_chat"`
		}
		if json.Unmarshal([]byte(model.JSON.ExtraFields["capabilities"].Raw()), &capabilities) == nil && capabilities.CompletionChat {
			ids = append(ids, model.ID)
		}
	}
	if err := iter.Err(); err != nil {
		return nil, compat_oai.WrapAPIError(err)
	}
	return ids, nil
}
