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
//
// SPDX-License-Identifier: Apache-2.0

// Package openrouter provides a Genkit plugin for OpenRouter, a gateway that
// serves models from many providers behind one OpenAI-compatible endpoint.
//
// OpenRouter is not a model vendor, so this plugin carries no model catalog.
// Every model resolves by name on demand, under an ID that keeps the
// upstream vendor's prefix:
//
//	ai.WithModelName("openrouter/anthropic/claude-sonnet-4.5")
//
// A resolved model is described with permissive capabilities, which is what
// makes an arbitrary model usable without an entry per model. Correct a model
// whose real capabilities are narrower through [OpenRouter.Models].
//
// See https://openrouter.ai/docs.
package openrouter

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	provider       = "openrouter"
	defaultBaseURL = "https://openrouter.ai/api/v1"
)

// ReasoningEffort is how much reasoning a reasoning-capable model spends
// before it answers. OpenRouter normalizes the level across vendors, so the
// same value reaches an OpenAI, an Anthropic, and a Gemini model.
//
// Which levels a model takes is the model's to decide: a level the upstream
// vendor does not offer is an error from OpenRouter rather than one from here.
type ReasoningEffort string

const (
	// ReasoningEffortNone disables reasoning. Models that always reason
	// reject it.
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortMinimal is the shallowest level that still reasons.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow is fast reasoning, for latency-sensitive work.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium is the level [ReasoningConfig.Enabled] selects on
	// its own.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh is deeper thinking, for hard multi-step problems.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh is deeper still, offered by a few vendors.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
	// ReasoningEffortMax is the deepest level any vendor offers.
	ReasoningEffortMax ReasoningEffort = "max"
)

// ServiceTier selects how a request is scheduled and billed upstream.
//
// It is declared here rather than reused from
// [openai.ChatCompletionNewParamsServiceTier] because the sets differ: a
// config advertising the SDK's type would omit "fast" and "scale".
type ServiceTier string

const (
	// ServiceTierAuto lets OpenRouter pick the tier.
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierDefault is standard scheduling and billing.
	ServiceTierDefault ServiceTier = "default"
	// ServiceTierFast buys faster scheduling.
	ServiceTierFast ServiceTier = "fast"
	// ServiceTierFlex trades latency for a lower rate.
	ServiceTierFlex ServiceTier = "flex"
	// ServiceTierPriority buys the fastest scheduling at the highest rate.
	ServiceTierPriority ServiceTier = "priority"
	// ServiceTierScale is the tier for sustained high volume.
	ServiceTierScale ServiceTier = "scale"
)

// ProviderSort is the metric upstream providers are ranked by, replacing
// OpenRouter's default load balancing.
type ProviderSort string

const (
	// ProviderSortPrice ranks the cheapest provider first.
	ProviderSortPrice ProviderSort = "price"
	// ProviderSortThroughput ranks the fastest provider first.
	ProviderSortThroughput ProviderSort = "throughput"
	// ProviderSortLatency ranks the provider with the lowest time to first
	// token first.
	ProviderSortLatency ProviderSort = "latency"
)

// DataCollection is whether a request may reach a provider that may store it.
type DataCollection string

const (
	// DataCollectionAllow permits any provider, which is the default.
	DataCollectionAllow DataCollection = "allow"
	// DataCollectionDeny restricts routing to providers that do not store
	// request data.
	DataCollectionDeny DataCollection = "deny"
)

// ReasoningConfig controls the reasoning a model does before it answers,
// sent as the API's reasoning object. See
// https://openrouter.ai/docs/use-cases/reasoning-tokens.
type ReasoningConfig struct {
	// Effort is how much reasoning to spend, from none to max. Vendors that
	// budget in tokens rather than levels take MaxTokens instead.
	Effort ReasoningEffort `json:"effort,omitempty" jsonschema:"enum=none,enum=minimal,enum=low,enum=medium,enum=high,enum=xhigh,enum=max" jsonschema_description:"How much reasoning to spend, from none to max. It overrides the model's own default."`
	// MaxTokens is the exact token budget to reason with, sent as the API's
	// max_tokens inside the reasoning object. It overrides Effort. Vendors
	// that take a budget usually reject one under 1,024 tokens, which is a
	// per-vendor limit rather than a documented API-wide one.
	MaxTokens int `json:"maxTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Exact token budget to reason with, which overrides effort. Vendors that take a budget usually reject one under 1,024 tokens."`
	// Exclude keeps the reasoning out of the response without stopping the
	// model from reasoning.
	Exclude *bool `json:"exclude,omitempty" jsonschema_description:"Keeps the reasoning out of the response without stopping the model from reasoning."`
	// Enabled turns reasoning on with the vendor's own defaults, which
	// OpenRouter treats as medium effort. Effort and MaxTokens imply it.
	Enabled *bool `json:"enabled,omitempty" jsonschema_description:"Turns reasoning on with the vendor's defaults, which OpenRouter treats as medium effort. Effort and maxTokens imply it."`
}

// wire returns the reasoning object to send, or nil when the config carries
// nothing. An empty object is not sent: it would enable reasoning on a model
// the caller only meant to configure conditionally.
func (c *ReasoningConfig) wire() map[string]any {
	if c == nil {
		return nil
	}
	reasoning := map[string]any{}
	if c.Effort != "" {
		reasoning["effort"] = string(c.Effort)
	}
	if c.MaxTokens > 0 {
		reasoning["max_tokens"] = c.MaxTokens
	}
	if c.Exclude != nil {
		reasoning["exclude"] = *c.Exclude
	}
	if c.Enabled != nil {
		reasoning["enabled"] = *c.Enabled
	}
	if len(reasoning) == 0 {
		return nil
	}
	return reasoning
}

// MaxPrice caps what a request may cost, in USD per million tokens for the
// token fields. A request that no provider can serve within the cap fails
// rather than falling back to a dearer one.
type MaxPrice struct {
	// Prompt is the cap on the input price, per million tokens.
	Prompt *float64 `json:"prompt,omitempty" jsonschema:"minimum=0" jsonschema_description:"Cap on the input price, in USD per million tokens."`
	// Completion is the cap on the output price, per million tokens.
	Completion *float64 `json:"completion,omitempty" jsonschema:"minimum=0" jsonschema_description:"Cap on the output price, in USD per million tokens."`
	// Request is the cap on the per-request price.
	Request *float64 `json:"request,omitempty" jsonschema:"minimum=0" jsonschema_description:"Cap on the per-request price, in USD."`
	// Image is the cap on the per-image price.
	Image *float64 `json:"image,omitempty" jsonschema:"minimum=0" jsonschema_description:"Cap on the per-image price, in USD."`
}

// wire returns the max_price object to send, or nil when nothing is capped.
func (p *MaxPrice) wire() map[string]any {
	if p == nil {
		return nil
	}
	price := map[string]any{}
	if p.Prompt != nil {
		price["prompt"] = *p.Prompt
	}
	if p.Completion != nil {
		price["completion"] = *p.Completion
	}
	if p.Request != nil {
		price["request"] = *p.Request
	}
	if p.Image != nil {
		price["image"] = *p.Image
	}
	if len(price) == 0 {
		return nil
	}
	return price
}

// ProviderRouting chooses which upstream providers may serve a request and in
// what order, sent as the API's provider object. This is the control the
// gateway exists for: the same model is served by several providers at
// different prices, speeds, and data policies.
//
// Sort, PreferredMinThroughput, and PreferredMaxLatency also have an object
// form (a partition, or per-percentile thresholds) that this struct does not
// declare. Send the whole provider object through [compat_oai.RequestConfig.Extra]
// to reach it: an extra wins over the field it collides with, so the raw
// object replaces this one wholesale.
//
// See https://openrouter.ai/docs/features/provider-routing.
type ProviderRouting struct {
	// Order lists provider slugs to try in order before any fallback.
	Order []string `json:"order,omitempty" jsonschema_description:"Provider slugs to try in order before any fallback, e.g. anthropic or together."`
	// Only restricts routing to these provider slugs.
	Only []string `json:"only,omitempty" jsonschema_description:"Restricts routing to these provider slugs."`
	// Ignore skips these provider slugs.
	Ignore []string `json:"ignore,omitempty" jsonschema_description:"Skips these provider slugs."`
	// AllowFallbacks lets another provider serve the request when the chosen
	// ones fail. It defaults to true; false fails the request instead.
	AllowFallbacks *bool `json:"allowFallbacks,omitempty" jsonschema_description:"Lets another provider serve the request when the chosen ones fail. Defaults to true; false fails the request instead."`
	// RequireParameters routes only to providers that honor every parameter
	// the request carries, rather than to one that would ignore some.
	RequireParameters *bool `json:"requireParameters,omitempty" jsonschema_description:"Routes only to providers that honor every parameter the request carries, rather than to one that would ignore some."`
	// DataCollection restricts routing to providers that do not store request
	// data when set to deny.
	DataCollection DataCollection `json:"dataCollection,omitempty" jsonschema:"enum=allow,enum=deny" jsonschema_description:"Restricts routing to providers that do not store request data when set to deny. Defaults to allow."`
	// ZDR restricts routing to zero data retention endpoints.
	ZDR *bool `json:"zdr,omitempty" jsonschema_description:"Restricts routing to zero data retention endpoints."`
	// Sort ranks providers by one metric instead of OpenRouter's default load
	// balancing.
	Sort ProviderSort `json:"sort,omitempty" jsonschema:"enum=price,enum=throughput,enum=latency" jsonschema_description:"Ranks providers by price, throughput, or latency instead of the default load balancing."`
	// Quantizations restricts routing to providers serving the model at these
	// quantization levels, e.g. int4, int8, fp8, fp16, bf16, or fp32. The
	// set is not enumerated in the schema: OpenRouter adds levels as hardware
	// gains them, and a closed list here would reject a level it accepts.
	Quantizations []string `json:"quantizations,omitempty" jsonschema_description:"Restricts routing to providers serving the model at these quantization levels, e.g. int4, int8, fp8, fp16, bf16, or fp32."`
	// MaxPrice caps what the request may cost.
	MaxPrice *MaxPrice `json:"maxPrice,omitempty" jsonschema_description:"Caps what the request may cost. A request no provider can serve within the cap fails rather than falling back to a dearer one."`
	// PreferredMinThroughput deprioritizes providers below this many output
	// tokens per second. They stay eligible as a fallback.
	PreferredMinThroughput *float64 `json:"preferredMinThroughput,omitempty" jsonschema:"minimum=0" jsonschema_description:"Deprioritizes providers below this many output tokens per second. They stay eligible as a fallback."`
	// PreferredMaxLatency deprioritizes providers slower than this many
	// seconds to first token. They stay eligible as a fallback.
	PreferredMaxLatency *float64 `json:"preferredMaxLatency,omitempty" jsonschema:"minimum=0" jsonschema_description:"Deprioritizes providers slower than this many seconds to first token. They stay eligible as a fallback."`
}

// wire returns the provider object to send, or nil when no preference is set.
func (p *ProviderRouting) wire() map[string]any {
	if p == nil {
		return nil
	}
	routing := map[string]any{}
	if len(p.Order) > 0 {
		routing["order"] = p.Order
	}
	if len(p.Only) > 0 {
		routing["only"] = p.Only
	}
	if len(p.Ignore) > 0 {
		routing["ignore"] = p.Ignore
	}
	if p.AllowFallbacks != nil {
		routing["allow_fallbacks"] = *p.AllowFallbacks
	}
	if p.RequireParameters != nil {
		routing["require_parameters"] = *p.RequireParameters
	}
	if p.DataCollection != "" {
		routing["data_collection"] = string(p.DataCollection)
	}
	if p.ZDR != nil {
		routing["zdr"] = *p.ZDR
	}
	if p.Sort != "" {
		routing["sort"] = string(p.Sort)
	}
	if len(p.Quantizations) > 0 {
		routing["quantizations"] = p.Quantizations
	}
	if price := p.MaxPrice.wire(); price != nil {
		routing["max_price"] = price
	}
	if p.PreferredMinThroughput != nil {
		routing["preferred_min_throughput"] = *p.PreferredMinThroughput
	}
	if p.PreferredMaxLatency != nil {
		routing["preferred_max_latency"] = *p.PreferredMaxLatency
	}
	if len(routing) == 0 {
		return nil
	}
	return routing
}

// ChatConfig is the per-request config for models served through OpenRouter:
// the sampling fields the gateway normalizes across vendors, plus the routing
// controls that are the reason to route through it at all. See
// https://openrouter.ai/docs/api-reference/chat-completion.
//
// Several documented request fields are deliberately absent. n asks for
// several completion choices and bills for all of them, while Genkit reads
// only the first. route and usage are deprecated by OpenRouter and have no
// effect. cache_control is a message annotation rather than a request field,
// so it has no home in a config. Anything undeclared still reaches the wire
// through [compat_oai.RequestConfig.Extra].
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 2."`
	// TopP is the nucleus sampling threshold, from 0 to 1.
	TopP *float64 `json:"topP,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Nucleus sampling threshold, from 0 to 1: only the tokens comprising the top P probability mass are considered."`
	// TopK limits sampling to the K most likely tokens, sent as the API's
	// top_k. 0, the default, applies no limit. Some vendors ignore it.
	TopK *int `json:"topK,omitempty" jsonschema:"minimum=0" jsonschema_description:"Limits sampling to the K most likely tokens, sent as the API's top_k. 0, the default, applies no limit."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as
	// the API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_tokens."`
	// StopSequences stop generation when produced by the model, up to four.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema:"maxItems=4" jsonschema_description:"Stop generation when produced by the model, up to four sequences."`
	// FrequencyPenalty penalizes tokens by their frequency so far, from -2.0
	// to 2.0.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens by their frequency so far, from -2.0 to 2.0."`
	// PresencePenalty penalizes tokens that have appeared at all, from -2.0
	// to 2.0.
	PresencePenalty *float64 `json:"presencePenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens that have appeared at all, from -2.0 to 2.0."`
	// RepetitionPenalty penalizes tokens by whether they appeared in the
	// input, from 0 to 2, sent as the API's repetition_penalty. 1 is neutral.
	RepetitionPenalty *float64 `json:"repetitionPenalty,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Penalizes tokens by whether they appeared in the input, from 0 to 2, sent as the API's repetition_penalty. 1 is neutral."`
	// MinP is the minimum probability a token needs relative to the most
	// likely one, from 0 to 1, sent as the API's min_p.
	MinP *float64 `json:"minP,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Minimum probability a token needs relative to the most likely one, from 0 to 1, sent as the API's min_p."`
	// TopA filters tokens by a threshold scaled from the most likely token's
	// probability, from 0 to 1, sent as the API's top_a.
	TopA *float64 `json:"topA,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Filters tokens by a threshold scaled from the most likely token's probability, from 0 to 1, sent as the API's top_a."`
	// Seed makes generation reproducible across calls when set, on a
	// best-effort basis.
	Seed *int `json:"seed,omitempty" jsonschema_description:"Makes generation reproducible across calls when set, on a best-effort basis."`
	// LogProbs requests log probabilities for the output tokens.
	LogProbs *bool `json:"logProbs,omitempty" jsonschema_description:"Requests log probabilities for the output tokens."`
	// TopLogProbs is how many of the most likely tokens to return log
	// probabilities for at each position, from 0 to 20; it requires LogProbs.
	TopLogProbs *int `json:"topLogProbs,omitempty" jsonschema:"minimum=0,maximum=20" jsonschema_description:"How many of the most likely tokens to return log probabilities for at each position, from 0 to 20; requires logProbs."`
	// ParallelToolCalls lets the model request several tool calls in one
	// response, which it may do by default. It applies to a request that
	// carries tools.
	ParallelToolCalls *bool `json:"parallelToolCalls,omitempty" jsonschema_description:"Lets the model request several tool calls in one response, which it may do by default; false caps it at one call per response."`
	// User identifies the end user a request is made for, which OpenRouter
	// uses to isolate abuse to one user rather than the whole key.
	User string `json:"user,omitempty" jsonschema_description:"Identifies the end user a request is made for, which OpenRouter uses to isolate abuse to one user rather than the whole key."`

	// Models lists further models to fall back to, in order, when the
	// requested one is unavailable, rate-limited, or refuses. The model the
	// request names is tried first.
	Models []string `json:"models,omitempty" jsonschema_description:"Further models to fall back to, in order, when the requested one is unavailable, rate-limited, or refuses. The requested model is tried first."`
	// Provider chooses which upstream providers may serve the request.
	Provider *ProviderRouting `json:"provider,omitempty" jsonschema_description:"Chooses which upstream providers may serve the request, and in what order."`
	// Reasoning controls the reasoning the model does before answering.
	Reasoning *ReasoningConfig `json:"reasoning,omitempty" jsonschema_description:"Controls the reasoning the model does before answering."`
	// Plugins enables OpenRouter's request plugins, such as web search and
	// file parsing, each an object with an id and that plugin's own options:
	//
	//	Plugins: []map[string]any{{"id": "web", "max_results": 3}}
	//
	// They are sent verbatim rather than typed: the roster and each plugin's
	// options change on OpenRouter's schedule, and a struct here would reject
	// an option the gateway accepts. See
	// https://openrouter.ai/docs/features/web-search.
	Plugins []map[string]any `json:"plugins,omitempty" jsonschema_description:"OpenRouter request plugins such as web search and file parsing, each an object with an id and that plugin's own options, e.g. {\"id\": \"web\", \"max_results\": 3}."`
	// Transforms are the prompt transforms to apply, currently "middle-out",
	// which compresses a prompt that would overflow the model's context by
	// dropping from the middle.
	Transforms []string `json:"transforms,omitempty" jsonschema_description:"Prompt transforms to apply, currently middle-out, which compresses a prompt that would overflow the model's context by dropping from the middle."`
	// SessionID groups related requests so they keep reaching the same
	// upstream provider, sent as the API's session_id. It is what keeps a
	// multi-turn conversation on one provider's prompt cache.
	SessionID string `json:"sessionId,omitempty" jsonschema_description:"Groups related requests so they keep reaching the same upstream provider, sent as the API's session_id. It keeps a multi-turn conversation on one provider's prompt cache."`
	// ServiceTier selects how the request is scheduled and billed upstream.
	ServiceTier ServiceTier `json:"serviceTier,omitempty" jsonschema:"enum=auto,enum=default,enum=fast,enum=flex,enum=priority,enum=scale" jsonschema_description:"How the request is scheduled and billed upstream: auto, default, fast, flex, priority, or scale."`
	// Metadata attaches key-value pairs to the request, readable later on
	// OpenRouter's activity pages. It takes up to 16 pairs, with keys up to
	// 64 characters and values up to 512.
	Metadata map[string]string `json:"metadata,omitempty" jsonschema_description:"Key-value pairs attached to the request, readable later on OpenRouter's activity pages. Up to 16 pairs, keys up to 64 characters and values up to 512."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the fields the
// OpenAI schema already has land on their chat completion counterparts, and
// the sampling knobs and routing controls OpenRouter adds ride as extra
// request fields.
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
	if c.Seed != nil {
		params.Seed = openai.Int(int64(*c.Seed))
	}
	if c.LogProbs != nil {
		params.Logprobs = openai.Bool(*c.LogProbs)
	}
	if c.TopLogProbs != nil {
		params.TopLogprobs = openai.Int(int64(*c.TopLogProbs))
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
	if len(c.Metadata) > 0 {
		params.Metadata = c.Metadata
	}

	extra := map[string]any{}
	if c.TopK != nil {
		extra["top_k"] = *c.TopK
	}
	if c.RepetitionPenalty != nil {
		extra["repetition_penalty"] = *c.RepetitionPenalty
	}
	if c.MinP != nil {
		extra["min_p"] = *c.MinP
	}
	if c.TopA != nil {
		extra["top_a"] = *c.TopA
	}
	if len(c.Models) > 0 {
		extra["models"] = c.Models
	}
	if len(c.Transforms) > 0 {
		extra["transforms"] = c.Transforms
	}
	if len(c.Plugins) > 0 {
		extra["plugins"] = c.Plugins
	}
	if c.SessionID != "" {
		extra["session_id"] = c.SessionID
	}
	if routing := c.Provider.wire(); routing != nil {
		extra["provider"] = routing
	}
	if reasoning := c.Reasoning.wire(); reasoning != nil {
		extra["reasoning"] = reasoning
	}
	compat_oai.AddExtraFields(params, extra)
}

// OpenRouter configures the OpenRouter plugin.
type OpenRouter struct {
	// APIKey is the OpenRouter API key. If empty, OPENROUTER_API_KEY is
	// consulted.
	APIKey string
	// SiteURL identifies the calling application on OpenRouter's rankings,
	// sent as the HTTP-Referer header. It is optional and affects
	// attribution only.
	SiteURL string
	// AppName is the title the calling application appears under on
	// OpenRouter's rankings, sent as the X-Title header. It is optional and
	// affects attribution only.
	AppName string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (OPENROUTER_BASE_URL
	// works too). Options supplied here are applied after the plugin
	// defaults, so they win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a model, keyed by model
	// ID, bare or provider-prefixed. Every model already works without an
	// entry: OpenRouter serves hundreds of models from dozens of vendors and
	// adds more weekly, so the plugin curates no catalog and describes every
	// model it resolves with the same deliberately permissive capabilities.
	// The two ways to be wrong are not symmetric: a capability declared too
	// narrow is refused by Genkit before the request is sent, which blocks a
	// model that would have worked, while one declared too wide reaches
	// OpenRouter, which answers with the real reason the model cannot serve
	// it. Constrained output is the exception, left unclaimed on purpose:
	// a large share of the catalog lacks it natively, and unset, Genkit
	// falls back to putting the schema in the prompt, which every model
	// handles and which returns the same typed result.
	//
	// Supply an entry to correct what a model can actually do:
	//
	//	&openrouter.OpenRouter{Models: map[string]ai.ModelOptions{
	//		// A text-only model, so Genkit refuses media locally rather
	//		// than paying for the upstream rejection.
	//		"mistralai/mistral-7b-instruct": {Supports: &ai.ModelSupports{
	//			Multiturn: true, Tools: true, SystemRole: true,
	//		}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the rest. The model ID
	// keeps the upstream vendor's prefix, so it contains a slash; the
	// optional provider prefix is this plugin's own, as in
	// "openrouter/mistralai/mistral-7b-instruct".
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (o *OpenRouter) Name() string {
	return provider
}

// Init implements genkit.Plugin. It registers no models: OpenRouter's catalog
// is too large and too fast-moving to enumerate, so every model is resolved on
// demand by [OpenRouter.ResolveAction].
func (o *OpenRouter) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("OPENROUTER_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := o.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_API_KEY")
	}
	if apiKey == "" {
		panic("openrouter plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	// Attribution only: OpenRouter lists the calling application under these
	// on its public rankings, and omitting them changes nothing else.
	if o.SiteURL != "" {
		opts = append(opts, option.WithHeader("HTTP-Referer", o.SiteURL))
	}
	if o.AppName != "" {
		opts = append(opts, option.WithHeader("X-Title", o.AppName))
	}
	opts = append(opts, o.Opts...)

	o.openAICompatible.Provider = provider
	o.openAICompatible.Opts = opts
	return o.openAICompatible.Init(ctx)
}

// modelOptions returns the ModelOptions for a model ID: the permissive
// defaults, with an entry from [OpenRouter.Models] overlaid on them.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative.
func (o *OpenRouter) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, nil, compat_oai.DefaultModelOptions(), o.Models)
}

// ModelRef names a model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(openrouter.ModelRef("anthropic/claude-sonnet-4.5", &openrouter.ChatConfig{
//		Provider: &openrouter.ProviderRouting{Sort: openrouter.ProviderSortThroughput},
//	}))
//
// id is the model ID, with or without this plugin's provider prefix. It keeps
// the upstream vendor's prefix either way.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions returns no descriptors. OpenRouter serves hundreds of models,
// and a descriptor carries a full copy of the request and response schemas,
// so listing the catalog would put megabytes on every reflection poll for a
// list nobody reads in full. Models stay reachable by name through
// [OpenRouter.ResolveAction].
func (o *OpenRouter) ListActions(ctx context.Context) []api.ActionDesc {
	return nil
}

// ResolveAction dynamically builds a model served by OpenRouter, described by
// the plugin's config schema and capabilities. Any model ID the gateway
// serves resolves, whether or not this plugin has heard of it.
func (o *OpenRouter) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&o.openAICompatible, atype, id, o.modelOptions)
}
