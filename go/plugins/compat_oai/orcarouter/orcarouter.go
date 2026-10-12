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

// Package orcarouter provides a Genkit plugin for OrcaRouter, a gateway that
// serves models from many providers behind one OpenAI-compatible endpoint.
//
// OrcaRouter is not a model vendor, so this plugin carries no model catalog.
// Every model resolves by name on demand, under an ID that keeps the upstream
// vendor's prefix:
//
//	ai.WithModelName("orcarouter/anthropic/claude-sonnet-4.5")
//
// A resolved model is described with permissive capabilities, which is what
// makes an arbitrary model usable without an entry per model. Correct a model
// whose real capabilities are narrower through [OrcaRouter.Models].
//
// See https://www.orcarouter.ai.
package orcarouter

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
	provider       = "orcarouter"
	defaultBaseURL = "https://api.orcarouter.ai/v1"
)

// ReasoningEffort is how hard a reasoning-capable model thinks before it
// answers. OrcaRouter normalizes the level across the vendors it fronts, so
// the same value reaches an OpenAI, an Anthropic, and a DeepSeek model.
//
// Which levels a model takes is the model's to decide: a level the upstream
// vendor does not offer is an error from OrcaRouter rather than from here.
type ReasoningEffort string

const (
	// ReasoningEffortLow is fast reasoning, for latency-sensitive work.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium is a balanced level.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh is deeper thinking, for hard multi-step problems.
	ReasoningEffortHigh ReasoningEffort = "high"
)

// ChatConfig is the per-request config for models served through OrcaRouter:
// the sampling fields the gateway accepts across the vendors it fronts. See
// https://docs.orcarouter.ai/api-reference/chat/create-a-chat-completion.
//
// Several documented request fields are deliberately absent. n asks for
// several completion choices and bills for all of them, while Genkit reads
// only the first. Anything undeclared still reaches the wire through
// [compat_oai.RequestConfig.Extra].
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 2."`
	// TopP is the nucleus sampling threshold, from 0 to 1.
	TopP *float64 `json:"topP,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Nucleus sampling threshold, from 0 to 1: only the tokens comprising the top P probability mass are considered."`
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
	// User identifies the end user a request is made for, which OrcaRouter
	// uses to isolate abuse to one user rather than the whole key.
	User string `json:"user,omitempty" jsonschema_description:"Identifies the end user a request is made for, which OrcaRouter uses to isolate abuse to one user rather than the whole key."`
	// ReasoningEffort adjusts how hard the model thinks, [ReasoningEffortLow]
	// to [ReasoningEffortHigh], sent as the API's reasoning_effort. It
	// applies to reasoning-capable models; others ignore it.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty" jsonschema:"enum=low,enum=medium,enum=high" jsonschema_description:"How hard the model thinks: low, medium, or high, for reasoning-capable models."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the fields the
// OpenAI schema already has land on their chat completion counterparts, and
// ReasoningEffort on the SDK's reasoning_effort. Every field here is part of
// the OpenAI-compatible surface OrcaRouter exposes, so nothing rides the
// extra map.
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
	if c.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(c.ReasoningEffort)
	}
}

// OrcaRouter configures the OrcaRouter plugin.
type OrcaRouter struct {
	// APIKey is the OrcaRouter API key. If empty, ORCAROUTER_API_KEY is
	// consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (ORCAROUTER_BASE_URL
	// works too). Options supplied here are applied after the plugin
	// defaults, so they win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a model, keyed by model
	// ID, bare or provider-prefixed. Every model already works without an
	// entry: OrcaRouter serves hundreds of models from dozens of vendors and
	// adds more regularly, so the plugin curates no catalog and describes
	// every model it resolves with the same deliberately permissive
	// capabilities. The two ways to be wrong are not symmetric: a capability
	// declared too narrow is refused by Genkit before the request is sent,
	// which blocks a model that would have worked, while one declared too
	// wide reaches OrcaRouter, which answers with the real reason the model
	// cannot serve it. Constrained output is the exception, left unclaimed on
	// purpose: a large share of the catalog lacks it natively, and unset,
	// Genkit falls back to putting the schema in the prompt, which every
	// model handles and which returns the same typed result.
	//
	// Supply an entry to correct what a model can actually do:
	//
	//	&orcarouter.OrcaRouter{Models: map[string]ai.ModelOptions{
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
	// "orcarouter/mistralai/mistral-7b-instruct".
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (o *OrcaRouter) Name() string {
	return provider
}

// Init implements genkit.Plugin. It registers no models: OrcaRouter's catalog
// is too large and too fast-moving to enumerate, so every model is resolved on
// demand by [OrcaRouter.ResolveAction].
func (o *OrcaRouter) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("ORCAROUTER_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := o.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ORCAROUTER_API_KEY")
	}
	if apiKey == "" {
		panic("orcarouter plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, o.Opts...)

	o.openAICompatible.Provider = provider
	o.openAICompatible.Opts = opts
	return o.openAICompatible.Init(ctx)
}

// modelOptions returns the ModelOptions for a model ID: the permissive
// defaults, with an entry from [OrcaRouter.Models] overlaid on them.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative.
func (o *OrcaRouter) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, nil, compat_oai.DefaultModelOptions(), o.Models)
}

// ModelRef names a model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(orcarouter.ModelRef("anthropic/claude-sonnet-4.5", &orcarouter.ChatConfig{
//		ReasoningEffort: orcarouter.ReasoningEffortHigh,
//	}))
//
// id is the model ID, with or without this plugin's provider prefix. It keeps
// the upstream vendor's prefix either way.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions returns no descriptors. OrcaRouter serves hundreds of models,
// and a descriptor carries a full copy of the request and response schemas,
// so listing the catalog would put megabytes on every reflection poll for a
// list nobody reads in full. Models stay reachable by name through
// [OrcaRouter.ResolveAction].
func (o *OrcaRouter) ListActions(ctx context.Context) []api.ActionDesc {
	return nil
}

// ResolveAction dynamically builds a model served by OrcaRouter, described by
// the plugin's config schema and capabilities. Any model ID the gateway
// serves resolves, whether or not this plugin has heard of it.
func (o *OrcaRouter) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&o.openAICompatible, atype, id, o.modelOptions)
}
