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
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

const (
	provider       = "xai"
	defaultBaseURL = "https://api.x.ai/v1"
)

// ChatConfig is the per-request config for Grok models: the common generation
// fields plus the xAI-specific controls. See
// https://docs.x.ai/docs/api-reference.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP is the nucleus sampling threshold.
	TopP *float64 `json:"topP,omitempty"`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_completion_tokens; xAI deprecated max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
	// StopSequences stop generation when produced by the model, up to four.
	// Reasoning models do not support them.
	StopSequences []string `json:"stopSequences,omitempty"`
	// FrequencyPenalty penalizes tokens by their frequency so far, from -2.0
	// to 2.0. Reasoning models do not support it.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`
	// PresencePenalty penalizes tokens that have appeared at all, from -2.0 to
	// 2.0. Reasoning models do not support it.
	PresencePenalty *float64 `json:"presencePenalty,omitempty"`
	// LogProbs requests log probabilities for the output tokens.
	LogProbs *bool `json:"logProbs,omitempty"`
	// TopLogProbs is how many of the most likely tokens to return log
	// probabilities for at each position, from 0 to 8; it requires LogProbs.
	TopLogProbs *int `json:"topLogProbs,omitempty"`
	// Seed makes generation reproducible across calls when set.
	Seed *int `json:"seed,omitempty"`
	// ReasoningEffort adjusts how hard a reasoning-capable Grok model thinks:
	// "none", "low", "medium", or "high".
	ReasoningEffort string `json:"reasoningEffort,omitempty"`
	// SearchParameters lets the model consult live web and X results, sent as
	// the API's search_parameters.
	SearchParameters *SearchParameters `json:"searchParameters,omitempty"`
}

// SearchParameters configures xAI's live search. See
// https://docs.x.ai/docs/guides/live-search.
type SearchParameters struct {
	// Mode turns live search "on", "off", or "auto" (the model decides).
	Mode string `json:"mode,omitempty"`
	// ReturnCitations asks for the sources behind the answer, sent as the
	// API's return_citations.
	ReturnCitations *bool `json:"returnCitations,omitempty"`
	// FromDate is the earliest date to search, as YYYY-MM-DD; sent as the
	// API's from_date.
	FromDate string `json:"fromDate,omitempty"`
	// ToDate is the latest date to search, as YYYY-MM-DD; sent as the API's
	// to_date.
	ToDate string `json:"toDate,omitempty"`
	// MaxSearchResults caps how many results the model may consult, sent as
	// the API's max_search_results.
	MaxSearchResults *int `json:"maxSearchResults,omitempty"`
	// Sources narrows what is searched, defaulting to the web and X. Each
	// entry is a source object as xAI documents it, e.g.
	//
	//	map[string]any{"type": "x", "x_handles": []any{"xai"}}
	//
	// The entries pass through unchanged, so their keys are the API's own
	// rather than this package's camelCase.
	Sources []map[string]any `json:"sources,omitempty"`
}

// wireFields renders the search parameters as the API's snake_case object,
// leaving out what the config does not carry, or nil for a config that carries
// nothing.
func (s *SearchParameters) wireFields() map[string]any {
	if s == nil {
		return nil
	}
	fields := map[string]any{}
	if s.Mode != "" {
		fields["mode"] = s.Mode
	}
	if s.ReturnCitations != nil {
		fields["return_citations"] = *s.ReturnCitations
	}
	if s.FromDate != "" {
		fields["from_date"] = s.FromDate
	}
	if s.ToDate != "" {
		fields["to_date"] = s.ToDate
	}
	if s.MaxSearchResults != nil {
		fields["max_search_results"] = *s.MaxSearchResults
	}
	if len(s.Sources) > 0 {
		fields["sources"] = s.Sources
	}
	return fields
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

	// An all-zero SearchParameters adds nothing rather than sending an empty
	// search_parameters object the API could reject.
	if search := c.SearchParameters.wireFields(); len(search) > 0 {
		compat_oai.AddExtraFields(params, map[string]any{"search_parameters": search})
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
	"grok-4.5": {Label: "Grok 4.5", Supports: &multimodal},
	// The long-context model xAI documents ChatConfig.ReasoningEffort for.
	"grok-4.3":                     {Label: "Grok 4.3", Supports: &multimodal},
	"grok-4.20-0309-reasoning":     {Label: "Grok 4.20 Reasoning", Supports: &multimodal},
	"grok-4.20-0309-non-reasoning": {Label: "Grok 4.20 Non-Reasoning", Supports: &multimodal},
	// The agentic coding model, also served as "grok-code-fast-1". Outside the
	// Grok 4 family, so structured output and tools cannot be combined.
	"grok-build-0.1": {Label: "Grok Build 0.1", Supports: &multimodalNoToolConstraint},
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
	x.openAICompatible.Opts = opts
	actions := x.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, x.newModel(model, modelOpts))
	}
	return actions
}

// newModel creates a Grok model without registering it.
func (x *XAI) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&x.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Grok model ID: curated
// capabilities for a known model and the Grok defaults for the rest.
func modelOptions(id string) ai.ModelOptions {
	if opts, ok := supportedModels[id]; ok {
		return opts
	}
	return dynamicModelOptions
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

// RegisterModel registers a Grok model with g and returns it. The plugin
// supplies the implementation; opts describes
// what the model supports, and a nil opts takes the capabilities the plugin
// resolves for that ID, curated for a known model and the Grok defaults for
// the rest.
//
// Registering an ID that is already registered panics; Init registers every
// curated model and generating with an ID registers it on demand, so define
// a model before its first use or guard with [IsDefinedModel]. name is the
// model ID, bare or provider-prefixed.
func (x *XAI) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return compat_oai.RegisterChatModel[ChatConfig](g, &x.openAICompatible, id, opts, modelOptions)
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [XAI.RegisterModel]).
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return compat_oai.IsDefinedModel(g, provider, id)
}

// ListActions lists the models the configured xAI endpoint exposes, described
// by the plugin's config schema and capabilities.
func (x *XAI) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &x.openAICompatible, modelOptions)
}

// ResolveAction dynamically builds a model exposed by the xAI endpoint,
// described by the plugin's config schema and capabilities.
func (x *XAI) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&x.openAICompatible, atype, id, modelOptions)
}
