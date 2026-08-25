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
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	provider       = "zai"
	defaultBaseURL = "https://api.z.ai/api/paas/v4"
)

// ThinkingType turns the chain-of-thought of GLM models on or off.
type ThinkingType string

const (
	// ThinkingTypeEnabled turns thinking on, which is Z.ai's default.
	ThinkingTypeEnabled ThinkingType = "enabled"
	// ThinkingTypeDisabled turns thinking off.
	ThinkingTypeDisabled ThinkingType = "disabled"
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
	// 0 to 1; the default varies by model.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 1. The default varies by model."`
	// TopP is the nucleus sampling threshold, from 0.01 to 1.
	TopP *float64 `json:"topP,omitempty" jsonschema:"minimum=0.01,maximum=1" jsonschema_description:"Nucleus sampling threshold, from 0.01 to 1."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens; Z.ai documents up to 131072.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1,maximum=131072" jsonschema_description:"Maximum number of tokens to generate, from 1 to 131072; sent as the API's max_tokens."`
	// StopSequences stop generation when produced by the model, up to four.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema:"maxItems=4" jsonschema_description:"Stop generation when produced by the model, up to four sequences."`
	// Thinking controls the chain-of-thought mode of GLM 4.5 and later
	// models, sent as the API's thinking field.
	Thinking *ThinkingConfig `json:"thinking,omitempty" jsonschema_description:"Chain-of-thought controls for GLM 4.5 and later models, sent as the API's thinking field."`
	// DoSample turns sampling off when set to false, making temperature and
	// TopP inert; sent as the API's do_sample.
	DoSample *bool `json:"doSample,omitempty" jsonschema_description:"Turns sampling off when false, making temperature and topP inert; sent as the API's do_sample."`
}

// ThinkingConfig configures the chain-of-thought mode of GLM models.
type ThinkingConfig struct {
	// Type turns thinking [ThinkingTypeEnabled] (the default) or
	// [ThinkingTypeDisabled].
	Type ThinkingType `json:"type,omitempty" jsonschema:"enum=enabled,enum=disabled" jsonschema_description:"Turns thinking enabled (the default) or disabled."`
	// ClearThinking controls whether the reasoning content is cleared from
	// the response, sent as the API's clear_thinking; Z.ai defaults it to
	// true.
	ClearThinking *bool `json:"clearThinking,omitempty" jsonschema_description:"Whether the reasoning content is cleared from the response, sent as the API's clear_thinking; defaults to true."`
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
			thinking["type"] = string(c.Thinking.Type)
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

// Capability sets shared by the entries below. No GLM model advertises
// ToolChoice, so the plugin always uses automatic tool selection and rejects a
// forced tool choice before the request goes out. Nor is constrained
// generation advertised: Z.ai's response_format takes text or json_object
// only, not json_schema, so a schema reaches the model as prompt
// instructions. See https://docs.z.ai/api-reference/llm/chat-completion.
var (
	textOnly = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
		ToolChoice: false,
		Output:     []string{"text", "json"},
	}
	multimodal = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
		ToolChoice: false,
		Output:     []string{"text", "json"},
	}
)

// supportedModels curates capabilities for well-known GLM models. It is not
// the set of usable models: any GLM model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works. No versions are
// declared, since Z.ai serves dated snapshots the plugin cannot enumerate,
// and an undeclared list leaves config version pinning unconstrained.
//
// Catalog: https://docs.z.ai/guides/llm/overview
var supportedModels = map[string]ai.ModelOptions{
	"glm-5.1":             {Label: "Z.ai GLM 5.1", Supports: &textOnly},
	"glm-5-turbo":         {Label: "Z.ai GLM 5 Turbo", Supports: &textOnly},
	"glm-5":               {Label: "Z.ai GLM 5", Supports: &textOnly},
	"glm-4.7":             {Label: "Z.ai GLM 4.7", Supports: &textOnly},
	"glm-4.7-flash":       {Label: "Z.ai GLM 4.7 Flash", Supports: &textOnly},
	"glm-4.7-flashx":      {Label: "Z.ai GLM 4.7 FlashX", Supports: &textOnly},
	"glm-4.6":             {Label: "Z.ai GLM 4.6", Supports: &textOnly},
	"glm-4.5":             {Label: "Z.ai GLM 4.5", Supports: &textOnly},
	"glm-4.5-air":         {Label: "Z.ai GLM 4.5 Air", Supports: &textOnly},
	"glm-4.5-x":           {Label: "Z.ai GLM 4.5 X", Supports: &textOnly},
	"glm-4.5-airx":        {Label: "Z.ai GLM 4.5 AirX", Supports: &textOnly},
	"glm-4.5-flash":       {Label: "Z.ai GLM 4.5 Flash", Supports: &textOnly},
	"glm-4-32b-0414-128k": {Label: "Z.ai GLM 4 32B 128K", Supports: &textOnly},

	// Vision models.
	"glm-5v-turbo":    {Label: "Z.ai GLM 5V Turbo", Supports: &multimodal},
	"glm-4.6v":        {Label: "Z.ai GLM 4.6V", Supports: &multimodal},
	"glm-4.6v-flash":  {Label: "Z.ai GLM 4.6V Flash", Supports: &multimodal},
	"glm-4.6v-flashx": {Label: "Z.ai GLM 4.6V FlashX", Supports: &multimodal},
	"glm-4.5v":        {Label: "Z.ai GLM 4.5V", Supports: &multimodal},
}

// dynamicModelOptions is advertised for GLM models that resolve dynamically
// rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &textOnly,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// ZAI configures the Z.ai GLM plugin.
type ZAI struct {
	// APIKey is the Z.ai API key. If empty, ZAI_API_KEY is consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (ZAI_BASE_URL works too).
	// Options supplied here are applied after the plugin defaults, so they
	// win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a GLM model, keyed by
	// model ID, bare or provider-prefixed. Every GLM model already works
	// without an entry: known IDs carry curated capabilities and the rest take
	// the GLM defaults. Supply an entry only to correct or extend what the
	// plugin resolves, most often for a model released after this version of
	// the plugin.
	//
	//	&zai.ZAI{Models: map[string]ai.ModelOptions{
	//		"glm-5.1": {Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [ZAI.ListActions] advertises and [ZAI.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (z *ZAI) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (z *ZAI) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("ZAI_BASE_URL")
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

	for model := range supportedModels {
		actions = append(actions, z.newModel(model, z.modelOptions(model)))
	}
	return actions
}

// newModel creates a GLM model without registering it.
func (z *ZAI) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&z.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a GLM model ID: curated
// capabilities for a known model and the GLM defaults for the rest, with
// an entry from [ZAI.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (z *ZAI) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, z.Models)
}

// ModelRef names a GLM model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(zai.ModelRef("glm-5", &zai.ChatConfig{
//		Thinking: &zai.ThinkingConfig{Type: "enabled"},
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions lists the models the configured Z.ai endpoint exposes,
// described by the plugin's config schema and capabilities.
func (z *ZAI) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &z.openAICompatible, z.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Z.ai endpoint,
// described by the plugin's config schema and capabilities.
func (z *ZAI) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&z.openAICompatible, atype, id, z.modelOptions)
}
