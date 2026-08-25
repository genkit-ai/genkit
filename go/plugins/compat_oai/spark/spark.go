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

// Package spark provides a Genkit plugin for iFLYTEK's Spark models.
package spark

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
	provider       = "spark"
	defaultBaseURL = "https://spark-api-open.xf-yun.com/v1"
)

// ChatConfig is the per-request config for iFLYTEK Spark models: the generation
// fields Spark's OpenAI-compatible HTTP API accepts. See
// https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html.
//
// Spark is reached over its OpenAI-compatible HTTP endpoint, authenticated with
// the single Bearer "API Password" from the iFLYTEK console, not the legacy
// WebSocket API's APPID/APIKey/APISecret triple.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2; Spark's default is 0.5.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 2. Spark's default is 0.5."`
	// TopP is the nucleus sampling threshold, from 0 to 1.
	TopP *float64 `json:"topP,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Nucleus sampling threshold, from 0 to 1."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_tokens."`
	// StopSequences stop generation when produced by the model.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema_description:"Stop generation when produced by the model."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts, with MaxOutputTokens on
// the max_tokens Spark reads.
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
}

// textOnly is the capability set every Spark model shares: text in, text or
// JSON out, and tools. No constrained generation is advertised: Spark's
// response_format takes json_object only, not json_schema, so a schema reaches
// the model as prompt instructions.
var textOnly = ai.ModelSupports{
	Multiturn:  true,
	Tools:      true,
	SystemRole: true,
	Media:      false,
	ToolChoice: true,
	Output:     []string{"text", "json"},
}

// supportedModels curates capabilities for well-known Spark models. It is not
// the set of usable models: any Spark model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works. No versions are
// declared, since the iFLYTEK HTTP API serves each model under one ID.
//
// Catalog: https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html
var supportedModels = map[string]ai.ModelOptions{
	"4.0Ultra":    {Label: "Spark 4.0 Ultra", Supports: &textOnly},
	"generalv3.5": {Label: "Spark Max", Supports: &textOnly},
	"max-32k":     {Label: "Spark Max-32K", Supports: &textOnly},
	"generalv3":   {Label: "Spark Pro", Supports: &textOnly},
	"pro-128k":    {Label: "Spark Pro-128K", Supports: &textOnly},
	"lite":        {Label: "Spark Lite", Supports: &textOnly},
}

// dynamicModelOptions is advertised for Spark models that resolve dynamically
// rather than appearing in supportedModels. A model iFLYTEK adds later is
// assumed to share the text-only shape.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &textOnly,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// Spark configures the iFLYTEK Spark plugin.
type Spark struct {
	// APIKey is the Spark HTTP service "API Password" — the single Bearer
	// credential shown as APIPassword in the iFLYTEK console, not the legacy
	// APPID/APIKey/APISecret triple. If empty, SPARK_API_KEY is consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (SPARK_BASE_URL works too).
	// Options supplied here are applied after the plugin defaults, so they win
	// on overlap.
	Opts []option.RequestOption
	// Models overrides what the plugin knows about a Spark model, keyed by
	// model ID, bare or provider-prefixed. Every Spark model already works
	// without an entry: known IDs carry curated capabilities and the rest take
	// the Spark defaults. Supply an entry only to correct or extend what the
	// plugin resolves, most often for a model released after this version of
	// the plugin.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (s *Spark) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (s *Spark) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("SPARK_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := s.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("SPARK_API_KEY")
	}
	if apiKey == "" {
		panic("spark plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, s.Opts...)

	s.openAICompatible.Provider = provider
	s.openAICompatible.Opts = opts
	actions := s.openAICompatible.Init(ctx)

	for model := range supportedModels {
		actions = append(actions, s.newModel(model, s.modelOptions(model)))
	}
	return actions
}

// newModel creates a Spark model without registering it.
func (s *Spark) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&s.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Spark model ID: curated
// capabilities for a known model and the Spark defaults for the rest, with an
// entry from [Spark.Models] overlaid on whichever applies.
func (s *Spark) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, s.Models)
}

// ModelRef names a Spark model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(spark.ModelRef("4.0Ultra", &spark.ChatConfig{
//		MaxOutputTokens: 1024,
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions lists the models the configured Spark endpoint exposes, described
// by the plugin's config schema and capabilities.
func (s *Spark) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &s.openAICompatible, s.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Spark endpoint,
// described by the plugin's config schema and capabilities.
func (s *Spark) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&s.openAICompatible, atype, id, s.modelOptions)
}
