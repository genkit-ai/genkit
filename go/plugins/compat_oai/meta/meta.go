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

// Package meta provides a Genkit plugin for models hosted by Meta Model API.
package meta

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
	provider       = "meta"
	defaultBaseURL = "https://api.meta.ai/v1"
)

// ReasoningEffort controls how much reasoning Muse Spark performs.
type ReasoningEffort string

const (
	// ReasoningEffortMinimal uses the least reasoning and lowest latency.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	// ReasoningEffortLow uses a small reasoning budget.
	ReasoningEffortLow ReasoningEffort = "low"
	// ReasoningEffortMedium balances reasoning depth and latency.
	ReasoningEffortMedium ReasoningEffort = "medium"
	// ReasoningEffortHigh uses a larger reasoning budget.
	ReasoningEffortHigh ReasoningEffort = "high"
	// ReasoningEffortXHigh uses the largest reasoning budget.
	ReasoningEffortXHigh ReasoningEffort = "xhigh"
)

// ChatConfig is the per-request config for Meta models. Fields not yet
// declared here can be sent through [compat_oai.RequestConfig.Extra].
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls randomness in token selection, from 0 to 2.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=2" jsonschema_description:"Controls randomness in token selection, from 0 to 2."`
	// TopP is the nucleus sampling threshold, from 0 to 1.
	TopP *float64 `json:"topP,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Nucleus sampling threshold, from 0 to 1."`
	// MaxOutputTokens is the maximum number of tokens to generate.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as max_completion_tokens."`
	// StopSequences stop generation when produced by the model, up to four.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema:"maxItems=4" jsonschema_description:"Stop generation when produced by the model, up to four sequences."`
	// ReasoningEffort controls how much reasoning Muse Spark performs, from
	// minimal through xhigh.
	ReasoningEffort ReasoningEffort `json:"reasoningEffort,omitempty" jsonschema:"enum=minimal,enum=low,enum=medium,enum=high,enum=xhigh" jsonschema_description:"How much reasoning Muse Spark performs: minimal, low, medium, high, or xhigh."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig].
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
	if c.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(c.ReasoningEffort)
	}
}

// multimodal is the capability set shared by Muse Spark models. Meta's chat
// API accepts response_format json_schema, so structured output is native.
var multimodal = ai.ModelSupports{
	Multiturn:   true,
	Tools:       true,
	SystemRole:  true,
	Media:       true,
	ToolChoice:  true,
	Output:      []string{"text", "json"},
	Constrained: ai.ConstrainedSupportAll,
}

// supportedModels curates the Muse Spark checkpoints exposed by Meta Model
// API. Model IDs are map keys rather than exported constants so old IDs can be
// removed without breaking the Go API. No Versions are declared: each
// checkpoint is already a complete model ID, and an empty list keeps version
// pinning open for future snapshots.
var supportedModels = map[string]ai.ModelOptions{
	"muse-spark-1.1":             {Label: "Muse Spark 1.1", Supports: &multimodal},
	"muse-spark-1.2":             {Label: "Muse Spark 1.2", Supports: &multimodal},
	"muse-spark-1.2-contributor": {Label: "Muse Spark 1.2 Contributor", Supports: &multimodal},
}

var dynamicModelOptions = ai.ModelOptions{
	Supports: &multimodal,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// Meta configures the Meta Model API plugin.
type Meta struct {
	// APIKey is the Meta Model API key. If empty, MODEL_API_KEY is consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (META_BASE_URL works too).
	// Options supplied here are applied after the plugin defaults, so they win
	// on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a Meta model, keyed by bare
	// or provider-prefixed model ID. Fields left at their zero value preserve
	// the catalog or dynamic defaults. Overrides apply consistently to Init,
	// ListActions, and ResolveAction.
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
		apiKey = os.Getenv("MODEL_API_KEY")
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
	m.openAICompatible.Opts = opts
	actions := m.openAICompatible.Init(ctx)
	for model := range supportedModels {
		actions = append(actions, m.newModel(model, m.modelOptions(model)))
	}
	return actions
}

func (m *Meta) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&m.openAICompatible, id, opts)
}

func (m *Meta) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, m.Models)
}

// ModelRef names a Meta model and carries its typed request config. id may be
// bare or provider-prefixed.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// ListActions lists models exposed by the configured Meta endpoint with the
// same config schema and capabilities used during registration and resolution.
func (m *Meta) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &m.openAICompatible, m.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the Meta endpoint with
// the same config schema and capabilities used during registration and listing.
func (m *Meta) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&m.openAICompatible, atype, id, m.modelOptions)
}
