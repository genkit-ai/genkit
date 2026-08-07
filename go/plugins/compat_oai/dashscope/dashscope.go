// Copyright 2025 Google LLC
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

// Package dashscope provides a Genkit plugin for Alibaba Cloud's Qwen models,
// served through DashScope's OpenAI-compatible mode.
package dashscope

import (
	"context"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	provider = "dashscope"
	// defaultBaseURL is the shared international endpoint, used as a fallback
	// when neither BaseURL nor DASHSCOPE_BASE_URL is set. Works for standard
	// API keys; mainland-China accounts or workspace-dedicated domains
	// (Alibaba's recommended production setup) should override via BaseURL or
	// DASHSCOPE_BASE_URL. See https://help.aliyun.com/en/model-studio/base-url
	// and the package README.
	defaultBaseURL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

// ChatConfig is the per-request config for Qwen models served through
// DashScope's OpenAI-compatible mode: the common generation fields plus the
// DashScope-specific controls the mode accepts as extra request fields. See
// https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, from
	// 0 to 2, exclusive of both ends.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP is the nucleus sampling threshold, from 0 to 1, exclusive of both
	// ends.
	TopP *float64 `json:"topP,omitempty"`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
	// StopSequences stop generation when produced by the model.
	StopSequences []string `json:"stopSequences,omitempty"`
	// PresencePenalty penalizes tokens that have appeared at all, from -2.0 to
	// 2.0. DashScope's compatible mode documents no frequency penalty.
	PresencePenalty *float64 `json:"presencePenalty,omitempty"`
	// Seed makes generation reproducible across calls when set.
	Seed *int `json:"seed,omitempty"`
	// EnableThinking turns the thinking mode of hybrid Qwen models on or off,
	// sent as the API's enable_thinking.
	EnableThinking *bool `json:"enableThinking,omitempty"`
	// ThinkingBudget is the maximum number of tokens the model may think
	// with, sent as the API's thinking_budget; it requires EnableThinking.
	ThinkingBudget *int `json:"thinkingBudget,omitempty"`
	// EnableSearch lets the model consult web search, sent as the API's
	// enable_search.
	EnableSearch *bool `json:"enableSearch,omitempty"`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts and the DashScope controls
// ride as the mode's extra request fields.
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
	if c.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*c.PresencePenalty)
	}
	if c.Seed != nil {
		params.Seed = openai.Int(int64(*c.Seed))
	}

	extra := map[string]any{}
	if c.EnableThinking != nil {
		extra["enable_thinking"] = *c.EnableThinking
	}
	if c.ThinkingBudget != nil {
		extra["thinking_budget"] = *c.ThinkingBudget
	}
	if c.EnableSearch != nil {
		extra["enable_search"] = *c.EnableSearch
	}
	compat_oai.AddExtraFields(params, extra)
}

// Capability sets shared by the entries below. Forced tool-choice modes carry
// model- and thinking-mode-specific restrictions on Qwen, so no model
// advertises ToolChoice and tool selection is always automatic. Constrained
// generation is likewise absent: DashScope's response_format takes
// json_object only, not json_schema, so a schema reaches the model as prompt
// instructions. See https://www.alibabacloud.com/help/en/model-studio/json-mode.
var (
	textOnly = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
		Output:     []string{"text", "json"},
	}
	multimodal = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
		Output:     []string{"text", "json"},
	}
)

// supportedModels curates capabilities for well-known Qwen models. It is not
// the set of usable models: any Qwen model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works. Dated snapshots are
// folded into Versions rather than registered as separate models, matching the
// anthropic and openai subplugins.
//
// Catalog: https://www.alibabacloud.com/help/en/model-studio/models
// Confirmed against the live GET /compatible-mode/v1/models response.
var supportedModels = map[string]ai.ModelOptions{
	"qwen-flash": {
		Label:    "Qwen Flash",
		Supports: &textOnly,
		Versions: []string{"qwen-flash", "qwen-flash-2025-07-28"},
	},
	"qwen-plus": {
		Label:    "Qwen Plus",
		Supports: &textOnly,
		Versions: []string{"qwen-plus", "qwen-plus-2025-07-28", "qwen-plus-2025-09-11", "qwen-plus-2025-12-01"},
	},
	"qwen3.5-flash": {
		Label:    "Qwen 3.5 Flash",
		Supports: &multimodal,
		Versions: []string{"qwen3.5-flash", "qwen3.5-flash-2026-02-23"},
	},
	"qwen3.5-plus": {
		Label:    "Qwen 3.5 Plus",
		Supports: &multimodal,
		Versions: []string{"qwen3.5-plus", "qwen3.5-plus-2026-02-15"},
	},
	"qwen3.6-flash": {
		Label:    "Qwen 3.6 Flash",
		Supports: &multimodal,
		Versions: []string{"qwen3.6-flash", "qwen3.6-flash-2026-04-16"},
	},
	"qwen3.6-plus": {
		Label:    "Qwen 3.6 Plus",
		Supports: &multimodal,
		Versions: []string{"qwen3.6-plus", "qwen3.6-plus-2026-04-02"},
	},
	"qwen3.7-plus": {
		Label:    "Qwen 3.7 Plus",
		Supports: &multimodal,
		Versions: []string{"qwen3.7-plus", "qwen3.7-plus-2026-05-26"},
	},
	"qwen3.7-max": {
		Label:    "Qwen 3.7 Max",
		Supports: &textOnly,
		Versions: []string{"qwen3.7-max", "qwen3.7-max-2026-06-08", "qwen3.7-max-2026-05-20"},
	},
	"qwen3-max": {
		Label:    "Qwen 3 Max",
		Supports: &textOnly,
		Versions: []string{"qwen3-max", "qwen3-max-2026-01-23", "qwen3-max-2025-09-23", "qwen3-max-preview"},
	},
	"qwen3-vl-plus": {
		Label:    "Qwen 3 VL Plus",
		Supports: &multimodal,
		Versions: []string{"qwen3-vl-plus", "qwen3-vl-plus-2025-12-19", "qwen3-vl-plus-2025-09-23"},
	},
	"qwen3-coder-plus": {
		Label:    "Qwen 3 Coder Plus",
		Supports: &textOnly,
		Versions: []string{"qwen3-coder-plus", "qwen3-coder-plus-2025-07-22", "qwen3-coder-plus-2025-09-23"},
	},
}

// dynamicModelOptions is advertised for Qwen models that resolve dynamically
// rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &textOnly,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// DashScope configures the Alibaba Cloud DashScope (Qwen) plugin.
type DashScope struct {
	// APIKey is the DashScope API key. If empty, DASHSCOPE_API_KEY is consulted.
	APIKey string
	// BaseURL overrides the DashScope API endpoint. If empty, DASHSCOPE_BASE_URL,
	// and then the default international endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (d *DashScope) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (d *DashScope) Init(ctx context.Context) []api.Action {
	baseURL := d.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("DASHSCOPE_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := d.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DASHSCOPE_API_KEY")
	}
	if apiKey == "" {
		panic("dashscope plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, d.Opts...)

	d.openAICompatible.Provider = provider
	d.openAICompatible.Opts = opts
	compatActions := d.openAICompatible.Init(ctx)

	var actions []api.Action
	actions = append(actions, compatActions...)

	// define default models
	for model, modelOpts := range supportedModels {
		actions = append(actions, d.newModel(model, modelOpts))
	}

	return actions
}

// newModel creates a Qwen model without registering it.
func (d *DashScope) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&d.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Qwen model ID: curated
// capabilities for a known model and the Qwen defaults for the rest.
func modelOptions(id string) ai.ModelOptions {
	if opts, ok := supportedModels[id]; ok {
		return opts
	}
	return dynamicModelOptions
}

// ModelRef names a Qwen model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(dashscope.ModelRef("qwen-plus", &dashscope.ChatConfig{
//		EnableThinking: openai.Ptr(true),
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// RegisterModel registers a Qwen model with g and returns it. The plugin
// supplies the implementation; opts describes
// what the model supports, and a nil opts takes the capabilities the plugin
// resolves for that ID, curated for a known model and the Qwen defaults for
// the rest.
//
// Registering an ID that is already registered panics; Init registers every
// curated model and generating with an ID registers it on demand, so define
// a model before its first use or guard with [IsDefinedModel]. name is the
// model ID, bare or provider-prefixed.
func (d *DashScope) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return compat_oai.RegisterChatModel[ChatConfig](g, &d.openAICompatible, id, opts, modelOptions)
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [DashScope.RegisterModel]).
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return compat_oai.IsDefinedModel(g, provider, id)
}

// ListActions lists the models the configured DashScope endpoint exposes,
// described by the plugin's config schema and capabilities.
func (d *DashScope) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &d.openAICompatible, modelOptions)
}

// ResolveAction dynamically builds a model exposed by the DashScope endpoint,
// described by the plugin's config schema and capabilities.
func (d *DashScope) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&d.openAICompatible, atype, id, modelOptions)
}
