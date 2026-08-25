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
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	provider = "dashscope"
	// defaultBaseURL is the shared international endpoint, used as a fallback
	// when DASHSCOPE_BASE_URL is not set. Works for standard API keys;
	// mainland-China accounts or workspace-dedicated domains (Alibaba's
	// recommended production setup) should override via DASHSCOPE_BASE_URL or
	// [option.WithBaseURL]. See https://help.aliyun.com/en/model-studio/base-url
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
	// 0 inclusive up to but not including 2.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,exclusiveMaximum=2" jsonschema_description:"Controls the degree of randomness in token selection, from 0 inclusive up to but not including 2."`
	// TopP is the nucleus sampling threshold, above 0 and up to 1 inclusive.
	TopP *float64 `json:"topP,omitempty" jsonschema:"exclusiveMinimum=0,maximum=1" jsonschema_description:"Nucleus sampling threshold, above 0 and up to 1 inclusive."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens; the default and the ceiling are both the model's
	// maximum output length.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_tokens. The default and the ceiling are both the model's maximum output length."`
	// StopSequences stop generation when produced by the model.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema_description:"Stop generation when produced by the model."`
	// PresencePenalty penalizes tokens that have appeared at all, from -2.0 to
	// 2.0. DashScope's compatible mode documents no frequency penalty.
	PresencePenalty *float64 `json:"presencePenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens that have appeared at all, from -2.0 to 2.0."`
	// Seed makes generation reproducible across calls when set, from 0 to
	// 2^31-1.
	Seed *int `json:"seed,omitempty" jsonschema:"minimum=0,maximum=2147483647" jsonschema_description:"Makes generation reproducible across calls when set, from 0 to 2^31-1."`
	// EnableThinking turns the thinking mode of hybrid Qwen models on or off,
	// sent as the API's enable_thinking.
	EnableThinking *bool `json:"enableThinking,omitempty" jsonschema_description:"Turns the thinking mode of hybrid Qwen models on or off, sent as the API's enable_thinking."`
	// ThinkingBudget is the maximum number of tokens the model may think
	// with, sent as the API's thinking_budget; it requires EnableThinking.
	// The default is the model's maximum chain-of-thought length.
	ThinkingBudget *int `json:"thinkingBudget,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens the model may think with, sent as the API's thinking_budget; requires enableThinking. Defaults to the model's maximum chain-of-thought length."`
	// EnableSearch lets the model consult web search, sent as the API's
	// enable_search.
	EnableSearch *bool `json:"enableSearch,omitempty" jsonschema_description:"Lets the model consult web search, sent as the API's enable_search."`
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
//
// The NoJSON sets serve the models whose capability tables say "Structured
// Outputs: Unsupported": response_format fails there, so they advertise text
// output only, the base sends no response_format, and the format instructions
// the framework injects carry a JSON request instead.
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
	textOnlyNoJSON = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
		Output:     []string{"text"},
	}
	multimodalNoJSON = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
		Output:     []string{"text"},
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
		Supports: &textOnlyNoJSON,
		Versions: []string{"qwen3.7-max", "qwen3.7-max-2026-06-08", "qwen3.7-max-2026-05-20"},
	},
	// The 2026-06-08 snapshot has capabilities of its own: DashScope documents
	// image and video input for it, which the floating qwen3.7-max and the May
	// snapshot do not take. It is registered standalone so media requests pass
	// validation, and stays in the base entry's Versions so pinning it there
	// keeps working.
	"qwen3.7-max-2026-06-08": {
		Label:    "Qwen 3.7 Max (2026-06-08)",
		Supports: &multimodalNoJSON,
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
		Supports: &textOnlyNoJSON,
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
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a different endpoint (DASHSCOPE_BASE_URL works
	// too). Options supplied here are applied after the plugin defaults, so
	// they win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a Qwen model, keyed by
	// model ID, bare or provider-prefixed. Every Qwen model already works
	// without an entry: known IDs carry curated capabilities and the rest take
	// the Qwen defaults. Supply an entry only to correct or extend what the
	// plugin resolves, most often for a model released after this version of
	// the plugin.
	//
	//	&dashscope.DashScope{Models: map[string]ai.ModelOptions{
	//		"qwen-plus": {Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [DashScope.ListActions] advertises and [DashScope.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (d *DashScope) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (d *DashScope) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("DASHSCOPE_BASE_URL")
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
	for model := range supportedModels {
		actions = append(actions, d.newModel(model, d.modelOptions(model)))
	}

	return actions
}

// newModel creates a Qwen model without registering it.
func (d *DashScope) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&d.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Qwen model ID: curated
// capabilities for a known model and the Qwen defaults for the rest, with
// an entry from [DashScope.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (d *DashScope) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, d.Models)
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

// ListActions lists the models the configured DashScope endpoint exposes,
// described by the plugin's config schema and capabilities.
func (d *DashScope) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &d.openAICompatible, d.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the DashScope endpoint,
// described by the plugin's config schema and capabilities.
func (d *DashScope) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&d.openAICompatible, atype, id, d.modelOptions)
}
