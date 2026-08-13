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

// Package anthropic provides a Genkit plugin for Claude models through
// Anthropic's OpenAI-compatible endpoint. Anthropic positions that endpoint
// for testing and comparison; the plugins/anthropic package speaks the native
// Anthropic API and is the primary way to use Claude models with Genkit.
package anthropic

import (
	"context"
	"net/url"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	provider = "anthropic"
	baseURL  = "https://api.anthropic.com/v1"
)

// ChatConfig is the per-request config for Claude models served through
// Anthropic's OpenAI-compatible endpoint. It carries the fields that endpoint
// honors; OpenAI fields Anthropic documents as ignored (penalties, log
// probabilities, seed, response_format) are deliberately absent. See
// https://platform.claude.com/docs/en/api/openai-sdk.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness, from 0 to 1; the
	// endpoint caps greater values at 1, so the schema stops where the
	// behavior does.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 1. The endpoint would cap greater values at 1 rather than honor them."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as
	// the API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_tokens."`
	// TopP is the nucleus sampling threshold. The endpoint documents no
	// range for it, so the schema declares none.
	TopP *float64 `json:"topP,omitempty" jsonschema_description:"Nucleus sampling threshold: only the tokens comprising the top P probability mass are considered."`
	// StopSequences stop generation when produced by the model; whitespace
	// stop sequences are not supported by the endpoint.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema_description:"Stop generation when produced by the model. Whitespace-only stop sequences are not supported by the endpoint."`
	// Thinking controls Claude's extended thinking.
	Thinking *ThinkingConfig `json:"thinking,omitempty" jsonschema_description:"Extended thinking controls. The OpenAI-compatible endpoint does not return the thinking content itself."`
}

// ThinkingConfig configures Claude's extended thinking through the
// OpenAI-compatible endpoint. The endpoint does not return the thinking
// content; the plugins/anthropic package does.
type ThinkingConfig struct {
	// Type turns thinking "enabled" or "disabled". It is not an enum in the
	// schema: the endpoint documents no closed set, and the native API's set
	// has grown (Claude 5 added adaptive thinking), so a list here would
	// reject a value Anthropic accepts.
	Type string `json:"type,omitempty" jsonschema_description:"Turns thinking enabled or disabled."`
	// BudgetTokens is the maximum number of tokens Claude may think with,
	// sent as the API's budget_tokens. Anthropic rejects budgets under 1,024
	// tokens, and the budget must stay below the request's max_tokens.
	BudgetTokens int `json:"budgetTokens,omitempty" jsonschema:"minimum=1024" jsonschema_description:"Maximum number of tokens Claude may think with, sent as the API's budget_tokens; at least 1,024, and less than maxOutputTokens."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the endpoint's
// generation fields land on their chat completion counterparts and thinking
// rides as the endpoint's thinking extra field.
func (c ChatConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
	c.ApplyVersion(params)

	if c.Temperature != nil {
		params.Temperature = openai.Float(*c.Temperature)
	}
	if c.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(c.MaxOutputTokens))
	}
	if c.TopP != nil {
		params.TopP = openai.Float(*c.TopP)
	}
	if len(c.StopSequences) > 0 {
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: c.StopSequences}
	}

	if c.Thinking != nil {
		thinking := map[string]any{}
		if c.Thinking.Type != "" {
			thinking["type"] = c.Thinking.Type
		}
		if c.Thinking.BudgetTokens > 0 {
			thinking["budget_tokens"] = c.Thinking.BudgetTokens
		}
		// An all-zero ThinkingConfig adds nothing rather than sending an
		// empty thinking object the endpoint could reject.
		if len(thinking) > 0 {
			compat_oai.AddExtraFields(params, map[string]any{"thinking": thinking})
		}
	}
}

// Capability sets shared by the entries below. The Claude 3 generation
// predates Anthropic's system-role support on the compatible endpoint.
// Models from the 4.5 generation on support structured outputs natively:
// the compatible endpoint takes response_format in its json_schema form and
// rejects every other form with a 400, so no set lists "json" among its
// native output formats and a schema-less JSON request rides the injected
// format instructions. See https://platform.claude.com/docs/en/api/openai-sdk.
var (
	multimodal = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		ToolChoice: true,
		SystemRole: true,
		Media:      true,
		Output:     []string{"text"},
	}
	multimodalConstrained = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		ToolChoice:  true,
		SystemRole:  true,
		Media:       true,
		Output:      []string{"text"},
		Constrained: ai.ConstrainedSupportAll,
	}
	multimodalNoSystemRole = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		ToolChoice: true,
		SystemRole: false,
		Media:      true,
		Output:     []string{"text"},
	}
)

// supportedModels curates capabilities for well-known Claude models. It is not
// the set of usable models: any Claude model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works. Dated snapshots are
// folded into Versions rather than registered as separate models. The Claude 4
// generations alias by bare name (claude-sonnet-4-5, never -latest), and IDs
// from the 4.6 generation on are dateless pinned snapshots.
//
// Catalog: https://platform.claude.com/docs/en/about-claude/models/overview
var supportedModels = map[string]ai.ModelOptions{
	"claude-fable-5": {
		Label:    "Claude Fable 5",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-fable-5"},
	},
	"claude-opus-5": {
		Label:    "Claude Opus 5",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-opus-5"},
	},
	"claude-sonnet-5": {
		Label:    "Claude Sonnet 5",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-sonnet-5"},
	},
	"claude-opus-4-8": {
		Label:    "Claude Opus 4.8",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-opus-4-8"},
	},
	"claude-opus-4-7": {
		Label:    "Claude Opus 4.7",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-opus-4-7"},
	},
	"claude-opus-4-6": {
		Label:    "Claude Opus 4.6",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-opus-4-6"},
	},
	"claude-sonnet-4-6": {
		Label:    "Claude Sonnet 4.6",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-sonnet-4-6"},
	},
	"claude-opus-4-5-20251101": {
		Label:    "Claude Opus 4.5",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-opus-4-5", "claude-opus-4-5-20251101"},
	},
	"claude-sonnet-4-5-20250929": {
		Label:    "Claude Sonnet 4.5",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-sonnet-4-5", "claude-sonnet-4-5-20250929"},
	},
	"claude-haiku-4-5-20251001": {
		Label:    "Claude Haiku 4.5",
		Supports: &multimodalConstrained,
		Versions: []string{"claude-haiku-4-5", "claude-haiku-4-5-20251001"},
	},
	"claude-opus-4-1-20250805": {
		Label:    "Claude Opus 4.1",
		Supports: &multimodal,
		Versions: []string{"claude-opus-4-1", "claude-opus-4-1-20250805"},
	},
	"claude-3-7-sonnet-20250219": {
		Label:    "Claude 3.7 Sonnet",
		Supports: &multimodal,
		Versions: []string{"claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"},
	},
	"claude-3-5-haiku-20241022": {
		Label:    "Claude 3.5 Haiku",
		Supports: &multimodal,
		Versions: []string{"claude-3-5-haiku-latest", "claude-3-5-haiku-20241022"},
	},
	"claude-3-5-sonnet-20240620": {
		Label:    "Claude 3.5 Sonnet",
		Supports: &multimodalNoSystemRole,
		Versions: []string{"claude-3-5-sonnet-20240620"},
	},
	"claude-3-opus-20240229": {
		Label:    "Claude 3 Opus",
		Supports: &multimodalNoSystemRole,
		Versions: []string{"claude-3-opus-latest", "claude-3-opus-20240229"},
	},
	"claude-3-haiku-20240307": {
		Label:    "Claude 3 Haiku",
		Supports: &multimodalNoSystemRole,
		Versions: []string{"claude-3-haiku-20240307"},
	},
}

// dynamicModelOptions is advertised for Claude models that resolve dynamically
// rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &multimodal,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

type Anthropic struct {
	// APIKey is the key requests are authenticated with. When empty, the
	// ANTHROPIC_API_KEY environment variable is used. A key set here or in
	// the environment authenticates both surfaces the plugin speaks to: the
	// chat endpoint as the bearer token and the native models list as
	// x-api-key.
	APIKey string

	// Opts are additional options for the underlying client, such as
	// [option.WithBaseURL] for a different endpoint; they are applied after
	// what the plugin composes, so they win on overlap. The credential
	// belongs in APIKey or the environment rather than here: a key inside
	// Opts is opaque to the plugin, so model listing cannot be authenticated
	// with it.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a Claude model, keyed by
	// model ID, bare or provider-prefixed. Every Claude model already works
	// without an entry: known IDs carry curated capabilities and the rest take
	// the Claude defaults. Supply an entry only to correct or extend what the
	// plugin resolves, most often for a model released after this version of
	// the plugin.
	//
	//	&anthropic.Anthropic{Models: map[string]ai.ModelOptions{
	//		"claude-sonnet-4-5-20250929": {Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [Anthropic.ListActions] advertises and [Anthropic.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (a *Anthropic) Name() string {
	return provider
}

func (a *Anthropic) Init(ctx context.Context) []api.Action {
	url := os.Getenv("ANTHROPIC_BASE_URL")
	if url == "" {
		url = baseURL
	}
	// ANTHROPIC_BASE_URL conventionally names the API origin without the
	// version segment: the native SDKs append v1 to every path themselves,
	// and environments like Claude Code export exactly that form. The OpenAI
	// SDK's paths carry no version, so restore the segment unless the value
	// already ends with it.
	url = strings.TrimSuffix(url, "/")
	if !strings.HasSuffix(url, "/v1") {
		url += "/v1"
	}
	a.Opts = append([]option.RequestOption{option.WithBaseURL(url)}, a.Opts...)

	apiKey := a.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey != "" {
		// The chat endpoint takes the OpenAI-style bearer token; the models
		// list is a native Anthropic endpoint on the same base URL and takes
		// x-api-key, so both ride along for listing to work.
		a.Opts = append([]option.RequestOption{
			option.WithAPIKey(apiKey),
			option.WithHeader("x-api-key", apiKey),
		}, a.Opts...)
	}

	// initialize OpenAICompatible
	a.openAICompatible.Provider = provider
	a.openAICompatible.Opts = a.Opts
	a.openAICompatible.ListModels = listClaudeModels
	actions := a.openAICompatible.Init(ctx)

	// define default models
	for model := range supportedModels {
		actions = append(actions, a.newModel(model, a.modelOptions(model)))
	}

	return actions
}

// newModel creates a Claude model without registering it.
func (a *Anthropic) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&a.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Claude model ID: curated
// capabilities for a known model and the Claude defaults for the rest, with
// an entry from [Anthropic.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first.
func (a *Anthropic) modelOptions(id string) ai.ModelOptions {
	return compat_oai.ModelOptionsFor(provider, id, supportedModels, dynamicModelOptions, a.Models)
}

// ModelRef names a Claude model and carries the config to generate with, so
// the config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(anthropic.ModelRef("claude-3-5-haiku-20241022", &anthropic.ChatConfig{
//		MaxOutputTokens: 1024,
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// DefineModel builds a Claude model and returns it, without registering it.
//
// Deprecated: describe the model through [Anthropic.Models] instead. This
// method builds the model and registers nothing, so the result carries only
// the model's name: generation resolves a model from that name and serves the
// request with the capabilities the plugin resolves, not the ones passed
// here. An entry in Models reaches both paths.
func (a *Anthropic) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return a.newModel(id, opts)
}

// Model returns a previously registered model.
//
// Deprecated: Generation resolves a model from its name, so looking one up
// first is rarely necessary: pass ai.WithModelName("anthropic/claude-3-5-haiku-20241022")
// or, to carry config with it, [ModelRef]. Use [genkit.LookupModel] when the
// action itself is what you need.
func (a *Anthropic) Model(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, compat_oai.ActionName(provider, id))
}

// listClaudeModels pages through the models list. It is a native Anthropic
// endpoint with its own cursoring, which the OpenAI-style lister does not
// follow: it would stop after the first page and silently list a fraction of
// the models. A page that fails to advance the cursor ends the walk, so an
// endpoint that misbehaves and repeats itself cannot loop the pager forever.
func listClaudeModels(ctx context.Context, client *openai.Client) ([]string, error) {
	var models []string
	after := ""
	for {
		var page struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
			HasMore bool   `json:"has_more"`
			LastID  string `json:"last_id"`
		}
		path := "models?limit=1000"
		if after != "" {
			path += "&after_id=" + url.QueryEscape(after)
		}
		if err := client.Get(ctx, path, nil, &page); err != nil {
			return nil, err
		}
		for _, m := range page.Data {
			models = append(models, m.ID)
		}
		if !page.HasMore || page.LastID == "" || page.LastID == after {
			return models, nil
		}
		after = page.LastID
	}
}

// ListActions lists the Claude models the configured endpoint exposes,
// described by the plugin's config schema and capabilities.
func (a *Anthropic) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &a.openAICompatible, a.modelOptions)
}

// ResolveAction dynamically builds a Claude model, described by the plugin's
// config schema and capabilities.
func (a *Anthropic) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&a.openAICompatible, atype, id, a.modelOptions)
}
