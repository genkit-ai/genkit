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
	// endpoint caps greater values at 1.
	Temperature *float64 `json:"temperature,omitempty"`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as
	// the API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
	// TopP is the nucleus sampling threshold.
	TopP *float64 `json:"topP,omitempty"`
	// StopSequences stop generation when produced by the model; whitespace
	// stop sequences are not supported by the endpoint.
	StopSequences []string `json:"stopSequences,omitempty"`
	// Thinking controls Claude's extended thinking.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// ThinkingConfig configures Claude's extended thinking through the
// OpenAI-compatible endpoint. The endpoint does not return the thinking
// content; the plugins/anthropic package does.
type ThinkingConfig struct {
	// Type turns thinking "enabled" or "disabled".
	Type string `json:"type,omitempty"`
	// BudgetTokens is the maximum number of tokens Claude may think with,
	// sent as the API's budget_tokens.
	BudgetTokens int `json:"budgetTokens,omitempty"`
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

// Supported models: https://docs.anthropic.com/en/docs/about-claude/models/all-models
var supportedModels = map[string]ai.ModelOptions{
	"claude-opus-4-1-20250805": {
		Label: "Claude 4.1 Opus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-opus-4-1-latest", "claude-opus-4-1-20250805"},
	},
	"claude-sonnet-4-5-20250929": {
		Label: "Claude 4.5 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-sonnet-4-5-latest", "claude-sonnet-4-5-20250929"},
	},
	"claude-haiku-4-5-20251001": {
		Label: "Claude 4.5 Haiku",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-haiku-4-5-latest", "claude-haiku-4-5-20251001"},
	},
	"claude-3-7-sonnet-20250219": {
		Label: "Claude 3.7 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"},
	},
	"claude-3-5-haiku-20241022": {
		Label: "Claude 3.5 Haiku",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: true,
			Media:      true,
		},
		Versions: []string{"claude-3-5-haiku-latest", "claude-3-5-haiku-20241022"},
	},
	"claude-3-5-sonnet-20240620": {
		Label: "Claude 3.5 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: false, // NOTE: This model does not support system role
			Media:      true,
		},
		Versions: []string{"claude-3-5-sonnet-20240620"},
	},
	"claude-3-opus-20240229": {
		Label: "Claude 3 Opus",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: false, // NOTE: This model does not support system role
			Media:      true,
		},
		Versions: []string{"claude-3-opus-latest", "claude-3-opus-20240229"},
	},
	"claude-3-haiku-20240307": {
		Label: "Claude 3 Haiku",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: false, // NOTE: This model does not support system role
			Media:      true,
		},
		Versions: []string{"claude-3-haiku-20240307"},
	},
}

// defaultClaudeOpts is the capability set advertised for Claude models not in
// the curated list.
var defaultClaudeOpts = ai.ModelOptions{
	Supports: &ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		ToolChoice: true,
		SystemRole: true,
		Media:      true,
	},
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

type Anthropic struct {
	Opts             []option.RequestOption
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
	a.Opts = append([]option.RequestOption{option.WithBaseURL(url)}, a.Opts...)

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
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
	for model, opts := range supportedModels {
		actions = append(actions, a.newModel(model, opts))
	}

	return actions
}

// newModel creates a Claude model without registering it.
func (a *Anthropic) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&a.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a Claude model ID: curated
// capabilities for a known model and the Claude defaults for the rest.
func modelOptions(id string) ai.ModelOptions {
	if opts, ok := supportedModels[id]; ok {
		return opts
	}
	return defaultClaudeOpts
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

// RegisterModel registers a Claude model with g and returns it. The plugin
// supplies the implementation; opts describes
// what the model supports, and a nil opts takes the capabilities the plugin
// resolves for that ID, curated for a known model and the Claude defaults
// for the rest.
//
// Registering an ID that is already registered panics; Init registers every
// curated model and generating with an ID registers it on demand, so define
// a model before its first use or guard with [IsDefinedModel]. name is the
// model ID, bare or provider-prefixed.
func (a *Anthropic) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return compat_oai.RegisterChatModel[ChatConfig](g, &a.openAICompatible, id, opts, modelOptions)
}

// DefineModel builds a Claude model and returns it, without registering it.
//
// Deprecated: use [Anthropic.RegisterModel]. The model this returns is not
// registered, and generation resolves a model from its name, so passing the
// result to ai.WithModel contributes only that name and serves the request
// with a model resolved from it instead. Registering it with
// [genkit.RegisterAction] is what makes these capabilities the ones used.
func (a *Anthropic) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return a.newModel(id, opts)
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [Anthropic.RegisterModel]).
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return compat_oai.IsDefinedModel(g, provider, id)
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
// the models.
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
		if !page.HasMore || page.LastID == "" {
			return models, nil
		}
		after = page.LastID
	}
}

// ListActions lists the Claude models the configured endpoint exposes,
// described by the plugin's config schema and capabilities.
func (a *Anthropic) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &a.openAICompatible, modelOptions)
}

// ResolveAction dynamically builds a Claude model, described by the plugin's
// config schema and capabilities.
func (a *Anthropic) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&a.openAICompatible, atype, id, modelOptions)
}
