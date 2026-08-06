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

// Package deepseek provides a Genkit plugin for DeepSeek's models.
package deepseek

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
	provider       = "deepseek"
	defaultBaseURL = "https://api.deepseek.com"

	// ModelV4Flash is the fast DeepSeek V4 model.
	ModelV4Flash = "deepseek-v4-flash"
	// ModelV4Pro is the most capable DeepSeek V4 model.
	ModelV4Pro = "deepseek-v4-pro"
)

// ChatConfig is the per-request config for DeepSeek models: the generation
// fields DeepSeek accepts plus its thinking controls. See
// https://api-docs.deepseek.com/api/create-chat-completion.
//
// DeepSeek no longer supports the frequency and presence penalties, so those
// are deliberately absent.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection, up to
	// 2.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP is the nucleus sampling threshold, up to 1.
	TopP *float64 `json:"topP,omitempty"`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
	// StopSequences stop generation when produced by the model, up to sixteen.
	StopSequences []string `json:"stopSequences,omitempty"`
	// LogProbs requests log probabilities for the output tokens.
	LogProbs *bool `json:"logProbs,omitempty"`
	// TopLogProbs is how many of the most likely tokens to return log
	// probabilities for at each position, from 0 to 20; it requires LogProbs.
	TopLogProbs *int `json:"topLogProbs,omitempty"`
	// Thinking controls the thinking mode of DeepSeek models, which is on by
	// default; sent as the API's thinking field.
	Thinking *ThinkingConfig `json:"thinking,omitempty"`
}

// ThinkingConfig configures the thinking mode of DeepSeek models.
type ThinkingConfig struct {
	// Type turns thinking "enabled" or "disabled".
	Type string `json:"type,omitempty"`
	// ReasoningEffort adjusts how hard the model thinks: "low", "high", or
	// "max"; sent as the API's reasoning_effort inside the thinking object,
	// not as a top-level field.
	ReasoningEffort string `json:"reasoningEffort,omitempty"`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: the generation
// fields land on their chat completion counterparts, MaxOutputTokens on the
// max_tokens DeepSeek reads, and thinking rides as DeepSeek's extra request
// field.
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
	if c.LogProbs != nil {
		params.Logprobs = openai.Bool(*c.LogProbs)
	}
	if c.TopLogProbs != nil {
		params.TopLogprobs = openai.Int(int64(*c.TopLogProbs))
	}

	if c.Thinking != nil {
		thinking := map[string]any{}
		if c.Thinking.Type != "" {
			thinking["type"] = c.Thinking.Type
		}
		if c.Thinking.ReasoningEffort != "" {
			thinking["reasoning_effort"] = c.Thinking.ReasoningEffort
		}
		// An all-zero ThinkingConfig adds nothing rather than sending an
		// empty thinking object the API could reject.
		if len(thinking) > 0 {
			compat_oai.AddExtraFields(params, map[string]any{"thinking": thinking})
		}
	}
}

// Supported models: https://api-docs.deepseek.com/quick_start/pricing
var supportedModels = map[string]ai.ModelOptions{
	ModelV4Flash: newModelOptions("DeepSeek V4 Flash"),
	ModelV4Pro:   newModelOptions("DeepSeek V4 Pro"),
}

// newModelOptions builds the curated options entry for a DeepSeek model. Both
// V4 models take text and answer with text or JSON, call tools, and think, so
// the entries differ only by label. No versions are declared: DeepSeek serves
// each model under one ID whose snapshot moves underneath it, and an
// undeclared list leaves config version pinning unconstrained.
func newModelOptions(label string) ai.ModelOptions {
	return ai.ModelOptions{
		Label: label,
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
	}
}

// DeepSeek configures the DeepSeek plugin.
type DeepSeek struct {
	// APIKey is the DeepSeek API key. If empty, DEEPSEEK_API_KEY is consulted.
	APIKey string
	// BaseURL overrides the DeepSeek API endpoint. If empty, DEEPSEEK_BASE_URL
	// and then the default endpoint are used.
	BaseURL string
	// Opts contains additional OpenAI client request options. Options supplied
	// here are applied after the plugin defaults.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (d *DeepSeek) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (d *DeepSeek) Init(ctx context.Context) []api.Action {
	baseURL := d.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("DEEPSEEK_BASE_URL")
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := d.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	if apiKey == "" {
		panic("deepseek plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, d.Opts...)

	d.openAICompatible.Provider = provider
	d.openAICompatible.Opts = opts
	actions := d.openAICompatible.Init(ctx)

	for model, modelOpts := range supportedModels {
		actions = append(actions, d.newModel(model, modelOpts))
	}
	return actions
}

// newModel creates a DeepSeek model without registering it.
func (d *DeepSeek) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&d.openAICompatible, id, opts)
}

// modelOptions returns the ModelOptions for a DeepSeek model ID: curated
// capabilities for a known model and the DeepSeek defaults for the rest.
func modelOptions(id string) ai.ModelOptions {
	if opts, ok := supportedModels[id]; ok {
		return opts
	}
	return ai.ModelOptions{
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			SystemRole: true,
			Media:      false,
			ToolChoice: true,
			Output:     []string{"text", "json"},
		},
		Versions: []string{},
		Stage:    ai.ModelStageStable,
	}
}

// ModelRef names a DeepSeek model and carries the config to generate with, so
// the config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(deepseek.ModelRef(deepseek.ModelV4Pro, &deepseek.ChatConfig{
//		Thinking: &deepseek.ThinkingConfig{Type: "disabled"},
//	}))
//
// id is the model ID, with or without the provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// RegisterModel registers a DeepSeek model with g and returns it. The plugin
// supplies the implementation; opts describes
// what the model supports, and a nil opts takes the capabilities the plugin
// resolves for that ID, curated for a known model and the DeepSeek defaults
// for the rest.
//
// Registering an ID that is already registered panics; Init registers every
// curated model and generating with an ID registers it on demand, so define
// a model before its first use or guard with [IsDefinedModel]. name is the
// model ID, bare or provider-prefixed.
func (d *DeepSeek) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	return compat_oai.RegisterChatModel[ChatConfig](g, &d.openAICompatible, id, opts, modelOptions)
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [DeepSeek.RegisterModel]).
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return compat_oai.IsDefinedModel(g, provider, id)
}

// Model returns a previously registered model.
//
// Deprecated: Generation resolves a model from its name, so looking one up
// first is rarely necessary: pass ai.WithModelName("deepseek/deepseek-v4-pro")
// or, to carry config with it, [ModelRef]. Use [genkit.LookupModel] when the
// action itself is what you need.
func (d *DeepSeek) Model(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, compat_oai.ActionName(provider, id))
}

// ListActions lists the models the configured DeepSeek endpoint exposes,
// described by the plugin's config schema and capabilities.
func (d *DeepSeek) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &d.openAICompatible, modelOptions)
}

// ResolveAction dynamically builds a model exposed by the DeepSeek endpoint,
// described by the plugin's config schema and capabilities.
func (d *DeepSeek) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&d.openAICompatible, atype, id, modelOptions)
}
