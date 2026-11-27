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

// Package nvidia provides a Genkit plugin for models served by NVIDIA NIM,
// either NVIDIA's hosted API or a self-hosted NIM.
package nvidia

import (
	"context"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	provider       = "nvidia"
	defaultBaseURL = "https://integrate.api.nvidia.com/v1"
)

// ChatConfig is the per-request config for NIM chat models: the generation
// fields of NIM's OpenAI-compatible chat completions API. See
// https://docs.api.nvidia.com/nim/reference/llm-apis.
//
// Accepted ranges vary by model, so the schema enforces only the limits the
// API defines for every model and leaves the rest to the descriptions.
type ChatConfig struct {
	compat_oai.RequestConfig

	// Temperature controls the degree of randomness in token selection; the
	// accepted range and the default vary by model.
	Temperature *float64 `json:"temperature,omitempty" jsonschema:"minimum=0" jsonschema_description:"Controls the degree of randomness in token selection. The accepted range and the default vary by model."`
	// TopP is the nucleus sampling threshold, from 0 to 1.
	TopP *float64 `json:"topP,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Nucleus sampling threshold, from 0 to 1."`
	// MaxOutputTokens is the maximum number of tokens to generate, sent as the
	// API's max_tokens. The ceiling varies by model.
	MaxOutputTokens int `json:"maxOutputTokens,omitempty" jsonschema:"minimum=1" jsonschema_description:"Maximum number of tokens to generate, sent as the API's max_tokens. The ceiling varies by model."`
	// StopSequences stop generation when produced by the model.
	StopSequences []string `json:"stopSequences,omitempty" jsonschema_description:"Stop generation when produced by the model."`
	// FrequencyPenalty penalizes tokens by how often they have appeared so
	// far, from -2 to 2.
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens by how often they have appeared so far, from -2 to 2."`
	// PresencePenalty penalizes tokens that have appeared at all so far, from
	// -2 to 2.
	PresencePenalty *float64 `json:"presencePenalty,omitempty" jsonschema:"minimum=-2,maximum=2" jsonschema_description:"Penalizes tokens that have appeared at all so far, from -2 to 2."`
	// Seed makes sampling repeatable on a best-effort basis.
	Seed *int `json:"seed,omitempty" jsonschema_description:"Makes sampling repeatable on a best-effort basis."`
}

// ApplyToChatCompletion implements [compat_oai.ChatConfig]: each generation
// field lands on its chat completion counterpart.
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
}

// Capability sets shared by the entries below. The curated Nemotron models
// take response_format json_schema, so structured output is generated
// natively; under a response format the model answers in JSON at once and
// never calls a tool, so the constraint is claimed only for requests without
// tools.
var (
	// multimodal is the vision-capable Nemotron set: text and images in,
	// text or JSON out, tools, and tool choice.
	multimodal = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       true,
		ToolChoice:  true,
		Output:      []string{"text", "json", "array", "enum"},
		Constrained: ai.ConstrainedSupportNoTools,
	}
	// textOnly is multimodal minus image input, for models that reject media.
	textOnly = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       false,
		ToolChoice:  true,
		Output:      []string{"text", "json", "array", "enum"},
		Constrained: ai.ConstrainedSupportNoTools,
	}
	// generic is what the plugin assumes of a model it does not curate: NIM
	// serves chat, embedding and specialist models side by side, so only
	// text chat with tools is claimed. [NVIDIA.Models] widens it per model.
	generic = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
		Output:     []string{"text", "json"},
	}
)

// mediaTypes are the media NIM's chat completions read that the plugin sends:
// images, as image_url content parts, on vision-capable models.
var mediaTypes = []string{"image/*"}

// supportedModels curates capabilities for well-known NIM models, keyed by
// their full NIM ID (publisher/model). It is not the set of usable models: any
// NIM model resolves on demand and takes [dynamicModelOptions], so an ID
// absent here still works. No versions are declared, since NIM serves each
// model under one ID.
//
// Catalog: https://build.nvidia.com/models
var supportedModels = map[string]ai.ModelOptions{
	"nvidia/nemotron-3.5-super-vl-120b-a12b": {Label: "Nemotron 3.5 Super VL 120B A12B", Supports: &multimodal},
	"nvidia/nemotron-3.5-lightning-30b-a3b":  {Label: "Nemotron 3.5 Lightning 30B A3B", Supports: &textOnly},
	"nvidia/nemotron-3.5-bolt-e4b":           {Label: "Nemotron 3.5 Bolt E4B", Supports: &multimodal},
}

// dynamicModelOptions is advertised for NIM models that resolve dynamically
// rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &generic,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// NVIDIA configures the NVIDIA NIM plugin.
type NVIDIA struct {
	// APIKey is the NVIDIA API key. If empty, NVIDIA_API_KEY is consulted.
	APIKey string
	// Opts contains additional OpenAI client request options, such as
	// [option.WithBaseURL] for a self-hosted NIM (NVIDIA_BASE_URL works too).
	// Options supplied here are applied after the plugin defaults, so they
	// win on overlap.
	Opts []option.RequestOption

	// Models overrides what the plugin knows about a NIM model, keyed by its
	// full NIM ID, bare ("nvidia/nemotron-3.5-bolt-e4b") or provider-prefixed
	// ("nvidia/nvidia/nemotron-3.5-bolt-e4b"). Every NIM model already works
	// without an entry: known IDs carry curated capabilities and the rest
	// take the NIM defaults. Supply an entry only to correct or extend what
	// the plugin resolves, most often to claim image input for a vision
	// model the plugin does not curate.
	//
	//	&nvidia.NVIDIA{Models: map[string]ai.ModelOptions{
	//		"google/gemma-3-12b-it": {Supports: &compat_oai.Multimodal},
	//	}}
	//
	// Fields left at their zero value keep what the plugin resolves, so an
	// entry can pin one capability without restating the label or the
	// versions. Entries apply to the models Init registers as well as the
	// ones [NVIDIA.ListActions] advertises and [NVIDIA.ResolveAction] builds,
	// which is the way to describe a curated model differently: Init has
	// already registered those and nothing can re-register them.
	Models map[string]ai.ModelOptions

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (n *NVIDIA) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (n *NVIDIA) Init(ctx context.Context) []api.Action {
	baseURL := os.Getenv("NVIDIA_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	apiKey := n.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("NVIDIA_API_KEY")
	}
	if apiKey == "" {
		panic("nvidia plugin initialization failed: apiKey is required")
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithBaseURL(baseURL),
	}
	opts = append(opts, n.Opts...)

	n.openAICompatible.Provider = provider
	n.openAICompatible.MediaTypes = mediaTypes
	n.openAICompatible.Opts = opts
	actions := n.openAICompatible.Init(ctx)

	for model := range supportedModels {
		actions = append(actions, n.newModel(model, n.modelOptions(model)))
	}
	return actions
}

// newModel creates a NIM model without registering it.
func (n *NVIDIA) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return compat_oai.NewChatModel[ChatConfig](&n.openAICompatible, id, opts)
}

// bareID returns the NIM model ID without the plugin's "nvidia/" prefix.
// NIM IDs are publisher/model, and NVIDIA's own models are published as
// "nvidia/...", so the prefix is removed only when what remains still names
// a publisher: "nvidia/nvidia/nemotron-3.5-bolt-e4b" and
// "nvidia/nemotron-3.5-bolt-e4b" both become the latter.
func bareID(id string) string {
	if rest, ok := strings.CutPrefix(id, provider+"/"); ok && strings.Contains(rest, "/") {
		return rest
	}
	return id
}

// modelOptions returns the ModelOptions for a NIM model ID: curated
// capabilities for a known model and the NIM defaults for the rest, with an
// entry from [NVIDIA.Models] overlaid on whichever applies.
//
// Every path that describes a model goes through this one, which is what
// makes a caller's entry authoritative whether Init, ListActions or
// ResolveAction gets there first. Entries are matched by [bareID] so that
// both key forms work for NVIDIA-published models, whose IDs already start
// with the provider name.
func (n *NVIDIA) modelOptions(id string) ai.ModelOptions {
	id = bareID(id)
	opts, ok := supportedModels[id]
	if !ok {
		opts = dynamicModelOptions
	}
	for key, override := range n.Models {
		if bareID(key) == id {
			return opts.Overlay(override)
		}
	}
	return opts
}

// ModelRef names a NIM model and carries the config to generate with, so the
// config is typed at the call site instead of an any the model checks at
// runtime. A nil config leaves the request's config unset.
//
//	ai.WithModel(nvidia.ModelRef("nvidia/nemotron-3.5-super-vl-120b-a12b", &nvidia.ChatConfig{
//		MaxOutputTokens: 1024,
//	}))
//
// id is the full NIM model ID (publisher/model), with or without the
// provider prefix.
func ModelRef(id string, config *ChatConfig) ai.ModelRef {
	return ai.NewModelRef(api.NewName(provider, bareID(id)), config)
}

// ListActions lists the models the configured NIM endpoint exposes,
// described by the plugin's config schema and capabilities.
func (n *NVIDIA) ListActions(ctx context.Context) []api.ActionDesc {
	return compat_oai.ListChatActions[ChatConfig](ctx, &n.openAICompatible, n.modelOptions)
}

// ResolveAction dynamically builds a model exposed by the NIM endpoint,
// described by the plugin's config schema and capabilities.
func (n *NVIDIA) ResolveAction(atype api.ActionType, id string) api.Action {
	return compat_oai.ResolveChatAction[ChatConfig](&n.openAICompatible, atype, id, n.modelOptions)
}
