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

// Package openai provides a Genkit plugin for OpenAI's models and embedders.
package openai

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const provider = "openai"

// TextEmbeddingConfig is the per-request config for OpenAI embedders.
type TextEmbeddingConfig = compat_oai.EmbeddingConfig

// Capability sets shared by the entries below. Structured outputs
// (response_format json_schema with strict) arrived with the gpt-4o-mini and
// gpt-4o-2024-08-06 snapshots, so the models predating them advertise no
// constrained generation and fall back to schema instructions in the prompt.
// See https://developers.openai.com/api/docs/guides/structured-outputs.
var (
	textOnly = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       false,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportAll,
	}
	multimodal = ai.ModelSupports{
		Multiturn:   true,
		Tools:       true,
		SystemRole:  true,
		Media:       true,
		ToolChoice:  true,
		Output:      []string{"text", "json"},
		Constrained: ai.ConstrainedSupportAll,
	}
	textOnlyLegacy = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
		Output:     []string{"text", "json"},
	}
	multimodalLegacy = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
		ToolChoice: true,
		Output:     []string{"text", "json"},
	}
)

// supportedModels curates capabilities for well-known OpenAI models. It is not
// the set of usable models: any OpenAI model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works. Dated snapshots are
// folded into Versions rather than registered as separate models. Model IDs are
// spelled out rather than taken from the SDK's ChatModel constants, which trail
// the catalog by whichever SDK version go.mod pins.
//
// Two kinds of model are deliberately absent. The "pro" tiers (gpt-5-pro,
// gpt-5.2-pro, gpt-5.4-pro, gpt-5.5-pro, o1-pro, o3-pro) are served only by
// the Responses API, and this plugin speaks Chat Completions. Audio, image,
// realtime, transcription, and moderation models are not chat models.
//
// Catalog: https://developers.openai.com/api/docs/models
// Retirements: https://developers.openai.com/api/docs/deprecations
var supportedModels = map[string]ai.ModelOptions{
	// GPT-5.6, the current frontier family. No dated snapshots yet; the
	// bare gpt-5.6 alias routes to sol and resolves dynamically.
	"gpt-5.6-sol": {
		Label:    "OpenAI GPT-5.6 Sol",
		Supports: &multimodal,
		Versions: []string{"gpt-5.6-sol"},
	},
	"gpt-5.6-terra": {
		Label:    "OpenAI GPT-5.6 Terra",
		Supports: &multimodal,
		Versions: []string{"gpt-5.6-terra"},
	},
	"gpt-5.6-luna": {
		Label:    "OpenAI GPT-5.6 Luna",
		Supports: &multimodal,
		Versions: []string{"gpt-5.6-luna"},
	},

	"gpt-5.5": {
		Label:    "OpenAI GPT-5.5",
		Supports: &multimodal,
		Versions: []string{"gpt-5.5", "gpt-5.5-2026-04-23"},
	},
	"gpt-5.4": {
		Label:    "OpenAI GPT-5.4",
		Supports: &multimodal,
		Versions: []string{"gpt-5.4", "gpt-5.4-2026-03-05"},
	},
	"gpt-5.4-mini": {
		Label:    "OpenAI GPT-5.4-mini",
		Supports: &multimodal,
		Versions: []string{"gpt-5.4-mini", "gpt-5.4-mini-2026-03-17"},
	},
	"gpt-5.4-nano": {
		Label:    "OpenAI GPT-5.4-nano",
		Supports: &multimodal,
		Versions: []string{"gpt-5.4-nano", "gpt-5.4-nano-2026-03-17"},
	},
	"gpt-5.2": {
		Label:    "OpenAI GPT-5.2",
		Supports: &multimodal,
		Versions: []string{"gpt-5.2", "gpt-5.2-2025-12-11"},
	},
	"gpt-5.1": {
		Label:    "OpenAI GPT-5.1",
		Supports: &multimodal,
		Versions: []string{"gpt-5.1", "gpt-5.1-2025-11-13"},
	},

	// GPT-5. The dated snapshots shut down 2026-12-11; the aliases stay.
	"gpt-5": {
		Label:    "OpenAI GPT-5",
		Supports: &multimodal,
		Versions: []string{"gpt-5", "gpt-5-2025-08-07"},
	},
	"gpt-5-mini": {
		Label:    "OpenAI GPT-5-mini",
		Supports: &multimodal,
		Versions: []string{"gpt-5-mini", "gpt-5-mini-2025-08-07"},
	},
	"gpt-5-nano": {
		Label:    "OpenAI GPT-5-nano",
		Supports: &multimodal,
		Versions: []string{"gpt-5-nano", "gpt-5-nano-2025-08-07"},
	},

	"gpt-4.1": {
		Label:    "OpenAI GPT-4.1",
		Supports: &multimodal,
		Versions: []string{"gpt-4.1", "gpt-4.1-2025-04-14"},
	},
	"gpt-4.1-mini": {
		Label:    "OpenAI GPT-4.1-mini",
		Supports: &multimodal,
		Versions: []string{"gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"},
	},
	// Shuts down 2026-10-23, replaced by gpt-5.6-luna.
	"gpt-4.1-nano": {
		Label:    "OpenAI GPT-4.1-nano",
		Supports: &multimodal,
		Versions: []string{"gpt-4.1-nano", "gpt-4.1-nano-2025-04-14"},
	},

	"gpt-4o": {
		Label:    "OpenAI GPT-4o",
		Supports: &multimodal,
		// gpt-4o-2024-05-13 shuts down 2026-10-23.
		Versions: []string{"gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13"},
	},
	"gpt-4o-mini": {
		Label:    "OpenAI GPT-4o-mini",
		Supports: &multimodal,
		Versions: []string{"gpt-4o-mini", "gpt-4o-mini-2024-07-18"},
	},

	// Reasoning models. o1, o3-mini, and o4-mini shut down 2026-10-23;
	// o3-2025-04-16 shuts down 2026-12-11.
	"o3": {
		Label:    "OpenAI o3",
		Supports: &multimodal,
		Versions: []string{"o3", "o3-2025-04-16"},
	},
	"o4-mini": {
		Label:    "OpenAI o4-mini",
		Supports: &multimodal,
		Versions: []string{"o4-mini", "o4-mini-2025-04-16"},
	},
	"o3-mini": {
		Label:    "OpenAI o3-mini",
		Supports: &textOnly,
		Versions: []string{"o3-mini", "o3-mini-2025-01-31"},
	},
	"o1": {
		Label:    "OpenAI o1",
		Supports: &multimodal,
		Versions: []string{"o1", "o1-2024-12-17"},
	},

	// Legacy models, all shutting down 2026-10-23 except
	// gpt-3.5-turbo-1106 and -instruct, which go 2026-09-28.
	"gpt-4-turbo": {
		Label:    "OpenAI GPT-4-turbo",
		Supports: &multimodalLegacy,
		Versions: []string{"gpt-4-turbo", "gpt-4-turbo-2024-04-09"},
	},
	"gpt-4": {
		Label:    "OpenAI GPT-4",
		Supports: &textOnlyLegacy,
		Versions: []string{"gpt-4", "gpt-4-0613"},
	},
	"gpt-3.5-turbo": {
		Label:    "OpenAI GPT-3.5-turbo",
		Supports: &textOnlyLegacy,
		Versions: []string{"gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct"},
	},
}

// dynamicModelOptions is advertised for OpenAI models that resolve dynamically
// rather than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
	Supports: &multimodal,
	Versions: []string{},
	Stage:    ai.ModelStageStable,
}

// supportedEmbeddingModels curates capabilities for well-known OpenAI
// embedders.
var supportedEmbeddingModels = map[string]ai.EmbedderOptions{
	"text-embedding-3-large": {
		Dimensions: 3072,
		Label:      "Open AI - Text Embedding 3 Large",
		Supports: &ai.EmbedderSupports{
			Input: []string{"text"},
		},
	},
	"text-embedding-3-small": {
		Dimensions: 1536,
		Label:      "Open AI - Text Embedding 3 Small",
		Supports: &ai.EmbedderSupports{
			Input: []string{"text"},
		},
	},
	"text-embedding-ada-002": {
		Dimensions: 1536,
		Label:      "Open AI - Text Embedding ADA 002",
		Supports: &ai.EmbedderSupports{
			Input: []string{"text"},
		},
	},
}

type OpenAI struct {
	// APIKey is the API key for the OpenAI API. If empty, the values of the environment variable "OPENAI_API_KEY" will be consulted.
	// Request a key at https://platform.openai.com/api-keys
	APIKey string
	// Optional: Opts are additional options for the OpenAI client.
	// Can include other options like WithOrganization, WithBaseURL, etc.
	Opts []option.RequestOption

	openAICompatible compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (o *OpenAI) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (o *OpenAI) Init(ctx context.Context) []api.Action {
	apiKey := o.APIKey

	// if api key is not set, get it from environment variable
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}

	if apiKey == "" {
		panic("openai plugin initialization failed: apiKey is required")
	}

	// set the options
	o.openAICompatible.Opts = []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if len(o.Opts) > 0 {
		o.openAICompatible.Opts = append(o.openAICompatible.Opts, o.Opts...)
	}

	o.openAICompatible.Provider = provider
	actions := o.openAICompatible.Init(ctx)

	// define default models
	for model, opts := range supportedModels {
		actions = append(actions, o.newModel(model, opts))
	}

	// define default embedders
	for name, opts := range supportedEmbeddingModels {
		actions = append(actions, o.newEmbedder(name, opts))
	}

	return actions
}

// newModel creates an OpenAI model without registering it.
func (o *OpenAI) newModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	return o.openAICompatible.NewModel(id, opts)
}

// newEmbedder creates an OpenAI embedder without registering it.
func (o *OpenAI) newEmbedder(id string, opts ai.EmbedderOptions) *ai.EmbedderAction {
	return o.openAICompatible.NewEmbedder(id, &opts)
}

// modelOptions returns the ModelOptions for an OpenAI model ID: curated
// capabilities for a known model and the generic multimodal defaults for the
// rest, with the label derived from the name.
func modelOptions(id string) ai.ModelOptions {
	if opts, ok := supportedModels[id]; ok {
		return opts
	}
	return dynamicModelOptions
}

// ModelRef names an OpenAI model and carries the config to generate with, so
// the config is typed at the call site instead of an any the model checks at
// runtime. The config is the OpenAI SDK's request params, the raw request the
// plugin sends; a nil config leaves the request's config unset.
//
//	ai.WithModel(openai.ModelRef("gpt-4o", &openaiGo.ChatCompletionNewParams{
//		Temperature: openaiGo.Float(0.7),
//	}))
//
// name is the model ID, with or without the provider prefix: "gpt-4o" and
// "openai/gpt-4o" name the same model, as they do everywhere else in this
// package.
func ModelRef(id string, config *openaiGo.ChatCompletionNewParams) ai.ModelRef {
	return ai.NewModelRef(compat_oai.ActionName(provider, id), config)
}

// RegisterModel registers an OpenAI model with g and returns it. The plugin
// supplies the implementation; opts describes what the model supports, and a
// nil opts takes the capabilities the plugin resolves for that ID, curated
// for a known model and the generic multimodal defaults for the rest.
//
// Most applications never need this. Every OpenAI model resolves on demand,
// so naming one that was never registered is enough:
//
//	genkit.Generate(ctx, g, ai.WithModelName("openai/gpt-4o"), ...)
//
// Reach for RegisterModel only to pin capabilities that differ from the ones
// the plugin resolves, which is what opts is for.
//
// Registering an ID that is already registered panics, and generating with an
// ID registers it, so register a model before its first use or guard with
// [IsDefinedModel]. id is the model ID, bare or provider-prefixed.
func (o *OpenAI) RegisterModel(g *genkit.Genkit, id string, opts *ai.ModelOptions) (ai.Model, error) {
	// Trim before resolving, so a prefixed ID still hits supportedModels.
	id = strings.TrimPrefix(id, provider+"/")

	var modelOpts ai.ModelOptions
	if opts != nil {
		modelOpts = *opts
	} else {
		modelOpts = modelOptions(id)
	}

	model := o.newModel(id, modelOpts)
	genkit.RegisterAction(g, model)
	return model, nil
}

// DefineModel builds an OpenAI model and returns it, without registering it.
//
// Deprecated: use [OpenAI.RegisterModel]. The model this returns is not
// registered, and generation resolves a model from its name, so passing the
// result to ai.WithModel contributes only that name and serves the request
// with a model resolved from it instead. Registering it with
// [genkit.RegisterAction] is what makes these capabilities the ones used.
func (o *OpenAI) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return o.newModel(id, opts)
}

// IsDefinedModel reports whether a model is already registered, which is the
// guard against registering one twice (see [OpenAI.RegisterModel]).
func IsDefinedModel(g *genkit.Genkit, id string) bool {
	return compat_oai.IsDefinedModel(g, provider, id)
}

// Model returns a previously registered model.
//
// Deprecated: Generation resolves a model from its name, so looking one up
// first is rarely necessary: pass ai.WithModelName("openai/gpt-4o") or, to
// carry config with it, [ModelRef]. Use [genkit.LookupModel] when the action
// itself is what you need.
func (o *OpenAI) Model(g *genkit.Genkit, id string) ai.Model {
	return genkit.LookupModel(g, compat_oai.ActionName(provider, id))
}

// NewEmbedderRef names an OpenAI embedder and carries the config to embed
// with, so the config is typed at the call site. A nil config leaves the
// request's options unset.
//
//	ai.WithEmbedder(openai.NewEmbedderRef("text-embedding-3-small", &openai.TextEmbeddingConfig{
//		Dimensions: 256,
//	}))
//
// id is the embedder ID, with or without the provider prefix.
func NewEmbedderRef(id string, config *TextEmbeddingConfig) ai.EmbedderRef {
	return ai.NewEmbedderRef(compat_oai.ActionName(provider, id), config)
}

// EmbedderRef describes an embedding model.
//
// Deprecated: use [NewEmbedderRef], which builds an [ai.EmbedderRef] naming an
// embedder and carrying its typed config, the form generation and embedding
// calls accept.
type EmbedderRef struct {
	Name         string
	ConfigSchema TextEmbeddingConfig // Represents the schema, can be used for default config
	Label        string
	Supports     *ai.EmbedderSupports
	Dimensions   int
}

// RegisterEmbedder registers an OpenAI embedder with g and returns it. A nil
// opts takes the options the plugin curates for a known embedder and empty
// options for the rest.
//
// Registering an ID that is already registered panics, so guard with
// [IsDefinedEmbedder] when in doubt. id is the embedder ID, bare or
// provider-prefixed.
func (o *OpenAI) RegisterEmbedder(g *genkit.Genkit, id string, opts *ai.EmbedderOptions) (ai.Embedder, error) {
	id = strings.TrimPrefix(id, provider+"/")

	var embedOpts ai.EmbedderOptions
	if opts != nil {
		embedOpts = *opts
	} else {
		embedOpts = supportedEmbeddingModels[id]
	}

	embedder := o.newEmbedder(id, embedOpts)
	genkit.RegisterAction(g, embedder)
	return embedder, nil
}

// DefineEmbedder builds an OpenAI embedder and returns it, without registering
// it.
//
// Deprecated: use [OpenAI.RegisterEmbedder]. Like [OpenAI.DefineModel], the
// embedder this returns is not registered, so embedding by that name serves
// the request with a different one unless the caller registers it with
// [genkit.RegisterAction].
func (o *OpenAI) DefineEmbedder(id string, opts *ai.EmbedderOptions) ai.Embedder {
	var embedOpts ai.EmbedderOptions
	if opts != nil {
		embedOpts = *opts
	}
	return o.newEmbedder(id, embedOpts)
}

// IsDefinedEmbedder reports whether an embedder is already registered, which
// is the guard against registering one twice (see [OpenAI.RegisterEmbedder]).
// Like [IsDefinedModel], the lookup does not resolve dynamically.
func IsDefinedEmbedder(g *genkit.Genkit, id string) bool {
	return genkit.LookupAction(g, fmt.Sprintf("/%s/%s", api.ActionTypeEmbedder, compat_oai.ActionName(provider, id))) != nil
}

// Embedder returns a previously registered embedder.
//
// Deprecated: Embedding resolves an embedder from its name, so looking one up
// first is rarely necessary: pass ai.WithEmbedderName("openai/text-embedding-3-small")
// or, to carry config with it, [EmbedderRef]. Use [genkit.LookupEmbedder]
// when the action itself is what you need.
func (o *OpenAI) Embedder(g *genkit.Genkit, id string) ai.Embedder {
	return genkit.LookupEmbedder(g, compat_oai.ActionName(provider, id))
}

func (o *OpenAI) ListActions(ctx context.Context) []api.ActionDesc {
	return o.openAICompatible.ListActions(ctx)
}

func (o *OpenAI) ResolveAction(atype api.ActionType, id string) api.Action {
	return o.openAICompatible.ResolveAction(atype, id)
}
