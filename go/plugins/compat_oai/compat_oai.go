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

package compat_oai

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	pluginjsonschema "github.com/firebase/genkit/go/plugins/internal/jsonschema"
	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

var (
	// BasicText describes model capabilities for text-only GPT models.
	BasicText = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      false,
	}

	// Multimodal describes model capabilities for multimodal GPT models.
	Multimodal = ai.ModelSupports{
		Multiturn:  true,
		Tools:      true,
		SystemRole: true,
		Media:      true,
		ToolChoice: true,
	}
)

// sdkConfigSchema is the schema advertised by models that take the OpenAI
// SDK's [openai.ChatCompletionNewParams] as their config. Reflecting the SDK
// params struct is expensive and the result is read-only, so it is built on
// first use and shared by every such model. Stop inlines a union that
// marshals as a string or a string array, which the shared reflector cannot
// know, so it is mapped here.
var sdkConfigSchema = sync.OnceValue(func() map[string]any {
	return pluginjsonschema.ReflectConfigSchema(openai.ChatCompletionNewParams{}, map[string]*jsonschema.Schema{
		"ChatCompletionNewParamsStopUnion": {AnyOf: []*jsonschema.Schema{
			{Type: "string"},
			{Type: "array", Items: &jsonschema.Schema{Type: "string"}},
		}},
	})
})

// OpenAICompatible is a plugin that provides compatibility with OpenAI's Compatible APIs.
// It allows defining models and embedders that can be used with Genkit.
type OpenAICompatible struct {
	// mu protects concurrent access to the client and initialization state
	mu sync.Mutex

	// initted tracks whether the plugin has been initialized
	initted bool

	// client is the OpenAI client used for making API requests
	// see https://github.com/openai/openai-go
	client *openai.Client

	// Opts contains request options for the OpenAI client.
	// Required: Must include at least WithAPIKey for authentication.
	// Optional: Can include other options like WithOrganization, WithBaseURL, etc.
	Opts []option.RequestOption

	// Provider is a unique identifier for the plugin.
	// This will be used as a prefix for model names (e.g., "myprovider/model-name").
	// Should be lowercase and match the plugin's Name() method.
	Provider string

	// API key to use with the desired plugin.
	APIKey string

	// Base URL to use for custom endpoints.
	// This should be used if you are running through a proxy or
	// using a non-official endpoint
	BaseURL string

	// ListModels optionally overrides how the provider's model IDs are
	// listed, for providers whose models endpoint does not speak OpenAI's
	// pagination. It must return every model the provider serves; nil uses
	// the OpenAI-style listing.
	ListModels func(ctx context.Context, client *openai.Client) ([]string, error)

	// descs caches the action descriptors of listed models by name; they are
	// deterministic per name, and rebuilding a full model action per listed
	// model on every reflection poll is wasteful. A plugin instance lists
	// through one config type, so the cache never mixes schemas.
	descs sync.Map
}

// Init implements genkit.Plugin.
func (o *OpenAICompatible) Init(ctx context.Context) []api.Action {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.initted {
		panic("compat_oai.Init already called")
	}

	if o.APIKey != "" {
		o.Opts = append([]option.RequestOption{option.WithAPIKey(o.APIKey)}, o.Opts...)
	}

	if o.BaseURL != "" {
		o.Opts = append([]option.RequestOption{option.WithBaseURL(o.BaseURL)}, o.Opts...)
	}

	// create client
	client := openai.NewClient(o.Opts...)
	o.client = &client
	o.initted = true

	return []api.Action{}
}

// Name implements genkit.Plugin.
func (o *OpenAICompatible) Name() string {
	return o.Provider
}

// checkInitted panics unless Init has run, which is what makes the client
// available to the model constructors.
func (o *OpenAICompatible) checkInitted() {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAICompatible.Init not called")
	}
}

// clientForKey returns the plugin's client, or a request-scoped client whose
// API key overrides the plugin's when key is non-empty. The plugin's options
// are cloned before appending so concurrent requests cannot write into each
// other's backing array, and the override is appended last so it wins over
// any key the options carry.
func (o *OpenAICompatible) clientForKey(key string) *openai.Client {
	if key == "" {
		return o.client
	}
	client := openai.NewClient(append(slices.Clip(o.Opts), option.WithAPIKey(key))...)
	return &client
}

// NewModel creates a model that takes the OpenAI SDK's
// [openai.ChatCompletionNewParams] as its config, the raw request the plugin
// sends to the provider. The framework validates the request's config against
// the SDK schema and deserializes it before the model function runs; the
// params' Messages, Tools, and ToolChoice are managed by Genkit and
// overwritten from the request, while a Model set in the config pins the
// exact version the request is served by. Providers with a curated config of
// their own use [NewChatModel] instead.
//
// The model is not registered: return it from a plugin's Init for the
// framework to register, or register it with [genkit.RegisterAction].
func (o *OpenAICompatible) NewModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	o.checkInitted()
	return newSDKModel(o.client, o.Provider, id, opts)
}

// DefineModel creates an unregistered model that takes the OpenAI SDK's
// request params as its config.
//
// Deprecated: use [OpenAICompatible.NewModel], which names what it does and
// takes the provider from the plugin. Define is the verb for a caller
// supplying the implementation, which this is not.
func (o *OpenAICompatible) DefineModel(provider, id string, opts ai.ModelOptions) ai.Model {
	o.checkInitted()
	return newSDKModel(o.client, provider, id, opts)
}

// newSDKModel creates an unregistered model whose config is the OpenAI SDK's
// request params type. A nil ConfigSchema defaults to the reflected SDK
// schema and an empty label is derived from the provider and the name.
func newSDKModel(client *openai.Client, provider, id string, opts ai.ModelOptions) *ai.ModelAction {
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = sdkConfigSchema()
	}
	if opts.Label == "" {
		opts.Label = fmt.Sprintf("%s - %s", provider, id)
	}

	return ai.NewModelAction(api.NewName(provider, id), &opts, func(
		ctx context.Context,
		input *ai.ModelRequest,
		config openai.ChatCompletionNewParams,
		cb ai.ModelStreamCallback,
	) (*ai.ModelResponse, error) {
		return NewModelGenerator(client, id).
			WithParams(config).
			WithMessages(input.Messages).
			WithTools(input.Tools).
			WithToolChoice(input.ToolChoice).
			Generate(ctx, input, cb)
	})
}

// NewChatModel creates an unregistered model whose config is the provider's
// own Config type; the framework validates the request's config against the
// schema inferred from Config and deserializes it before the model function
// runs, and the config merges itself into the outgoing request through
// [ChatConfig]. A config Version pins the model version the request is served
// by, and a config carrying a request API key (see
// [ChatCompletionConfig.APIKey]) is served by a request-scoped client. An
// empty label is derived from the plugin's provider and the name.
//
// Return the model from the plugin's Init for the framework to register, or
// register it with [genkit.RegisterAction].
func NewChatModel[Config ChatConfig](o *OpenAICompatible, id string, opts ai.ModelOptions) *ai.ModelAction {
	o.checkInitted()
	if opts.Label == "" {
		opts.Label = fmt.Sprintf("%s - %s", o.Provider, id)
	}

	return ai.NewModelAction(api.NewName(o.Provider, id), &opts, func(
		ctx context.Context,
		input *ai.ModelRequest,
		config Config,
		cb ai.ModelStreamCallback,
	) (*ai.ModelResponse, error) {
		// A config Version lands on params.Model, which WithParams lets win
		// over the model ID: it names the exact version the request is
		// served by.
		var params openai.ChatCompletionNewParams
		config.ApplyToChatCompletion(&params)

		return NewModelGenerator(o.clientForKey(config.RequestAPIKey()), id).
			WithParams(params).
			WithMessages(input.Messages).
			WithTools(input.Tools).
			WithToolChoice(input.ToolChoice).
			Generate(ctx, input, cb)
	})
}

// NewEmbedder creates an embedder that takes an [EmbeddingConfig] as its
// per-request config; a config carrying a request API key is served by a
// request-scoped client. The embedder is not registered: return it from a
// plugin's Init for the framework to register, or register it with
// [genkit.RegisterAction].
func (o *OpenAICompatible) NewEmbedder(id string, embedOpts *ai.EmbedderOptions) *ai.EmbedderAction {
	return o.newEmbedder(o.Provider, id, embedOpts)
}

// DefineEmbedder creates an unregistered embedder.
//
// Deprecated: use [OpenAICompatible.NewEmbedder], which names what it does and
// takes the provider from the plugin. Define is the verb for a caller
// supplying the implementation, which this is not.
func (o *OpenAICompatible) DefineEmbedder(provider, id string, embedOpts *ai.EmbedderOptions) ai.Embedder {
	return o.newEmbedder(provider, id, embedOpts)
}

// newEmbedder builds the embedder both entry points return.
func (o *OpenAICompatible) newEmbedder(provider, id string, embedOpts *ai.EmbedderOptions) *ai.EmbedderAction {
	o.checkInitted()

	return ai.NewEmbedderAction(api.NewName(provider, id), embedOpts, func(ctx context.Context, req *ai.EmbedRequest, config EmbeddingConfig) (*ai.EmbedResponse, error) {
		var data openai.EmbeddingNewParamsInputUnion
		for _, doc := range req.Input {
			for _, p := range doc.Content {
				data.OfArrayOfStrings = append(data.OfArrayOfStrings, p.Text)
			}
		}

		params := openai.EmbeddingNewParams{
			Input:          data,
			Model:          id,
			EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
		}
		config.applyToEmbedding(&params)

		embeddingResp, err := o.clientForKey(config.APIKey).Embeddings.New(ctx, params)
		if err != nil {
			return nil, err
		}

		resp := &ai.EmbedResponse{}
		for _, emb := range embeddingResp.Data {
			embedding, err := embeddingFloats(emb)
			if err != nil {
				return nil, err
			}
			resp.Embeddings = append(resp.Embeddings, &ai.Embedding{Embedding: embedding})
		}
		return resp, nil
	})
}

// embeddingFloats extracts the vector from an embedding in either encoding
// the API can return: a float array, or (with encoding_format base64) a
// base64 string of little-endian float32s, which the SDK leaves undecoded in
// the raw JSON.
func embeddingFloats(emb openai.Embedding) ([]float32, error) {
	if len(emb.Embedding) > 0 {
		embedding := make([]float32, len(emb.Embedding))
		for i, val := range emb.Embedding {
			embedding[i] = float32(val)
		}
		return embedding, nil
	}

	var encoded string
	if err := json.Unmarshal([]byte(emb.JSON.Embedding.Raw()), &encoded); err != nil || encoded == "" {
		// Not a base64 string: a genuinely empty float vector.
		return []float32{}, nil
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("compat_oai: decoding base64 embedding: %w", err)
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("compat_oai: base64 embedding has %d bytes, want a multiple of 4", len(data))
	}
	embedding := make([]float32, len(data)/4)
	for i := range embedding {
		embedding[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return embedding, nil
}

// IsDefinedEmbedder reports whether the named [Embedder] is defined by this
// plugin. name is the full action name, provider prefix included; the
// subplugins take a bare or prefixed embedder ID instead.
func (o *OpenAICompatible) IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, name) != nil
}

// Embedder returns the [ai.Embedder] with the given name, the full action name
// with its provider prefix. It returns nil if the embedder was not defined.
//
// Deprecated: Embedding resolves an embedder from its name, so passing
// [ai.WithEmbedderName] is usually enough; a plugin's EmbedderRef carries a
// typed config with it. Use [genkit.LookupEmbedder] when the action itself is
// what you need.
func (o *OpenAICompatible) Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, name)
}

// Model returns the [ai.Model] with the given name, the full action name with
// its provider prefix. It returns nil if the model was not defined.
//
// Deprecated: Generation resolves a model from its name, so passing
// [ai.WithModelName] is usually enough; a plugin's ModelRef carries a typed
// config with it. Use [genkit.LookupModel] when the action itself is what you
// need.
func (o *OpenAICompatible) Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, name)
}

// IsDefinedModel reports whether the named [Model] is defined by this plugin.
// name is the full action name, provider prefix included; the subplugins take
// a bare or prefixed model ID instead.
func (o *OpenAICompatible) IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, name) != nil
}

// ListActions lists the models the provider's API reports, described with the
// SDK config schema and generic multimodal capabilities. Plugins with a config
// type and curated capabilities of their own use [ListChatActions].
func (o *OpenAICompatible) ListActions(ctx context.Context) []api.ActionDesc {
	return listActions(ctx, o, func(id string) api.ActionDesc {
		return newSDKModel(o.client, o.Provider, id, sdkModelOptions(o.Provider, id)).Desc()
	})
}

// ResolveAction resolves a model not registered up front, described with the
// SDK config schema and generic multimodal capabilities. Plugins with a config
// type and curated capabilities of their own use [ResolveChatAction].
func (o *OpenAICompatible) ResolveAction(atype api.ActionType, id string) api.Action {
	switch atype {
	case api.ActionTypeModel:
		return newSDKModel(o.client, o.Provider, id, sdkModelOptions(o.Provider, id))
	}
	return nil
}

// sdkModelOptions is [DefaultModelOptions] with the label the SDK-config
// constructors would otherwise derive.
func sdkModelOptions(provider, id string) ai.ModelOptions {
	opts := DefaultModelOptions()
	opts.Label = fmt.Sprintf("%s - %s", provider, id)
	return opts
}

// DefaultModelOptions is the capability set advertised for models that are
// discovered or resolved dynamically rather than curated by a plugin.
func DefaultModelOptions() ai.ModelOptions {
	return ai.ModelOptions{
		Stage:    ai.ModelStageStable,
		Versions: []string{},
		Supports: &Multimodal,
	}
}

// ActionName builds the action name for a model or embedder ID under
// provider, taking the ID either bare or already provider-prefixed. The
// prefix is applied by concatenation, so without the trim an
// already-prefixed name would double up and name an action that resolves
// nowhere.
func ActionName(provider, id string) string {
	return api.NewName(provider, strings.TrimPrefix(id, provider+"/"))
}

// IsDefinedModel reports whether the model is registered with g, which is the
// guard against defining one twice. id is the model ID, bare or
// provider-prefixed. The lookup deliberately does not resolve dynamically: on
// plugins that resolve models on demand, a resolving lookup would register the
// very model the caller is checking for and answer true for any ID.
func IsDefinedModel(g *genkit.Genkit, provider, id string) bool {
	return genkit.LookupAction(g, fmt.Sprintf("/%s/%s", api.ActionTypeModel, ActionName(provider, id))) != nil
}

// RegisterChatModel registers a model built from the plugin's Config type with
// g and returns it (see [NewChatModel]). id is the model ID, bare or
// provider-prefixed; a nil opts falls back to modelOptions(id), which is
// how plugins resolve curated capabilities. Registering an ID that is
// already registered panics, so guard with [IsDefinedModel] when in doubt.
func RegisterChatModel[Config ChatConfig](g *genkit.Genkit, o *OpenAICompatible, id string, opts *ai.ModelOptions, modelOptions func(id string) ai.ModelOptions) (ai.Model, error) {
	// Trim before resolving, so a prefixed ID still hits the curated list.
	id = strings.TrimPrefix(id, o.Provider+"/")

	var modelOpts ai.ModelOptions
	if opts != nil {
		modelOpts = *opts
	} else {
		modelOpts = modelOptions(id)
	}

	model := NewChatModel[Config](o, id, modelOpts)
	genkit.RegisterAction(g, model)
	return model, nil
}

// ListChatActions lists the models the provider's API reports, each described
// by modelOptions and the schema of the plugin's Config type. Plugins with
// curated capabilities pass the same options lookup their ResolveAction uses,
// so listing and resolving a model can never disagree.
func ListChatActions[Config ChatConfig](ctx context.Context, o *OpenAICompatible, modelOptions func(id string) ai.ModelOptions) []api.ActionDesc {
	return listActions(ctx, o, func(id string) api.ActionDesc {
		return NewChatModel[Config](o, id, modelOptions(id)).Desc()
	})
}

// ResolveChatAction resolves a model not registered up front, described by
// modelOptions and the schema of the plugin's Config type; see
// [ListChatActions].
func ResolveChatAction[Config ChatConfig](o *OpenAICompatible, atype api.ActionType, id string, modelOptions func(id string) ai.ModelOptions) api.Action {
	switch atype {
	case api.ActionTypeModel:
		return NewChatModel[Config](o, id, modelOptions(id))
	}
	return nil
}

// listActions lists the models the provider's API reports, described by desc.
// Descriptors are cached per ID: they are deterministic, and reflection
// polls the list often.
func listActions(ctx context.Context, o *OpenAICompatible, desc func(id string) api.ActionDesc) []api.ActionDesc {
	listModels := o.ListModels
	if listModels == nil {
		listModels = listOpenAIModels
	}
	models, err := listModels(ctx, o.client)
	if err != nil {
		return nil
	}
	actions := make([]api.ActionDesc, 0, len(models))
	for _, id := range models {
		if cached, ok := o.descs.Load(id); ok {
			actions = append(actions, cached.(api.ActionDesc))
			continue
		}
		d := desc(id)
		o.descs.Store(id, d)
		actions = append(actions, d)
	}
	return actions
}

func listOpenAIModels(ctx context.Context, client *openai.Client) ([]string, error) {
	models := []string{}
	iter := client.Models.ListAutoPaging(ctx)
	for iter.Next() {
		m := iter.Current()
		models = append(models, m.ID)
	}
	if err := iter.Err(); err != nil {
		return nil, err
	}

	return models, nil
}
