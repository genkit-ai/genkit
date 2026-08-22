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
	"maps"
	"math"
	"slices"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal"
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

	// ImageGeneration describes text-to-image model capabilities.
	ImageGeneration = ai.ModelSupports{
		Multiturn:  false,
		Tools:      false,
		SystemRole: false,
		Media:      false,
		Output:     []string{"media"},
	}
)

// managedRequestFields are the request fields Genkit owns, stripped from the
// advertised SDK config schema and cleared from any config that carries them
// (see [ModelGenerator.WithParams]). Messages, tools, and the tool choice are
// built from the Genkit request, and functions and function_call are the same
// surface under the names OpenAI used before tools: a config that set either
// pair would offer the model a callable the framework has no handler for, so
// the model could answer with a call Genkit cannot dispatch.
//
// The schema is what the framework validates against, so a config naming one
// of these is rejected rather than silently dropped.
var managedRequestFields = []string{
	"messages", "tools", "tool_choice", "functions", "function_call",
	// Built from the request's output config by applyResponseFormat; an extra
	// naming it would marshal over what Genkit computed.
	"response_format",
}

// sdkSchemaOverrides is the presentation layered onto the reflected
// [openai.ChatCompletionNewParams] schema before it is advertised: the SDK
// structs carry no schema descriptions, and the fields a Genkit option owns
// are hidden from the dev UI, with [rejectManagedConfig] telling a caller who
// sets one anyway which option to use. See [internal.SchemaOverrides] for the
// path notation; a stale entry is a no-op, which
// TestSDKSchemaOverridePathsResolve catches.
var sdkSchemaOverrides = internal.SchemaOverrides{
	Descriptions: map[string]string{
		"audio":                        "Audio output settings, required when modalities includes \"audio\": the voice to answer with and the format the audio comes back in.",
		"audio.format":                 "Output audio format, e.g. wav, mp3, flac, opus, or pcm16.",
		"audio.voice":                  "Voice the model answers with, e.g. alloy, ash, echo.",
		"frequency_penalty":            "Number between -2.0 and 2.0. Positive values penalize tokens by how often they have appeared so far, lowering the chance of repeating the same line verbatim.",
		"logit_bias":                   "Maps token IDs to a bias from -100 to 100 added before sampling. Values near the extremes effectively ban or force the token.",
		"logprobs":                     "Whether to return log probabilities of the output tokens.",
		"max_completion_tokens":        "Upper bound on tokens generated for this request, visible output and reasoning tokens both.",
		"max_tokens":                   "Legacy cap on generated tokens, not compatible with reasoning models. Prefer max_completion_tokens.",
		"metadata":                     "Up to 16 key-value string pairs attached to the request, readable when the completion is stored via the store field.",
		"modalities":                   "Output types the model should generate. Most models produce [\"text\"]; audio-capable models also accept [\"text\", \"audio\"].",
		"model":                        "Pins the exact model version the request is served by. When unset, the request is served by the model named in the action.",
		"parallel_tool_calls":          "Whether the model may call several tools in parallel during a single turn. Defaults to true.",
		"prediction":                   "Predicted output content, which can cut latency when much of the response is known ahead of time, such as when editing a file.",
		"presence_penalty":             "Number between -2.0 and 2.0. Positive values penalize tokens that have appeared at all, nudging the model toward new topics.",
		"reasoning_effort":             "How much reasoning a reasoning model spends before answering, e.g. minimal, low, medium, or high. Lower effort answers faster and spends fewer tokens.",
		"seed":                         "Best-effort determinism: repeated requests with the same seed and parameters should return the same result.",
		"service_tier":                 "Processing tier the request runs on, e.g. auto, default, flex, or priority, subject to the project's settings.",
		"stop":                         "Up to 4 sequences that stop generation when emitted; the response does not include the stop text. Not supported by reasoning models.",
		"store":                        "Whether to store the completion for later retrieval, for the model distillation and evals products.",
		"stream_options":               "Streaming options. Genkit always asks for usage on the final chunk; the rest pass through.",
		"stream_options.include_usage": "Send token usage on a final chunk before [DONE]. Genkit sets this on every streaming call so streamed responses report usage.",
		"temperature":                  "Sampling temperature from 0 to 2. Higher values increase randomness; lower values make output more focused. Tune this or top_p, not both.",
		"top_logprobs":                 "Number of most likely tokens, 0 to 20, to return log probabilities for at each position. Requires logprobs.",
		"top_p":                        "Nucleus sampling: consider only the tokens making up this cumulative probability mass. Tune this or temperature, not both.",
		"user":                         "Stable identifier for the end user, which OpenAI uses to detect abuse. Use a UUID or hash, never identifying information.",
		"web_search_options":           "Enables and configures web search for the request on search-capable models.",
	},
	Hidden: []string{
		// Owned by Genkit primitives; rejectManagedConfig names the option.
		"messages",        // ai.WithMessages / ai.WithPrompt
		"tools",           // ai.WithTools
		"tool_choice",     // ai.WithToolChoice
		"functions",       // the deprecated form of tools
		"function_call",   // the deprecated form of tool_choice
		"response_format", // ai.WithOutputType / ai.WithOutputFormat
		// The response path serves the first choice only, so extra candidates
		// would be billed and dropped.
		"n",
	},
}

// sdkConfigSchema is the schema advertised by models that take the OpenAI
// SDK's [openai.ChatCompletionNewParams] as their config. Reflecting the SDK
// params struct is expensive and the result is read-only, so it is built on
// first use and shared by every such model. Stop inlines a union that
// marshals as a string or a string array, which the shared reflector cannot
// know, so it is mapped here; the result is then curated by
// [sdkSchemaOverrides].
var sdkConfigSchema = sync.OnceValue(func() map[string]any {
	schema := pluginjsonschema.ReflectConfigSchema(openai.ChatCompletionNewParams{}, map[string]*jsonschema.Schema{
		"ChatCompletionNewParamsStopUnion": {AnyOf: []*jsonschema.Schema{
			{Type: "string"},
			{Type: "array", Items: &jsonschema.Schema{Type: "string"}},
		}},
	})
	stripParamObjArtifact(schema)
	internal.ApplySchemaOverrides(schema, sdkSchemaOverrides)
	return schema
})

// stripParamObjArtifact removes the "any" property the SDK's embedded
// param.APIObject (an anonymous any carrying the raw decoded message)
// reflects into every object in the schema. It is machinery rather than a
// request field, and it appears at every depth, so the dev UI would render a
// junk field on each object.
func stripParamObjArtifact(schema map[string]any) {
	drop := func(s map[string]any) map[string]any {
		if props, ok := s["properties"].(map[string]any); ok {
			delete(props, "any")
		}
		return s
	}
	drop(schema)
	base.WalkSubschemas(schema, drop)
}

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
// the SDK schema and deserializes it before the model function runs. A Model
// set in the config pins the exact version the request is served by.
// Providers with a curated config of their own use [NewChatModel] instead.
//
// The schema is the SDK's minus the fields Genkit owns (see
// [managedRequestFields]), so a config naming one of them is rejected rather
// than silently dropped.
//
// The model is not registered: return it from a plugin's Init for the
// framework to register, or register it with [genkit.RegisterAction].
func (o *OpenAICompatible) NewModel(id string, opts ai.ModelOptions) *ai.ModelAction {
	o.checkInitted()
	return newSDKModel(o.client, o.Provider, id, opts)
}

// DefineModel creates an unregistered model that takes its config untyped:
// the OpenAI SDK's request params, or a map of them whose unknown keys ride
// to the wire as JSON extras. Nothing is validated before the model function
// runs; a config the [ModelGenerator.WithConfig] type switch does not
// recognize fails the request instead.
//
// Deprecated: use [OpenAICompatible.NewModel], which names what it does,
// takes the provider from the plugin, and has the framework validate the
// config against the SDK schema before the model function runs.
func (o *OpenAICompatible) DefineModel(provider, id string, opts ai.ModelOptions) ai.Model {
	o.checkInitted()
	return ai.NewModel(api.NewName(provider, id), &opts, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb ai.ModelStreamCallback,
	) (*ai.ModelResponse, error) {
		return NewModelGenerator(o.client, id).
			WithMessages(input.Messages).
			WithConfig(input.Config).
			WithTools(input.Tools).
			WithToolChoice(input.ToolChoice).
			Generate(ctx, input, cb)
	})
}

// DefineImageModel defines an OpenAI-compatible image generation model in the registry.
func (o *OpenAICompatible) DefineImageModel(provider, id string, opts ai.ModelOptions) ai.Model {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAICompatible.Init not called")
	}

	return ai.NewModel(api.NewName(provider, id), &opts, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return generateImage(ctx, o.client, id, input, cb)
	})
}

// newSDKModel creates an unregistered model whose config is the OpenAI SDK's
// request params type. A nil ConfigSchema defaults to the reflected SDK
// schema and an empty label is derived from the provider and the name.
func newSDKModel(client *openai.Client, provider, id string, opts ai.ModelOptions) *ai.ModelAction {
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = sdkConfigSchema()
	}
	if opts.Label == "" {
		opts.Label = internal.ProviderLabel(provider, id)
	}

	return ai.NewModelAction(api.NewName(provider, id), &opts, func(
		ctx context.Context,
		input *ai.ModelRequest,
		config openai.ChatCompletionNewParams,
		cb ai.ModelStreamCallback,
	) (*ai.ModelResponse, error) {
		if err := rejectManagedConfig(&config); err != nil {
			return nil, err
		}
		return NewModelGenerator(client, id).
			WithParams(config).
			WithMessages(input.Messages).
			WithTools(input.Tools).
			WithToolChoice(input.ToolChoice).
			Generate(ctx, input, cb)
	})
}

// rejectManagedConfig reports a config field that a Genkit primitive owns.
//
// Each one is overwritten while the request is built, so accepting it would
// drop the caller's value on the floor. Failing with the option to use
// instead beats a request that silently ignores half of what it was given. n
// is refused for a different reason: the response path serves the first
// choice only, so extra candidates would be billed and never surfaced.
//
// These fields are hidden from the advertised schema (see
// [sdkSchemaOverrides]) by being replaced with a permissive schema rather
// than deleted, so a value passes boundary validation and reaches here rather
// than failing as an unknown property.
//
// Classified ErrInvalidArgument rather than ErrInvalidInput: the value passed
// the action's input schema and is refused on what it means, and the request
// is the caller's to fix.
func rejectManagedConfig(config *openai.ChatCompletionNewParams) error {
	switch {
	case len(config.Messages) > 0:
		return status.Errorf(status.ErrInvalidArgument, "messages must be set using Genkit feature: ai.WithMessages() or ai.WithPrompt()")
	case len(config.Tools) > 0 || len(config.Functions) > 0:
		return status.Errorf(status.ErrInvalidArgument, "tools must be set using Genkit feature: ai.WithTools()")
	case config.ToolChoice.OfAuto.Valid() || config.ToolChoice.OfChatCompletionNamedToolChoice != nil ||
		config.FunctionCall.OfFunctionCallMode.Valid() || config.FunctionCall.OfFunctionCallOption != nil:
		return status.Errorf(status.ErrInvalidArgument, "tool choice must be set using Genkit feature: ai.WithToolChoice()")
	case config.ResponseFormat.OfText != nil || config.ResponseFormat.OfJSONSchema != nil || config.ResponseFormat.OfJSONObject != nil:
		return status.Errorf(status.ErrInvalidArgument, "output format must be set using Genkit feature: ai.WithOutputType(), ai.WithOutputFormat(), or ai.WithOutputInstructions()")
	case config.N.Valid():
		return status.Errorf(status.ErrInvalidArgument, "n is not supported: the response carries the first candidate only, so extra candidates would be billed and dropped")
	}
	return nil
}

// NewChatModel creates an unregistered model whose config is the provider's
// own Config type; the framework validates the request's config against the
// schema inferred from Config and deserializes it before the model function
// runs, and the config merges itself into the outgoing request through
// [ChatConfig], with its [RequestConfig.Extra] fields forwarded after that
// merge so a collision resolves in their favor. A config Version pins the
// model version the request is served by, and a config carrying a request
// API key (see
// [ChatCompletionConfig.APIKey]) is served by a request-scoped client. An
// empty label is derived from the plugin's provider and the name.
//
// Return the model from the plugin's Init for the framework to register, or
// register it with [genkit.RegisterAction].
func NewChatModel[Config ChatConfig](o *OpenAICompatible, id string, opts ai.ModelOptions) *ai.ModelAction {
	o.checkInitted()
	if opts.Label == "" {
		opts.Label = internal.ProviderLabel(o.Provider, id)
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
		if err := forwardRequestExtra(&params, config.RequestExtra()); err != nil {
			return nil, err
		}

		var outputFormats []string
		if opts.Supports != nil {
			outputFormats = opts.Supports.Output
		}
		return NewModelGenerator(o.clientForKey(config.RequestAPIKey()), id).
			WithParams(params).
			WithMessages(input.Messages).
			WithTools(input.Tools).
			WithToolChoice(input.ToolChoice).
			WithOutputFormats(outputFormats).
			Generate(ctx, input, cb)
	})
}

// forwardRequestExtra merges the config's undeclared request fields into
// params after the config's own merge, which is what makes a colliding key
// win: [AddExtraFields] keeps the later write, and the SDK marshals an extra
// field over the struct field it collides with. The fields Genkit builds from
// the request are rejected loudly: the schema cannot see inside the
// passthrough, and clearing them silently would hide that the config tried to
// replace the conversation.
func forwardRequestExtra(params *openai.ChatCompletionNewParams, extra map[string]any) error {
	if len(extra) == 0 {
		return nil
	}
	for _, field := range managedRequestFields {
		if _, ok := extra[field]; ok {
			return status.Errorf(status.ErrInvalidArgument, "compat_oai: extra field %q is built by Genkit from the request and cannot be set from config", field)
		}
	}
	AddExtraFields(params, extra)
	return nil
}

// forwardEmbeddingExtra is [forwardRequestExtra] for the embeddings request,
// whose only Genkit-built field is the input. The model stays settable: it is
// this request's version pin.
func forwardEmbeddingExtra(params *openai.EmbeddingNewParams, extra map[string]any) error {
	if len(extra) == 0 {
		return nil
	}
	if _, ok := extra["input"]; ok {
		return status.Errorf(status.ErrInvalidArgument, "compat_oai: extra field %q is built by Genkit from the request and cannot be set from config", "input")
	}
	params.SetExtraFields(maps.Clone(extra))
	return nil
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
		if err := forwardEmbeddingExtra(&params, config.Extra); err != nil {
			return nil, err
		}

		embeddingResp, err := o.clientForKey(config.APIKey).Embeddings.New(ctx, params)
		if err != nil {
			return nil, WrapAPIError(err)
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
		return nil, status.Errorf(status.ErrInvalidOutput, "compat_oai: decoding base64 embedding: %w", err)
	}
	if len(data)%4 != 0 {
		return nil, status.Errorf(status.ErrInvalidOutput, "compat_oai: base64 embedding has %d bytes, want a multiple of 4", len(data))
	}
	embedding := make([]float32, len(data)/4)
	for i := range embedding {
		embedding[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return embedding, nil
}

// IsDefinedEmbedder reports whether the named [Embedder] is defined by this
// plugin. name is the full action name, provider prefix included.
//
// Deprecated: this existed to guard a registration call that panics on a
// duplicate. Embedder options now come from a plugin's Embedders field, which
// nothing has to register and no ordering can defeat, leaving this a question
// about registry state that applications do not need to ask.
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
// name is the full action name, provider prefix included.
//
// Deprecated: this existed to guard a registration call that panics on a
// duplicate. Capabilities now come from a plugin's Models field, which nothing
// has to register and no ordering can defeat, leaving this a question about
// registry state that applications do not need to ask.
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
	opts.Label = internal.ProviderLabel(provider, id)
	return opts
}

// DefaultModelOptions is the capability set advertised for models that are
// discovered or resolved dynamically rather than curated by a plugin. Each
// call returns its own copy of the capabilities, so a caller that adjusts
// them adjusts one model's description rather than the package's default.
func DefaultModelOptions() ai.ModelOptions {
	supports := Multimodal
	return ai.ModelOptions{
		Stage:    ai.ModelStageStable,
		Versions: []string{},
		Supports: &supports,
	}
}

// ActionName builds the action name for a model or embedder ID under
// provider, taking the ID either bare or already provider-prefixed. The
// prefix is applied by concatenation, so without the trim an
// already-prefixed name would double up and name an action that resolves
// nowhere.
func ActionName(provider, id string) string {
	return api.NewName(provider, internal.TrimProvider(provider, id))
}

// ModelOptionsFor resolves the options a plugin describes a model with:
// curated capabilities for a known ID, dynamic ones for the rest, and a
// caller's own entry from models overlaid on whichever applies. Overlaying
// rather than replacing lets an entry pin one capability without restating
// the label, the versions and the rest.
//
// This is how a plugin's catalog stays correctable. Every path that describes
// a model goes through it, so a caller's entry is authoritative no matter
// whether Init, ListActions or ResolveAction gets there first, and it applies
// to the models Init registers, which nothing can re-register afterwards.
//
// id is the bare model ID; models is keyed either bare or provider-prefixed
// (see [internal.LookupOverride]).
func ModelOptionsFor(provider, id string, curated map[string]ai.ModelOptions, dynamic ai.ModelOptions, models map[string]ai.ModelOptions) ai.ModelOptions {
	opts, ok := curated[id]
	if !ok {
		opts = dynamic
	}
	if override, ok := internal.LookupOverride(models, provider, id); ok {
		return opts.Overlay(override)
	}
	return opts
}

// ListModelActions lists the models the provider's API reports, each described
// by modelOptions and the SDK config schema. It is [ListChatActions] for a
// plugin whose config is the SDK request type: the plugin passes the same
// options lookup its ResolveAction and its Init use, so no path can describe a
// model differently from the others.
//
// [OpenAICompatible.ListActions] is the same listing for a plugin with no
// catalog of its own, describing every model with the generic defaults.
func ListModelActions(ctx context.Context, o *OpenAICompatible, modelOptions func(id string) ai.ModelOptions) []api.ActionDesc {
	return listActions(ctx, o, func(id string) api.ActionDesc {
		return newSDKModel(o.client, o.Provider, id, modelOptions(id)).Desc()
	})
}

// ResolveModelAction resolves a model not registered up front, described by
// modelOptions and the SDK config schema; see [ListModelActions].
func ResolveModelAction(o *OpenAICompatible, atype api.ActionType, id string, modelOptions func(id string) ai.ModelOptions) api.Action {
	switch atype {
	case api.ActionTypeModel:
		return newSDKModel(o.client, o.Provider, id, modelOptions(id))
	}
	return nil
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
		return nil, WrapAPIError(err)
	}

	return models, nil
}
