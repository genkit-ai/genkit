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

// Package openai contains the Genkit Plugin implementation for OpenAI provider
package openai

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

var (
	openaiLabelPrefix = "OpenAI"
	openaiProvider    = "openai"
	defaultOpenAIOpts = ai.ModelOptions{
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportAll,
		},
		Versions: []string{},
		Stage:    ai.ModelStageUnstable,
	}
)

var defaultEmbedOpts = ai.EmbedderOptions{}

type OpenAI struct {
	mu          sync.Mutex             // protects concurrent access to the client and init state
	initted     bool                   // tracks weter the plugin has been initialized
	client      *openai.Client         // openAI client used for making requests
	Opts        []option.RequestOption // request options for the OpenAI client
	APIKey      string                 // API key to use with the desired plugin
	BaseURL     string                 // Base URL for custom endpoints
	Provider    string                 // Provider name (defaults to "openai")
	LabelPrefix string                 // Provider label prefix (defaults to "OpenAI")
}

func (o *OpenAI) Name() string {
	return openaiProvider
}

func (o *OpenAI) Init(ctx context.Context) []api.Action {
	if o == nil {
		o = &OpenAI{}
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.initted {
		panic("plugin already initialized")
	}

	apiKey := o.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey != "" {
		o.Opts = append([]option.RequestOption{option.WithAPIKey(apiKey)}, o.Opts...)
	}

	baseURL := o.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("OPENAI_BASE_URL")
	}
	if baseURL != "" {
		o.Opts = append([]option.RequestOption{option.WithBaseURL(baseURL)}, o.Opts...)
	}

	if o.Provider != "" {
		openaiProvider = o.Provider
	}
	if o.LabelPrefix != "" {
		openaiLabelPrefix = o.LabelPrefix
	}

	client := openai.NewClient(o.Opts...)
	o.client = &client
	o.initted = true

	return []api.Action{}
}

// DefineModel defines an unknown model with the given name.
func (o *OpenAI) DefineModel(g *genkit.Genkit, name string, opts *ai.ModelOptions) (ai.Model, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAI.Init not called")
	}
	if name == "" {
		return nil, fmt.Errorf("OpenAI.DefineModel: called with empty model name")
	}

	if opts == nil {
		return nil, fmt.Errorf("OpenAI.DefineModel: called with unknown model options")
	}
	return newModel(o.client, name, opts), nil
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not previously defined.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(openaiProvider, name))
}

// ModelRef creates a new ModelRef for an OpenAI model with the given ID and configuration
func ModelRef(name string, config *responses.ResponseNewParams) ai.ModelRef {
	return ai.NewModelRef(openaiProvider+"/"+name, config)
}

// IsDefinedModel reports whether the named [ai.Model] is defined by this plugin
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, name) != nil
}

// DefineEmbedder defines an embedder with a given name
func (o *OpenAI) DefineEmbedder(g *genkit.Genkit, name string, embedOpts *ai.EmbedderOptions) (ai.Embedder, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAI.Init not called")
	}
	return newEmbedder(o.client, name, embedOpts), nil
}

// Embedder returns the [ai.Embedder] with the given name.
// It returns nil if the embedder was not previously defined.
func Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, name)
}

// IsDefinedEmbedder reports whether the named [ai.Embedder] is defined by this plugin
func IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, name) != nil
}

// ListActions lists all the actions supported by the OpenAI plugin.
func (o *OpenAI) ListActions(ctx context.Context) []api.ActionDesc {
	actions := []api.ActionDesc{}
	models, err := listOpenAIModels(ctx, o.client)
	if err != nil {
		slog.Error("unable to fetch models from OpenAI API")
		return nil
	}

	for _, name := range models.chat {
		model := newModel(o.client, name, &defaultOpenAIOpts)
		if actionDef, ok := model.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}
	for _, e := range models.embedders {
		embedder := newEmbedder(o.client, e, &defaultEmbedOpts)
		if actionDef, ok := embedder.(api.Action); ok {
			actions = append(actions, actionDef.Desc())
		}
	}
	return actions
}

// ResolveAction resolves an action with the given name.
func (o *OpenAI) ResolveAction(atype api.ActionType, name string) api.Action {
	switch atype {
	case api.ActionTypeEmbedder:
		return newEmbedder(o.client, name, &ai.EmbedderOptions{}).(api.Action)
	case api.ActionTypeModel:
		var supports *ai.ModelSupports
		var config any

		switch {
		// TODO: add image and video models
		default:
			supports = &ai.ModelSupports{
				Multiturn:   true,
				Tools:       true,
				ToolChoice:  true,
				SystemRole:  true,
				Media:       true,
				Constrained: ai.ConstrainedSupportAll,
			}
			config = &responses.ResponseNewParams{}
		}
		return newModel(o.client, name, &ai.ModelOptions{
			Label:        fmt.Sprintf("%s - %s", openaiLabelPrefix, name),
			Stage:        ai.ModelStageStable,
			Versions:     []string{},
			Supports:     supports,
			ConfigSchema: configToMap(config),
		}).(api.Action)
	}
	return nil
}

// openaiModels contains the collection of supported OpenAI models
type openaiModels struct {
	chat      []string // gpt, tts, o1, o2, o3...
	image     []string // gpt-image
	video     []string // sora
	embedders []string // text-embedding...
}

// listOpenAIModels returns a list of models available in the OpenAI API
// The returned struct is a filtered list of models based on plain string comparisons:
// chat: gpt, tts, o1, o2, o3...
// image: gpt-image
// video: sora
// embedders: text-embedding
// NOTE: the returned list from the SDK is just a plain slice of model names.
// No extra information about the model stage or type is provided.
// See: platform.openai.com/docs/models
func listOpenAIModels(ctx context.Context, client *openai.Client) (openaiModels, error) {
	models := openaiModels{}
	iter := client.Models.ListAutoPaging(ctx)
	for iter.Next() {
		m := iter.Current()
		if strings.Contains(m.ID, "sora") {
			models.video = append(models.video, m.ID)
			continue
		}
		if strings.Contains(m.ID, "image") {
			models.image = append(models.image, m.ID)
			continue
		}
		if strings.Contains(m.ID, "embedding") {
			models.embedders = append(models.embedders, m.ID)
			continue
		}
		models.chat = append(models.chat, m.ID)
	}
	if err := iter.Err(); err != nil {
		return openaiModels{}, err
	}

	return models, nil
}

// newEmbedder creates a new embedder without registering it
func newEmbedder(client *openai.Client, name string, embedOpts *ai.EmbedderOptions) ai.Embedder {
	return ai.NewEmbedder(api.NewName(openaiProvider, name), embedOpts, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		var data openai.EmbeddingNewParamsInputUnion
		for _, doc := range req.Input {
			for _, p := range doc.Content {
				data.OfArrayOfStrings = append(data.OfArrayOfStrings, p.Text)
			}
		}

		params := openai.EmbeddingNewParams{
			Input:          openai.EmbeddingNewParamsInputUnion(data),
			Model:          name,
			EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
		}

		embeddingResp, err := client.Embeddings.New(ctx, params)
		if err != nil {
			return nil, err
		}

		resp := &ai.EmbedResponse{}
		for _, e := range embeddingResp.Data {
			embedding := make([]float32, len(e.Embedding))
			for i, v := range e.Embedding {
				embedding[i] = float32(v)
			}
			resp.Embeddings = append(resp.Embeddings, &ai.Embedding{Embedding: embedding})
		}
		return resp, nil
	})
}

// newModel creates a new model without registering it in the registry
func newModel(client *openai.Client, name string, opts *ai.ModelOptions) ai.Model {
	config := &responses.ResponseNewParams{}
	meta := &ai.ModelOptions{
		Label:        opts.Label,
		Supports:     opts.Supports,
		Versions:     opts.Versions,
		ConfigSchema: configToMap(config),
		Stage:        opts.Stage,
	}

	fn := func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		// TODO: add support for imagen and video
		return generate(ctx, client, name, input, cb)
	}

	return ai.NewModel(api.NewName(openaiProvider, name), meta, fn)
}
