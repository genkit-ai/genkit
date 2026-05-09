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
	"fmt"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
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

// DefineModel defines a model in the registry
func (o *OpenAICompatible) DefineModel(provider, id string, opts ai.ModelOptions) ai.Model {
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
		// Configure the response generator with input
		generator := NewModelGenerator(o.client, id).WithMessages(input.Messages).WithConfig(input.Config).WithTools(input.Tools)

		// Generate response
		resp, err := generator.Generate(ctx, input, cb)
		if err != nil {
			return nil, err
		}

		return resp, nil
	})
}

// DefineEmbedder defines an embedder with a given name.
func (o *OpenAICompatible) DefineEmbedder(provider, name string, embedOpts *ai.EmbedderOptions) ai.Embedder {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("OpenAICompatible.Init not called")
	}

	return ai.NewEmbedder(api.NewName(provider, name), embedOpts, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
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

		embeddingResp, err := o.client.Embeddings.New(ctx, params)
		if err != nil {
			return nil, err
		}

		resp := &ai.EmbedResponse{}
		for _, emb := range embeddingResp.Data {
			embedding := make([]float32, len(emb.Embedding))
			for i, val := range emb.Embedding {
				embedding[i] = float32(val)
			}
			resp.Embeddings = append(resp.Embeddings, &ai.Embedding{Embedding: embedding})
		}
		return resp, nil
	})
}

// IsDefinedEmbedder reports whether the named [Embedder] is defined by this plugin.
func (o *OpenAICompatible) IsDefinedEmbedder(g *genkit.Genkit, name string) bool {
	return genkit.LookupEmbedder(g, name) != nil
}

// Embedder returns the [ai.Embedder] with the given name.
// It returns nil if the embedder was not defined.
func (o *OpenAICompatible) Embedder(g *genkit.Genkit, name string) ai.Embedder {
	return genkit.LookupEmbedder(g, name)
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not defined.
func (o *OpenAICompatible) Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, name)
}

// IsDefinedModel reports whether the named [Model] is defined by this plugin.
func (o *OpenAICompatible) IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, name) != nil
}

func (o *OpenAICompatible) ListActions(ctx context.Context) []api.ActionDesc {
	actions := []api.ActionDesc{}

	models, err := listOpenAIModels(ctx, o.client)
	if err != nil {
		return nil
	}
	for _, name := range models {
		metadata := map[string]any{
			"model": map[string]any{
				"supports": map[string]any{
					"media":       true,
					"multiturn":   true,
					"systemRole":  true,
					"tools":       true,
					"toolChoice":  true,
					"constrained": "all",
				},
			},
			"versions": []string{},
			"stage":    string(ai.ModelStageStable),
		}
		metadata["label"] = fmt.Sprintf("%s - %s", o.Provider, name)

		actions = append(actions, api.ActionDesc{
			Type:     api.ActionTypeModel,
			Name:     fmt.Sprintf("%s/%s", o.Provider, name),
			Key:      fmt.Sprintf("/%s/%s/%s", api.ActionTypeModel, o.Provider, name),
			Metadata: metadata,
		})
	}

	return actions
}

func (o *OpenAICompatible) ResolveAction(atype api.ActionType, name string) api.Action {
	switch atype {
	case api.ActionTypeModel:
		if model := o.DefineModel(o.Provider, name, ai.ModelOptions{
			Label:    fmt.Sprintf("%s - %s", o.Provider, name),
			Stage:    ai.ModelStageStable,
			Versions: []string{},
			Supports: &Multimodal,
		}); model != nil {
			return model.(api.Action)
		}
	}

	return nil
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
