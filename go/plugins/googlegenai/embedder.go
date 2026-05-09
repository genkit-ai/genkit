// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

// newEmbedder creates an embedder without registering it.
func newEmbedder(client *genai.Client, name string, embedOpts *ai.EmbedderOptions) ai.Embedder {
	provider := googleAIProvider
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}

	if embedOpts.ConfigSchema == nil {
		embedOpts.ConfigSchema = core.InferSchemaMap(genai.EmbedContentConfig{})
	}

	return ai.NewEmbedder(api.NewName(provider, name), embedOpts, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		var content []*genai.Content
		var embedConfig *genai.EmbedContentConfig

		if config, ok := req.Options.(*genai.EmbedContentConfig); ok {
			embedConfig = config
		}

		for _, doc := range req.Input {
			parts, err := toGeminiParts(doc.Content)
			if err != nil {
				return nil, err
			}
			content = append(content, &genai.Content{
				Parts: parts,
			})
		}

		r, err := genai.Models.EmbedContent(*client.Models, ctx, name, content, embedConfig)
		if err != nil {
			return nil, err
		}
		var res ai.EmbedResponse
		for _, emb := range r.Embeddings {
			res.Embeddings = append(res.Embeddings, &ai.Embedding{Embedding: emb.Values})
		}
		return &res, nil
	})
}
