// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

// newEmbedder creates an embedder without registering it. The framework
// validates and deserializes the request's options into
// [genai.EmbedContentConfig] before the embedder function runs; the config
// schema is inferred from that type unless the caller overrides it.
func newEmbedder(client *genai.Client, name string, embedOpts *ai.EmbedderOptions) *ai.EmbedderAction {
	provider := googleAIProvider
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}

	return ai.NewEmbedderAction(api.NewName(provider, name), embedOpts, func(ctx context.Context, req *ai.EmbedRequest, embedConfig genai.EmbedContentConfig) (*ai.EmbedResponse, error) {
		var content []*genai.Content

		for _, doc := range req.Input {
			parts, err := toGeminiParts(doc.Content)
			if err != nil {
				return nil, err
			}
			content = append(content, &genai.Content{
				Parts: parts,
			})
		}

		r, err := client.Models.EmbedContent(ctx, name, content, &embedConfig)
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
