// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

// embedConfigSchema caches the embed config schema so that dynamic discovery
// paths (listActions, resolveAction), which construct an embedder per call,
// don't re-run schema reflection every time.
var embedConfigSchema = sync.OnceValue(func() map[string]any {
	return core.InferSchemaMap(genai.EmbedContentConfig{})
})

// newEmbedder creates an embedder without registering it.
func newEmbedder(client *genai.Client, name string, embedOpts *ai.EmbedderOptions) *ai.Embedder {
	provider := googleAIProvider
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}

	if embedOpts.ConfigSchema == nil {
		embedOpts.ConfigSchema = embedConfigSchema()
	}

	return ai.NewEmbedder(api.NewName(provider, name), embedOpts, func(ctx context.Context, req *ai.EmbedRequest, embedConfig genai.EmbedContentConfig) (*ai.EmbedResponse, error) {
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

		r, err := genai.Models.EmbedContent(*client.Models, ctx, name, content, &embedConfig)
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
