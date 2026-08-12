// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

// Per-request input limits for embedding calls. The Gemini API's
// batchEmbedContents endpoint accepts up to 100 requests per call; the
// Vertex AI prediction service accepts up to 250 instances per call.
const (
	googleAIEmbedBatchSize = 100
	vertexAIEmbedBatchSize = 250
)

// embedBatchSize returns how many inputs a single EmbedContent call may
// carry for the given backend and model. Vertex AI serves its Gemini
// embedding models and the open-source MaaS embedding models through
// endpoints that take one input per request (the SDK enforces this for most
// of them, and the gemini-embedding-001 prediction service enforces it
// server side), so those batch one at a time.
func embedBatchSize(backend genai.Backend, model string) int {
	if backend != genai.BackendVertexAI {
		return googleAIEmbedBatchSize
	}
	if strings.Contains(model, "gemini") || strings.Contains(model, "maas") {
		return 1
	}
	return vertexAIEmbedBatchSize
}

// newEmbedder creates an embedder without registering it. The framework
// validates and deserializes the request's options into
// [genai.EmbedContentConfig] before the embedder function runs; the config
// schema is inferred from that type unless the caller overrides it.
//
// Requests with more inputs than the service accepts per call are split into
// sequential batches transparently; the response carries one embedding per
// input, in input order.
func newEmbedder(client *genai.Client, id string, embedOpts *ai.EmbedderOptions) *ai.EmbedderAction {
	backend := client.ClientConfig().Backend
	provider := googleAIProvider
	if backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}
	batch := embedBatchSize(backend, id)

	return ai.NewEmbedderAction(api.NewName(provider, id), embedOpts, func(ctx context.Context, req *ai.EmbedRequest, embedConfig genai.EmbedContentConfig) (*ai.EmbedResponse, error) {
		content := make([]*genai.Content, 0, len(req.Input))
		for _, doc := range req.Input {
			parts, err := toGeminiParts(doc.Content)
			if err != nil {
				return nil, err
			}
			content = append(content, &genai.Content{
				Parts: parts,
			})
		}

		res := ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(content))}
		for group := range slices.Chunk(content, batch) {
			r, err := client.Models.EmbedContent(ctx, id, group, &embedConfig)
			if err != nil {
				return nil, wrapAPIError(err)
			}
			for _, emb := range r.Embeddings {
				res.Embeddings = append(res.Embeddings, &ai.Embedding{Embedding: emb.Values})
			}
		}
		return &res, nil
	})
}
