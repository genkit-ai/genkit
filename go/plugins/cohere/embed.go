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
//
// SPDX-License-Identifier: Apache-2.0

package cohere

import (
	"context"
	"errors"
	"fmt"
	"strings"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal"
)

// EmbedOptions configures a Cohere embedding request. Pass it to
// [NewEmbedderRef] to carry typed configuration with the embedder.
type EmbedOptions struct {
	// InputType tunes the embedding for the downstream task. One of
	// "search_document" (default), "search_query", "classification" or
	// "clustering".
	InputType string `json:"inputType,omitempty"`
	// OutputDimension overrides the embedding dimensionality. Only supported by
	// embed-v4 and newer models (256, 512, 1024 or 1536).
	OutputDimension int `json:"outputDimension,omitempty"`
	// Truncate controls handling of over-length inputs: "NONE", "START" or "END".
	Truncate string `json:"truncate,omitempty"`
	// EmbeddingType selects the returned representation: "float" (default),
	// "int8", "uint8", "binary" or "ubinary". Quantized and packed integer
	// values are converted to float32 values in Genkit's embedding response.
	EmbeddingType cohere.EmbeddingType `json:"embeddingType,omitempty"`
}

// newEmbedder creates an embedder without registering it.
func (c *Cohere) newEmbedder(id string) *ai.EmbedderAction {
	id = internal.TrimProvider(provider, id)
	info := GetEmbedderOptions(id)
	embedOpts := &ai.EmbedderOptions{
		Label:      info.Label,
		Dimensions: info.Dimensions,
		Supports:   &ai.EmbedderSupports{Input: []string{"text"}},
	}

	client := c.client
	return ai.NewEmbedderAction(actionName(id), embedOpts, func(ctx context.Context, req *ai.EmbedRequest, options EmbedOptions) (*ai.EmbedResponse, error) {
		return embed(ctx, client, id, req, options)
	})
}

// NewEmbedderRef names a Cohere embedder and carries its typed configuration.
// A nil config leaves the request configuration unset.
func NewEmbedderRef(id string, config *EmbedOptions) ai.EmbedderRef {
	return ai.NewEmbedderRef(actionName(id), config)
}

// embed runs a Cohere V2 Embed request over the request's documents.
func embed(ctx context.Context, client *cohereclient.Client, name string, req *ai.EmbedRequest, opts EmbedOptions) (*ai.EmbedResponse, error) {
	embeddingType, err := normalizeEmbeddingType(opts.EmbeddingType)
	if err != nil {
		return nil, err
	}

	inputType := cohere.EmbedInputTypeSearchDocument
	if opts.InputType != "" {
		inputType = cohere.EmbedInputType(opts.InputType)
	}

	texts := make([]string, 0, len(req.Input))
	for _, doc := range req.Input {
		texts = append(texts, documentText(doc))
	}

	embedReq := &cohere.V2EmbedRequest{
		Model:          name,
		Texts:          texts,
		InputType:      inputType,
		EmbeddingTypes: []cohere.EmbeddingType{embeddingType},
	}
	if opts.OutputDimension > 0 {
		dim := opts.OutputDimension
		embedReq.OutputDimension = &dim
	}
	if opts.Truncate != "" {
		truncate := cohere.V2EmbedRequestTruncate(opts.Truncate)
		embedReq.Truncate = &truncate
	}

	resp, err := client.V2.Embed(ctx, embedReq)
	if err != nil {
		return nil, fmt.Errorf("cohere: %w", err)
	}

	vectors, err := embeddingVectors(resp.Embeddings, embeddingType)
	if err != nil {
		return nil, err
	}
	res := ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(vectors))}
	for _, vector := range vectors {
		res.Embeddings = append(res.Embeddings, &ai.Embedding{Embedding: vector})
	}
	return &res, nil
}

func normalizeEmbeddingType(value cohere.EmbeddingType) (cohere.EmbeddingType, error) {
	if value == "" {
		return cohere.EmbeddingTypeFloat, nil
	}
	switch value {
	case cohere.EmbeddingTypeFloat,
		cohere.EmbeddingTypeInt8,
		cohere.EmbeddingTypeUint8,
		cohere.EmbeddingTypeBinary,
		cohere.EmbeddingTypeUbinary:
		return value, nil
	default:
		return "", fmt.Errorf("cohere: unsupported embedding type %q", value)
	}
}

func embeddingVectors(embeddings *cohere.EmbedByTypeResponseEmbeddings, embeddingType cohere.EmbeddingType) ([][]float32, error) {
	if embeddings == nil {
		return nil, errors.New("cohere: embed response did not contain embeddings")
	}

	var vectors [][]float32
	switch embeddingType {
	case cohere.EmbeddingTypeFloat:
		vectors = floatVectors(embeddings.Float)
	case cohere.EmbeddingTypeInt8:
		vectors = intVectors(embeddings.Int8)
	case cohere.EmbeddingTypeUint8:
		vectors = intVectors(embeddings.Uint8)
	case cohere.EmbeddingTypeBinary:
		vectors = intVectors(embeddings.Binary)
	case cohere.EmbeddingTypeUbinary:
		vectors = intVectors(embeddings.Ubinary)
	}
	if vectors == nil {
		return nil, fmt.Errorf("cohere: embed response did not contain %q embeddings", embeddingType)
	}
	return vectors, nil
}

// documentText concatenates the text parts of a document.
func documentText(doc *ai.Document) string {
	var sb strings.Builder
	for _, p := range doc.Content {
		if p.IsText() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// floatVectors narrows float64 embedding vectors to float32 for ai.Embedding.
func floatVectors(in [][]float64) [][]float32 {
	out := make([][]float32, len(in))
	for i, vector := range in {
		out[i] = make([]float32, len(vector))
		for j, value := range vector {
			out[i][j] = float32(value)
		}
	}
	return out
}

func intVectors(in [][]int) [][]float32 {
	out := make([][]float32, len(in))
	for i, vector := range in {
		out[i] = make([]float32, len(vector))
		for j, value := range vector {
			out[i][j] = float32(value)
		}
	}
	return out
}
