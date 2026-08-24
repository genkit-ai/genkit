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
	"fmt"
	"strings"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

// EmbedOptions configures a Cohere embedding request. Pass it via
// ai.WithEmbedderOptions / the EmbedRequest.Options field.
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
}

// newEmbedder creates an embedder without registering it.
func (c *Cohere) newEmbedder(name string) ai.Embedder {
	info := GetEmbedderOptions(name)
	embedOpts := &ai.EmbedderOptions{
		Label:        info.Label,
		Dimensions:   info.Dimensions,
		ConfigSchema: core.InferSchemaMap(EmbedOptions{}),
		Supports:     &ai.EmbedderSupports{Input: []string{"text"}},
	}

	client := c.client
	return ai.NewEmbedder(api.NewName(provider, name), embedOpts, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		return embed(ctx, client, name, req)
	})
}

// embed runs a Cohere V2 Embed request over the request's documents.
func embed(ctx context.Context, client *cohereclient.Client, name string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	opts := embedOptionsFromRequest(req.Options)

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
		EmbeddingTypes: []cohere.EmbeddingType{cohere.EmbeddingTypeFloat},
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

	var res ai.EmbedResponse
	if resp.Embeddings != nil {
		for _, vec := range resp.Embeddings.Float {
			res.Embeddings = append(res.Embeddings, &ai.Embedding{Embedding: toFloat32(vec)})
		}
	}
	return &res, nil
}

// embedOptionsFromRequest extracts EmbedOptions from the request options,
// accepting the struct, a pointer, or a map. Unrecognized values yield defaults.
func embedOptionsFromRequest(options any) EmbedOptions {
	switch o := options.(type) {
	case EmbedOptions:
		return o
	case *EmbedOptions:
		if o != nil {
			return *o
		}
	case map[string]any:
		if opts, err := base.MapToStruct[EmbedOptions](o); err == nil {
			return opts
		}
	}
	return EmbedOptions{}
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

// toFloat32 narrows a float64 embedding vector to float32 for ai.Embedding.
func toFloat32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
