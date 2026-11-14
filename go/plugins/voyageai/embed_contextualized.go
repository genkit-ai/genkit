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
//
// SPDX-License-Identifier: Apache-2.0

package voyageai

import (
	"context"
	"fmt"
	"sort"

	"github.com/firebase/genkit/go/ai"
)

// autoChunkSize is the chunk size, in tokens, sent with every document-mode
// request: the largest the API accepts, so a document is split only when it
// exceeds the model's per-chunk context.
const autoChunkSize = 32000

// ContextualizedEmbedConfig is the per-request configuration of the
// contextualized chunk embedders (voyage-context-4 and voyage-context-3).
type ContextualizedEmbedConfig struct {
	// InputType is required: "document" to embed full documents, which Voyage
	// splits into chunks and embeds with the context of the whole document, or
	// "query" to embed search queries.
	InputType string `json:"inputType" jsonschema:"enum=document,enum=query"`
	// OutputDimension asks for a vector of that size: 1024 (the default),
	// 2048, 512, or 256.
	OutputDimension int `json:"outputDimension,omitempty"`
}

type contextualizedRequest struct {
	Inputs             []string `json:"inputs"`
	Model              string   `json:"model"`
	InputType          string   `json:"input_type"`
	OutputDimension    int      `json:"output_dimension,omitempty"`
	EnableAutoChunking bool     `json:"enable_auto_chunking,omitempty"`
	ChunkSize          int      `json:"chunk_size,omitempty"`
}

type contextualizedResponse struct {
	Data []struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
			Text      *string   `json:"text"`
		} `json:"data"`
		Index int `json:"index"`
	} `json:"data"`
}

// embedContextualized embeds each input document as one string. In document
// mode Voyage chunks the document itself, and the response holds one
// embedding per chunk, flattened in (document, chunk) order, each with
// metadata documentIndex, chunkIndex, and, when the API chunked, text. In query
// mode each input is one query and yields one embedding.
func embedContextualized(ctx context.Context, c *client, m *model, req *ai.EmbedRequest, cfg ContextualizedEmbedConfig) (*ai.EmbedResponse, error) {
	if len(req.Input) == 0 {
		return emptyResponse(), nil
	}
	if err := checkInputCount(len(req.Input)); err != nil {
		return nil, err
	}
	if err := checkInputType(cfg.InputType, true); err != nil {
		return nil, err
	}
	if err := m.validateDimension(cfg.OutputDimension); err != nil {
		return nil, err
	}
	inputs := make([]string, len(req.Input))
	for i, doc := range req.Input {
		if doc == nil || len(doc.Content) != 1 || doc.Content[0] == nil || !doc.Content[0].IsText() {
			return nil, invalidArgument("input %d: a contextualized embedder takes exactly one text part per document, each document being embedded whole (pre-chunked documents are not supported, Voyage chunks them), and no media", i)
		}
		inputs[i] = doc.Content[0].Text
	}

	body := contextualizedRequest{
		Inputs:          inputs,
		Model:           m.name,
		InputType:       cfg.InputType,
		OutputDimension: cfg.OutputDimension,
	}
	if cfg.InputType == InputTypeDocument {
		body.EnableAutoChunking = true
		body.ChunkSize = autoChunkSize
	}
	var resp contextualizedResponse
	if err := c.post(ctx, "/v1/contextualizedembeddings", body, &resp); err != nil {
		return nil, err
	}
	if len(resp.Data) != len(inputs) {
		return nil, fmt.Errorf("voyageai: response has %d results for %d inputs", len(resp.Data), len(inputs))
	}

	sort.SliceStable(resp.Data, func(a, b int) bool { return resp.Data[a].Index < resp.Data[b].Index })
	out := &ai.EmbedResponse{}
	for doc, result := range resp.Data {
		if result.Index != doc {
			return nil, fmt.Errorf("voyageai: response has invalid or repeated result index %d", result.Index)
		}
		chunks := result.Data
		if len(chunks) == 0 {
			return nil, fmt.Errorf("voyageai: response has no embeddings for input %d", doc)
		}
		sort.SliceStable(chunks, func(a, b int) bool { return chunks[a].Index < chunks[b].Index })
		for chunk, e := range chunks {
			if e.Index != chunk {
				return nil, fmt.Errorf("voyageai: response has invalid or repeated chunk index %d for input %d", e.Index, doc)
			}
			if len(e.Embedding) == 0 {
				return nil, fmt.Errorf("voyageai: response has an empty embedding for input %d, chunk %d", doc, chunk)
			}
			meta := map[string]any{"documentIndex": doc, "chunkIndex": chunk}
			if e.Text != nil {
				meta["text"] = *e.Text
			}
			out.Embeddings = append(out.Embeddings, &ai.Embedding{Embedding: e.Embedding, Metadata: meta})
		}
	}
	return out, nil
}
