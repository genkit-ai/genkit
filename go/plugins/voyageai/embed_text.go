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
	"errors"
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// TextEmbedConfig is the per-request configuration of the text embedders
// (voyage-4, voyage-3.5, voyage-code-4, and the rest of the text catalog). It
// goes through [ai.WithConfig]; every field is optional.
type TextEmbedConfig struct {
	// InputType is "document" or "query". Voyage prepends a retrieval prompt
	// for the type; leave it empty to embed the text as is.
	InputType string `json:"inputType,omitempty" jsonschema:"enum=document,enum=query"`
	// OutputDimension asks for a vector of that size. It must be one of the
	// dimensions the model offers: 1024 (the default), 2048, 512, or 256, or
	// only 1024 for voyage-finance-2 and voyage-law-2.
	OutputDimension int `json:"outputDimension,omitempty"`
	// Truncation controls whether an input longer than the model's context
	// is truncated (the API default) or rejected.
	Truncation *bool `json:"truncation,omitempty"`
}

type textRequest struct {
	Input           []string `json:"input"`
	Model           string   `json:"model"`
	InputType       string   `json:"input_type,omitempty"`
	OutputDimension int      `json:"output_dimension,omitempty"`
	Truncation      *bool    `json:"truncation,omitempty"`
}

type textResponse struct {
	Data []embeddingItem `json:"data"`
}

func embedText(ctx context.Context, c *client, m *model, req *ai.EmbedRequest, cfg TextEmbedConfig) (*ai.EmbedResponse, error) {
	if len(req.Input) == 0 {
		return emptyResponse(), nil
	}
	if err := checkInputCount(len(req.Input)); err != nil {
		return nil, err
	}
	if err := checkInputType(cfg.InputType, false); err != nil {
		return nil, err
	}
	if err := m.validateDimension(cfg.OutputDimension); err != nil {
		return nil, err
	}
	texts := make([]string, len(req.Input))
	for i, doc := range req.Input {
		text, err := documentText(doc)
		if err != nil {
			return nil, invalidArgument("input %d: %v (use a voyage-multimodal model for media)", i, err)
		}
		texts[i] = text
	}

	var resp textResponse
	err := c.post(ctx, "/v1/embeddings", textRequest{
		Input:           texts,
		Model:           m.name,
		InputType:       cfg.InputType,
		OutputDimension: cfg.OutputDimension,
		Truncation:      cfg.Truncation,
	}, &resp)
	if err != nil {
		return nil, err
	}
	vectors, err := orderByIndex(resp.Data, len(texts))
	if err != nil {
		return nil, err
	}
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, len(vectors))}
	for i, v := range vectors {
		out.Embeddings[i] = &ai.Embedding{Embedding: v}
	}
	return out, nil
}

// documentText joins the text parts of doc. A document with any other kind of
// part is rejected rather than embedded without it.
func documentText(doc *ai.Document) (string, error) {
	if doc == nil {
		return "", errors.New("document is nil")
	}
	var b strings.Builder
	for _, p := range doc.Content {
		if p == nil || !p.IsText() {
			return "", errors.New("only text parts are supported")
		}
		b.WriteString(p.Text)
	}
	return b.String(), nil
}
