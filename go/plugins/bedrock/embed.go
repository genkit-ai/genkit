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

package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/firebase/genkit/go/ai"
)

// embedderFunc dispatches to the Titan or Cohere request shape based on the
// model-ID prefix. Bedrock embedders are InvokeModel-based; the SDK's body
// is opaque []byte and the JSON shape is provider-specific.
func embedderFunc(client *bedrockruntime.Client, modelID string) ai.EmbedderFunc {
	return func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		if len(req.Input) == 0 {
			return &ai.EmbedResponse{}, nil
		}
		switch {
		case strings.HasPrefix(modelID, "amazon.titan-embed-"):
			return embedTitan(ctx, client, modelID, req)
		case strings.HasPrefix(modelID, "cohere.embed-"):
			return embedCohere(ctx, client, modelID, req)
		default:
			return nil, fmt.Errorf("bedrock: unrecognised embedder model %q (expected amazon.titan-embed-* or cohere.embed-*)", modelID)
		}
	}
}

// embedTitan submits one InvokeModel call per input document. Titan embedders
// accept a single text per call and return a single vector.
type titanEmbedReq struct {
	InputText string `json:"inputText"`
}

type titanEmbedResp struct {
	Embedding []float32 `json:"embedding"`
}

func embedTitan(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(req.Input))}
	for i, doc := range req.Input {
		text := docText(doc)
		if text == "" {
			return nil, fmt.Errorf("bedrock: titan embedder: document %d has no text", i)
		}
		var resp titanEmbedResp
		if err := invokeJSON(ctx, client, modelID, titanEmbedReq{InputText: text}, &resp); err != nil {
			return nil, err
		}
		out.Embeddings = append(out.Embeddings, &ai.Embedding{Embedding: resp.Embedding})
	}
	return out, nil
}

// embedCohere batches every input document into a single InvokeModel call.
// Cohere accepts an array of texts and returns parallel embeddings.
type cohereEmbedReq struct {
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
}

type cohereEmbedResp struct {
	Embeddings [][]float32 `json:"embeddings"`
}

const cohereInputTypeDefault = "search_document"

func embedCohere(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	texts := make([]string, 0, len(req.Input))
	for i, doc := range req.Input {
		t := docText(doc)
		if t == "" {
			return nil, fmt.Errorf("bedrock: cohere embedder: document %d has no text", i)
		}
		texts = append(texts, t)
	}
	var resp cohereEmbedResp
	if err := invokeJSON(ctx, client, modelID, cohereEmbedReq{Texts: texts, InputType: cohereInputTypeDefault}, &resp); err != nil {
		return nil, err
	}
	if len(resp.Embeddings) != len(texts) {
		return nil, fmt.Errorf("bedrock: cohere embedder: got %d embeddings for %d texts", len(resp.Embeddings), len(texts))
	}
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(resp.Embeddings))}
	for _, e := range resp.Embeddings {
		out.Embeddings = append(out.Embeddings, &ai.Embedding{Embedding: e})
	}
	return out, nil
}

// docText concatenates the text parts of a document. Non-text parts (media)
// are skipped — embedders only see text content.
func docText(d *ai.Document) string {
	if d == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range d.Content {
		if p.Text != "" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// invokeJSON marshals in, calls InvokeModel, and decodes the response body
// into out. Used by both embedders and the image-gen helpers.
func invokeJSON(ctx context.Context, client *bedrockruntime.Client, modelID string, in any, out any) error {
	if client == nil {
		return errors.New("bedrock: client is nil")
	}
	body, err := json.Marshal(in)
	if err != nil {
		return fmt.Errorf("bedrock: marshal request: %w", err)
	}
	resp, err := client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return fmt.Errorf("bedrock: InvokeModel(%s): %w", modelID, err)
	}
	if err := json.Unmarshal(resp.Body, out); err != nil {
		return fmt.Errorf("bedrock: decode response body: %w", err)
	}
	return nil
}
