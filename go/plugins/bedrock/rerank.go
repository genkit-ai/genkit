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

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// RerankOptions configures a [Rerank] call. Pass it via
// [ai.RerankerRequest.Options].
type RerankOptions struct {
	// TopN caps the number of ranked documents returned. If <= 0, every input
	// document is returned, ordered by descending relevance.
	TopN int `json:"topN,omitempty"`
}

// cohereRerankReq is the Cohere Rerank InvokeModel body. api_version 2 is the
// current Bedrock contract.
type cohereRerankReq struct {
	Query      string   `json:"query"`
	Documents  []string `json:"documents"`
	TopN       int      `json:"top_n"`
	APIVersion int      `json:"api_version"`
}

type cohereRerankResp struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// Rerank reranks req.Documents by relevance to req.Query using a Bedrock
// reranking model (e.g. "cohere.rerank-v3-5:0") and returns them ordered by
// descending score, each annotated with its relevance score.
//
// The genkit framework does not yet expose a first-class reranker action, so
// this is a standalone call rather than a registered [ai.Reranker]. It looks up
// the already-initialised Bedrock plugin on g for its credentials and client.
func Rerank(ctx context.Context, g *genkit.Genkit, modelID string, req *ai.RerankerRequest) (*ai.RerankerResponse, error) {
	p, _ := genkit.LookupPlugin(g, provider).(*Bedrock)
	if p == nil {
		return nil, errors.New("bedrock plugin not registered; pass &bedrock.Bedrock{...} to genkit.WithPlugins")
	}
	if !p.initted {
		return nil, errors.New("bedrock.Rerank: plugin not initialized")
	}
	if modelID == "" {
		return nil, errors.New("bedrock.Rerank: model ID required")
	}
	if req == nil {
		return nil, errors.New("bedrock.Rerank: request required")
	}
	return rerank(ctx, p.client, modelID, req)
}

func rerank(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.RerankerRequest) (*ai.RerankerResponse, error) {
	query := docText(req.Query)
	if query == "" {
		return nil, errors.New("bedrock.Rerank: query has no text content")
	}
	docs := make([]string, 0, len(req.Documents))
	for i, d := range req.Documents {
		t := docText(d)
		if t == "" {
			return nil, fmt.Errorf("bedrock.Rerank: document %d has no text content", i)
		}
		docs = append(docs, t)
	}
	if len(docs) == 0 {
		return &ai.RerankerResponse{}, nil
	}

	topN := len(docs)
	if opts := rerankOptions(req.Options); opts != nil && opts.TopN > 0 && opts.TopN < topN {
		topN = opts.TopN
	}

	body := cohereRerankReq{
		Query:      query,
		Documents:  docs,
		TopN:       topN,
		APIVersion: 2,
	}
	var resp cohereRerankResp
	if err := invokeJSON(ctx, client, modelID, body, &resp); err != nil {
		return nil, err
	}
	return buildRerankResponse(resp, req.Documents)
}

// buildRerankResponse maps Cohere's score results back onto the original
// documents, preserving the reranked order and attaching each relevance score.
func buildRerankResponse(resp cohereRerankResp, docs []*ai.Document) (*ai.RerankerResponse, error) {
	out := &ai.RerankerResponse{Documents: make([]*ai.RankedDocumentData, 0, len(resp.Results))}
	for _, r := range resp.Results {
		if r.Index < 0 || r.Index >= len(docs) {
			return nil, fmt.Errorf("bedrock.Rerank: result index %d out of range for %d documents", r.Index, len(docs))
		}
		out.Documents = append(out.Documents, &ai.RankedDocumentData{
			Content:  docs[r.Index].Content,
			Metadata: &ai.RankedDocumentMetadata{Score: r.RelevanceScore},
		})
	}
	return out, nil
}

// rerankOptions extracts [RerankOptions] from the request's Options field,
// accepting either a value, pointer, or JSON-deserialised map. Returns nil
// when absent or malformed.
func rerankOptions(o any) *RerankOptions {
	switch v := o.(type) {
	case *RerankOptions:
		return v
	case RerankOptions:
		return &v
	case map[string]any:
		b, err := json.Marshal(v)
		if err != nil {
			return nil
		}
		var opts RerankOptions
		if err := json.Unmarshal(b, &opts); err != nil {
			return nil
		}
		return &opts
	default:
		return nil
	}
}
