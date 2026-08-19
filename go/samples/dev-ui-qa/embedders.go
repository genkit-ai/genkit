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

package main

import (
	"context"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// registerEmbedderCases covers the embedder rows of "Discovery and action
// list" and "Model runner": cross-backend catalog leak (GGA-40), typo'd ID
// resolving as a live embedder (GGA-60), and multimodal input hitting an SDK
// refusal or process panic (GGA-32). Also home for the local retriever used
// by the retriever discovery row - it needs no plugin, so it works keyless.
func registerEmbedderCases(g *genkit.Genkit) {
	// TODO: embedder refs, including a deliberately misspelled ID and a
	// multimodal input fixture.

	corpus := []string{
		"Genkit is an open source framework for building AI-powered apps.",
		"The Dev UI lets you run flows, tools, and retrievers locally.",
		"Retrievers return ranked documents for a query.",
	}
	genkit.DefineRetriever(g, "staticRetriever",
		&ai.RetrieverOptions{Label: "Static QA retriever"},
		func(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
			query := strings.ToLower(req.Query.Content[0].Text)
			res := &ai.RetrieverResponse{}
			for i, text := range corpus {
				if query == "" || strings.Contains(strings.ToLower(text), query) {
					res.Documents = append(res.Documents, &ai.Document{
						Content:  []*ai.Part{ai.NewTextPart(text)},
						Metadata: map[string]any{"index": i},
					})
				}
			}
			return res, nil
		},
	)
}
