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

package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

type EmbedOptions struct {
	Model string `json:"model"`
}

type ollamaEmbedRequest struct {
	Model   string         `json:"model"`
	Input   any            `json:"input"` // todo: using any to handle both string and []string, figure out better solution
	Options map[string]any `json:"options,omitempty"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func embed(ctx context.Context, serverAddress string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	options, ok := req.Options.(*EmbedOptions)
	if !ok && req.Options != nil {
		return nil, fmt.Errorf("invalid options type: expected *EmbedOptions")
	}
	if options == nil || options.Model == "" {
		return nil, fmt.Errorf("invalid embedding model: model must be specified")
	}

	if serverAddress == "" {
		return nil, fmt.Errorf("invalid server address: address cannot be empty")
	}

	ollamaReq := newOllamaEmbedRequest(options.Model, req.Input)

	jsonData, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embed request: %w", err)
	}

	resp, err := sendEmbedRequest(ctx, serverAddress, jsonData)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embed request failed with status code %d", resp.StatusCode)
	}

	var ollamaResp ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode embed response: %w", err)
	}

	return newEmbedResponse(ollamaResp.Embeddings), nil
}

func sendEmbedRequest(ctx context.Context, serverAddress string, jsonData []byte) (*http.Response, error) {
	client := &http.Client{}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", serverAddress+"/api/embed", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	return client.Do(httpReq)
}

func newOllamaEmbedRequest(model string, documents []*ai.Document) ollamaEmbedRequest {
	var input any
	if len(documents) == 1 {
		input = concatenateText(documents[0])
	} else {
		texts := make([]string, len(documents))
		for i, doc := range documents {
			texts[i] = concatenateText(doc)
		}
		input = texts
	}

	return ollamaEmbedRequest{
		Model: model,
		Input: input,
	}
}

func newEmbedResponse(embeddings [][]float32) *ai.EmbedResponse {
	resp := &ai.EmbedResponse{
		Embeddings: make([]*ai.Embedding, len(embeddings)),
	}
	for i, embedding := range embeddings {
		resp.Embeddings[i] = &ai.Embedding{Embedding: embedding}
	}
	return resp
}

func concatenateText(doc *ai.Document) string {
	var builder strings.Builder
	for _, part := range doc.Content {
		builder.WriteString(part.Text)
	}
	result := builder.String()
	return result
}

// DefineEmbedder defines an embedder with a given server address.
func (o *Ollama) DefineEmbedder(g *genkit.Genkit, serverAddress string, model string, embedOpts *ai.EmbedderOptions) ai.Embedder {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.initted {
		panic("ollama.Init not called")
	}
	return genkit.DefineEmbedder(g, api.NewName(provider, serverAddress), embedOpts, func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		if req.Options == nil {
			req.Options = &EmbedOptions{Model: model}
		} else if opts, ok := req.Options.(*EmbedOptions); ok {
			if opts.Model == "" {
				opts.Model = model
			}
		}
		return embed(ctx, serverAddress, req)
	})
}

// IsDefinedEmbedder reports whether the embedder with the given server address is defined by this plugin.
func IsDefinedEmbedder(g *genkit.Genkit, serverAddress string) bool {
	return genkit.LookupEmbedder(g, api.NewName(provider, serverAddress)) != nil
}

// Embedder returns the [ai.Embedder] with the given server address.
// It returns nil if the embedder was not defined.
func Embedder(g *genkit.Genkit, serverAddress string) ai.Embedder {
	return genkit.LookupEmbedder(g, api.NewName(provider, serverAddress))
}
