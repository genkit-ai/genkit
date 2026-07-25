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

// Package fakeembedder provides a fake implementation of
// genkit.Embedder for testing purposes.
// The caller must register the values that the fake embedder should
// return for each document. Asking for the values of an unregistered
// document panics.
package fakeembedder

import (
	"context"
	"errors"

	"github.com/firebase/genkit/go/ai"
)

// Embedder is a fake implementation of an Embedder.
type Embedder struct {
	registry map[*ai.Document][]float32
}

// New returns a new fake embedder.
func New() *Embedder {
	return &Embedder{
		registry: make(map[*ai.Document][]float32),
	}
}

// Register records the value to return for a Document.
func (e *Embedder) Register(d *ai.Document, vals []float32) {
	e.registry[d] = vals
}

func (e *Embedder) Embed(ctx context.Context, req *ai.EmbedRequest, _ any) (*ai.EmbedResponse, error) {
	res := &ai.EmbedResponse{}
	for _, doc := range req.Input {
		vals, ok := e.registry[doc]
		if !ok {
			return nil, errors.New("fake embedder called with unregistered document")
		}
		res.Embeddings = append(res.Embeddings, &ai.Embedding{Embedding: vals})
	}
	return res, nil
}
