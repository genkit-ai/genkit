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

package ai

import (
	"context"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestEmbedderRef(t *testing.T) {
	t.Run("NewEmbedderRef creates ref with name and config", func(t *testing.T) {
		config := map[string]any{"dimension": 768}
		ref := NewEmbedderRef("test/embedder", config)

		if ref.Name() != "test/embedder" {
			t.Errorf("Name() = %q, want %q", ref.Name(), "test/embedder")
		}
		if diff := cmp.Diff(config, ref.Config()); diff != "" {
			t.Errorf("Config() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("NewEmbedderRef with nil config", func(t *testing.T) {
		ref := NewEmbedderRef("test/embedder", nil)

		if ref.Name() != "test/embedder" {
			t.Errorf("Name() = %q, want %q", ref.Name(), "test/embedder")
		}
		if ref.Config() != nil {
			t.Errorf("Config() = %v, want nil", ref.Config())
		}
	})
}

func TestNewEmbedder(t *testing.T) {
	t.Run("creates embedder with valid name", func(t *testing.T) {
		e := NewEmbedder("test/embedder", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return &EmbedResponse{}, nil
		})

		if e == nil {
			t.Fatal("expected embedder, got nil")
		}
		if e.Name() != "test/embedder" {
			t.Errorf("Name() = %q, want %q", e.Name(), "test/embedder")
		}
	})

	t.Run("panics with empty name", func(t *testing.T) {
		assertPanic(t, func() {
			NewEmbedder("", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
				return &EmbedResponse{}, nil
			})
		}, "name is required")
	})

	t.Run("applies options correctly", func(t *testing.T) {
		opts := &EmbedderOptions{
			Label:      "Test Embedder",
			Dimensions: 768,
			Supports: &EmbedderSupports{
				Input:        []string{"text", "image"},
				Multilingual: true,
			},
			ConfigSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"temperature": map[string]any{"type": "number"},
				},
			},
		}

		e := NewEmbedder("test/embedder", opts, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return &EmbedResponse{}, nil
		})

		if e == nil {
			t.Fatal("expected embedder, got nil")
		}
	})

	t.Run("uses defaults when options nil", func(t *testing.T) {
		e := NewEmbedder("test/embedder", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return &EmbedResponse{}, nil
		})

		if e == nil {
			t.Fatal("expected embedder, got nil")
		}
	})
}

func TestDefineEmbedder(t *testing.T) {
	t.Run("registers and returns embedder", func(t *testing.T) {
		r := newTestRegistry(t)
		called := false

		e := DefineEmbedder(r, "test/defineEmbedder", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			called = true
			return &EmbedResponse{
				Embeddings: []*Embedding{{Embedding: []float32{0.1, 0.2, 0.3}}},
			}, nil
		})

		if e == nil {
			t.Fatal("expected embedder, got nil")
		}

		// Verify it's registered by looking it up
		found := LookupEmbedder(r, "test/defineEmbedder")
		if found == nil {
			t.Fatal("LookupEmbedder returned nil for registered embedder")
		}

		// Verify the function works
		resp, err := e.Embed(context.Background(), &EmbedRequest{
			Input: []*Document{DocumentFromText("test", nil)},
		})
		assertNoError(t, err)
		if !called {
			t.Error("embedder function was not called")
		}
		if len(resp.Embeddings) != 1 {
			t.Errorf("len(Embeddings) = %d, want 1", len(resp.Embeddings))
		}
	})
}

func TestLookupEmbedder(t *testing.T) {
	t.Run("returns embedder when found", func(t *testing.T) {
		r := newTestRegistry(t)
		DefineEmbedder(r, "test/lookupEmbedder", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return &EmbedResponse{}, nil
		})

		e := LookupEmbedder(r, "test/lookupEmbedder")
		if e == nil {
			t.Error("expected embedder, got nil")
		}
	})

	t.Run("returns nil when not found", func(t *testing.T) {
		r := newTestRegistry(t)

		e := LookupEmbedder(r, "nonexistent")
		if e != nil {
			t.Error("expected nil for non-existent embedder")
		}
	})
}

func TestEmbedderEmbed(t *testing.T) {
	t.Run("embeds documents successfully", func(t *testing.T) {
		r := newTestRegistry(t)
		var capturedReq *EmbedRequest

		e := DefineEmbedder(r, "test/embedDocuments", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			capturedReq = req
			embeddings := make([]*Embedding, len(req.Input))
			for i := range req.Input {
				embeddings[i] = &Embedding{
					Embedding: []float32{float32(i) * 0.1, float32(i) * 0.2, float32(i) * 0.3},
				}
			}
			return &EmbedResponse{Embeddings: embeddings}, nil
		})

		docs := []*Document{
			DocumentFromText("first document", nil),
			DocumentFromText("second document", nil),
		}

		resp, err := e.Embed(context.Background(), &EmbedRequest{Input: docs})
		assertNoError(t, err)

		if len(capturedReq.Input) != 2 {
			t.Errorf("captured input len = %d, want 2", len(capturedReq.Input))
		}
		if len(resp.Embeddings) != 2 {
			t.Errorf("len(Embeddings) = %d, want 2", len(resp.Embeddings))
		}
	})

	t.Run("returns error on nil embedder", func(t *testing.T) {
		var e *embedder
		_, err := e.Embed(context.Background(), &EmbedRequest{})
		if err == nil {
			t.Error("expected error for nil embedder")
		}
	})

	t.Run("propagates function errors", func(t *testing.T) {
		r := newTestRegistry(t)
		expectedErr := errors.New("embedding failed")

		e := DefineEmbedder(r, "test/embedError", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return nil, expectedErr
		})

		_, err := e.Embed(context.Background(), &EmbedRequest{
			Input: []*Document{DocumentFromText("test", nil)},
		})
		if err == nil {
			t.Error("expected error, got nil")
		}
	})

	t.Run("passes options through request", func(t *testing.T) {
		r := newTestRegistry(t)
		var capturedOpts any

		e := DefineEmbedder(r, "test/embedOpts", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			capturedOpts = req.Options
			return &EmbedResponse{Embeddings: []*Embedding{{Embedding: []float32{0.1}}}}, nil
		})

		opts := map[string]any{"dimension": 768}
		_, err := e.Embed(context.Background(), &EmbedRequest{
			Input:   []*Document{DocumentFromText("test", nil)},
			Options: opts,
		})
		assertNoError(t, err)

		if diff := cmp.Diff(opts, capturedOpts); diff != "" {
			t.Errorf("Options mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestEmbedFunction(t *testing.T) {
	t.Run("embeds with embedder directly", func(t *testing.T) {
		r := newTestRegistry(t)
		e := DefineEmbedder(r, "test/embedFunc", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return &EmbedResponse{
				Embeddings: []*Embedding{{Embedding: []float32{0.1, 0.2, 0.3}}},
			}, nil
		})

		resp, err := Embed(context.Background(), r,
			WithEmbedder(e),
			WithTextDocs("test document"),
		)
		assertNoError(t, err)

		if len(resp.Embeddings) != 1 {
			t.Errorf("len(Embeddings) = %d, want 1", len(resp.Embeddings))
		}
	})

	t.Run("embeds with embedder ref", func(t *testing.T) {
		r := newTestRegistry(t)
		DefineEmbedder(r, "test/embedFuncRef", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return &EmbedResponse{
				Embeddings: []*Embedding{{Embedding: []float32{0.1, 0.2, 0.3}}},
			}, nil
		})

		ref := NewEmbedderRef("test/embedFuncRef", nil)
		resp, err := Embed(context.Background(), r,
			WithEmbedder(ref),
			WithTextDocs("test document"),
		)
		assertNoError(t, err)

		if len(resp.Embeddings) != 1 {
			t.Errorf("len(Embeddings) = %d, want 1", len(resp.Embeddings))
		}
	})

	t.Run("embeds with embedder name", func(t *testing.T) {
		r := newTestRegistry(t)
		DefineEmbedder(r, "test/embedFuncName", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			return &EmbedResponse{
				Embeddings: []*Embedding{{Embedding: []float32{0.1, 0.2, 0.3}}},
			}, nil
		})

		resp, err := Embed(context.Background(), r,
			WithEmbedderName("test/embedFuncName"),
			WithTextDocs("test document"),
		)
		assertNoError(t, err)

		if len(resp.Embeddings) != 1 {
			t.Errorf("len(Embeddings) = %d, want 1", len(resp.Embeddings))
		}
	})

	t.Run("uses config from EmbedderRef", func(t *testing.T) {
		r := newTestRegistry(t)
		var capturedOpts any

		DefineEmbedder(r, "test/embedRefConfig", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			capturedOpts = req.Options
			return &EmbedResponse{Embeddings: []*Embedding{{Embedding: []float32{0.1}}}}, nil
		})

		config := map[string]any{"dimension": 768}
		ref := NewEmbedderRef("test/embedRefConfig", config)

		_, err := Embed(context.Background(), r,
			WithEmbedder(ref),
			WithTextDocs("test"),
		)
		assertNoError(t, err)

		if diff := cmp.Diff(config, capturedOpts); diff != "" {
			t.Errorf("Options mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("explicit config overrides EmbedderRef config", func(t *testing.T) {
		r := newTestRegistry(t)
		var capturedOpts any

		DefineEmbedder(r, "test/embedOverrideConfig", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			capturedOpts = req.Options
			return &EmbedResponse{Embeddings: []*Embedding{{Embedding: []float32{0.1}}}}, nil
		})

		refConfig := map[string]any{"dimension": 768}
		explicitConfig := map[string]any{"dimension": 512}
		ref := NewEmbedderRef("test/embedOverrideConfig", refConfig)

		_, err := Embed(context.Background(), r,
			WithEmbedder(ref),
			WithConfig(explicitConfig),
			WithTextDocs("test"),
		)
		assertNoError(t, err)

		if diff := cmp.Diff(explicitConfig, capturedOpts); diff != "" {
			t.Errorf("Options mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns error when embedder not set", func(t *testing.T) {
		r := newTestRegistry(t)

		_, err := Embed(context.Background(), r,
			WithTextDocs("test"),
		)
		assertError(t, err, "embedder must be set")
	})

	t.Run("returns error when embedder not found", func(t *testing.T) {
		r := newTestRegistry(t)

		_, err := Embed(context.Background(), r,
			WithEmbedderName("nonexistent"),
			WithTextDocs("test"),
		)
		assertError(t, err, "embedder not found")
	})

	t.Run("embeds with document options", func(t *testing.T) {
		r := newTestRegistry(t)
		var capturedDocs []*Document

		DefineEmbedder(r, "test/embedDocs", nil, func(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
			capturedDocs = req.Input
			embeddings := make([]*Embedding, len(req.Input))
			for i := range req.Input {
				embeddings[i] = &Embedding{Embedding: []float32{0.1}}
			}
			return &EmbedResponse{Embeddings: embeddings}, nil
		})

		doc := DocumentFromText("custom document", map[string]any{"custom": "metadata"})
		_, err := Embed(context.Background(), r,
			WithEmbedderName("test/embedDocs"),
			WithDocs(doc),
		)
		assertNoError(t, err)

		if len(capturedDocs) != 1 {
			t.Fatalf("len(docs) = %d, want 1", len(capturedDocs))
		}
		if capturedDocs[0].Metadata["custom"] != "metadata" {
			t.Error("document metadata not passed correctly")
		}
	})
}
