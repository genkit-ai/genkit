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

func TestRetrieverRef(t *testing.T) {
	t.Run("NewRetrieverRef creates ref with name and config", func(t *testing.T) {
		config := map[string]any{"topK": 10}
		ref := NewRetrieverRef("test/retriever", config)

		if ref.Name() != "test/retriever" {
			t.Errorf("Name() = %q, want %q", ref.Name(), "test/retriever")
		}
		if diff := cmp.Diff(config, ref.Config()); diff != "" {
			t.Errorf("Config() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("NewRetrieverRef with nil config", func(t *testing.T) {
		ref := NewRetrieverRef("test/retriever", nil)

		if ref.Name() != "test/retriever" {
			t.Errorf("Name() = %q, want %q", ref.Name(), "test/retriever")
		}
		if ref.Config() != nil {
			t.Errorf("Config() = %v, want nil", ref.Config())
		}
	})
}

func TestNewRetriever(t *testing.T) {
	t.Run("creates retriever with valid name", func(t *testing.T) {
		r := NewRetriever("test/retriever", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{}, nil
		})

		if r == nil {
			t.Fatal("expected retriever, got nil")
		}
		if r.Name() != "test/retriever" {
			t.Errorf("Name() = %q, want %q", r.Name(), "test/retriever")
		}
	})

	t.Run("panics with empty name", func(t *testing.T) {
		assertPanic(t, func() {
			NewRetriever("", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
				return &RetrieverResponse{}, nil
			})
		}, "name is required")
	})

	t.Run("applies options correctly", func(t *testing.T) {
		opts := &RetrieverOptions{
			Label: "Test Retriever",
			Supports: &RetrieverSupports{
				Media: true,
			},
			ConfigSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"topK": map[string]any{"type": "integer"},
				},
			},
		}

		r := NewRetriever("test/retriever", opts, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{}, nil
		})

		if r == nil {
			t.Fatal("expected retriever, got nil")
		}
	})

	t.Run("uses defaults when options nil", func(t *testing.T) {
		r := NewRetriever("test/retriever", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{}, nil
		})

		if r == nil {
			t.Fatal("expected retriever, got nil")
		}
	})
}

func TestDefineRetriever(t *testing.T) {
	t.Run("registers and returns retriever", func(t *testing.T) {
		reg := newTestRegistry(t)
		called := false
		expectedDocs := []*Document{
			DocumentFromText("result 1", nil),
			DocumentFromText("result 2", nil),
		}

		r := DefineRetriever(reg, "test/defineRetriever", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			called = true
			return &RetrieverResponse{Documents: expectedDocs}, nil
		})

		if r == nil {
			t.Fatal("expected retriever, got nil")
		}

		// Verify it's registered by looking it up
		found := LookupRetriever(reg, "test/defineRetriever")
		if found == nil {
			t.Fatal("LookupRetriever returned nil for registered retriever")
		}

		// Verify the function works
		resp, err := r.Retrieve(context.Background(), &RetrieverRequest{
			Query: DocumentFromText("query", nil),
		})
		assertNoError(t, err)
		if !called {
			t.Error("retriever function was not called")
		}
		if len(resp.Documents) != 2 {
			t.Errorf("len(Documents) = %d, want 2", len(resp.Documents))
		}
	})
}

func TestLookupRetriever(t *testing.T) {
	t.Run("returns retriever when found", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineRetriever(reg, "test/lookupRetriever", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{}, nil
		})

		r := LookupRetriever(reg, "test/lookupRetriever")
		if r == nil {
			t.Error("expected retriever, got nil")
		}
	})

	t.Run("returns nil when not found", func(t *testing.T) {
		reg := newTestRegistry(t)

		r := LookupRetriever(reg, "nonexistent")
		if r != nil {
			t.Error("expected nil for non-existent retriever")
		}
	})
}

func TestRetrieverRetrieve(t *testing.T) {
	t.Run("retrieves documents successfully", func(t *testing.T) {
		reg := newTestRegistry(t)
		var capturedReq *RetrieverRequest

		expectedDocs := []*Document{
			DocumentFromText("relevant result 1", map[string]any{"score": 0.9}),
			DocumentFromText("relevant result 2", map[string]any{"score": 0.8}),
		}

		r := DefineRetriever(reg, "test/retrieveDocs", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			capturedReq = req
			return &RetrieverResponse{Documents: expectedDocs}, nil
		})

		query := DocumentFromText("search query", nil)
		resp, err := r.Retrieve(context.Background(), &RetrieverRequest{Query: query})
		assertNoError(t, err)

		if len(capturedReq.Query.Content) == 0 || capturedReq.Query.Content[0].Text != "search query" {
			t.Errorf("captured query content mismatch")
		}
		if len(resp.Documents) != 2 {
			t.Errorf("len(Documents) = %d, want 2", len(resp.Documents))
		}
	})

	t.Run("returns error on nil retriever", func(t *testing.T) {
		var r *retriever
		_, err := r.Retrieve(context.Background(), &RetrieverRequest{})
		if err == nil {
			t.Error("expected error for nil retriever")
		}
	})

	t.Run("propagates function errors", func(t *testing.T) {
		reg := newTestRegistry(t)
		expectedErr := errors.New("retrieval failed")

		r := DefineRetriever(reg, "test/retrieveError", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return nil, expectedErr
		})

		_, err := r.Retrieve(context.Background(), &RetrieverRequest{
			Query: DocumentFromText("query", nil),
		})
		if err == nil {
			t.Error("expected error, got nil")
		}
	})

	t.Run("passes options through request", func(t *testing.T) {
		reg := newTestRegistry(t)
		var capturedOpts any

		r := DefineRetriever(reg, "test/retrieveOpts", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			capturedOpts = req.Options
			return &RetrieverResponse{Documents: []*Document{}}, nil
		})

		opts := map[string]any{"topK": 5, "threshold": 0.7}
		_, err := r.Retrieve(context.Background(), &RetrieverRequest{
			Query:   DocumentFromText("query", nil),
			Options: opts,
		})
		assertNoError(t, err)

		if diff := cmp.Diff(opts, capturedOpts); diff != "" {
			t.Errorf("Options mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestRetrieveFunction(t *testing.T) {
	t.Run("retrieves with retriever directly", func(t *testing.T) {
		reg := newTestRegistry(t)
		r := DefineRetriever(reg, "test/retrieveFunc", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{
				Documents: []*Document{DocumentFromText("result", nil)},
			}, nil
		})

		resp, err := Retrieve(context.Background(), reg,
			WithRetriever(r),
			WithTextDocs("query"),
		)
		assertNoError(t, err)

		if len(resp.Documents) != 1 {
			t.Errorf("len(Documents) = %d, want 1", len(resp.Documents))
		}
	})

	t.Run("retrieves with retriever ref", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineRetriever(reg, "test/retrieveFuncRef", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{
				Documents: []*Document{DocumentFromText("result", nil)},
			}, nil
		})

		ref := NewRetrieverRef("test/retrieveFuncRef", nil)
		resp, err := Retrieve(context.Background(), reg,
			WithRetriever(ref),
			WithTextDocs("query"),
		)
		assertNoError(t, err)

		if len(resp.Documents) != 1 {
			t.Errorf("len(Documents) = %d, want 1", len(resp.Documents))
		}
	})

	t.Run("retrieves with retriever name", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineRetriever(reg, "test/retrieveFuncName", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{
				Documents: []*Document{DocumentFromText("result", nil)},
			}, nil
		})

		resp, err := Retrieve(context.Background(), reg,
			WithRetrieverName("test/retrieveFuncName"),
			WithTextDocs("query"),
		)
		assertNoError(t, err)

		if len(resp.Documents) != 1 {
			t.Errorf("len(Documents) = %d, want 1", len(resp.Documents))
		}
	})

	t.Run("uses config from RetrieverRef", func(t *testing.T) {
		reg := newTestRegistry(t)
		var capturedOpts any

		DefineRetriever(reg, "test/retrieveRefConfig", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			capturedOpts = req.Options
			return &RetrieverResponse{Documents: []*Document{}}, nil
		})

		config := map[string]any{"topK": 10}
		ref := NewRetrieverRef("test/retrieveRefConfig", config)

		_, err := Retrieve(context.Background(), reg,
			WithRetriever(ref),
			WithTextDocs("query"),
		)
		assertNoError(t, err)

		if diff := cmp.Diff(config, capturedOpts); diff != "" {
			t.Errorf("Options mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("explicit config overrides RetrieverRef config", func(t *testing.T) {
		reg := newTestRegistry(t)
		var capturedOpts any

		DefineRetriever(reg, "test/retrieveOverrideConfig", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			capturedOpts = req.Options
			return &RetrieverResponse{Documents: []*Document{}}, nil
		})

		refConfig := map[string]any{"topK": 10}
		explicitConfig := map[string]any{"topK": 5}
		ref := NewRetrieverRef("test/retrieveOverrideConfig", refConfig)

		_, err := Retrieve(context.Background(), reg,
			WithRetriever(ref),
			WithConfig(explicitConfig),
			WithTextDocs("query"),
		)
		assertNoError(t, err)

		if diff := cmp.Diff(explicitConfig, capturedOpts); diff != "" {
			t.Errorf("Options mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("returns error when retriever not set", func(t *testing.T) {
		reg := newTestRegistry(t)

		_, err := Retrieve(context.Background(), reg,
			WithTextDocs("query"),
		)
		assertError(t, err, "retriever must be set")
	})

	t.Run("returns error when retriever not found", func(t *testing.T) {
		reg := newTestRegistry(t)

		_, err := Retrieve(context.Background(), reg,
			WithRetrieverName("nonexistent"),
			WithTextDocs("query"),
		)
		assertError(t, err, "retriever not found")
	})

	t.Run("returns error with multiple documents", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineRetriever(reg, "test/retrieveMultiDoc", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			return &RetrieverResponse{Documents: []*Document{}}, nil
		})

		_, err := Retrieve(context.Background(), reg,
			WithRetrieverName("test/retrieveMultiDoc"),
			WithDocs(
				DocumentFromText("doc1", nil),
				DocumentFromText("doc2", nil),
			),
		)
		assertError(t, err, "only supports a single document")
	})

	t.Run("retrieves with document options", func(t *testing.T) {
		reg := newTestRegistry(t)
		var capturedQuery *Document

		DefineRetriever(reg, "test/retrieveDocOpts", nil, func(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
			capturedQuery = req.Query
			return &RetrieverResponse{Documents: []*Document{}}, nil
		})

		query := DocumentFromText("custom query", map[string]any{"custom": "metadata"})
		_, err := Retrieve(context.Background(), reg,
			WithRetrieverName("test/retrieveDocOpts"),
			WithDocs(query),
		)
		assertNoError(t, err)

		if len(capturedQuery.Content) == 0 || capturedQuery.Content[0].Text != "custom query" {
			t.Errorf("query content mismatch")
		}
		if capturedQuery.Metadata["custom"] != "metadata" {
			t.Error("query metadata not passed correctly")
		}
	})
}
