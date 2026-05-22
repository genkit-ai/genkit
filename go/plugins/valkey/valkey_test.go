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

package valkey

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/fakeembedder"
	"github.com/valkey-io/valkey-glide/go/v2/config"
	"github.com/valkey-io/valkey-glide/go/v2/servermodules/glideft"
)

var testValkeyAddr = flag.String("test-valkey-addr", "", "Valkey address (host:port) to use for integration tests")

func TestPluginName(t *testing.T) {
	v := &Valkey{}
	if got := v.Name(); got != "valkey" {
		t.Errorf("Name() = %q, want %q", got, "valkey")
	}
}

func TestDocID(t *testing.T) {
	d1 := ai.DocumentFromText("hello", nil)
	d2 := ai.DocumentFromText("hello", nil)
	d3 := ai.DocumentFromText("world", nil)

	id1, err := docID(d1)
	if err != nil {
		t.Fatal(err)
	}
	id2, err := docID(d2)
	if err != nil {
		t.Fatal(err)
	}
	id3, err := docID(d3)
	if err != nil {
		t.Fatal(err)
	}

	if id1 != id2 {
		t.Errorf("same document produced different IDs: %q vs %q", id1, id2)
	}
	if id1 == id3 {
		t.Errorf("different documents produced same ID: %q", id1)
	}
	// MD5 hex is 32 characters.
	if len(id1) != 32 {
		t.Errorf("docID length = %d, want 32", len(id1))
	}
}

func TestDocIDWithMetadata(t *testing.T) {
	meta := map[string]any{"b": 1, "a": 2, "c": "three"}
	d1 := ai.DocumentFromText("hello", meta)
	d2 := ai.DocumentFromText("hello", meta)

	// Verify determinism across multiple calls (Go map iteration is random).
	for i := 0; i < 10; i++ {
		id1, err := docID(d1)
		if err != nil {
			t.Fatal(err)
		}
		id2, err := docID(d2)
		if err != nil {
			t.Fatal(err)
		}
		if id1 != id2 {
			t.Fatalf("iteration %d: same document with metadata produced different IDs: %q vs %q", i, id1, id2)
		}
	}
}

func TestFloat32SliceToBytes(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	result := float32SliceToBytes(input)

	if len(result) != 12 {
		t.Fatalf("expected 12 bytes, got %d", len(result))
	}

	// Verify round-trip.
	for i, expected := range input {
		bits := binary.LittleEndian.Uint32(result[i*4:])
		got := math.Float32frombits(bits)
		if got != expected {
			t.Errorf("index %d: got %f, want %f", i, got, expected)
		}
	}
}

func TestFloat32SliceToBytesEmpty(t *testing.T) {
	result := float32SliceToBytes(nil)
	if len(result) != 0 {
		t.Errorf("expected empty slice, got %d bytes", len(result))
	}
}

// errorEmbedder is a mock embedder that always returns an error.
type errorEmbedder struct {
	err error
}

func (e *errorEmbedder) Name() string { return "error/embedder" }
func (e *errorEmbedder) Embed(_ context.Context, _ *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	return nil, e.err
}
func (e *errorEmbedder) Register(_ api.Registry) {}

// mismatchEmbedder returns a fixed number of embeddings regardless of input count.
type mismatchEmbedder struct {
	count int
	dim   int
}

func (m *mismatchEmbedder) Name() string { return "mismatch/embedder" }
func (m *mismatchEmbedder) Embed(_ context.Context, _ *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	resp := &ai.EmbedResponse{}
	for i := 0; i < m.count; i++ {
		resp.Embeddings = append(resp.Embeddings, &ai.Embedding{Embedding: make([]float32, m.dim)})
	}
	return resp, nil
}
func (m *mismatchEmbedder) Register(_ api.Registry) {}

func TestIndexEmbedderError(t *testing.T) {
	ds := &Docstore{
		Embedder:  &errorEmbedder{err: fmt.Errorf("connection timeout")},
		Prefix:    "test",
		Dimension: 3,
	}
	docs := []*ai.Document{ai.DocumentFromText("hello", nil)}
	err := Index(context.Background(), docs, ds)
	if err == nil {
		t.Fatal("expected error from embedder, got nil")
	}
	if !strings.Contains(err.Error(), "connection timeout") {
		t.Errorf("expected error to contain 'connection timeout', got: %v", err)
	}
}

func TestIndexEmbeddingCountMismatch(t *testing.T) {
	ds := &Docstore{
		Embedder:  &mismatchEmbedder{count: 1, dim: 3},
		Prefix:    "test",
		Dimension: 3,
	}
	docs := []*ai.Document{
		ai.DocumentFromText("doc1", nil),
		ai.DocumentFromText("doc2", nil),
	}
	err := Index(context.Background(), docs, ds)
	if err == nil {
		t.Fatal("expected embedding count mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "1 embeddings for 2 docs") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestIndexEmptyDocs(t *testing.T) {
	ds := &Docstore{
		Embedder:  &errorEmbedder{err: fmt.Errorf("should not be called")},
		Prefix:    "test",
		Dimension: 3,
	}
	err := Index(context.Background(), nil, ds)
	if err != nil {
		t.Fatalf("expected nil error for empty docs, got: %v", err)
	}
}

func TestRetrieveEmbedderError(t *testing.T) {
	ds := &Docstore{
		Embedder:  &errorEmbedder{err: fmt.Errorf("rate limited")},
		IndexName: "test-index",
		Prefix:    "test",
		Dimension: 3,
	}
	req := &ai.RetrieverRequest{
		Query: ai.DocumentFromText("query", nil),
	}
	_, err := ds.Retrieve(context.Background(), req)
	if err == nil {
		t.Fatal("expected error from embedder, got nil")
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("expected error to contain 'rate limited', got: %v", err)
	}
}

func TestRetrieveInvalidOptions(t *testing.T) {
	ds := &Docstore{
		Embedder:  &errorEmbedder{err: fmt.Errorf("should not be called")},
		IndexName: "test-index",
		Prefix:    "test",
		Dimension: 3,
	}
	req := &ai.RetrieverRequest{
		Query:   ai.DocumentFromText("query", nil),
		Options: "invalid-type", // wrong type
	}
	_, err := ds.Retrieve(context.Background(), req)
	if err == nil {
		t.Fatal("expected type error for invalid options, got nil")
	}
	if !strings.Contains(err.Error(), "RetrieverOptions") {
		t.Errorf("expected error about RetrieverOptions type, got: %v", err)
	}
}

func TestRetrieveEmptyEmbeddings(t *testing.T) {
	ds := &Docstore{
		Embedder:  &mismatchEmbedder{count: 0, dim: 3},
		IndexName: "test-index",
		Prefix:    "test",
		Dimension: 3,
	}
	req := &ai.RetrieverRequest{
		Query: ai.DocumentFromText("query", nil),
	}
	_, err := ds.Retrieve(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for empty embeddings, got nil")
	}
	if !strings.Contains(err.Error(), "no embeddings") {
		t.Errorf("expected 'no embeddings' error, got: %v", err)
	}
}

func TestIntegration(t *testing.T) {
	if *testValkeyAddr == "" {
		t.Skip("skipping integration test: -test-valkey-addr flag not set")
	}

	parts := strings.SplitN(*testValkeyAddr, ":", 2)
	host := parts[0]
	port := 6379
	if len(parts) == 2 {
		fmt.Sscanf(parts[1], "%d", &port)
	}

	ctx := context.Background()

	addresses := []config.NodeAddress{
		{Host: host, Port: port},
	}

	g := genkit.Init(ctx, genkit.WithPlugins(&Valkey{Addresses: addresses}))

	const dim = 3
	indexName := "test-genkit-valkey"

	// Clean up the test index before and after the test.
	plugin := genkit.LookupPlugin(g, provider).(*Valkey)
	t.Cleanup(func() {
		glideft.FtDropIndex(ctx, plugin.Client(), indexName)
	})
	// Drop any leftover index from a previous run.
	glideft.FtDropIndex(ctx, plugin.Client(), indexName)

	// Create fake embedder with known vectors.
	d1 := ai.DocumentFromText("espresso is strong coffee", nil)
	d2 := ai.DocumentFromText("latte has steamed milk", nil)
	d3 := ai.DocumentFromText("croissant is a pastry", nil)

	v1 := []float32{1.0, 0.0, 0.0}
	v2 := []float32{0.9, 0.1, 0.0}
	v3 := []float32{0.0, 0.0, 1.0}

	embedder := fakeembedder.New()
	embedder.Register(d1, v1)
	embedder.Register(d2, v2)
	embedder.Register(d3, v3)

	emdOpts := &ai.EmbedderOptions{
		Dimensions: dim,
		Label:      "fake-embedder",
		Supports: &ai.EmbedderSupports{
			Input: []string{"text"},
		},
	}

	cfg := Config{
		IndexName: indexName,
		Embedder:  genkit.DefineEmbedder(g, "fake/embedder", emdOpts, embedder.Embed),
		Dimension: dim,
	}

	retOpts := &ai.RetrieverOptions{
		ConfigSchema: core.InferSchemaMap(RetrieverOptions{}),
		Label:        "valkey-test",
		Supports: &ai.RetrieverSupports{
			Media: false,
		},
	}

	ds, retriever, err := DefineRetriever(ctx, g, cfg, retOpts)
	if err != nil {
		t.Fatal(err)
	}

	// Index documents.
	if err := Index(ctx, []*ai.Document{d1, d2, d3}, ds); err != nil {
		t.Fatalf("Index failed: %v", err)
	}

	// Retrieve documents similar to d1 (should return d1 and d2).
	retrieverOptions := &RetrieverOptions{K: 2}
	resp, err := genkit.Retrieve(ctx, g,
		ai.WithRetriever(retriever),
		ai.WithDocs(d1),
		ai.WithConfig(retrieverOptions))
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}

	if len(resp.Documents) != 2 {
		t.Errorf("got %d results, want 2", len(resp.Documents))
	}

	// Both results should be coffee-related, not the pastry.
	for _, d := range resp.Documents {
		text := d.Content[0].Text
		if strings.Contains(text, "croissant") {
			t.Errorf("unexpected result: %q (should not contain pastry)", text)
		}
		if !strings.Contains(text, "espresso") && !strings.Contains(text, "latte") {
			t.Errorf("unexpected result: %q (expected espresso or latte)", text)
		}
	}
}
