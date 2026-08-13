// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"google.golang.org/genai"
)

func TestEmbedBatchSize(t *testing.T) {
	cases := []struct {
		backend genai.Backend
		model   string
		want    int
	}{
		{genai.BackendGeminiAPI, "gemini-embedding-001", googleAIEmbedBatchSize},
		{genai.BackendGeminiAPI, "text-embedding-004", googleAIEmbedBatchSize},
		{genai.BackendVertexAI, "text-embedding-005", vertexAIEmbedBatchSize},
		{genai.BackendVertexAI, "multimodalembedding", vertexAIEmbedBatchSize},
		// Vertex serves most Gemini and all MaaS embedding models through the
		// one-content embedContent API, but gemini-embedding-001 through the
		// batching prediction service, mirroring the SDK's routing predicate.
		{genai.BackendVertexAI, "gemini-embedding-001", vertexAIEmbedBatchSize},
		{genai.BackendVertexAI, "gemini-embedding-2", 1},
		{genai.BackendVertexAI, "some-embedding-maas", 1},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%v/%s", tc.backend, tc.model), func(t *testing.T) {
			if got := embedBatchSize(tc.backend, tc.model); got != tc.want {
				t.Errorf("embedBatchSize(%v, %q) = %d, want %d", tc.backend, tc.model, got, tc.want)
			}
		})
	}
}

// TestEmbedderBatchesLargeInputs verifies that an embed request with more
// inputs than the service accepts per call is split into multiple calls and
// that the embeddings come back in input order.
func TestEmbedderBatchesLargeInputs(t *testing.T) {
	const docCount = googleAIEmbedBatchSize + 50

	var batchSizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Requests []json.RawMessage `json:"requests"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		batchSizes = append(batchSizes, len(body.Requests))

		embeddings := make([]map[string]any, len(body.Requests))
		for i := range body.Requests {
			embeddings[i] = map[string]any{"values": []float64{float64(len(batchSizes)), float64(i)}}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"embeddings": embeddings})
	}))
	defer srv.Close()

	embedder := newEmbedder(newTestClient(t, srv.URL), "text-embedding-004", &ai.EmbedderOptions{})

	req := &ai.EmbedRequest{}
	for i := 0; i < docCount; i++ {
		req.Input = append(req.Input, ai.DocumentFromText(fmt.Sprintf("doc %d", i), nil))
	}

	resp, err := embedder.Embed(context.Background(), req)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	if len(batchSizes) != 2 || batchSizes[0] != googleAIEmbedBatchSize || batchSizes[1] != 50 {
		t.Errorf("batch sizes = %v, want [%d 50]", batchSizes, googleAIEmbedBatchSize)
	}
	if len(resp.Embeddings) != docCount {
		t.Fatalf("got %d embeddings, want %d", len(resp.Embeddings), docCount)
	}
	// First embedding of the second batch carries batch number 2, index 0,
	// proving order is preserved across batches.
	second := resp.Embeddings[googleAIEmbedBatchSize].Embedding
	if len(second) != 2 || second[0] != 2 || second[1] != 0 {
		t.Errorf("first embedding of second batch = %v, want [2 0]", second)
	}
}

// TestEmbedderWrapsAPIErrors verifies that service errors surface with their
// status classification intact.
func TestEmbedderWrapsAPIErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"}}`)
	}))
	defer srv.Close()

	embedder := newEmbedder(newTestClient(t, srv.URL), "text-embedding-004", &ai.EmbedderOptions{})
	_, err := embedder.Embed(context.Background(), &ai.EmbedRequest{
		Input: []*ai.Document{ai.DocumentFromText("doc", nil)},
	})
	if err == nil {
		t.Fatal("Embed = nil error, want quota error")
	}
	var ge *core.GenkitError
	if !errors.As(err, &ge) {
		t.Fatalf("error %v (%T) is not status-classified", err, err)
	}
	if ge.Status != core.RESOURCE_EXHAUSTED {
		t.Errorf("status = %q, want RESOURCE_EXHAUSTED", ge.Status)
	}
}
