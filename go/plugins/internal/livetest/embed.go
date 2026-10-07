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

package livetest

import (
	"math"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// EmbedderSuite describes how to drive one embedder through the shared
// checklist.
type EmbedderSuite struct {
	// Embedder is the embedder under test. Required.
	Embedder ai.EmbedderArg
	// Dimensions is the length every vector must have. Required.
	Dimensions int
	// Normalized reports whether the provider returns unit-length vectors.
	Normalized bool
}

// RunEmbedder walks the embedder through the shared checklist, as subtests of
// an "embedder" subtest.
func RunEmbedder(t *testing.T, g *genkit.Genkit, s EmbedderSuite) {
	t.Helper()
	if s.Embedder == nil || s.Dimensions == 0 {
		t.Fatal("livetest: EmbedderSuite.Embedder and Dimensions are required")
	}
	ctx := t.Context()
	t.Run("embedder", func(t *testing.T) {
		// Two sentences about the same scene in different words, and one
		// about something else: the pair must land closer together than
		// either does to the third, in the order the documents were sent.
		docs := []string{
			"The cat sat on the mat.",
			"Stock markets fell sharply today.",
			"A kitten rested on the rug.",
		}
		res, err := genkit.Embed(ctx, g, ai.WithEmbedder(s.Embedder), ai.WithTextDocs(docs...))
		if err != nil {
			t.Fatalf("Embed() error = %v", err)
		}
		if len(res.Embeddings) != len(docs) {
			t.Fatalf("Embeddings = %d, want one per document (%d)", len(res.Embeddings), len(docs))
		}
		vecs := make([][]float32, len(docs))
		for i, e := range res.Embeddings {
			vecs[i] = e.Embedding
			if len(e.Embedding) != s.Dimensions {
				t.Errorf("Embeddings[%d] has %d dimensions, want %d", i, len(e.Embedding), s.Dimensions)
			}
			// A zero vector would make the similarity checks below NaN,
			// and every comparison with NaN is false.
			switch n := norm(e.Embedding); {
			case n == 0:
				t.Errorf("Embeddings[%d] is all zeros", i)
			case s.Normalized && math.Abs(n-1) > 0.01:
				t.Errorf("Embeddings[%d] has norm %v, want unit length", i, n)
			}
		}
		if t.Failed() {
			return
		}
		if cosine(vecs[0], vecs[2]) <= cosine(vecs[0], vecs[1]) || cosine(vecs[2], vecs[0]) <= cosine(vecs[2], vecs[1]) {
			t.Error("the two cat sentences are not each other's nearest neighbor, want vectors in document order")
		}
	})
}

func norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot / (norm(a) * norm(b))
}
