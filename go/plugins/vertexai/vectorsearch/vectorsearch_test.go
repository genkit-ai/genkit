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

package vectorsearch

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

type fakeEmbedder struct {
	resp *ai.EmbedResponse
	err  error
}

func (f *fakeEmbedder) Name() string { return "fake-embedder" }

func (f *fakeEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.resp, nil
}

func (f *fakeEmbedder) Register(r api.Registry) {
	// This is a no-op for the fake embedder, as it doesn't need to register anything.
}

// ---------- tests: Name ----------

func TestVectorsearch_Name(t *testing.T) {
	v := &VertexAIVectorSearch{}
	if got, want := v.Name(), vectorsearchProvider; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
}

// ---------- tests: Init ----------

func TestInit_AlreadyInitialized(t *testing.T) {
	v := &VertexAIVectorSearch{initted: true}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected Init to panic when the plugin is already initialized")
		}
	}()
	v.Init(context.Background())
}

func TestInit_MissingProjectID_Panics(t *testing.T) {
	unset := func(k string) func() {
		old, had := os.LookupEnv(k)
		_ = os.Unsetenv(k)
		return func() {
			if had {
				_ = os.Setenv(k, old)
			} else {
				_ = os.Unsetenv(k)
			}
		}
	}
	restore := []func(){}
	for _, e := range []string{"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION"} {
		restore = append(restore, unset(e))
	}
	defer func() {
		for _, r := range restore {
			r()
		}
	}()

	v := &VertexAIVectorSearch{}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected Init to panic when GOOGLE_CLOUD_PROJECT is not set")
		}
	}()
	v.Init(context.Background())
}

func TestInit_MissingLocation_Panics(t *testing.T) {
	restoreProj := setEnv(t, "GOOGLE_CLOUD_PROJECT", "proj")
	defer restoreProj()
	restoreLoc := clearEnv(t, "GOOGLE_CLOUD_LOCATION")
	defer restoreLoc()
	restoreReg := clearEnv(t, "GOOGLE_CLOUD_REGION")
	defer restoreReg()

	v := &VertexAIVectorSearch{}

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected Init to panic when GOOGLE_CLOUD_LOCATION is not set")
		}
	}()
	v.Init(context.Background())
}

func setEnv(t *testing.T, k, v string) func() {
	t.Helper()
	old, had := os.LookupEnv(k)
	if err := os.Setenv(k, v); err != nil {
		t.Fatal(err)
	}
	return func() {
		if had {
			_ = os.Setenv(k, old)
		} else {
			_ = os.Unsetenv(k)
		}
	}
}

func clearEnv(t *testing.T, k string) func() {
	t.Helper()
	old, had := os.LookupEnv(k)
	_ = os.Unsetenv(k)
	return func() {
		if had {
			_ = os.Setenv(k, old)
		} else {
			_ = os.Unsetenv(k)
		}
	}
}

// ---------- tests: Retrieve ----------

func TestRetrieve_Success(t *testing.T) {
	embedder := &fakeEmbedder{
		resp: &ai.EmbedResponse{
			Embeddings: []*ai.Embedding{{Embedding: []float32{0.1}}},
		},
	}
	v := newVSWithToken("tok123", nil)
	findNeighborsJSON := `{"nearestNeighbors":[{"neighbors":[{"datapoint":{"datapointId":"doc-1"},"distance":0.4}]}]}`
	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		if got, want := r.Header.Get("Authorization"), "Bearer tok123"; got != want {
			t.Errorf("Authorization header = %q, want %q", got, want)
		}
		return newHTTPResponse(http.StatusOK, findNeighborsJSON), nil
	}), func() {
		docRetriever := func(ctx context.Context, neighbors []Neighbor, options any) ([]*ai.Document, error) {
			if len(neighbors) != 1 {
				t.Errorf("len(neighbors) = %d, want 1", len(neighbors))
			}
			if got, want := neighbors[0].Datapoint.DatapointId, "doc-1"; got != want {
				t.Errorf("datapointId = %q, want %q", got, want)
			}
			return []*ai.Document{{Content: []*ai.Part{{Text: "Hello"}}}}, nil
		}

		req := &ai.RetrieverRequest{
			Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
			Options: &RetrieveParams{
				Embedder:          embedder,
				NeighborCount:     1,
				ProjectNumber:     "123",
				IndexEndpointID:   "ep",
				PublicDomainName:  "public.example",
				DeployedIndexID:   "dep",
				DocumentRetriever: docRetriever,
			},
		}

		resp, err := v.Retrieve(context.Background(), req)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if resp == nil {
			t.Fatal("expected response, got nil")
		}
		if got, want := resp.Documents[0].Content[0].Text, "Hello"; got != want {
			t.Errorf("document content = %q, want %q", got, want)
		}
	})
}

func TestRetrieve_EmbedderError(t *testing.T) {
	v := newVSWithToken("tok", nil)
	req := &ai.RetrieverRequest{
		Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
		Options: &RetrieveParams{
			Embedder: &fakeEmbedder{err: errors.New("fail")},
			DocumentRetriever: func(context.Context, []Neighbor, any) ([]*ai.Document, error) {
				return nil, nil
			},
		},
	}
	_, err := v.Retrieve(context.Background(), req)
	if err == nil {
		t.Error("expected error, got nil")
	}
	if want := "error generating embedding for query"; !strings.Contains(err.Error(), want) {
		t.Errorf("error = %q, want it to contain %q", err.Error(), want)
	}
}

func TestRetrieve_NoEmbeddings(t *testing.T) {
	v := newVSWithToken("tok", nil)
	req := &ai.RetrieverRequest{
		Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
		Options: &RetrieveParams{
			Embedder: &fakeEmbedder{resp: &ai.EmbedResponse{Embeddings: []*ai.Embedding{}}},
			DocumentRetriever: func(context.Context, []Neighbor, any) ([]*ai.Document, error) {
				return nil, nil
			},
		},
	}
	_, err := v.Retrieve(context.Background(), req)
	if err == nil {
		t.Error("expected error, got nil")
	}
	if want := "no embeddings generated for query"; !strings.Contains(err.Error(), want) {
		t.Errorf("error = %q, want it to contain %q", err.Error(), want)
	}
}

func TestRetrieve_NoNeighborsReturnsNil(t *testing.T) {
	embedder := &fakeEmbedder{
		resp: &ai.EmbedResponse{Embeddings: []*ai.Embedding{{Embedding: []float32{1}}}},
	}
	v := newVSWithToken("tok", nil)
	findNeighborsJSON := `{"nearestNeighbors":[{"neighbors":[]}]}`
	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusOK, findNeighborsJSON), nil
	}), func() {
		req := &ai.RetrieverRequest{
			Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
			Options: &RetrieveParams{
				Embedder:          embedder,
				DocumentRetriever: func(context.Context, []Neighbor, any) ([]*ai.Document, error) { return nil, nil },
			},
		}
		resp, err := v.Retrieve(context.Background(), req)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if resp != nil {
			t.Errorf("Retrieve() resp = %v, want nil", resp)
		}
	})
}
