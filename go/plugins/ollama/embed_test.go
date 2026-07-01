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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

func TestEmbedValidRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		})
	}))
	defer server.Close()

	req := &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("test", nil),
		},
		Options: &EmbedOptions{Model: "all-minilm"},
	}

	resp, err := embed(context.Background(), server.URL, req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if len(resp.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(resp.Embeddings))
	}
}

func TestEmbedInvalidServerAddress(t *testing.T) {
	req := &ai.EmbedRequest{
		Input: []*ai.Document{
			ai.DocumentFromText("test", nil),
		},
		Options: &EmbedOptions{Model: "all-minilm"},
	}

	_, err := embed(context.Background(), "", req)
	if err == nil || !strings.Contains(err.Error(), "invalid server address") {
		t.Fatalf("expected invalid server address error, got %v", err)
	}
}

func TestDefineEmbedderRegistersByModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		})
	}))
	defer server.Close()

	model := "nomic-embed-text"
	g, o := newTestGenkit(t, server.URL)
	o.DefineEmbedder(g, model, 768, nil)

	if !IsDefinedEmbedder(g, model) {
		t.Fatal("expected embedder to be registered by model")
	}
	if got := Embedder(g, model); got == nil {
		t.Fatal("expected embedder lookup by model to succeed")
	}
	if IsDefinedEmbedder(g, server.URL) {
		t.Fatal("expected server-address lookup to fail")
	}
}

func TestDefineEmbedderRegistersDistinctModelsOnSameServer(t *testing.T) {
	g, o := newTestGenkit(t, "http://localhost:11434")

	firstModel := "nomic-embed-text"
	secondModel := "mxbai-embed-large"

	firstDefined := o.DefineEmbedder(g, firstModel, 768, nil)
	secondDefined := o.DefineEmbedder(g, secondModel, 1024, nil)

	if !IsDefinedEmbedder(g, firstModel) {
		t.Fatalf("expected %q embedder to be registered", firstModel)
	}
	if !IsDefinedEmbedder(g, secondModel) {
		t.Fatalf("expected %q embedder to be registered", secondModel)
	}

	firstLookup := Embedder(g, firstModel)
	secondLookup := Embedder(g, secondModel)

	if firstLookup == nil {
		t.Fatalf("expected lookup for %q to succeed", firstModel)
	}
	if secondLookup == nil {
		t.Fatalf("expected lookup for %q to succeed", secondModel)
	}

	firstDefinedAction, ok := firstDefined.(api.Action)
	if !ok {
		t.Fatalf("expected %q embedder to implement api.Action", firstModel)
	}
	secondDefinedAction, ok := secondDefined.(api.Action)
	if !ok {
		t.Fatalf("expected %q embedder to implement api.Action", secondModel)
	}

	firstLookupAction, ok := firstLookup.(api.Action)
	if !ok {
		t.Fatalf("expected lookup for %q to implement api.Action", firstModel)
	}
	secondLookupAction, ok := secondLookup.(api.Action)
	if !ok {
		t.Fatalf("expected lookup for %q to implement api.Action", secondModel)
	}

	if got, want := firstDefinedAction.Desc().Name, api.NewName(provider, firstModel); got != want {
		t.Fatalf("first Desc().Name = %q, want %q", got, want)
	}
	if got, want := secondDefinedAction.Desc().Name, api.NewName(provider, secondModel); got != want {
		t.Fatalf("second Desc().Name = %q, want %q", got, want)
	}
	if got, want := firstLookupAction.Desc().Name, api.NewName(provider, firstModel); got != want {
		t.Fatalf("lookup first Desc().Name = %q, want %q", got, want)
	}
	if got, want := secondLookupAction.Desc().Name, api.NewName(provider, secondModel); got != want {
		t.Fatalf("lookup second Desc().Name = %q, want %q", got, want)
	}
	if firstDefinedAction.Desc().Name == secondDefinedAction.Desc().Name {
		t.Fatal("expected embedders with different models to have distinct registration names")
	}
}

func TestDefineEmbedderSetsDefaultMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		})
	}))
	defer server.Close()

	model := "nomic-embed-text"
	g, o := newTestGenkit(t, server.URL)
	embedder := o.DefineEmbedder(g, model, 768, nil)

	action, ok := embedder.(api.Action)
	if !ok {
		t.Fatal("expected embedder to implement api.Action")
	}

	desc := action.Desc()
	if got, want := desc.Name, api.NewName(provider, model); got != want {
		t.Fatalf("Desc().Name = %q, want %q", got, want)
	}

	info, ok := desc.Metadata["info"].(map[string]any)
	if !ok {
		t.Fatalf("expected info metadata, got %T", desc.Metadata["info"])
	}
	if got, want := info["label"], "Ollama Embedding - "+model; got != want {
		t.Fatalf("label = %v, want %q", got, want)
	}
	if got, want := info["dimensions"], 768; got != want {
		t.Fatalf("dimensions = %v, want %d", got, want)
	}

	supports, ok := info["supports"].(map[string]any)
	if !ok {
		t.Fatalf("expected supports metadata, got %T", info["supports"])
	}
	input, ok := supports["input"].([]string)
	if !ok {
		t.Fatalf("expected supports.input to be []string, got %T", supports["input"])
	}
	if want := []string{"text"}; !reflect.DeepEqual(input, want) {
		t.Fatalf("supports.input = %v, want %v", input, want)
	}
}

func TestDefineEmbedderPreservesProvidedMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		})
	}))
	defer server.Close()

	g, o := newTestGenkit(t, server.URL)
	embedder := o.DefineEmbedder(g, "nomic-embed-text", 768, &ai.EmbedderOptions{
		Label: "Custom Label",
		Supports: &ai.EmbedderSupports{
			Input:        []string{"text", "image"},
			Multilingual: true,
		},
		ConfigSchema: map[string]any{"type": "object"},
	})

	action := embedder.(api.Action)
	desc := action.Desc()

	info := desc.Metadata["info"].(map[string]any)
	if got, want := info["label"], "Custom Label"; got != want {
		t.Fatalf("label = %v, want %q", got, want)
	}
	if got, want := info["dimensions"], 768; got != want {
		t.Fatalf("dimensions = %v, want %d", got, want)
	}

	supports := info["supports"].(map[string]any)
	input := supports["input"].([]string)
	if want := []string{"text", "image"}; !reflect.DeepEqual(input, want) {
		t.Fatalf("supports.input = %v, want %v", input, want)
	}
	if got, want := supports["multilingual"], true; got != want {
		t.Fatalf("supports.multilingual = %v, want %v", got, want)
	}

	embedderMeta, ok := desc.Metadata["embedder"].(map[string]any)
	if !ok {
		t.Fatalf("expected embedder metadata, got %T", desc.Metadata["embedder"])
	}
	if got, ok := embedderMeta["customOptions"].(map[string]any); !ok || got["type"] != "object" {
		t.Fatalf("customOptions = %v, want schema map", embedderMeta["customOptions"])
	}
}

func TestDefineEmbedderRequestOptionsHandling(t *testing.T) {
	var seenModels []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/api/embed"; got != want {
			t.Fatalf("path = %q, want %q", got, want)
		}

		var body ollamaEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		seenModels = append(seenModels, body.Model)

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		})
	}))
	defer server.Close()

	model := "nomic-embed-text"
	g, o := newTestGenkit(t, server.URL)
	embedder := o.DefineEmbedder(g, model, 768, nil)

	tests := []struct {
		name          string
		options       any
		wantModel     string
		wantErr       string
		wantHTTPCalls int
		assertOpts    func(*testing.T, any)
	}{
		{
			name:          "nil options use bound model",
			options:       nil,
			wantModel:     model,
			wantHTTPCalls: 1,
		},
		{
			name:          "typed nil options use bound model",
			options:       (*EmbedOptions)(nil),
			wantModel:     model,
			wantHTTPCalls: 1,
		},
		{
			name:          "empty model uses bound model without mutating caller options",
			options:       &EmbedOptions{},
			wantModel:     model,
			wantHTTPCalls: 1,
			assertOpts: func(t *testing.T, options any) {
				t.Helper()
				opts := options.(*EmbedOptions)
				if opts.Model != "" {
					t.Fatalf("caller options mutated to %q, want empty string", opts.Model)
				}
			},
		},
		{
			name:          "same model allowed",
			options:       &EmbedOptions{Model: model},
			wantModel:     model,
			wantHTTPCalls: 1,
		},
		{
			name:          "different model rejected",
			options:       &EmbedOptions{Model: "other-model"},
			wantErr:       `invalid embedding model: embedder bound to model "nomic-embed-text", got "other-model"`,
			wantHTTPCalls: 0,
		},
		{
			name:          "wrong options type preserves existing error",
			options:       map[string]any{"model": model},
			wantErr:       "invalid options type: expected *EmbedOptions",
			wantHTTPCalls: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			beforeCalls := len(seenModels)
			resp, err := embedder.Embed(context.Background(), &ai.EmbedRequest{
				Input: []*ai.Document{
					ai.DocumentFromText("test", nil),
				},
				Options: tt.options,
			})

			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("err = %v, want %q", err, tt.wantErr)
				}
				if got := len(seenModels) - beforeCalls; got != tt.wantHTTPCalls {
					t.Fatalf("http calls = %d, want %d", got, tt.wantHTTPCalls)
				}
				return
			}

			if err != nil {
				t.Fatalf("Embed() error = %v", err)
			}
			if resp == nil || len(resp.Embeddings) != 1 {
				t.Fatalf("expected 1 embedding, got %#v", resp)
			}
			if got := len(seenModels) - beforeCalls; got != tt.wantHTTPCalls {
				t.Fatalf("http calls = %d, want %d", got, tt.wantHTTPCalls)
			}
			if got := seenModels[len(seenModels)-1]; got != tt.wantModel {
				t.Fatalf("request model = %q, want %q", got, tt.wantModel)
			}
			if tt.assertOpts != nil {
				tt.assertOpts(t, tt.options)
			}
		})
	}
}

func TestDefineEmbedderRejectsNonPositiveDimensions(t *testing.T) {
	g, o := newTestGenkit(t, "http://localhost:11434")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		if got, want := r, "ollama.DefineEmbedder: dimensions must be greater than 0"; got != want {
			t.Fatalf("panic = %v, want %q", got, want)
		}
	}()

	o.DefineEmbedder(g, "nomic-embed-text", 0, nil)
}

func TestDefineEmbedderDuplicateModelPanics(t *testing.T) {
	g, o := newTestGenkit(t, "http://localhost:11434")
	o.DefineEmbedder(g, "nomic-embed-text", 768, nil)

	defer func() {
		if recover() == nil {
			t.Fatal("expected duplicate registration panic")
		}
	}()

	o.DefineEmbedder(g, "nomic-embed-text", 768, nil)
}

func newTestGenkit(t *testing.T, serverAddress string) (*genkit.Genkit, *Ollama) {
	t.Helper()

	o := &Ollama{ServerAddress: serverAddress}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))
	return g, o
}
