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

package cohere

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/cohere-ai/cohere-go/v2/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func TestEmbedRequestAndResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/embed" {
			t.Errorf("request path = %q, want /v2/embed", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["model"] != "embed-v4.0" || body["input_type"] != "search_query" {
			t.Errorf("model/input_type = %q/%q", body["model"], body["input_type"])
		}
		if body["output_dimension"] != float64(256) || body["truncate"] != "END" {
			t.Errorf("dimension/truncate = %v/%v", body["output_dimension"], body["truncate"])
		}
		texts, ok := body["texts"].([]any)
		if !ok || len(texts) != 2 || texts[0] != "hello world" || texts[1] != "second" {
			t.Errorf("texts = %#v", body["texts"])
		}
		types, ok := body["embedding_types"].([]any)
		if !ok || len(types) != 1 || types[0] != "float" {
			t.Errorf("embedding_types = %#v", body["embedding_types"])
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"response_type":"embeddings_by_type","id":"embed_1","embeddings":{"float":[[1.25,2.5],[-3,4]]},"texts":["hello world","second"]}`)
	}))
	defer server.Close()

	client := cohereclient.NewClient(option.WithToken("test-key"), option.WithBaseURL(server.URL))
	response, err := embed(context.Background(), client, "embed-v4.0", &ai.EmbedRequest{
		Input: []*ai.Document{
			{Content: []*ai.Part{ai.NewTextPart("hello"), ai.NewTextPart(" world")}},
			{Content: []*ai.Part{ai.NewTextPart("second")}},
		},
	}, EmbedOptions{InputType: "search_query", OutputDimension: 256, Truncate: "END"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(response.Embeddings) != 2 {
		t.Fatalf("embedding count = %d, want 2", len(response.Embeddings))
	}
	if got := response.Embeddings[0].Embedding; len(got) != 2 || got[0] != 1.25 || got[1] != 2.5 {
		t.Errorf("first embedding = %v", got)
	}
	if got := response.Embeddings[1].Embedding; len(got) != 2 || got[0] != -3 || got[1] != 4 {
		t.Errorf("second embedding = %v", got)
	}
}

func TestEmbedTypes(t *testing.T) {
	tests := []struct {
		name          string
		embeddingType cohere.EmbeddingType
		response      string
		want          []float32
	}{
		{name: "float", embeddingType: cohere.EmbeddingTypeFloat, response: `{"float":[[1.25,-2.5]]}`, want: []float32{1.25, -2.5}},
		{name: "int8", embeddingType: cohere.EmbeddingTypeInt8, response: `{"int8":[[-128,127]]}`, want: []float32{-128, 127}},
		{name: "uint8", embeddingType: cohere.EmbeddingTypeUint8, response: `{"uint8":[[0,255]]}`, want: []float32{0, 255}},
		{name: "binary", embeddingType: cohere.EmbeddingTypeBinary, response: `{"binary":[[-128,127]]}`, want: []float32{-128, 127}},
		{name: "ubinary", embeddingType: cohere.EmbeddingTypeUbinary, response: `{"ubinary":[[0,255]]}`, want: []float32{0, 255}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatalf("decode request: %v", err)
				}
				types, ok := body["embedding_types"].([]any)
				if !ok || len(types) != 1 || types[0] != string(tc.embeddingType) {
					t.Errorf("embedding_types = %#v, want [%q]", body["embedding_types"], tc.embeddingType)
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintf(w, `{"response_type":"embeddings_by_type","id":"embed_1","embeddings":%s,"texts":["hello"]}`, tc.response)
			}))
			defer server.Close()

			client := cohereclient.NewClient(option.WithToken("test-key"), option.WithBaseURL(server.URL))
			response, err := embed(context.Background(), client, "embed-v4.0", &ai.EmbedRequest{
				Input: []*ai.Document{{Content: []*ai.Part{ai.NewTextPart("hello")}}},
			}, EmbedOptions{EmbeddingType: tc.embeddingType})
			if err != nil {
				t.Fatalf("embed: %v", err)
			}
			if len(response.Embeddings) != 1 {
				t.Fatalf("embedding count = %d, want 1", len(response.Embeddings))
			}
			got := response.Embeddings[0].Embedding
			if len(got) != len(tc.want) {
				t.Fatalf("embedding = %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("embedding[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestEmbedRejectsUnsupportedType(t *testing.T) {
	_, err := embed(context.Background(), nil, "embed-v4.0", &ai.EmbedRequest{}, EmbedOptions{EmbeddingType: "base64"})
	if err == nil || !strings.Contains(err.Error(), `unsupported embedding type "base64"`) {
		t.Fatalf("embed error = %v", err)
	}
}

func TestNewEmbedderRef(t *testing.T) {
	config := &EmbedOptions{InputType: "classification", OutputDimension: 512, Truncate: "START", EmbeddingType: cohere.EmbeddingTypeInt8}
	ref := NewEmbedderRef("embed-v4.0", config)
	if ref.Name() != "cohere/embed-v4.0" {
		t.Fatalf("ref name = %q", ref.Name())
	}
	if ref.Config() != config {
		t.Fatalf("ref config = %#v, want original pointer", ref.Config())
	}
	prefixed := NewEmbedderRef("cohere/embed-v4.0", nil)
	if prefixed.Name() != "cohere/embed-v4.0" {
		t.Fatalf("prefixed ref name = %q", prefixed.Name())
	}
}

func TestEmbedOptionsPassActionSchema(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["input_type"] != "classification" || body["output_dimension"] != float64(512) || body["truncate"] != "START" {
			t.Errorf("typed embed config was not forwarded: %#v", body)
		}
		types, ok := body["embedding_types"].([]any)
		if !ok || len(types) != 1 || types[0] != "int8" {
			t.Errorf("typed embedding type was not forwarded: %#v", body["embedding_types"])
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"response_type":"embeddings_by_type","id":"embed_1","embeddings":{"int8":[[1,2]]},"texts":["hello"]}`)
	}))
	defer server.Close()

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&Cohere{APIKey: "test-key", BaseURL: server.URL}))
	response, err := genkit.Embed(ctx, g,
		ai.WithEmbedder(NewEmbedderRef("embed-v4.0", &EmbedOptions{
			InputType:       "classification",
			OutputDimension: 512,
			Truncate:        "START",
			EmbeddingType:   cohere.EmbeddingTypeInt8,
		})),
		ai.WithTextDocs("hello"),
	)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(response.Embeddings) != 1 || len(response.Embeddings[0].Embedding) != 2 {
		t.Fatalf("embeddings = %#v", response.Embeddings)
	}
}

func TestEmbedWrapsAPIErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"message":"bad embed input"}`)
	}))
	defer server.Close()

	client := cohereclient.NewClient(option.WithToken("test-key"), option.WithBaseURL(server.URL))
	_, err := embed(context.Background(), client, "embed-v4.0", &ai.EmbedRequest{
		Input: []*ai.Document{{Content: []*ai.Part{ai.NewTextPart("hello")}}},
	}, EmbedOptions{})
	if err == nil || !strings.Contains(err.Error(), "cohere:") || !strings.Contains(err.Error(), "bad embed input") {
		t.Fatalf("embed error = %v", err)
	}
}
