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

	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/cohere-ai/cohere-go/v2/option"
	"github.com/firebase/genkit/go/ai"
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
		Options: &EmbedOptions{InputType: "search_query", OutputDimension: 256, Truncate: "END"},
	})
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

func TestEmbedDefaultsAndMapOptions(t *testing.T) {
	value := EmbedOptions{InputType: "search_document"}
	if got := embedOptionsFromRequest(value); got != value {
		t.Fatalf("value options = %+v", got)
	}
	options := embedOptionsFromRequest(map[string]any{
		"inputType":       "classification",
		"outputDimension": 512,
		"truncate":        "START",
	})
	if options.InputType != "classification" || options.OutputDimension != 512 || options.Truncate != "START" {
		t.Fatalf("map options = %+v", options)
	}
	if got := embedOptionsFromRequest(42); got != (EmbedOptions{}) {
		t.Fatalf("unsupported options = %+v, want defaults", got)
	}
	if got := embedOptionsFromRequest((*EmbedOptions)(nil)); got != (EmbedOptions{}) {
		t.Fatalf("nil pointer options = %+v, want defaults", got)
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
	})
	if err == nil || !strings.Contains(err.Error(), "cohere:") || !strings.Contains(err.Error(), "bad embed input") {
		t.Fatalf("embed error = %v", err)
	}
}
