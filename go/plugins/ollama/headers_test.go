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

package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestStaticRequestHeadersOnTags(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{}})
	}))
	defer server.Close()

	o := newTestOllama(server.URL)
	o.RequestHeaders = map[string]string{"Authorization": "Bearer static-token"}

	if _, err := o.listLocalModels(context.Background()); err != nil {
		t.Fatalf("listLocalModels() error = %v", err)
	}
	if gotAuth != "Bearer static-token" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer static-token")
	}
}

func TestDynamicRequestHeadersOnGenerate(t *testing.T) {
	var gotAuth, gotModel string
	var sawContentType bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		gotModel = r.Header.Get("X-Model")
		sawContentType = r.Header.Get("Content-Type") == "application/json"
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"model":"llama3","message":{"role":"assistant","content":"hi"},"done":true}`))
	}))
	defer server.Close()

	g := &generator{
		model:         ModelDefinition{Name: "llama3", Type: "chat"},
		serverAddress: server.URL,
		timeout:       30,
		requestHeaderFunc: func(ctx context.Context, params HeaderParams) (map[string]string, error) {
			if params.ServerAddress != server.URL {
				t.Errorf("ServerAddress = %q, want %q", params.ServerAddress, server.URL)
			}
			if params.Model == nil || params.Model.Name != "llama3" {
				t.Errorf("Model = %+v, want llama3", params.Model)
			}
			if params.ModelRequest == nil {
				t.Error("ModelRequest is nil")
			}
			return map[string]string{
				"Authorization": "Bearer dynamic-token",
				"X-Model":       params.Model.Name,
			}, nil
		},
	}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hello")}},
		},
	}
	if _, err := g.generate(context.Background(), req, nil); err != nil {
		t.Fatalf("generate() error = %v", err)
	}
	if !sawContentType {
		t.Fatal("expected Content-Type application/json")
	}
	if gotAuth != "Bearer dynamic-token" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer dynamic-token")
	}
	if gotModel != "llama3" {
		t.Fatalf("X-Model = %q, want llama3", gotModel)
	}
}

func TestRequestHeaderFuncTakesPrecedenceOverStatic(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(ollamaTagsResponse{Models: []ollamaLocalModel{}})
	}))
	defer server.Close()

	o := newTestOllama(server.URL)
	o.RequestHeaders = map[string]string{"Authorization": "Bearer static-token"}
	o.RequestHeaderFunc = func(ctx context.Context, params HeaderParams) (map[string]string, error) {
		return map[string]string{"Authorization": "Bearer from-func"}, nil
	}

	if _, err := o.listLocalModels(context.Background()); err != nil {
		t.Fatalf("listLocalModels() error = %v", err)
	}
	if gotAuth != "Bearer from-func" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer from-func")
	}
}

func TestStaticRequestHeadersOnEmbed(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1, 0.2}},
		})
	}))
	defer server.Close()

	req := &ai.EmbedRequest{
		Input:   []*ai.Document{ai.DocumentFromText("test", nil)},
		Options: &EmbedOptions{Model: "all-minilm"},
	}
	headers := map[string]string{"Authorization": "Bearer embed-token"}
	if _, err := embed(context.Background(), server.URL, req, headers, nil); err != nil {
		t.Fatalf("embed() error = %v", err)
	}
	if gotAuth != "Bearer embed-token" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer embed-token")
	}
}

func TestDynamicRequestHeadersOnEmbed(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(ollamaEmbedResponse{
			Embeddings: [][]float32{{0.1}},
		})
	}))
	defer server.Close()

	req := &ai.EmbedRequest{
		Input:   []*ai.Document{ai.DocumentFromText("test", nil)},
		Options: &EmbedOptions{Model: "all-minilm"},
	}
	headerFunc := func(ctx context.Context, params HeaderParams) (map[string]string, error) {
		if params.EmbedRequest == nil {
			t.Error("EmbedRequest is nil")
		}
		if params.Model == nil || params.Model.Name != "all-minilm" {
			t.Errorf("Model = %+v, want all-minilm", params.Model)
		}
		return map[string]string{"Authorization": "Bearer embed-dyn"}, nil
	}
	if _, err := embed(context.Background(), server.URL, req, map[string]string{"Authorization": "ignored"}, headerFunc); err != nil {
		t.Fatalf("embed() error = %v", err)
	}
	if gotAuth != "Bearer embed-dyn" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer embed-dyn")
	}
}

func TestStaticRequestHeadersOnShow(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/show" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(ollamaShowResponse{Capabilities: []string{"completion"}})
	}))
	defer server.Close()

	o := newTestOllama(server.URL)
	o.RequestHeaders = map[string]string{"Authorization": "Bearer show-token"}

	if _, err := o.getModelCapabilities(context.Background(), "llama3"); err != nil {
		t.Fatalf("getModelCapabilities() error = %v", err)
	}
	if gotAuth != "Bearer show-token" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer show-token")
	}
}
