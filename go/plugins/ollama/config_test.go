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
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

func TestInitDefaultsServerAddress(t *testing.T) {
	o := &Ollama{}
	o.Init(context.Background())
	if o.ServerAddress != defaultOllamaServerAddress {
		t.Fatalf("ServerAddress = %q, want %q", o.ServerAddress, defaultOllamaServerAddress)
	}
}

func TestInitPreservesExplicitServerAddress(t *testing.T) {
	want := "http://example.com:11434"
	o := &Ollama{ServerAddress: want}
	o.Init(context.Background())
	if o.ServerAddress != want {
		t.Fatalf("ServerAddress = %q, want %q", o.ServerAddress, want)
	}
}

func TestInitDefinesModelsAndEmbedders(t *testing.T) {
	o := &Ollama{
		Models: []ModelDefinition{
			{Name: "tinyllama"}, // Type defaults to chat
			{Name: "custom-gen", Type: "generate"},
		},
		Embedders: []EmbedderDefinition{
			{Name: "nomic-embed-text", Dimensions: 768},
		},
	}
	g := genkit.Init(context.Background(), genkit.WithPlugins(o))

	if !IsDefinedModel(g, "tinyllama") {
		t.Fatal("expected tinyllama model to be defined at Init")
	}
	if !IsDefinedModel(g, "custom-gen") {
		t.Fatal("expected custom-gen model to be defined at Init")
	}
	if !IsDefinedEmbedder(g, "nomic-embed-text") {
		t.Fatal("expected nomic-embed-text embedder to be defined at Init")
	}

	if got := Model(g, "tinyllama"); got == nil {
		t.Fatal("Model(tinyllama) returned nil")
	}
	if got := Embedder(g, "nomic-embed-text"); got == nil {
		t.Fatal("Embedder(nomic-embed-text) returned nil")
	}

	// Empty Type should have been treated as chat for the tinyllama generator.
	tiny := Model(g, "tinyllama")
	if action, ok := tiny.(api.Action); ok {
		if action.Desc().Name != api.NewName(provider, "tinyllama") {
			t.Fatalf("tinyllama name = %q", action.Desc().Name)
		}
	}
}
