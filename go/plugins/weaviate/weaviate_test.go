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

package weaviate

import (
	"context"
	"flag"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/fakeembedder"
)

var (
	testAddr   = flag.String("test-weaviate-addr", "", "Weaviate address to use for tests")
	testScheme = flag.String("test-weaviate-scheme", "", "Weaviate scheme to use for tests")
	testAPIKey = flag.String("test-weaviate-api-key", "", "Weaviate API key to use for tests")
	testClass  = flag.String("test-weaviate-class", "", "Weaviate class to use for tests")
)

func TestGenkit(t *testing.T) {
	if *testAddr == "" {
		t.Skip("skipping test because -test-weaviate-addr flag not used")
	}
	if *testClass == "" {
		t.Skip("skipping test because -test-weaviate-class flag not used")
	}

	ctx := context.Background()

	g := genkit.Init(context.Background())

	// Make two very similar vectors and one different vector.
	// Arrange for a fake embedder to return those vectors
	// when provided with documents.

	const dim = 768

	v1 := make([]float32, dim)
	v2 := make([]float32, dim)
	v3 := make([]float32, dim)
	for i := range v1 {
		v1[i] = float32(i)
		v2[i] = float32(i)
		v3[i] = float32(dim - i)
	}
	v2[0] = 1

	d1 := ai.DocumentFromText("hello1", map[string]any{"name": "hello1"})
	d2 := ai.DocumentFromText("hello2", map[string]any{"name": "hello2"})
	d3 := ai.DocumentFromText("goodbye", map[string]any{"name": "goodbye"})

	embedder := fakeembedder.New()
	embedder.Register(d1, v1)
	embedder.Register(d2, v2)
	embedder.Register(d3, v3)

	w := &Weaviate{
		Addr:   *testAddr,
		Scheme: *testScheme,
		APIKey: *testAPIKey,
	}

	actions := w.Init(ctx)
	if len(actions) != 0 {
		t.Errorf("actions = %v, want empty", actions)
	}

	// Delete our test class so that earlier runs don't mess us up.
	if err := w.client.Schema().ClassDeleter().WithClassName(*testClass).Do(ctx); err != nil {
		t.Fatal(err)
	}

	emdOpts := &ai.EmbedderOptions{
		Dimensions: 768,
		Label:      "",
		Supports: &ai.EmbedderSupports{
			Input: []string{"text"},
		},
		ConfigSchema: nil,
	}

	classCfg := ClassConfig{
		Class:    *testClass,
		Embedder: genkit.DefineEmbedder(g, "fake/embedder3", emdOpts, embedder.Embed),
	}
	retOpts := &ai.RetrieverOptions{
		ConfigSchema: core.InferSchemaMap(RetrieverOptions{}),
		Label:        "weaviate",
		Supports: &ai.RetrieverSupports{
			Media: false,
		},
	}
	ds, retriever, err := DefineRetriever(ctx, g, classCfg, retOpts)
	if err != nil {
		t.Fatal(err)
	}

	err = Index(ctx, []*ai.Document{d1, d2, d3}, ds)
	if err != nil {
		t.Fatalf("Index operation failed: %v", err)
	}

	retrieverOptions := &RetrieverOptions{
		Count:        2,
		MetadataKeys: []string{"name"},
	}
	retrieverResp, err := genkit.Retrieve(ctx, g,
		ai.WithRetriever(retriever),
		ai.WithDocs(d1),
		ai.WithConfig(retrieverOptions))
	if err != nil {
		t.Fatalf("Retrieve operation failed: %v", err)
	}

	docs := retrieverResp.Documents
	if len(docs) != 2 {
		t.Errorf("got %d results, expected 2", len(docs))
	}
	for _, d := range docs {
		text := d.Content[0].Text
		if !strings.HasPrefix(text, "hello") {
			t.Errorf("returned doc text %q does not start with %q", text, "hello")
		}
		name, ok := d.Metadata["name"]
		if !ok {
			t.Errorf("missing metadata entry for name: %v", d.Metadata)
		} else if !strings.HasPrefix(name.(string), "hello") {
			t.Errorf("metadata name entry %q does not start with %q", name, "hello")
		}
	}
}
