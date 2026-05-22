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

// Full RAG demo: Valkey vector search via the Genkit Go plugin.
// Indexes 4 documents into Valkey, retrieves the top-2 matches for a query,
// then prints the retrieved documents.
//
// Prerequisites:
//  1. Valkey 8+ with valkey-search module:
//     docker run -d --name valkey -p 6379:6379 valkey/valkey:8-alpine
//  2. Ollama running locally with nomic-embed-text pulled:
//     ollama pull nomic-embed-text
//  3. From the go/ directory: go mod tidy
//
// Run (from go/ directory):
//
//	go run ./samples/valkey/main.go -valkey-addr localhost:6379

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/ollama"
	valkeyplugin "github.com/firebase/genkit/go/plugins/valkey"
	"github.com/valkey-io/valkey-glide/go/v2/config"
)

var (
	valkeyAddr = flag.String("valkey-addr", "localhost:6379", "Valkey address host:port")
	ollamaAddr = flag.String("ollama-addr", "http://localhost:11434", "Ollama server address")
)

const (
	indexName = "coffee-menu-go"
	dimension = 768 // nomic-embed-text output dimension
)

func main() {
	flag.Parse()

	ctx := context.Background()

	// --- Parse Valkey address ---
	parts := strings.SplitN(*valkeyAddr, ":", 2)
	host := parts[0]
	port := 6379
	if len(parts) == 2 {
		if _, err := fmt.Sscanf(parts[1], "%d", &port); err != nil {
			log.Fatalf("invalid port in address %q: %v", *valkeyAddr, err)
		}
	}

	// --- Bootstrap: init Genkit with Ollama + Valkey plugins ---
	ollamaPlugin := &ollama.Ollama{ServerAddress: *ollamaAddr}

	g := genkit.Init(ctx, genkit.WithPlugins(
		ollamaPlugin,
		&valkeyplugin.Valkey{
			Addresses: []config.NodeAddress{{Host: host, Port: port}},
		},
	))

	// --- Define embedder using Ollama nomic-embed-text ---
	embedder := ollamaPlugin.DefineEmbedder(g, *ollamaAddr, "nomic-embed-text", &ai.EmbedderOptions{
		Dimensions: dimension,
		Label:      "nomic-embed-text",
		Supports:   &ai.EmbedderSupports{Input: []string{"text"}},
	})

	// --- Define retriever (creates FT index if absent) ---
	cfg := valkeyplugin.Config{
		IndexName: indexName,
		Embedder:  embedder,
		Dimension: dimension,
	}
	retOpts := &ai.RetrieverOptions{
		ConfigSchema: core.InferSchemaMap(valkeyplugin.RetrieverOptions{}),
		Label:        "valkey-coffee-menu",
		Supports:     &ai.RetrieverSupports{Media: false},
	}
	ds, retriever, err := valkeyplugin.DefineRetriever(ctx, g, cfg, retOpts)
	if err != nil {
		log.Fatalf("DefineRetriever: %v", err)
	}

	// --- Index documents ---
	docs := []*ai.Document{
		ai.DocumentFromText("Espresso: a concentrated coffee brewed by forcing hot water through finely-ground beans. $3.50", nil),
		ai.DocumentFromText("Latte: espresso with steamed milk and a thin layer of foam. $4.75", nil),
		ai.DocumentFromText("Cold Brew: coffee steeped in cold water for 12-24 hours, served chilled. $4.25", nil),
		ai.DocumentFromText("Croissant: a buttery, flaky pastry of French origin. $3.00", nil),
	}

	if err := ds.Index(ctx, docs); err != nil {
		log.Fatalf("Index: %v", err)
	}
	fmt.Printf("Indexed %d documents into %q.\n", len(docs), indexName)

	// --- Retrieve ---
	query := ai.DocumentFromText("What cold coffee drinks do you have?", nil)

	resp, err := genkit.Retrieve(ctx, g,
		ai.WithRetriever(retriever),
		ai.WithDocs(query),
		ai.WithConfig(&valkeyplugin.RetrieverOptions{K: 2}),
	)
	if err != nil {
		log.Fatalf("Retrieve: %v", err)
	}

	fmt.Printf("\nRetrieved %d documents for: %q\n", len(resp.Documents), query.Content[0].Text)
	for i, d := range resp.Documents {
		fmt.Printf("  [%d] %s\n", i+1, d.Content[0].Text)
	}
}
