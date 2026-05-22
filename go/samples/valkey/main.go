// Full RAG demo: Valkey vector search via the Genkit Go plugin.
// Indexes 4 documents into Valkey, retrieves the top-2 matches for a query,
// then prints the retrieved documents.
//
// Prerequisites:
//  1. Valkey 8+ with valkey-search module:
//     docker run -d --name valkey -p 6379:6379 valkey/valkey:8-alpine
//  2. From the repo root: go mod tidy
//  3. An embedder registered in your Genkit setup. This sample uses the
//     fakeembedder for self-contained execution; swap for a real embedder
//     (e.g., googleai, ollama) in production.
//     If using Ollama: ollama pull nomic-embed-text
//
// Run (from repo root):
//
//	go run go/plugins/valkey/valkey_sample.go -valkey-addr localhost:6379

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
	"github.com/firebase/genkit/go/internal/fakeembedder"
	valkeyplugin "github.com/firebase/genkit/go/plugins/valkey"
	"github.com/valkey-io/valkey-glide/go/v2/config"
)

var valkeyAddr = flag.String("valkey-addr", "localhost:6379", "Valkey address host:port")

func main() {
	flag.Parse()

	ctx := context.Background()

	// --- Parse address ---
	parts := strings.SplitN(*valkeyAddr, ":", 2)
	host := parts[0]
	port := 6379
	if len(parts) == 2 {
		if _, err := fmt.Sscanf(parts[1], "%d", &port); err != nil {
			log.Fatalf("invalid port in address %q: %v", *valkeyAddr, err)
		}
	}

	// --- Bootstrap: init Genkit with the Valkey plugin ---
	g := genkit.Init(ctx, genkit.WithPlugins(&valkeyplugin.Valkey{
		Addresses: []config.NodeAddress{{Host: host, Port: port}},
	}))

	// --- Define a fake embedder (3-dimensional) ---
	const dim = 3
	const indexName = "coffee-menu-go"

	docs := []*ai.Document{
		ai.DocumentFromText("Espresso: concentrated coffee brewed under pressure. $3.50", nil),
		ai.DocumentFromText("Latte: espresso with steamed milk and foam. $4.75", nil),
		ai.DocumentFromText("Cold Brew: steeped in cold water 12-24h, served chilled. $4.25", nil),
		ai.DocumentFromText("Croissant: buttery flaky French pastry. $3.00", nil),
	}

	// Assign hand-crafted vectors: coffee drinks cluster near [1,0,0]; food near [0,0,1].
	vectors := [][]float32{
		{1.0, 0.0, 0.0},
		{0.9, 0.1, 0.0},
		{0.8, 0.2, 0.0},
		{0.0, 0.0, 1.0},
	}

	fake := fakeembedder.New()
	for i, d := range docs {
		fake.Register(d, vectors[i])
	}

	emdOpts := &ai.EmbedderOptions{
		Dimensions: dim,
		Label:      "fake-embedder",
		Supports:   &ai.EmbedderSupports{Input: []string{"text"}},
	}
	embedder := genkit.DefineEmbedder(g, "fake/embedder", emdOpts, fake.Embed)

	// --- Define retriever (creates FT index if absent) ---
	cfg := valkeyplugin.Config{
		IndexName: indexName,
		Embedder:  embedder,
		Dimension: dim,
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

	// --- Indexing flow ---
	if err := valkeyplugin.Index(ctx, docs, ds); err != nil {
		log.Fatalf("Index: %v", err)
	}
	fmt.Printf("Indexed %d documents into %q.\n", len(docs), indexName)

	// --- Retrieval flow ---
	query := ai.DocumentFromText("What cold coffee drinks do you have?", nil)
	// Register query vector (similar to espresso/latte/cold-brew cluster).
	fake.Register(query, []float32{0.85, 0.15, 0.0})

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
