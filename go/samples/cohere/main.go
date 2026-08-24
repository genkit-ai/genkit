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

// Command cohere demonstrates the Cohere plugin: non-streaming generation,
// streaming, tool calling, and embeddings. Requires COHERE_API_KEY.
package main

import (
	"context"
	"fmt"
	"log"

	cohere "github.com/cohere-ai/cohere-go/v2"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	coheregenkit "github.com/firebase/genkit/go/plugins/cohere"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&coheregenkit.Cohere{}))

	maxTokens := 512
	temperature := 0.3
	config := &cohere.V2ChatRequest{MaxTokens: &maxTokens, Temperature: &temperature}

	// 1. Non-streaming generation.
	fmt.Println("=== generate ===")
	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("cohere/command-r"),
		ai.WithConfig(config),
		ai.WithPrompt("In one sentence, what is Genkit?"),
	)
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	fmt.Println(resp.Text())

	// 2. Streaming generation.
	fmt.Println("\n=== streaming ===")
	_, err = genkit.Generate(ctx, g,
		ai.WithModelName("cohere/command-r"),
		ai.WithConfig(config),
		ai.WithPrompt("Write a two-line poem about the ocean."),
		ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
			for _, p := range chunk.Content {
				if p.IsText() {
					fmt.Print(p.Text)
				}
			}
			return nil
		}),
	)
	if err != nil {
		log.Fatalf("streaming: %v", err)
	}
	fmt.Println()

	// 3. Tool calling.
	fmt.Println("\n=== tool calling ===")
	addTool := genkit.DefineTool(g, "add", "adds two integers",
		func(toolCtx *ai.ToolContext, input struct {
			A int `json:"a"`
			B int `json:"b"`
		}) (int, error) {
			return input.A + input.B, nil
		},
	)
	toolResp, err := genkit.Generate(ctx, g,
		ai.WithModelName("cohere/command-r-plus"),
		ai.WithConfig(config),
		ai.WithPrompt("Use the add tool to compute 17 plus 25, then state only the result."),
		ai.WithTools(addTool),
	)
	if err != nil {
		log.Fatalf("tool calling: %v", err)
	}
	fmt.Println(toolResp.Text())

	// 4. Embeddings.
	fmt.Println("\n=== embed ===")
	embedResp, err := genkit.Embed(ctx, g,
		ai.WithEmbedderName("cohere/embed-v4.0"),
		ai.WithConfig(&coheregenkit.EmbedOptions{InputType: "search_document"}),
		ai.WithTextDocs("the quick brown fox", "jumps over the lazy dog"),
	)
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	for i, e := range embedResp.Embeddings {
		fmt.Printf("embedding %d: %d dims, first values %.4f\n", i, len(e.Embedding), e.Embedding[:min(3, len(e.Embedding))])
	}
}
