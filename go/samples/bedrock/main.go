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

// Bedrock-rag is a minimal sample showing how to use the Bedrock plugin
// for text generation (Claude), tool calling, and embeddings (Titan).
//
// The sample assumes:
//   - You have AWS credentials configured via the standard chain (env vars,
//     ~/.aws/credentials, instance role, etc.).
//   - You have requested + been granted access to the configured Claude and
//     Titan models in the target region via the Bedrock console.
//
// Run:
//
//	export AWS_REGION=us-east-1
//	go run .
//
// Trigger the chat flow:
//
//	curl -X POST http://localhost:8080/chatFlow \
//	    -H "Content-Type: application/json" \
//	    -d '{"data": "What is the temperature in San Francisco?"}'
//
// Trigger the embedding flow:
//
//	curl -X POST http://localhost:8080/embedFlow \
//	    -H "Content-Type: application/json" \
//	    -d '{"data": "Embed this sentence."}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/bedrock"
	"github.com/firebase/genkit/go/plugins/server"
)

const (
	claudeModelID  = "anthropic.claude-3-5-sonnet-20241022-v2:0"
	titanEmbedID   = "amazon.titan-embed-text-v2:0"
)

type weatherIn struct {
	Location string `json:"location" jsonschema:"description=City name"`
}

type weatherOut struct {
	TemperatureF float32 `json:"temperatureF"`
	Conditions   string  `json:"conditions"`
}

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx, genkit.WithPlugins(&bedrock.Bedrock{
		// Empty Region falls through to AWS_REGION env / shared config.
	}))

	claude, err := bedrock.DefineModel(g, claudeModelID, nil)
	if err != nil {
		log.Fatalf("DefineModel: %v", err)
	}
	titan, err := bedrock.DefineEmbedder(g, titanEmbedID, nil)
	if err != nil {
		log.Fatalf("DefineEmbedder: %v", err)
	}

	// A trivial tool that pretends to fetch weather. Real flows would call an
	// API here; the point is to exercise the tool round-trip end-to-end.
	weather := genkit.DefineTool(g, "get_weather", "Get the current temperature and conditions for a city.",
		func(ctx *ai.ToolContext, in weatherIn) (weatherOut, error) {
			return weatherOut{TemperatureF: 68, Conditions: "foggy"}, nil
		})

	genkit.DefineFlow(g, "chatFlow", func(ctx context.Context, question string) (string, error) {
		if question == "" {
			question = "What's the weather like in San Francisco?"
		}
		return genkit.GenerateText(ctx, g,
			ai.WithModel(claude),
			ai.WithSystem("You are a weather concierge. Use the get_weather tool when asked about temperature or conditions."),
			ai.WithPrompt(question),
			ai.WithTools(weather),
			ai.WithMaxTurns(5),
			ai.WithConfig(&bedrock.Config{MaxTokens: 512}),
		)
	})

	genkit.DefineFlow(g, "embedFlow", func(ctx context.Context, text string) (int, error) {
		if text == "" {
			text = "Genkit on Bedrock"
		}
		resp, err := genkit.Embed(ctx, g, ai.WithEmbedder(titan), ai.WithTextDocs(text))
		if err != nil {
			return 0, fmt.Errorf("embed: %w", err)
		}
		if len(resp.Embeddings) == 0 {
			return 0, fmt.Errorf("empty embedding")
		}
		return len(resp.Embeddings[0].Embedding), nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
