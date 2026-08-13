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

// This sample demonstrates the base compat_oai plugin pointed at a custom
// OpenAI-compatible provider, here OpenRouter: models resolve dynamically by
// name and take the OpenAI SDK's own request type as their config.
//
// To run:
//
//	export OPENROUTER_API_KEY=...
//	go run .
//
// In another terminal:
//
//	curl -X POST http://localhost:8080/jokesFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "bananas"}'
package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/openai/openai-go"
)

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY environment variable not set")
	}

	g := genkit.Init(ctx, genkit.WithPlugins(&compat_oai.OpenAICompatible{
		Provider: "openrouter",
		APIKey:   apiKey,
		BaseURL:  "https://openrouter.ai/api/v1",
	}))

	// Define a flow that generates a joke about a given topic. Any model the
	// provider serves resolves by name under the plugin's provider prefix.
	genkit.DefineFlow(g, "jokesFlow", func(ctx context.Context, topic string) (string, error) {
		if topic == "" {
			topic = "airplane food"
		}

		return genkit.GenerateText(ctx, g,
			ai.WithModelName("openrouter/tngtech/deepseek-r1t2-chimera:free"),
			ai.WithConfig(&openai.ChatCompletionNewParams{
				Temperature: openai.Float(0.7),
				MaxTokens:   openai.Int(1024),
			}),
			ai.WithPrompt("Share a joke about %s.", topic),
		)
	})

	// Optionally, start a web server to make the flow callable via HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
