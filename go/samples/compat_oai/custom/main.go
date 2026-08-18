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
// OpenAI-compatible provider, here OpenRouter: a streaming flow that generates
// a joke with a model that resolves dynamically by name and takes the OpenAI
// SDK's own request type as its config.
//
// This is the shape for a provider with no plugin of its own. OpenRouter has
// one, in plugins/compat_oai/openrouter, and it is the better way to reach
// OpenRouter: the SDK request type has no home for the routing, fallback, and
// reasoning fields that are the reason to use the gateway, and the config
// schema rejects them. See samples/compat_oai/openrouter.
//
// Run it:
//
//	export OPENROUTER_API_KEY=...
//	go run .
//
// Or with the Dev UI, to call the flow from a browser and read a trace of
// every run at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Streaming needs ?stream=true:
//
//	curl -N -X POST 'http://localhost:8080/jokesFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"topic": "bananas"}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/openai/openai-go"
)

// JokeRequest is what the flow takes. A struct rather than a bare string lets
// the field carry a description and a default, which the Dev UI pre-fills its
// form from. The default is not applied in transit, and a field without
// omitempty is required.
type JokeRequest struct {
	Topic string `json:"topic" jsonschema:"default=airplane food" jsonschema_description:"What the joke should be about"`
}

// model pins the model and its config in one place, so switching either is a
// one-line change. The base plugin ships no typed ModelRef helper, since it
// knows nothing about the provider it is pointed at, so ai.NewModelRef pins any
// model the provider serves by name under the plugin's provider prefix.
//
// The name after the prefix is OpenRouter's, not Genkit's, and OpenRouter
// retires models on its own schedule; a 404 saying "no endpoints found" means
// this one is gone. The current catalog is at https://openrouter.ai/models.
var model = ai.NewModelRef("openrouter/deepseek/deepseek-chat", &openai.ChatCompletionNewParams{
	Temperature: openai.Float(0.7),
	MaxTokens:   openai.Int(1024),
})

func main() {
	ctx := context.Background()

	// A custom provider has no environment variable of its own, so the plugin
	// takes the key, the provider prefix, and the endpoint as fields.
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY environment variable not set")
	}

	g := genkit.Init(ctx, genkit.WithPlugins(&compat_oai.OpenAICompatible{
		Provider: "openrouter",
		APIKey:   apiKey,
		BaseURL:  "https://openrouter.ai/api/v1",
	}))

	// Passing sendChunk straight to WithStreaming forwards the model's chunks
	// to the caller untouched.
	genkit.DefineStreamingFlow(g, "jokesFlow",
		func(ctx context.Context, input JokeRequest, sendChunk ai.ModelStreamCallback) (string, error) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(model),
				ai.WithPrompt("Share a joke about %s.", input.Topic),
				ai.WithStreaming(sendChunk),
			)
			if err != nil {
				return "", fmt.Errorf("could not generate joke: %w", err)
			}

			return resp.Text(), nil
		},
	)

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
