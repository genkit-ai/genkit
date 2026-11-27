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

// This sample demonstrates the Mistral plugin: a streaming flow that
// generates a joke with a model pinned through mistral.ModelRef and its typed
// config, and a flow that embeds text with a Mistral embedder.
//
// Run it:
//
//	export MISTRAL_API_KEY=...
//	go run .
//
// Or with the Dev UI, to call the flows from a browser and read a trace of
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
//
//	curl -X POST 'http://localhost:8080/embedFlow' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"text": "Genkit is an open source framework."}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/mistral"
	"github.com/firebase/genkit/go/plugins/server"
)

// JokeRequest is what the jokes flow takes. A struct rather than a bare
// string lets the field carry a description and a default, which the Dev UI
// pre-fills its form from. The default is not applied in transit, and a field
// without omitempty is required.
type JokeRequest struct {
	Topic string `json:"topic" jsonschema:"default=airplane food" jsonschema_description:"What the joke should be about"`
}

// EmbedRequest is what the embed flow takes.
type EmbedRequest struct {
	Text string `json:"text" jsonschema:"default=Genkit is an open source framework." jsonschema_description:"The text to embed"`
}

// model pins the model and its config in one place, so switching either is a
// one-line change. Mistral Small reasons only when asked, so the config
// leaves reasoning off and caps the answer for a quick conversational reply.
var model = mistral.ModelRef("mistral-small-latest", &mistral.ChatConfig{
	MaxOutputTokens: 256,
})

// embedder pins the embedder and its config. Codestral Embed lets the caller
// choose the output size.
var embedder = mistral.EmbedderRef("codestral-embed", &mistral.EmbedConfig{
	Dimensions: 256,
})

func main() {
	ctx := context.Background()

	// The plugin reads the API key from the MISTRAL_API_KEY environment variable.
	g := genkit.Init(ctx, genkit.WithPlugins(&mistral.Mistral{}))

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

	genkit.DefineFlow(g, "embedFlow",
		func(ctx context.Context, input EmbedRequest) ([]float32, error) {
			res, err := genkit.Embed(ctx, g,
				ai.WithEmbedder(embedder),
				ai.WithTextDocs(input.Text),
			)
			if err != nil {
				return nil, fmt.Errorf("could not embed text: %w", err)
			}

			return res.Embeddings[0].Embedding, nil
		},
	)

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	if err := server.Start(ctx, "127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
