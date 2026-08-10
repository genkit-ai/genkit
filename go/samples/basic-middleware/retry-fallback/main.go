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

// This sample demonstrates composing the Retry and Fallback middlewares in a
// single Generate call to build a resilient model pipeline.
//
// The composition order passed to ai.WithUse is outer-to-inner: the first
// middleware wraps the second, which wraps the actual model call. Here:
//
//	ai.WithUse(&middleware.Retry{...}, &middleware.Fallback{...})
//
// expands to Retry { Fallback { model } } at call time:
//
//   - The primary model is invoked first.
//   - If it fails with a fallback-eligible status (UNAVAILABLE, NOT_FOUND,
//     DEADLINE_EXCEEDED, INTERNAL, ...), Fallback forwards the request to
//     the next model in its list and keeps trying until one succeeds or the
//     list is exhausted.
//   - If the whole Fallback cascade still returns a retryable error, Retry
//     sleeps with exponential backoff and runs the cascade again, up to
//     MaxRetries times.
//
// To make the fallback path visibly fire on a fresh run, the primary model
// is deliberately set to a non-existent model id — Google AI returns a
// NOT_FOUND, Fallback catches it, and the real model answers. If you switch
// the primary to a valid model, the sample still works; Fallback simply
// never triggers.
//
// To run:
//
//	go run .
//
// In another terminal, trigger the flow (the response will be produced by
// the fallback model since the primary is intentionally invalid):
//
//	curl -X POST http://localhost:8080/resilientFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "quantum computing"}'
package main

import (
	"context"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/middleware"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin and the Middleware plugin.
	// Registering the Middleware plugin exposes the built-in middleware
	// (Retry, Fallback, Filesystem, Skills, ...) to the Dev UI.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}))

	DefineResilientFlow(g)

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineResilientFlow asks the model for a one-paragraph explainer, wrapped
// in Retry + Fallback. The primary model id is intentionally bogus so the
// fallback path runs every time.
func DefineResilientFlow(g *genkit.Genkit) {
	genkit.DefineFlow(g, "resilientFlow", func(ctx context.Context, topic string) (string, error) {
		if topic == "" {
			topic = "photosynthesis"
		}

		return genkit.GenerateText(ctx, g,
			// Primary model is deliberately non-existent — this forces the
			// Fallback middleware to kick in on every request so the sample
			// is observably using the fallback path.
			ai.WithModel(googlegenai.ModelRef("googleai/gemini-does-not-exist", nil)),
			ai.WithPrompt("Explain %s in one concise paragraph.", topic),
			ai.WithUse(
				&middleware.Retry{MaxRetries: 1},
				&middleware.Fallback{
					Models: []ai.ModelRef{
						googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
							ThinkingConfig: &genai.ThinkingConfig{
								ThinkingBudget: genai.Ptr[int32](0),
							},
						}),
					},
				},
			),
		)
	})
}
