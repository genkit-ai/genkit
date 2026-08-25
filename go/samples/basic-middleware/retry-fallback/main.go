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

// This sample demonstrates composing the Retry and Fallback middlewares into a
// resilient model pipeline.
//
// ai.WithUse composes outer-to-inner, so this:
//
//	ai.WithUse(&middleware.Retry{...}, &middleware.Fallback{...})
//
// becomes Retry { Fallback { model } } at call time. Fallback moves down its
// model list on a fallback-eligible status (UNAVAILABLE, NOT_FOUND,
// DEADLINE_EXCEEDED, INTERNAL, ...); if the whole cascade still fails with a
// retryable error, Retry backs off and runs it again.
//
// The primary model here is deliberately a non-existent id, so Google AI
// returns NOT_FOUND, Fallback catches it, and the real model answers on every
// run. Pointing the primary at a valid model just means Fallback never fires.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to watch both attempts in a trace of every run at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP:
//
//	curl -N -X POST 'http://localhost:8080/resilientFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"topic": "quantum computing"}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/middleware"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// TopicRequest is what the flow takes. The field carries a description and a
// default, which the Dev UI pre-fills its form from.
type TopicRequest struct {
	Topic string `json:"topic" jsonschema:"default=photosynthesis" jsonschema_description:"What to explain"`
}

// model is the working model this sample falls back to. The primary model in
// the flow below is deliberately non-existent, so this is the one that answers.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

func main() {
	ctx := context.Background()

	// Registering the Middleware plugin exposes the built-in middleware
	// (Retry, Fallback, Filesystem, Skills, ...) to the Dev UI.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}))

	// Streaming starts only once the cascade settles on a model that answers,
	// so nothing arrives until Fallback has done its work.
	genkit.DefineStreamingFlow(g, "resilientFlow",
		func(ctx context.Context, input TopicRequest, sendChunk core.StreamCallback[string]) (string, error) {
			text, err := genkit.GenerateText(ctx, g,
				// Deliberately non-existent, so Fallback fires on every request.
				ai.WithModelName("googleai/gemini-does-not-exist"),
				ai.WithPrompt("Explain %s in one concise paragraph.", input.Topic),
				ai.WithUse(
					&middleware.Retry{MaxRetries: 1},
					&middleware.Fallback{
						Models: []ai.ModelRef{model},
					},
				),
				// Forwarding every chunk is all this flow does, so WithStreaming
				// says it in one option. GenerateStream is for a flow that has to
				// act on the chunks, as basic-tools does.
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not explain the topic: %w", err)
			}
			return text, nil
		})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
