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

// This sample demonstrates basic Genkit flows: a non-streaming flow and a
// streaming flow that generate jokes about a given topic.
//
// To run:
//
//	go run .
//
// In another terminal, test the non-streaming flow:
//
//	curl -X POST http://localhost:8080/jokesFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "bananas"}'
//
// Test the streaming flow:
//
//	curl -N -X POST http://localhost:8080/streamingJokesFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "bananas"}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the recommended
	// practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Define a non-streaming flow that generates jokes about a given topic.
	genkit.DefineFlow(g, "jokesFlow", func(ctx context.Context, input string) (string, error) {
		if input == "" {
			input = "airplane food"
		}

		return genkit.GenerateText(ctx, g,
			ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			})),
			ai.WithPrompt("Share a joke about %s.", input),
		)
	})

	// Define a streaming flow that generates jokes about a given topic with passthrough streaming.
	genkit.DefineStreamingFlow(g, "streamingJokesFlow",
		func(ctx context.Context, input string, sendChunk ai.ModelStreamCallback) (string, error) {
			if input == "" {
				input = "airplane food"
			}

			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
					ThinkingConfig: &genai.ThinkingConfig{
						ThinkingBudget: genai.Ptr[int32](0),
					},
				})),
				ai.WithPrompt("Share a joke about %s.", input),
				ai.WithStreaming(sendChunk),
			)
			if err != nil {
				return "", fmt.Errorf("could not generate joke: %w", err)
			}

			return resp.Text(), nil
		},
	)

	// Optionally, start a web server to make the flow callable via HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
