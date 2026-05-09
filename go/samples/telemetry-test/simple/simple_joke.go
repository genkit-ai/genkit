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

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Firebase telemetry
	firebase.EnableFirebaseTelemetry(&firebase.FirebaseTelemetryOptions{
		ForceDevExport: true, // Force telemetry export in development
	})

	// Initialize Genkit with plugins
	g := genkit.Init(ctx, genkit.WithPlugins(
		&googlegenai.GoogleAI{},
	))

	// Define a simple joke flow
	genkit.DefineFlow(g, "jokeFlow", func(ctx context.Context, topic string) (string, error) {
		// Generate a joke using Gemini
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.5-flash"),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](1.0),
			}),
			ai.WithPrompt("Tell me a clean, family-friendly joke about %s. Just return the joke, nothing else.", topic))
		if err != nil {
			return "", fmt.Errorf("failed to generate joke: %w", err)
		}

		return resp.Text(), nil
	})

	// Start the server
	fmt.Println("Simple Joke Flow with Firebase Telemetry")
	fmt.Println("Server: http://localhost:3400")
	fmt.Println("")
	fmt.Println("Test with:")
	fmt.Println(`curl -X POST http://localhost:3400/jokeFlow -H 'Content-Type: application/json' -d '{"data": "cats"}'`)

	mux := http.NewServeMux()
	for _, flow := range genkit.ListFlows(g) {
		fmt.Printf("Registered flow: %s\n", flow.Name())
		mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))
	}

	fmt.Println("\nStarting server on http://127.0.0.1:3400...")
	if err := server.Start(ctx, "127.0.0.1:3400", mux); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
