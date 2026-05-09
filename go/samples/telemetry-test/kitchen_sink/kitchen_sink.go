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

type SearchInput struct {
	Query string `json:"query" description:"The search query or topic to search for"`
}

type SearchResult struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}

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

	// Define a tool for web search simulation - using struct input for model compatibility
	searchTool := genkit.DefineTool(g, "webSearch",
		"Search the web for information about a topic",
		func(ctx *ai.ToolContext, input SearchInput) (*SearchResult, error) {
			// Simulate search with some realistic results
			return &SearchResult{
				Title:   fmt.Sprintf("Search results for: %s", input.Query),
				Content: fmt.Sprintf("Here are the top results about %s with detailed information and recent updates.", input.Query),
			}, nil
		},
	)

	// Flow 1: Text with Tools
	genkit.DefineFlow(g, "textFlow", func(ctx context.Context, topic string) (string, error) {
		if topic == "" {
			topic = "artificial intelligence"
		}

		searchResp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.5-flash"),
			ai.WithTools(searchTool),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.7),
			}),
			ai.WithPrompt("Research information about %s. Use the webSearch tool to find relevant details.", topic))
		if err != nil {
			return "", fmt.Errorf("research failed: %w", err)
		}
		return searchResp.Text(), nil
	})

	// Flow 2: Image Analysis
	type ImageRequest struct {
		ImageURL string `json:"imageUrl"`
		Prompt   string `json:"prompt,omitempty"`
	}

	genkit.DefineFlow(g, "imageFlow", func(ctx context.Context, req ImageRequest) (string, error) {
		prompt := req.Prompt
		if prompt == "" {
			prompt = "Describe what you see in this image"
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.5-flash"),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart(prompt),
					ai.NewMediaPart("", req.ImageURL),
				),
			),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.7),
			}))
		if err != nil {
			return "", fmt.Errorf("image analysis failed: %w", err)
		}
		return resp.Text(), nil
	})

	// Flow 3: Batch processing flow
	genkit.DefineFlow(g, "batchFlow", func(ctx context.Context, topics []string) (map[string]string, error) {
		results := make(map[string]string)

		for _, topic := range topics {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModelName("googleai/gemini-2.5-flash"),
				ai.WithConfig(&genai.GenerateContentConfig{
					Temperature: genai.Ptr[float32](0.8),
				}),
				ai.WithPrompt("Generate a brief, interesting fact about %s", topic))
			if err != nil {
				results[topic] = fmt.Sprintf("Error: %v", err)
			} else {
				results[topic] = resp.Text()
			}
		}

		return results, nil
	})

	// Start the server
	fmt.Println("Kitchen Sink Telemetry Demo")
	fmt.Println("Firebase telemetry with various scenarios:")
	fmt.Println("   • Tool calls and function execution")
	fmt.Println("   • Multi-step RAG operations")
	fmt.Println("   • Batch processing flows")
	fmt.Println("")
	fmt.Println("Server: http://localhost:3400")
	fmt.Println("")
	fmt.Println("Test endpoints:")
	fmt.Println()
	fmt.Println("Text + Tools:")
	fmt.Println(`curl -X POST http://localhost:3400/textFlow -H 'Content-Type: application/json' -d '{"data": "machine learning"}'`)
	fmt.Println()
	fmt.Println("Image Analysis:")
	fmt.Println(`curl -X POST http://localhost:3400/imageFlow -H 'Content-Type: application/json' -d '{"data": {"imageUrl": "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png", "prompt": "What logo is this?"}}'`)
	fmt.Println()
	fmt.Println("Batch Processing:")
	fmt.Println(`curl -X POST http://localhost:3400/batchFlow -H 'Content-Type: application/json' -d '{"data": ["AI", "robotics", "quantum"]}'`)

	mux := http.NewServeMux()
	for _, flow := range genkit.ListFlows(g) {
		fmt.Printf("Registered flow: %s\n", flow.Name())
		mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))
	}

	fmt.Println("\nAll telemetry modules active - check Google Cloud Console!")
	log.Fatal(server.Start(ctx, "127.0.0.1:3400", mux))
}
