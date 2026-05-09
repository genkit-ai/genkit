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

// This sample demonstrates how to use sessions to maintain state across
// multiple requests. It implements a shopping cart where items persist
// between calls using the session API.
//
// To run:
//
//	go run .
//
// In another terminal, test (items persist across requests):
//
//	curl -X POST http://localhost:8080/manageCart \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "Add apples and bananas to my cart"}'
//
//	curl -X POST http://localhost:8080/manageCart \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "What is in my cart?"}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// CartState holds the shopping cart items.
type CartState struct {
	Items []string `json:"items"`
}

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Create in-memory store (shared across requests).
	store := session.NewInMemoryStore[CartState]()

	// Fixed session ID for simplicity.
	const sessionID = "shopping-session"

	// Define addToCart tool - adds an item to the cart stored in session state.
	addToCartTool := genkit.DefineTool(g, "addToCart",
		"Adds items to the shopping cart",
		func(ctx *ai.ToolContext, input struct{ Items []string }) ([]string, error) {
			sess := session.FromContext[CartState](ctx.Context)
			if sess == nil {
				return nil, fmt.Errorf("no session in context")
			}
			state := sess.State()
			state.Items = append(state.Items, input.Items...)
			if err := sess.UpdateState(ctx.Context, state); err != nil {
				return nil, err
			}
			return state.Items, nil
		},
	)

	// Define getCart tool - returns all items currently in the cart.
	getCartTool := genkit.DefineTool(g, "getCart",
		"Returns all items currently in the shopping cart",
		func(ctx *ai.ToolContext, input struct{}) ([]string, error) {
			sess := session.FromContext[CartState](ctx.Context)
			if sess == nil {
				return nil, fmt.Errorf("no session in context")
			}
			return sess.State().Items, nil
		},
	)

	// Define flow that uses session to maintain cart state across requests.
	genkit.DefineFlow(g, "manageCart", func(ctx context.Context, input string) (string, error) {
		// Load existing session or create new one.
		sess, err := session.Load(ctx, store, sessionID)
		if err != nil {
			// Session doesn't exist, create it.
			sess, err = session.New(ctx,
				session.WithID[CartState](sessionID),
				session.WithStore(store),
				session.WithInitialState(CartState{Items: []string{}}),
			)
			if err != nil {
				return "", err
			}
		}

		// Attach session to context for tools.
		ctx = session.NewContext(ctx, sess)

		return genkit.GenerateText(ctx, g,
			ai.WithModel(googlegenai.ModelRef("gemini-2.5-flash", &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			})),
			ai.WithSystem("You are a helpful shopping assistant. Use the provided tools to manage the user's cart."),
			ai.WithTools(addToCartTool, getCartTool),
			ai.WithPrompt(input),
		)
	})

	// Start server.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
