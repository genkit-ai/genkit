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

// This sample demonstrates structured input/output with strongly-typed Go
// structs. It shows GenerateStream for simple output and GenerateDataStream
// for typed JSON output with streaming partial results.
//
// To run:
//
//	go run .
//
// In another terminal, test a simple joke flow:
//
//	curl -N -X POST http://localhost:8080/simpleJokesFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "bananas"}'
//
// Test a structured joke flow (returns JSON):
//
//	curl -N -X POST http://localhost:8080/structuredJokesFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"topic": "bananas"}}'
//
// Test a recipe flow:
//
//	curl -N -X POST http://localhost:8080/recipeFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"dish": "tacos", "cuisine": "Mexican", "servingSize": 4}}'
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
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

type JokeRequest struct {
	Topic string `json:"topic" jsonschema:"default=airplane food"`
}

// Note how the fields are annotated with jsonschema tags to describe the output schema.
// This is vital for the model to understand the intent of the fields.
type Joke struct {
	Joke     string `json:"joke" jsonschema:"description=The joke text"`
	Category string `json:"category" jsonschema:"description=The joke category"`
}

type RecipeRequest struct {
	Dish                string   `json:"dish" jsonschema:"default=pasta"`
	Cuisine             string   `json:"cuisine" jsonschema:"default=Italian"`
	ServingSize         int      `json:"servingSize" jsonschema:"default=4"`
	MaxPrepMinutes      int      `json:"maxPrepMinutes" jsonschema:"default=30"`
	DietaryRestrictions []string `json:"dietaryRestrictions,omitempty"`
}

type Ingredient struct {
	Name     string `json:"name" jsonschema:"description=The ingredient name"`
	Amount   string `json:"amount" jsonschema:"description=The ingredient amount (e.g. 1 cup, 2 tablespoons, etc.)"`
	Optional bool   `json:"optional,omitempty" jsonschema:"description=Whether the ingredient is optional in the recipe"`
}

type Recipe struct {
	Title        string        `json:"title" jsonschema:"description=The recipe title (e.g. 'Spicy Chicken Tacos')"`
	Description  string        `json:"description,omitempty" jsonschema:"description=The recipe description (under 100 characters)"`
	Ingredients  []*Ingredient `json:"ingredients" jsonschema:"description=The recipe ingredients (order by type first and then importance)"`
	Instructions []string      `json:"instructions" jsonschema:"description=The recipe instructions (step by step)"`
	PrepTime     string        `json:"prepTime" jsonschema:"description=The recipe preparation time (e.g. 10 minutes, 30 minutes, etc.)"`
	Difficulty   string        `json:"difficulty" jsonschema:"enum=easy,enum=medium,enum=hard"`
}

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the recommended
	// practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Define the flows.
	DefineSimpleJoke(g)
	DefineStructuredJoke(g)
	DefineRecipe(g)

	// Optionally, start a web server to make the flows callable via HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineSimpleJoke demonstrates defining a streaming flow that generates a joke about a given topic.
func DefineSimpleJoke(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "simpleJokesFlow",
		func(ctx context.Context, input string, sendChunk core.StreamCallback[string]) (string, error) {
			stream := genkit.GenerateStream(ctx, g,
				ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
					ThinkingConfig: &genai.ThinkingConfig{
						ThinkingBudget: genai.Ptr[int32](0),
					},
				})),
				ai.WithPrompt("Share a long joke about %s.", input),
			)

			for result, err := range stream {
				if err != nil {
					return "", fmt.Errorf("could not generate joke: %w", err)
				}
				if result.Done {
					return result.Response.Text(), nil
				}
				sendChunk(ctx, result.Chunk.Text())
			}

			return "", nil
		},
	)
}

// DefineStructuredJoke demonstrates defining a streaming flow that generates a joke about a given topic.
// The input is a strongly-typed JokeRequest struct and the output is a strongly-typed Joke struct.
func DefineStructuredJoke(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "structuredJokesFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[*Joke]) (*Joke, error) {
			stream := genkit.GenerateDataStream[*Joke](ctx, g,
				ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
					ThinkingConfig: &genai.ThinkingConfig{
						ThinkingBudget: genai.Ptr[int32](0),
					},
				})),
				ai.WithPrompt("Share a long joke about %s.", input.Topic),
			)

			for result, err := range stream {
				if err != nil {
					return nil, fmt.Errorf("could not generate joke: %w", err)
				}
				if result.Done {
					return result.Output, nil
				}
				sendChunk(ctx, result.Chunk)
			}

			return nil, nil
		})
}

// DefineRecipe demonstrates defining a streaming flow that generates a recipe based on a given RecipeRequest struct.
// The input is a strongly-typed RecipeRequest struct and the output is a strongly-typed Recipe struct.
func DefineRecipe(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "recipeFlow",
		func(ctx context.Context, input RecipeRequest, sendChunk core.StreamCallback[[]*Ingredient]) (*Recipe, error) {
			stream := genkit.GenerateDataStream[*Recipe](ctx, g,
				ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
					ThinkingConfig: &genai.ThinkingConfig{
						ThinkingBudget: genai.Ptr[int32](0),
					},
				})),
				ai.WithSystem("You are an experienced chef. Come up with easy, creative recipes."),
				// Here we are passing WithPromptFn() since our prompt takes some string manipulation to build.
				// Alternatively, we could pass WithPrompt() with the complete prompt string.
				ai.WithPromptFn(func(ctx context.Context, _ any) (string, error) {
					prompt := fmt.Sprintf(
						"Create a %s %s recipe for %d people that takes under %d minutes to prepare.",
						input.Cuisine, input.Dish, input.ServingSize, input.MaxPrepMinutes,
					)
					if len(input.DietaryRestrictions) > 0 {
						prompt += fmt.Sprintf(" Dietary restrictions: %v.", input.DietaryRestrictions)
					}
					return prompt, nil
				}),
			)

			for result, err := range stream {
				if err != nil {
					return nil, fmt.Errorf("could not generate recipe: %w", err)
				}
				if result.Done {
					return result.Output, nil
				}
				sendChunk(ctx, result.Chunk.Ingredients)
			}

			return nil, nil
		})
}
