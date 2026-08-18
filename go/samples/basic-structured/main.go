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

// This sample demonstrates typed output: the Go type you ask for is the schema
// the model is held to, so the answer comes back as a struct, not a string.
//
//   - simpleJokesFlow streams plain text with WithStreaming.
//   - structuredJokesFlow streams a typed Joke with GenerateDataStream.
//   - recipeFlow does the same for a nested Recipe.
//
// jsonschema tags describe each field to the model, which is what makes the
// generated values sensible.
//
// Run it:
//
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
//	curl -N -X POST 'http://localhost:8080/structuredJokesFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"topic": "bananas"}}'
//
//	curl -N -X POST 'http://localhost:8080/recipeFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"dish": "tacos", "cuisine": "Mexican", "servingSize": 4, "maxPrepMinutes": 30}}'
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

type JokeRequest struct {
	Topic string `json:"topic" jsonschema:"default=airplane food" jsonschema_description:"What the joke should be about"`
}

type Joke struct {
	Joke     string `json:"joke" jsonschema_description:"The joke text"`
	Category string `json:"category" jsonschema_description:"The joke category"`
}

// A jsonschema "default" is form fill for the Dev UI, not a value applied on
// the way through: every field here without omitempty is required.
type RecipeRequest struct {
	Dish                string   `json:"dish" jsonschema:"default=pasta" jsonschema_description:"The dish to cook"`
	Cuisine             string   `json:"cuisine" jsonschema:"default=Italian" jsonschema_description:"The cuisine to cook it in"`
	ServingSize         int      `json:"servingSize" jsonschema:"default=4" jsonschema_description:"How many people it should feed"`
	MaxPrepMinutes      int      `json:"maxPrepMinutes" jsonschema:"default=30" jsonschema_description:"The longest the recipe may take to prepare"`
	DietaryRestrictions []string `json:"dietaryRestrictions,omitempty" jsonschema_description:"Any dietary restrictions to respect"`
}

type Ingredient struct {
	Name     string `json:"name" jsonschema_description:"The ingredient name"`
	Amount   string `json:"amount" jsonschema_description:"The ingredient amount (e.g. 1 cup, 2 tablespoons, etc.)"`
	Optional bool   `json:"optional,omitempty" jsonschema_description:"Whether the ingredient is optional in the recipe"`
}

type Recipe struct {
	Title        string       `json:"title" jsonschema_description:"The recipe title (e.g. 'Spicy Chicken Tacos')"`
	Description  string       `json:"description,omitempty" jsonschema_description:"The recipe description (under 100 characters)"`
	Ingredients  []Ingredient `json:"ingredients" jsonschema_description:"The recipe ingredients (order by type first and then importance)"`
	Instructions []string     `json:"instructions" jsonschema_description:"The recipe instructions (step by step)"`
	PrepTime     string       `json:"prepTime" jsonschema_description:"The recipe preparation time (e.g. 10 minutes, 30 minutes, etc.)"`
	Difficulty   string       `json:"difficulty" jsonschema:"enum=easy,enum=medium,enum=hard"`
}

// model is shared by every flow below, so switching models or thinking levels
// for the whole sample is a one-line change.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

func main() {
	ctx := context.Background()

	// The Google AI plugin reads the API key from GEMINI_API_KEY or
	// GOOGLE_API_KEY, which is the recommended practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	DefineSimpleJoke(g)
	DefineStructuredJoke(g)
	DefineRecipe(g)

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineSimpleJoke streams plain text: WithStreaming forwards raw chunks, and
// the format stays "text", so nothing is parsed.
func DefineSimpleJoke(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "simpleJokesFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[string]) (string, error) {
			text, err := genkit.GenerateText(ctx, g,
				ai.WithModel(model),
				ai.WithPrompt("Share a long joke about %s.", input.Topic),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not generate joke: %w", err)
			}
			return text, nil
		},
	)
}

// DefineStructuredJoke streams a typed value: GenerateDataStream reads the
// output schema off its type parameter, so naming Joke is all it takes.
//
// Naming the value rather than *Joke is what keeps the loop free of nil
// checks: a chunk is the joke so far, and a half-filled Joke reads the same
// as one whose fields have not arrived yet.
func DefineStructuredJoke(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "structuredJokesFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[Joke]) (Joke, error) {
			for val, err := range genkit.GenerateDataStream[Joke](ctx, g,
				ai.WithModel(model),
				ai.WithPrompt("Share a long joke about %s.", input.Topic),
			) {
				if err != nil {
					return Joke{}, fmt.Errorf("could not generate joke: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				sendChunk(ctx, val.Chunk)
			}

			return Joke{}, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		})
}

// DefineRecipe does the same for a nested type, and streams only the part of
// it a caller wants to watch fill in.
func DefineRecipe(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "recipeFlow",
		func(ctx context.Context, input RecipeRequest, sendChunk core.StreamCallback[[]Ingredient]) (Recipe, error) {
			// Generate takes the request the caller already has in hand, so a
			// prompt needing string manipulation is simply built first. See
			// basic-prompt-content for the content functions registered prompts
			// use instead.
			prompt := fmt.Sprintf(
				"Create a %s %s recipe for %d people that takes under %d minutes to prepare.",
				input.Cuisine, input.Dish, input.ServingSize, input.MaxPrepMinutes,
			)
			if len(input.DietaryRestrictions) > 0 {
				prompt += fmt.Sprintf(" Dietary restrictions: %v.", input.DietaryRestrictions)
			}

			for val, err := range genkit.GenerateDataStream[Recipe](ctx, g,
				ai.WithModel(model),
				ai.WithSystem("You are an experienced chef. Come up with easy, creative recipes."),
				ai.WithPrompt(prompt),
			) {
				if err != nil {
					return Recipe{}, fmt.Errorf("could not generate recipe: %w", err)
				}
				if val.Done {
					return val.Output, nil
				}
				// Reaching into the chunk needs no guard, since the value is
				// always there and its fields fill in as they arrive. The
				// early chunks land before the list does, so this streams only
				// once there is something to show.
				if len(val.Chunk.Ingredients) > 0 {
					sendChunk(ctx, val.Chunk.Ingredients)
				}
			}

			return Recipe{}, status.Errorf(status.ErrInternal, "the stream ended without a final result")
		})
}
