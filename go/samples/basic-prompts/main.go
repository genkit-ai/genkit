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

// This sample demonstrates prompts using both inline code definitions and
// .prompt files (Dotprompt). It shows simple prompts, structured output with
// typed schemas, and complex prompts with Handlebars conditionals.
//
// To run:
//
//	go run .
//
// In another terminal, test a simple joke flow:
//
//	curl -N -X POST http://localhost:8080/simpleJokePromptFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "bananas"}'
//
// Test a structured joke flow (returns JSON):
//
//	curl -N -X POST http://localhost:8080/structuredJokePromptFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"topic": "bananas"}}'
//
// Test a recipe flow:
//
//	curl -N -X POST http://localhost:8080/recipePromptFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"dish": "tacos", "cuisine": "Mexican", "servingSize": 4}}'
//
// Test an assistant flow (with all middleware: retry, fallback, filesystem, skills):
//
//	curl -N -X POST http://localhost:8080/assistantPromptFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": "what files are in my current directory?"}'
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
	Ingredients  []*Ingredient `json:"ingredients" jsonschema:"description=The recipe ingredients (group by type and order by importance)"`
	Instructions []string      `json:"instructions" jsonschema:"description=The recipe instructions (step by step)"`
	PrepTime     string        `json:"prepTime" jsonschema:"description=The recipe preparation time (e.g. 10 minutes, 30 minutes, etc.)"`
	Difficulty   string        `json:"difficulty" jsonschema:"enum=easy,enum=medium,enum=hard"`
}

type AssistantRequest struct {
	Query string `json:"query" jsonschema:"description=The user's query or request"`
}

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the recommended
	// practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &middleware.Middleware{}))

	// Define schemas for the expected input and output types so that the Dotprompt files can reference them.
	// Alternatively, you can specify the JSON schema by hand in the Dotprompt metadata.
	// Code-defined prompts do not need to have schemas defined in advance but they too can reference them.
	genkit.DefineSchemaFor[JokeRequest](g)
	genkit.DefineSchemaFor[Joke](g)
	genkit.DefineSchemaFor[RecipeRequest](g)
	genkit.DefineSchemaFor[Recipe](g)
	genkit.DefineSchemaFor[AssistantRequest](g)

	// TODO: Include partials and helpers.

	// Define the prompts and flows.
	DefineSimpleJokeWithInlinePrompt(g)
	DefineSimpleJokeWithDotprompt(g)
	DefineStructuredJokeWithInlinePrompt(g)
	DefineStructuredJokeWithDotprompt(g)
	DefineRecipeWithInlinePrompt(g)
	DefineRecipeWithDotprompt(g)
	DefineAssistantWithInlinePrompt(g)
	DefineAssistantWithDotprompt(g)

	// Optionally, start a web server to make the flows callable via HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// DefineSimpleJokeWithInlinePrompt demonstrates defining a prompt in code using DefinePrompt.
// The prompt has no output schema defined so it will always return a string.
// When executing the prompt, we pass in a map[string]any with the input fields.
func DefineSimpleJokeWithInlinePrompt(g *genkit.Genkit) {
	jokePrompt := genkit.DefinePrompt(
		g, "joke.code",
		ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{
				ThinkingBudget: genai.Ptr[int32](0),
			},
		})),
		// Despite JokeRequest having defaults set in jsonschema tags, we can override it with values set in WithInputType.
		ai.WithInputType(JokeRequest{Topic: "rush hour traffic"}),
		ai.WithPrompt("Share a long joke about {{topic}}."),
	)

	genkit.DefineStreamingFlow(g, "simpleJokePromptFlow",
		func(ctx context.Context, topic string, sendChunk core.StreamCallback[string]) (string, error) {
			// One way to pass input is using a map[string]any. This is useful when there is no structured input type.
			stream := jokePrompt.ExecuteStream(ctx, ai.WithInput(map[string]any{"topic": topic}))
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

// DefineSimpleJokeWithDotprompt demonstrates loading a prompt from a .prompt file using
// LoadPrompt. The prompt configuration (model, input schema, defaults) is defined in the
// file. Input is passed as a map since the .prompt file defines its own schema.
func DefineSimpleJokeWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "simpleJokeDotpromptFlow",
		func(ctx context.Context, topic string, sendChunk core.StreamCallback[string]) (string, error) {
			jokePrompt := genkit.LookupPrompt(g, "joke")
			// One way to pass input is using a map[string]any. This is useful when there is no structured input type.
			stream := jokePrompt.ExecuteStream(ctx, ai.WithInput(map[string]any{"topic": topic}))
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

// DefineStructuredJokeWithInlinePrompt demonstrates DefineDataPrompt for strongly-typed
// input and output. The type parameters automatically configure input/output schemas
// and JSON output format. ExecuteStream returns typed chunks and final output.
func DefineStructuredJokeWithInlinePrompt(g *genkit.Genkit) {
	jokePrompt := genkit.DefineDataPrompt[JokeRequest, *Joke](
		g, "structured-joke.code",
		ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{
				ThinkingBudget: genai.Ptr[int32](0),
			},
		})),
		ai.WithPrompt("Share a long joke about {{topic}}."),
	)

	genkit.DefineStreamingFlow(g, "structuredJokePromptFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[*Joke]) (*Joke, error) {
			for result, err := range jokePrompt.ExecuteStream(ctx, input) {
				if err != nil {
					return nil, fmt.Errorf("could not generate joke: %w", err)
				}
				if result.Done {
					return result.Output, nil
				}
				sendChunk(ctx, result.Chunk)
			}

			return nil, nil
		},
	)
}

// DefineStructuredJokeWithDotprompt demonstrates LookupDataPrompt to wrap a .prompt file
// with Go type information. The .prompt file references registered schemas by name
// (e.g., "schema: Joke"), which must be defined via DefineSchemaFor before loading.
func DefineStructuredJokeWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "structuredJokeDotpromptFlow",
		func(ctx context.Context, input JokeRequest, sendChunk core.StreamCallback[*Joke]) (*Joke, error) {
			jokePrompt := genkit.LookupDataPrompt[JokeRequest, *Joke](g, "structured-joke")
			stream := jokePrompt.ExecuteStream(ctx, input)
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
		},
	)
}

// DefineRecipeWithInlinePrompt demonstrates DefineDataPrompt with complex nested types
// and Handlebars conditionals/loops in the prompt template. The streaming flow applies
// default values before execution and streams partial ingredients as they arrive.
func DefineRecipeWithInlinePrompt(g *genkit.Genkit) {
	recipePrompt := genkit.DefineDataPrompt[RecipeRequest, *Recipe](
		g, "recipe.code",
		ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
			ThinkingConfig: &genai.ThinkingConfig{
				ThinkingBudget: genai.Ptr[int32](0),
			},
		})),
		ai.WithSystem("You are an experienced chef. Come up with easy, creative recipes."),
		ai.WithPrompt("Create a {{cuisine}} {{dish}} recipe for {{servingSize}} people that takes under {{maxPrepMinutes}} minutes to prepare. "+
			"{{#if dietaryRestrictions}}Dietary restrictions: {{#each dietaryRestrictions}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}.{{/if}}"),
	)

	genkit.DefineStreamingFlow(g, "recipePromptFlow",
		func(ctx context.Context, input RecipeRequest, sendChunk core.StreamCallback[*Ingredient]) (*Recipe, error) {
			// This is not necessary for this example but it shows how to easily have more control over what you stream.
			filterNew := newIngredientFilter()
			for result, err := range recipePrompt.ExecuteStream(ctx, input) {
				if err != nil {
					return nil, fmt.Errorf("could not generate recipe: %w", err)
				}
				if result.Done {
					return result.Output, nil
				}
				for _, i := range filterNew(result.Chunk.Ingredients) {
					sendChunk(ctx, i)
				}
			}
			return nil, nil
		},
	)
}

// DefineRecipeWithDotprompt demonstrates LookupDataPrompt with a .prompt file that uses
// multi-message format (system/user roles) and references registered schemas.
// Streams partial ingredients as they arrive via ExecuteStream.
func DefineRecipeWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "recipeDotpromptFlow",
		func(ctx context.Context, input RecipeRequest, sendChunk core.StreamCallback[*Ingredient]) (*Recipe, error) {
			// This is not necessary for this example but it shows how to easily have more control over what you stream.
			filterNew := newIngredientFilter()
			recipePrompt := genkit.LookupDataPrompt[RecipeRequest, *Recipe](g, "recipe")
			stream := recipePrompt.ExecuteStream(ctx, input)
			for result, err := range stream {
				if err != nil {
					return nil, fmt.Errorf("could not generate recipe: %w", err)
				}
				if result.Done {
					return result.Output, nil
				}
				for _, i := range filterNew(result.Chunk.Ingredients) {
					sendChunk(ctx, i)
				}
			}
			return nil, nil
		},
	)
}

// newIngredientFilter is a helper function to filter out duplicate ingredients.
// This allows us to stream only new ingredients as they are identified, avoiding duplicates.
func newIngredientFilter() func([]*Ingredient) []*Ingredient {
	seen := map[string]struct{}{}
	return func(ings []*Ingredient) (newIngs []*Ingredient) {
		for _, ing := range ings {
			if _, ok := seen[ing.Name]; !ok {
				seen[ing.Name] = struct{}{}
				newIngs = append(newIngs, ing)
			}
		}
		return
	}
}

// DefineAssistantWithInlinePrompt demonstrates attaching multiple middlewares
// (Retry, Fallback, Filesystem, and Skills) directly to a prompt definition.
// This creates a highly capable and resilient prompt that is also fully
// transparent in the Dev UI metadata.
func DefineAssistantWithInlinePrompt(g *genkit.Genkit) {
	assistantPrompt := genkit.DefinePrompt(
		g, "assistant.code",
		ai.WithModelName("googleai/gemini-flash-latest"),
		ai.WithPrompt("{{query}}"),
		ai.WithInputType(AssistantRequest{}),
		ai.WithUse(
			&middleware.Retry{MaxRetries: 2},
			&middleware.Fallback{
				Models: []ai.ModelRef{
					googlegenai.ModelRef("googleai/gemini-3.1-flash-preview", nil),
					googlegenai.ModelRef("googleai/gemini-3.1-pro-preview", &genai.GenerateContentConfig{
						Temperature: genai.Ptr[float32](2.0),
					}),
				},
			},
			&middleware.Filesystem{RootDir: "."},
			&middleware.Skills{SkillPaths: []string{"./skills"}},
		),
	)

	genkit.DefineStreamingFlow(g, "assistantPromptFlow",
		func(ctx context.Context, query string, sendChunk core.StreamCallback[string]) (string, error) {
			stream := assistantPrompt.ExecuteStream(ctx, ai.WithInput(&AssistantRequest{Query: query}))
			for result, err := range stream {
				if err != nil {
					return "", fmt.Errorf("assistant error: %w", err)
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

// DefineAssistantWithDotprompt demonstrates loading a prompt from a .prompt file
// that includes a full suite of middleware configuration in its YAML frontmatter.
func DefineAssistantWithDotprompt(g *genkit.Genkit) {
	genkit.DefineStreamingFlow(g, "assistantDotpromptFlow",
		func(ctx context.Context, query string, sendChunk core.StreamCallback[string]) (string, error) {
			// The "assistant" prompt file includes all middleware in its frontmatter.
			assistantPrompt := genkit.LookupPrompt(g, "assistant")
			stream := assistantPrompt.ExecuteStream(ctx, ai.WithInput(&AssistantRequest{Query: query}))
			for result, err := range stream {
				if err != nil {
					return "", fmt.Errorf("agent error: %w", err)
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
