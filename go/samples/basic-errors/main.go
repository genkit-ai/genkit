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

// This sample demonstrates error handling with the core/status package.
//
// The pattern: classify a failure once, where its meaning is known, with
// status.Errorf and a sentinel (PublicErrorf when the message is safe to show
// a client); add context with fmt.Errorf and %w, which keeps the
// classification; branch with errors.Is, never on message text. At the HTTP
// boundary the status picks the response code, and only public messages reach
// the client.
//
//   - cookbookFlow produces classified errors and lets them propagate.
//   - improviseFlow consumes them, recovering from each with errors.Is.
//   - leakyFlow fails unclassified, so you can watch the boundary redact it.
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
// Or over HTTP, where the status becomes the response code:
//
//	# 400, public message: dish must not be empty.
//	curl -X POST http://localhost:8080/cookbookFlow \
//	  -H "Content-Type: application/json" -d '{"data": {"dish": ""}}'
//
//	# 404, public message: no recipe for "lasagna" in the cookbook.
//	curl -X POST http://localhost:8080/cookbookFlow \
//	  -H "Content-Type: application/json" -d '{"data": {"dish": "lasagna"}}'
//
//	# 200: the same miss, recovered from by improvising a recipe.
//	curl -X POST http://localhost:8080/improviseFlow \
//	  -H "Content-Type: application/json" -d '{"data": {"dish": "lasagna"}}'
//
//	# 500, generic message: the real text and its fake credentials stay in
//	# the server log and never reach the client.
//	curl -X POST http://localhost:8080/leakyFlow \
//	  -H "Content-Type: application/json" -d '{"data": null}'
//
// Both recipe flows stream. Asking for the stream changes where a failure
// lands, since the response has answered 200 before the flow runs: the status
// and the public message arrive in the body rather than on the status line.
//
//	# 200, with {"error":{"status":"INVALID_ARGUMENT", ...}} in the stream.
//	curl -N -X POST 'http://localhost:8080/cookbookFlow?stream=true' \
//	  -H "Content-Type: application/json" -d '{"data": {"dish": ""}}'
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

type (
	// CookbookRequest is what cookbookFlow takes. The field is required, but an
	// empty string satisfies that, which is how the INVALID_ARGUMENT branch
	// below is reached.
	CookbookRequest struct {
		Dish string `json:"dish" jsonschema:"default=pancakes" jsonschema_description:"A dish to look up in the cookbook"`
	}

	// ImproviseRequest is what improviseFlow takes.
	ImproviseRequest struct {
		Dish string `json:"dish" jsonschema:"default=lasagna" jsonschema_description:"The dish to cook"`
	}
)

// ErrRecipeNotFound classifies lookups for dishes the cookbook doesn't have.
// A subtype keeps its parent's status (404 here), and errors.Is matches it at
// either granularity: ErrRecipeNotFound for this failure, status.ErrNotFound
// for any not-found.
var ErrRecipeNotFound = status.ErrNotFound.Subtype("recipe not found")

// cookbook is the sample's tiny data store.
var cookbook = map[string]string{
	"pancakes":  "Whisk 1 cup flour, 1 tbsp sugar, 1 tsp baking powder, 1 egg, and 3/4 cup milk. Fry ladlefuls in butter until golden on both sides.",
	"shakshuka": "Simmer a can of crushed tomatoes with sauteed onion, garlic, and paprika. Crack in 4 eggs, cover, and cook until just set.",
}

// model is shared by every flow below, so switching models or thinking levels
// for the whole sample is a one-line change.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

// lookupRecipe classifies the not-found case once, at the source. The message
// only reflects what the caller sent, so PublicErrorf returns it to them.
func lookupRecipe(dish string) (string, error) {
	recipe, ok := cookbook[strings.ToLower(dish)]
	if !ok {
		return "", status.PublicErrorf(ErrRecipeNotFound, "no recipe for %q in the cookbook", dish)
	}
	return recipe, nil
}

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	// Note what this flow does NOT do: it never inspects err.Error() text, and
	// it wraps with fmt.Errorf rather than a new status, so the classification
	// chosen at the source survives to the HTTP boundary.
	genkit.DefineStreamingFlow(g, "cookbookFlow",
		func(ctx context.Context, input CookbookRequest, sendChunk core.StreamCallback[string]) (string, error) {
			dish := input.Dish
			if strings.TrimSpace(dish) == "" {
				// A bad request: classify it INVALID_ARGUMENT and make the
				// message public so the client sees what to fix. Over HTTP
				// that is a 400, except on a streaming request, which has
				// already answered 200 to open the stream; there the same
				// status and message arrive in the body instead.
				return "", status.PublicErrorf(status.ErrInvalidArgument, "dish must not be empty")
			}

			recipe, err := lookupRecipe(dish)
			if err != nil {
				// Wrapping does not reclassify: %w keeps the sentinel, the
				// status, and the public message reachable, so this still
				// answers 404 with the message lookupRecipe wrote. Say what
				// failed, not which flow it failed in: the trace already knows
				// the flow, and the reader wants the operation.
				return "", fmt.Errorf("could not look up the recipe: %w", err)
			}

			// Forwarding every chunk is all this flow does, so WithStreaming
			// says it in one option. improviseFlow below needs to act on a
			// failure mid-stream, which is what GenerateStream is for.
			text, err := genkit.GenerateText(ctx, g,
				ai.WithModel(model),
				ai.WithPrompt("Rewrite this recipe as three cheerful numbered steps: %s", recipe),
				ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					return sendChunk(ctx, chunk.Text())
				}),
			)
			if err != nil {
				return "", fmt.Errorf("could not rewrite the recipe: %w", err)
			}
			return text, nil
		})

	// improviseFlow branches on sentinels with errors.Is and recovers instead
	// of letting failures propagate.
	genkit.DefineStreamingFlow(g, "improviseFlow",
		func(ctx context.Context, input ImproviseRequest, sendChunk core.StreamCallback[string]) (string, error) {
			if strings.TrimSpace(input.Dish) == "" {
				return "", status.PublicErrorf(status.ErrInvalidArgument, "dish must not be empty")
			}
			prompt := ""
			recipe, lookupErr := lookupRecipe(input.Dish)
			switch {
			case errors.Is(lookupErr, ErrRecipeNotFound):
				// The failure this flow knows how to recover from.
				logger.Warn(ctx, "dish not in the cookbook, improvising", "dish", input.Dish)
				prompt = fmt.Sprintf("Invent a plausible three-step recipe for %s.", input.Dish)
			case lookupErr != nil:
				return "", lookupErr
			default:
				prompt = fmt.Sprintf("Rewrite this recipe as three cheerful numbered steps: %s", recipe)
			}

			started := false
			for val, err := range genkit.GenerateStream(ctx, g, ai.WithModel(model), ai.WithPrompt(prompt)) {
				switch {
				case errors.Is(err, status.ErrUnavailable), errors.Is(err, status.ErrResourceExhausted):
					// Transient trouble: degrade instead of surfacing a 5xx.
					// The middleware in basic-middleware automates this.
					logger.Warn(ctx, "model unavailable, serving the recipe unrewritten", "status", status.Of(err))
					// Recovery has a deadline that a non-streaming flow does
					// not: a chunk already sent cannot be taken back, so a
					// stand-in only works while the caller has read nothing.
					if recipe != "" && !started {
						sendChunk(ctx, recipe)
						return recipe, nil
					}
					return "", fmt.Errorf("could not improvise the recipe: %w", err)
				case err != nil:
					return "", fmt.Errorf("could not improvise the recipe: %w", err)
				case val.Done:
					return val.Response.Text(), nil
				}
				started = true
				sendChunk(ctx, val.Chunk.Text())
			}
			return "", status.Errorf(status.ErrInternal, "the stream ended without a final result")
		})

	// The error below is unclassified, so it reports INTERNAL and the client
	// gets a generic 500. The full text lands in the server log only, and
	// GENKIT_ENV=dev shows it unredacted during development.
	genkit.DefineFlow(g, "leakyFlow", func(ctx context.Context, _ any) (string, error) {
		return "", errors.New("connecting to db at 10.0.0.3 as admin: password rejected")
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
