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

// This sample demonstrates how errors are classified, propagated, and
// handled in Genkit Go using the core/status package.
//
// The pattern, in one paragraph: classify a failure once, where its meaning
// is known, with status.Errorf and a sentinel (use status.PublicErrorf when
// the message is safe to show a client); add context up the stack with plain
// fmt.Errorf and %w, which preserves the classification; and branch on
// failures with errors.Is against a sentinel instead of matching message
// text. At the flow HTTP boundary the status picks the response code, and
// only public messages reach the client; everything else is redacted to a
// generic string (and still fully logged on the server).
//
// Three flows exercise the pieces:
//
//   - cookbookFlow produces classified errors: a public INVALID_ARGUMENT for
//     a bad request, and a custom NOT_FOUND subtype for an unknown dish.
//   - improviseFlow consumes them: it branches on the custom sentinel to
//     recover (improvise a recipe instead of failing), falls back to the
//     default model when the requested model doesn't exist, and degrades
//     gracefully when the model is overloaded.
//   - leakyFlow fails with an unclassified error so you can see the boundary
//     redact it.
//
// To run:
//
//	go run .
//
// In another terminal:
//
//	# Public INVALID_ARGUMENT: 400, the message reaches the client.
//	curl -X POST http://localhost:8080/cookbookFlow \
//	  -H "Content-Type: application/json" -d '{"data": ""}'
//
//	# Custom NOT_FOUND subtype: 404 with the public message.
//	curl -X POST http://localhost:8080/cookbookFlow \
//	  -H "Content-Type: application/json" -d '{"data": "lasagna"}'
//
//	# A dish in the cookbook: the model rewrites the stored recipe.
//	curl -X POST http://localhost:8080/cookbookFlow \
//	  -H "Content-Type: application/json" -d '{"data": "pancakes"}'
//
//	# Recovery: not in the cookbook, so the flow improvises instead of 404ing.
//	curl -X POST http://localhost:8080/improviseFlow \
//	  -H "Content-Type: application/json" -d '{"data": {"dish": "lasagna"}}'
//
//	# Misconfigured model: the flow catches the NOT_FOUND and retries with
//	# the default model instead of failing.
//	curl -X POST http://localhost:8080/improviseFlow \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"dish": "pancakes", "model": "googleai/not-a-real-model"}}'
//
//	# Unclassified error: 500, and the client sees only a generic message.
//	# The real text (with its fake credentials) never leaves the process.
//	curl -X POST http://localhost:8080/leakyFlow \
//	  -H "Content-Type: application/json" -d '{"data": null}'
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
)

const defaultModel = "googleai/gemini-flash-latest"

// ErrRecipeNotFound classifies lookups for dishes the cookbook doesn't have.
// Deriving a subtype from a base sentinel keeps the parent's status (and so
// its HTTP code, 404 here), and errors.Is matches it at either granularity:
// errors.Is(err, ErrRecipeNotFound) for this exact failure, or
// errors.Is(err, status.ErrNotFound) for any not-found.
var ErrRecipeNotFound = status.ErrNotFound.Subtype("recipe not found")

// cookbook is the sample's tiny data store.
var cookbook = map[string]string{
	"pancakes":  "Whisk 1 cup flour, 1 tbsp sugar, 1 tsp baking powder, 1 egg, and 3/4 cup milk. Fry ladlefuls in butter until golden on both sides.",
	"shakshuka": "Simmer a can of crushed tomatoes with sauteed onion, garlic, and paprika. Crack in 4 eggs, cover, and cook until just set.",
}

// lookupRecipe classifies the not-found case once, at the source. The
// message is built with PublicErrorf because it only reflects what the
// caller sent, so it is safe (and useful) to return to them.
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

	// cookbookFlow produces classified errors and lets them propagate. Note
	// what the flow does NOT do: it never inspects err.Error() text, and it
	// wraps with fmt.Errorf (not a new status) when adding context, so the
	// classification chosen at the source survives to the HTTP boundary.
	genkit.DefineFlow(g, "cookbookFlow", func(ctx context.Context, dish string) (string, error) {
		if strings.TrimSpace(dish) == "" {
			// A bad request, described in terms of the caller's input:
			// classify it INVALID_ARGUMENT and mark the message public so
			// the client sees what to fix. Over HTTP this becomes a 400.
			return "", status.PublicErrorf(status.ErrInvalidArgument, "dish must not be empty")
		}

		recipe, err := lookupRecipe(dish)
		if err != nil {
			// Add context without reclassifying: %w keeps the sentinel, the
			// status, and the public message reachable, so this still
			// surfaces as a 404 rather than turning into a 500.
			return "", fmt.Errorf("cookbookFlow: %w", err)
		}

		return genkit.GenerateText(ctx, g,
			ai.WithModelName(defaultModel),
			ai.WithPrompt("Rewrite this recipe as three cheerful numbered steps: %s", recipe),
		)
	})

	// improviseFlow consumes classified errors: instead of letting failures
	// propagate, it branches on sentinels with errors.Is and recovers.
	type ImproviseInput struct {
		Dish string `json:"dish"`
		// Model optionally overrides the model name, so you can point the
		// flow at a model that doesn't exist and watch the fallback branch.
		Model string `json:"model,omitempty"`
	}
	genkit.DefineFlow(g, "improviseFlow", func(ctx context.Context, input ImproviseInput) (string, error) {
		if strings.TrimSpace(input.Dish) == "" {
			return "", status.PublicErrorf(status.ErrInvalidArgument, "dish must not be empty")
		}
		model := input.Model
		if model == "" {
			model = defaultModel
		}

		prompt := ""
		recipe, err := lookupRecipe(input.Dish)
		switch {
		case errors.Is(err, ErrRecipeNotFound):
			// The exact failure this flow knows how to recover from:
			// improvise a recipe rather than failing the request.
			log.Printf("improviseFlow: %q not in the cookbook, improvising", input.Dish)
			prompt = fmt.Sprintf("Invent a plausible three-step recipe for %s.", input.Dish)
		case err != nil:
			// Anything else is unexpected here: add context and propagate.
			return "", fmt.Errorf("improviseFlow: %w", err)
		default:
			prompt = fmt.Sprintf("Rewrite this recipe as three cheerful numbered steps: %s", recipe)
		}

		text, err := genkit.GenerateText(ctx, g,
			ai.WithModelName(model),
			ai.WithPrompt("%s", prompt),
		)
		switch {
		case errors.Is(err, status.ErrNotFound):
			// A misconfigured model name is recoverable: fall back to the
			// default model. Matching the base sentinel catches both ways
			// the miss can surface: ai.ErrModelNotFound (a subtype of
			// ErrNotFound) when the name resolves to no registered model,
			// and the provider's own NOT_FOUND when the API rejects a name
			// it doesn't recognize. status.Of extracts the status for
			// logging.
			log.Printf("improviseFlow: model %q not found (status %s), falling back to %s", model, status.Of(err), defaultModel)
			text, err = genkit.GenerateText(ctx, g,
				ai.WithModelName(defaultModel),
				ai.WithPrompt("%s", prompt),
			)
			if err != nil {
				return "", fmt.Errorf("improviseFlow: fallback model: %w", err)
			}
		case errors.Is(err, status.ErrUnavailable), errors.Is(err, status.ErrResourceExhausted):
			// Transient provider trouble: degrade gracefully instead of
			// surfacing a 5xx. (The retry and fallback middleware in
			// samples/basic-middleware automate this pattern.)
			log.Printf("improviseFlow: model temporarily unavailable (%s), serving the plain recipe", status.Of(err))
			if recipe != "" {
				return recipe, nil
			}
			return "", fmt.Errorf("improviseFlow: %w", err)
		case err != nil:
			return "", fmt.Errorf("improviseFlow: %w", err)
		}
		return text, nil
	})

	// leakyFlow shows the boundary protecting you: the error below is
	// unclassified, so status.Of reports INTERNAL and the client gets a 500
	// with a generic message. The full text lands in the server log only.
	// (Run with GENKIT_ENV=dev to see it unredacted during development.)
	genkit.DefineFlow(g, "leakyFlow", func(ctx context.Context, _ any) (string, error) {
		return "", errors.New("connecting to db at 10.0.0.3 as admin: password rejected")
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
