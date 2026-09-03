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
//
// SPDX-License-Identifier: Apache-2.0

// This sample serves an A2UI-enabled agent over HTTP, compatible with the
// browser frontend in js/testapps/a2ui/web. The whole A2UI integration is the
// a2ui middleware in the agent's inline prompt (`ai.WithUse(&a2uix.Surfaces{})`):
// it injects the catalog's capabilities into the system prompt and rewrites the
// model's a2ui fenced blocks into a2ui data parts that the client renderer
// consumes.
//
// The agent is served at POST /api/uiAgent (plus the /getSnapshot and /abort
// companions the `remoteAgent` client expects), the exact endpoint the web UI
// talks to via `remoteAgent({ url: '/api/uiAgent' })`.
//
// To run the whole thing:
//
//  1. Start this backend (needs a Gemini API key in the environment):
//
//     go run .
//
//  2. Build and preview the web UI (in js/testapps/a2ui/web):
//
//     pnpm install
//     pnpm build
//     pnpm preview
//
//     Then open the printed URL (http://localhost:4173). The Vite preview
//     server proxies /api to this backend on :8080.
//
// You can also drive the agent directly with curl:
//
//	curl -N -X POST 'http://localhost:8080/api/uiAgent?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "What is the weather in Tokyo?"}]}}}'
package main

import (
	"context"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	a2uix "github.com/firebase/genkit/go/plugins/a2ui/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/middleware"
	"github.com/firebase/genkit/go/plugins/server"
)

// weatherInput is the getWeather tool's input.
type weatherInput struct {
	City string `json:"city" jsonschema:"description=The city to get the weather for."`
}

// weatherOutput is the getWeather tool's (fake) result.
type weatherOutput struct {
	City      string  `json:"city"`
	TempC     float64 `json:"tempC"`
	Condition string  `json:"condition"`
	Humidity  int     `json:"humidity"`
}

func main() {
	ctx := context.Background()

	// Experimental mode is required for agents. Registering the A2UI plugin is
	// optional (it surfaces the middleware in the Dev UI); the middleware works
	// via ai.WithUse regardless.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}, &a2uix.A2UI{}),
		genkit.WithExperimental(),
	)

	// Register the bundled basic catalog in the registry so it shows up in the
	// Dev UI (GET /api/values?type=a2ui-catalog) alongside any custom catalogs.
	// The middleware falls back to the basic catalog even without this, so it is
	// optional; it is here to demonstrate registry-backed catalogs.
	if err := a2uix.RegisterBasicCatalog(g); err != nil {
		log.Fatalf("registering basic catalog: %v", err)
	}

	// A demo tool the model can call to fetch (fake) weather data.
	getWeather := genkit.DefineTool(g, "getWeather", "Gets the current weather for a given city.",
		func(_ *ai.ToolContext, in weatherInput) (weatherOutput, error) {
			// Deterministic pseudo-random values so the demo is stable per-city.
			var seed int
			for _, c := range in.City {
				seed += int(c)
			}
			conditions := []string{"Sunny", "Partly cloudy", "Rainy", "Windy", "Foggy"}
			return weatherOutput{
				City:      in.City,
				TempC:     float64(10 + seed%20),
				Condition: conditions[seed%len(conditions)],
				Humidity:  40 + seed%50,
			}, nil
		},
	)

	// The A2UI-enabled agent. The whole integration is a2uix.Surfaces in WithUse.
	// An in-memory session store makes state server-managed, so the browser
	// only needs to pass a session id (remoteAgent handles that for it).
	uiAgent := genkitx.DefineAgent(g, "uiAgent",
		aix.InlinePrompt{
			ai.WithModelName("googleai/gemini-flash-latest"),
			ai.WithSystem(`You are a helpful assistant that can render rich UI.
Prefer rendering an A2UI surface whenever a result is clearer shown than told —
for example weather, comparisons, lists, forms, or anything interactive. Keep any
prose brief; put the substance in the UI. When asked about weather, call the
getWeather tool, then render a nice Card/Column summarizing it (temperature,
condition, humidity). Feel free to add a Button (e.g. "Refresh") when useful.`),
			ai.WithTools(getWeather),
			// Retry transient model failures (UNAVAILABLE, RESOURCE_EXHAUSTED,
			// etc.) with exponential backoff before giving up.
			ai.WithUse(&middleware.Retry{MaxRetries: 5}),
			ai.WithUse(&a2uix.Surfaces{}), // defaults to the bundled basic catalog
		},
		aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
	)

	// Serve the agent at /api/uiAgent, the endpoint the web UI's
	// remoteAgent({ url: '/api/uiAgent' }) talks to, plus the companion
	// endpoints it derives from that base (/getSnapshot and /abort). Each
	// handler is wrapped with permissive CORS so the browser can reach it
	// (a no-op behind the Vite proxy, but handy for `pnpm dev`). The routes
	// also register bare paths (without the POST method prefix) so the CORS
	// preflight OPTIONS requests are matched.
	mux := http.NewServeMux()
	handle := func(path string, a api.Action) {
		mux.Handle(path, withCORS(genkit.Handler(a)))
	}
	handle("/api/uiAgent", uiAgent)
	handle("/api/uiAgent/getSnapshot", uiAgent.GetSnapshotAction())
	handle("/api/uiAgent/abort", uiAgent.AbortAction())

	log.Print("A2UI agent server listening on http://localhost:8080 (POST /api/uiAgent)")
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}

// withCORS wraps h with permissive CORS headers so a browser served from a
// different origin (e.g. the Vite dev server on :5173) can call the agent
// directly. When using `pnpm preview`, requests are same-origin via the Vite
// proxy and CORS is a no-op, but this keeps `pnpm dev` working too.
func withCORS(h http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Accept, x-genkit-stream-id")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h.ServeHTTP(w, r)
	}
}
