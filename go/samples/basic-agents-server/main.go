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

// This sample demonstrates serving Genkit agents as plain HTTP endpoints.
//
// Agents are bidirectional streaming actions, but the standard action
// handler also runs them one turn per request: "data" carries the turn's
// input (the user message), and the optional "init" field carries the
// session source that lets a conversation span requests.
//
// Two agents show the two session-state modes:
//
//   - "chat" configures a session store (server-managed state). Each turn
//     persists a snapshot; the response carries sessionId and snapshotId,
//     and a later request resumes the conversation by sending
//     {"init": {"sessionId": ...}} (or {"snapshotId": ...} to resume from
//     a specific point in history). The store also gives the agent
//     snapshot companion actions, served here at
//     /agents/chat/getSnapshot and /agents/chat/abort.
//   - "statelessChat" has no store (client-managed state). The response
//     carries the full conversation state; the client sends it back
//     verbatim as {"init": {"state": ...}} on the next turn. The server
//     keeps nothing between requests.
//
// To run:
//
//	go run .
//
// Start a conversation (no init starts a fresh session):
//
//	curl -X POST http://localhost:8080/agents/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "My name is Alex and I am planning a trip to Japan."}]}}}'
//
// Continue it, using the sessionId from the response:
//
//	curl -X POST http://localhost:8080/agents/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "What is my name?"}]}}, "init": {"sessionId": "SESSION_ID"}}'
//
// Stream a turn's model chunks and lifecycle events as server-sent events:
//
//	curl -N -X POST 'http://localhost:8080/agents/chat?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "Suggest three day trips from Tokyo."}]}}}'
//
// For statelessChat, resume by round-tripping the returned state instead:
//
//	curl -X POST http://localhost:8080/agents/statelessChat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "What is my name?"}]}}, "init": {"state": STATE_FROM_PREVIOUS_RESPONSE}}'
//
// Server-managed state also unlocks background continuation. Send a turn
// with "detach": true and the response comes back immediately with
// finishReason "detached" and a pending snapshotId, while the turn keeps
// running on the server:
//
//	curl -X POST http://localhost:8080/agents/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "Plan a two-week Japan itinerary."}]}, "detach": true}}'
//
// The companion endpoints follow the conversation from there. Each is a
// POST that carries the snapshotId in the {"data": ...} body and returns
// the {"result": ...} envelope, the same convention as the turn route.
// Poll the pending snapshot until its status leaves "pending" (the final
// state carries the result), using the snapshotId from the detach
// response:
//
//	curl -X POST http://localhost:8080/agents/chat/getSnapshot \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"snapshotId": "SNAPSHOT_ID"}}'
//
// Or abort the background work instead; an aborted snapshot finalizes
// with status "aborted":
//
//	curl -X POST http://localhost:8080/agents/chat/abort \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"snapshotId": "SNAPSHOT_ID"}}'
//
// Failures come in two tiers. A failed turn (e.g. the model call errors)
// still returns HTTP 200: the result reports finishReason "failed", a
// structured error ({status, message, details}), and the last-good
// conversation state (or a recovery snapshot ID), so the client can retry
// the turn without losing the conversation. A rejected init (an unknown
// session or snapshot ID, state sent to a store-backed agent) fails the
// request itself with a 4xx error before any turn runs.
package main

import (
	"context"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the
	// recommended practice.
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	model := googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
		ThinkingConfig: &genai.ThinkingConfig{
			ThinkingBudget: genai.Ptr[int32](0),
		},
	})

	// "chat" persists every conversation to a snapshot store, so a client
	// only needs to hold on to the sessionId between requests. Snapshots
	// land under ./.genkit/snapshots/chat/.
	store, err := localstore.NewFileSessionStore[any]("./.genkit/snapshots/chat")
	if err != nil {
		log.Fatalf("creating session store: %v", err)
	}
	genkitx.DefineAgent(g, "chat",
		aix.InlinePrompt{
			ai.WithModel(model),
			ai.WithSystem("You are a helpful travel assistant. Keep responses to a couple of sentences."),
		},
		aix.WithSessionStore(store),
	)

	// "statelessChat" keeps no state on the server: each response carries
	// the full conversation state and the client round-trips it on the next
	// request. This suits deployments where the server must stay stateless.
	genkitx.DefineAgent[any](g, "statelessChat",
		aix.InlinePrompt{
			ai.WithModel(model),
			ai.WithSystem("You are a helpful travel assistant. Keep responses to a couple of sentences."),
		},
	)

	// genkitx.AllAgentRoutes lays out a default HTTP surface for every
	// registered agent; range over the routes and wire each onto the mux.
	// The layout follows each agent's capabilities, so server-managed and
	// client-managed agents can be deployed side by side from one call:
	//
	//   "chat" (store-backed):
	//     POST /agents/chat                one turn per request
	//     POST /agents/chat/getSnapshot    read a snapshot by ID
	//     POST /agents/chat/abort          abort background work
	//   "statelessChat" (client-managed):
	//     POST /agents/statelessChat       one turn per request
	//
	// Every route is a POST taking the standard {"data": ...} envelope and
	// returning {"result": ...}; the companions read the snapshotId from
	// that body. route.Pattern() is its "METHOD /path" and route.Handler()
	// builds the genkit.Handler; pass HandlerOptions (e.g. context providers
	// for auth) to Handler() to apply them per route. Any router works the
	// same way (Gin, Chi, Echo): read Pattern and serve Handler.
	//
	// To serve specific agents instead of all of them, use
	// genkitx.AgentRoutes(agent); to expose flows, genkitx.AllFlowRoutes(g).
	// Mix them by concatenating the route slices. The genkitx (genkit/exp)
	// package holds these helpers while the routing layer is experimental.
	mux := http.NewServeMux()
	for _, route := range genkitx.AllAgentRoutes(g) {
		mux.HandleFunc(route.Pattern(), route.Handler())
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
