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

// This sample demonstrates serving agents as plain HTTP endpoints.
//
// Agents are bidirectional streaming actions, but the standard action handler
// also runs them one turn per request: "data" carries the user message, and the
// optional "init" carries the session source that spans requests.
//
// Two agents show the two ways to hold session state:
//
//   - chat has a session store, so the server keeps the state. Each turn
//     persists a snapshot and the response carries sessionId and snapshotId;
//     resume with {"init": {"sessionId": ...}}. The store also brings the
//     companion actions, served under the agent's own path as getSnapshot and
//     abort.
//   - statelessChat has no store, so the client keeps the state. The response
//     carries the whole thing; send it back as {"init": {"state": ...}}.
//
// Failures come in two tiers. A failed turn still returns 200, reporting
// finishReason "failed" with a structured error and the last-good state, so a
// client can retry without losing the conversation. A rejected init (unknown
// session, or state sent to a store-backed agent) fails the request with a 4xx
// before any turn runs.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to call the agents from a browser and read a trace of
// every turn at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. Start a conversation, then continue it with the sessionId the
// response carried:
//
//	curl -X POST http://localhost:8080/agents/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "My name is Alex and I am planning a trip to Japan."}]}}}'
//
//	curl -X POST http://localhost:8080/agents/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "What is my name?"}]}}, "init": {"sessionId": "SESSION_ID"}}'
//
// Stream a turn's chunks and lifecycle events as server-sent events:
//
//	curl -N -X POST 'http://localhost:8080/agents/chat?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "Suggest three day trips from Tokyo."}]}}}'
//
// Or detach, which returns immediately with finishReason "detached" and a
// pending snapshotId while the turn keeps running. Poll getSnapshot until its
// status leaves "pending", or abort it:
//
//	curl -X POST http://localhost:8080/agents/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "Plan a two-week Japan itinerary."}]}, "detach": true}}'
//
//	curl -X POST http://localhost:8080/agents/chat/getSnapshot \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"snapshotId": "SNAPSHOT_ID"}}'
//
// Or stop the background turn instead, which finalizes it as "aborted":
//
//	curl -X POST http://localhost:8080/agents/chat/abort \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"snapshotId": "SNAPSHOT_ID"}}'
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

// model is shared by every agent below, so switching models or thinking levels
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
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}), genkit.WithExperimental())

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

	// AllAgentRoutes lays out a default HTTP surface for every registered
	// agent, following each one's capabilities, so the store-backed and
	// client-managed agents can be served side by side from one call:
	//
	//     POST /agents/chat                one turn per request
	//     POST /agents/chat/getSnapshot    read a snapshot by ID
	//     POST /agents/chat/abort          abort background work
	//     POST /agents/statelessChat       one turn per request
	//
	// route.Pattern() is its "METHOD /path" and route.Handler() builds the
	// genkit.Handler, so any router works the same way. For a subset use
	// genkitx.AgentRoutes(agent), and for flows genkitx.AllFlowRoutes(g).
	mux := http.NewServeMux()
	for _, route := range genkitx.AllAgentRoutes(g) {
		mux.HandleFunc(route.Pattern(), route.Handler())
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
