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
//     a specific point in history).
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
//	curl -X POST http://localhost:8080/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "My name is Alex and I am planning a trip to Japan."}]}}}'
//
// Continue it, using the sessionId from the response:
//
//	curl -X POST http://localhost:8080/chat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "What is my name?"}]}}, "init": {"sessionId": "SESSION_ID"}}'
//
// Stream a turn's model chunks and lifecycle events as server-sent events:
//
//	curl -N -X POST 'http://localhost:8080/chat?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "Suggest three day trips from Tokyo."}]}}}'
//
// For statelessChat, resume by round-tripping the returned state instead:
//
//	curl -X POST http://localhost:8080/statelessChat \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "What is my name?"}]}}, "init": {"state": STATE_FROM_PREVIOUS_RESPONSE}}'
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
	genkit.DefineAgent(g, "chat",
		aix.FromInline(
			ai.WithModel(model),
			ai.WithSystem("You are a helpful travel assistant. Keep responses to a couple of sentences."),
		),
		aix.WithSessionStore(store),
	)

	// "statelessChat" keeps no state on the server: each response carries
	// the full conversation state and the client round-trips it on the next
	// request. This suits deployments where the server must stay stateless.
	genkit.DefineAgent[any](g, "statelessChat",
		aix.FromInline(
			ai.WithModel(model),
			ai.WithSystem("You are a helpful travel assistant. Keep responses to a couple of sentences."),
		),
	)

	// Agents register under their own "agent" action type; ListAgents
	// surfaces them and the standard action handler serves them one turn
	// per request.
	mux := http.NewServeMux()
	for _, a := range genkit.ListAgents(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
