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

// Command researcher serves the researcher agent of the basic-agents-remote
// sample on 127.0.0.1:8081. It is an ordinary Genkit app serving an agent over
// HTTP, which is all a remote agent needs; nothing here knows about the
// orchestrator.
//
// The sample starts it. To run it on its own, from the sample's directory:
//
//	go run ./researcher
package main

import (
	"context"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// The sample holds this process's stdin open, so end of input means the
	// sample is gone. Run from a terminal, stdin stays open.
	go func() {
		io.Copy(io.Discard, os.Stdin)
		cancel()
	}()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}), genkit.WithExperimental())

	// The session store is what lets a caller run this agent in the
	// background: a background task is a pending snapshot in this store, which
	// the caller follows through the getSnapshot, waitForSnapshot, and abort
	// routes. A deployed service would use a store all its replicas share.
	genkitx.DefineAgent(g, "researcher",
		aix.InlinePrompt{
			ai.WithModelName("googleai/gemini-flash-latest"),
			ai.WithSystem("You are a research assistant. You get one question at a time. " +
				"Answer it from what you know in at most five sentences, and say when you are unsure."),
		},
		aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
	)

	mux := http.NewServeMux()
	for _, route := range genkitx.AllAgentRoutes(g) {
		mux.HandleFunc(route.Pattern(), route.Handler())
	}
	if err := server.Start(ctx, "127.0.0.1:8081", mux); err != nil {
		log.Fatal(err)
	}
}
