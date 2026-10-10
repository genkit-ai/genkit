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

package main

import (
	"context"
	"io"
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

// serveResearcher runs the researcher service: one agent, served over HTTP on
// addr with the AllAgentRoutes layout, until ctx ends or its input closes.
//
// Nothing here knows about the orchestrator. The service is an ordinary
// Genkit app serving an agent, which is all a remote agent needs.
func serveResearcher(ctx context.Context, addr string) error {
	// The orchestrator holds this process's stdin open, so end of input means
	// the orchestrator is gone, however it ended.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		io.Copy(io.Discard, os.Stdin)
		cancel()
	}()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}), genkit.WithExperimental())

	// The session store is what lets the orchestrator run this agent in the
	// background: a background task is a pending snapshot in this store, and
	// the orchestrator follows it through the getSnapshot, waitForSnapshot,
	// and abort routes. The in-memory store suits a service that lives as long
	// as the demo; a deployed one would use a store all its replicas share.
	genkitx.DefineAgent(g, "researcher",
		aix.InlinePrompt{
			ai.WithModel(model),
			ai.WithSystem("You are a research assistant. You get one question at a time. " +
				"Answer it from what you know in at most five sentences, and say when you are unsure."),
		},
		aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
	)

	mux := http.NewServeMux()
	for _, route := range genkitx.AllAgentRoutes(g) {
		mux.HandleFunc(route.Pattern(), route.Handler())
	}
	return server.Start(ctx, addr, mux)
}
