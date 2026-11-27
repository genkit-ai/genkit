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

// This sample demonstrates a sub-agent that runs in another process. An
// orchestrator delegates research to a researcher agent that a second Genkit
// app (./researcher) serves over HTTP, and the delegation works the way it does
// for an agent in the same process, background tasks included.
//
// The researcher is an ordinary Genkit app serving an agent. This program
// registers it with genkitx.DefineRemoteAgent and gives it to the agents
// middleware by name, exactly as it would a local agent.
//
// Run it from this directory (the Google AI plugin reads GEMINI_API_KEY). It
// starts the researcher on 127.0.0.1:8081 and serves the orchestrator on
// 127.0.0.1:8080:
//
//	go run .
//
// Chat with the orchestrator in the Dev UI, and read the trace of every
// delegation at http://localhost:4000/traces:
//
//	genkit start -- go run .
//
// Or over HTTP, streaming the turn:
//
//	curl -N -X POST 'http://localhost:8080/agents/orchestrator?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"message": {"role": "user", "content": [{"text": "Compare the history of Tokyo and Kyoto as capitals of Japan."}]}}}'
//
// The orchestrator starts one researcher task per part of the question in the
// background, waits for them, and combines the answers.
package main

import (
	"context"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	middlewarex "github.com/firebase/genkit/go/plugins/middleware/exp"
	"github.com/firebase/genkit/go/plugins/server"
)

func main() {
	ctx := context.Background()

	researcherStdin, err := startResearcher()
	if err != nil {
		log.Fatalf("starting the researcher: %v", err)
	}
	defer researcherStdin.Close()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}), genkit.WithExperimental())

	// The researcher is now an agent of this app. The URL is its turn route;
	// the snapshot routes that background tasks use sit next to it. The
	// metadata declares what its server supports: it keeps the session and
	// can stop background work. The middleware gates background tasks on it.
	researcher := genkitx.DefineRemoteAgent(g, "researcher", "http://127.0.0.1:8081/agents/researcher",
		aix.WithDescription[any]("Researches one question and answers in a few sentences."),
		aix.WithAgentMetadata(&aix.AgentMetadata{
			StateManagement: aix.AgentStateManagementServer,
			Abortable:       true,
		}),
	)

	genkitx.DefineAgent(g, "orchestrator",
		aix.InlinePrompt{
			ai.WithModelName("googleai/gemini-flash-latest"),
			ai.WithSystem("You are a planning assistant. When a question has several parts, " +
				"give each part to the researcher as its own task, all of them in the background " +
				"in one turn, then wait for every task and combine the answers into one reply. " +
				"Answer a simple question yourself."),
			// Nothing here says that the researcher is remote.
			ai.WithUse(&middlewarex.Agents{
				Agents: []aix.AgentRef{researcher.Ref()},
				Async:  true,
			}),
			// Launch, wait, and answer are three tool rounds at least.
			ai.WithMaxTurns(10),
		},
		aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
	)

	// AllAgentRoutes serves the orchestrator and skips the researcher, which
	// is another service's agent.
	mux := http.NewServeMux()
	for _, route := range genkitx.AllAgentRoutes(g) {
		mux.HandleFunc(route.Pattern(), route.Handler())
	}
	if err := server.Start(ctx, "127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}

// startResearcher runs ./researcher as a second process and returns its
// stdin. The researcher exits when its stdin closes, so keep the returned
// writer open for as long as this process runs: when this process ends, even
// by a kill, the researcher ends with it.
//
// This is demo plumbing. A deployed orchestrator only knows the researcher's
// URL.
func startResearcher() (io.WriteCloser, error) {
	cmd := exec.Command("go", "run", "./researcher")
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	// Production mode keeps the researcher out of the Dev UI, so the Dev UI
	// shows this app alone.
	cmd.Env = append(os.Environ(), "GENKIT_ENV=prod")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	return stdin, cmd.Start()
}
