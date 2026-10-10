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
// orchestrator delegates research questions to a researcher agent that a second
// Genkit app serves over HTTP, and the delegation works the way it does for an
// agent in the same process, background tasks included.
//
// There are two processes, and one command starts both:
//
//   - The researcher service (researcher.go) defines the researcher agent with
//     a session store and serves it with genkitx.AllAgentRoutes. Any Genkit
//     app that serves an agent this way can be a remote agent.
//   - The orchestrator (orchestrator.go) registers the researcher with
//     genkitx.DefineRemoteAgent and delegates to it through the agents
//     middleware, by name, exactly as it would to a local agent.
//
// spawn.go starts the researcher service as a copy of this program run with
// -serve, prefixes its output with [researcher], and stops it when the
// orchestrator exits.
//
// Run it (the Google AI plugin reads GEMINI_API_KEY):
//
//	go run .
//
// Then ask a question with more than one part, for example:
//
//	Compare the history of Tokyo and Kyoto as capitals of Japan.
//
// The orchestrator starts one researcher task per part in the background, so
// the tool call lines show the delegations, the task IDs they return, and the
// wait that collects them. The [researcher] lines are the other process
// answering those requests.
//
// To run the researcher service on its own, for example to call it with curl
// as in the basic-agents-server sample, start it with an address. It stops at
// end of input (Ctrl-D):
//
//	go run . -serve 127.0.0.1:8081
//
// With the Dev UI (genkit start -- go run .), the orchestrator is the runtime
// you see. The researcher service runs in production mode, so its traces stay
// out of the Dev UI and an internal error in it reaches the orchestrator as
// its status with a generic message, as it would from a deployed service.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

// model is shared by both agents.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
})

func main() {
	serve := flag.String("serve", "", "serve the researcher agent on this address instead of running the orchestrator")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var err error
	if *serve != "" {
		err = serveResearcher(ctx, *serve)
	} else {
		err = runOrchestrator(ctx)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
