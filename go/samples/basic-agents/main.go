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

// This sample demonstrates Genkit's agent APIs by defining four agents and
// exposing all of them through a single CLI. Each agent lives in its own file:
//
//   - "pirate" (pirate.go) uses DefineAgent + aix.InlinePrompt. The prompt is
//     declared inline next to the agent.
//   - "chef" (chef.go) uses DefinePromptAgent. With no source option it defaults
//     to the prompt registered under the agent's name, loaded from
//     ./prompts/chef.prompt.
//   - "coder" (coder.go) uses DefineCustomAgent. The per-turn loop (model
//     selection, history management, streaming) is wired by hand.
//   - "orchestrator" (orchestrator.go) uses the experimental Agents middleware
//     to delegate to specialized sub-agents.
//
// The first three persist their conversation state to a per-agent
// FileSessionStore under ./.genkit/snapshots/<agent>/; the orchestrator does
// too, while its sub-agents run statelessly per delegation.
//
// To run:
//
//	go run .
//
// The CLI prints a numbered list of agents. Pick one, choose to resume
// from the last snapshot or start fresh, and chat. Inside a chat:
//
//	(text)             send a message and stream the reply
//	/detach (text...)  send the text (optional) as the final input, then
//	                   detach. The server keeps processing in the
//	                   background; you get a pending snapshot ID and
//	                   return to the agent list. Re-pick the agent later
//	                   to wait for the snapshot to finalize and resume
//	                   from the cumulative final state.
//	/back              return to the agent list (snapshot is still
//	                   written by the agent's normal turn-end hook)
//	/quit              exit the program
//
// Tip: try "/detach write me a long pirate story" to see the detach loop
// end-to-end. After the CLI returns to the agent list, pick "pirate"
// again; if the snapshot is still pending, you'll get a three-way menu
// (wait, start new, back). Picking wait blocks on the in-process
// status subscription and resumes from the cumulative final state.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	genkit.DefineSchemaFor[ChatPromptInput](g) // input schema for ./prompts/chef.prompt (see chef.go)

	// Each define function (in its own file) registers an agent and returns
	// it. The CLI drives all of them through the same surface: a.Name() and
	// a.Desc().Description for the list view, a.Connect(...) to chat, and
	// a.Store() for snapshot reads. Nothing the CLI does is tied to a concrete
	// store type, so swapping in a different SessionStore would not touch a
	// line of it.
	agents := []*aix.Agent[any]{
		defineInlineAgent(g),
		definePromptAgent(g),
		defineCustomAgent(g),
		defineOrchestratorAgent(g),
	}

	if err := runCLI(ctx, agents); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// flashModel is the model shared by the agents in this sample:
// gemini-flash-latest with thinking disabled for snappy, low-cost turns.
// genkit copies request config per call rather than mutating it, so one shared
// reference is safe across all the agents.
var flashModel = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](0)},
})

// mustStore creates a FileSessionStore rooted at the per-agent dir under
// ./.genkit/snapshots/, or exits the process on failure. Used during
// agent setup where there's nowhere sensible to return an error.
//
// A dir per agent keeps each agent's snapshots on disk separately, which
// is tidy for browsing but not required: resumes are resolved by session
// ID (see SnapshotReader.GetLatestSnapshot), so one shared store would
// work the same.
func mustStore(agentName string) *localstore.FileSessionStore[any] {
	store, err := localstore.NewFileSessionStore[any]("./.genkit/snapshots/" + agentName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating store for %q: %v\n", agentName, err)
		os.Exit(1)
	}
	return store
}
