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

// This sample demonstrates the agent APIs by defining six agents in different
// styles, one per file, behind a single CLI:
//
//   - pirate (pirate.go): DefineAgent with the prompt declared inline.
//   - chef (chef.go): DefinePromptAgent, which defaults to the prompt
//     registered under the agent's name, ./prompts/chef.prompt.
//   - coder (coder.go): DefineCustomAgent, with the per-turn loop wired by
//     hand.
//   - banker (banker.go): a prompt agent with an interruptible tool, so a turn
//     pauses for approval before moving money and resumes with the answer.
//   - barista (barista.go): an Agent[BaristaOrder] whose system instruction is
//     a function. A tool writes the order into session state and the prompt
//     reads it back, so each turn knows what earlier ones established.
//   - orchestrator (orchestrator.go): delegates to sub-agents through
//     per-agent tools, merging their artifacts into its own session.
//
// cli.go holds the CLI: it lists the agents, streams each turn, renders tool
// calls as they happen, and routes interrupts. Conversation state persists to a
// per-agent FileSessionStore under ./.genkit/snapshots/<agent>/, except for the
// orchestrator's sub-agents, which run statelessly per delegation.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, which keeps the CLI on the terminal and adds a trace of
// every turn at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Pick an agent, resume from its last snapshot or start fresh, then chat.
// Inside a chat:
//
//	(text)             send a message and stream the reply
//	/detach (text...)  send the text, then leave it running in the background
//	                   and return to the agent list. Re-pick the agent to wait
//	                   for the snapshot, or to stop it and resume from the
//	                   turns it finished.
//	/back              return to the agent list
//	/quit              exit
//
// Three worth trying: "/detach write me a long pirate story", then re-pick
// pirate to wait on it; banker with "send $200 to alice" (over the $150
// balance) or "send $120 to bob" (large enough to need approval); and barista,
// where ordering a drink and later asking "what have I ordered?" is answered
// from session state rather than from the transcript.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

// model is the default model shared by every agent in this sample. The pirate,
// coder, and orchestrator agents reference it directly; the chef and banker
// agents set the same model in their .prompt frontmatter.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium},
})

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}), genkit.WithExperimental())

	// Each define function registers an agent and returns it, paired with
	// the optional hooks the CLI needs to drive it (see agentEntry). The
	// CLI drives all of them through the same surface: a.Name() and
	// a.Desc().Description for the list view, a.Connect(...) to chat,
	// and a.Store() for snapshot reads. Nothing the CLI does is tied to a
	// concrete store type, so swapping in a different SessionStore would
	// not touch a line of it.
	//
	// The banker is the only agent with an interruptible tool, so it is the
	// only one that supplies an onInterrupt handler; the others leave it
	// nil and the CLI streams them exactly as before.
	agents := []agentEntry{
		newEntry(defineInlineAgent(g), nil),
		newEntry(definePromptAgent(g), nil),
		newEntry(defineCustomAgent(g), nil),
		newEntry(defineBankerAgent(g), handleTransferInterrupt),
		newEntry(defineBaristaAgent(g), nil),
		newEntry(defineOrchestratorAgent(g), nil),
	}

	if err := runCLI(ctx, agents); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// mustStore creates a FileSessionStore rooted at the per-agent dir under
// ./.genkit/snapshots/, or exits the process on failure. Used during
// agent setup where there's nowhere sensible to return an error.
//
// The store is typed by the agent's session state, so each agent gets a store
// that reads and writes its own shape: mustStore[any] for the agents that keep
// no custom state, mustStore[BaristaOrder] for the one that does.
//
// A dir per agent keeps each agent's snapshots on disk separately, which
// is tidy for browsing but not required: resumes are resolved by session
// ID (see SnapshotReader.GetLatestSnapshot), so one shared store would
// work the same.
func mustStore[State any](agentName string) *localstore.FileSessionStore[State] {
	store, err := localstore.NewFileSessionStore[State]("./.genkit/snapshots/" + agentName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating store for %q: %v\n", agentName, err)
		os.Exit(1)
	}
	return store
}
