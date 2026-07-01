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

// This sample demonstrates Genkit's agent APIs by defining five agents in
// different styles and exposing all of them through a single CLI. Each agent
// lives in its own file so the styles can be compared side by side:
//
//   - "pirate" (pirate.go) uses DefineAgent + aix.InlinePrompt. The prompt
//     is declared inline next to the agent.
//   - "chef" (chef.go) uses DefinePromptAgent. With no source option it
//     defaults to the prompt registered under the agent's name, loaded from
//     ./prompts/chef.prompt.
//   - "coder" (coder.go) uses DefineCustomAgent. The per-turn loop (model
//     selection, history management, streaming) is wired by hand.
//   - "banker" (banker.go) uses DefinePromptAgent (prompt loaded from
//     ./prompts/banker.prompt) with an interruptible tool. It pauses
//     mid-turn to ask the user for approval before moving money, then
//     resumes the tool with their answer. This exercises the tool
//     interrupt / resume flow through the same CLI.
//   - "orchestrator" (orchestrator.go) uses the experimental Agents
//     middleware to delegate to specialized sub-agents (researcher,
//     engineer) through per-agent tools, merging their artifacts into its
//     own session.
//
// The interactive CLI lives in cli.go: it lists the agents, streams each
// turn, renders tool calls inline as the agent makes them, and routes tool
// interrupts to the right handler.
//
// The first four agents persist their conversation state to a per-agent
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
//
// Tip: pick "banker" and try "send $200 to alice" (more than the $150
// balance) or "send $120 to bob" (a large transfer) to see the tool
// interrupt flow: the turn pauses, the CLI asks you to approve or adjust,
// and the tool resumes with your answer.
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

// flashModel is the default model shared by every agent in this sample:
// gemini-flash-latest with thinking disabled for snappy, low-cost turns. The
// pirate, coder, and orchestrator agents reference it directly; the chef and
// banker agents set the same model in their .prompt frontmatter.
var flashModel = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](0)},
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
		{agent: defineInlineAgent(g)},
		{agent: definePromptAgent(g)},
		{agent: defineCustomAgent(g)},
		{agent: defineBankerAgent(g), onInterrupt: handleTransferInterrupt},
		{agent: defineOrchestratorAgent(g)},
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
