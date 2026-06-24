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

// This sample demonstrates Genkit's agent APIs by defining three agents in
// three different styles and exposing all of them through a single CLI:
//
//   - "pirate" uses DefineAgent + aix.InlinePrompt. The prompt is declared
//     inline next to the agent.
//   - "chef" uses DefinePromptAgent. With no source option it defaults to the
//     prompt registered under the agent's name, loaded from
//     ./prompts/chef.prompt.
//   - "coder" uses DefineCustomAgent. The per-turn loop (model selection,
//     history management, streaming) is wired by hand.
//
// All three agents persist their conversation state to a per-agent
// FileSessionStore under ./.genkit/snapshots/<agent>/.
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

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

// ChatPromptInput is the input schema referenced by ./prompts/chef.prompt.
// Registering it via DefineSchemaFor lets the .prompt file refer to it by
// name in its YAML frontmatter.
type ChatPromptInput struct {
	Personality string `json:"personality"`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	genkit.DefineSchemaFor[ChatPromptInput](g)

	// Each define function registers an agent and returns it. The CLI
	// drives all three through the same surface: a.Name() and
	// a.Desc().Description for the list view, a.Connect(...) to chat,
	// and a.Store() for snapshot reads. Nothing the CLI does is tied to a
	// concrete store type, so swapping in a different SessionStore would
	// not touch a line of it.
	agents := []*aix.Agent[any]{
		defineInlineAgent(g),
		definePromptAgent(g),
		defineCustomAgent(g),
	}

	if err := runCLI(ctx, agents); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// defineInlineAgent demonstrates DefineAgent with aix.InlinePrompt. The
// prompt is declared right next to the agent definition; the registered
// prompt and the agent share a name. Each turn the framework renders the
// prompt, appends the conversation history, calls the model, and updates
// session state. This is the shortest path from "I want a chat agent" to
// a working one.
func defineInlineAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "pirate"
	return genkit.DefineAgent(g, name,
		aix.InlinePrompt{
			ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			})),
			ai.WithSystem("You are a sarcastic pirate. Keep responses concise."),
		},
		aix.WithSessionStore(mustStore(name)),
		aix.WithDescription[any]("Sarcastic pirate (inline-defined prompt)"),
	)
}

// definePromptAgent demonstrates DefinePromptAgent. The prompt is loaded from
// ./prompts/<agent-name>.prompt by genkit's prompt registry. Defining the
// prompt in a file lets you tune model, config, schema, template, and default
// input independently of the Go code, which is useful when prompt authors are
// not the people writing the agent wiring.
//
// With no source option, DefinePromptAgent defaults to the prompt registered
// under the agent's own name and renders it with the prompt's own default
// input each turn (here, the personality set in chef.prompt's frontmatter).
// The prompt source is a typed option, so it sits in the same variadic as the
// other agent options. To supply an input from code, or to back several agents
// with one shared prompt, add aix.WithNamedPrompt(name, input).
func definePromptAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "chef"
	return genkit.DefinePromptAgent(g, name,
		aix.WithSessionStore(mustStore(name)),
		aix.WithDescription[any]("Michelin-starred chef (prompt loaded from ./prompts/chef.prompt)"),
	)
}

// defineCustomAgent demonstrates DefineCustomAgent. The per-turn function
// is fully under your control: it picks the model, manages the message
// list, streams chunks back to the client, and decides what to put in the
// final result. Use this form when the prompt-backed agent loop doesn't
// fit (e.g. you want to pre/post-process every turn, swap models
// dynamically, or wire up custom tool plumbing).
//
// Even with full control over the loop, the framework still owns session
// state, snapshot writes, and the detach lifecycle.
func defineCustomAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "coder"
	return genkit.DefineCustomAgent(g, name,
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				for chunk, err := range genkit.GenerateStream(ctx, g,
					ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
						ThinkingConfig: &genai.ThinkingConfig{
							ThinkingBudget: genai.Ptr[int32](0),
						},
					})),
					ai.WithSystem("You are a senior software engineer. Answer briefly. Use fenced code blocks when showing code."),
					ai.WithMessages(sess.Messages()...),
				) {
					if err != nil {
						return nil, err
					}
					if chunk.Done {
						sess.AddMessages(chunk.Response.Message)
						// Report how the turn ended so the framework can
						// forward it on the TurnEnd chunk and persist it
						// on the snapshot.
						return &aix.TurnResult{
							FinishReason: aix.AgentFinishReason(chunk.Response.FinishReason),
						}, nil
					}
					resp.SendModelChunk(chunk.Chunk)
				}
				return nil, nil
			}); err != nil {
				return nil, err
			}
			return sess.Result(), nil
		},
		aix.WithSessionStore(mustStore(name)),
		aix.WithDescription[any]("Concise code helper (custom per-turn loop)"),
	)
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
