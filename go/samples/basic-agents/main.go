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
// different styles and exposing all of them through a single CLI:
//
//   - "pirate" uses DefineAgent + aix.InlinePrompt. The prompt is declared
//     inline next to the agent.
//   - "chef" uses DefinePromptAgent. With no source option it defaults to the
//     prompt registered under the agent's name, loaded from
//     ./prompts/chef.prompt.
//   - "coder" uses DefineCustomAgent. The per-turn loop (model selection,
//     history management, streaming) is wired by hand.
//   - "banker" uses DefinePromptAgent (prompt loaded from
//     ./prompts/banker.prompt) with an interruptible tool. It pauses
//     mid-turn to ask the user for approval before moving money, then
//     resumes the tool with their answer. This exercises the tool
//     interrupt / resume flow through the same CLI.
//   - "orchestrator" uses the experimental Agents middleware to delegate to
//     specialized sub-agents (researcher, engineer) through per-agent tools,
//     merging their artifacts into its own session. See orchestrator.go.
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

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
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

// defineInlineAgent demonstrates DefineAgent with aix.InlinePrompt. The
// prompt is declared right next to the agent definition; the registered
// prompt and the agent share a name. Each turn the framework renders the
// prompt, appends the conversation history, calls the model, and updates
// session state. This is the shortest path from "I want a chat agent" to
// a working one.
func defineInlineAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "pirate"
	return genkitx.DefineAgent(g, name,
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
	return genkitx.DefinePromptAgent(g, name,
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
	return genkitx.DefineCustomAgent(g, name,
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

// --- banker: tool interrupt / resume demo ---
//
// The banker is the fourth agent. Unlike the others (which just stream
// text), it pauses mid-turn to get the user's approval before moving
// money, then resumes the tool with their answer. The split mirrors how
// interrupts are meant to be used:
//
//   - the tool (transferMoney) decides when human input is needed and
//     calls tool.Interrupt with typed data describing why it paused;
//   - the CLI client (see InterruptHandler / Prompter in cli.go) collects
//     the interrupt at turn end, asks the user, and resumes the tool.
//
// The agent itself stays trivial: an ordinary prompt-backed agent whose
// prompt lists the transferMoney tool. All the interrupt-specific wiring
// lives in transferMoney and handleTransferInterrupt.

// TransferInput and TransferOutput are the tool's contract: the JSON
// schemas inferred from these field names are what the model sees.
type TransferInput struct {
	ToAccount string  `json:"toAccount" jsonschema:"description=destination account ID"`
	Amount    float64 `json:"amount" jsonschema:"description=amount in dollars (e.g. 50.00 for $50)"`
}

type TransferOutput struct {
	Status     string  `json:"status"`
	Message    string  `json:"message,omitempty"`
	NewBalance float64 `json:"newBalance,omitempty"`
}

// TransferInterrupt is the payload the tool hands the client when it needs
// a human decision. Reason discriminates the cases the handler switches on.
type TransferInterrupt struct {
	Reason    string  `json:"reason"`
	ToAccount string  `json:"toAccount"`
	Amount    float64 `json:"amount"`
	Balance   float64 `json:"balance,omitempty"`
}

// Confirmation is the resume payload the client sends back. It arrives as
// the tool function's resume parameter when the tool is re-executed.
type Confirmation struct {
	Approved       bool     `json:"approved"`
	AdjustedAmount *float64 `json:"adjustedAmount,omitempty"`
}

// accountBalance is the demo's single mutable "account". It is process
// state, not session state, so it is shared across conversations and reset
// on restart — fine for illustrating the interrupt flow.
var accountBalance = 150.00

// defineBankerAgent registers the transferMoney tool and a prompt-backed
// agent that uses it, then returns the agent. Wire it into the CLI with
// handleTransferInterrupt as its interrupt handler (see main).
func defineBankerAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "banker"

	// transferMoney is an interruptible tool: rather than always returning a
	// result, it can pause (tool.Interrupt) to get the user's approval. Its
	// third parameter (*Confirmation) is the resume payload — nil on the
	// first call, populated when the client resumes.
	genkitx.DefineInterruptibleTool(g, "transferMoney",
		"Transfers money to another account. Use when the user wants to send money.",
		func(ctx context.Context, input TransferInput, confirm *Confirmation) (*TransferOutput, error) {
			if confirm != nil {
				if !confirm.Approved {
					return &TransferOutput{Status: "cancelled", Message: "Transfer cancelled by user.", NewBalance: accountBalance}, nil
				}
				if confirm.AdjustedAmount != nil {
					input.Amount = *confirm.AdjustedAmount
				}
			}

			if input.Amount > accountBalance {
				if accountBalance <= 0 {
					return &TransferOutput{Status: "rejected", Message: "Account balance is 0. Please add funds.", NewBalance: accountBalance}, nil
				}
				// Not enough money: pause and ask whether to send what's left.
				return nil, tool.Interrupt(TransferInterrupt{
					Reason: "insufficient_balance", ToAccount: input.ToAccount,
					Amount: input.Amount, Balance: accountBalance,
				})
			}

			if confirm == nil && input.Amount > 100 {
				// Large transfer on the first pass: ask for explicit confirmation.
				return nil, tool.Interrupt(TransferInterrupt{
					Reason: "confirm_large", ToAccount: input.ToAccount,
					Amount: input.Amount, Balance: accountBalance,
				})
			}

			accountBalance -= input.Amount
			return &TransferOutput{
				Status:     "completed",
				Message:    fmt.Sprintf("Transferred $%.2f to %s.", input.Amount, input.ToAccount),
				NewBalance: accountBalance,
			}, nil
		})

	return genkitx.DefinePromptAgent[any](g, name,
		aix.WithSessionStore(mustStore(name)),
		aix.WithDescription[any]("Money transfer assistant (interruptible tool + human approval)"),
	)
}

// handleTransferInterrupt is the banker's InterruptHandler. It reads the
// typed interrupt payload, asks the user through the Prompter, and returns
// a restart part (tool.Resume) carrying their decision. Returning a resume
// part — instead of touching the connection — is what keeps the handler
// decoupled from the CLI's streaming loop.
func handleTransferInterrupt(p *Prompter, part *ai.Part) (*ai.Part, error) {
	meta, ok := tool.InterruptAs[TransferInterrupt](part)
	if !ok {
		// Not our interrupt type; let the CLI report it as unresolved.
		return nil, nil
	}

	switch meta.Reason {
	case "insufficient_balance":
		p.Printf("\nInsufficient balance: $%.2f requested to %s, but only $%.2f available.\n",
			meta.Amount, meta.ToAccount, meta.Balance)
		switch p.Choose("How do you want to proceed?",
			fmt.Sprintf("Transfer $%.2f instead", meta.Balance),
			"Cancel the transfer") {
		case 0:
			return tool.Resume(part, Confirmation{Approved: true, AdjustedAmount: &meta.Balance})
		default:
			return tool.Resume(part, Confirmation{Approved: false})
		}

	case "confirm_large":
		approved := p.Confirm(fmt.Sprintf("\nConfirm large transfer of $%.2f to %s?", meta.Amount, meta.ToAccount))
		return tool.Resume(part, Confirmation{Approved: approved})

	default:
		p.Printf("\nUnrecognized approval request (%q); cancelling the transfer.\n", meta.Reason)
		return tool.Resume(part, Confirmation{Approved: false})
	}
}
