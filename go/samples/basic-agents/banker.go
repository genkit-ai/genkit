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

// banker.go is the tool interrupt / resume demo.
//
// Unlike the other agents (which just stream text), the banker pauses
// mid-turn to get the user's approval before moving money, then resumes the
// tool with their answer. The split mirrors how interrupts are meant to be
// used:
//
//   - the tool (transferMoney) decides when human input is needed and
//     calls tool.Interrupt with typed data describing why it paused;
//   - the CLI client (see InterruptHandler / Prompter in cli.go) collects
//     the interrupt at turn end, asks the user, and resumes the tool.
//
// The agent itself stays trivial: an ordinary prompt-backed agent whose
// prompt lists the transferMoney tool. All the interrupt-specific wiring
// lives in transferMoney and handleTransferInterrupt.

package main

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

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
		p.Printf("Insufficient balance: $%.2f requested to %s, but only $%.2f available.\n",
			meta.Amount, meta.ToAccount, meta.Balance)
		switch p.Choose("",
			fmt.Sprintf("Transfer $%.2f instead", meta.Balance),
			"Cancel the transfer") {
		case 0:
			return tool.Resume(part, Confirmation{Approved: true, AdjustedAmount: &meta.Balance})
		default:
			return tool.Resume(part, Confirmation{Approved: false})
		}

	case "confirm_large":
		approved := p.Confirm(fmt.Sprintf("Confirm large transfer of $%.2f to %s?", meta.Amount, meta.ToAccount))
		return tool.Resume(part, Confirmation{Approved: approved})

	default:
		p.Printf("Unrecognized approval request (%q); cancelling the transfer.\n", meta.Reason)
		return tool.Resume(part, Confirmation{Approved: false})
	}
}
