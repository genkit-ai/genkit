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

// This sample is basic-tool-interrupts, written against the in-preview tools
// API in genkit/exp. It is slated to replace the one basic-tool-interrupts
// uses in the next major version, so the two are worth reading side by side:
// same human in the loop (HITL) approval, same flow, and the differences are
// all API.
//
// A tool interrupt is how Genkit does HITL: transferMoney pauses generation
// when a transfer is large enough to need approving, and the flow runs a second
// turn that resumes the tool with the answer.
//
// DefineInterruptibleTool takes a third type parameter for what comes back on
// the resume, which is what the differences all follow from:
//
//   - The tool function takes a plain context.Context and an *Approval, rather
//     than *[ai.ToolContext]. The parameter is nil on the first call and set on
//     the resume, so the tool reads a typed value instead of asking whether it
//     was resumed and then pulling metadata out by key.
//   - tool.Interrupt pauses with a typed value, in place of ai.InterruptWith.
//   - Tool.Resume carries a typed Approval, in place of RestartWith and an
//     ai.WithResumedMetadata map.
//
// The approve field stands in for the person: a real app would hand the pending
// interrupt to a client and run the second turn when they answer.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to call the flow from a browser and read a trace of both
// turns at http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP. The balance carries across requests, so run these in order.
// Decline a transfer that needs approving, which leaves the money where it is:
//
//	curl -N -X POST 'http://localhost:8080/transferFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"request": "Send $120 to bob", "approve": false}}'
//
// Then approve the same one, which spends it:
//
//	curl -N -X POST 'http://localhost:8080/transferFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"request": "Send $120 to bob", "approve": true}}'
//
// A small transfer needs no approval, so it finishes in one turn:
//
//	curl -N -X POST 'http://localhost:8080/transferFlow?stream=true' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"request": "Send $20 to alice", "approve": false}}'
//
// Repeating the $120 one now asks for more than is left, which the tool answers
// outright rather than pausing on.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
	"google.golang.org/genai"
)

// Transfers above this need a human to approve them.
const approvalLimit = 100

// accountBalance stands in for a bank. A real app would use a database, which
// would also survive a restart and two requests at once.
var accountBalance = 150.00

type (
	// TransferInput is what the model fills in to call the tool.
	TransferInput struct {
		ToAccount string  `json:"toAccount" jsonschema_description:"The destination account"`
		Amount    float64 `json:"amount" jsonschema_description:"The amount in dollars"`
	}

	// TransferResult is what the tool answers with.
	TransferResult struct {
		Status  string  `json:"status" jsonschema:"enum=completed,enum=declined,enum=rejected"`
		Balance float64 `json:"balance" jsonschema_description:"The balance after the transfer"`
	}

	// Approval is the answer carried back into the tool when it is resumed. It
	// is the third type parameter of the tool, so a shape can be given to it
	// here rather than agreed on key by key between the tool and the flow.
	Approval struct {
		Approved bool `json:"approved"`
	}

	// TransferInterrupt is the typed metadata the tool attaches when it pauses,
	// so whoever answers knows what they are approving.
	TransferInterrupt struct {
		ToAccount string  `json:"toAccount"`
		Amount    float64 `json:"amount"`
	}

	// TransferRequest is what the flow takes.
	TransferRequest struct {
		Request string `json:"request" jsonschema:"default=Send $120 to bob" jsonschema_description:"What to do with the money"`
		Approve bool   `json:"approve,omitempty" jsonschema:"default=true" jsonschema_description:"Whether to approve a transfer that needs it"`
	}

	// Transfer is what the flow returns.
	Transfer struct {
		Reply    string  `json:"reply"`
		Balance  float64 `json:"balance"`
		Approved bool    `json:"approved"`
	}
)

// model is shared by every flow below, so switching models or thinking levels
// for the whole sample is a one-line change.
var model = googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
	ThinkingConfig: &genai.ThinkingConfig{
		ThinkingLevel: genai.ThinkingLevelMedium,
	},
})

func main() {
	ctx := context.Background()

	// The Google AI plugin reads the API key from GEMINI_API_KEY or
	// GOOGLE_API_KEY, which is the recommended practice.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithExperimental(), // This sample uses the new in-preview tools API.
	)

	// An interruptible tool declares what it is resumed with, so the answer
	// arrives as a third parameter rather than as metadata to look up.
	transferMoney := genkitx.DefineInterruptibleTool(g, "transferMoney",
		"Transfers money to another account.",
		func(ctx context.Context, input TransferInput, approval *Approval) (*TransferResult, error) {
			if input.Amount > accountBalance {
				// An ordinary answer, not an interrupt: the model can explain
				// this to the user without anyone being asked anything.
				return &TransferResult{Status: "rejected", Balance: accountBalance}, nil
			}
			// approval is nil on the first call and set when the tool is
			// resumed, which is what tells a fresh large transfer from an
			// answered one.
			if approval == nil && input.Amount > approvalLimit {
				return nil, tool.Interrupt(TransferInterrupt{
					ToAccount: input.ToAccount,
					Amount:    input.Amount,
				})
			}
			// Letting the tool decide, rather than the flow, is what keeps the
			// rule it paused on in one place.
			if approval != nil && !approval.Approved {
				return &TransferResult{Status: "declined", Balance: accountBalance}, nil
			}

			accountBalance -= input.Amount
			return &TransferResult{Status: "completed", Balance: accountBalance}, nil
		})

	genkit.DefineStreamingFlow(g, "transferFlow",
		func(ctx context.Context, input TransferRequest, sendChunk core.StreamCallback[string]) (*Transfer, error) {
			// Forwarding the model's text is all the streaming this flow does,
			// so WithStreaming carries it and Generate still returns the
			// finished response to inspect. The tool turn streams as well and
			// carries no text, so the check keeps blank chunks off the wire.
			forward := ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				if text := chunk.Text(); text != "" {
					return sendChunk(ctx, text)
				}
				return nil
			})

			// Turn one. It ends early if the tool interrupts, so the response
			// holds a paused tool call rather than a finished answer. Both
			// turns stream, so the pause shows up as a gap in the output.
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(model),
				ai.WithSystem("You are a payment assistant. Use transferMoney to move money, and tell the user what happened."),
				ai.WithPrompt(input.Request),
				ai.WithTools(transferMoney),
				forward,
			)
			if err != nil {
				return nil, fmt.Errorf("could not start the transfer: %w", err)
			}

			interrupts := resp.Interrupts()
			if len(interrupts) == 0 {
				// Nothing needed approving, so the first turn is the whole answer.
				return &Transfer{Reply: resp.Text(), Balance: accountBalance}, nil
			}

			// Answer every interrupt. Resume builds a part rather than calling
			// the model, so the decision travels with the next request.
			var restarts []*ai.Part
			for _, interrupt := range interrupts {
				meta, ok := tool.InterruptAs[TransferInterrupt](interrupt)
				if !ok {
					return nil, status.Errorf(status.ErrInternal, "unexpected interrupt: %s", interrupt.ToolRequest.Name)
				}
				logger.Info(ctx, "transfer needs approval",
					"amount", meta.Amount, "toAccount", meta.ToAccount, "approve", input.Approve)

				part, err := transferMoney.Resume(interrupt, Approval{Approved: input.Approve})
				if err != nil {
					return nil, fmt.Errorf("could not answer the approval: %w", err)
				}
				restarts = append(restarts, part)
			}

			// Turn two: the same conversation, plus the answers. History
			// carries the paused call, so the model picks up where it left off.
			resp, err = genkit.Generate(ctx, g,
				ai.WithModel(model),
				ai.WithMessages(resp.History()...),
				ai.WithTools(transferMoney),
				ai.WithToolRestarts(restarts...),
				forward,
			)
			if err != nil {
				return nil, fmt.Errorf("could not finish the transfer: %w", err)
			}
			return &Transfer{Reply: resp.Text(), Balance: accountBalance, Approved: input.Approve}, nil
		})

	// Serve every flow over HTTP.
	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
