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

// This sample demonstrates the experimental Budget middleware, which caps the
// tokens a generate run spends across its whole tool loop.
//
// The model follows a chain of boxes to a treasure. Each box names the next,
// so every box costs a turn, and a small budget runs out before the end:
//
//   - huntFlow stops the run once it spends maxTokens.
//   - huntWithApprovalFlow pauses the run instead, and approves another
//     maxTokens up to "approvals" times.
//
// Run it:
//
//	go run .
//
// Or with the Dev UI, to watch each turn and its usage at
// http://localhost:4000/traces:
//
//	curl -sL cli.genkit.dev | bash    # install the Genkit CLI, once
//	genkit start -- go run .
//
// Or over HTTP:
//
//	curl -X POST 'http://localhost:8080/huntWithApprovalFlow' \
//	  -H "Content-Type: application/json" \
//	  -d '{"data": {"maxTokens": 3000, "approvals": 3}}'
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	middlewarex "github.com/firebase/genkit/go/plugins/middleware/exp"
	"github.com/firebase/genkit/go/plugins/server"
)

// HuntRequest is what both flows take. The Dev UI pre-fills its form from
// the defaults.
type HuntRequest struct {
	MaxTokens int `json:"maxTokens" jsonschema:"default=3000" jsonschema_description:"Tokens one run may spend, thinking included"`
	Approvals int `json:"approvals,omitempty" jsonschema:"default=3" jsonschema_description:"How many times huntWithApprovalFlow approves another maxTokens"`
}

// boxes is the chain of clues. Each note names the next box, so the model
// cannot open them in parallel.
var boxes = map[int]string{
	1: "A note: the next clue is in box 7.",
	7: "A note: try box 3.",
	3: "A note: box 9 holds the next clue.",
	9: "A note: go to box 4.",
	4: "A note: the treasure is in box 6.",
	6: "The treasure: a brass key engraved with the word GENKIT.",
}

func main() {
	ctx := context.Background()

	// Registering the experimental Middleware plugin exposes Budget to the
	// Dev UI.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}, &middlewarex.Middleware{}),
		genkit.WithDefaultModel("googleai/gemini-flash-latest"),
	)

	openBox := genkit.DefineTool(g, "openBox", "Opens a numbered box and says what is inside.",
		func(ctx *ai.ToolContext, in struct {
			Box int `json:"box"`
		}) (string, error) {
			if note, ok := boxes[in.Box]; ok {
				return note, nil
			}
			return "The box is empty.", nil
		})

	const hunt = "Find the treasure: open box 1 and follow the notes. Then say what it is."

	genkit.DefineFlow(g, "huntFlow", func(ctx context.Context, in HuntRequest) (string, error) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt(hunt),
			ai.WithTools(openBox),
			ai.WithUse(&middlewarex.Budget{Limit: ai.GenerationUsage{TotalTokens: in.MaxTokens}}),
		)
		// The run stops before the turn that would go past the limit, and
		// still returns its response, whose TotalUsage is what it spent.
		if errors.Is(err, ai.ErrBudgetExceeded) {
			return fmt.Sprintf("Out of budget after %d tokens.", resp.TotalUsage.TotalTokens), nil
		}
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	genkit.DefineFlow(g, "huntWithApprovalFlow", func(ctx context.Context, in HuntRequest) (string, error) {
		budget := &middlewarex.Budget{Limit: ai.GenerationUsage{TotalTokens: in.MaxTokens}, Interrupt: true}

		resp, err := genkit.Generate(ctx, g, ai.WithPrompt(hunt), ai.WithTools(openBox), ai.WithUse(budget))
		for range in.Approvals {
			if err != nil || resp.FinishReason != ai.FinishReasonInterrupted {
				break
			}
			// The budget held the tool calls that would lead to the next
			// turn. Approving them grants another maxTokens.
			var approvals []*ai.Part
			for _, part := range resp.Interrupts() {
				if call, ok := middlewarex.BudgetInterrupted(part); ok {
					hold, _ := ai.InterruptAs[middlewarex.BudgetExceeded](call.Part)
					log.Printf("%s, approving more", hold.Message)
					approvals = append(approvals, call.Restart(middlewarex.BudgetDecision{Approved: true}))
				}
			}
			resp, err = genkit.Generate(ctx, g,
				ai.WithMessages(resp.History()...),
				ai.WithResume(approvals...),
				ai.WithTools(openBox),
				ai.WithUse(budget),
			)
		}
		if err != nil {
			return "", err
		}
		if resp.FinishReason == ai.FinishReasonInterrupted {
			return "Paused: out of approvals.", nil
		}
		return resp.Text(), nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
