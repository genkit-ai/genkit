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

// barista.go is the session-state demo.
//
// The other agents carry only their conversation from turn to turn. The
// barista also carries a running order, and its system instruction is rebuilt
// from that order every turn, so what the agent is told grows as the
// conversation does. The two halves are:
//
//   - the tool (addToOrder) writes session state through the live session it
//     finds on its context;
//   - the prompt (ai.WithSystemFn) reads that state back and turns it into
//     instructions.
//
// The state is a Go struct, not a bag of keys: the agent is an
// Agent[BaristaOrder], so its store, its snapshots, and every read of
// aix.SessionFromContext are typed end to end. The other agents in this sample
// keep no state of their own and are Agent[any].
//
// This is what a prompt-backed agent can do that a fixed system string cannot,
// and it is the reason the agent's replies stay consistent over a long order
// without the model having to re-read the whole transcript.

package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// BaristaOrder is the agent's session state: what the customer has ordered so
// far. It is the State type parameter on the agent, so the framework infers its
// JSON schema, the file store persists it, and reads come back as this struct
// rather than a map.
//
// omitempty is load-bearing on the slice: a nil Go slice marshals to null,
// which does not satisfy the array schema inferred from this type.
type BaristaOrder struct {
	Drinks []string `json:"drinks,omitempty"`
}

// defineBaristaAgent registers the addToOrder tool and an inline-prompt agent
// whose system instruction is computed from session state on every turn.
func defineBaristaAgent(g *genkit.Genkit) *aix.Agent[BaristaOrder] {
	const name = "barista"

	// The write side, defined with the experimental tool API: the function
	// takes a plain context.Context, so the session lookup reads the same way
	// it does anywhere else. Use tool.AttachParts if a tool needs to return
	// content alongside its output. The banker's interruptible tool is the
	// other half of this API.
	//
	// A tool reaches the live session through its context and mutates the
	// custom state; what it writes is visible to the next turn's prompt
	// render. UpdateCustom takes and returns the state as its own type, so
	// appending a drink is one line and a typo is a compile error.
	addToOrder := genkitx.DefineTool(g, "addToOrder",
		"Records one drink the customer ordered. Call it every time they ask for a drink.",
		func(ctx context.Context, in struct {
			Drink string `json:"drink" jsonschema:"description=The drink to add e.g. flat white"`
		}) (string, error) {
			sess := aix.SessionFromContext[BaristaOrder](ctx)
			if sess == nil {
				// Classify rather than returning a bare error: the runtime
				// wraps a tool failure as ai.ErrToolFailed with the cause
				// preserved, so this stays matchable with errors.Is.
				return "", status.Errorf(status.ErrFailedPrecondition,
					"addToOrder must be called inside a session")
			}
			sess.UpdateCustom(func(order BaristaOrder) BaristaOrder {
				order.Drinks = append(order.Drinks, in.Drink)
				return order
			})
			return "Added " + in.Drink + " to the order.", nil
		})

	return genkitx.DefineAgent(g, name,
		aix.InlinePrompt{
			ai.WithModel(flashModel),
			ai.WithTools(addToOrder),
			// The read side. WithSystemFn runs once per turn, so the
			// instruction is rebuilt from whatever the earlier turns left
			// behind. Its result is used verbatim, so a drink name with a
			// brace in it never has to be escaped. A template could reach the
			// same values as {{@state.drinks}}; a function is what lets the
			// wording branch on them, at the cost of naming the state type.
			ai.WithSystemFn(func(ctx context.Context, _ any) (string, error) {
				sess := aix.SessionFromContext[BaristaOrder](ctx)
				if sess == nil {
					return "", status.Errorf(status.ErrFailedPrecondition,
						"barista prompt rendered outside a session")
				}
				order := sess.Custom()
				if len(order.Drinks) == 0 {
					return "You are a brisk, friendly barista. Take the customer's order one drink at a time, calling addToOrder for each. Keep every reply to a sentence.", nil
				}
				return fmt.Sprintf(
					"You are a brisk, friendly barista. The customer has ordered so far: %s. Call addToOrder for anything they add, read the order back when they ask, and keep every reply to a sentence.",
					strings.Join(order.Drinks, ", "),
				), nil
			}),
		},
		aix.WithSessionStore(mustStore[BaristaOrder](name)),
		aix.WithDescription[BaristaOrder]("Coffee order taker (typed session state read back into the prompt)"),
	)
}
