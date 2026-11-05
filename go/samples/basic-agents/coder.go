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

package main

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// defineCustomAgent demonstrates DefineCustomAgent. The per-turn function
// is fully under your control: it picks the model, manages the message
// list, streams chunks back to the client, and decides what to put in the
// final result. Use this form when the prompt-backed agent loop doesn't
// fit (e.g. you want to pre/post-process every turn, swap models
// dynamically, or wire up custom tool plumbing).
//
// Even with full control over the loop, the framework still owns session
// state, snapshot writes, and the detach lifecycle. What a custom loop does
// own is whether a turn that failed is worth keeping; see the error arm
// below.
func defineCustomAgent(g *genkit.Genkit) *aix.Agent[any] {
	const name = "coder"
	return genkitx.DefineCustomAgent(g, name,
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				for chunk, err := range genkit.GenerateStream(ctx, g,
					ai.WithModel(model),
					ai.WithSystem("You are a senior software engineer. Answer in as few words as possible. Use fenced code blocks for any code."),
					ai.WithMessages(sess.Messages()...),
				) {
					if err != nil {
						// Commit the turn rather than discard it: sess.Run
						// already stored this turn's input, so the session
						// holds a conversation ending on the user's message,
						// which is a turn seam. Returning a TurnResult beside
						// the error is what says so, and it makes the failed
						// snapshot a resume point that re-runs the turn on
						// the same message. A bare error would roll the turn
						// back and the user would have to type it again.
						return &aix.TurnResult{}, fmt.Errorf("could not answer the question: %w", err)
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
		aix.WithSessionStore(mustStore[any](name)),
		aix.WithDescription[any]("Concise code helper (custom per-turn loop)"),
	)
}
