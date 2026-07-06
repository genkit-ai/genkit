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
//
// SPDX-License-Identifier: Apache-2.0

package exp

import (
	"sort"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/genkitbridge"
)

// DefineAgent defines an agent backed by an inline prompt and registers it as
// an action on the registry. Returns an [aix.Agent].
//
// An Agent is a stateful, multi-turn conversational action. It builds on
// bidirectional streaming to enable ongoing conversations where each turn's
// input and output are streamed between client and server. The framework
// handles session state, conversation history, and optional snapshot
// persistence automatically.
//
// The prompt is defined inline via [aix.InlinePrompt] and registered under the
// agent's name. To back the agent with a prompt already in the registry (e.g.
// one from a .prompt file), use [DefinePromptAgent] instead.
//
// The State type parameter is inferred from the typed agent options
// (e.g. [aix.WithSessionStore], [aix.WithStateTransform]); pass an explicit
// [State] only when no typed option is provided.
//
// The returned agent is an [api.BidiAction]; pass it to [genkit.Handler] to
// serve it over HTTP, one turn per request. Server-managed agents also
// register companion actions for the snapshot lifecycle; serve them
// alongside the agent via [aix.Agent.GetSnapshotAction] and
// [aix.Agent.AbortAction].
//
// For full control over the per-turn loop, use [DefineCustomAgent].
//
// # Options
//
//   - [aix.WithSessionStore]: Enable snapshot persistence
//   - [aix.WithStateTransform]: Rewrite session state on its way out to the client
//   - [aix.WithStreamTransform]: Rewrite stream chunks on their way out to the client
//
// Example:
//
//	chatAgent := exp.DefineAgent(g, "chat",
//		aix.InlinePrompt{
//			ai.WithModelName("googleai/gemini-flash-latest"),
//			ai.WithSystem("You are a helpful assistant."),
//		},
//		aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
//	)
func DefineAgent[State any](
	g *genkit.Genkit,
	name string,
	prompt aix.InlinePrompt,
	opts ...aix.AgentOption[State],
) *aix.Agent[State] {
	requireExperimental(g, "DefineAgent")
	return aix.DefineAgent(genkitbridge.RegistryOf(g), name, prompt, opts...)
}

// DefinePromptAgent defines a prompt-backed agent sourced from the registry by
// name and registers it as an action. Returns an [aix.Agent].
//
// By default the agent uses the prompt registered under its own name (e.g. one
// defined via [genkit.DefinePrompt] or loaded from a .prompt file), so no source
// option is required. Pass [aix.WithNamedPrompt] to reference a differently
// named prompt and supply its render input from code, so a single prompt can
// back many agents.
//
// It is the registry-backed counterpart of [DefineAgent]: where [DefineAgent]
// defines the prompt inline, DefinePromptAgent points at a prompt already in
// the registry. The prompt source is a typed option ([aix.WithNamedPrompt])
// rather than a positional argument, so it composes with the other agent
// options in one variadic. For full control over the per-turn loop, use
// [DefineCustomAgent].
//
// The State type parameter is inferred from the typed agent options; pass an
// explicit [State] only when no typed option provides it.
//
// # Options
//
//   - [aix.WithNamedPrompt]: Source from a differently named prompt with a code-supplied input
//   - [aix.WithSessionStore]: Enable snapshot persistence
//   - [aix.WithStateTransform]: Rewrite session state on its way out to the client
//   - [aix.WithStreamTransform]: Rewrite stream chunks on their way out to the client
//
// Example (same-named prompt loaded from ./prompts/chef.prompt):
//
//	chef := exp.DefinePromptAgent(g, "chef",
//		aix.WithSessionStore(localstore.NewInMemorySessionStore[any]()),
//	)
//
// Example (a shared prompt, parameterized per agent):
//
//	pirate := exp.DefinePromptAgent(g, "pirate",
//		aix.WithNamedPrompt[any]("chat", ChatInput{Personality: "a sarcastic pirate"}),
//	)
func DefinePromptAgent[State any](
	g *genkit.Genkit,
	name string,
	opts ...aix.PromptAgentOption[State],
) *aix.Agent[State] {
	requireExperimental(g, "DefinePromptAgent")
	return aix.DefinePromptAgent(genkitbridge.RegistryOf(g), name, opts...)
}

// DefineCustomAgent defines an agent with full control over the conversation
// loop, registers it as an action of type agent, and returns an
// [aix.Agent].
//
// The provided function fn receives a [aix.Responder] for streaming output
// to the client and an [aix.SessionRunner] for accessing conversation state.
// Call [aix.SessionRunner.Run] to enter the turn loop, which blocks until the
// client sends the next message.
//
// Like [DefineAgent], the returned agent is an [api.BidiAction] servable
// via [genkit.Handler], with companion actions on [aix.Agent.GetSnapshotAction]
// and [aix.Agent.AbortAction].
//
// For agents backed by a prompt, use [DefineAgent] (inline prompt) or
// [DefinePromptAgent] (a prompt already in the registry) instead.
//
// # Options
//
//   - [aix.WithSessionStore]: Enable snapshot persistence
//   - [aix.WithStateTransform]: Rewrite session state on its way out to the client
//   - [aix.WithStreamTransform]: Rewrite stream chunks on their way out to the client
//
// The State type parameter is the shape of the conversation's custom state
// ([aix.SessionState.Custom]); mutating it via [aix.Session.UpdateCustom]
// streams an [aix.AgentStreamChunk.CustomPatch] delta to the client.
//
// Example:
//
//	chatAgent := exp.DefineCustomAgent(g, "chat",
//		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
//			var lastMessage *ai.Message
//			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
//				var reason aix.AgentFinishReason
//				for result, err := range genkit.GenerateStream(ctx, g,
//					ai.WithModelName("googleai/gemini-3-flash-preview"),
//					ai.WithMessages(sess.Messages()...),
//				) {
//					if err != nil {
//						return nil, err
//					}
//					if result.Done {
//						lastMessage = result.Response.Message
//						reason = aix.AgentFinishReason(result.Response.FinishReason)
//						sess.AddMessages(lastMessage)
//					} else {
//						resp.SendModelChunk(result.Chunk)
//					}
//				}
//				// Report how the turn ended; the framework forwards it on
//				// the TurnEnd chunk and persists it on the snapshot.
//				return &aix.TurnResult{FinishReason: reason}, nil
//			})
//			if err != nil {
//				return nil, err
//			}
//			return &aix.AgentResult{Message: lastMessage}, nil
//		},
//	)
func DefineCustomAgent[State any](
	g *genkit.Genkit,
	name string,
	fn aix.AgentFunc[State],
	opts ...aix.AgentOption[State],
) *aix.Agent[State] {
	requireExperimental(g, "DefineCustomAgent")
	return aix.DefineCustomAgent(genkitbridge.RegistryOf(g), name, fn, opts...)
}

// ListAgents returns a slice of all [api.Action] instances that represent
// agents registered with the Genkit instance g. Like [genkit.ListFlows], this
// is useful for introspection or for dynamically exposing agent endpoints in an
// HTTP server; an agent served via [genkit.Handler] runs one turn per request.
func ListAgents(g *genkit.Genkit) []api.Action {
	reg := genkitbridge.RegistryOf(g)
	agents := []api.Action{}
	for _, act := range reg.ListActions() {
		if act.Desc().Type == api.ActionTypeAgent {
			agents = append(agents, act)
		}
	}
	sort.Slice(agents, func(i, j int) bool {
		return agents[i].Name() < agents[j].Name()
	})
	return agents
}
