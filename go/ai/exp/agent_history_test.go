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
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
)

// A prompt-backed agent hands its conversation to the prompt's render rather
// than splicing it in afterwards, so where the history lands is the prompt's
// decision. These tests pin each placement rule against the model requests the
// turns actually produce, and against what the session is left holding.

// recordingModel defines a model that records every request it receives and
// replies with scripted text, falling back to "reply" once the script is
// exhausted.
func recordingModel(t *testing.T, r api.Registry, name string, replies ...string) *[]*ai.ModelRequest {
	t.Helper()
	ai.ConfigureFormats(r)
	ai.DefineGenerateAction(context.Background(), r)
	var requests []*ai.ModelRequest
	turn := 0
	defineTestModel(r, name, &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		requests = append(requests, req)
		text := "reply"
		if turn < len(replies) {
			text = replies[turn]
		}
		turn++
		resp := &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage(text)}
		if cb != nil {
			if err := cb(ctx, &ai.ModelResponseChunk{Content: resp.Message.Content}); err != nil {
				return nil, err
			}
		}
		return resp, nil
	})
	return &requests
}

// summarize renders messages as "role:text" for whole-conversation assertions.
func summarize(messages []*ai.Message) []string {
	out := make([]string, 0, len(messages))
	for _, m := range messages {
		out = append(out, string(m.Role)+":"+strings.TrimSpace(m.Text()))
	}
	return out
}

func assertConversation(t *testing.T, got []*ai.Message, want []string) {
	t.Helper()
	if summary := summarize(got); !slices.Equal(summary, want) {
		t.Errorf("conversation =\n  %q\nwant\n  %q", summary, want)
	}
}

// runTurns drives an agent through the given user messages on one connection
// and returns the final session messages.
func runTurns(t *testing.T, af *Agent[testState], texts ...string) []*ai.Message {
	t.Helper()
	ctx := context.Background()
	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	for _, text := range texts {
		sendTurn(t, conn, text)
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	return out.State.Messages
}

// TestPromptAgent_HistoryPlacement covers each way a prompt can place the
// conversation it is handed, across two turns so accumulation is visible.
func TestPromptAgent_HistoryPlacement(t *testing.T) {
	// The prompt declares nothing, so the conversation is the whole middle:
	// [system, ...history].
	t.Run("system only", func(t *testing.T) {
		reg := newTestRegistry(t)
		reqs := recordingModel(t, reg, "test/order1", "a1", "a2")
		af := DefineAgent[testState](reg, "orderSystemOnly", InlinePrompt{
			ai.WithModelName("test/order1"),
			ai.WithSystem("Be helpful."),
		})

		session := runTurns(t, af, "q1", "q2")

		assertConversation(t, (*reqs)[1].Messages, []string{
			"system:Be helpful.", "user:q1", "model:a1", "user:q2",
		})
		// The system message is the prompt's, so it is re-rendered every
		// turn and never accumulates in the session.
		assertConversation(t, session, []string{
			"user:q1", "model:a1", "user:q2", "model:a2",
		})
	})

	// With a user prompt the conversation goes between it and the system
	// message, so the template's turn stays last. The agent loop used to
	// append history after the whole render, which put the template's user
	// turn ahead of the conversation.
	t.Run("system and user prompt", func(t *testing.T) {
		reg := newTestRegistry(t)
		reqs := recordingModel(t, reg, "test/order2", "a1", "a2")
		af := DefineAgent[testState](reg, "orderSystemPrompt", InlinePrompt{
			ai.WithModelName("test/order2"),
			ai.WithSystem("Be helpful."),
			ai.WithPrompt("Be concise."),
		})

		session := runTurns(t, af, "q1", "q2")

		assertConversation(t, (*reqs)[1].Messages, []string{
			"system:Be helpful.", "user:q1", "model:a1", "user:q2", "user:Be concise.",
		})
		assertConversation(t, session, []string{
			"user:q1", "model:a1", "user:q2", "model:a2",
		})
	})

	// {{history}} puts the conversation exactly where the template says.
	t.Run("messages template with history marker", func(t *testing.T) {
		reg := newTestRegistry(t)
		reqs := recordingModel(t, reg, "test/order3", "a1", "a2")
		af := DefineAgent[testState](reg, "orderMarker", InlinePrompt{
			ai.WithModelName("test/order3"),
			ai.WithSystem("Be helpful."),
			ai.WithMessagesTemplate(`{{role "user"}}Here is the conversation so far:
{{history}}
{{role "model"}}Now respond to the latest message.`),
		})

		session := runTurns(t, af, "q1", "q2")

		assertConversation(t, (*reqs)[1].Messages, []string{
			"system:Be helpful.",
			"user:Here is the conversation so far:",
			"user:q1", "model:a1", "user:q2",
			"model:Now respond to the latest message.",
		})
		assertConversation(t, session, []string{
			"user:q1", "model:a1", "user:q2", "model:a2",
		})
	})

	// Without the marker, dotprompt inserts the conversation before the
	// template's final user message.
	t.Run("messages template without marker", func(t *testing.T) {
		reg := newTestRegistry(t)
		reqs := recordingModel(t, reg, "test/order4", "a1", "a2")
		af := DefineAgent[testState](reg, "orderNoMarker", InlinePrompt{
			ai.WithModelName("test/order4"),
			ai.WithMessagesTemplate(`{{role "model"}}I am ready.
{{role "user"}}Answer carefully.`),
		})

		session := runTurns(t, af, "q1", "q2")

		assertConversation(t, (*reqs)[1].Messages, []string{
			"model:I am ready.",
			"user:q1", "model:a1", "user:q2",
			"user:Answer carefully.",
		})
		assertConversation(t, session, []string{
			"user:q1", "model:a1", "user:q2", "model:a2",
		})
	})

	// A function owns the conversation outright: it reads what it was handed
	// and decides what survives. Trimming is the context-window case, and it
	// is what the loop could not express while history was appended after
	// the render.
	t.Run("messages function trims history", func(t *testing.T) {
		reg := newTestRegistry(t)
		reqs := recordingModel(t, reg, "test/order5", "a1", "a2", "a3")
		af := DefineAgent[testState](reg, "orderTrim", InlinePrompt{
			ai.WithModelName("test/order5"),
			ai.WithSystem("Be helpful."),
			ai.WithMessagesFn(func(ctx context.Context, _ any) ([]*ai.Message, error) {
				history := ai.HistoryFromContext(ctx)
				if len(history) <= 2 {
					return history, nil
				}
				dropped := len(history) - 2
				note := ai.NewUserTextMessage(fmt.Sprintf("(%d earlier messages omitted)", dropped))
				return append([]*ai.Message{note}, history[dropped:]...), nil
			}),
		})

		session := runTurns(t, af, "q1", "q2", "q3")

		// Turn 3 sees the note plus the last two messages, not the whole
		// transcript.
		assertConversation(t, (*reqs)[2].Messages, []string{
			"system:Be helpful.",
			"user:(2 earlier messages omitted)",
			"model:a2", "user:q3",
		})
		// The session holds the conversation the prompt placed, so the
		// compaction persists rather than being recomputed from an
		// ever-growing transcript every turn. The note is the prompt's, so
		// it is dropped as scaffolding.
		assertConversation(t, session, []string{
			"model:a2", "user:q3", "model:a3",
		})
	})

	// Static messages are content the prompt already produced, so the
	// conversation is not spliced in on top of them. The caller's messages
	// are placed only by a prompt that asks for them, or by one that
	// declares no conversation at all.
	t.Run("static messages own the conversation", func(t *testing.T) {
		reg := newTestRegistry(t)
		reqs := recordingModel(t, reg, "test/order6", "a1", "a2")
		af := DefineAgent[testState](reg, "orderStatic", InlinePrompt{
			ai.WithModelName("test/order6"),
			ai.WithSystem("Be helpful."),
			ai.WithMessages(ai.NewUserTextMessage("example in"), ai.NewModelTextMessage("example out")),
		})

		session := runTurns(t, af, "q1", "q2")

		assertConversation(t, (*reqs)[1].Messages, []string{
			"system:Be helpful.", "user:example in", "model:example out",
		})
		// Nothing from the session reached the request, so the turn's own
		// reply is all that comes back into it and the conversation never
		// accumulates. This is the sharp end of the ownership rule: a
		// prompt that declares static messages and never places the
		// conversation has no conversation.
		assertConversation(t, session, []string{"model:a2"})
	})
}

// TestPromptAgent_HistoryTagStrippedBeforeModel pins that the marker used to
// tell the session's messages from the prompt's scaffolding never reaches the
// model, and that neither marker survives into the session.
func TestPromptAgent_HistoryTagStrippedBeforeModel(t *testing.T) {
	reg := newTestRegistry(t)
	reqs := recordingModel(t, reg, "test/tagcheck", "a1", "a2")
	af := DefineAgent[testState](reg, "tagCheck", InlinePrompt{
		ai.WithModelName("test/tagcheck"),
		ai.WithSystem("Be helpful."),
	})

	session := runTurns(t, af, "q1", "q2")

	for i, m := range (*reqs)[1].Messages {
		if _, ok := m.Metadata[sessionMessageKey]; ok {
			t.Errorf("request message %d (%s) carries %s: %v", i, m.Role, sessionMessageKey, m.Metadata)
		}
	}
	// The scaffolding tag is what the loop filters on, so it must be present
	// on the system message and absent from the conversation.
	if !hasTag((*reqs)[1].Messages[0], promptMessageKey) {
		t.Errorf("system message is not tagged %s", promptMessageKey)
	}
	for i, m := range (*reqs)[1].Messages[1:] {
		if hasTag(m, promptMessageKey) {
			t.Errorf("history message %d (%s) is tagged %s", i, m.Role, promptMessageKey)
		}
	}
	for i, m := range session {
		if _, ok := m.Metadata[sessionMessageKey]; ok {
			t.Errorf("session message %d carries %s: %v", i, sessionMessageKey, m.Metadata)
		}
		if _, ok := m.Metadata[promptMessageKey]; ok {
			t.Errorf("session message %d carries %s: %v", i, promptMessageKey, m.Metadata)
		}
	}
}

// TestPromptAgent_StateReachesContentFunction covers the other half of what a
// content function can read during an agent turn: the live session, and through
// it the custom state, which is how a prompt adapts to what earlier turns
// established.
func TestPromptAgent_StateReachesContentFunction(t *testing.T) {
	reg := newTestRegistry(t)
	ai.ConfigureFormats(reg)
	ai.DefineGenerateAction(context.Background(), reg)

	// A tool is the write side: it reaches the live session from its context
	// and updates the custom state, which the next turn's render sees.
	bump := defineTestTool(reg, "bump", "increments the counter",
		func(tc *ai.ToolContext, _ struct{}) (string, error) {
			sess := SessionFromContext[testState](tc)
			if sess == nil {
				return "", fmt.Errorf("no session on the tool context")
			}
			sess.UpdateCustom(func(s testState) testState {
				s.Counter++
				return s
			})
			return "ok", nil
		})

	// Calls bump once per turn, then answers on the pass that carries the
	// tool's response.
	var systemSeen []string
	defineTestModel(reg, "test/statefn", &ai.ModelOptions{
		Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true},
	}, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		for _, m := range req.Messages {
			if m.Role == ai.RoleSystem {
				systemSeen = append(systemSeen, m.Text())
			}
		}
		for _, p := range req.Messages[len(req.Messages)-1].Content {
			if p.IsToolResponse() {
				return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("done")}, nil
			}
		}
		return &ai.ModelResponse{Request: req, Message: ai.NewModelMessage(
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "bump", Input: map[string]any{}}),
		)}, nil
	})

	var renders int
	af := DefineAgent[testState](reg, "stateFnAgent", InlinePrompt{
		ai.WithModelName("test/statefn"),
		ai.WithTools(bump),
		ai.WithSystemFn(func(ctx context.Context, _ any) (string, error) {
			renders++
			sess := SessionFromContext[testState](ctx)
			if sess == nil {
				return "", fmt.Errorf("no session on the render context")
			}
			return fmt.Sprintf("counter=%d", sess.Custom().Counter), nil
		}),
	})

	runTurns(t, af, "q1", "q2")

	if renders != 2 {
		t.Errorf("system function ran %d times, want one render per turn", renders)
	}
	// One render per turn, so turn 2 is told what turn 1's tool wrote. The
	// system message is rendered once and reused across the tool loop, so it
	// reaches the model twice per turn with the same text.
	if want := []string{"counter=0", "counter=0", "counter=1", "counter=1"}; !slices.Equal(systemSeen, want) {
		t.Errorf("system messages = %q, want %q", systemSeen, want)
	}
}
