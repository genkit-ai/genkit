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

package livetest

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// The agent cases run each conversation through a file-backed session store
// and start every turn as a fresh invocation resumed by session ID. Every
// turn therefore replays a history that went through JSON on disk, which is
// where provider metadata such as reasoning signatures is lost if a plugin
// only reads the in-memory form of it.
//
// Each case keeps one feature going across turns and checks that the turns
// after it still work, since a history the provider rejects only shows up on
// the turn that sends it.

// session is one conversation with an agent.
type session struct {
	r     *runner
	agent *aix.Agent[any]
	id    string
}

// turn is what one agent turn produced.
type turn struct {
	out *aix.AgentOutput[any]
	// text and reasoning are what streamed.
	text, reasoning string
	chunks          int
	end             *aix.TurnEnd
}

// newSession defines an agent over model with the extra prompt options and
// returns an empty conversation with it.
func (r *runner) newSession(t *testing.T, model ai.ModelArg, opts ...ai.PromptOption) *session {
	t.Helper()
	store, err := localstore.NewFileSessionStore[any](t.TempDir())
	if err != nil {
		t.Fatalf("NewFileSessionStore() error = %v", err)
	}
	r.agents++
	agent := genkitx.DefineAgent(r.g, fmt.Sprintf("livetestAgent%d", r.agents),
		aix.InlinePrompt(append([]ai.PromptOption{ai.WithModel(model)}, opts...)),
		aix.WithSessionStore[any](store))
	return &session{r: r, agent: agent}
}

// send runs input as one turn on a fresh invocation, resuming the session
// after its first turn, and returns what the turn produced. It fails t only
// when the invocation itself breaks; the turn's outcome is the caller's to
// check.
func (s *session) send(t *testing.T, ctx context.Context, input *aix.AgentInput) *turn {
	t.Helper()
	var opts []aix.InvocationOption[any]
	if s.id != "" {
		opts = append(opts, aix.WithSessionID[any](s.id))
	}
	conn, err := s.agent.Connect(ctx, opts...)
	if err != nil {
		t.Fatalf("Connect() error = %v", err)
	}
	if err := conn.Send(input); err != nil {
		t.Fatalf("Send() error = %v", err)
	}
	res := &turn{}
	var text, reasoning strings.Builder
	for chunk, err := range conn.Receive() {
		if err != nil {
			break // the invocation's outcome is on Output
		}
		if chunk.ModelChunk != nil {
			res.chunks++
			text.WriteString(chunk.ModelChunk.Text())
			reasoning.WriteString(chunk.ModelChunk.Reasoning())
		}
		if chunk.TurnEnd != nil {
			res.end = chunk.TurnEnd
			break
		}
	}
	res.text, res.reasoning = text.String(), reasoning.String()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output() error = %v", err)
	}
	res.out = out
	if s.id == "" {
		s.id = out.SessionID
	} else if out.SessionID != s.id {
		t.Errorf("SessionID = %q, want the resumed %q", out.SessionID, s.id)
	}
	return res
}

// ask runs a user message as a turn that must complete, streamed, with a
// reply containing every one of want.
func (s *session) ask(t *testing.T, msg *ai.Message, want ...string) *turn {
	t.Helper()
	res := s.send(t, s.r.ctx, &aix.AgentInput{Message: msg})
	res.wantCompleted(t)
	res.wantReply(t, want...)
	return res
}

// askText is ask with a text message.
func (s *session) askText(t *testing.T, text string, want ...string) *turn {
	t.Helper()
	return s.ask(t, ai.NewUserTextMessage(text), want...)
}

// wantCompleted fails t unless the turn ended normally and streamed.
func (res *turn) wantCompleted(t *testing.T) {
	t.Helper()
	if res.out.FinishReason != aix.AgentFinishReasonStop {
		t.Fatalf("FinishReason = %q, want %q (error %v, reply %q)", res.out.FinishReason, aix.AgentFinishReasonStop, res.out.Error, res.reply())
	}
	if res.chunks == 0 {
		t.Error("the turn streamed no model chunks")
	}
	if res.end == nil {
		t.Error("the turn streamed no turn end")
	}
}

// wantReply fails t unless the turn's reply contains every one of want,
// ignoring case.
func (res *turn) wantReply(t *testing.T, want ...string) {
	t.Helper()
	for _, w := range want {
		if !containsFold(res.reply(), w) {
			t.Errorf("reply = %q (finish reason %q), want it to contain %q", res.reply(), res.out.FinishReason, w)
		}
	}
}

// reply is the text of the turn's last model message.
func (res *turn) reply() string {
	if res.out == nil || res.out.Message == nil {
		return ""
	}
	return res.out.Message.Text()
}

// interrupt returns the single interrupt the turn stopped on.
func (res *turn) interrupt(t *testing.T) *ai.Part {
	t.Helper()
	if res.out.FinishReason != aix.AgentFinishReasonInterrupted {
		t.Fatalf("FinishReason = %q, want %q (error %v, reply %q)", res.out.FinishReason, aix.AgentFinishReasonInterrupted, res.out.Error, res.reply())
	}
	var interrupts []*ai.Part
	if res.out.Message != nil {
		for _, p := range res.out.Message.Content {
			if p.IsInterrupt() {
				interrupts = append(interrupts, p)
			}
		}
	}
	if len(interrupts) != 1 {
		t.Fatalf("interrupts = %d parts, want the one transfer", len(interrupts))
	}
	return interrupts[0]
}

// history returns the conversation the store holds for the session.
func (s *session) history(t *testing.T) []*ai.Message {
	t.Helper()
	snap, err := s.agent.GetLatestSnapshot(s.r.ctx, s.id)
	if err != nil {
		t.Fatalf("GetLatestSnapshot() error = %v", err)
	}
	if snap.State == nil {
		return nil
	}
	return snap.State.Messages
}

// snapStatus is snap's status, or "" for a nil snapshot.
func snapStatus(snap *aix.SessionSnapshot[any]) aix.SnapshotStatus {
	if snap == nil {
		return ""
	}
	return snap.Status
}

// snapReply is the text of the last message snap holds, or "".
func snapReply(snap *aix.SessionSnapshot[any]) string {
	if snap == nil || snap.State == nil || len(snap.State.Messages) == 0 {
		return ""
	}
	return snap.State.Messages[len(snap.State.Messages)-1].Text()
}

func agentCases() []liveCase {
	return []liveCase{
		{"multi-turn chat", always, func(t *testing.T, r *runner) {
			s := r.newSession(t, r.s.Model)
			s.askText(t, "My name is Zebulon Quixote and I live in Reykjavik. Reply with just OK.")
			s.askText(t, "What is my name? Reply with just the name.", "zebulon")
			s.askText(t, "Which city do I live in? Reply with just the city.", "reykjavik")
		}},
		{"tool calls across turns", needTools, func(t *testing.T, r *runner) {
			s := r.newSession(t, r.s.Model, ai.WithTools(r.tools.gablorken))
			s.askText(t, "Use the gablorken tool with value 4 and over 2. Reply with just the number.", "17")
			s.askText(t, "Call the gablorken tool exactly once, with value 17 and over 2. Reply with just the number it returns.", "290")
			s.askText(t, "Now get two gablorkens at once: value 2 over 3, and value 3 over 2. Reply with both numbers.", "9", "10")
			s.askText(t, "List every gablorken result so far, separated by commas.", "17", "290", "9", "10")
		}},
		{"interrupts across turns", needTools, func(t *testing.T, r *runner) {
			s := r.newSession(t, r.s.Model, ai.WithTools(r.tools.transfer, r.tools.gablorken))

			first := s.send(t, r.ctx, &aix.AgentInput{Message: ai.NewUserTextMessage(
				transferPrompt)})
			restart, err := r.tools.transfer.RestartWith(first.interrupt(t),
				ai.WithResumedMetadata[transferInput](map[string]any{"approved": true}))
			if err != nil {
				t.Fatalf("RestartWith() error = %v", err)
			}
			approved := s.send(t, r.ctx, &aix.AgentInput{Resume: &aix.ToolResume{Restart: []*ai.Part{restart}}})
			approved.wantCompleted(t)
			approved.wantReply(t, transferCode)

			second := s.send(t, r.ctx, &aix.AgentInput{Message: ai.NewUserTextMessage(
				"Now send 20 dollars to Grace with the transferFunds tool and tell me whether it went through.")})
			respond, err := r.tools.transfer.RespondWith(second.interrupt(t),
				transferResult{Status: "rejected by the account owner"})
			if err != nil {
				t.Fatalf("RespondWith() error = %v", err)
			}
			rejected := s.send(t, r.ctx, &aix.AgentInput{Resume: &aix.ToolResume{Respond: []*ai.Part{respond}}})
			rejected.wantCompleted(t)
			if !containsAnyFold(rejected.reply(), "reject", "declin", "denied", "not go through", "did not", "didn't") {
				t.Errorf("reply = %q, want it to report the rejected transfer", rejected.reply())
			}

			s.askText(t, "Use the gablorken tool with value 4 and over 2. Reply with just the number.", "17")
			s.askText(t, "What was the confirmation code of the transfer to Ada?", transferCode)
		}},
		{"reasoning across turns", needReasoning, func(t *testing.T, r *runner) {
			var opts []ai.PromptOption
			if r.reasoningCaps.Tools {
				opts = append(opts, ai.WithTools(r.tools.gablorken))
			}
			s := r.newSession(t, r.s.ReasoningModel, opts...)
			turns := []*turn{s.askText(t, "Is 91 a prime number? Answer yes or no.", "no")}
			if len(opts) > 0 {
				turns = append(turns,
					s.askText(t, "Use the gablorken tool with value 3 and over 2. Reply with just the number.", "10"),
					s.askText(t, "Call the gablorken tool exactly once, with value 10 and over 2. Reply with just the number it returns.", "101"),
					s.askText(t, "What were the two gablorken results? Reply with both numbers.", "10", "101"))
			} else {
				turns = append(turns, s.askText(t, "Is 97 a prime number? Answer yes or no.", "yes"))
			}
			if r.s.ReasoningContent {
				for i, res := range turns {
					if res.reasoning == "" {
						t.Errorf("turn %d streamed no reasoning, want the thinking content", i+1)
					}
				}
				if !hasReasoning(s.history(t)) {
					t.Error("the stored history holds no reasoning, want the thinking kept for later turns")
				}
			}
		}},
		{"media across turns", needVision, func(t *testing.T, r *runner) {
			caps := r.visionCaps
			var opts []ai.PromptOption
			if caps.Tools {
				tools := []ai.ToolRef{r.tools.gablorken}
				if r.s.ToolResponseMedia {
					tools = append(tools, r.tools.swatch)
				}
				opts = append(opts, ai.WithTools(tools...))
			}
			s := r.newSession(t, r.vision, opts...)
			s.ask(t, ai.NewUserMessage(
				ai.NewMediaPart("image/png", RedImage),
				ai.NewTextPart("Remember this image. Reply with just OK.")))
			s.askText(t, "What is the dominant color of the image I sent? Reply with one word.", "red")
			if !caps.Tools {
				return
			}
			s.askText(t, "Use the gablorken tool with value 4 and over 2. Reply with just the number.", "17")
			if r.s.ToolResponseMedia {
				s.askText(t, "Fetch the swatch named primary with the fetchSwatch tool and tell me its color in one word.", "red")
				s.askText(t, "What color was the swatch? Reply with one word.", "red")
			}
		}},
		{"structured output across turns", always, func(t *testing.T, r *runner) {
			s := r.newSession(t, r.s.Model, ai.WithOutputType(capitalFacts{}))
			for _, c := range []struct{ prompt, city string }{
				{"What is the capital of France?", "paris"},
				{"And the capital of Japan?", "tokyo"},
				{"What was the first country I asked about? Give its capital.", "paris"},
			} {
				res := s.askText(t, c.prompt)
				var facts capitalFacts
				if err := json.Unmarshal([]byte(res.reply()), &facts); err != nil {
					t.Fatalf("reply %q is not the structured output: %v", res.reply(), err)
				}
				wantText(t, facts.City, c.city)
			}
		}},
		{"abort during a tool call", needTools, func(t *testing.T, r *runner) {
			s := r.newSession(t, r.s.Model, ai.WithTools(r.tools.gablorken, r.tools.diagnostics))
			s.askText(t, "Use the gablorken tool with value 4 and over 2. Reply with just the number.", "17")

			select {
			case <-r.tools.started:
			default:
			}
			task, err := s.agent.RunDetached(r.ctx, &aix.AgentInput{Message: ai.NewUserTextMessage(
				`Call the runDiagnostics tool with target "database" and tell me what it reports.`)}, aix.WithSessionID[any](s.id))
			if err != nil {
				t.Fatalf("RunDetached() error = %v", err)
			}
			// A case that fails before its own abort must not leave the turn
			// running into the cases after it, or into the session store's
			// removed directory.
			defer func() {
				ctx, cancel := context.WithTimeout(context.WithoutCancel(r.ctx), time.Minute)
				defer cancel()
				task.Abort(ctx)
				task.Wait(ctx)
			}()
			// A model that answers without the tool settles the task, so
			// waiting on it reports that rather than a timeout.
			settled := make(chan *aix.SessionSnapshot[any], 1)
			go func() {
				snap, _ := task.Wait(r.ctx)
				settled <- snap
			}()
			select {
			case <-r.tools.started:
			case snap := <-settled:
				t.Fatalf("the turn settled (status %q) without calling the diagnostics tool, replying %q", snapStatus(snap), snapReply(snap))
			case <-time.After(2 * time.Minute):
				t.Fatal("the model never called the diagnostics tool")
			}
			if _, err := task.Abort(r.ctx); err != nil {
				t.Fatalf("Abort() error = %v", err)
			}
			waitCtx, cancel := context.WithTimeout(r.ctx, 2*time.Minute)
			defer cancel()
			snap, err := task.Wait(waitCtx)
			if err != nil {
				t.Fatalf("Wait() error = %v", err)
			}
			if snap.Status != aix.SnapshotStatusAborted {
				t.Fatalf("Status = %q, want %q", snap.Status, aix.SnapshotStatusAborted)
			}

			// The aborted turn left a tool request nothing answered; the
			// conversation must carry on without it.
			s.askText(t, "Forget the diagnostics. Use the gablorken tool with value 3 and over 2. Reply with just the number.", "10")
			s.askText(t, "What was the first gablorken result in this conversation?", "17")
		}},
		{"abort while streaming", always, func(t *testing.T, r *runner) {
			s := r.newSession(t, r.s.Model)
			s.askText(t, "My favorite animal is the quokka. Reply with just OK.")

			ctx, cancel := context.WithCancel(r.ctx)
			defer cancel()
			conn, err := s.agent.Connect(ctx, aix.WithSessionID[any](s.id))
			if err != nil {
				t.Fatalf("Connect() error = %v", err)
			}
			if err := conn.SendText("Count from 1 to 500, one number per line, with no other text."); err != nil {
				t.Fatalf("SendText() error = %v", err)
			}
			streamed := false
			for chunk, err := range conn.Receive() {
				if err != nil {
					break
				}
				if !streamed && chunk.ModelChunk != nil && chunk.ModelChunk.Text() != "" {
					streamed = true
					cancel()
				}
			}
			if !streamed {
				t.Fatal("the turn ended before any text streamed")
			}
			if out, err := conn.Output(); err == nil && out.FinishReason != aix.AgentFinishReasonAborted {
				t.Errorf("FinishReason = %q, want %q after the caller hung up", out.FinishReason, aix.AgentFinishReasonAborted)
			}

			s.askText(t, "Stop counting. What is my favorite animal? Reply with just the animal.", "quokka")
		}},
		{"failed turn then retry", needTools, func(t *testing.T, r *runner) {
			s := r.newSession(t, r.s.Model, ai.WithTools(r.tools.lookupOrder))
			s.askText(t, "My name is Ada Lovelace. Reply with just OK.")

			r.tools.failLookups.Store(1)
			defer r.tools.failLookups.Store(0)
			failed := s.send(t, r.ctx, &aix.AgentInput{Message: ai.NewUserTextMessage(
				"Call the lookupOrder tool for order 1234 and tell me its carrier.")})
			if r.tools.failLookups.Load() > 0 {
				t.Fatalf("the model answered without calling lookupOrder (reply %q)", failed.reply())
			}
			if failed.out.FinishReason != aix.AgentFinishReasonFailed || failed.out.Error == nil {
				t.Fatalf("FinishReason = %q (error %v), want the failed tool to fail the turn", failed.out.FinishReason, failed.out.Error)
			}

			// An input with no payload re-attempts the failed turn from what
			// it committed.
			retry := s.send(t, r.ctx, &aix.AgentInput{})
			retry.wantCompleted(t)
			retry.wantReply(t, "quokka")
			s.askText(t, "What is my name, and which order number did we look up?", "ada", "1234")
		}},
	}
}
