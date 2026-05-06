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
//
// SPDX-License-Identifier: Apache-2.0

// Package exp provides experimental AI primitives for Genkit.
//
// APIs in this package are under active development and may change in any
// minor version release.
package exp

import (
	"context"
	"fmt"
	"iter"
	"runtime/debug"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/google/uuid"
)

// --- AgentSession ---

// AgentSession extends Session with agent-runtime functionality:
// turn management, snapshot persistence, and input channel handling.
type AgentSession[State any] struct {
	*Session[State]

	// InputCh is the channel that delivers per-turn inputs from the client.
	// It is consumed automatically by [AgentSession.Run], but is exposed
	// for advanced use cases that need direct access to the input stream
	// (e.g., custom turn loops or fan-out patterns).
	InputCh <-chan *AgentInput
	// TurnIndex is the zero-based index of the current conversation turn.
	// It is incremented automatically by [AgentSession.Run], but is exposed
	// for advanced use cases that need to track or manipulate turn ordering
	// directly.
	TurnIndex int

	snapshotCallback    SnapshotCallback[State]
	onEndTurn           func(ctx context.Context)
	lastSnapshot        *SessionSnapshot[State]
	lastSnapshotVersion uint64
	collectTurnOutput   func() any
}

// Run loops over the input channel, calling fn for each turn. Each turn is
// wrapped in a trace span for observability. Input messages are automatically
// added to the session before fn is called. After fn returns successfully, a
// TurnEnd chunk is sent and a snapshot check is triggered.
func (s *AgentSession[State]) Run(ctx context.Context, fn func(ctx context.Context, input *AgentInput) error) error {
	for input := range s.InputCh {
		spanMeta := &tracing.SpanMetadata{
			Name:    fmt.Sprintf("agent/turn/%d", s.TurnIndex),
			Type:    "flowStep",
			Subtype: "flowStep",
		}

		_, err := tracing.RunInNewSpan(ctx, spanMeta, input,
			func(ctx context.Context, input *AgentInput) (any, error) {
				s.AddMessages(input.Messages...)

				if err := fn(ctx, input); err != nil {
					return nil, err
				}

				s.onEndTurn(ctx)
				s.TurnIndex++

				if s.collectTurnOutput != nil {
					return s.collectTurnOutput(), nil
				}
				return nil, nil
			},
		)
		if err != nil {
			return err
		}
	}
	return nil
}

// Result returns an [AgentResult] populated from the current session state:
// the last message in the conversation history and all artifacts.
// It is a convenience for custom agents that don't need to construct the
// result manually.
func (s *AgentSession[State]) Result() *AgentResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := &AgentResult{}
	if msgs := s.state.Messages; len(msgs) > 0 {
		result.Message = msgs[len(msgs)-1]
	}
	if len(s.state.Artifacts) > 0 {
		arts := make([]*Artifact, len(s.state.Artifacts))
		copy(arts, s.state.Artifacts)
		result.Artifacts = arts
	}
	return result
}

// maybeSnapshot creates a snapshot if conditions are met (store configured,
// callback approves, state changed). Returns the snapshot ID or empty string.
func (s *AgentSession[State]) maybeSnapshot(ctx context.Context, event SnapshotEvent) string {
	if s.store == nil {
		return ""
	}

	s.mu.RLock()
	currentVersion := s.version
	currentState := s.copyStateLocked()
	s.mu.RUnlock()

	// Skip if state hasn't changed since the last snapshot. This avoids
	// redundant snapshots, e.g. the invocation-end snapshot after a
	// single-turn Run where the turn-end snapshot already captured the
	// same state.
	if s.lastSnapshot != nil && currentVersion == s.lastSnapshotVersion {
		return ""
	}

	if s.snapshotCallback != nil {
		var prevState *SessionState[State]
		if s.lastSnapshot != nil {
			prevState = &s.lastSnapshot.State
		}
		if !s.snapshotCallback(ctx, &SnapshotContext[State]{
			State:     &currentState,
			PrevState: prevState,
			TurnIndex: s.TurnIndex,
			Event:     event,
		}) {
			return ""
		}
	}

	snapshot := &SessionSnapshot[State]{
		SnapshotID: uuid.New().String(),
		CreatedAt:  time.Now(),
		Event:      event,
		State:      currentState,
	}
	if s.lastSnapshot != nil {
		snapshot.ParentID = s.lastSnapshot.SnapshotID
	}

	if err := s.store.SaveSnapshot(ctx, snapshot); err != nil {
		logger.FromContext(ctx).Error("agent: failed to save snapshot", "err", err)
		return ""
	}

	// Set snapshotId in last message metadata.
	s.mu.Lock()
	if msgs := s.state.Messages; len(msgs) > 0 {
		lastMsg := msgs[len(msgs)-1]
		if lastMsg.Metadata == nil {
			lastMsg.Metadata = make(map[string]any)
		}
		lastMsg.Metadata["snapshotId"] = snapshot.SnapshotID
	}
	s.mu.Unlock()

	s.lastSnapshot = snapshot
	s.lastSnapshotVersion = currentVersion

	return snapshot.SnapshotID
}

// --- Responder ---

// Responder is the output channel for an agent. Artifacts sent through
// it are automatically added to the session before being forwarded to the
// client.
type Responder[Stream any] chan<- *AgentStreamChunk[Stream]

// SendModelChunk sends a generation chunk (token-level streaming).
func (r Responder[Stream]) SendModelChunk(chunk *ai.ModelResponseChunk) {
	r <- &AgentStreamChunk[Stream]{ModelChunk: chunk}
}

// SendStatus sends a user-defined status update.
func (r Responder[Stream]) SendStatus(status Stream) {
	r <- &AgentStreamChunk[Stream]{Status: status}
}

// SendArtifact sends an artifact to the stream and adds it to the session.
// If an artifact with the same name already exists in the session, it is replaced.
func (r Responder[Stream]) SendArtifact(artifact *Artifact) {
	r <- &AgentStreamChunk[Stream]{Artifact: artifact}
}

// --- Agent ---

// AgentFunc is the function signature for custom agents.
// Type parameters:
//   - Stream: Type for status updates sent via the responder
//   - State: Type for user-defined state in snapshots
type AgentFunc[Stream, State any] = func(ctx context.Context, resp Responder[Stream], sess *AgentSession[State]) (*AgentResult, error)

// Agent is a bidirectional streaming agent with automatic snapshot management.
type Agent[Stream, State any] struct {
	flow *core.Flow[*AgentInit[State], *AgentOutput[State], *AgentStreamChunk[Stream], *AgentInput]
}

// DefineAgent defines an agent that wraps a prompt defined inline from the
// given options, and registers both under name. Each turn renders the prompt,
// appends conversation history, calls the model with streaming, and updates
// session state.
//
// opts is a mixed list of [github.com/firebase/genkit/go/ai.PromptOption]
// values (which configure the prompt) and [AgentOption] values (which
// configure the agent itself, e.g., [WithSessionStore]).
//
// State is phantom in the variadic, so it cannot be inferred. Specify [any]
// when no typed Custom state is needed; specify [Foo] when a
// [SessionStore[Foo]] is provided. A mismatch panics at definition time with
// a clear message.
//
// For an agent backed by an existing prompt, use [DefinePromptAgent]. For
// full control over the per-turn loop, use [DefineCustomAgent].
func DefineAgent[State any](
	r api.Registry,
	name string,
	opts ...AgentDefineOption[State],
) *Agent[any, State] {
	var promptOpts []ai.PromptOption
	var agentOpts []AgentOption[State]
	for _, opt := range opts {
		if ao, ok := opt.(AgentOption[State]); ok {
			agentOpts = append(agentOpts, ao)
			continue
		}
		if po, ok := opt.(ai.PromptOption); ok {
			promptOpts = append(promptOpts, po)
			continue
		}
		panic(fmt.Sprintf("DefineAgent %q: option of type %T does not match agent State %T (likely a typed AgentOption with a different State than the one declared on DefineAgent)", name, opt, *new(State)))
	}

	prompt := ai.DefinePrompt(r, name, promptOpts...)
	return DefineCustomAgent(r, name, agentLoop[State](r, prompt, nil), agentOpts...)
}

// DefinePromptAgent defines an agent backed by a prompt already registered
// with the registry (via [ai.DefinePrompt] or loaded from a .prompt file).
// The agent is registered under the same name as the prompt, sharing its
// namespace.
//
// defaultInput is used to render the prompt on every turn. PromptIn is
// captured for compile-time type checking on defaultInput; it is not
// propagated through the [Agent] type.
//
// For an agent that defines its prompt inline, use [DefineAgent]. For full
// control over the per-turn loop, use [DefineCustomAgent].
func DefinePromptAgent[State, PromptIn any](
	r api.Registry,
	promptName string,
	defaultInput PromptIn,
	opts ...AgentOption[State],
) *Agent[any, State] {
	prompt := ai.LookupPrompt(r, promptName)
	if prompt == nil {
		panic(fmt.Sprintf("DefinePromptAgent: prompt %q not found", promptName))
	}
	return DefineCustomAgent(r, promptName, agentLoop[State](r, prompt, defaultInput), opts...)
}

// DefineCustomAgent defines an agent with full control over the conversation
// loop and registers it with the registry.
func DefineCustomAgent[Stream, State any](
	r api.Registry,
	name string,
	fn AgentFunc[Stream, State],
	opts ...AgentOption[State],
) *Agent[Stream, State] {
	agOpts := &agentOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyAgent(agOpts); err != nil {
			panic(fmt.Errorf("DefineCustomAgent %q: %w", name, err))
		}
	}

	store := agOpts.store
	snapshotCallback := agOpts.callback

	flow := core.DefineBidiFlow(r, name, func(
		ctx context.Context,
		in *AgentInit[State],
		inCh <-chan *AgentInput,
		outCh chan<- *AgentStreamChunk[Stream],
	) (*AgentOutput[State], error) {
		session, snapshot, err := newSessionFromInit(ctx, in, store)
		if err != nil {
			return nil, err
		}
		ctx = NewSessionContext(ctx, session)

		agentSess := &AgentSession[State]{
			Session:          session,
			snapshotCallback: snapshotCallback,
			InputCh:          inCh,
			lastSnapshot:     snapshot,
		}

		var (
			turnMu     sync.Mutex
			turnChunks []*AgentStreamChunk[Stream]
		)

		agentSess.collectTurnOutput = func() any {
			turnMu.Lock()
			defer turnMu.Unlock()
			result := turnChunks
			turnChunks = nil
			return result
		}

		respCh := make(chan *AgentStreamChunk[Stream])
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for chunk := range respCh {
				if chunk.Artifact != nil {
					session.AddArtifacts(chunk.Artifact)
				}
				if chunk.TurnEnd == nil {
					turnMu.Lock()
					turnChunks = append(turnChunks, chunk)
					turnMu.Unlock()
				}
				// Once ctx is cancelled, downstream is gone but fn may
				// still be mid-flight. Keep draining respCh so fn doesn't
				// deadlock on its next send; just stop forwarding.
				if ctx.Err() != nil {
					continue
				}
				select {
				case outCh <- chunk:
				case <-ctx.Done():
				}
			}
		}()

		// Writes through respCh (not outCh) so the TurnEnd signal stays
		// ordered after any user chunks emitted during the turn.
		agentSess.onEndTurn = func(turnCtx context.Context) {
			snapshotID := agentSess.maybeSnapshot(turnCtx, SnapshotEventTurnEnd)
			respCh <- &AgentStreamChunk[Stream]{
				TurnEnd: &TurnEnd{SnapshotID: snapshotID},
			}
		}

		// Run fn under deferred cleanup so the streaming goroutine is
		// always released, even if fn panics. The panic is recovered and
		// surfaced as an error rather than crashing the process.
		var (
			result *AgentResult
			fnErr  error
		)
		func() {
			defer wg.Wait()
			defer close(respCh)
			defer func() {
				if r := recover(); r != nil {
					logger.FromContext(ctx).Error("agent fn panicked", "panic", r, "stack", string(debug.Stack()))
					fnErr = core.NewError(core.INTERNAL, "agent fn panicked: %v", r)
				}
			}()
			result, fnErr = fn(ctx, Responder[Stream](respCh), agentSess)
		}()

		if fnErr != nil {
			return nil, fnErr
		}

		// Final snapshot at invocation end. If skipped (state unchanged
		// since last turn-end snapshot), use the last snapshot's ID so
		// the output always reflects the latest snapshot.
		snapshotID := agentSess.maybeSnapshot(ctx, SnapshotEventInvocationEnd)
		if snapshotID == "" && agentSess.lastSnapshot != nil {
			snapshotID = agentSess.lastSnapshot.SnapshotID
		}

		out := &AgentOutput[State]{
			SnapshotID: snapshotID,
		}
		if result != nil {
			out.Message = result.Message
			out.Artifacts = result.Artifacts
		}

		// Only include full state when client-managed (no store).
		if store == nil {
			out.State = session.State()
		}

		return out, nil
	})

	return &Agent[Stream, State]{flow: flow}
}

// promptMessageKey is the metadata key used to tag base messages from the
// agent config (system prompt, prompt template output, etc.) so they can be
// excluded from session history after generation.
const promptMessageKey = "_genkit_prompt"

// agentLoop returns the per-turn function for a prompt-backed agent. Each
// turn renders the prompt, appends conversation history, calls the model
// with streaming, and updates the session.
//
// defaultInput is the prompt input passed to Render on every turn. It is
// nil for [DefineAgent], where the inline-defined prompt has no per-turn
// input.
func agentLoop[State any](r api.Registry, prompt ai.Prompt, defaultInput any) AgentFunc[any, State] {
	return func(ctx context.Context, resp Responder[any], sess *AgentSession[State]) (*AgentResult, error) {
		if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
			actionOpts, err := prompt.Render(ctx, defaultInput)
			if err != nil {
				return fmt.Errorf("prompt render: %w", err)
			}

			// Tag base messages so they can be filtered out of session
			// history after generation.
			for _, m := range actionOpts.Messages {
				if m.Metadata == nil {
					m.Metadata = make(map[string]any)
				}
				m.Metadata[promptMessageKey] = true
			}

			// Append conversation history after the base messages.
			actionOpts.Messages = append(actionOpts.Messages, sess.Messages()...)

			// If tool restarts were provided, set the resume option so
			// handleResumeOption re-executes the interrupted tools.
			if len(input.ToolRestarts) > 0 {
				for _, p := range input.ToolRestarts {
					if !p.IsToolRequest() {
						return core.NewError(core.INVALID_ARGUMENT, "ToolRestarts: part is not a tool request")
					}
				}
				actionOpts.Resume = ai.NewResume(input.ToolRestarts, nil)
			}

			modelResp, err := ai.GenerateWithRequest(ctx, r, actionOpts, nil,
				func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					resp.SendModelChunk(chunk)
					return nil
				},
			)
			if err != nil {
				return fmt.Errorf("generate: %w", err)
			}

			// Replace session messages with the full history minus base
			// messages. This captures intermediate tool call/response
			// messages from the tool loop, not just the final response.
			if modelResp.Request != nil {
				history := modelResp.History()
				msgs := make([]*ai.Message, 0, len(history))
				for _, m := range history {
					if m.Metadata != nil && m.Metadata[promptMessageKey] == true {
						continue
					}
					msgs = append(msgs, m)
				}
				sess.SetMessages(msgs)
			} else if modelResp.Message != nil {
				sess.AddMessages(modelResp.Message)
			}

			// Stream interrupt parts so the client can detect and
			// handle them (e.g. prompt the user for confirmation).
			if modelResp.FinishReason == ai.FinishReasonInterrupted {
				if parts := modelResp.Interrupts(); len(parts) > 0 {
					resp.SendModelChunk(&ai.ModelResponseChunk{
						Role:    ai.RoleTool,
						Content: parts,
					})
				}
			}

			return nil
		}); err != nil {
			return nil, err
		}
		return sess.Result(), nil
	}
}

// StreamBidi starts a new agent invocation with bidirectional streaming.
// Use this for multi-turn interactions where you need to send multiple inputs
// and receive streaming chunks. For single-turn usage, see Run and RunText.
func (a *Agent[Stream, State]) StreamBidi(
	ctx context.Context,
	opts ...InvocationOption[State],
) (*AgentConnection[Stream, State], error) {
	invOpts, err := a.resolveOptions(opts)
	if err != nil {
		return nil, err
	}

	conn, err := a.flow.StreamBidi(ctx, invOpts)
	if err != nil {
		return nil, err
	}

	return &AgentConnection[Stream, State]{conn: conn}, nil
}

// Run starts a single-turn agent invocation with the given input.
// It sends the input, waits for the agent to complete, and returns the output.
// For multi-turn interactions or streaming, use StreamBidi instead.
func (a *Agent[Stream, State]) Run(
	ctx context.Context,
	input *AgentInput,
	opts ...InvocationOption[State],
) (*AgentOutput[State], error) {
	conn, err := a.StreamBidi(ctx, opts...)
	if err != nil {
		return nil, err
	}

	if err := conn.Send(input); err != nil {
		return nil, err
	}
	if err := conn.Close(); err != nil {
		return nil, err
	}

	// Drain stream chunks.
	for _, err := range conn.Receive() {
		if err != nil {
			return nil, err
		}
	}

	return conn.Output()
}

// RunText is a convenience method that starts a single-turn agent invocation
// with a user text message. It is equivalent to calling Run with an
// AgentInput containing a single user text message.
func (a *Agent[Stream, State]) RunText(
	ctx context.Context,
	text string,
	opts ...InvocationOption[State],
) (*AgentOutput[State], error) {
	return a.Run(ctx, &AgentInput{
		Messages: []*ai.Message{ai.NewUserTextMessage(text)},
	}, opts...)
}

// resolveOptions applies invocation options and returns the init struct.
func (a *Agent[Stream, State]) resolveOptions(opts []InvocationOption[State]) (*AgentInit[State], error) {
	invOpts := &invocationOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyInvocation(invOpts); err != nil {
			return nil, fmt.Errorf("Agent %q: %w", a.flow.Name(), err)
		}
	}

	return &AgentInit[State]{
		SnapshotID: invOpts.snapshotID,
		State:      invOpts.state,
	}, nil
}

// newSessionFromInit creates a Session from initialization data.
// If resuming from a snapshot, the loaded snapshot is also returned.
func newSessionFromInit[State any](
	ctx context.Context,
	init *AgentInit[State],
	store SessionStore[State],
) (*Session[State], *SessionSnapshot[State], error) {
	s := &Session[State]{store: store}

	var snapshot *SessionSnapshot[State]
	if init != nil {
		if init.SnapshotID != "" && init.State != nil {
			return nil, nil, core.NewError(core.INVALID_ARGUMENT, "snapshot ID and state are mutually exclusive")
		}
		if init.SnapshotID != "" && store == nil {
			return nil, nil, core.NewError(core.FAILED_PRECONDITION, "snapshot ID %q provided but no session store configured", init.SnapshotID)
		}
		if init.SnapshotID != "" && store != nil {
			var err error
			snapshot, err = store.GetSnapshot(ctx, init.SnapshotID)
			if err != nil {
				return nil, nil, core.NewError(core.INTERNAL, "failed to load snapshot %q: %v", init.SnapshotID, err)
			}
			if snapshot == nil {
				return nil, nil, core.NewError(core.NOT_FOUND, "snapshot %q not found", init.SnapshotID)
			}
			s.state = snapshot.State
		} else if init.State != nil {
			s.state = *init.State
		}
	}

	return s, snapshot, nil
}

// --- AgentConnection ---

// AgentConnection wraps BidiConnection with agent-specific functionality.
// It provides a Receive() iterator that supports multi-turn patterns: breaking
// out of the iterator between turns does not cancel the underlying connection.
type AgentConnection[Stream, State any] struct {
	conn *core.BidiConnection[*AgentInput, *AgentStreamChunk[Stream], *AgentOutput[State]]

	// chunks buffers stream chunks from the underlying connection so that
	// breaking from Receive() between turns doesn't cancel the context.
	chunks   chan *AgentStreamChunk[Stream]
	chunkErr error
	initOnce sync.Once
}

// initReceiver starts a goroutine that drains the underlying BidiConnection's
// Receive into a channel. This goroutine never breaks from the underlying
// iterator, preventing context cancellation.
func (c *AgentConnection[Stream, State]) initReceiver() {
	c.initOnce.Do(func() {
		c.chunks = make(chan *AgentStreamChunk[Stream], 1)
		go func() {
			defer close(c.chunks)
			for chunk, err := range c.conn.Receive() {
				if err != nil {
					c.chunkErr = err
					return
				}
				c.chunks <- chunk
			}
		}()
	})
}

// Send sends an AgentInput to the agent.
func (c *AgentConnection[Stream, State]) Send(input *AgentInput) error {
	return c.conn.Send(input)
}

// SendMessages sends messages to the agent.
func (c *AgentConnection[Stream, State]) SendMessages(messages ...*ai.Message) error {
	return c.conn.Send(&AgentInput{Messages: messages})
}

// SendText sends a single user text message to the agent.
func (c *AgentConnection[Stream, State]) SendText(text string) error {
	return c.conn.Send(&AgentInput{
		Messages: []*ai.Message{ai.NewUserTextMessage(text)},
	})
}

// SendToolRestarts sends tool restart parts to resume interrupted tool calls.
// Parts should be created via [ai.ToolDef.RestartWith].
func (c *AgentConnection[Stream, State]) SendToolRestarts(parts ...*ai.Part) error {
	return c.conn.Send(&AgentInput{ToolRestarts: parts})
}

// Close signals that no more inputs will be sent.
func (c *AgentConnection[Stream, State]) Close() error {
	return c.conn.Close()
}

// Receive returns an iterator for receiving stream chunks.
// Unlike the underlying BidiConnection.Receive, breaking out of this iterator
// does not cancel the connection. This enables multi-turn patterns where the
// caller breaks on TurnEnd, sends the next input, then calls Receive again.
func (c *AgentConnection[Stream, State]) Receive() iter.Seq2[*AgentStreamChunk[Stream], error] {
	c.initReceiver()
	return func(yield func(*AgentStreamChunk[Stream], error) bool) {
		for {
			chunk, ok := <-c.chunks
			if !ok {
				if err := c.chunkErr; err != nil {
					yield(nil, err)
				}
				return
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

// Output returns the final response after the agent completes.
func (c *AgentConnection[Stream, State]) Output() (*AgentOutput[State], error) {
	return c.conn.Output()
}

// Done returns a channel closed when the connection completes.
func (c *AgentConnection[Stream, State]) Done() <-chan struct{} {
	return c.conn.Done()
}
