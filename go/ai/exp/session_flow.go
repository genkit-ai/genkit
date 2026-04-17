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
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/uuid"
)

// --- SessionRunner ---

// SessionRunner extends Session with session-flow-specific functionality:
// turn management, snapshot persistence, and input channel handling.
type SessionRunner[State any] struct {
	*Session[State]

	// InputCh is the channel that delivers per-turn inputs from the client.
	// It is consumed automatically by [SessionRunner.Run], but is exposed
	// for advanced use cases that need direct access to the input stream
	// (e.g., custom turn loops or fan-out patterns).
	InputCh <-chan *SessionFlowInput
	// TurnIndex is the zero-based index of the current conversation turn.
	// It is incremented automatically by [SessionRunner.Run], but is exposed
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
func (a *SessionRunner[State]) Run(ctx context.Context, fn func(ctx context.Context, input *SessionFlowInput) error) error {
	for input := range a.InputCh {
		spanMeta := &tracing.SpanMetadata{
			Name:    fmt.Sprintf("sessionFlow/turn/%d", a.TurnIndex),
			Type:    "flowStep",
			Subtype: "flowStep",
		}

		_, err := tracing.RunInNewSpan(ctx, spanMeta, input,
			func(ctx context.Context, input *SessionFlowInput) (any, error) {
				a.AddMessages(input.Messages...)

				if err := fn(ctx, input); err != nil {
					return nil, err
				}

				a.onEndTurn(ctx)
				a.TurnIndex++

				if a.collectTurnOutput != nil {
					return a.collectTurnOutput(), nil
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

// Result returns an [SessionFlowResult] populated from the current session state:
// the last message in the conversation history and all artifacts.
// It is a convenience for custom session flows that don't need to construct the
// result manually.
func (a *SessionRunner[State]) Result() *SessionFlowResult {
	a.mu.RLock()
	defer a.mu.RUnlock()

	result := &SessionFlowResult{}
	if msgs := a.state.Messages; len(msgs) > 0 {
		result.Message = msgs[len(msgs)-1]
	}
	if len(a.state.Artifacts) > 0 {
		arts := make([]*Artifact, len(a.state.Artifacts))
		copy(arts, a.state.Artifacts)
		result.Artifacts = arts
	}
	return result
}

// maybeSnapshot creates a snapshot if conditions are met (store configured,
// callback approves, state changed). Returns the snapshot ID or empty string.
func (a *SessionRunner[State]) maybeSnapshot(ctx context.Context, event SnapshotEvent) string {
	if a.store == nil {
		return ""
	}

	a.mu.RLock()
	currentVersion := a.version
	currentState := a.copyStateLocked()
	a.mu.RUnlock()

	// Skip if state hasn't changed since the last snapshot. This avoids
	// redundant snapshots, e.g. the invocation-end snapshot after a
	// single-turn Run where the turn-end snapshot already captured the
	// same state.
	if a.lastSnapshot != nil && currentVersion == a.lastSnapshotVersion {
		return ""
	}

	if a.snapshotCallback != nil {
		var prevState *SessionState[State]
		if a.lastSnapshot != nil {
			prevState = &a.lastSnapshot.State
		}
		if !a.snapshotCallback(ctx, &SnapshotContext[State]{
			State:     &currentState,
			PrevState: prevState,
			TurnIndex: a.TurnIndex,
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
	if a.lastSnapshot != nil {
		snapshot.ParentID = a.lastSnapshot.SnapshotID
	}

	if err := a.store.SaveSnapshot(ctx, snapshot); err != nil {
		logger.FromContext(ctx).Error("session flow: failed to save snapshot", "err", err)
		return ""
	}

	// Set snapshotId in last message metadata.
	a.mu.Lock()
	if msgs := a.state.Messages; len(msgs) > 0 {
		lastMsg := msgs[len(msgs)-1]
		if lastMsg.Metadata == nil {
			lastMsg.Metadata = make(map[string]any)
		}
		lastMsg.Metadata["snapshotId"] = snapshot.SnapshotID
	}
	a.mu.Unlock()

	a.lastSnapshot = snapshot
	a.lastSnapshotVersion = currentVersion

	return snapshot.SnapshotID
}

// --- Responder ---

// Responder is the output channel for an session flow. Artifacts sent through
// it are automatically added to the session before being forwarded to the
// client.
type Responder[Stream any] chan<- *SessionFlowStreamChunk[Stream]

// SendModelChunk sends a generation chunk (token-level streaming).
func (r Responder[Stream]) SendModelChunk(chunk *ai.ModelResponseChunk) {
	r <- &SessionFlowStreamChunk[Stream]{ModelChunk: chunk}
}

// SendStatus sends a user-defined status update.
func (r Responder[Stream]) SendStatus(status Stream) {
	r <- &SessionFlowStreamChunk[Stream]{Status: status}
}

// SendArtifact sends an artifact to the stream and adds it to the session.
// If an artifact with the same name already exists in the session, it is replaced.
func (r Responder[Stream]) SendArtifact(artifact *Artifact) {
	r <- &SessionFlowStreamChunk[Stream]{Artifact: artifact}
}

// --- SessionFlow ---

// SessionFlowFunc is the function signature for session flows.
// Type parameters:
//   - Stream: Type for status updates sent via the responder
//   - State: Type for user-defined state in snapshots
type SessionFlowFunc[Stream, State any] = func(ctx context.Context, resp Responder[Stream], sess *SessionRunner[State]) (*SessionFlowResult, error)

// SessionFlow is a bidirectional streaming flow with automatic snapshot management.
type SessionFlow[Stream, State any] struct {
	flow *core.Flow[*SessionFlowInit[State], *SessionFlowOutput[State], *SessionFlowStreamChunk[Stream], *SessionFlowInput]
}

// DefineSessionFlow creates an SessionFlow with automatic snapshot management and registers it.
func DefineSessionFlow[Stream, State any](
	r api.Registry,
	name string,
	fn SessionFlowFunc[Stream, State],
	opts ...SessionFlowOption[State],
) *SessionFlow[Stream, State] {
	afOpts := &sessionFlowOptions[State]{}
	for _, opt := range opts {
		if err := opt.applySessionFlow(afOpts); err != nil {
			panic(fmt.Errorf("DefineSessionFlow %q: %w", name, err))
		}
	}

	store := afOpts.store
	snapshotCallback := afOpts.callback

	flow := core.DefineBidiFlow(r, name, func(
		ctx context.Context,
		in *SessionFlowInit[State],
		inCh <-chan *SessionFlowInput,
		outCh chan<- *SessionFlowStreamChunk[Stream],
	) (*SessionFlowOutput[State], error) {
		session, snapshot, err := newSessionFromInit(ctx, in, store)
		if err != nil {
			return nil, err
		}
		ctx = NewSessionContext(ctx, session)

		agentSess := &SessionRunner[State]{
			Session:          session,
			snapshotCallback: snapshotCallback,
			InputCh:          inCh,
			lastSnapshot:     snapshot,
		}

		// Turn output accumulator: collects content chunks per turn for span output.
		var (
			turnMu     sync.Mutex
			turnChunks []*SessionFlowStreamChunk[Stream]
		)

		agentSess.collectTurnOutput = func() any {
			turnMu.Lock()
			defer turnMu.Unlock()
			result := turnChunks
			turnChunks = nil
			return result
		}

		// Intermediary channel: intercepts artifacts, accumulates turn output,
		// and forwards to outCh.
		respCh := make(chan *SessionFlowStreamChunk[Stream])
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for chunk := range respCh {
				if chunk.Artifact != nil {
					session.AddArtifacts(chunk.Artifact)
				}
				// Accumulate content chunks (exclude the TurnEnd control signal).
				if chunk.TurnEnd == nil {
					turnMu.Lock()
					turnChunks = append(turnChunks, chunk)
					turnMu.Unlock()
				}
				outCh <- chunk
			}
		}()

		// Wire up onEndTurn: triggers snapshot + sends TurnEnd chunk.
		// Writes through respCh to preserve ordering with user chunks.
		agentSess.onEndTurn = func(turnCtx context.Context) {
			snapshotID := agentSess.maybeSnapshot(turnCtx, SnapshotEventTurnEnd)
			respCh <- &SessionFlowStreamChunk[Stream]{
				TurnEnd: &TurnEnd{SnapshotID: snapshotID},
			}
		}

		result, fnErr := fn(ctx, Responder[Stream](respCh), agentSess)
		close(respCh)
		wg.Wait()

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

		out := &SessionFlowOutput[State]{
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

	return &SessionFlow[Stream, State]{flow: flow}
}

// promptMessageKey is the metadata key used to tag prompt-rendered messages
// so they can be excluded from session history after generation.
const promptMessageKey = "_genkit_prompt"

// DefineSessionFlowFromPrompt creates a prompt-backed SessionFlow with an
// automatic conversation loop. Each turn renders the prompt, appends
// conversation history, calls GenerateWithRequest, streams chunks to the
// client, and adds the model response to the session.
//
// The prompt is looked up by name from the registry using
// [ai.LookupDataPrompt]. The defaultInput is used for prompt rendering
// unless overridden per invocation via WithInputVariables.
func DefineSessionFlowFromPrompt[State, PromptIn any](
	r api.Registry,
	promptName string,
	defaultInput PromptIn,
	opts ...SessionFlowOption[State],
) *SessionFlow[any, State] {
	p := ai.LookupDataPrompt[PromptIn, string](r, promptName)
	if p == nil {
		panic(fmt.Sprintf("DefineSessionFlowFromPrompt: prompt %q not found", promptName))
	}

	fn := func(ctx context.Context, resp Responder[any], sess *SessionRunner[State]) (*SessionFlowResult, error) {
		if err := sess.Run(ctx, func(ctx context.Context, input *SessionFlowInput) error {
			// Resolve prompt input: session state override > default.
			promptInput := defaultInput
			if stored := sess.InputVariables(); stored != nil {
				typed, ok := base.ConvertTo[PromptIn](stored)
				if !ok {
					return core.NewError(core.INVALID_ARGUMENT, "input variables type mismatch: got %T, want %T", stored, promptInput)
				}
				promptInput = typed
			}

			// Render the prompt template.
			genOpts, err := p.Render(ctx, promptInput)
			if err != nil {
				return fmt.Errorf("prompt render: %w", err)
			}

			// Tag prompt-rendered messages so we can exclude them from
			// session history after generation.
			for _, m := range genOpts.Messages {
				if m.Metadata == nil {
					m.Metadata = make(map[string]any)
				}
				m.Metadata[promptMessageKey] = true
			}

			// Append conversation history after the prompt-rendered messages.
			genOpts.Messages = append(genOpts.Messages, sess.Messages()...)

			// If tool restarts were provided, set the resume option so
			// handleResumeOption re-executes the interrupted tools.
			if len(input.ToolRestarts) > 0 {
				for _, p := range input.ToolRestarts {
					if !p.IsToolRequest() {
						return core.NewError(core.INVALID_ARGUMENT, "ToolRestarts: part is not a tool request")
					}
				}
				genOpts.Resume = ai.NewResume(input.ToolRestarts, nil)
			}

			// Call the model with streaming.
			modelResp, err := ai.GenerateWithRequest(ctx, r, genOpts, nil,
				func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
					resp.SendModelChunk(chunk)
					return nil
				},
			)
			if err != nil {
				return fmt.Errorf("generate: %w", err)
			}

			// Replace session messages with the full history minus prompt
			// messages. This captures intermediate tool call/response
			// messages from the tool loop, not just the final response.
			if modelResp.Request != nil {
				var msgs []*ai.Message
				for _, m := range modelResp.History() {
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

	return DefineSessionFlow(r, promptName, fn, opts...)
}

// StreamBidi starts a new session flow invocation with bidirectional streaming.
// Use this for multi-turn interactions where you need to send multiple inputs
// and receive streaming chunks. For single-turn usage, see Run and RunText.
func (af *SessionFlow[Stream, State]) StreamBidi(
	ctx context.Context,
	opts ...InvocationOption[State],
) (*SessionFlowConnection[Stream, State], error) {
	invOpts, err := af.resolveOptions(opts)
	if err != nil {
		return nil, err
	}

	conn, err := af.flow.StreamBidi(ctx, invOpts)
	if err != nil {
		return nil, err
	}

	return &SessionFlowConnection[Stream, State]{conn: conn}, nil
}

// Run starts a single-turn session flow invocation with the given input.
// It sends the input, waits for the flow to complete, and returns the output.
// For multi-turn interactions or streaming, use StreamBidi instead.
func (af *SessionFlow[Stream, State]) Run(
	ctx context.Context,
	input *SessionFlowInput,
	opts ...InvocationOption[State],
) (*SessionFlowOutput[State], error) {
	conn, err := af.StreamBidi(ctx, opts...)
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

// RunText is a convenience method that starts a single-turn session flow
// invocation with a user text message. It is equivalent to calling Run with
// an SessionFlowInput containing a single user text message.
func (af *SessionFlow[Stream, State]) RunText(
	ctx context.Context,
	text string,
	opts ...InvocationOption[State],
) (*SessionFlowOutput[State], error) {
	return af.Run(ctx, &SessionFlowInput{
		Messages: []*ai.Message{ai.NewUserTextMessage(text)},
	}, opts...)
}

// resolveOptions applies invocation options and returns the init struct.
func (af *SessionFlow[Stream, State]) resolveOptions(opts []InvocationOption[State]) (*SessionFlowInit[State], error) {
	invOpts := &invocationOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyInvocation(invOpts); err != nil {
			return nil, fmt.Errorf("SessionFlow %q: %w", af.flow.Name(), err)
		}
	}

	init := &SessionFlowInit[State]{
		SnapshotID: invOpts.snapshotID,
		State:      invOpts.state,
	}
	if invOpts.promptInput != nil {
		if init.State == nil {
			init.State = &SessionState[State]{}
		}
		init.State.InputVariables = invOpts.promptInput
	}

	return init, nil
}

// newSessionFromInit creates a Session from initialization data.
// If resuming from a snapshot, the loaded snapshot is also returned.
func newSessionFromInit[State any](
	ctx context.Context,
	init *SessionFlowInit[State],
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

// --- SessionFlowConnection ---

// SessionFlowConnection wraps BidiConnection with session flow-specific functionality.
// It provides a Receive() iterator that supports multi-turn patterns: breaking out
// of the iterator between turns does not cancel the underlying connection.
type SessionFlowConnection[Stream, State any] struct {
	conn *core.BidiConnection[*SessionFlowInput, *SessionFlowStreamChunk[Stream], *SessionFlowOutput[State]]

	// chunks buffers stream chunks from the underlying connection so that
	// breaking from Receive() between turns doesn't cancel the context.
	chunks   chan *SessionFlowStreamChunk[Stream]
	chunkErr error
	initOnce sync.Once
}

// initReceiver starts a goroutine that drains the underlying BidiConnection's
// Receive into a channel. This goroutine never breaks from the underlying
// iterator, preventing context cancellation.
func (c *SessionFlowConnection[Stream, State]) initReceiver() {
	c.initOnce.Do(func() {
		c.chunks = make(chan *SessionFlowStreamChunk[Stream], 1)
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

// Send sends an SessionFlowInput to the session flow.
func (c *SessionFlowConnection[Stream, State]) Send(input *SessionFlowInput) error {
	return c.conn.Send(input)
}

// SendMessages sends messages to the session flow.
func (c *SessionFlowConnection[Stream, State]) SendMessages(messages ...*ai.Message) error {
	return c.conn.Send(&SessionFlowInput{Messages: messages})
}

// SendText sends a single user text message to the session flow.
func (c *SessionFlowConnection[Stream, State]) SendText(text string) error {
	return c.conn.Send(&SessionFlowInput{
		Messages: []*ai.Message{ai.NewUserTextMessage(text)},
	})
}

// SendToolRestarts sends tool restart parts to resume interrupted tool calls.
// Parts should be created via [ai.ToolDef.RestartWith].
func (c *SessionFlowConnection[Stream, State]) SendToolRestarts(parts ...*ai.Part) error {
	return c.conn.Send(&SessionFlowInput{ToolRestarts: parts})
}

// Close signals that no more inputs will be sent.
func (c *SessionFlowConnection[Stream, State]) Close() error {
	return c.conn.Close()
}

// Receive returns an iterator for receiving stream chunks.
// Unlike the underlying BidiConnection.Receive, breaking out of this iterator
// does not cancel the connection. This enables multi-turn patterns where the
// caller breaks on TurnEnd, sends the next input, then calls Receive again.
func (c *SessionFlowConnection[Stream, State]) Receive() iter.Seq2[*SessionFlowStreamChunk[Stream], error] {
	c.initReceiver()
	return func(yield func(*SessionFlowStreamChunk[Stream], error) bool) {
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

// Output returns the final response after the session flow completes.
func (c *SessionFlowConnection[Stream, State]) Output() (*SessionFlowOutput[State], error) {
	return c.conn.Output()
}

// Done returns a channel closed when the connection completes.
func (c *SessionFlowConnection[Stream, State]) Done() <-chan struct{} {
	return c.conn.Done()
}
