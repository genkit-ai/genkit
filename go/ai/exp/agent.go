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

// --- AgentSession ---

// AgentSession extends Session with agent-flow-specific functionality:
// turn management, snapshot persistence, and input channel handling.
type AgentSession[State any] struct {
	*Session[State]

	// InputCh is the channel that delivers per-turn inputs from the client.
	// It is consumed automatically by [AgentSession.Run], but is exposed
	// for advanced use cases that need direct access to the input stream
	// (e.g., custom turn loops or fan-out patterns).
	InputCh <-chan *AgentFlowInput
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
// added to the session before fn is called. After fn returns successfully, an
// EndTurn chunk is sent and a snapshot check is triggered.
func (a *AgentSession[State]) Run(ctx context.Context, fn func(ctx context.Context, input *AgentFlowInput) error) error {
	for input := range a.InputCh {
		spanMeta := &tracing.SpanMetadata{
			Name:    fmt.Sprintf("agentFlow/turn/%d", a.TurnIndex),
			Type:    "agentFlowTurn",
			Subtype: "agentFlowTurn",
		}

		_, err := tracing.RunInNewSpan(ctx, spanMeta, input,
			func(ctx context.Context, input *AgentFlowInput) (any, error) {
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

// Result returns an [AgentFlowResult] populated from the current session state:
// the last message in the conversation history and all artifacts.
// It is a convenience for custom agent flows that don't need to construct the
// result manually.
func (a *AgentSession[State]) Result() *AgentFlowResult {
	a.mu.RLock()
	defer a.mu.RUnlock()

	result := &AgentFlowResult{}
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
func (a *AgentSession[State]) maybeSnapshot(ctx context.Context, event SnapshotEvent) string {
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
		logger.FromContext(ctx).Error("agent flow: failed to save snapshot", "err", err)
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

// Responder is the output channel for an agent flow. Artifacts sent through
// it are automatically added to the session before being forwarded to the
// client.
type Responder[Stream any] chan<- *AgentFlowStreamChunk[Stream]

// SendModelChunk sends a generation chunk (token-level streaming).
func (r Responder[Stream]) SendModelChunk(chunk *ai.ModelResponseChunk) {
	r <- &AgentFlowStreamChunk[Stream]{ModelChunk: chunk}
}

// SendStatus sends a user-defined status update.
func (r Responder[Stream]) SendStatus(status Stream) {
	r <- &AgentFlowStreamChunk[Stream]{Status: status}
}

// SendArtifact sends an artifact to the stream and adds it to the session.
// If an artifact with the same name already exists in the session, it is replaced.
func (r Responder[Stream]) SendArtifact(artifact *Artifact) {
	r <- &AgentFlowStreamChunk[Stream]{Artifact: artifact}
}

// --- AgentFlow ---

// AgentFlowFunc is the function signature for agent flows.
// Type parameters:
//   - Stream: Type for status updates sent via the responder
//   - State: Type for user-defined state in snapshots
type AgentFlowFunc[Stream, State any] = func(ctx context.Context, resp Responder[Stream], sess *AgentSession[State]) (*AgentFlowResult, error)

// AgentFlow is a bidirectional streaming flow with automatic snapshot management.
type AgentFlow[Stream, State any] struct {
	flow *core.Flow[*AgentFlowInput, *AgentFlowOutput[State], *AgentFlowStreamChunk[Stream], *AgentFlowInit[State]]
}

// DefineCustomAgent creates an AgentFlow with automatic snapshot management and registers it.
func DefineCustomAgent[Stream, State any](
	r api.Registry,
	name string,
	fn AgentFlowFunc[Stream, State],
	opts ...AgentFlowOption[State],
) *AgentFlow[Stream, State] {
	afOpts := &agentFlowOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyAgentFlow(afOpts); err != nil {
			panic(fmt.Errorf("DefineCustomAgent %q: %w", name, err))
		}
	}

	store := afOpts.store
	snapshotCallback := afOpts.callback

	flow := core.DefineBidiFlow(r, name, func(
		ctx context.Context,
		init *AgentFlowInit[State],
		inCh <-chan *AgentFlowInput,
		outCh chan<- *AgentFlowStreamChunk[Stream],
	) (*AgentFlowOutput[State], error) {
		session, snapshot, err := newSessionFromInit(ctx, init, store)
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

		// Turn output accumulator: collects content chunks per turn for span output.
		var (
			turnMu     sync.Mutex
			turnChunks []*AgentFlowStreamChunk[Stream]
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
		respCh := make(chan *AgentFlowStreamChunk[Stream])
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			for chunk := range respCh {
				if chunk.Artifact != nil {
					session.AddArtifacts(chunk.Artifact)
				}
				// Accumulate content chunks (exclude control signals from onEndTurn).
				if !chunk.EndTurn && chunk.SnapshotID == "" {
					turnMu.Lock()
					turnChunks = append(turnChunks, chunk)
					turnMu.Unlock()
				}
				outCh <- chunk
			}
		}()

		// Wire up onEndTurn: triggers snapshot + sends EndTurn chunk.
		// Writes through respCh to preserve ordering with user chunks.
		agentSess.onEndTurn = func(turnCtx context.Context) {
			snapshotID := agentSess.maybeSnapshot(turnCtx, SnapshotEventTurnEnd)
			if snapshotID != "" {
				respCh <- &AgentFlowStreamChunk[Stream]{SnapshotID: snapshotID}
			}
			respCh <- &AgentFlowStreamChunk[Stream]{EndTurn: true}
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

		out := &AgentFlowOutput[State]{
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

	return &AgentFlow[Stream, State]{flow: flow}
}

// promptMessageKey is the metadata key used to tag prompt-rendered messages
// so they can be excluded from session history after generation.
const promptMessageKey = "_genkit_prompt"

// DefinePromptAgent creates a prompt-backed AgentFlow with an
// automatic conversation loop. Each turn renders the prompt, appends
// conversation history, calls GenerateWithRequest, streams chunks to the
// client, and adds the model response to the session.
//
// The prompt is looked up by name from the registry using
// [ai.LookupDataPrompt]. The defaultInput is used for prompt rendering
// unless overridden per invocation via WithInputVariables.
func DefinePromptAgent[State, PromptIn any](
	r api.Registry,
	promptName string,
	defaultInput PromptIn,
	opts ...AgentFlowOption[State],
) *AgentFlow[any, State] {
	p := ai.LookupDataPrompt[PromptIn, string](r, promptName)
	if p == nil {
		panic(fmt.Sprintf("DefinePromptAgent: prompt %q not found", promptName))
	}

	fn := func(ctx context.Context, resp Responder[any], sess *AgentSession[State]) (*AgentFlowResult, error) {
		if err := sess.Run(ctx, func(ctx context.Context, input *AgentFlowInput) error {
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

	return DefineCustomAgent(r, promptName, fn, opts...)
}

// StreamBidi starts a new agent flow invocation with bidirectional streaming.
// Use this for multi-turn interactions where you need to send multiple inputs
// and receive streaming chunks. For single-turn usage, see Run and RunText.
func (af *AgentFlow[Stream, State]) StreamBidi(
	ctx context.Context,
	opts ...InvocationOption[State],
) (*AgentFlowConnection[Stream, State], error) {
	invOpts, err := af.resolveOptions(opts)
	if err != nil {
		return nil, err
	}

	conn, err := af.flow.StreamBidi(ctx, invOpts)
	if err != nil {
		return nil, err
	}

	return &AgentFlowConnection[Stream, State]{conn: conn}, nil
}

// Run starts a single-turn agent flow invocation with the given input.
// It sends the input, waits for the flow to complete, and returns the output.
// For multi-turn interactions or streaming, use StreamBidi instead.
func (af *AgentFlow[Stream, State]) Run(
	ctx context.Context,
	input *AgentFlowInput,
	opts ...InvocationOption[State],
) (*AgentFlowOutput[State], error) {
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

// RunText is a convenience method that starts a single-turn agent flow
// invocation with a user text message. It is equivalent to calling Run with
// an AgentFlowInput containing a single user text message.
func (af *AgentFlow[Stream, State]) RunText(
	ctx context.Context,
	text string,
	opts ...InvocationOption[State],
) (*AgentFlowOutput[State], error) {
	return af.Run(ctx, &AgentFlowInput{
		Messages: []*ai.Message{ai.NewUserTextMessage(text)},
	}, opts...)
}

// resolveOptions applies invocation options and returns the init struct.
func (af *AgentFlow[Stream, State]) resolveOptions(opts []InvocationOption[State]) (*AgentFlowInit[State], error) {
	invOpts := &invocationOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyInvocation(invOpts); err != nil {
			return nil, fmt.Errorf("AgentFlow %q: %w", af.flow.Name(), err)
		}
	}

	init := &AgentFlowInit[State]{
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
	init *AgentFlowInit[State],
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

// --- AgentFlowConnection ---

// AgentFlowConnection wraps BidiConnection with agent flow-specific functionality.
// It provides a Receive() iterator that supports multi-turn patterns: breaking out
// of the iterator between turns does not cancel the underlying connection.
type AgentFlowConnection[Stream, State any] struct {
	conn *core.BidiConnection[*AgentFlowInput, *AgentFlowOutput[State], *AgentFlowStreamChunk[Stream]]
	// chunks buffers stream chunks from the underlying connection so that
	// breaking from Receive() between turns doesn't cancel the context.
	chunks   chan *AgentFlowStreamChunk[Stream]
	chunkErr error
	initOnce sync.Once
}

// initReceiver starts a goroutine that drains the underlying BidiConnection's
// Receive into a channel. This goroutine never breaks from the underlying
// iterator, preventing context cancellation.
func (c *AgentFlowConnection[Stream, State]) initReceiver() {
	c.initOnce.Do(func() {
		c.chunks = make(chan *AgentFlowStreamChunk[Stream], 1)
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

// Send sends an AgentFlowInput to the agent flow.
func (c *AgentFlowConnection[Stream, State]) Send(input *AgentFlowInput) error {
	return c.conn.Send(input)
}

// SendMessages sends messages to the agent flow.
func (c *AgentFlowConnection[Stream, State]) SendMessages(messages ...*ai.Message) error {
	return c.conn.Send(&AgentFlowInput{Messages: messages})
}

// SendText sends a single user text message to the agent flow.
func (c *AgentFlowConnection[Stream, State]) SendText(text string) error {
	return c.conn.Send(&AgentFlowInput{
		Messages: []*ai.Message{ai.NewUserTextMessage(text)},
	})
}

// SendToolRestarts sends tool restart parts to resume interrupted tool calls.
// Parts should be created via [ai.ToolDef.RestartWith].
func (c *AgentFlowConnection[Stream, State]) SendToolRestarts(parts ...*ai.Part) error {
	return c.conn.Send(&AgentFlowInput{ToolRestarts: parts})
}

// Close signals that no more inputs will be sent.
func (c *AgentFlowConnection[Stream, State]) Close() error {
	return c.conn.Close()
}

// Receive returns an iterator for receiving stream chunks.
// Unlike the underlying BidiConnection.Receive, breaking out of this iterator
// does not cancel the connection. This enables multi-turn patterns where the
// caller breaks on EndTurn, sends the next input, then calls Receive again.
func (c *AgentFlowConnection[Stream, State]) Receive() iter.Seq2[*AgentFlowStreamChunk[Stream], error] {
	c.initReceiver()
	return func(yield func(*AgentFlowStreamChunk[Stream], error) bool) {
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

// Output returns the final response after the agent flow completes.
func (c *AgentFlowConnection[Stream, State]) Output() (*AgentFlowOutput[State], error) {
	return c.conn.Output()
}

// Done returns a channel closed when the connection completes.
func (c *AgentFlowConnection[Stream, State]) Done() <-chan struct{} {
	return c.conn.Done()
}
