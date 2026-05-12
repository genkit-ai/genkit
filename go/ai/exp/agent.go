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
	"sync/atomic"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
)

// --- SessionRunner ---

// SessionRunner extends Session with agent-runtime functionality:
// turn management, snapshot persistence, and input channel handling.
type SessionRunner[State any] struct {
	*Session[State]

	// InputCh is the channel that delivers per-turn inputs from the client.
	// It is consumed automatically by [SessionRunner.Run], but is exposed
	// for advanced use cases that need direct access to the input stream
	// (e.g., custom turn loops or fan-out patterns).
	InputCh <-chan *AgentInput
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

	// intake is the source of truth for in-flight tracking, queue state,
	// and suspended state. The session consults it via beginTurnEnd (in
	// maybeSnapshot) so per-turn snapshot writes and detach captures
	// cannot race over the same input.
	intake *detachIntake
}

// parentSnapshotID returns the ID of the most recent snapshot in this
// invocation (used to chain new snapshots via ParentID), or "" if no
// snapshot has been written yet.
func (s *SessionRunner[State]) parentSnapshotID() string {
	if s.lastSnapshot == nil {
		return ""
	}
	return s.lastSnapshot.SnapshotID
}

// Run loops over the input channel, calling fn for each turn. Each turn is
// wrapped in a trace span for observability. Input messages are automatically
// added to the session before fn is called. After fn returns successfully, a
// TurnEnd chunk is sent and a snapshot check is triggered.
func (s *SessionRunner[State]) Run(ctx context.Context, fn func(ctx context.Context, input *AgentInput) error) error {
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
func (s *SessionRunner[State]) Result() *AgentResult {
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
// callback approves, state changed, detach has not suspended snapshots).
// Returns the snapshot ID or empty string.
//
// For turn-end events, the session asks the intake whether snapshots
// have been suspended (i.e. detach has landed). If so, the session skips
// the turn-end snapshot — the pending row already captures the
// invocation and a single finalize rewrite will record the cumulative
// state once the queued inputs drain.
func (s *SessionRunner[State]) maybeSnapshot(ctx context.Context, event SnapshotEvent) string {
	if event == SnapshotEventTurnEnd && s.intake != nil {
		if suspended := s.intake.beginTurnEnd(); suspended {
			return ""
		}
	}

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
			prevState = s.lastSnapshot.State
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

	parentID := s.parentSnapshotID()

	saved, err := s.store.SaveSnapshot(ctx, "",
		func(_ *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			return &SessionSnapshot[State]{
				ParentID: parentID,
				Event:    event,
				Status:   SnapshotStatusSucceeded,
				State:    &currentState,
			}, nil
		})
	if err != nil {
		// Snapshot persistence is best-effort: a store failure must not
		// kill the in-flight turn. Surface enough context in the log
		// that the failure is diagnosable without the caller having to
		// thread the error back up.
		logger.FromContext(ctx).Error("agent: failed to save snapshot",
			"parentId", parentID,
			"event", event,
			"err", err)
		return ""
	}

	s.lastSnapshot = saved
	s.lastSnapshotVersion = currentVersion
	return saved.SnapshotID
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
// If an artifact with the same name already exists in the session, it is
// replaced. The session-level side effect happens whether or not detach
// has landed; only the wire forward to the client is suppressed
// post-detach, when there is no longer a client to receive it.
func (r Responder[Stream]) SendArtifact(artifact *Artifact) {
	r <- &AgentStreamChunk[Stream]{Artifact: artifact}
}

// --- Agent ---

// AgentFunc is the function signature for custom agents.
// Type parameters:
//   - Stream: Type for status updates sent via the responder
//   - State: Type for user-defined state in snapshots
type AgentFunc[Stream, State any] = func(ctx context.Context, resp Responder[Stream], sess *SessionRunner[State]) (*AgentResult, error)

// Agent is a bidirectional streaming agent with automatic snapshot management.
type Agent[Stream, State any] struct {
	action *core.Action[*AgentInit[State], *AgentOutput[State], *AgentStreamChunk[Stream], *AgentInput]
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
// loop and registers it with the registry. The underlying action is created
// via [core.DefineBidiAction] (rather than [core.DefineBidiFlow]) so the
// agent capability metadata can be set at construction time — actions
// must be immutable once registered. The flow-context wrapping that makes
// [core.Run] work inside fn is preserved via [core.WithFlowContext].
func DefineCustomAgent[Stream, State any](
	r api.Registry,
	name string,
	fn AgentFunc[Stream, State],
	opts ...AgentOption[State],
) *Agent[Stream, State] {
	cfg := &agentOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyAgent(cfg); err != nil {
			panic(fmt.Errorf("DefineCustomAgent %q: %w", name, err))
		}
	}

	action := core.DefineBidiAction(r, name, api.ActionTypeFlow,
		&core.ActionOptions{
			Metadata: map[string]any{"agent": agentMetadataFor(cfg.store)},
		},
		func(
			ctx context.Context,
			in *AgentInit[State],
			inCh <-chan *AgentInput,
			outCh chan<- *AgentStreamChunk[Stream],
		) (*AgentOutput[State], error) {
			ctx = core.WithFlowContext(ctx, name)
			rt, err := newAgentRuntime(ctx, name, cfg, in, inCh, outCh)
			if err != nil {
				return nil, err
			}
			return rt.run(ctx, fn)
		})

	registerSnapshotActions(r, name, cfg.store, cfg.transform)

	return &Agent[Stream, State]{action: action}
}

// agentMetadataFor derives the [AgentMetadata] value attached to the
// agent's action descriptor under the "agent" key. [AgentMetadata]
// itself is generated from agent.ts; this constructor is hand-written
// because it inspects the configured store's optional capabilities.
func agentMetadataFor[State any](store SessionStore[State]) AgentMetadata {
	mgmt := AgentStateManagementClient
	abortable := false
	if store != nil {
		mgmt = AgentStateManagementServer
		_, abortable = store.(SnapshotAborter)
	}
	return AgentMetadata{
		StateManagement: mgmt,
		Abortable:       abortable,
	}
}

// --- agentRuntime ---

// agentRuntime owns the per-invocation wiring of an agent:
// session, runner, output router, input intake, and the goroutine that runs
// the user fn. Its methods implement the three terminal paths the agent can
// take: detach, fn-completion, and client-cancel.
type agentRuntime[Stream, State any] struct {
	name string
	cfg  *agentOptions[State]

	session *Session[State]
	sess    *SessionRunner[State]
	router  *chunkRouter[Stream, State]
	intake  *detachIntake

	fnDone chan fnDoneResult[State]
}

// fnDoneResult carries the user fn's return values across the goroutine
// boundary that runs it. A named type keeps the channel signatures readable.
type fnDoneResult[State any] struct {
	result *AgentResult
	err    error
}

func newAgentRuntime[Stream, State any](
	ctx context.Context,
	name string,
	cfg *agentOptions[State],
	in *AgentInit[State],
	inCh <-chan *AgentInput,
	outCh chan<- *AgentStreamChunk[Stream],
) (*agentRuntime[Stream, State], error) {
	session, parent, err := loadSession(ctx, in, cfg.store)
	if err != nil {
		return nil, err
	}

	rt := &agentRuntime[Stream, State]{
		name:    name,
		cfg:     cfg,
		session: session,
		router:  startChunkRouter(session, outCh),
		intake:  startDetachIntake(inCh),
		fnDone:  make(chan fnDoneResult[State], 1),
	}

	rt.sess = &SessionRunner[State]{
		Session:          session,
		InputCh:          rt.intake.out(),
		snapshotCallback: cfg.callback,
		lastSnapshot:     parent,
		intake:           rt.intake,
	}
	rt.sess.collectTurnOutput = func() any { return rt.router.collectTurnChunks() }
	rt.sess.onEndTurn = rt.emitTurnEnd

	return rt, nil
}

// emitTurnEnd is called by the session after each successful turn. It writes
// a turn-end snapshot (if applicable) and forwards the resulting [TurnEnd]
// chunk through the router so clients see it on the output stream.
func (rt *agentRuntime[Stream, State]) emitTurnEnd(ctx context.Context) {
	snapshotID := rt.sess.maybeSnapshot(ctx, SnapshotEventTurnEnd)
	rt.router.send() <- &AgentStreamChunk[Stream]{TurnEnd: &TurnEnd{
		SnapshotID: snapshotID,
	}}
}

// run drives the user fn to completion and returns the agent output.
//
// workCtx carries the session and is decoupled from clientCtx: pre-detach a
// watcher mirrors clientCtx so a disconnect cancels the work; on detach the
// watcher exits and the finalizer goroutine owns workCtx until fn returns.
func (rt *agentRuntime[Stream, State]) run(
	clientCtx context.Context,
	fn AgentFunc[Stream, State],
) (*AgentOutput[State], error) {
	workCtx, cancelWork := context.WithCancel(context.WithoutCancel(clientCtx))
	workCtx = NewSessionContext(workCtx, rt.session)

	var detachOnce sync.Once
	detached := make(chan struct{})
	markDetached := func() { detachOnce.Do(func() { close(detached) }) }
	defer markDetached() // ensure the watcher exits on every return path

	go func() {
		select {
		case <-clientCtx.Done():
			cancelWork()
		case <-detached:
		}
	}()

	go func() {
		// Run fn under deferred panic recovery so a panic surfaces as
		// an error rather than crashing the process or leaking the
		// fnDone channel.
		var (
			result *AgentResult
			fnErr  error
		)
		func() {
			defer func() {
				if r := recover(); r != nil {
					logger.FromContext(workCtx).Error("agent fn panicked", "panic", r, "stack", string(debug.Stack()))
					fnErr = core.NewError(core.INTERNAL, "agent fn panicked: %v", r)
				}
			}()
			result, fnErr = fn(workCtx, rt.router.responder(), rt.sess)
		}()
		rt.fnDone <- fnDoneResult[State]{result: result, err: fnErr}
	}()

	select {
	case <-rt.intake.detachSignal():
		if err := rt.checkDetachCapabilities(); err != nil {
			rt.drainAndWait(cancelWork)
			return nil, err
		}
		return rt.handleDetach(clientCtx, workCtx, cancelWork, markDetached)

	case res := <-rt.fnDone:
		return rt.handleFnDone(clientCtx, cancelWork, res)

	case <-clientCtx.Done():
		res := rt.drainAndWait(cancelWork)
		if res.err != nil {
			return nil, res.err
		}
		return nil, clientCtx.Err()
	}
}

// checkDetachCapabilities reports whether the configured store is capable
// of supporting detach. Detach requires a writable store (to persist the
// pending snapshot) and a [SnapshotAborter] (which bundles both abort
// triggering and status-change subscription so the runtime can react to
// the abort without polling).
func (rt *agentRuntime[Stream, State]) checkDetachCapabilities() error {
	if rt.cfg.store == nil {
		return core.NewError(core.FAILED_PRECONDITION,
			"agent %q: detach requires a session store", rt.name)
	}
	if _, ok := rt.cfg.store.(SnapshotAborter); !ok {
		return core.NewError(core.FAILED_PRECONDITION,
			"agent %q: detach requires a session store implementing SnapshotAborter", rt.name)
	}
	return nil
}

// drainAndWait performs a synchronous shutdown: cancel work, stop router
// writes (so a fn mid-send doesn't deadlock once outCh's consumer is
// gone), wait for the intake reader/forwarder to finish, drain fnDone,
// and close the router. Returns the fn's result for callers that need
// to surface its error.
func (rt *agentRuntime[Stream, State]) drainAndWait(cancelWork context.CancelFunc) fnDoneResult[State] {
	cancelWork()
	// Switch the router to side-effects-only mode before waiting on fn.
	// Without this, a fn mid-SendStatus blocks on the router's r.in
	// receive while the router blocks on r.out send (consumer is gone),
	// so fn never observes ctx and we deadlock waiting on fnDone.
	rt.router.stopAndWait()
	rt.intake.stopAndWait()
	res := <-rt.fnDone
	rt.router.close()
	return res
}

// handleFnDone is the synchronous-completion path: fn returned before any
// detach signal. Capture an invocation-end snapshot if state advanced past
// the last turn-end snapshot, then assemble the output.
func (rt *agentRuntime[Stream, State]) handleFnDone(
	ctx context.Context,
	cancelWork context.CancelFunc,
	res fnDoneResult[State],
) (*AgentOutput[State], error) {
	cancelWork()
	rt.intake.stopAndWait()
	rt.router.close()

	if res.err != nil {
		return nil, res.err
	}

	snapshotID := rt.sess.maybeSnapshot(ctx, SnapshotEventInvocationEnd)
	if snapshotID == "" && rt.sess.lastSnapshot != nil {
		// State unchanged since the last turn-end snapshot — reuse it so
		// the response always carries an ID when a store is configured.
		snapshotID = rt.sess.lastSnapshot.SnapshotID
	}

	out := &AgentOutput[State]{SnapshotID: snapshotID}
	if res.result != nil {
		out.Message = res.result.Message
		out.Artifacts = res.result.Artifacts
	}
	if rt.cfg.store == nil {
		out.State = applyTransform(ctx, rt.cfg.transform, rt.session.State())
	}
	return out, nil
}

// handleDetach commits the pending snapshot, returns its ID, and spawns the
// status-subscriber and finalizer goroutines that own the rest of the
// invocation. Per-turn snapshots are suspended for the remainder so the
// queued inputs roll into a single finalize rewrite; the chunk router
// stops writing to outCh but keeps applying in-process side effects
// (e.g. artifacts added via Responder.SendArtifact) so user code does
// not have to branch on detach.
func (rt *agentRuntime[Stream, State]) handleDetach(
	clientCtx, workCtx context.Context,
	cancelWork context.CancelFunc,
	markDetached func(),
) (*AgentOutput[State], error) {
	// Stop mirroring clientCtx. From here, only the abort subscription or
	// fn completion can cancel workCtx.
	markDetached()

	rt.intake.suspend()

	parentID := rt.sess.parentSnapshotID()

	// Detach intends to outlive the client connection. If clientCtx was
	// already cancelled (or cancels mid-write), we still want the pending
	// row durable so observers can find it later. Decouple this write.
	pending, err := rt.cfg.store.SaveSnapshot(context.WithoutCancel(clientCtx), "",
		func(_ *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			return &SessionSnapshot[State]{
				ParentID: parentID,
				Event:    SnapshotEventDetach,
				Status:   SnapshotStatusPending,
			}, nil
		})
	if err != nil {
		rt.drainAndWait(cancelWork)
		return nil, core.NewError(core.INTERNAL,
			"agent %q: detach: save pending snapshot: %v", rt.name, err)
	}

	// The router can no longer write to outCh once we return; the bidi
	// framework closes it shortly after. The router stops writing and
	// trashes any further chunks.
	rt.router.stopAndWait()

	abortedByUser := &atomic.Bool{}
	subCtx, stopSub := context.WithCancel(workCtx)
	aborter := rt.cfg.store.(SnapshotAborter) // safe: checkDetachCapabilities ran already
	statusCh := aborter.OnSnapshotStatusChange(subCtx, pending.SnapshotID)
	go func() {
		for status := range statusCh {
			if status == SnapshotStatusAborted {
				abortedByUser.Store(true)
				cancelWork()
				return
			}
		}
	}()

	finalizeCtx := context.WithoutCancel(clientCtx)
	go func() {
		res := <-rt.fnDone
		stopSub()
		rt.intake.stopAndWait()
		rt.router.close()
		rt.finalizePendingSnapshot(finalizeCtx, pending, res.err, abortedByUser.Load())
		cancelWork()
	}()

	return &AgentOutput[State]{SnapshotID: pending.SnapshotID}, nil
}

// finalizePendingSnapshot rewrites the pending snapshot row with the
// terminal state and status. abortedByUser distinguishes a context
// cancellation from abortSnapshot (status=aborted) from an internal
// failure (status=failed). The write is funneled through SaveSnapshot
// so the read-and-rewrite is one atomic step: if the row has already
// transitioned to aborted (a late abort racing this finalize),
// SaveSnapshot sees it inside fn and we leave the row untouched.
func (rt *agentRuntime[Stream, State]) finalizePendingSnapshot(
	ctx context.Context,
	pending *SessionSnapshot[State],
	fnErr error,
	abortedByUser bool,
) {
	finalState := *rt.session.State()

	_, err := rt.cfg.store.SaveSnapshot(ctx, pending.SnapshotID,
		func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			// Late abort wins over the terminal we were about to land.
			if existing != nil && existing.Status == SnapshotStatusAborted {
				return nil, nil
			}

			status := SnapshotStatusSucceeded
			var snapErr *core.GenkitError
			switch {
			case abortedByUser:
				status = SnapshotStatusAborted
				if fnErr != nil {
					snapErr = core.AsGenkitError(fnErr) // aborted wins, preserve text
				}
			case fnErr != nil:
				status = SnapshotStatusFailed
				snapErr = core.AsGenkitError(fnErr)
			}

			return &SessionSnapshot[State]{
				ParentID: pending.ParentID,
				Event:    SnapshotEventDetach,
				Status:   status,
				Error:    snapErr,
				State:    &finalState,
			}, nil
		})
	if err != nil {
		logger.FromContext(ctx).Error("agent: failed to finalize pending snapshot",
			"snapshotId", pending.SnapshotID, "err", err)
	}
}

// loadSession constructs a Session from the invocation's init payload,
// loading from the store when a snapshot ID is provided. Returns the
// snapshot too so the runtime can chain ParentID off it.
func loadSession[State any](
	ctx context.Context,
	init *AgentInit[State],
	store SessionStore[State],
) (*Session[State], *SessionSnapshot[State], error) {
	s := &Session[State]{store: store}
	if init == nil {
		return s, nil, nil
	}

	if init.SnapshotID != "" && init.State != nil {
		return nil, nil, core.NewError(core.INVALID_ARGUMENT, "snapshot ID and state are mutually exclusive")
	}

	if init.SnapshotID == "" {
		if init.State != nil {
			if store != nil {
				return nil, nil, core.NewError(core.FAILED_PRECONDITION,
					"state provided but agent has a session store configured (server-managed state); use snapshot ID instead")
			}
			s.state = *init.State
		}
		return s, nil, nil
	}

	if store == nil {
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot ID %q provided but agent has no session store configured (client-managed state); use state instead", init.SnapshotID)
	}
	snap, err := store.GetSnapshot(ctx, init.SnapshotID)
	if err != nil {
		return nil, nil, core.NewError(core.INTERNAL, "failed to load snapshot %q: %v", init.SnapshotID, err)
	}
	if snap == nil {
		return nil, nil, core.NewError(core.NOT_FOUND, "snapshot %q not found", init.SnapshotID)
	}
	switch snap.Status {
	case SnapshotStatusFailed:
		msg := "snapshot recorded an error"
		if snap.Error != nil && snap.Error.Message != "" {
			msg = snap.Error.Message
		}
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q terminated with error: %s", init.SnapshotID, msg)
	case SnapshotStatusPending:
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q is still pending; wait for it to finalize before resuming", init.SnapshotID)
	case SnapshotStatusAborted:
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q was aborted", init.SnapshotID)
	}
	if snap.State != nil {
		s.state = *snap.State
	}
	return s, snap, nil
}

// --- chunkRouter ---
//
// chunkRouter owns the intermediate stream channel that all chunks flow
// through on their way to outCh. Every chunk gets the same in-process
// side effects (adding artifacts to the session, accumulating turn
// chunks for span output) regardless of whether detach has landed; the
// wire forward to outCh is the only thing detach suppresses, since the
// bidi framework closes outCh shortly after bidiFn returns. The router
// commits to not writing before we return so that close is safe, and
// keeps draining its input so the user fn never blocks on a responder
// send.

type chunkRouter[Stream, State any] struct {
	in      chan *AgentStreamChunk[Stream]
	out     chan<- *AgentStreamChunk[Stream]
	session *Session[State]

	turnMu     sync.Mutex
	turnChunks []*AgentStreamChunk[Stream]

	done          chan struct{}
	stopWriting   chan struct{}
	writerStopped chan struct{}
}

func startChunkRouter[Stream, State any](
	session *Session[State],
	out chan<- *AgentStreamChunk[Stream],
) *chunkRouter[Stream, State] {
	r := &chunkRouter[Stream, State]{
		in:            make(chan *AgentStreamChunk[Stream]),
		out:           out,
		session:       session,
		done:          make(chan struct{}),
		stopWriting:   make(chan struct{}),
		writerStopped: make(chan struct{}),
	}
	go r.run()
	return r
}

func (r *chunkRouter[Stream, State]) run() {
	defer close(r.done)
	if !r.forward() {
		// r.in closed before detach; nothing left to do.
		return
	}
	close(r.writerStopped)
	// Detached: keep applying side effects so the user fn's
	// SendArtifact/SendModelChunk calls behave the same way they did
	// pre-detach. Only the wire forward to outCh is suppressed.
	for chunk := range r.in {
		r.applySideEffects(chunk)
	}
}

// applySideEffects records the chunk's effect on session state and turn
// span output. Invoked from both forward (pre-detach) and the post-detach
// drain so a Send call is observably the same in either mode.
func (r *chunkRouter[Stream, State]) applySideEffects(chunk *AgentStreamChunk[Stream]) {
	if chunk.Artifact != nil {
		r.session.AddArtifacts(chunk.Artifact)
	}
	if chunk.TurnEnd == nil {
		r.turnMu.Lock()
		r.turnChunks = append(r.turnChunks, chunk)
		r.turnMu.Unlock()
	}
}

// forward delivers chunks to outCh and applies side effects until detach
// or r.in closes. Returns true if it stopped because of detach.
func (r *chunkRouter[Stream, State]) forward() bool {
	for {
		select {
		case chunk, ok := <-r.in:
			if !ok {
				return false
			}
			r.applySideEffects(chunk)
			select {
			case r.out <- chunk:
			case <-r.stopWriting:
				return true
			}
		case <-r.stopWriting:
			return true
		}
	}
}

// responder returns a [Responder] that sends chunks into the router.
func (r *chunkRouter[Stream, State]) responder() Responder[Stream] {
	return Responder[Stream](r.in)
}

// send returns the internal chunk channel for producers other than the user
// agent function (e.g. the runtime's emitTurnEnd).
func (r *chunkRouter[Stream, State]) send() chan<- *AgentStreamChunk[Stream] {
	return r.in
}

// collectTurnChunks returns and resets accumulated turn chunks.
func (r *chunkRouter[Stream, State]) collectTurnChunks() []*AgentStreamChunk[Stream] {
	r.turnMu.Lock()
	defer r.turnMu.Unlock()
	result := r.turnChunks
	r.turnChunks = nil
	return result
}

// stopAndWait tells the router to stop writing to out and blocks until it
// has committed. After it returns, it is safe for the framework to close
// out without risking a write-to-closed-channel panic.
func (r *chunkRouter[Stream, State]) stopAndWait() {
	close(r.stopWriting)
	<-r.writerStopped
}

// close signals end-of-input and waits for the router to drain.
func (r *chunkRouter[Stream, State]) close() {
	close(r.in)
	<-r.done
}

// --- detachIntake ---
//
// detachIntake separates eager src reading from runner-paced forwarding,
// and owns the queue and suspend state.
//
// The reader goroutine pulls from the bidi framework's inCh as soon as
// inputs arrive and appends them to an internal queue. This is what makes
// detach detection immediate: the moment an input with [AgentInput.Detach]
// lands in src, the reader sees it without waiting for the runner to
// finish whatever it's processing.
//
// The forwarder goroutine pops the queue and writes to dst, blocking on
// the runner via turnDone so it stays in step with turn pacing.
//
// The runner asks beginTurnEnd at the end of each turn: if suspended
// (detach has landed), the runner skips its turn-end snapshot — the
// pending row already captures the invocation and a single finalize
// will rewrite it with the cumulative state once the queued inputs
// drain. If not suspended, a normal turn-end snapshot is written.
//
// suspend is called once by the detach handler under the same mutex
// that beginTurnEnd reads from, ensuring memory ordering: any
// beginTurnEnd that returns after suspend completes sees suspended=true.

type detachIntake struct {
	src    <-chan *AgentInput
	dst    chan *AgentInput
	notify chan struct{} // buffered size 1; wakes forwarder when queue grows

	// turnDone is signaled by beginTurnEnd to release the forwarder so it
	// may pop the next input. Initialized with one token so the very
	// first turn can start without a preceding turn end.
	turnDone chan struct{}

	mu        sync.Mutex
	suspended bool
	queue     []*AgentInput

	readDone atomic.Bool
	detachCh chan struct{} // signaled by reader when detach observed

	stop     chan struct{}
	stopOnce sync.Once
	done     chan struct{}
}

func startDetachIntake(src <-chan *AgentInput) *detachIntake {
	i := &detachIntake{
		src:      src,
		dst:      make(chan *AgentInput),
		notify:   make(chan struct{}, 1),
		turnDone: make(chan struct{}, 1),
		detachCh: make(chan struct{}, 1),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	i.turnDone <- struct{}{} // initial credit for the first turn
	go i.run()
	return i
}

func (i *detachIntake) run() {
	defer close(i.done)

	forwarderDone := make(chan struct{})
	go func() {
		defer close(forwarderDone)
		defer close(i.dst)
		i.forward()
	}()

	i.read()
	<-forwarderDone
}

// signal wakes the forwarder. Non-blocking: the channel is buffered size
// 1, so a pending signal is enough.
func (i *detachIntake) signal() {
	select {
	case i.notify <- struct{}{}:
	default:
	}
}

// read pulls eagerly from src into the internal queue and detects detach
// the moment it lands. When detach is observed, it drains any remaining
// buffered src non-blockingly (so all pre-detach inputs are accounted
// for), signals the detach handler, and exits.
func (i *detachIntake) read() {
	defer func() {
		i.readDone.Store(true)
		i.signal()
	}()

	for {
		select {
		case input, ok := <-i.src:
			if !ok {
				return
			}
			if input.Detach {
				i.handleDetach(input)
				return
			}
			i.enqueue(input)
		case <-i.stop:
			return
		}
	}
}

func (i *detachIntake) enqueue(input *AgentInput) {
	i.mu.Lock()
	i.queue = append(i.queue, input)
	i.mu.Unlock()
	i.signal()
}

// handleDetach drains any buffered src inputs into the queue and signals
// the detach handler. The detach handler then calls suspend to halt
// turn-end snapshots while the queued inputs finish processing.
//
// A pure detach signal (no Messages, no Resume payload) is dropped
// rather than enqueued: it carries no payload to process, so it would
// just trigger a no-op turn. Callers that want to ride a final input
// on the detach signal can do so by calling
// Send(&AgentInput{Detach: true, Messages: ...}) explicitly.
func (i *detachIntake) handleDetach(first *AgentInput) {
	var drained []*AgentInput
	if hasInputPayload(first) {
		drained = append(drained, first)
	}
drainLoop:
	for {
		select {
		case more, ok := <-i.src:
			if !ok {
				break drainLoop
			}
			drained = append(drained, more)
		default:
			break drainLoop
		}
	}

	if len(drained) > 0 {
		i.mu.Lock()
		i.queue = append(i.queue, drained...)
		i.mu.Unlock()
		i.signal()
	}

	select {
	case i.detachCh <- struct{}{}:
	case <-i.stop:
	}
}

// hasInputPayload reports whether the input carries data the runner would
// otherwise process. Used to filter pure detach signals out of the
// queue so they don't trigger no-op turns.
func hasInputPayload(in *AgentInput) bool {
	if in == nil {
		return false
	}
	if len(in.Messages) > 0 {
		return true
	}
	if in.Resume != nil && (len(in.Resume.Respond) > 0 || len(in.Resume.Restart) > 0) {
		return true
	}
	return false
}

// forward pops the queue and writes to dst at the runner's pace. The
// runner signals turnDone via beginTurnEnd when it's ready for the next
// input; until then the forwarder waits, so it never gets ahead of the
// runner.
func (i *detachIntake) forward() {
	for {
		// Wait for the previous turn to release us (initial credit lets
		// the first turn through immediately).
		select {
		case <-i.turnDone:
		case <-i.stop:
			return
		}
		input := i.awaitInput()
		if input == nil {
			return // reader done with empty queue, or stop signaled
		}
		forwarded := *input
		forwarded.Detach = false
		select {
		case i.dst <- &forwarded:
		case <-i.stop:
			return
		}
	}
}

// awaitInput blocks until the queue has an input, the reader is done, or
// stop is signaled. Returns the popped input or nil if no further inputs
// will arrive.
func (i *detachIntake) awaitInput() *AgentInput {
	for {
		i.mu.Lock()
		if len(i.queue) > 0 {
			input := i.queue[0]
			i.queue = i.queue[1:]
			i.mu.Unlock()
			return input
		}
		done := i.readDone.Load()
		i.mu.Unlock()
		if done {
			return nil
		}
		select {
		case <-i.notify:
		case <-i.stop:
			return nil
		}
	}
}

// releaseForward releases the forwarder so it can pop the next input.
// Must be called from beginTurnEnd (and only there) so the forwarder
// stays in step with the runner's turn pacing.
func (i *detachIntake) releaseForward() {
	select {
	case i.turnDone <- struct{}{}:
	default:
	}
}

func (i *detachIntake) out() <-chan *AgentInput {
	return i.dst
}

func (i *detachIntake) detachSignal() <-chan struct{} {
	return i.detachCh
}

// beginTurnEnd is called by [SessionRunner.maybeSnapshot] before writing
// a turn-end snapshot. If the intake has been suspended (detach landed),
// it returns suspended=true and the runner skips the snapshot.
//
// In all cases (including suspended) the forwarder is released so it can
// pop the next queued input — suspension stops snapshot writing, not
// processing.
func (i *detachIntake) beginTurnEnd() (suspended bool) {
	i.mu.Lock()
	suspended = i.suspended
	i.mu.Unlock()
	i.releaseForward()
	return suspended
}

// suspend is called once by the detach handler. It flips suspended=true
// under the mutex so subsequent beginTurnEnd calls observe the change
// and skip their turn-end snapshot writes; the queued inputs roll into
// a single finalize rewrite of the pending row instead.
func (i *detachIntake) suspend() {
	i.mu.Lock()
	i.suspended = true
	i.mu.Unlock()
}

// stopAndWait forces the intake to exit and waits for both reader and
// forwarder goroutines.
func (i *detachIntake) stopAndWait() {
	i.stopOnce.Do(func() { close(i.stop) })
	<-i.done
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
	return func(ctx context.Context, resp Responder[any], sess *SessionRunner[State]) (*AgentResult, error) {
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

			// If a resume payload was provided, forward it to the
			// generate call so handleResumeOption re-executes the
			// interrupted tools and / or applies the responses.
			if input.Resume != nil {
				actionOpts.Resume = &ai.GenerateActionResume{
					Respond: input.Resume.Respond,
					Restart: input.Resume.Restart,
				}
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

// --- Agent client API ---

// StreamBidi starts a new agent invocation with bidirectional streaming.
// Use this for multi-turn interactions where you need to send multiple inputs
// and receive streaming chunks. For single-turn usage, see Run and RunText.
func (a *Agent[Stream, State]) StreamBidi(
	ctx context.Context,
	opts ...InvocationOption[State],
) (*AgentConnection[Stream, State], error) {
	init, err := a.resolveOptions(opts)
	if err != nil {
		return nil, err
	}
	conn, err := a.action.StreamBidi(ctx, init)
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
	// If the bidi function fails fast (e.g. resuming from an errored
	// snapshot rejects in newAgentRuntime), Send / Close / Receive
	// see a closed connection and return generic "action has completed"
	// errors. The real fn error is on Output(). Prefer it whenever it's
	// non-nil so callers get the meaningful failure.
	if err := conn.Send(input); err != nil {
		if _, outErr := conn.Output(); outErr != nil {
			return nil, outErr
		}
		return nil, err
	}
	if err := conn.Close(); err != nil {
		if _, outErr := conn.Output(); outErr != nil {
			return nil, outErr
		}
		return nil, err
	}
	for _, err := range conn.Receive() {
		if err != nil {
			if _, outErr := conn.Output(); outErr != nil {
				return nil, outErr
			}
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
			return nil, fmt.Errorf("Agent %q: %w", a.action.Name(), err)
		}
	}

	return &AgentInit[State]{
		SnapshotID: invOpts.snapshotID,
		State:      invOpts.state,
	}, nil
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

// SendResume sends a resume payload to continue an interrupted generation.
// Construct the payload with [ai.ToolDef.RestartWith] or
// [ai.ToolDef.RespondWith] parts.
func (c *AgentConnection[Stream, State]) SendResume(resume *ToolResume) error {
	return c.conn.Send(&AgentInput{Resume: resume})
}

// Detach asks the server to write a pending snapshot, close the
// connection, and continue processing any already-buffered inputs in
// the background. Output() returns the pending snapshot ID; the client
// can later call AbortSnapshot to stop the background work or
// GetSnapshot to observe its progression. The pending snapshot is
// finalized with the cumulative final state once the queued inputs
// are processed.
//
// Streamed chunks emitted after detach are not forwarded over the wire
// (the connection is gone), but their session-level side effects still
// apply: artifacts sent via [Responder.SendArtifact] land in the
// session and end up in the final snapshot's state.
//
// To send a final input as part of the same wire message, use
// Send(&AgentInput{Detach: true, Messages: ...}) directly.
func (c *AgentConnection[Stream, State]) Detach() error {
	return c.conn.Send(&AgentInput{Detach: true})
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
		for chunk := range c.chunks {
			if !yield(chunk, nil) {
				return
			}
		}
		if err := c.chunkErr; err != nil {
			yield(nil, err)
		}
	}
}

// Output returns the final response after the agent completes.
//
// Unlike the underlying BidiConnection, Output waits for the agent to
// finalize before returning. This is important for detached invocations:
// when the client sends Detach, the agent function returns promptly with a
// pending snapshot ID, and callers need to observe that output rather than
// the context cancellation error.
func (c *AgentConnection[Stream, State]) Output() (*AgentOutput[State], error) {
	<-c.conn.Done()
	return c.conn.Output()
}

// Done returns a channel closed when the connection completes.
func (c *AgentConnection[Stream, State]) Done() <-chan struct{} {
	return c.conn.Done()
}
