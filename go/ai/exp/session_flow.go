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
	"sync/atomic"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/uuid"
)

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
	cfg := &sessionFlowOptions[State]{}
	for _, opt := range opts {
		if err := opt.applySessionFlow(cfg); err != nil {
			panic(fmt.Errorf("DefineSessionFlow %q: %w", name, err))
		}
	}

	flow := core.DefineBidiFlow(r, name, func(
		ctx context.Context,
		in *SessionFlowInit[State],
		inCh <-chan *SessionFlowInput,
		outCh chan<- *SessionFlowStreamChunk[Stream],
	) (*SessionFlowOutput[State], error) {
		rt, err := newSessionFlowRuntime(ctx, name, cfg, in, inCh, outCh)
		if err != nil {
			return nil, err
		}
		return rt.run(ctx, fn)
	})

	registerSnapshotActions(r, name, cfg.store, cfg.transform)

	return &SessionFlow[Stream, State]{flow: flow}
}

// --- sessionFlowRuntime ---

// sessionFlowRuntime owns the per-invocation wiring of a session flow:
// session, runner, output router, input intake, and the goroutine that runs
// the user fn. Its methods implement the three terminal paths the flow can
// take: detach, fn-completion, and client-cancel.
type sessionFlowRuntime[Stream, State any] struct {
	name string
	cfg  *sessionFlowOptions[State]

	session *Session[State]
	runner  *SessionRunner[State]
	router  *chunkRouter[Stream, State]
	intake  *detachIntake

	fnDone chan fnDoneResult[State]
}

// fnDoneResult carries the user fn's return values across the goroutine
// boundary that runs it. A named type keeps the channel signatures readable.
type fnDoneResult[State any] struct {
	result *SessionFlowResult
	err    error
}

func newSessionFlowRuntime[Stream, State any](
	ctx context.Context,
	name string,
	cfg *sessionFlowOptions[State],
	in *SessionFlowInit[State],
	inCh <-chan *SessionFlowInput,
	outCh chan<- *SessionFlowStreamChunk[Stream],
) (*sessionFlowRuntime[Stream, State], error) {
	session, parent, err := loadSession(ctx, in, cfg.store)
	if err != nil {
		return nil, err
	}

	rt := &sessionFlowRuntime[Stream, State]{
		name:    name,
		cfg:     cfg,
		session: session,
		router:  startChunkRouter(session, outCh),
		intake:  startDetachIntake(inCh),
		fnDone:  make(chan fnDoneResult[State], 1),
	}

	rt.runner = &SessionRunner[State]{
		Session:          session,
		InputCh:          rt.intake.out(),
		snapshotCallback: cfg.callback,
		lastSnapshot:     parent,
		intake:           rt.intake,
	}
	rt.runner.collectTurnOutput = func() any { return rt.router.collectTurnChunks() }
	rt.runner.onEndTurn = rt.emitTurnEnd

	return rt, nil
}

// emitTurnEnd is called by the runner after each successful turn. It writes
// a turn-end snapshot (if applicable) and forwards the resulting [TurnEnd]
// chunk through the router so clients see it on the output stream.
func (rt *sessionFlowRuntime[Stream, State]) emitTurnEnd(ctx context.Context) {
	snapshotID := rt.runner.maybeSnapshot(ctx, SnapshotEventTurnEnd)
	rt.router.send() <- &SessionFlowStreamChunk[Stream]{TurnEnd: &TurnEnd{
		SnapshotID: snapshotID,
	}}
}

// run drives the user fn to completion and returns the flow output.
//
// workCtx carries the session and is decoupled from clientCtx: pre-detach a
// watcher mirrors clientCtx so a disconnect cancels the work; on detach the
// watcher exits and the finalizer goroutine owns workCtx until fn returns.
func (rt *sessionFlowRuntime[Stream, State]) run(
	clientCtx context.Context,
	fn SessionFlowFunc[Stream, State],
) (*SessionFlowOutput[State], error) {
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
		result, err := fn(workCtx, rt.router.responder(), rt.runner)
		rt.fnDone <- fnDoneResult[State]{result: result, err: err}
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
// pending snapshot), an aborter (to cancel mid-flight), and a status
// subscriber (so the runtime can react to the abort without polling).
func (rt *sessionFlowRuntime[Stream, State]) checkDetachCapabilities() error {
	if rt.cfg.store == nil {
		return core.NewError(core.FAILED_PRECONDITION,
			"session flow %q: detach requires a session store", rt.name)
	}
	if _, ok := rt.cfg.store.(SnapshotAborter); !ok {
		return core.NewError(core.FAILED_PRECONDITION,
			"session flow %q: detach requires a session store implementing SnapshotAborter", rt.name)
	}
	if _, ok := rt.cfg.store.(SnapshotStatusSubscriber); !ok {
		return core.NewError(core.FAILED_PRECONDITION,
			"session flow %q: detach requires a session store implementing SnapshotStatusSubscriber", rt.name)
	}
	return nil
}

// drainAndWait performs a synchronous shutdown: cancel work, wait for the
// intake reader/forwarder to finish, drain fnDone, and close the router.
// Returns the fn's result for callers that need to surface its error.
func (rt *sessionFlowRuntime[Stream, State]) drainAndWait(cancelWork context.CancelFunc) fnDoneResult[State] {
	cancelWork()
	rt.intake.stopAndWait()
	res := <-rt.fnDone
	rt.router.close()
	return res
}

// handleFnDone is the synchronous-completion path: fn returned before any
// detach signal. Capture an invocation-end snapshot if state advanced past
// the last turn-end snapshot, then assemble the output.
func (rt *sessionFlowRuntime[Stream, State]) handleFnDone(
	ctx context.Context,
	cancelWork context.CancelFunc,
	res fnDoneResult[State],
) (*SessionFlowOutput[State], error) {
	cancelWork()
	rt.intake.stopAndWait()
	rt.router.close()

	if res.err != nil {
		return nil, res.err
	}

	snapshotID := rt.runner.maybeSnapshot(ctx, SnapshotEventInvocationEnd)
	if snapshotID == "" && rt.runner.lastSnapshot != nil {
		// State unchanged since the last turn-end snapshot — reuse it so
		// the response always carries an ID when a store is configured.
		snapshotID = rt.runner.lastSnapshot.SnapshotID
	}

	out := &SessionFlowOutput[State]{SnapshotID: snapshotID}
	if res.result != nil {
		out.Message = res.result.Message
		out.Artifacts = res.result.Artifacts
	}
	if rt.cfg.store == nil {
		out.State = applyTransform(rt.cfg.transform, rt.session.State())
	}
	return out, nil
}

// handleDetach commits the pending snapshot, returns its ID, and spawns the
// status-subscriber and finalizer goroutines that own the rest of the
// invocation. Per-turn snapshots are suspended for the remainder, and the
// chunk router enters trash mode (no further outCh writes, no side effects).
//
// PendingInputs is FIFO: in-flight input (atomically captured with the
// suspend flag, so it's never both turn-end-snapshotted and listed here),
// followed by the intake's queue, followed by anything the reader drains
// from src after detach.
func (rt *sessionFlowRuntime[Stream, State]) handleDetach(
	clientCtx, workCtx context.Context,
	cancelWork context.CancelFunc,
	markDetached func(),
) (*SessionFlowOutput[State], error) {
	// Stop mirroring clientCtx. From here, only the abort subscription or
	// fn completion can cancel workCtx.
	markDetached()

	combined := rt.intake.suspendAndCapture()

	parentID := ""
	if rt.runner.lastSnapshot != nil {
		parentID = rt.runner.lastSnapshot.SnapshotID
	}

	now := time.Now()
	pending := &SessionSnapshot[State]{
		SnapshotID:    uuid.New().String(),
		ParentID:      parentID,
		CreatedAt:     now,
		UpdatedAt:     now,
		Event:         SnapshotEventDetach,
		Status:        SnapshotStatusPending,
		PendingInputs: combined,
	}
	if err := rt.cfg.store.SaveSnapshot(clientCtx, pending); err != nil {
		rt.drainAndWait(cancelWork)
		return nil, core.NewError(core.INTERNAL,
			"session flow %q: failed to write pending snapshot: %v", rt.name, err)
	}

	// The router can no longer write to outCh once we return; the bidi
	// framework closes it shortly after. The router stops writing and
	// trashes any further chunks.
	rt.router.stopAndWait()

	canceledByUser := &atomic.Bool{}
	subCtx, stopSub := context.WithCancel(workCtx)
	subscriber := rt.cfg.store.(SnapshotStatusSubscriber)
	statusCh := subscriber.OnSnapshotStatusChange(subCtx, pending.SnapshotID)
	go func() {
		for status := range statusCh {
			if status == SnapshotStatusCanceled {
				canceledByUser.Store(true)
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
		rt.finalizePendingSnapshot(finalizeCtx, pending, res.err, canceledByUser.Load())
		cancelWork()
	}()

	return &SessionFlowOutput[State]{SnapshotID: pending.SnapshotID}, nil
}

// finalizePendingSnapshot rewrites the pending snapshot row with the
// terminal state and status. canceledByUser distinguishes a context
// cancellation from abortSnapshot (status=canceled) from an internal
// failure (status=error). Re-checks the store's current status before
// writing so an abort that lands between fn-return and this write is
// honored rather than clobbered with status=complete.
func (rt *sessionFlowRuntime[Stream, State]) finalizePendingSnapshot(
	ctx context.Context,
	pending *SessionSnapshot[State],
	fnErr error,
	canceledByUser bool,
) {
	if !canceledByUser {
		if snap, err := rt.cfg.store.GetSnapshot(ctx, pending.SnapshotID); err == nil && snap != nil && snap.Status == SnapshotStatusCanceled {
			canceledByUser = true
		}
	}

	status := SnapshotStatusComplete
	errMsg := ""
	switch {
	case canceledByUser:
		status = SnapshotStatusCanceled
		if fnErr != nil {
			errMsg = fnErr.Error() // canceled status takes precedence; preserve text
		}
	case fnErr != nil:
		status = SnapshotStatusError
		errMsg = fnErr.Error()
	}

	final := &SessionSnapshot[State]{
		SnapshotID: pending.SnapshotID,
		ParentID:   pending.ParentID,
		CreatedAt:  pending.CreatedAt,
		UpdatedAt:  time.Now(),
		Event:      SnapshotEventDetach,
		Status:     status,
		Error:      errMsg,
		State:      *rt.session.State(),
	}
	if err := rt.cfg.store.SaveSnapshot(ctx, final); err != nil {
		logger.FromContext(ctx).Error("session flow: failed to finalize pending snapshot",
			"snapshotId", pending.SnapshotID, "err", err)
	}
}

// loadSession constructs a Session from the invocation's init payload,
// loading from the store when a snapshot ID is provided. Returns the
// snapshot too so the runtime can chain ParentID off it.
func loadSession[State any](
	ctx context.Context,
	init *SessionFlowInit[State],
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
			s.state = *init.State
		}
		return s, nil, nil
	}

	if store == nil {
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot ID %q provided but no session store configured", init.SnapshotID)
	}
	snap, err := store.GetSnapshot(ctx, init.SnapshotID)
	if err != nil {
		return nil, nil, core.NewError(core.INTERNAL, "failed to load snapshot %q: %v", init.SnapshotID, err)
	}
	if snap == nil {
		return nil, nil, core.NewError(core.NOT_FOUND, "snapshot %q not found", init.SnapshotID)
	}
	switch snap.Status {
	case SnapshotStatusError:
		msg := snap.Error
		if msg == "" {
			msg = "snapshot recorded an error"
		}
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q terminated with error: %s", init.SnapshotID, msg)
	case SnapshotStatusPending:
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q is still pending; wait for it to finalize before resuming", init.SnapshotID)
	case SnapshotStatusCanceled:
		return nil, nil, core.NewError(core.FAILED_PRECONDITION,
			"snapshot %q was canceled", init.SnapshotID)
	}
	s.state = snap.State
	return s, snap, nil
}

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

	// intake is the source of truth for in-flight tracking, queue state,
	// and suspended state. The runner consults it via beginTurnEnd (in
	// maybeSnapshot) so per-turn snapshot writes and detach captures
	// cannot race over the same input.
	intake *detachIntake
}

// Run loops over the input channel, calling fn for each turn. Each turn is
// wrapped in a trace span for observability. Input messages are automatically
// added to the session before fn is called. After fn returns successfully, a
// TurnEnd chunk is sent and a snapshot check is triggered.
func (s *SessionRunner[State]) Run(ctx context.Context, fn func(ctx context.Context, input *SessionFlowInput) error) error {
	for input := range s.InputCh {
		spanMeta := &tracing.SpanMetadata{
			Name:    fmt.Sprintf("sessionFlow/turn/%d", s.TurnIndex),
			Type:    "flowStep",
			Subtype: "flowStep",
		}
		_, err := tracing.RunInNewSpan(ctx, spanMeta, input,
			func(ctx context.Context, input *SessionFlowInput) (any, error) {
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

// Result returns an [SessionFlowResult] populated from the current session state:
// the last message in the conversation history and all artifacts.
// It is a convenience for custom session flows that don't need to construct the
// result manually.
func (s *SessionRunner[State]) Result() *SessionFlowResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := &SessionFlowResult{}
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
// For turn-end events, the runner consults the intake atomically: if
// detach has suspended snapshots, intake.beginTurnEnd reports that and
// the runner skips. Otherwise intake.beginTurnEnd clears the in-flight
// input. This is what guarantees the in-flight input is never both
// turn-end-snapshotted AND included in the pending snapshot's
// PendingInputs.
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

	now := time.Now()
	snapshot := &SessionSnapshot[State]{
		SnapshotID: uuid.New().String(),
		CreatedAt:  now,
		UpdatedAt:  now,
		Event:      event,
		Status:     SnapshotStatusComplete,
		State:      currentState,
	}
	if s.lastSnapshot != nil {
		snapshot.ParentID = s.lastSnapshot.SnapshotID
	}

	if err := s.store.SaveSnapshot(ctx, snapshot); err != nil {
		logger.FromContext(ctx).Error("session flow: failed to save snapshot", "err", err)
		return ""
	}

	s.tagLastMessage(snapshot.SnapshotID)
	s.lastSnapshot = snapshot
	s.lastSnapshotVersion = currentVersion
	return snapshot.SnapshotID
}

// tagLastMessage records snapshotID in the metadata of the last message in
// session history, so clients can correlate messages with the snapshot that
// captured them.
func (s *SessionRunner[State]) tagLastMessage(snapshotID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	msgs := s.state.Messages
	if len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	if last.Metadata == nil {
		last.Metadata = make(map[string]any)
	}
	last.Metadata["snapshotId"] = snapshotID
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
// If an artifact with the same name already exists in the session, it is
// replaced. The session-level side effect happens whether or not detach
// has landed; only the wire forward to the client is suppressed
// post-detach, when there is no longer a client to receive it.
func (r Responder[Stream]) SendArtifact(artifact *Artifact) {
	r <- &SessionFlowStreamChunk[Stream]{Artifact: artifact}
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
	in      chan *SessionFlowStreamChunk[Stream]
	out     chan<- *SessionFlowStreamChunk[Stream]
	session *Session[State]

	turnMu     sync.Mutex
	turnChunks []*SessionFlowStreamChunk[Stream]

	done          chan struct{}
	stopWriting   chan struct{}
	writerStopped chan struct{}
}

func startChunkRouter[Stream, State any](
	session *Session[State],
	out chan<- *SessionFlowStreamChunk[Stream],
) *chunkRouter[Stream, State] {
	r := &chunkRouter[Stream, State]{
		in:            make(chan *SessionFlowStreamChunk[Stream]),
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
func (r *chunkRouter[Stream, State]) applySideEffects(chunk *SessionFlowStreamChunk[Stream]) {
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
// flow function (e.g. the runtime's emitTurnEnd).
func (r *chunkRouter[Stream, State]) send() chan<- *SessionFlowStreamChunk[Stream] {
	return r.in
}

// collectTurnChunks returns and resets accumulated turn chunks.
func (r *chunkRouter[Stream, State]) collectTurnChunks() []*SessionFlowStreamChunk[Stream] {
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
// and is the single source of truth for in-flight tracking, queue state,
// and suspend state.
//
// The reader goroutine pulls from the bidi framework's inCh as soon as
// inputs arrive and appends them to an internal queue. This is what makes
// detach detection immediate: the moment an input with
// [SessionFlowInput.Detach] lands in src, the reader sees it without
// waiting for the runner to finish whatever it's processing.
//
// The forwarder goroutine pops the queue and writes to dst, blocking on
// the runner. Crucially, it sets in-flight (under the same mutex it pops
// under) BEFORE the dst send, so an input is never "in transit" without
// being tracked — closing the gap that would otherwise let detach miss it.
//
// The runner asks beginTurnEnd at the end of each turn: if not suspended,
// in-flight is cleared. If suspended, the runner skips its snapshot and
// the input rolls into the pending snapshot instead.
//
// suspendAndCapture is the detach handler's atomic read: flips suspended,
// reads in-flight, reads queue, all under one lock so there's no race
// with maybeSnapshot or the forwarder.

type detachIntake struct {
	src    <-chan *SessionFlowInput
	dst    chan *SessionFlowInput
	notify chan struct{} // buffered size 1; wakes forwarder when queue grows

	// turnDone is signaled by beginTurnEnd to release the forwarder so it
	// may pop the next input. Initialized with one token so the very
	// first turn can start without a preceding turn end.
	turnDone chan struct{}

	mu        sync.Mutex
	suspended bool
	inFlight  *SessionFlowInput
	queue     []*SessionFlowInput

	readDone atomic.Bool
	detachCh chan struct{} // signaled by reader when detach observed

	stop     chan struct{}
	stopOnce sync.Once
	done     chan struct{}
}

func startDetachIntake(src <-chan *SessionFlowInput) *detachIntake {
	i := &detachIntake{
		src:      src,
		dst:      make(chan *SessionFlowInput),
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

func (i *detachIntake) enqueue(input *SessionFlowInput) {
	i.mu.Lock()
	i.queue = append(i.queue, input)
	i.mu.Unlock()
	i.signal()
}

// handleDetach drains any buffered src inputs into the queue and signals
// the detach handler. The detach handler then calls suspendAndCapture for
// the atomic read of the full state.
//
// A pure detach signal (no Messages, no ToolRestarts) is dropped rather
// than enqueued: it carries no payload to process, so adding it to
// PendingInputs would just leave a stray empty input there. Callers that
// want to ride a final input on the detach signal can do so by calling
// Send(&SessionFlowInput{Detach: true, Messages: ...}) explicitly.
func (i *detachIntake) handleDetach(first *SessionFlowInput) {
	var drained []*SessionFlowInput
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
// otherwise process. Used to filter pure detach signals out of
// PendingInputs.
func hasInputPayload(in *SessionFlowInput) bool {
	return in != nil && (len(in.Messages) > 0 || len(in.ToolRestarts) > 0)
}

// forward pops the queue and writes to dst at the runner's pace. The
// runner signals turnDone via beginTurnEnd when it's ready for the next
// input; until then the forwarder waits, so inFlight always reflects the
// input actually being processed (or in transit to the runner) rather
// than running ahead.
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
// stop is signaled. Returns the popped input (with inFlight set under mu)
// or nil if no further inputs will arrive.
func (i *detachIntake) awaitInput() *SessionFlowInput {
	for {
		i.mu.Lock()
		if len(i.queue) > 0 {
			input := i.queue[0]
			i.queue = i.queue[1:]
			i.inFlight = input
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

func (i *detachIntake) out() <-chan *SessionFlowInput {
	return i.dst
}

func (i *detachIntake) detachSignal() <-chan struct{} {
	return i.detachCh
}

// beginTurnEnd is called by [SessionRunner.maybeSnapshot] before writing
// a turn-end snapshot. If the intake has been suspended (detach landed),
// it returns suspended=true and the runner skips the snapshot. Otherwise
// it clears in-flight.
//
// In all cases (including suspended) the forwarder is released so it can
// pop the next queued input — suspension stops snapshot writing, not
// processing.
func (i *detachIntake) beginTurnEnd() (suspended bool) {
	i.mu.Lock()
	if i.suspended {
		i.mu.Unlock()
		i.releaseForward()
		return true
	}
	i.inFlight = nil
	i.mu.Unlock()
	i.releaseForward()
	return false
}

// suspendAndCapture is called once by the detach handler. It flips
// suspended=true and reads the in-flight input (if any) and the queue
// snapshot. Inputs are returned in FIFO order with Detach cleared.
func (i *detachIntake) suspendAndCapture() (combined []*SessionFlowInput) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.suspended = true

	if i.inFlight != nil {
		c := *i.inFlight
		c.Detach = false
		combined = append(combined, &c)
	}
	for _, q := range i.queue {
		c := *q
		c.Detach = false
		combined = append(combined, &c)
	}
	return
}

// stopAndWait forces the intake to exit and waits for both reader and
// forwarder goroutines.
func (i *detachIntake) stopAndWait() {
	i.stopOnce.Do(func() { close(i.stop) })
	<-i.done
}

// --- SessionFlow client API ---

// StreamBidi starts a new session flow invocation with bidirectional streaming.
// Use this for multi-turn interactions where you need to send multiple inputs
// and receive streaming chunks. For single-turn usage, see Run and RunText.
func (af *SessionFlow[Stream, State]) StreamBidi(
	ctx context.Context,
	opts ...InvocationOption[State],
) (*SessionFlowConnection[Stream, State], error) {
	init, err := af.resolveOptions(opts)
	if err != nil {
		return nil, err
	}
	conn, err := af.flow.StreamBidi(ctx, init)
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
	// If the bidi function fails fast (e.g. resuming from an errored
	// snapshot rejects in newSessionFlowRuntime), Send / Close / Receive
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
	cfg := &invocationOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyInvocation(cfg); err != nil {
			return nil, fmt.Errorf("SessionFlow %q: %w", af.flow.Name(), err)
		}
	}
	init := &SessionFlowInit[State]{
		SnapshotID: cfg.snapshotID,
		State:      cfg.state,
	}
	if cfg.promptInput != nil {
		if init.State == nil {
			init.State = &SessionState[State]{}
		}
		init.State.InputVariables = cfg.promptInput
	}
	return init, nil
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

// Detach asks the server to write a pending snapshot capturing any inputs
// already buffered, close the connection, and continue processing in the
// background. Output() returns the pending snapshot ID; the client can
// later call AbortSnapshot to stop the background work or GetSnapshot to
// observe its progression. Streamed chunks emitted after detach are not
// forwarded over the wire (the connection is gone), but their session-
// level side effects still apply: artifacts sent via [Responder.SendArtifact]
// land in the session and end up in the final snapshot's state.
//
// To send a final input as part of the same wire message, use
// Send(&SessionFlowInput{Detach: true, Messages: ...}) directly.
func (c *SessionFlowConnection[Stream, State]) Detach() error {
	return c.conn.Send(&SessionFlowInput{Detach: true})
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

// Output returns the final response after the session flow completes.
//
// Unlike the underlying BidiConnection, Output waits for the flow to
// finalize before returning. This is important for detached invocations:
// when the client sends Detach, the flow function returns promptly with a
// pending snapshot ID, and callers need to observe that output rather than
// the context cancellation error.
func (c *SessionFlowConnection[Stream, State]) Output() (*SessionFlowOutput[State], error) {
	<-c.conn.Done()
	return c.conn.Output()
}

// Done returns a channel closed when the connection completes.
func (c *SessionFlowConnection[Stream, State]) Done() <-chan struct{} {
	return c.conn.Done()
}

// --- DefineSessionFlowFromPrompt ---

// promptMessageKey tags prompt-rendered messages so they can be excluded
// from session history after generation. They're rendered fresh each turn
// from the registered prompt, so persisting them in history would cause
// duplication on resume.
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

	turn := func(ctx context.Context, resp Responder[any], sess *SessionRunner[State], input *SessionFlowInput) error {
		genOpts, err := renderPromptForTurn(ctx, p, sess, defaultInput)
		if err != nil {
			return err
		}

		if len(input.ToolRestarts) > 0 {
			for _, part := range input.ToolRestarts {
				if !part.IsToolRequest() {
					return core.NewError(core.INVALID_ARGUMENT, "ToolRestarts: part is not a tool request")
				}
			}
			genOpts.Resume = ai.NewResume(input.ToolRestarts, nil)
		}

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
		// messages. This captures intermediate tool call/response messages
		// from the tool loop, not just the final response.
		if modelResp.Request != nil {
			var msgs []*ai.Message
			for _, m := range modelResp.History() {
				if m.Metadata[promptMessageKey] == true {
					continue
				}
				msgs = append(msgs, m)
			}
			sess.SetMessages(msgs)
		} else if modelResp.Message != nil {
			sess.AddMessages(modelResp.Message)
		}

		// Stream interrupt parts so the client can detect and handle them
		// (e.g. prompt the user for confirmation).
		if modelResp.FinishReason == ai.FinishReasonInterrupted {
			if parts := modelResp.Interrupts(); len(parts) > 0 {
				resp.SendModelChunk(&ai.ModelResponseChunk{
					Role:    ai.RoleTool,
					Content: parts,
				})
			}
		}
		return nil
	}

	fn := func(ctx context.Context, resp Responder[any], sess *SessionRunner[State]) (*SessionFlowResult, error) {
		err := sess.Run(ctx, func(ctx context.Context, input *SessionFlowInput) error {
			return turn(ctx, resp, sess, input)
		})
		if err != nil {
			return nil, err
		}
		return sess.Result(), nil
	}

	return DefineSessionFlow(r, promptName, fn, opts...)
}

// renderPromptForTurn renders the prompt with the active input variables
// (session override > default), tags the prompt-rendered messages so they
// can be excluded from history, and appends conversation history.
func renderPromptForTurn[State, PromptIn any](
	ctx context.Context,
	p *ai.DataPrompt[PromptIn, string],
	sess *SessionRunner[State],
	defaultInput PromptIn,
) (*ai.GenerateActionOptions, error) {
	promptInput := defaultInput
	if stored := sess.InputVariables(); stored != nil {
		typed, ok := base.ConvertTo[PromptIn](stored)
		if !ok {
			return nil, core.NewError(core.INVALID_ARGUMENT,
				"input variables type mismatch: got %T, want %T", stored, promptInput)
		}
		promptInput = typed
	}

	genOpts, err := p.Render(ctx, promptInput)
	if err != nil {
		return nil, fmt.Errorf("prompt render: %w", err)
	}

	for _, m := range genOpts.Messages {
		if m.Metadata == nil {
			m.Metadata = make(map[string]any)
		}
		m.Metadata[promptMessageKey] = true
	}

	genOpts.Messages = append(genOpts.Messages, sess.Messages()...)
	return genOpts, nil
}
