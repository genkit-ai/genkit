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

// Package exp provides experimental AI primitives for Genkit.
//
// APIs in this package are under active development and may change in any
// minor version release.
package exp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"reflect"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/internal/genkitbridge"
)

// --- Heartbeat ---

// A detached (background) turn refreshes its pending snapshot's heartbeat on an
// interval so a reader can tell a live background worker from a dead one. Each
// beat is a store write; if the beats stop (the worker died) the heartbeat goes
// stale and reads surface the pending snapshot as [SnapshotStatusExpired]
// rather than leaving it pending forever.
const (
	// defaultHeartbeatInterval is how often a detached turn refreshes its
	// pending snapshot's heartbeat.
	defaultHeartbeatInterval = 30 * time.Second
	// defaultHeartbeatTimeout is the staleness threshold after which a pending
	// snapshot whose heartbeat has not advanced is reported as expired on read.
	// It is comfortably larger than defaultHeartbeatInterval so a single missed
	// beat does not trip expiry.
	defaultHeartbeatTimeout = 60 * time.Second
	// windDownHeartbeatBudget bounds how long an aborting invocation keeps
	// heartbeating while its worker drains toward the finalize write. The
	// beats are what keep a live, winding-down worker's row reading as
	// aborting instead of expired the moment the abort flip lands: the flip
	// stops the work, not the worker, and a drain through a slow tool call
	// can outlast defaultHeartbeatTimeout. A worker that has not finalized
	// within the budget is presumed wedged; its beats stop, the row goes
	// stale and reads as expired, and the task stays recoverable instead of
	// winding down forever.
	windDownHeartbeatBudget = 5 * time.Minute
)

// settleGrace is how long [AgentConnection.Output] waits for a cancelled
// invocation to produce its result before falling back to reporting the bare
// cancellation. An agent function that honors its context unwinds in well
// under this, and the wait is what lets the caller see the snapshot the
// aborted run stopped at. One that ignores its context has nothing to hand
// back however long the wait, so the grace expires and the caller escapes.
const settleGrace = 2 * time.Second

// isHeartbeatExpired reports whether snap is an in-flight detached snapshot
// (pending, or aborting while its worker drains toward the finalize) whose
// heartbeat is older than timeout, i.e. its background worker is presumed
// dead. A row that has not yet written a first heartbeat is not considered
// expired (the beat may simply not have fired yet).
func isHeartbeatExpired[State any](snap *SessionSnapshot[State], timeout time.Duration) bool {
	if snap.Status.Terminal() || snap.HeartbeatAt == nil {
		return false
	}
	return time.Since(*snap.HeartbeatAt) > timeout
}

// --- SessionRunner ---

// SessionRunner extends Session with agent-runtime functionality:
// turn management, snapshot persistence, and input channel handling.
type SessionRunner[State any] struct {
	*Session[State]

	// inputCh delivers per-turn inputs from the client; consumed by Run.
	inputCh <-chan *AgentInput
	// turnIndex is the zero-based index of the current conversation turn,
	// incremented by Run after each turn completes.
	turnIndex int
	// turnSnapshotID is the ID reserved at the start of the current turn that
	// its turn-end snapshot is saved under (server-managed only; "" for a
	// client-managed agent). Reserved up front so the per-turn fn can read it
	// from its [TurnContext]. Written by Run at turn start and read by
	// snapshotTurnEnd at turn end, both in the fn goroutine, so it needs no
	// lock (same confinement as turnIndex).
	turnSnapshotID string

	onStartTurn func()
	onEndTurn   func(ctx context.Context)

	// snapMu serializes the turn-end snapshot write (snapshotTurnEnd)
	// against the detach handler's suspend-and-capture (suspendSnapshots).
	// snapshotsSuspended and lastSnapshotID are written under it; the
	// terminal paths that read lastSnapshotID without it (handleFnDone,
	// failedOutput) run after fn completes, with a happens-before edge
	// through the fnDone channel.
	snapMu             sync.Mutex
	snapshotsSuspended bool
	// lastSnapshotID is the ID of the most recent turn-end snapshot, or of
	// the snapshot the invocation resumed from until the first turn commits,
	// or "" when no store is configured or nothing has been written yet. It
	// is the parent of the next snapshot and the resume point the failed and
	// detached outputs report.
	lastSnapshotID string

	// lastTurnFinishReason is the finish reason reported by the most recent
	// turn (via the [TurnResult] its callback returned), or "" if the turn
	// reported none. It is written by endTurn before [SessionRunner.onEndTurn]
	// and read by the runtime when emitting the turn-end signal and when
	// defaulting the invocation's finish reason. All accesses are confined
	// to the fn goroutine (Run and its synchronous onEndTurn callback) until
	// fn completes, after which the terminal paths read it with a
	// happens-before edge through the fnDone channel, so no lock is needed.
	// The same confinement applies to lastTurnErr, lastTurnCommitted, and
	// lastGoodState; the terminal paths that read them (handleFnDone and the
	// detach-failure paths) all wait on fnDone first.
	lastTurnFinishReason AgentFinishReason

	// lastTurnErr is the error the most recent turn ended with, or nil when
	// it succeeded. Set by endTurn each turn, and recorded on the turn's
	// snapshot so a client reading the row can branch on the status the
	// failure carried.
	lastTurnErr error

	// lastTurnCommitted reports whether the most recent turn left state
	// worth continuing from. A successful turn always has; a failed one has
	// when its callback returned a [TurnResult] alongside the error. Set by
	// endTurn each turn, and what decides whether the turn snapshots.
	lastTurnCommitted bool

	// lastGoodState is a deep copy of the session state as of the most
	// recent committed turn (or the initial state when no turn has committed
	// yet). It is kept for a client-managed agent (no store), whose failure
	// path returns it inline, and for a detached one, whose finalize writes
	// it to the pending row; both then resume from the last committed turn,
	// excluding a later turn that failed before committing. Nil and unused
	// for an attached server-managed agent, whose failure path returns the
	// last turn snapshot instead. See captureLastGood.
	lastGoodState *SessionState[State]

	// initialState is a deep copy of the state the invocation began with:
	// fresh, client-provided, or loaded from the snapshot it resumed. It is
	// the floor committedState falls to when no turn has committed and there
	// is no snapshot to read one from. Written once by captureInitial and
	// never mutated after.
	initialState *SessionState[State]
}

// suspendSnapshots stops all further turn-end snapshot writes for this
// invocation and returns the ID of the newest persisted snapshot (the
// parent for the detach handler's pending row). Taking snapMu makes the
// two steps atomic with respect to an in-flight turn-end write: a write
// already inside snapshotTurnEnd completes first (so the returned parent
// is current, not stale), and any later turn end observes the suspension
// and skips its write. Called by the detach handler, after which the
// queued inputs roll into a single finalize rewrite of the pending row.
func (s *SessionRunner[State]) suspendSnapshots() (parentID string) {
	s.snapMu.Lock()
	defer s.snapMu.Unlock()
	s.snapshotsSuspended = true
	return s.lastSnapshotID
}

// snapshotsAreSuspended reports whether a detach has stopped turn-end
// snapshot writes. Read by captureLastGood in the fn goroutine, so it takes
// snapMu against the detach handler's write.
func (s *SessionRunner[State]) snapshotsAreSuspended() bool {
	s.snapMu.Lock()
	defer s.snapMu.Unlock()
	return s.snapshotsSuspended
}

// TurnResult is the optional return value of a [SessionRunner.Run] per-turn
// callback. It lets a custom agent report how the turn ended; the framework
// forwards the reason on the turn's [TurnEnd] chunk, persists it on the
// turn-end snapshot, and uses it to default the invocation's finish reason.
//
// Returning nil (or a zero TurnResult) omits the reason: the framework
// performs no implicit inference. A prompt-backed agent populates it
// automatically from the underlying generate response.
//
// Returned alongside an error, it also says the failed turn left state worth
// continuing from, which is what makes the turn snapshot and the session
// resumable; see [SessionRunner.Run].
type TurnResult struct {
	// FinishReason is why this turn ended (e.g. [AgentFinishReasonStop],
	// [AgentFinishReasonInterrupted]). Empty to report no reason.
	FinishReason AgentFinishReason
}

// turnCtxKey carries the current turn's [TurnContext] on the per-turn
// function's context, so a handler reads it via [TurnContextFromContext]
// instead of the framework threading it through the [SessionRunner.Run]
// callback signature.
var turnCtxKey = base.NewContextKey[*TurnContext]()

// TurnContext carries per-turn metadata the runtime exposes to the per-turn
// function through its context. Retrieve it with [TurnContextFromContext].
//
// Its SnapshotID is reserved before the turn runs (for a server-managed agent),
// so a handler can name external, snapshot-correlated resources (e.g. a git
// worktree named after the snapshot) up front and have them line up with the
// snapshot the turn persists at its end.
type TurnContext struct {
	// SnapshotID is the ID the turn-end snapshot will be saved under, minted
	// before the turn runs so the handler knows it in advance. A server-managed
	// turn persists its snapshot under this ID, whether it succeeded or failed.
	// It is empty for a client-managed agent (no store, so no snapshot) and is
	// the reserved-but-unused ID for a turn that writes none (one that failed
	// before committing anything, or one whose snapshots a detach suspended).
	SnapshotID string
	// ParentSnapshotID is the ID of the snapshot this turn continues from: the
	// previous turn's snapshot, or the snapshot the invocation resumed from. It
	// is empty on the first turn of a fresh conversation and for a
	// client-managed agent.
	ParentSnapshotID string
	// TurnIndex is the zero-based index of this turn within the invocation.
	TurnIndex int
}

// TurnContextFromContext returns the [TurnContext] for the turn currently
// executing, or nil if ctx is not a per-turn context (e.g. it was called
// outside the [SessionRunner.Run] callback). The returned pointer is read-only;
// the runtime owns it.
func TurnContextFromContext(ctx context.Context) *TurnContext {
	return turnCtxKey.FromContext(ctx)
}

// turnSpanOutput is the value recorded as a turn span's genkit:output. It
// wraps the committed session state captured at turn end under a "state" key,
// so the span output serializes as {"state": <session state>}. The state is
// raw: a configured [StateTransform] shapes only client-facing surfaces, not
// telemetry or persisted state, so this matches what a server-managed turn
// writes to its turn-end snapshot.
type turnSpanOutput[State any] struct {
	State *SessionState[State] `json:"state"`
}

// Run loops over the input channel, calling fn for each turn. Each turn is
// wrapped in a trace span for observability. Input messages are automatically
// added to the session before fn is called. After fn returns successfully, a
// snapshot check is triggered and a [TurnEnd] chunk (carrying any new
// snapshot's ID) is sent.
//
// fn may return a [TurnResult] to report how the turn ended (e.g. its finish
// reason); returning nil reports nothing. The reason rides the turn's
// [TurnEnd] chunk and is persisted on the turn-end snapshot.
//
// When fn returns an error, Run records the failure, stops looping, and
// returns the error. What it does with the turn's state depends on what fn
// returned beside the error: a [TurnResult] commits the turn, which snapshots
// the session under [SnapshotStatusFailed] with the error on the row, and nil
// rolls it back, leaving the previous snapshot as the resume point. A
// prompt-backed agent commits whenever the generate call produced a partial
// response, because that partial ends at a turn seam and is a conversation
// the caller can continue from; a custom agent commits when it knows the same
// of its own state.
//
// A custom agent may then recover (e.g. call Run again to keep processing
// inputs) or propagate the error out of the agent function, which resolves
// the invocation with a failed [AgentOutput] carrying the error and the
// resume point rather than failing the action.
func (s *SessionRunner[State]) Run(ctx context.Context, fn func(ctx context.Context, input *AgentInput) (*TurnResult, error)) error {
	for input := range s.inputCh {
		// Deep-copy at the framework boundary: an in-process caller
		// retains the pointers it sent (message, resume parts) and may
		// mutate them after Send returns, so everything past this point
		// (trace marshaling, session state, snapshot writes) must work
		// on private memory rather than race the caller.
		input = jsonClone(input)
		// Mark the start of the turn so the first customPatch emitted during
		// it is a whole-document replace that re-bases the client.
		if s.onStartTurn != nil {
			s.onStartTurn()
		}
		// Reserve this turn's snapshot ID before the turn runs so the per-turn
		// fn can read it (with the parent ID and turn index) from its context
		// via [TurnContextFromContext]; the turn-end write persists under it.
		// Empty for a client-managed agent, which writes no snapshot.
		s.turnSnapshotID = s.reserveTurnSnapshotID()
		turnCtx := &TurnContext{
			SnapshotID:       s.turnSnapshotID,
			ParentSnapshotID: s.lastSnapshotID,
			TurnIndex:        s.turnIndex,
		}
		spanMeta := &tracing.SpanMetadata{
			// The turn span's shape is fixed so traces from every SDK
			// line up: name "runTurn-N" (1-indexed) and type flowStep
			// with no subtype (genkit:type only, no
			// genkit:metadata:subtype).
			Name: fmt.Sprintf("runTurn-%d", s.turnIndex+1),
			Type: "flowStep",
		}
		// The result of a turn that errored, captured on the way out of the
		// span so the failure arm below can read it: non-nil commits.
		var failedResult *TurnResult
		_, err := tracing.RunInNewSpan(ctx, spanMeta, input,
			func(ctx context.Context, input *AgentInput) (any, error) {
				// Carry the reserved turn context on the per-turn fn's context
				// rather than threading it through Run's callback signature.
				ctx = turnCtxKey.NewContext(ctx, turnCtx)
				if input.Message != nil {
					// The session owns its history: store a copy so fn's
					// view of the input stays independent of session state.
					s.AddMessages(jsonClone(input.Message))
				}
				tr, err := fn(ctx, input)
				if err != nil {
					failedResult = tr
					return nil, err
				}
				// A returned TurnResult sets the reason, nil reports none.
				var reason AgentFinishReason
				if tr != nil {
					reason = tr.FinishReason
				}
				s.endTurn(ctx, reason, nil, true)
				// The turn span's output is the committed session state at
				// turn end, recorded as {state: ...} (see turnSpanOutput).
				return turnSpanOutput[State]{State: s.State()}, nil
			},
		)
		if err != nil {
			reason := AgentFinishReasonFailed
			if failedResult != nil && failedResult.FinishReason != "" {
				reason = failedResult.FinishReason
			}
			// The caller stopping the run wins over whatever the turn
			// reported. The detached path settles the same race the same
			// way: an abort that lands while a turn is settling keeps the
			// aborted terminal.
			if callerStopped(ctx, err) {
				reason = AgentFinishReasonAborted
			}
			s.endTurn(ctx, reason, err, failedResult != nil)
			return err
		}
	}
	return nil
}

// reserveTurnSnapshotID mints the ID the upcoming turn will persist its
// snapshot under, or "" for a client-managed agent (no store), which writes
// none. The runtime mints it here, rather than letting the store mint at write
// time, so it is known before the turn runs and can ride on the turn's
// [TurnContext]; snapshotTurnEnd then saves under it.
func (s *SessionRunner[State]) reserveTurnSnapshotID() string {
	if s.store == nil {
		return ""
	}
	return uuid.New().String()
}

// endTurn records how the turn ended and runs the shared turn-end tail:
// the turn-end emit, the last-good capture, and the turn advance. cause is
// the turn's error, nil on success; committed says whether its state is a
// resume point (always so on success).
func (s *SessionRunner[State]) endTurn(ctx context.Context, reason AgentFinishReason, cause error, committed bool) {
	s.lastTurnFinishReason = reason
	s.lastTurnErr = cause
	s.lastTurnCommitted = committed
	s.onEndTurn(ctx)
	if committed {
		s.captureLastGood()
	}
	s.turnIndex++
}

// committedState resolves the session state as of the last turn that
// committed. It is what a run stopping mid-turn must land: the live state
// holds the unfinished turn's mutations, and an attached turn sheds them by
// not snapshotting at all.
//
// It always resolves, taking the first source that holds an answer:
//
//   - the live state, when the last turn committed and there is nothing to
//     shed;
//   - the copy captureLastGood took at the last committed turn, which a
//     client-managed agent always has and a detached one has for every turn
//     since the detach;
//   - the last turn-end snapshot, which is where a detached run's turns
//     before the detach went, since suspending them froze lastSnapshotID
//     there;
//   - the state the invocation began with, the answer when nothing has
//     committed at all.
//
// The returned state is the caller's to keep: every branch is a copy or a
// value nothing mutates afterwards.
func (s *SessionRunner[State]) committedState(ctx context.Context) *SessionState[State] {
	if s.lastTurnCommitted {
		return s.State()
	}
	if s.lastGoodState != nil {
		return s.lastGoodState
	}
	if snapped := s.snapshotState(ctx, s.lastSnapshotID); snapped != nil {
		return snapped
	}
	return s.initialState
}

// snapshotState reads the state off a snapshot this session already wrote.
// Returns nil when there is no such snapshot, or when it cannot be read or
// carries no state, which costs committedState this source but never an
// answer.
func (s *SessionRunner[State]) snapshotState(ctx context.Context, snapshotID string) *SessionState[State] {
	if s.store == nil || snapshotID == "" {
		return nil
	}
	snap, err := s.store.GetSnapshot(ctx, snapshotID)
	if err != nil {
		logger.Error(ctx, "agent: failed to read snapshot for committed state",
			"snapshotId", snapshotID, "error", err)
		return nil
	}
	if snap == nil || snap.State == nil {
		return nil
	}
	return jsonClone(snap.State)
}

// captureInitial deep-copies the state the invocation began with. It is the
// floor committedState falls to when nothing has committed and no snapshot
// holds an earlier turn, and, for a client-managed agent, the failure
// fallback until the first turn commits. One copy serves both: neither is
// mutated afterwards, only replaced.
func (s *SessionRunner[State]) captureInitial() {
	s.mu.RLock()
	state := s.copyStateLocked()
	s.mu.RUnlock()
	s.initialState = &state
	if s.store == nil {
		s.lastGoodState = &state
	}
}

// captureLastGood deep-copies the committed session state as the failure
// fallback, excluding the partial mutations of a later turn that failed
// before committing. Called after every committed turn, failed or not.
//
// It is kept for a client-managed agent, whose failed invocation returns it
// inline (see failedOutput), and for a detached one, whose finalize has no
// per-turn snapshot to fall back on because detach suspended them. An
// attached server-managed agent needs neither and pays no per-turn copy: its
// last turn snapshot is the resume point.
func (s *SessionRunner[State]) captureLastGood() {
	if s.store != nil && !s.snapshotsAreSuspended() {
		return
	}
	s.mu.RLock()
	state := s.copyStateLocked()
	s.mu.RUnlock()
	s.lastGoodState = &state
}

// Result returns an [AgentResult] populated from the current session state:
// the last message in the conversation history and all artifacts. The
// returned value is independent of the session; callers may mutate it
// without affecting session state. An artifact sent through the
// [Responder] is visible here as soon as the Send call returns.
//
// It is a convenience for custom agents that don't need to construct the
// result manually.
func (s *SessionRunner[State]) Result() *AgentResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := &AgentResult{}
	if msgs := s.state.Messages; len(msgs) > 0 {
		result.Message = jsonClone(msgs[len(msgs)-1])
	}
	if len(s.state.Artifacts) > 0 {
		result.Artifacts = cloneArtifacts(s.state.Artifacts)
	}
	return result
}

// invocationReason resolves the finish reason for the whole invocation: the
// last turn's reason, unless the agent's result overrides it with a non-empty
// one. Shared by the synchronous-completion and detach-finalize paths.
func (s *SessionRunner[State]) invocationReason(result *AgentResult) AgentFinishReason {
	if result != nil && result.FinishReason != "" {
		return result.FinishReason
	}
	return s.lastTurnFinishReason
}

// snapshotTurnEnd persists a turn-end snapshot capturing the committed
// session state, chained off the previous snapshot via ParentID, and
// returns its ID. The row is saved under the turn's pre-reserved ID
// (turnSnapshotID; see [SessionRunner.Run]), so the ID the per-turn fn read
// from its [TurnContext] is the ID the snapshot persists under. It is a no-op
// returning "" when no store is configured or snapshots have been suspended by
// a detach. finishReason records how the captured turn ended so a resumed task
// can report it, and cause is the turn's error: nil writes
// [SnapshotStatusCompleted], and an error writes [SnapshotStatusFailed] with
// the error on the row for a client to branch on, or [SnapshotStatusAborted]
// when finishReason says the turn was stopped rather than broken. The status
// follows the reason so the two cannot disagree, and Run stamps the aborted
// reason whenever callerStopped says so. That is what makes an attached run's
// abort indistinguishable from a detached one's: both land an aborted row
// holding the work up to the turn that did not finish.
//
// The turn-end snapshot is the agent's only routine persistence point, and
// every committed turn writes one, so the newest snapshot is the resume point
// the failed and detached outputs report. Only a turn that failed before
// committing anything writes none, leaving its predecessor as that point.
//
// The body runs under snapMu so the detach handler's suspend-and-capture
// (suspendSnapshots) cannot interleave with a write: it either waits for
// this write to commit or suspends before it starts. Persistence is
// best-effort: a store failure must not kill the in-flight turn, so it is
// logged and "" is returned.
//
// The write is decoupled from ctx. A turn that ends because the invocation's
// context was cancelled is exactly the turn whose snapshot a client needs, so
// the row must land even though the context it ran under is gone. ctx is
// still what the write is traced and logged under.
func (s *SessionRunner[State]) snapshotTurnEnd(ctx context.Context, finishReason AgentFinishReason, cause error) string {
	if s.store == nil {
		return ""
	}

	s.snapMu.Lock()
	defer s.snapMu.Unlock()
	if s.snapshotsSuspended {
		return ""
	}

	s.mu.RLock()
	state := s.copyStateLocked()
	s.mu.RUnlock()

	parentID := s.lastSnapshotID
	sessionID := s.SessionID()
	// Timestamps are caller-managed (the store persists them verbatim); a fresh
	// turn-end snapshot is created now, so CreatedAt and UpdatedAt are equal.
	now := time.Now()
	snapStatus := SnapshotStatusCompleted
	if cause != nil {
		snapStatus = terminalStatus(finishReason)
	}
	saved, err := s.store.SaveSnapshot(context.WithoutCancel(ctx), s.turnSnapshotID,
		func(_ *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			return &SessionSnapshot[State]{
				SessionID:    sessionID,
				ParentID:     parentID,
				Status:       snapStatus,
				FinishReason: finishReason,
				Error:        convertKeepText(cause),
				State:        &state,
				CreatedAt:    now,
				UpdatedAt:    now,
			}, nil
		})
	if err != nil {
		logger.Error(ctx, "agent failed to save snapshot",
			"parentId", parentID,
			"error", err)
		return ""
	}

	s.lastSnapshotID = saved.SnapshotID
	return saved.SnapshotID
}

// --- Responder ---

// Responder is an agent's output channel to the client. Its Send methods are
// fire-and-forget: they return no error, and the agent function should stop
// producing once its own context is cancelled.
//
// A Send applies its session-level side effects synchronously, so a state read
// ([SessionRunner.Result], [Session.Artifacts]) or turn-end snapshot taken
// afterward observes them. Only the forward to the client is asynchronous, and
// it is dropped once the work context is cancelled (client disconnect, abort,
// or agent completion) or the run has detached; the side effects still apply.
type Responder struct {
	in  chan<- *AgentStreamChunk
	ctx context.Context
	// effects applies the chunk's in-process side effects (adding an
	// artifact chunk's artifact to the session) synchronously in send, in
	// the sender's goroutine, so reads and snapshots that follow a Send
	// cannot miss the chunk.
	effects func(*AgentStreamChunk)
	// detached mirrors the router's ingress latch: once the run detaches,
	// send skips the wire forward (a detached run streams nothing) while
	// still applying side effects.
	detached *atomic.Bool
}

// SendModelChunk sends a generation chunk (token-level streaming).
func (r Responder) SendModelChunk(chunk *ai.ModelResponseChunk) {
	r.send(&AgentStreamChunk{ModelChunk: chunk})
}

// SendArtifact streams an artifact to the client and adds it to the session,
// replacing any existing artifact with the same name. The session keeps a deep
// copy, so later mutations of artifact do not affect session state.
func (r Responder) SendArtifact(artifact *Artifact) {
	r.send(&AgentStreamChunk{Artifact: artifact})
}

// send applies chunk's in-process side effects, then delivers it to the
// router for the wire forward, returning promptly if r.ctx is cancelled.
// Applying side effects synchronously (in the sender's goroutine, before
// the channel send) orders them before everything the caller does after
// Send, so a state read or a turn-end snapshot cannot miss a chunk whose
// Send already returned. Dropping the wire forward on cancel decouples
// fn liveness from the runtime's shutdown choreography: a Send issued
// after workCtx cancellation completes immediately rather than blocking
// on a router that has not yet been put into drain mode by a terminal
// path.
func (r Responder) send(chunk *AgentStreamChunk) {
	if r.effects != nil {
		r.effects(chunk)
	}
	if r.detached != nil && r.detached.Load() {
		return
	}
	select {
	case r.in <- chunk:
	case <-r.ctx.Done():
	}
}

// --- Agent ---

// AgentFunc is the function signature for custom agents. The State type
// parameter is the shape of the conversation's custom state (see
// [SessionState.Custom]). The agent streams output through resp and reads or
// mutates state through sess; mutating custom state via [Session.UpdateCustom]
// automatically streams a [AgentStreamChunk.CustomPatch] delta to the client.
type AgentFunc[State any] = func(ctx context.Context, resp Responder, sess *SessionRunner[State]) (*AgentResult, error)

// Agent is a bidirectional streaming agent with automatic snapshot management.
//
// Agent implements [api.BidiAction], so generic transports accept it directly
// (e.g. pass it to genkit.Handler to serve it over HTTP, one turn per request).
// [Agent.Run], [Agent.RunText], and [Agent.Connect] are typed conveniences
// over the same underlying action.
//
// Server-managed agents (those with a [SessionStore] configured) also
// register companion actions for the snapshot lifecycle, available via
// [Agent.GetSnapshotAction], [Agent.WaitForSnapshotAction], and
// [Agent.AbortAction] for serving alongside the agent, and expose the store
// itself via [Agent.Store].
type Agent[State any] struct {
	action *core.BidiAction[*AgentInput, *AgentOutput[State], *AgentStreamChunk, *AgentInit[State]]
	// Companion actions, retained so transports can serve them without a
	// registry lookup. Nil when the corresponding capability is absent;
	// see newSnapshotActions.
	getSnapshot api.Action
	wait        api.Action
	abort       api.Action
	// store is the configured session store, or nil for a client-managed
	// agent. Retained so callers can reach it via Store without threading
	// a separate reference.
	store SessionStore[State]
	// transform shapes session state on the way out to a client; see
	// [WithStateTransform]. Retained so the typed read facade ([Agent.GetSnapshot],
	// [Agent.GetLatestSnapshot]) applies it, matching the getSnapshot companion
	// action. Nil when none was configured.
	transform StateTransform[State]
}

// Name returns the agent's registered name. This is also the name under
// which any inline-defined prompt and companion actions (getSnapshot,
// waitForSnapshot, abort) are registered.
func (a *Agent[State]) Name() string {
	return a.action.Name()
}

// GetSnapshotAction returns the agent's getSnapshot companion action,
// which fetches a session snapshot by ID (input [GetSnapshotRequest],
// output [SessionSnapshot]). It returns nil when the agent is
// client-managed (no [SessionStore] configured): there is no server-side
// snapshot to fetch.
//
// Use it to expose snapshot polling over a transport (e.g. mount it with
// genkit.Handler next to the agent itself); local Go code should use
// [Agent.GetSnapshot], which applies the configured state transform.
func (a *Agent[State]) GetSnapshotAction() api.Action {
	return a.getSnapshot
}

// WaitForSnapshotAction returns the agent's waitForSnapshot companion action,
// which resolves a snapshot the same way getSnapshot does and returns once it
// settles (input [GetSnapshotRequest], output [SessionSnapshot]). It returns
// nil when the agent is client-managed (no [SessionStore] configured).
//
// Use it to expose following a detached invocation over a transport (e.g.
// mount it with genkit.Handler next to the agent itself), so a remote caller
// waits in one request instead of a read per tick; local Go code should use
// [Agent.WaitForSnapshot], which applies the configured state transform.
func (a *Agent[State]) WaitForSnapshotAction() api.Action {
	return a.wait
}

// AbortAction returns the agent's abort companion action,
// which asks the background work behind a pending snapshot (e.g. a
// detached invocation) to stop (input [AgentAbortRequest], output
// [AgentAbortResponse]). It returns nil when the agent has no
// [SessionStore] or the store does not implement [SnapshotSubscriber].
//
// Use it to expose aborting over a transport (e.g. mount it with
// genkit.Handler next to the agent itself); local Go code aborts with
// [Agent.Abort]; a store-only caller uses this companion action.
func (a *Agent[State]) AbortAction() api.Action {
	return a.abort
}

// Store returns the [SessionStore] the agent was configured with via
// [WithSessionStore], or nil when the agent is client-managed (no store).
//
// For reads and aborts prefer the typed facade [Agent.GetSnapshot],
// [Agent.GetLatestSnapshot], and [Agent.Abort]: they apply the
// configured [WithStateTransform] and read-time shaping. Store exposes the raw
// backend for advanced use; a direct [SnapshotReader.GetSnapshot] returns
// untransformed state.
//
// The store is returned as the [SessionStore] interface, not its concrete
// type; a caller needing a store-specific capability (e.g.
// [SnapshotSubscriber]) type-asserts for it.
func (a *Agent[State]) Store() SessionStore[State] {
	return a.store
}

// GetSnapshot fetches a session snapshot by ID through the agent, applying the
// configured [WithStateTransform] and the same read-time shaping the getSnapshot
// companion action performs (a pending or aborting row whose heartbeat went
// stale is surfaced as [SnapshotStatusExpired]; an empty status or zero
// UpdatedAt is defaulted).
// Prefer it to reading [Agent.Store] directly, which returns raw, untransformed
// state. Pass [WithMetadataOnly] to read the shaped metadata without the state.
//
// It returns FAILED_PRECONDITION on a client-managed agent (no store) and
// INVALID_ARGUMENT when snapshotID is empty; a missing snapshot is NOT_FOUND.
func (a *Agent[State]) GetSnapshot(ctx context.Context, snapshotID string, opts ...SnapshotReadOption) (*SessionSnapshot[State], error) {
	if a.store == nil {
		return nil, status.Errorf(ErrSessionStoreNotConfigured, "agent %q: GetSnapshot requires a session store", a.Name())
	}
	if snapshotID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: GetSnapshot: snapshotID is required", a.Name())
	}
	req := resolveSnapshotRead(&GetSnapshotRequest{SnapshotID: snapshotID}, opts)
	return readSnapshot(ctx, a.store, a.transform, "getSnapshot", snapshotID, "", req.MetadataOnly)
}

// WaitForSnapshot fetches a session snapshot by ID and blocks until it settles,
// returning the terminal snapshot with the same transform and shaping as
// [Agent.GetSnapshot]. An already-terminal snapshot returns at once. It is how
// an owner follows a detached invocation to its end; a caller holding only the
// agent's name waits through [AgentHandle.WaitForSnapshot] instead.
//
// A snapshot that failed, aborted, or expired is returned like any other, so a
// non-nil error means the wait itself could not proceed: reads failed past the
// wait's transient-retry budget, or ctx ended and its error is returned. Bound
// the wait with [context.WithTimeout] and read the snapshot afterwards to
// learn where it stands.
//
// It returns FAILED_PRECONDITION on a client-managed agent (no store) and
// INVALID_ARGUMENT when snapshotID is empty; a missing snapshot is NOT_FOUND.
func (a *Agent[State]) WaitForSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error) {
	if a.store == nil {
		return nil, status.Errorf(ErrSessionStoreNotConfigured, "agent %q: WaitForSnapshot requires a session store", a.Name())
	}
	if snapshotID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: WaitForSnapshot: snapshotID is required", a.Name())
	}
	return waitSnapshot(ctx, a.store, a.transform, "waitForSnapshot", snapshotID, "")
}

// GetLatestSnapshot fetches a session's most recently created snapshot (whatever
// its status) through the agent, with the same transform, shaping, and
// [SnapshotReadOption] projections as [Agent.GetSnapshot]. It is the
// transform-applying counterpart to [SnapshotReader.GetLatestSnapshot] and
// backs resume-by-session lookups.
//
// It returns FAILED_PRECONDITION on a client-managed agent and INVALID_ARGUMENT
// when sessionID is empty; an unknown session is NOT_FOUND.
func (a *Agent[State]) GetLatestSnapshot(ctx context.Context, sessionID string, opts ...SnapshotReadOption) (*SessionSnapshot[State], error) {
	if a.store == nil {
		return nil, status.Errorf(ErrSessionStoreNotConfigured, "agent %q: GetLatestSnapshot requires a session store", a.Name())
	}
	if sessionID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: GetLatestSnapshot: sessionID is required", a.Name())
	}
	req := resolveSnapshotRead(&GetSnapshotRequest{SessionID: sessionID}, opts)
	return readSnapshot(ctx, a.store, a.transform, "getSnapshot", "", sessionID, req.MetadataOnly)
}

// Abort aborts the detached invocation behind a pending snapshot by
// flipping it to [SnapshotStatusAborting]; the runtime observes the flip,
// cancels the background work, and lands the row as [SnapshotStatusAborted]
// with its state once the work drains. A caller that has only a store (no
// agent) aborts through the abort companion action instead. It is a no-op on
// a missing snapshot (returns "") or an already-settled one (returns the
// existing status), and idempotent on a row already aborting.
//
// It returns FAILED_PRECONDITION on a client-managed agent and INVALID_ARGUMENT
// when snapshotID is empty.
func (a *Agent[State]) Abort(ctx context.Context, snapshotID string) (SnapshotStatus, error) {
	if a.store == nil {
		return "", status.Errorf(ErrSessionStoreNotConfigured, "agent %q: Abort requires a session store", a.Name())
	}
	if snapshotID == "" {
		return "", status.Errorf(status.ErrInvalidArgument, "agent %q: Abort: snapshotID is required", a.Name())
	}
	return abortPendingSnapshot(ctx, a.store, snapshotID)
}

// --- api.BidiAction implementation ---

// Agent is itself an [api.BidiAction]: transports that accept an
// [api.Action] (or [api.BidiAction]) take an Agent directly. The bidi
// methods matter beyond mere interface completeness: generic transports
// type-assert to [api.BidiAction] to route session init (the wire
// counterpart of [WithSessionID], [WithSnapshotID], and [WithState]), so
// satisfying only [api.Action] would silently break session resume.
var _ api.BidiAction = (*Agent[any])(nil)

// Register registers the agent's run action and any companion actions
// (getSnapshot, waitForSnapshot, abort) with the registry. Agents defined via
// [DefineAgent] or [DefineCustomAgent] are already registered; this
// exists so an agent can travel to another registry as a unit. An
// inline-defined prompt does not travel: the agent holds it directly, so
// execution is unaffected, but the prompt action stays in the registry it
// was defined in.
func (a *Agent[State]) Register(r api.Registry) {
	// Register the wrapped bidi action under the agent key, the same way
	// every other action registers itself; the registry holds a uniform
	// api.BidiAction that the reflection servers, ListAgents, and the route
	// builders consume without knowing about the Agent type.
	//
	// The companion actions register independently under their own keys, so
	// registry consumers recover them by key (genkit.LookupAction) rather
	// than by reaching through the agent action; see newSnapshotActions.
	a.action.Register(r)
	for _, companion := range []api.Action{a.getSnapshot, a.wait, a.abort} {
		if companion != nil {
			companion.Register(r)
		}
	}
}

// Desc returns the descriptor of the agent's run action.
func (a *Agent[State]) Desc() api.ActionDesc {
	return a.action.Desc()
}

// RunJSON runs a one-shot invocation with no init (a fresh session):
// input is the turn's [AgentInput] and the result is the final
// [AgentOutput]. To supply a session source, use [Agent.RunBidiJSON].
func (a *Agent[State]) RunJSON(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error) (json.RawMessage, error) {
	return a.action.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [Agent.RunJSON] with trace information on the
// result.
func (a *Agent[State]) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error) (*api.ActionRunResult[json.RawMessage], error) {
	return a.action.RunJSONWithTelemetry(ctx, input, cb)
}

// RunBidiJSON runs a one-shot invocation whose session init (the wire
// counterpart of the [InvocationOption] values) rides in opts: input is
// delivered as the only chunk on the input stream and outgoing chunks are
// forwarded to cb.
func (a *Agent[State]) RunBidiJSON(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error, opts *api.BidiJSONOptions) (*api.ActionRunResult[json.RawMessage], error) {
	return a.action.RunBidiJSON(ctx, input, cb, opts)
}

// ConnectJSON starts a bidirectional streaming session using
// JSON-encoded messages. Local Go callers should prefer the typed
// [Agent.Connect].
func (a *Agent[State]) ConnectJSON(ctx context.Context, opts *api.BidiJSONOptions) (api.BidiJSONConnection, error) {
	return a.action.ConnectJSON(ctx, opts)
}

// DefineAgent defines an agent backed by an inline prompt and registers it. The
// prompt is defined from prompt's [ai.PromptOption] values and registered under
// the agent's name; each turn renders it, appends conversation history, calls
// the model with streaming, and updates session state.
//
// The prompt is an [InlinePrompt], a list of [ai.PromptOption] values:
//
//	agent := DefineAgent(r, "pirate",
//		InlinePrompt{
//			ai.WithModelName("googleai/gemini-flash-latest"),
//			ai.WithSystem("You are a sarcastic pirate."),
//		},
//		WithSessionStore(store),
//	)
//
// State is inferred from the typed agent options (e.g. [WithSessionStore],
// [WithStateTransform]); pass an explicit [State] only when no typed option is
// provided. A typed option that disagrees with the inferred State fails at
// compile time.
//
// To back an agent with a prompt already in the registry (e.g. one from a
// .prompt file), use [DefinePromptAgent]. For full control over the per-turn
// loop, use [DefineCustomAgent].
func DefineAgent[State any](
	r api.Registry,
	name string,
	prompt InlinePrompt,
	opts ...AgentOption[State],
) *Agent[State] {
	p := ai.DefinePrompt(r, name, prompt...)
	return DefineCustomAgent(r, name, agentLoop[State](r, p, nil), opts...)
}

// genkitContextSeed returns a context decorator that seeds the host Genkit
// instance into each agent invocation, so the agent's prompt, tools, and
// middleware can retrieve it via genkit.FromContext and resolve or run other
// actions. The instance is reconstructed from r by the genkit package through
// the internal bridge, so the registry-level constructors below wire seeding up
// themselves and there is no public option for it.
//
// It returns nil when the genkit package is not linked into the build, leaving
// an agent defined directly on a bare registry untouched.
func genkitContextSeed(r api.Registry) func(context.Context) context.Context {
	if genkitbridge.SeedContextForRegistry == nil {
		return nil
	}
	return func(ctx context.Context) context.Context {
		return genkitbridge.SeedContextForRegistry(ctx, r)
	}
}

// DefinePromptAgent defines a prompt-backed agent and registers it, sourcing
// its prompt from the registry by name. Each turn renders the prompt, appends
// conversation history, calls the model with streaming, and updates session
// state, exactly like [DefineAgent].
//
// By default the agent uses the prompt registered under its own name (e.g. one
// defined via [ai.DefinePrompt] or loaded from a .prompt file), so no source
// option is required. Pass [WithNamedPrompt] to reference a differently named
// prompt and supply its render input from code, so a single prompt can back
// many agents.
//
// It is the registry-backed counterpart of [DefineAgent]: where [DefineAgent]
// defines the prompt inline, DefinePromptAgent points at a prompt already in
// the registry. The prompt source is a typed option ([WithNamedPrompt]) rather
// than a positional argument, so it composes with the other agent options
// ([WithSessionStore], [WithStateTransform], [WithStreamTransform],
// [WithDescription]) in a single variadic. For full control over the per-turn
// loop, use [DefineCustomAgent].
//
// State is inferred from the typed agent options; pass an explicit [State] only
// when no typed option provides it (e.g. only [WithNamedPrompt] and
// [WithDescription], whose State cannot be deduced from their arguments).
func DefinePromptAgent[State any](
	r api.Registry,
	name string,
	opts ...PromptAgentOption[State],
) *Agent[State] {
	if seed := genkitContextSeed(r); seed != nil {
		opts = append(opts, &agentOptions[State]{contextFunc: seed})
	}
	cfg := &promptAgentOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyPromptAgent(cfg); err != nil {
			panic(fmt.Errorf("DefinePromptAgent %q: %w", name, err))
		}
	}

	promptName := cfg.promptName
	if promptName == "" {
		promptName = name // default: the prompt registered under the agent's own name
	}
	prompt := ai.LookupPrompt(r, promptName)
	if prompt == nil {
		panic(fmt.Sprintf("DefinePromptAgent %q: prompt %q not found", name, promptName))
	}
	if _, err := prompt.Render(context.Background(), cfg.promptInput); err != nil {
		panic(fmt.Sprintf("DefinePromptAgent %q: prompt input does not satisfy prompt schema: %v", name, err))
	}

	a := newCustomAgent(name, agentLoop[State](r, prompt, cfg.promptInput), &cfg.agentOptions)
	a.Register(r)
	return a
}

// NewCustomAgent creates an agent with full control over the conversation
// loop without registering it. Register it later with the registry (e.g.
// genkit.RegisterAction), which also registers its companion actions; see
// [Agent.Register]. fn receives a [Responder] for streaming output and a
// [SessionRunner] for turn and state management; call [SessionRunner.Run]
// to enter the per-turn loop.
//
// This is the agent counterpart of [core.NewStreamingActionOf]: use it when
// the agent must outlive or precede a registry (e.g. built in a library,
// registered conditionally, or moved between registries). For the common
// case, [DefineCustomAgent] creates and registers in one step.
//
// There is no NewAgent for prompt-backed agents: a prompt is bound to the
// registry it renders against, so it cannot be built before one exists. For
// prompt-like behavior without registration, render and generate with your own
// [genkit.Genkit] inside a custom fn.
func NewCustomAgent[State any](
	name string,
	fn AgentFunc[State],
	opts ...AgentOption[State],
) *Agent[State] {
	cfg := &agentOptions[State]{}
	for _, opt := range opts {
		if err := opt.applyAgent(cfg); err != nil {
			panic(fmt.Errorf("NewCustomAgent %q: %w", name, err))
		}
	}
	return newCustomAgent(name, fn, cfg)
}

// newCustomAgent builds (without registering) an agent from already-applied
// base options. It is the shared core of [NewCustomAgent] and the prompt-backed
// [DefinePromptAgent], which resolve their prompt source into an agentLoop fn
// and reuse the same base option set.
func newCustomAgent[State any](
	name string,
	fn AgentFunc[State],
	cfg *agentOptions[State],
) *Agent[State] {
	// Typed under ActionTypeAgent so agents surface as their own action
	// kind rather than as flows (genkit/exp.ListAgents vs genkit.ListFlows). Built on
	// NewBidiAction so the agent capability metadata is set at construction
	// time; actions must be immutable once registered. WithFlowContext
	// below preserves the flow-context wrapping that makes core.Run work
	// inside fn.
	//
	// metadata["agent"] carries the capability info for the Dev UI;
	// metadata["description"], if set, is lifted to the descriptor's
	// top-level Description by core (see core.newAction), the standard
	// place reflective tooling reads an action's description.
	metadata := map[string]any{"agent": agentMetadataFor(cfg.store)}
	if cfg.description != "" {
		metadata["description"] = cfg.description
	}
	action := core.NewBidiActionOf(api.ActionTypeAgent, name,
		&core.BidiActionOptions{Metadata: metadata},
		func(
			ctx context.Context,
			in *AgentInit[State],
			inCh <-chan *AgentInput,
			outCh chan<- *AgentStreamChunk,
		) (*AgentOutput[State], error) {
			ctx = core.WithFlowContext(ctx, name)
			// Apply any context decorators (e.g. the genkit package seeding its
			// instance) before the runtime derives the per-turn work context, so
			// the decorated values reach each turn's prompt, tools, and middleware.
			if cfg.contextFunc != nil {
				ctx = cfg.contextFunc(ctx)
			}
			rt, err := newAgentRuntime(ctx, name, cfg, in, inCh, outCh)
			if err != nil {
				// Init failures (a rejected init payload, a failed
				// snapshot load) fail the action outright rather than
				// resolving as a failed output: the invocation never
				// reached the input phase of its lifecycle, so there is
				// no conversation state to hand back and nothing to
				// snapshot.
				return nil, err
			}
			return rt.run(ctx, fn)
		})

	getSnapshot, wait, abort := newSnapshotActions(name, cfg.store, cfg.transform)

	return &Agent[State]{
		action:      action,
		getSnapshot: getSnapshot,
		wait:        wait,
		abort:       abort,
		store:       cfg.store,
		transform:   cfg.transform,
	}
}

// DefineCustomAgent defines an agent with full control over the
// conversation loop and registers it (and any companion actions) with the
// registry. fn receives a [Responder] for streaming output and a
// [SessionRunner] for turn and state management; call [SessionRunner.Run]
// to enter the per-turn loop.
//
// It is [NewCustomAgent] followed by [Agent.Register]. To build an agent
// without registering it, use [NewCustomAgent] directly. For agents backed
// by a prompt, use [DefineAgent] instead.
func DefineCustomAgent[State any](
	r api.Registry,
	name string,
	fn AgentFunc[State],
	opts ...AgentOption[State],
) *Agent[State] {
	if seed := genkitContextSeed(r); seed != nil {
		opts = append(opts, &agentOptions[State]{contextFunc: seed})
	}
	a := NewCustomAgent(name, fn, opts...)
	a.Register(r)
	return a
}

// agentMetadataFor derives the [AgentMetadata] value attached to the
// agent's action descriptor under the "agent" key. [AgentMetadata]
// itself is generated from agent.ts; this constructor is hand-written
// because it inspects the configured store's optional capabilities and
// infers the custom-state schema from the State type parameter.
func agentMetadataFor[State any](store SessionStore[State]) AgentMetadata {
	mgmt := AgentStateManagementClient
	abortable := false
	if store != nil {
		mgmt = AgentStateManagementServer
		// Abortable iff the runtime can observe the abort it writes via
		// SaveSnapshot, i.e. the store can subscribe to status changes.
		_, abortable = store.(SnapshotSubscriber)
	}
	return AgentMetadata{
		StateManagement: mgmt,
		Abortable:       abortable,
		StateSchema:     stateSchemaFor[State](),
	}
}

// stateSchemaFor infers the JSON schema for an agent's custom state type,
// the same way core derives an action's input/output schemas. It returns
// nil for an interface State (e.g. Agent[any]), whose zero value carries
// no type information to infer from, so [AgentMetadata.StateSchema] is
// simply omitted rather than advertising an empty object.
func stateSchemaFor[State any]() map[string]any {
	var zero State
	if reflect.ValueOf(&zero).Elem().Kind() == reflect.Interface {
		return nil
	}
	return core.InferSchemaMap(zero)
}

// --- agentRuntime ---

// agentRuntime owns the per-invocation wiring of an agent:
// session, runner, output router, input intake, and the goroutine that runs
// the user fn. Its methods implement the three terminal paths the agent can
// take: detach, fn-completion, and client-cancel.
type agentRuntime[State any] struct {
	name string
	cfg  *agentOptions[State]

	session *Session[State]
	sess    *SessionRunner[State]
	router  *chunkRouter[State]
	patcher *customPatcher[State]
	intake  *detachIntake

	fnDone chan fnDoneResult[State]
	// fatalErr latches the first fail-closed error from a streaming transform
	// (the stream transform in the router, or the state transform behind a
	// custom-state patch). Buffered to one and written non-blocking, so the
	// producer never blocks and only the first error wins; the run loop drains
	// it to resolve the invocation as a failed output. See failTransform.
	fatalErr chan error
}

// failTransform records a fail-closed error from a streaming transform without
// blocking the producer that hit it. The buffered, non-blocking send keeps the
// first error and discards the rest; the run loop observes it (directly via its
// select arm, or after the fact via handleFnDone) and resolves the invocation
// as a failed output. Safe to call from the router and the fn goroutines.
func (rt *agentRuntime[State]) failTransform(err error) {
	select {
	case rt.fatalErr <- err:
	default: // a fatal error is already latched; first one wins
	}
}

// takeFatal returns the latched streaming-transform error, or nil if none.
// Non-blocking, so a terminal path can fold a fatal error that raced fn's
// completion into a failed output.
func (rt *agentRuntime[State]) takeFatal() error {
	select {
	case err := <-rt.fatalErr:
		return err
	default:
		return nil
	}
}

// panicError logs a recovered panic with its stack and returns it as an
// INTERNAL error; what names the code that panicked (e.g. "agent fn"). Call it
// from a deferred recover, where the stack still reaches the panic site. It is
// the shared shape of the runtime's two recover sites: the agent fn and the
// stream transform, both of which contain a panic in user code rather than let
// it crash the process.
func panicError(ctx context.Context, what string, rec any) error {
	logger.Error(ctx, what+" panicked", "panic", rec, "stack", string(debug.Stack()))
	return status.Errorf(status.ErrPanic, "%s panicked: %v", what, rec)
}

// fnDoneResult carries the user fn's return values across the goroutine
// boundary that runs it. A named type keeps the channel signatures readable.
type fnDoneResult[State any] struct {
	result *AgentResult
	err    error
}

// sessionIDSpanAttrKey and snapshotIDSpanAttrKey are the full span-attribute
// keys under which an agent records its identifiers: the session ID on the
// root action span, and the turn-end snapshot ID on each server-managed turn
// span. They are the "genkit:metadata:"-prefixed forms of the
// "agent:sessionId" / "agent:snapshotId" custom-metadata keys every agent
// records; the prefix is inlined here because Go's tracing package exposes no
// custom-metadata helper.
const (
	sessionIDSpanAttrKey  = "genkit:metadata:agent:sessionId"
	snapshotIDSpanAttrKey = "genkit:metadata:agent:snapshotId"
)

func newAgentRuntime[State any](
	ctx context.Context,
	name string,
	cfg *agentOptions[State],
	in *AgentInit[State],
	inCh <-chan *AgentInput,
	outCh chan<- *AgentStreamChunk,
) (*agentRuntime[State], error) {
	session, parent, err := loadSession(ctx, in, cfg.store)
	if err != nil {
		return nil, err
	}

	// The session ID is settled up front, before the agent function runs,
	// so it exists for the whole invocation regardless of when (or
	// whether) the first snapshot is written. It lives inside the session
	// state ([SessionState.SessionID]), so it rides along wherever the
	// state goes: every persisted snapshot's state, and the state
	// returned to (and resent by) client-managed callers.
	if cfg.store != nil {
		// Server-managed: the store row is canonical, overriding whatever the
		// loaded state blob claims (a third-party writer could have let them
		// drift).
		switch {
		case parent != nil && parent.SessionID != "":
			// Resumed an existing chain: inherit its ID.
			session.state.SessionID = parent.SessionID
		case parent == nil && in != nil && in.SessionID != "":
			// No snapshot resolved for the caller-supplied session ID: the
			// client is starting a brand-new conversation under an ID of its
			// own choosing. Honor it for the whole session lifecycle so every
			// snapshot it persists carries it (rather than minting a server ID
			// and stranding the client's ID).
			session.state.SessionID = in.SessionID
		default:
			// Fresh conversation, or one resumed from a snapshot that predates
			// session IDs: mint one.
			session.state.SessionID = uuid.New().String()
		}
	} else if session.state.SessionID == "" {
		// Client-managed: the state object is canonical; keep the ID it
		// carried. Mint one when absent (a fresh conversation) so the
		// output state is self-describing from the first turn and the
		// client can round-trip it without tracking a separate field.
		session.state.SessionID = uuid.New().String()
	}

	// Tag the agent's root action span (the current span here, before any turn
	// span is opened) with the session ID so traces from the same conversation
	// can be correlated. It is written once at the start of the action body.
	// trace.SpanFromContext never returns nil (it yields a no-op span when none
	// is active), so the SetAttributes is always safe.
	trace.SpanFromContext(ctx).SetAttributes(
		attribute.String(sessionIDSpanAttrKey, session.state.SessionID))

	rt := &agentRuntime[State]{
		name:     name,
		cfg:      cfg,
		session:  session,
		fnDone:   make(chan fnDoneResult[State], 1),
		fatalErr: make(chan error, 1),
	}
	// Started after rt exists so the router can signal a fail-closed stream
	// transform error back through rt.failTransform.
	rt.router = startChunkRouter(ctx, session, outCh, cfg.streamTransform, rt.failTransform)
	// Started after the router so the intake's reader can latch the router's
	// ingress filter the moment a detach directive lands, before the inputs
	// riding it reach the runner (see detachIntake.handleDetach).
	rt.intake = startDetachIntake(inCh, rt.router.markDetached)

	rt.sess = &SessionRunner[State]{
		Session: session,
		inputCh: rt.intake.out(),
	}
	if parent != nil {
		// Resumed: chain the first turn's snapshot off the one we loaded, and
		// make it the resume point a first-turn failure falls back to.
		rt.sess.lastSnapshotID = parent.SnapshotID
	}
	rt.sess.onEndTurn = rt.emitTurnEnd
	// Stream custom-state mutations as customPatch chunks. beginTurn is armed
	// per turn by the runner; the session's onCustomChange hook is wired in
	// run, once the work context and responder exist.
	rt.patcher = &customPatcher[State]{
		transform:   cfg.transform,
		session:     session,
		firstInTurn: true,
		fail:        rt.failTransform,
	}
	rt.sess.onStartTurn = rt.patcher.beginTurn
	rt.sess.captureInitial()

	return rt, nil
}

// emitTurnEnd is called by the session after each turn. It paces the
// intake (releasing the forwarder for the next input), writes a turn-end
// snapshot (if applicable), and forwards the resulting [TurnEnd] chunk
// through the router so clients see it on the output stream.
//
// The snapshot sees everything the turn produced: the side effects of
// the turn's Send calls (e.g. artifacts) are applied synchronously in
// [Responder] before each Send returns, and fn returned before this
// runs, so there is no in-flight router work to wait out.
//
// The snapshot is skipped when the turn failed without committing (the live
// state holds mutations nothing can be resumed from) and when detach has
// landed (snapshotTurnEnd observes the suspension under snapMu; the pending
// row already captures the invocation and a single finalize rewrite records
// the cumulative state once the queued inputs drain).
func (rt *agentRuntime[State]) emitTurnEnd(ctx context.Context) {
	rt.intake.releaseForward()
	reason := rt.sess.lastTurnFinishReason
	var snapshotID string
	if rt.sess.lastTurnCommitted {
		snapshotID = rt.sess.snapshotTurnEnd(ctx, reason, rt.sess.lastTurnErr)
	}
	// Tag the turn span with the snapshot it persisted, so a server-managed
	// turn's trace links to its snapshot. ctx is the turn span's context (this
	// runs inside the runTurn-N span via onEndTurn). The ID is empty, and the
	// attribute omitted, when client-managed, when the turn failed without
	// committing, or when a detach suspended snapshots.
	if snapshotID != "" {
		trace.SpanFromContext(ctx).SetAttributes(
			attribute.String(snapshotIDSpanAttrKey, snapshotID))
	}
	rt.router.sendChunk(ctx, &AgentStreamChunk{TurnEnd: &TurnEnd{
		SnapshotID:   snapshotID,
		FinishReason: reason,
	}})
}

// run drives the user fn to completion and returns the agent output.
//
// workCtx carries the session and is decoupled from clientCtx: pre-detach a
// watcher mirrors clientCtx so a disconnect cancels the work; on detach the
// watcher exits and the finalizer goroutine owns workCtx until fn returns.
func (rt *agentRuntime[State]) run(
	clientCtx context.Context,
	fn AgentFunc[State],
) (*AgentOutput[State], error) {
	workCtx, cancelWork := context.WithCancel(context.WithoutCancel(clientCtx))
	workCtx = NewSessionContext(workCtx, rt.session)

	// Wire custom-state streaming now that the work context exists: every
	// UpdateCustom mutation during the invocation emits a customPatch chunk
	// through the same responder fn uses (so the chunk is forwarded on the
	// wire, dropping post-detach like any other chunk). The session mutation
	// itself still applies regardless.
	resp := rt.router.responder(workCtx)
	rt.patcher.bind(workCtx, resp.send)
	rt.session.onCustomChange = rt.patcher.onChange

	var detachOnce sync.Once
	detached := make(chan struct{})
	markDetached := func() { detachOnce.Do(func() { close(detached) }) }
	defer markDetached() // ensure the watcher exits on every return path

	go func() {
		select {
		case <-clientCtx.Done():
			// Arbitrate atomically against markDetached: whichever claims
			// the Once first wins. clientCtx ends not only on a true
			// disconnect but also when the framework releases the action
			// context right after run returns the detached output; by then
			// both select arms are ready and this arm may be picked, so
			// claiming the Once (rather than cancelling outright) is what
			// keeps an already-landed detach's background work alive.
			detachOnce.Do(func() {
				close(detached)
				cancelWork()
			})
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
					fnErr = panicError(workCtx, "agent fn", r)
				}
			}()
			result, fnErr = fn(workCtx, resp, rt.sess)
		}()
		rt.fnDone <- fnDoneResult[State]{result: result, err: fnErr}
	}()

	select {
	case <-rt.intake.detachSignal():
		return rt.handleDetach(clientCtx, workCtx, cancelWork, markDetached)

	case res := <-rt.fnDone:
		// A detach that raced fn's completion still wins: the intake
		// suppressed the wire the moment it read the directive, so
		// resolving this as a synchronous completion would hand back a
		// completed output whose stream was silently truncated. Both arms
		// being ready is a coin flip (select picks among ready cases at
		// random); checking the signal explicitly settles the race at
		// directive-read time, like the JS runtime's input pump.
		select {
		case <-rt.intake.detachSignal():
			// Hand the settled result back to handleDetach's finalizer
			// (or drainAndWait, on the capability-rejection path). fnDone
			// is buffered and fn sends exactly once, so this cannot block.
			rt.fnDone <- res
			return rt.handleDetach(clientCtx, workCtx, cancelWork, markDetached)
		default:
			return rt.handleFnDone(clientCtx, cancelWork, res)
		}

	case cause := <-rt.fatalErr:
		return rt.handleTransformFailure(clientCtx, cancelWork, cause)

	case <-clientCtx.Done():
		res := rt.drainAndWait(cancelWork)
		cause := res.err
		if cause == nil {
			cause = clientCtx.Err()
		}
		// Both, the way ai.Generate hands back a partial response beside the
		// error that ended it: the error is what stopped the run, and the
		// output names the snapshot it stopped at. Returning the error alone
		// left an in-process caller holding nothing to resume from.
		return rt.failedOutput(clientCtx, cause), cause
	}
}

// handleTransformFailure is the fail-closed terminal path for a streaming
// transform that returned an error (or panicked): the stream transform in the
// router, or the state transform behind a custom-state patch. It tears the
// invocation down like a fn error and resolves it as a failed output carrying
// the transform's cause, so no unshaped chunk reaches the client and the
// offending chunk's side effects never surface in a completed output.
//
// drainAndWait cancels the work context (stopping fn), switches the router to
// discard mode, and drains fn; the router has typically already stopped writing
// the moment shape returned the error, but a custom-patch failure trips this
// path while the router is still forwarding, so the stop here is what halts it.
func (rt *agentRuntime[State]) handleTransformFailure(
	clientCtx context.Context,
	cancelWork context.CancelFunc,
	cause error,
) (*AgentOutput[State], error) {
	rt.drainAndWait(cancelWork)
	return rt.failedOutput(clientCtx, cause), disconnectErr(clientCtx, cause)
}

// disconnectErr is the error a terminal path returns beside its failed
// output: the cause when the client is already gone, and nil when it is still
// there to be handed a graceful failure. Both carry the resume point on the
// output either way; the error is what tells an in-process caller that the
// run did not finish on its own terms.
func disconnectErr(clientCtx context.Context, cause error) error {
	if clientCtx.Err() != nil {
		return cause
	}
	return nil
}

// checkDetachCapabilities reports whether the configured store is capable
// of supporting detach. Detach requires a writable store (to persist the
// pending snapshot, and to abort it and refresh its heartbeat via ordinary
// SaveSnapshot writes) and a [SnapshotSubscriber] (so the runtime can observe
// the abort flip and promptly cancel the background work without polling).
func (rt *agentRuntime[State]) checkDetachCapabilities() error {
	if rt.cfg.store == nil {
		return status.Errorf(ErrSessionStoreNotConfigured,
			"agent %q: detach requires a session store", rt.name)
	}
	if _, ok := rt.cfg.store.(SnapshotSubscriber); !ok {
		return status.Errorf(status.ErrFailedPrecondition,
			"agent %q: detach requires a session store implementing SnapshotSubscriber", rt.name)
	}
	return nil
}

// drainAndWait performs a synchronous shutdown: cancel work, stop router
// writes (so a fn mid-send doesn't deadlock once outCh's consumer is
// gone), wait for the intake reader/forwarder to finish, drain fnDone,
// and close the router. Returns the fn's result for callers that need
// to surface its error.
func (rt *agentRuntime[State]) drainAndWait(cancelWork context.CancelFunc) fnDoneResult[State] {
	cancelWork()
	// Switch the router to discard mode before waiting on fn. Without
	// this, a fn mid-send (or a customPatch emit) blocks on the router's
	// r.in receive while the router blocks on r.out send (consumer is
	// gone), so fn never observes ctx and we deadlock waiting on fnDone.
	rt.router.stopAndWait()
	rt.intake.stopAndWait()
	res := <-rt.fnDone
	rt.router.close()
	return res
}

// handleFnDone is the synchronous-completion path: fn returned before any
// detach signal. The output reports the last turn-end snapshot as its
// resume point; there is no separate invocation-end write, so state a
// custom agent mutates after its turn loop rides on the returned output
// but is not persisted. When fn returned an error, the invocation resolves
// gracefully as a failed output instead (see failedOutput).
//
// router.close blocks on the forward goroutine exiting, and fn returning
// does not imply the router is idle: fn's last accepted chunk may still be
// in the router's hands, parked on the send to a full out buffer. On the
// error path stopAndWait closes stopWriting first, deliberately dropping
// the failed turn's in-flight chunks so close cannot wedge behind a
// slow/gone consumer. On the success path those chunks are wanted, so
// close relies on the parked send being released instead: a consuming
// client drains it, a disconnected client's ctx cancellation trips
// forward's ctx arm, and a client that stopped receiving unparks it when
// its Output call drains the stream.
func (rt *agentRuntime[State]) handleFnDone(
	ctx context.Context,
	cancelWork context.CancelFunc,
	res fnDoneResult[State],
) (*AgentOutput[State], error) {
	cancelWork()
	rt.intake.stopAndWait()
	// A custom-state patch whose transform failed closed latches during fn, so
	// it is readable now; a failed turn likewise wants its in-flight chunks
	// dropped. Either way stop router writes before close so it cannot wedge
	// behind a slow or gone consumer. A stream-transform failure instead puts
	// the router into discard mode the instant it occurs (forward never parks),
	// so it needs no stop here and is picked up after close below.
	fatal := rt.takeFatal()
	if res.err != nil || fatal != nil {
		rt.router.stopAndWait()
	}
	rt.router.close()
	if fatal == nil {
		fatal = rt.takeFatal()
	}

	// A streaming transform that failed closed resolves the invocation as
	// failed regardless of what fn returned, so no completed output leaks the
	// data it refused to shape.
	if fatal != nil {
		return rt.failedOutput(ctx, fatal), disconnectErr(ctx, fatal)
	}

	if res.err != nil {
		// The clientCtx.Done arm of the run select handles the common
		// disconnect ordering; disconnectErr guards the race where fn observes
		// the cancellation first and its result wins the select.
		return rt.failedOutput(ctx, res.err), disconnectErr(ctx, res.err)
	}

	// The resume point is the last turn-end snapshot (lastSnapshotID), or ""
	// when no store is configured or no turn committed. A custom agent that
	// overrode the invocation's finish reason on its AgentResult sees it on
	// the output below, but the snapshot keeps the turn's own reason.
	out := &AgentOutput[State]{
		SessionID:    rt.session.SessionID(),
		SnapshotID:   rt.sess.lastSnapshotID,
		FinishReason: rt.sess.invocationReason(res.result),
	}
	if res.result != nil {
		// Deep-copy at the framework boundary so the caller cannot
		// mutate session contents through the returned output, even
		// if a custom fn constructed AgentResult with raw session
		// pointers rather than going through [SessionRunner.Result].
		out.Message = jsonClone(res.result.Message)
		out.Artifacts = cloneArtifacts(res.result.Artifacts)
	}
	if rt.cfg.store == nil {
		// A final-output state transform that fails closed turns the otherwise
		// successful invocation into a failed output, so unshaped state is
		// never handed back.
		state, err := rt.outboundState(ctx, rt.session.State())
		if err != nil {
			if ctx.Err() != nil {
				return nil, err
			}
			return rt.failedOutput(ctx, err), nil
		}
		out.State = state
	}
	return out, nil
}

// outboundState applies the configured state transform and re-stamps the
// framework-owned SessionID, so the state handed to a client-managed
// caller always carries the conversation's identity even if a transform
// rewrote or dropped it. Returns (nil, nil) if state is nil, and a non-nil
// error if the transform failed closed.
func (rt *agentRuntime[State]) outboundState(ctx context.Context, state *SessionState[State]) (*SessionState[State], error) {
	out, err := applyTransform(ctx, rt.cfg.transform, state)
	if err != nil {
		return nil, err
	}
	if out != nil {
		out.SessionID = rt.session.SessionID()
	}
	return out, nil
}

// convertKeepText returns cause as a *status.Error for a persisted payload
// (AgentOutput.Error, SessionSnapshot.Error), preserving both halves of the
// failure: the classification of a buried *status.Error (a store's own status
// survives, per [status.Convert]) and the full chain text of any context
// wrapped around it with fmt.Errorf, which Convert alone would drop. A public
// error is exempt from the text merge: its message may reach a client and must
// stay exactly what was cleared as public. A cause that is itself a non-nil
// interface holding a nil *status.Error still yields a payload, because a
// failed invocation must carry one.
func convertKeepText(cause error) *status.Error {
	e := status.Convert(cause)
	if e == nil {
		if cause == nil {
			return nil
		}
		return status.Errorf(status.ErrInternal, "%w", cause)
	}
	if !e.Public && e.Message != cause.Error() {
		ne := *e
		ne.Message = cause.Error()
		return &ne
	}
	return e
}

// callerStopped reports whether the invocation ended because the caller
// stopped it rather than because something inside it broke. Two roads reach
// the same place:
//
// The context ended. A caller holding a live connection cancels it directly
// or by closing the transport under it, a deadline it set expires, or a
// detached caller calls the abort companion action, which cancels the work
// context on the status flip.
//
// Or the run reached a limit the caller set ([ai.ErrMaxTurnsExceeded]). A turn
// that propagates such an error unchanged is stopped, not broken, and reports
// so without having to say it in a [TurnResult].
//
// Both roads are read from the context and the sentinels, never from the
// classified status, which is a wider set than the caller's own doing: a
// service answering 409 or 504 lands on ABORTED or DEADLINE_EXCEEDED through
// the HTTP mapping in [status]. Persisting that as an aborted row would tell a
// client the run stopped on request when a provider dropped it, which is the
// class a retry loop is most likely to leave alone. Same rule as
// [ai.Generate]'s, so the snapshot status and the partial's finish reason
// agree on who ended the run.
func callerStopped(ctx context.Context, cause error) bool {
	return ctx.Err() != nil ||
		errors.Is(cause, context.Canceled) ||
		errors.Is(cause, context.DeadlineExceeded) ||
		errors.Is(cause, ai.ErrMaxTurnsExceeded)
}

// terminalReason is how an invocation or turn that ended with cause reports
// itself: aborted when the caller stopped it, failed when it broke.
func terminalReason(ctx context.Context, cause error) AgentFinishReason {
	if callerStopped(ctx, cause) {
		return AgentFinishReasonAborted
	}
	return AgentFinishReasonFailed
}

// terminalStatus is the snapshot status that goes with the reason a turn or
// invocation ended on, for the rows that ended with an error. Deriving it
// keeps the two from disagreeing: a row is aborted exactly when it says the
// caller stopped the run.
func terminalStatus(reason AgentFinishReason) SnapshotStatus {
	if reason == AgentFinishReasonAborted {
		return SnapshotStatusAborted
	}
	return SnapshotStatusFailed
}

// failedOutput assembles the output for an invocation that did not run to
// completion: [AgentFinishReasonFailed], or [AgentFinishReasonAborted] when
// the caller stopped it (see callerStopped), the error with its original
// status, and the resume point: the last turn snapshot's ID when server-managed, or
// the last-good state inline when client-managed. Both hold the state
// through the last committed turn, which is the failed turn itself when it
// committed and its predecessor when it did not, since only a committed turn
// snapshots and updates lastGoodState. When no turn committed at all, the
// server-managed ID is "" (or the resumed snapshot's ID) and the
// client-managed state is the initial state. Message and Artifacts are left
// empty; they describe the result of a completed run.
func (rt *agentRuntime[State]) failedOutput(ctx context.Context, cause error) *AgentOutput[State] {
	out := &AgentOutput[State]{
		SessionID:    rt.session.SessionID(),
		FinishReason: terminalReason(ctx, cause),
		Error:        convertKeepText(cause),
	}
	if rt.cfg.store == nil {
		// This is already the failure path, so a transform that also fails
		// closed while shaping the last-good state cannot escalate further:
		// omit state (fail closed, no leak) rather than recurse. The original
		// cause is what the caller needs and is preserved on Error above.
		if state, err := rt.outboundState(ctx, rt.sess.lastGoodState); err != nil {
			logger.Error(ctx,
				"agent state transform failed shaping failed-output state; omitting state",
				"error", err)
		} else {
			out.State = state
		}
	} else {
		out.SnapshotID = rt.sess.lastSnapshotID
	}
	return out
}

// handleDetach resolves an observed detach signal: it rejects the detach
// when the store cannot support it, and otherwise commits the pending
// snapshot, returns its ID, and spawns the status-subscriber and finalizer
// goroutines that own the rest of the invocation. Per-turn snapshots are
// suspended for the remainder so the queued inputs roll into a single
// finalize rewrite.
//
// The router's ingress filter has been latched since the intake read the
// detach directive, so nothing the detached turn produces enters the pipe.
// Filtered chunks keep their in-process side effects (e.g. artifacts added
// via Responder.SendArtifact), which apply at Send time, so user code does
// not have to branch on detach.
func (rt *agentRuntime[State]) handleDetach(
	clientCtx, workCtx context.Context,
	cancelWork context.CancelFunc,
	markDetached func(),
) (*AgentOutput[State], error) {
	if err := rt.checkDetachCapabilities(); err != nil {
		rt.drainAndWait(cancelWork)
		return rt.failedOutput(clientCtx, err), nil
	}

	// Stop mirroring clientCtx. From here, only the abort subscription or
	// fn completion can cancel workCtx.
	markDetached()

	// Atomically suspend per-turn snapshots and capture the chain tip: a
	// turn-end write already in flight commits first (so the pending row
	// chains off the real tip instead of becoming its sibling), and any
	// later turn end skips its write.
	parentID := rt.sess.suspendSnapshots()
	sessionID := rt.session.SessionID()

	// Detach intends to outlive the client connection. If clientCtx was
	// already cancelled (or cancels mid-write), we still want the pending
	// row durable so observers can find it later. Decouple this write.
	//
	// The capability check above guarantees the store is a
	// SnapshotSubscriber, so the runtime can observe the abort flip.
	subscriber := rt.cfg.store.(SnapshotSubscriber)

	// Stamp the pending row's timestamps and an initial heartbeat (refreshed on
	// an interval below). Timestamps are caller-managed; a reader treats a
	// pending snapshot whose heartbeat has gone stale as expired (its background
	// worker is presumed dead).
	//
	// The runtime mints the pending row's ID, as reserveTurnSnapshotID does for
	// turn snapshots, rather than letting the store mint at write time: task
	// handles built on this ID rely on the runtime's UUID format (in
	// particular, it contains no ':'), which a store-minted ID would not
	// guarantee.
	now := time.Now()
	pending, err := rt.cfg.store.SaveSnapshot(context.WithoutCancel(clientCtx), uuid.New().String(),
		func(_ *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			return &SessionSnapshot[State]{
				SessionID:   sessionID,
				ParentID:    parentID,
				Status:      SnapshotStatusPending,
				CreatedAt:   now,
				UpdatedAt:   now,
				HeartbeatAt: &now,
			}, nil
		})
	if err != nil {
		rt.drainAndWait(cancelWork)
		return rt.failedOutput(clientCtx, fmt.Errorf("agent %q: detach: save pending snapshot: %w", rt.name, err)), nil
	}
	// The router can no longer write to outCh once we return; the bidi
	// framework closes it shortly after. Post-detach chunks never entered
	// the pipe (ingress filter), so this only flushes or drops a pre-detach
	// turn's in-flight tail.
	rt.router.stopAndWait()

	// Refresh the heartbeat on an interval, decoupled from clientCtx (the work
	// outlives the client connection); stopped when the turn settles, below.
	hbCtx, stopHeartbeat := context.WithCancel(context.WithoutCancel(clientCtx))
	go rt.runHeartbeat(hbCtx, pending.SnapshotID)

	abortedByUser := &atomic.Bool{}
	subCtx, stopSub := context.WithCancel(workCtx)
	statusCh := subscriber.OnSnapshotStatusChange(subCtx, pending.SnapshotID)
	go func() {
		for status := range statusCh {
			// The runtime's own abort flips the row to aborting; a foreign
			// writer that lands aborted directly still means stop.
			if status != SnapshotStatusAborting && status != SnapshotStatusAborted {
				continue
			}
			abortedByUser.Store(true)
			// The flip stops the work, not the worker: the run is winding
			// down toward its finalize write now, and the heartbeat keeps
			// running so readers see a live aborting row instead of presuming
			// the worker dead the moment the flip lands. The fnDone goroutine
			// below stops the beats right before the finalize; the budget
			// bounds a worker whose fn never observes the cancellation, so a
			// truly wedged run still goes stale and reads as expired. The
			// timer is released as soon as the heartbeat stops for any
			// reason, so a normal wind-down does not keep it (and the
			// contexts it holds) alive for the whole budget.
			budget := time.AfterFunc(windDownHeartbeatBudget, stopHeartbeat)
			context.AfterFunc(hbCtx, func() { budget.Stop() })
			cancelWork()
			return
		}
	}()

	finalizeCtx := context.WithoutCancel(clientCtx)
	go func() {
		res := <-rt.fnDone
		stopSub()
		// The turn has settled; stop refreshing the heartbeat before the
		// finalize write so no beat races it. (A stray beat would be a no-op
		// anyway: the mutator only touches rows still awaiting their
		// finalize, and SaveSnapshot's atomicity keeps either order safe.)
		stopHeartbeat()
		rt.intake.stopAndWait()
		rt.router.close()
		rt.finalizePendingSnapshot(finalizeCtx, pending, res.result, res.err, abortedByUser.Load())
		cancelWork()
	}()

	// The invocation, from the client's perspective, ended by detaching. The
	// pending snapshot is finalized later with how the background work
	// actually ended (see finalizePendingSnapshot).
	return &AgentOutput[State]{
		SessionID:    pending.SessionID,
		SnapshotID:   pending.SnapshotID,
		FinishReason: AgentFinishReasonDetached,
	}, nil
}

// runHeartbeat refreshes the detached snapshot's heartbeat every
// defaultHeartbeatInterval until ctx is cancelled (the turn settled, or the
// wind-down budget elapsed after an abort). A transient store error is logged
// and the loop continues; a persistently failing worker simply stops beating,
// which is exactly the staleness a reader detects as expired.
func (rt *agentRuntime[State]) runHeartbeat(ctx context.Context, snapshotID string) {
	ticker := time.NewTicker(defaultHeartbeatInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := beatHeartbeat(ctx, rt.cfg.store, snapshotID); err != nil {
				logger.Debug(ctx, "agent: heartbeat refresh failed",
					"snapshotId", snapshotID, "error", err)
			}
		}
	}
}

// beatHeartbeat refreshes a detached snapshot's HeartbeatAt via an ordinary
// SaveSnapshot: the mutator carries the existing row through unchanged but for
// HeartbeatAt, so the caller-managed CreatedAt/UpdatedAt are preserved and a
// beat does not register as a state change - no dedicated store method needed.
// A beat lands on the two rows a live worker is still owed a write for: a
// pending one while the detached turn runs, and an aborting one while the
// stopped turn drains toward its finalize. Any other row is settled, and the
// beat no-ops rather than resurrect it; a beat racing the finalize is safe in
// either order (SaveSnapshot is atomic, and a beat after the finalize sees a
// settled row and no-ops). Shared by runHeartbeat and exercised directly in
// tests.
func beatHeartbeat[State any](ctx context.Context, store SnapshotWriter[State], snapshotID string) error {
	now := time.Now()
	_, err := store.SaveSnapshot(ctx, snapshotID,
		func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			if existing == nil || existing.Status.Terminal() {
				return nil, nil
			}
			updated := *existing
			updated.HeartbeatAt = &now
			return &updated, nil
		})
	return err
}

// abortPendingSnapshot flips a pending snapshot to aborting via an ordinary
// SaveSnapshot and returns the resulting status: aborting when the row was
// pending (or already aborting, since a second abort is an idempotent verbatim
// rewrite), the existing terminal status when the row had settled, or "" when
// the snapshot does not exist. The flip leaves HeartbeatAt where the worker
// left it: the beat is the worker's liveness signal, and a dead worker's row
// must read as expired at once rather than a heartbeat timeout later.
// SaveSnapshot's atomic read-mutate-write makes the flip safe against a racing
// finalize, and the status change drives the runtime's
// [SnapshotSubscriber.OnSnapshotStatusChange] subscription, so the store needs
// no dedicated abort method. It backs [Agent.Abort] and the abort companion
// action.
func abortPendingSnapshot[State any](ctx context.Context, store SnapshotWriter[State], snapshotID string) (SnapshotStatus, error) {
	now := time.Now()
	saved, err := store.SaveSnapshot(ctx, snapshotID,
		func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			if existing == nil {
				return nil, nil // not found
			}
			if existing.Status != SnapshotStatusPending {
				return existing, nil // settled or already aborting: re-persist so the return carries its status
			}
			updated := *existing
			updated.Status = SnapshotStatusAborting
			updated.UpdatedAt = now
			return &updated, nil
		})
	if err != nil {
		return "", err
	}
	if saved == nil {
		return "", nil
	}
	return saved.Status, nil
}

// finalizePendingSnapshot rewrites the pending snapshot row with the
// terminal state and status. abortedByUser distinguishes a context
// cancellation from abort (status=aborted) from an internal
// failure (status=failed). The write is funneled through SaveSnapshot
// so the read-and-rewrite is one atomic step: the row's status inside fn
// decides the write, so an abort flip that landed first (the row reads
// aborting) settles as aborted with the state, and a row that has already
// settled is left untouched.
func (rt *agentRuntime[State]) finalizePendingSnapshot(
	ctx context.Context,
	pending *SessionSnapshot[State],
	result *AgentResult,
	fnErr error,
	abortedByUser bool,
) {
	// The state the row lands with: everything through the last turn that
	// committed, so an unfinished turn's mutations do not ride onto a row
	// that is now resumable.
	finalState := *rt.sess.committedState(ctx)
	// Captured outside the SaveSnapshot callback (which must stay pure): the
	// finalizer runs after fn returned, so these are stable. The abort/error
	// branches below own their reasons and ignore this clean-success default.
	completedReason := rt.sess.invocationReason(result)
	now := time.Now()

	_, err := rt.cfg.store.SaveSnapshot(ctx, pending.SnapshotID,
		func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			if existing != nil {
				switch {
				case existing.Status == SnapshotStatusAborting:
					// The abort flip landed first: keep the stop, and settle
					// the row as aborted by stamping the finish reason and
					// the state, so the snapshot is self-describing and
					// resumable. (The flip only changes the status of a
					// pending row that carries no state; the runtime owns the
					// semantic reason and the state.)
					annotated := *existing
					annotated.Status = SnapshotStatusAborted
					annotated.FinishReason = AgentFinishReasonAborted
					annotated.State = &finalState
					annotated.UpdatedAt = now
					// The row is terminal now; drop the liveness heartbeat so
					// it does not linger on a settled snapshot. CreatedAt is
					// preserved from the copy, so recency ordering is
					// unaffected.
					annotated.HeartbeatAt = nil
					return &annotated, nil
				case existing.Status.Terminal():
					// Already settled (a retried finalize, or a writer that
					// landed a terminal status directly): the row describes
					// itself, so leave it.
					return nil, nil
				}
			}

			snapStatus := SnapshotStatusCompleted
			// The persisted finish reason records how the background work
			// actually ended, distinct from the detached reason the client
			// already saw on AgentOutput.
			finishReason := completedReason
			var snapErr *status.Error
			switch {
			case abortedByUser:
				snapStatus = SnapshotStatusAborted
				finishReason = AgentFinishReasonAborted
				if fnErr != nil {
					snapErr = convertKeepText(fnErr) // aborted wins, preserve text
				}
			case fnErr != nil:
				// ctx is decoupled from the client's and still live here, so
				// this reads the error: a background run that reached a
				// caller-set limit was stopped, not broken.
				finishReason = terminalReason(ctx, fnErr)
				snapStatus = terminalStatus(finishReason)
				snapErr = convertKeepText(fnErr)
			}

			// Preserve the pending row's CreatedAt (so the finalize does not
			// move it ahead of newer rows in createdAt-ordered resolution) and
			// advance UpdatedAt: this rewrite is a real state change.
			return &SessionSnapshot[State]{
				SessionID:    pending.SessionID,
				ParentID:     pending.ParentID,
				Status:       snapStatus,
				FinishReason: finishReason,
				Error:        snapErr,
				State:        &finalState,
				CreatedAt:    pending.CreatedAt,
				UpdatedAt:    now,
			}, nil
		})
	if err != nil {
		logger.Error(ctx, "agent: failed to finalize pending snapshot",
			"snapshotId", pending.SnapshotID, "error", err)
	}
}

// loadSession constructs a Session from the invocation's init payload,
// loading from the store when a snapshot or session ID is provided.
// Returns the loaded snapshot too so the runtime can chain ParentID (and
// carry the session ID) off it.
//
// State is mutually exclusive with both SessionID and SnapshotID: it is
// the client-managed conversation source and carries its own identity
// ([SessionState.SessionID]), while the two IDs resolve against a store.
// SessionID and SnapshotID compose: the snapshot picks the exact resume
// point and the session ID is asserted against it.
func loadSession[State any](
	ctx context.Context,
	init *AgentInit[State],
	store SessionStore[State],
) (*Session[State], *SessionSnapshot[State], error) {
	s := &Session[State]{store: store}
	if init == nil {
		return s, nil, nil
	}

	if init.State != nil && (init.SessionID != "" || init.SnapshotID != "") {
		return nil, nil, status.Errorf(status.ErrInvalidArgument,
			"state is mutually exclusive with session ID and snapshot ID; a client-managed conversation's identity rides inside the state (SessionState.SessionID)")
	}

	// The three store-mode mismatches below stay internal: they describe how the
	// agent was wired, not what the caller sent, so an anonymous client should
	// not learn from them whether state is server- or client-managed. A
	// developer integrating against the agent sees the full text under
	// GENKIT_ENV=dev and in the server log.
	switch {
	case init.State != nil:
		if store != nil {
			return nil, nil, status.Errorf(status.ErrFailedPrecondition,
				"state provided but agent has a session store configured (server-managed state); use snapshot ID instead")
		}
		// Deep-copy at the entry boundary: an in-process caller retains
		// its state object ([WithState] documents resending it), so the
		// session must own private memory. Without this, AddArtifacts'
		// in-place replace writes into the caller's array and the
		// caller's later mutations race snapshot marshaling.
		s.state = *jsonClone(init.State)
		return s, nil, nil

	case init.SnapshotID != "":
		if store == nil {
			return nil, nil, status.Errorf(status.ErrFailedPrecondition,
				"snapshot ID %q provided but agent has no session store configured (client-managed state); use state instead", init.SnapshotID)
		}
		snap, err := store.GetSnapshot(ctx, init.SnapshotID)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to load snapshot %q: %w", init.SnapshotID, err)
		}
		if snap == nil {
			return nil, nil, status.PublicErrorf(ErrSnapshotNotFound, "snapshot %q not found", init.SnapshotID)
		}
		// A session ID sent alongside the snapshot ID asserts which
		// conversation the snapshot belongs to; a mismatch means the
		// caller would silently continue the wrong conversation.
		if init.SessionID != "" && snap.SessionID != init.SessionID {
			return nil, nil, status.Errorf(status.ErrInvalidArgument,
				"snapshot %q does not belong to session %q (snapshot's session: %q)", init.SnapshotID, init.SessionID, snap.SessionID)
		}
		return resumeSessionFrom(s, snap)

	case init.SessionID != "":
		if store == nil {
			return nil, nil, status.Errorf(status.ErrFailedPrecondition,
				"session ID %q provided but agent has no session store configured (client-managed state); the conversation's identity rides inside the state object (SessionState.SessionID)", init.SessionID)
		}
		snap, err := store.GetLatestSnapshot(ctx, init.SessionID)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to resolve latest snapshot for session %q: %w", init.SessionID, err)
		}
		if snap == nil {
			// No snapshot exists for this session ID yet: the caller is
			// starting a brand-new conversation under an ID of its own
			// choosing, not resuming one. Return a fresh session with no
			// parent; newAgentRuntime stamps the chosen ID so every snapshot
			// the new session persists carries it.
			return s, nil, nil
		}
		if snap.SessionID != init.SessionID {
			return nil, nil, status.Errorf(status.ErrInternal,
				"store resolved session %q to snapshot %q, which belongs to session %q; the store violates the GetLatestSnapshot contract", init.SessionID, snap.SnapshotID, snap.SessionID)
		}
		return resumeSessionFrom(s, snap)
	}
	return s, nil, nil
}

// resumeSessionFrom validates that snap is in a resumable status and loads
// its state into s. Shared by the snapshot-ID and session-ID init paths:
// both reject a pending snapshot, whose invocation is still writing to it.
// A failed or aborted one resumes: the turn that wrote it committed a
// conversation ending at a turn seam, and whether to continue from a run
// that broke or one that was stopped is the caller's judgement, not the
// framework's. The session-ID path reaches the rejection too, because
// GetLatestSnapshot returns the literal latest row whatever its status; a
// caller wanting to continue past a dead-end tip must name an earlier
// snapshot explicitly via SnapshotID.
func resumeSessionFrom[State any](s *Session[State], snap *SessionSnapshot[State]) (*Session[State], *SessionSnapshot[State], error) {
	switch snap.Status {
	case SnapshotStatusPending, SnapshotStatusAborting:
		// Neither row is resumable: a pending row carries no state, and an
		// aborting one sits between the flip that stopped its work and the
		// finalize that stamps the state on. What the caller should do next
		// depends on whether the worker is alive, and the heartbeat says, by
		// the same staleness rule the read shaping applies
		// (isHeartbeatExpired). A live beat means the row is still being
		// written and this same ID is the thing to wait on; sending that
		// caller to an earlier snapshot would fork the run away from the work
		// about to be committed. A stale beat means the write is never coming
		// (the process died, or the write failed), and the resume point is
		// the parent: the last snapshot committed before the detach, which
		// the error names so the caller is not sent hunting for it. A row
		// with no parent belongs to a run that detached before committing
		// anything, and there is genuinely nothing to resume.
		if isHeartbeatExpired(snap, defaultHeartbeatTimeout) {
			fence := "abort it, then "
			if snap.Status == SnapshotStatusAborting {
				fence = ""
			}
			if snap.ParentID != "" {
				return nil, nil, status.Errorf(status.ErrFailedPrecondition,
					"snapshot %q is still %s but its worker stopped heartbeating and is presumed dead; %sresume from its parent snapshot %q", snap.SnapshotID, snap.Status, fence, snap.ParentID)
			}
			return nil, nil, status.Errorf(status.ErrFailedPrecondition,
				"snapshot %q is still %s but its worker stopped heartbeating and is presumed dead. It recorded no progress, so there is nothing to resume", snap.SnapshotID, snap.Status)
		}
		if snap.Status == SnapshotStatusAborting {
			return nil, nil, status.Errorf(status.ErrFailedPrecondition,
				"snapshot %q is still being finalized: its invocation was aborted and has not recorded the state yet; retry this same snapshot ID", snap.SnapshotID)
		}
		return nil, nil, status.Errorf(status.ErrFailedPrecondition,
			"snapshot %q is still pending: its detached invocation is still running; wait for it to finalize or abort it before resuming", snap.SnapshotID)
	case SnapshotStatusAborted:
		// The runtime lands an aborted row together with its state, so one
		// without state was written by something else. Resuming it would
		// silently hand back an empty session in place of the conversation
		// the caller asked to continue; the parent is the resume point.
		if snap.State == nil {
			if snap.ParentID != "" {
				return nil, nil, status.Errorf(status.ErrFailedPrecondition,
					"snapshot %q was aborted before its invocation recorded any state; resume from its parent snapshot %q", snap.SnapshotID, snap.ParentID)
			}
			return nil, nil, status.Errorf(status.ErrFailedPrecondition,
				"snapshot %q was aborted before its invocation recorded any state and has no parent; there is nothing to resume", snap.SnapshotID)
		}
	}
	if snap.State != nil {
		// Stores may return rows sharing memory with their internal
		// copies (the [SnapshotReader] contract does not require fresh
		// memory), so the session takes a private copy; otherwise two
		// invocations resumed from the same snapshot would cross-corrupt
		// through the shared backing arrays.
		s.state = *jsonClone(snap.State)
	}
	return s, snap, nil
}

// --- chunkRouter ---
//
// chunkRouter owns the intermediate stream channel that all chunks flow
// through on their way to outCh. A chunk's in-process side effect (adding
// an artifact chunk's artifact to the session) is applied synchronously by
// Responder.send before the chunk enters the router, so every chunk gets it
// in its sender's goroutine regardless of whether detach has landed; the
// wire forward is the one thing detach suppresses, by latching the
// detached ingress filter so a detached run's chunks never enter the pipe.
// The router commits to not writing before we return so that close is safe
// (the bidi framework closes outCh shortly after bidiFn returns), and
// keeps draining its input so the user fn never blocks on a responder
// send.

type chunkRouter[State any] struct {
	ctx     context.Context // action context; ends on client disconnect (or completion)
	in      chan *AgentStreamChunk
	out     chan<- *AgentStreamChunk
	session *Session[State]
	// transform shapes each chunk on the wire; see [WithStreamTransform]. Nil
	// forwards chunks verbatim.
	transform StreamTransform
	// fail reports a fail-closed transform error (or panic) to the runtime so
	// the invocation resolves as a failed output. Nil only when no transform is
	// configured, since that is the only thing that can fail here.
	fail func(error)

	// detached latches when the intake reader observes a detach directive:
	// from then on chunks are filtered at ingress (Responder.send and
	// sendChunk skip the r.in send), so nothing a detached run produces
	// enters the pipe. The latch store happens-before the detach's inputs
	// are released to the runner, so every chunk of the detached turn
	// observes it; in-process side effects still apply at Send time.
	detached atomic.Bool

	done          chan struct{}
	stopWriting   chan struct{}
	writerStopped chan struct{}
}

func startChunkRouter[State any](
	ctx context.Context,
	session *Session[State],
	out chan<- *AgentStreamChunk,
	transform StreamTransform,
	fail func(error),
) *chunkRouter[State] {
	r := &chunkRouter[State]{
		ctx:           ctx,
		in:            make(chan *AgentStreamChunk),
		out:           out,
		session:       session,
		transform:     transform,
		fail:          fail,
		done:          make(chan struct{}),
		stopWriting:   make(chan struct{}),
		writerStopped: make(chan struct{}),
	}
	go r.run()
	return r
}

func (r *chunkRouter[State]) run() {
	defer close(r.done)
	r.forward()
	// forward has committed to not writing. Signal before draining so a
	// stopAndWait waiter is released whichever way forward exited.
	close(r.writerStopped)
	// Keep draining so a producer mid-send never blocks. The chunks' side
	// effects already happened at Send time; only the wire forward to
	// outCh is suppressed, so the chunks are simply discarded. (When
	// forward exited because r.in closed, this exits immediately.)
	for range r.in {
	}
}

// applySideEffects records the chunk's effect on session state: an artifact
// chunk adds its artifact to the session. Invoked synchronously from
// Responder.send, in the sender's goroutine, so the effect is ordered before
// everything the sender does after Send: a state read, a turn-end snapshot, or
// [SessionRunner.Result] immediately after SendArtifact observes the artifact.
// The artifact is deep-copied on its way into the session so the sender's
// retained pointer (which also rides the wire chunk) cannot alias live session
// state.
func (r *chunkRouter[State]) applySideEffects(chunk *AgentStreamChunk) {
	if chunk.Artifact != nil {
		r.session.AddArtifacts(jsonClone(chunk.Artifact))
	}
}

// forward delivers chunks to outCh until told to stop writing, the
// action context ends, or r.in closes. On every return it has committed
// to never writing to out again.
func (r *chunkRouter[State]) forward() {
	for {
		select {
		case chunk, ok := <-r.in:
			if !ok {
				return
			}
			shaped, err := r.shape(chunk)
			if err != nil {
				// The stream transform failed closed (returned an error or
				// panicked). Report it so the invocation resolves as a failed
				// output, and switch to discard mode so no further chunk
				// reaches the wire: fail-closed means stop forwarding entirely.
				r.fail(err)
				return
			}
			if shaped == nil {
				// The stream transform dropped the chunk from the wire. Its
				// side effects already applied at Send time, so there is
				// nothing else to do; carry on draining the next chunk.
				continue
			}
			chunk = shaped
			select {
			case r.out <- chunk:
			case <-r.stopWriting:
				return
			case <-r.ctx.Done():
				// The client is gone (disconnect cancels the action
				// context), so nothing will drain out again and a blocked
				// forward would wedge close. Drop the chunk and switch to
				// side-effects-only mode.
				return
			}
		case <-r.stopWriting:
			return
		}
	}
}

// shape applies the configured stream transform to chunk, returning the chunk
// to forward on the wire, nil to drop it, or a non-nil error to fail the
// invocation closed; with no transform it returns chunk unchanged. The
// transform receives a fresh deep copy it owns, so mutating it in place cannot
// disturb the chunk's already-applied side effects (an artifact recorded on the
// session) or any pointer the sender retained. r.ctx is the action context,
// which carries the caller's identity for RBAC-aware redaction; the transform
// only runs on chunks bound for a live client, since forward stops calling it
// once writes cease.
//
// The transform is user code running in the router's own goroutine, which
// nothing else recovers (unlike the agent fn and the state transform, whose
// goroutines are covered), so a panic here would crash the process rather than
// fail just the invocation. Contain it the way the fn path does (log with a
// stack) and surface it as a fail-closed error, the same outcome as an explicit
// error return: the invocation fails rather than leaking the unshaped chunk.
func (r *chunkRouter[State]) shape(chunk *AgentStreamChunk) (out *AgentStreamChunk, err error) {
	if r.transform == nil {
		return chunk, nil
	}
	defer func() {
		if rec := recover(); rec != nil {
			out, err = nil, panicError(r.ctx, "agent stream transform", rec)
		}
	}()
	return r.transform(r.ctx, jsonClone(chunk))
}

// responder returns a [Responder] that applies chunk side effects
// synchronously and sends chunks into the router for the wire forward.
// The returned Responder's Send methods drop the forward (returning
// promptly) when ctx is cancelled or the run has detached.
func (r *chunkRouter[State]) responder(ctx context.Context) Responder {
	return Responder{in: r.in, ctx: ctx, effects: r.applySideEffects, detached: &r.detached}
}

// sendChunk delivers chunk to the router for producers other than the
// user agent function (e.g. the runtime's emitTurnEnd). It skips the
// in-process side effects (the only runtime-produced chunk is TurnEnd,
// which has none: no artifact) and returns promptly if ctx is cancelled
// or the run has detached, dropping the chunk.
func (r *chunkRouter[State]) sendChunk(ctx context.Context, chunk *AgentStreamChunk) {
	if r.detached.Load() {
		return
	}
	select {
	case r.in <- chunk:
	case <-ctx.Done():
	}
}

// markDetached latches the router into detached mode, filtering chunks at
// ingress from here on (see the detached field). Called by the intake
// reader the instant it observes a detach directive.
func (r *chunkRouter[State]) markDetached() {
	r.detached.Store(true)
}

// stopAndWait tells the router to stop writing to out and blocks until it
// has committed. After it returns, it is safe for the framework to close
// out without risking a write-to-closed-channel panic.
func (r *chunkRouter[State]) stopAndWait() {
	close(r.stopWriting)
	<-r.writerStopped
}

// close signals end-of-input and waits for the router to drain.
func (r *chunkRouter[State]) close() {
	close(r.in)
	<-r.done
}

// --- customPatcher ---

// customPatcher streams the agent's custom state to the client as RFC 6902
// JSON Patches. The runtime wires it to the session's onCustomChange hook so
// every [Session.UpdateCustom] mutation emits a [AgentStreamChunk.CustomPatch]
// describing the delta, exactly as adding an artifact emits an artifact chunk.
//
// The diff is computed on the client-facing custom value (after the configured
// [StateTransform]), so streamed deltas honor redaction and stay consistent
// with the full state in turn-end snapshots and final output. Because a client
// may begin a turn without having loaded the full state, the first patch of
// each turn is a whole-document replace at the root pointer that re-bases it;
// subsequent patches are incremental diffs against the last sent value.
type customPatcher[State any] struct {
	transform StateTransform[State]
	session   *Session[State]
	// fail reports a fail-closed transform error to the runtime so the
	// invocation resolves as a failed output rather than streaming a delta
	// derived from state the transform refused to shape.
	fail func(error)

	ctx  context.Context         // invocation work context, for the transform
	send func(*AgentStreamChunk) // forwards the chunk (side effects + wire)

	mu          sync.Mutex
	firstInTurn bool
	baseline    any // last sent custom, normalized; the diff baseline
}

// bind attaches the invocation's work context and chunk sink. Called once in
// run, before the agent fn (the only producer of custom mutations) starts.
func (p *customPatcher[State]) bind(ctx context.Context, send func(*AgentStreamChunk)) {
	p.ctx = ctx
	p.send = send
}

// beginTurn arms the next emitted patch to be a whole-document replace,
// re-basing a client that may not share the server's baseline. Called by the
// runner at the start of every turn.
func (p *customPatcher[State]) beginTurn() {
	p.mu.Lock()
	p.firstInTurn = true
	p.mu.Unlock()
}

// onChange computes and emits the patch for the current custom state. It is
// invoked (outside the session lock) after every UpdateCustom mutation. The
// state read, diff, baseline update, and send all happen under p.mu so
// concurrent mutations produce a single, consistently ordered patch stream.
func (p *customPatcher[State]) onChange() {
	if p.send == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	// Diff the client-facing custom value (after the transform), matching what
	// turn-end snapshots and final output expose. With no transform we only need
	// the custom value, so take a custom-only normalized copy instead of
	// deep-copying the whole session state (messages and artifacts included) on
	// every mutation. With a transform we honor it on the full state, exactly as
	// the snapshot and output paths do, so the streamed delta stays consistent
	// with them.
	var next any
	if p.transform == nil {
		next = p.session.customJSON()
	} else {
		t, err := applyTransform(p.ctx, p.transform, p.session.State())
		if err != nil {
			// The state transform failed closed while shaping the streamed
			// custom delta. Withhold the patch and fail the invocation; the run
			// loop tears it down as a failed output, the same fail-closed
			// outcome as a stream-transform error in the router.
			p.fail(err)
			return
		}
		var custom any
		if t != nil {
			custom = t.Custom
		}
		next = normalizeJSON(custom)
	}

	var patch JSONPatch
	if p.firstInTurn {
		patch = JSONPatch{{Op: JSONPatchOpReplace, Path: "", Value: cloneJSON(next)}}
		p.firstInTurn = false
	} else {
		patch = diffValues(p.baseline, next)
	}
	p.baseline = next
	if len(patch) > 0 {
		p.send(&AgentStreamChunk{CustomPatch: patch})
	}
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
// Snapshot suspension after detach is not the intake's concern: the
// runner gates writes itself (see SessionRunner.suspendSnapshots), so a
// detach can atomically wait out an in-flight turn-end write.
//
// Wire suppression is the intake's concern, though, because it is an
// ordering constraint on input release that only the reader can enforce:
// a detached run streams nothing to the client, so the router's ingress
// filter must be latched before the inputs riding the detach directive
// reach the runner. The runtime's detach handler cannot guarantee that
// itself - it runs in another goroutine, behind a store write, by which
// time the background turn may already have emitted.

type detachIntake struct {
	src    <-chan *AgentInput
	dst    chan *AgentInput
	notify chan struct{} // buffered size 1; wakes forwarder when queue grows

	// onDetach suppresses the client stream, called by the reader the
	// instant a detach directive is observed. Never nil: the runtime wires
	// it to the router's markDetached; standalone intake tests pass a no-op.
	onDetach func()

	// turnDone is signaled at each turn end to release the forwarder so
	// it may pop the next input. Initialized with one token so the very
	// first turn can start without a preceding turn end.
	turnDone chan struct{}

	mu    sync.Mutex
	queue []*AgentInput

	readDone atomic.Bool
	detachCh chan struct{} // closed by reader when detach observed

	stop     chan struct{}
	stopOnce sync.Once
	done     chan struct{}
}

func startDetachIntake(src <-chan *AgentInput, onDetach func()) *detachIntake {
	i := &detachIntake{
		src:      src,
		dst:      make(chan *AgentInput),
		notify:   make(chan struct{}, 1),
		onDetach: onDetach,
		turnDone: make(chan struct{}, 1),
		detachCh: make(chan struct{}),
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
			if input == nil {
				// A nil input (e.g. a JSON null decoded by a transport)
				// carries nothing to process; dropping it here also keeps
				// nil out of the queue, where the forwarder would read it
				// as end-of-input.
				continue
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

// handleDetach suppresses the client stream, drains any buffered src
// inputs into the queue, and signals the detach handler. The detach
// handler then suspends turn-end snapshots (via the runner) while the
// queued inputs finish processing.
//
// onDetach comes first: the inputs released below are the ones whose turn
// produces the chunks a detached run must not stream (see the type
// comment above for why only the reader can enforce that ordering).
//
// A pure detach signal (no Messages, no Resume payload) is dropped
// rather than enqueued: it carries no payload to process, so it would
// just trigger a no-op turn. Callers that want to ride a final input
// on the detach signal can do so by calling
// Send(&AgentInput{Detach: true, Message: ...}) explicitly.
func (i *detachIntake) handleDetach(first *AgentInput) {
	i.onDetach()

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
			if more != nil {
				drained = append(drained, more)
			}
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

	// A latch rather than a token: closing lets any number of observers
	// (the run select's detach arm and its fnDone-arm re-check) see the
	// signal without consuming it. handleDetach runs at most once (read
	// returns right after it), so the close cannot double-fire.
	close(i.detachCh)
}

// hasInputPayload reports whether the input carries data of its own. It
// filters pure detach signals out of the queue so they don't trigger no-op
// turns, and it marks the inputs a turn can start from: one without a
// payload runs on the conversation already in the session, so there has to
// be a conversation there.
func hasInputPayload(in *AgentInput) bool {
	if in == nil {
		return false
	}
	if in.Message != nil {
		return true
	}
	if in.Resume != nil && (len(in.Resume.Respond) > 0 || len(in.Resume.Restart) > 0) {
		return true
	}
	return false
}

// forward pops the queue and writes to dst at the runner's pace. The
// runtime signals turnDone via releaseForward when it's ready for the
// next input; until then the forwarder waits, so it never gets ahead of
// the runner.
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
// Called by the runtime's emitTurnEnd at each turn end (and only there)
// so the forwarder stays in step with the runner's turn pacing.
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

// sessionMessageKey marks the session's own messages on their way into the
// prompt render. The prompt decides where they land, so afterwards this is the
// only thing separating them from its scaffolding. [tagRenderedMessages] strips
// it before the model sees anything.
const sessionMessageKey = "_genkit_history"

// taggedMessage returns a copy of m carrying key in its metadata. Copying
// matters: Render may alias metadata from shared prompt config, so writing in
// place would leak the tag into the registered prompt and race with concurrent
// invocations.
func taggedMessage(m *ai.Message, key string) *ai.Message {
	tagged := *m
	tagged.Metadata = maps.Clone(tagged.Metadata)
	if tagged.Metadata == nil {
		tagged.Metadata = make(map[string]any, 1)
	}
	tagged.Metadata[key] = true
	return &tagged
}

// hasTag reports whether m carries key.
func hasTag(m *ai.Message, key string) bool {
	return m.Metadata != nil && m.Metadata[key] == true
}

// untaggedMessage returns m without key, copying only when it is present.
func untaggedMessage(m *ai.Message, key string) *ai.Message {
	if !hasTag(m, key) {
		return m
	}
	stripped := *m
	stripped.Metadata = maps.Clone(stripped.Metadata)
	delete(stripped.Metadata, key)
	if len(stripped.Metadata) == 0 {
		stripped.Metadata = nil
	}
	return &stripped
}

// markSessionMessages copies the session's messages with [sessionMessageKey]
// so the render can be handed them without the tag reaching session state.
func markSessionMessages(history []*ai.Message) []*ai.Message {
	marked := make([]*ai.Message, 0, len(history))
	for _, m := range history {
		if m == nil {
			continue
		}
		marked = append(marked, taggedMessage(m, sessionMessageKey))
	}
	return marked
}

// tagRenderedMessages splits the render output in one pass: what came from the
// session loses its marker, and everything else is tagged as the prompt's
// scaffolding so it can be dropped from session history after generation.
func tagRenderedMessages(messages []*ai.Message) []*ai.Message {
	tagged := make([]*ai.Message, 0, len(messages))
	for _, m := range messages {
		if m == nil {
			continue
		}
		if hasTag(m, sessionMessageKey) {
			tagged = append(tagged, untaggedMessage(m, sessionMessageKey))
			continue
		}
		tagged = append(tagged, taggedMessage(m, promptMessageKey))
	}
	return tagged
}

// turnSessionMessages reduces the generate response's full message list to what
// the session should hold: everything except the prompt's scaffolding, which
// leaves the placed conversation, the tool loop's messages, and the reply.
func turnSessionMessages(full []*ai.Message) []*ai.Message {
	msgs := make([]*ai.Message, 0, len(full))
	for _, m := range full {
		if m == nil || hasTag(m, promptMessageKey) {
			continue
		}
		msgs = append(msgs, m)
	}
	return msgs
}

// validateUserMessage rejects inputs the prompt-backed agent loop can't
// safely consume: a non-user role would be appended to history under the
// wrong speaker, and tool request / response parts belong on the
// [AgentInput.Resume] payload, not on a turn message.
func validateUserMessage(m *ai.Message) error {
	if m == nil {
		return nil
	}
	if m.Role != "" && m.Role != ai.RoleUser {
		return status.Errorf(status.ErrInvalidArgument,
			"agent input message must have role %q, got %q", ai.RoleUser, m.Role)
	}
	for _, p := range m.Content {
		if p == nil {
			continue
		}
		if p.IsToolRequest() || p.IsToolResponse() {
			return status.Errorf(status.ErrInvalidArgument,
				"agent input message must not contain tool request or response parts; use AgentInput.Resume instead")
		}
	}
	return nil
}

// ValidateResumeAgainstHistory ensures every restart and respond entry on a
// resume payload references a tool request that is actually pending, so a
// caller cannot drive a tool the model never asked for and interrupted on.
// For restart entries it additionally checks the input is unchanged from the
// original request, preventing a client from forging tool inputs on the
// interrupted call.
//
// Only the most recent model message in history is searched, and within it
// newest-first. Resume handling resolves exactly that response's tool
// requests, so an entry aimed at an earlier turn does nothing downstream;
// worse, because refs are only made unique within a single response, a tool
// name + ref recurs across turns, and searching further back would let a
// client validate a restart whose input was forged from an already-settled
// call. An entry naming an earlier turn is rejected as stale, separately
// from one naming a tool request that never existed. On a violation it
// returns an INVALID_ARGUMENT error.
//
// The prompt-backed agent loop ([DefineAgent]) calls this automatically. A
// custom agent ([DefineCustomAgent]) that accepts an [AgentInput.Resume] from
// untrusted callers should call it before forwarding the payload to the model:
//
//	if input.Resume != nil {
//		if err := ValidateResumeAgainstHistory(input.Resume, sess.Messages()); err != nil {
//			return nil, err
//		}
//	}
func ValidateResumeAgainstHistory(resume *ToolResume, history []*ai.Message) error {
	if resume == nil {
		return nil
	}

	// The pending response is the last model message. Trailing non-model
	// messages are skipped rather than treated as "nothing pending": a
	// caller that sends a turn message alongside a resume payload leaves a
	// user message last, and that combination must keep failing downstream
	// on generate's own precondition instead of being rejected here.
	pending := -1
	for i := len(history) - 1; i >= 0; i-- {
		if msg := history[i]; msg != nil && msg.Role == ai.RoleModel {
			pending = i
			break
		}
	}

	// findIn searches one model message's tool requests newest-first, so a
	// malformed response repeating a name + ref resolves to its most recent
	// occurrence.
	findIn := func(idx int, name, ref string) *ai.ToolRequest {
		if idx < 0 {
			return nil
		}
		msg := history[idx]
		if msg == nil || msg.Role != ai.RoleModel {
			return nil
		}
		for j := len(msg.Content) - 1; j >= 0; j-- {
			p := msg.Content[j]
			if p.IsToolRequest() && p.ToolRequest != nil &&
				p.ToolRequest.Name == name && p.ToolRequest.Ref == ref {
				return p.ToolRequest
			}
		}
		return nil
	}

	// unresolved renders the rejection for an entry that matches nothing in
	// the pending response. Earlier turns are consulted only to pick the
	// message: a caller replaying a settled payload should not be told the
	// tool request never existed.
	unresolved := func(field, name, ref string) error {
		for i := pending - 1; i >= 0; i-- {
			if findIn(i, name, ref) != nil {
				return status.Errorf(status.ErrInvalidArgument,
					"resume.%s references tool %q%s from an earlier turn which is no longer pending; only tool requests from the most recent model response can be resumed",
					field, name, toolRefSuffix(ref))
			}
		}
		return status.Errorf(status.ErrInvalidArgument,
			"resume.%s references tool %q%s which was not found in session history",
			field, name, toolRefSuffix(ref))
	}

	// Restart entries: name + ref must be pending and the input must match
	// the original request exactly. IsToolRequest only checks the part kind,
	// so guard the pointer too: a hand-built NewToolRequestPart(nil) is kind
	// PartToolRequest with a nil ToolRequest.
	for _, p := range resume.Restart {
		if !p.IsToolRequest() || p.ToolRequest == nil {
			continue
		}
		req := p.ToolRequest
		match := findIn(pending, req.Name, req.Ref)
		if match == nil {
			return unresolved("restart", req.Name, req.Ref)
		}
		if !jsonEqual(normalizeJSON(req.Input), normalizeJSON(match.Input)) {
			return status.Errorf(status.ErrInvalidArgument,
				"resume.restart for tool %q%s has modified inputs that do not match the original tool request in session history; restart inputs must exactly match the interrupted tool request",
				req.Name, toolRefSuffix(req.Ref))
		}
	}

	// Respond entries: name + ref must match a pending tool request.
	for _, p := range resume.Respond {
		if !p.IsToolResponse() || p.ToolResponse == nil {
			continue
		}
		resp := p.ToolResponse
		if findIn(pending, resp.Name, resp.Ref) == nil {
			return unresolved("respond", resp.Name, resp.Ref)
		}
	}

	return nil
}

// toolRefSuffix renders a " (ref: X)" clause for resume validation errors, or
// "" when the tool request carried no ref.
func toolRefSuffix(ref string) string {
	if ref == "" {
		return ""
	}
	return fmt.Sprintf(" (ref: %s)", ref)
}

// agentLoop returns the per-turn function for a prompt-backed agent. Each
// turn renders the prompt, appends conversation history, calls the model
// with streaming, and updates the session.
//
// defaultInput is the prompt input passed to Render on every turn. It is
// nil for inline-defined prompts ([InlinePrompt]), which take no per-turn
// input.
func agentLoop[State any](r api.Registry, prompt ai.Prompt, defaultInput any) AgentFunc[State] {
	return func(ctx context.Context, resp Responder, sess *SessionRunner[State]) (*AgentResult, error) {
		if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
			if err := validateUserMessage(input.Message); err != nil {
				return nil, err
			}

			// Hand the conversation to the render instead of splicing it
			// in afterwards: a prompt-backed agent is a prompt, so where
			// the conversation goes is the prompt's decision, which is
			// what makes trimming or summarizing possible.
			//
			// This is the whole session, including this turn's message,
			// since sess.Run has already stored it. Only the render sees
			// the context: generation below runs on the original, so the
			// conversation does not ride along into tools and prompts
			// invoked inside the generate loop.
			history := sess.Messages()

			// An input with no payload of its own runs the turn on the
			// conversation as it stands, which is how a failed turn is
			// re-attempted: the failed snapshot holds the messages the turn
			// committed, and the model is called on them again. There has to
			// be something to continue.
			if !hasInputPayload(input) && len(history) == 0 {
				return nil, status.Errorf(status.ErrInvalidArgument,
					"agent input message or resume is required to start a conversation")
			}

			actionOpts, err := prompt.Render(ai.NewHistoryContext(ctx, markSessionMessages(history)), defaultInput)
			if err != nil {
				return nil, fmt.Errorf("prompt render: %w", err)
			}

			// Whatever the render produced that did not come from the
			// session is the prompt's own scaffolding (system message,
			// template output, few-shot examples). Tag it so it can be
			// filtered out of session history after generation.
			actionOpts.Messages = tagRenderedMessages(actionOpts.Messages)

			// If a resume payload was provided, validate that every
			// restart / respond entry references a tool request the model
			// actually issued, then forward it to the generate call so
			// handleResumeOption re-executes the interrupted tools and / or
			// applies the responses.
			if input.Resume != nil {
				if err := ValidateResumeAgainstHistory(input.Resume, history); err != nil {
					return nil, err
				}
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
			// An interrupt is a turn outcome, not a failure, even when it
			// arrives as one. [ai.Generate] reports a restarted tool that
			// interrupted again with a FAILED_PRECONDITION, because its
			// caller asked for a completed generation; the agent's caller did
			// not. The tip that comes back is the same answerable interrupt a
			// first-run interrupt leaves, and only [Resume] can answer either,
			// so the turn takes the success path and commits the same way
			// both times. No other error carries this finish reason.
			if err != nil && modelResp != nil && modelResp.FinishReason == ai.FinishReasonInterrupted {
				err = nil
			}
			if err != nil {
				// The partial's history ends at a turn seam (see
				// [ai.Generate]), so it is a conversation the caller can
				// continue. Fold it into the session and commit the turn as a
				// resume point: the [TurnResult] beside the error is what says
				// so; see [SessionRunner.Run]. Without a partial the call
				// never reached the model, and the turn rolls back instead.
				if modelResp == nil || modelResp.Request == nil {
					return nil, fmt.Errorf("generate: %w", err)
				}
				sess.SetMessages(turnSessionMessages(modelResp.History()))
				// No reason: the TurnResult here only says the turn committed,
				// and [SessionRunner.Run] derives the rest from the error it
				// is handed. The success arm forwards generate's reason
				// verbatim, but this arm must not, because a response the loop
				// completed and post-processing then rejected still carries
				// the model's own "stop".
				return &TurnResult{}, fmt.Errorf("generate: %w", err)
			}

			// Replace session messages with the full history minus the
			// prompt's scaffolding. This captures intermediate tool
			// call/response messages from the tool loop, not just the
			// final response, and it is what carries a resumed tool
			// request's resolved content back into the session.
			//
			// It is the conversation the prompt actually placed, so a
			// prompt that trims or summarizes its history trims the
			// session too, rather than recompacting an ever-growing
			// transcript every turn.
			if modelResp.Request != nil {
				sess.SetMessages(turnSessionMessages(modelResp.History()))
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

			// Forward the generate response's finish reason verbatim: the
			// agent enum is a superset of the model enum for these values,
			// so the turn (and a single-turn invocation) reports e.g.
			// "interrupted" without the client scanning message content.
			return &TurnResult{FinishReason: AgentFinishReason(modelResp.FinishReason)}, nil
		}); err != nil {
			return nil, err
		}
		return sess.Result(), nil
	}
}

// --- Agent client API ---

// Connect starts a new agent invocation with bidirectional streaming.
// Use this for multi-turn interactions where you need to send multiple inputs
// and receive streaming chunks. For single-turn usage, see Run and RunText;
// for a single turn that keeps working after the call returns, RunDetached.
func (a *Agent[State]) Connect(
	ctx context.Context,
	opts ...InvocationOption[State],
) (*AgentConnection[State], error) {
	// The merge rules live in resolveInvocationInit (option.go), shared with
	// the untyped [AgentHandle] surface.
	init, err := resolveInvocationInit(a.action.Name(), opts)
	if err != nil {
		return nil, err
	}
	conn, err := a.action.Connect(ctx, init)
	if err != nil {
		return nil, err
	}
	return &AgentConnection[State]{conn: conn}, nil
}

// Run starts a single-turn agent invocation with the given input.
// It sends the input, waits for the agent to complete, and returns the output.
// For multi-turn interactions or streaming, use Connect instead.
//
// In-band failures (e.g. a failed turn) resolve as a failed [AgentOutput]
// rather than an error; a rejected init payload fails with an error, since
// the invocation never starts. See [AgentConnection.Output].
func (a *Agent[State]) Run(
	ctx context.Context,
	input *AgentInput,
	opts ...InvocationOption[State],
) (*AgentOutput[State], error) {
	conn, err := a.Connect(ctx, opts...)
	if err != nil {
		return nil, err
	}
	// The invocation may resolve before consuming the input (e.g. an init
	// validation failure errors out before the first turn); the outcome,
	// whether output or error, is on Output regardless.
	if err := conn.Send(input); err != nil && !errors.Is(err, core.ErrActionCompleted) {
		return nil, err
	}
	return conn.Output()
}

// RunText is a convenience method that starts a single-turn agent invocation
// with a user text message. It is equivalent to calling Run with an
// AgentInput whose Message is a user text message.
func (a *Agent[State]) RunText(
	ctx context.Context,
	text string,
	opts ...InvocationOption[State],
) (*AgentOutput[State], error) {
	return a.Run(ctx, &AgentInput{
		Message: ai.NewUserTextMessage(text),
	}, opts...)
}

// RunDetached launches input as a detached (background) invocation and returns
// the task tracking it. It is the one-shot counterpart of
// [AgentConnection.Detach]: the input is delivered with [AgentInput.Detach]
// set, whatever the caller set it to; the runtime persists a pending snapshot
// and keeps working on a context decoupled from ctx; and the returned
// [DetachedTask] polls, waits on, or aborts that snapshot, with custom state
// typed as State. The pending snapshot is the durable record: record
// [DetachedTask.SnapshotID] and rehydrate with [Agent.Task] to pick the
// work up later, including from another process.
//
// The launch is rejected with FAILED_PRECONDITION when the agent cannot
// support detach: it has no session store, or the store does not implement
// [SnapshotSubscriber]. The rejection is the invocation's failed output
// ([AgentOutput.Error]) surfaced as the returned error; match it by status.
//
// An agent may also settle the invocation synchronously, before the runtime
// observes the detach directive (e.g. a custom agent whose fn returns without
// consuming the input). When a turn committed, RunDetached returns a task over
// its snapshot, which is already terminal, so Poll and Wait resolve
// immediately; when nothing was recorded, it fails with FAILED_PRECONDITION
// naming the finish reason, since there is no durable record to track.
func (a *Agent[State]) RunDetached(
	ctx context.Context,
	input *AgentInput,
	opts ...InvocationOption[State],
) (*DetachedTask[State], error) {
	if input == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: input must not be nil", a.Name())
	}
	out, err := a.Run(ctx, detachedInput(input), opts...)
	if err != nil {
		return nil, err
	}
	return detachedTaskFrom(a.Name(), a, out)
}

// Task returns the task tracking a snapshot ID recorded earlier (e.g.
// by a prior process that called [Agent.RunDetached]), with custom state typed
// as State. It performs no I/O and does not verify the snapshot exists; the
// first [DetachedTask.Poll] or [DetachedTask.Wait] surfaces NOT_FOUND for an
// unknown ID.
func (a *Agent[State]) Task(snapshotID string) *DetachedTask[State] {
	return &DetachedTask[State]{ops: a, snapshotID: snapshotID}
}

// --- AgentConnection ---

// AgentConnection is an active agent invocation with bidirectional streaming,
// adding agent-specific Send helpers (SendMessage, SendText, SendResume,
// Detach) over the core connection.
//
// It also tracks custom state live: as [AgentConnection.Receive] yields chunks,
// it applies each chunk's [AgentStreamChunk.CustomPatch] to an internal copy,
// exposed by [AgentConnection.Custom], so callers see custom state as it
// streams without applying patches themselves.
type AgentConnection[State any] struct {
	conn *core.BidiConnection[*AgentInput, *AgentOutput[State], *AgentStreamChunk]

	mu     sync.Mutex
	custom any // live custom state (normalized JSON), updated as patches stream
}

// Send sends an AgentInput to the agent. The input must not be nil.
//
// Once the invocation has resolved (e.g. a failed turn ended it), Send
// fails with an error matching [core.ErrActionCompleted]; the outcome is
// on [AgentConnection.Output]. The same applies to the SendMessage,
// SendText, SendResume, and Detach helpers.
func (c *AgentConnection[State]) Send(input *AgentInput) error {
	if input == nil {
		return status.Errorf(status.ErrInvalidArgument, "agent input must not be nil")
	}
	return c.conn.Send(input)
}

// SendMessage sends a message to the agent for one turn.
func (c *AgentConnection[State]) SendMessage(message *ai.Message) error {
	return c.conn.Send(&AgentInput{Message: message})
}

// SendText sends a user text message to the agent.
func (c *AgentConnection[State]) SendText(text string) error {
	return c.conn.Send(&AgentInput{
		Message: ai.NewUserTextMessage(text),
	})
}

// SendResume sends a resume payload to continue an interrupted generation.
// Construct the payload with [ai.ToolAction.RestartWith] or
// [ai.ToolAction.RespondWith] parts.
func (c *AgentConnection[State]) SendResume(resume *ToolResume) error {
	return c.conn.Send(&AgentInput{Resume: resume})
}

// Detach asks the server to write a pending snapshot, close the
// connection, and continue processing any already-buffered inputs in
// the background. Output() returns the pending snapshot ID; the client
// can later call Abort to stop the background work or
// GetSnapshot to observe its progression. The pending snapshot is
// finalized with the cumulative final state once the queued inputs
// are processed.
//
// Chunks emitted after detach are not forwarded over the wire, but their
// session-level side effects still apply: an artifact sent via
// [Responder.SendArtifact] still lands in the final snapshot's state.
//
// To send a final input in the same wire message, call
// Send(&AgentInput{Detach: true, Message: ...}) directly.
func (c *AgentConnection[State]) Detach() error {
	return c.conn.Send(&AgentInput{Detach: true})
}

// Close signals that no more inputs will be sent.
func (c *AgentConnection[State]) Close() error {
	return c.conn.Close()
}

// Receive returns an iterator for receiving stream chunks. Breaking out
// of the iterator does not cancel the connection; multi-turn callers
// routinely break on [TurnEnd], send the next input, then call Receive
// again to consume the next batch. Call [AgentConnection.Output] to
// finish the invocation, or cancel the ctx passed to Connect to
// abort it.
//
// Each yielded chunk's [AgentStreamChunk.CustomPatch] is applied to the
// connection's tracked custom state before the chunk is yielded, so
// [AgentConnection.Custom] reflects every delta observed so far.
func (c *AgentConnection[State]) Receive() iter.Seq2[*AgentStreamChunk, error] {
	return func(yield func(*AgentStreamChunk, error) bool) {
		for chunk, err := range c.conn.Receive() {
			if err == nil && chunk != nil && len(chunk.CustomPatch) > 0 {
				c.applyCustomPatch(chunk.CustomPatch)
			}
			if !yield(chunk, err) {
				return
			}
		}
	}
}

// applyCustomPatch applies a streamed patch to the tracked custom state. A
// malformed patch (only possible from a non-conforming server) leaves the last
// good value in place; the next turn's whole-document replace re-bases it.
func (c *AgentConnection[State]) applyCustomPatch(patch JSONPatch) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if next, err := applyOps(cloneJSON(c.custom), patch); err == nil {
		c.custom = next
	}
}

// Custom returns the conversation's custom state as tracked from the streamed
// patches observed via [AgentConnection.Receive]. It reflects the deltas
// consumed so far, so reading it as a turn streams shows the live state; before
// any patch arrives it returns the zero value. The authoritative final state is
// on [AgentOutput.State] (client-managed) or the turn-end snapshot
// (server-managed).
func (c *AgentConnection[State]) Custom() (State, error) {
	c.mu.Lock()
	tree := cloneJSON(c.custom)
	c.mu.Unlock()

	var out State
	b, err := json.Marshal(tree)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(b, &out); err != nil {
		return out, err
	}
	return out, nil
}

// Output finalizes the connection and returns the agent's result. It closes
// the input side, drains any chunks not consumed via Receive, and blocks until
// the agent finalizes. It is idempotent: later calls return the same value, and
// the returned pointer is shared, so treat it as read-only.
//
// In-band failures resolve rather than error. A failed turn returns an
// [AgentOutput] with [AgentFinishReasonFailed], the error on [AgentOutput.Error],
// and the last-good state on [AgentOutput.State] (client-managed) or behind
// [AgentOutput.SnapshotID] (server-managed), so a failure costs only the failed
// turn, not the session. A detached invocation resolves with the pending
// snapshot ID.
//
// A run the caller stopped returns both: the error that stopped it, and an
// [AgentOutput] with [AgentFinishReasonAborted] naming the same resume point,
// the way [ai.Generate] hands back a partial response beside its error. Check
// the output even when the error is non-nil, or the work up to the stop is
// stranded. It is nil when the invocation never started (a rejected init
// payload), and when an agent function that ignores its context has not
// settled within settleGrace, since there is nothing yet to report.
//
// Do not call Output concurrently with a goroutine iterating Receive; both
// consume the stream and would split chunks between them. Finish Receive first.
func (c *AgentConnection[State]) Output() (*AgentOutput[State], error) {
	_ = c.conn.Close()
	// The core connection applies backpressure and its Output does not
	// consume the stream, so drain the chunks the caller did not Receive;
	// the agent must never wedge publishing to a stream nobody reads.
	// Receive ends on completion or cancellation either way, and the core
	// Output prefers the finalized result when both are ready.
	for range c.conn.Receive() {
	}
	out, err := c.conn.Output()
	if out != nil || err == nil {
		return out, err
	}
	// The caller's context died before the invocation settled, so both the
	// drain above and the core Output took their cancellation arms and the
	// result is not stored yet. Wait for it: the runtime is unwinding towards
	// a failed or aborted output naming the snapshot to resume from, and
	// returning the bare cancellation would strand that snapshot with no
	// handle on it.
	//
	// Bounded by settleGrace, whose doc carries why.
	timer := time.NewTimer(settleGrace)
	defer timer.Stop()
	select {
	case <-c.conn.Done():
		return c.conn.Output()
	case <-timer.C:
		return out, err
	}
}

// Done returns a channel closed when the connection completes.
func (c *AgentConnection[State]) Done() <-chan struct{} {
	return c.conn.Done()
}
