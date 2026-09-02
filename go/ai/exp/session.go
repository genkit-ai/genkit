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
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// --- Snapshot ---

// applyTransform returns the result of applying t to state, or state
// unchanged if t is nil. A nil state is returned as-is. A non-nil error from
// the transform is propagated so callers can fail closed.
func applyTransform[State any](ctx context.Context, t StateTransform[State], state *SessionState[State]) (*SessionState[State], error) {
	if t == nil || state == nil {
		return state, nil
	}
	return t(ctx, state)
}

// Terminal reports whether the status is settled: no further transition will
// happen on its own. [SnapshotStatusPending] and [SnapshotStatusAborting] are
// the non-terminal statuses, the two rows a live worker is still owed a write
// for (an empty status counts as [SnapshotStatusCompleted], matching the
// documented default), so waiters stop on completed, failed, aborted, and
// expired alike. An expired snapshot's raw store row is still pending or
// aborting, but its worker is presumed dead, so nothing will finalize it.
func (s SnapshotStatus) Terminal() bool {
	return s != SnapshotStatusPending && s != SnapshotStatusAborting
}

// CarriesResult reports whether a turn that ended for this reason produced an
// answer its caller can use. True for the reasons that mean the agent spoke and
// stopped: [AgentFinishReasonStop], the two catch-alls, and the empty reason a
// turn may report when it names none. Every other reason ended the turn with
// nothing to hand back.
//
// Callers use it to decide whether a turn's final message is the answer or an
// account of why there is none. That question has one answer per reason, so it
// belongs on the reason: an agent that reports a turn's outcome and one that
// folds it into a tool result would otherwise each keep their own list, and a
// reason added later would be an answer to one of them and not the other.
//
// It names the reasons that do carry a result rather than the ones that do
// not, so a reason added later defaults to "no answer". That default is the
// safe one: mistaking an explanation for an answer hands a caller partial work
// as though it were final, while the reverse only asks it to look at the text.
func (r AgentFinishReason) CarriesResult() bool {
	switch r {
	case AgentFinishReasonStop, AgentFinishReasonOther, AgentFinishReasonUnknown, "":
		return true
	}
	return false
}

// --- Session store ---

// SnapshotReader retrieves snapshots. The minimum any session store must
// implement to be used with [WithSessionStore].
type SnapshotReader[State any] interface {
	// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
	GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error)

	// GetLatestSnapshot returns the session's most recently created
	// snapshot, whatever its status: a pending, failed, or aborted row is
	// returned like any other, and the caller applies its own policy.
	// Returns nil if the session has no rows, and an error if sessionID is
	// empty.
	//
	// "Most recently created" means the greatest [SessionSnapshot.CreatedAt];
	// break ties deterministically (e.g. by SnapshotID). This is a plain
	// max-timestamp lookup, implementable as a single indexed query (e.g.
	// WHERE sessionId = ? ORDER BY createdAt DESC LIMIT 1). A later rewrite of
	// an older row (e.g. a detach finalize) does not move it ahead of a
	// newer-created sibling, since CreatedAt is preserved across rewrites.
	// ParentID is informational lineage and plays no part in resolution: when
	// history forks, the most recently created branch wins.
	GetLatestSnapshot(ctx context.Context, sessionID string) (*SessionSnapshot[State], error)
}

// SnapshotMetadataReader is the optional capability layered on [SessionStore]
// that answers a metadata-only read without loading the row's state. The
// runtime takes it for every read that only needs to know where a snapshot
// stands ([GetSnapshotRequest.MetadataOnly], [WithMetadataOnly]): its status,
// finish reason, parent, session, timestamps, error, and heartbeat. A store
// that does not implement it is still correct, since the runtime reads the
// row in full and drops the state, but it pays to load a conversation history
// it will not use. Both bundled local stores and the Firestore store implement
// it. Implement both methods or neither: the capability is detected as a
// whole.
type SnapshotMetadataReader[State any] interface {
	// GetSnapshotMetadata is [SnapshotReader.GetSnapshot] without the state:
	// the returned row carries every other field as stored, with State nil,
	// and the same nil-if-not-found contract.
	GetSnapshotMetadata(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error)

	// GetLatestSnapshotMetadata is [SnapshotReader.GetLatestSnapshot] without
	// the state: the same resolution, the same nil-if-none and
	// empty-session-ID contract, and a row with State nil.
	GetLatestSnapshotMetadata(ctx context.Context, sessionID string) (*SessionSnapshot[State], error)
}

// getSnapshotMetadata reads a snapshot for a metadata-only caller: through
// [SnapshotMetadataReader] when the store offers it, and as a full read
// otherwise. The caller drops the state either way, so the two paths differ
// only in what the store had to load.
func getSnapshotMetadata[State any](ctx context.Context, store SnapshotReader[State], snapshotID string) (*SessionSnapshot[State], error) {
	if mr, ok := store.(SnapshotMetadataReader[State]); ok {
		return mr.GetSnapshotMetadata(ctx, snapshotID)
	}
	return store.GetSnapshot(ctx, snapshotID)
}

// getLatestSnapshotMetadata is getSnapshotMetadata for a session's latest row.
func getLatestSnapshotMetadata[State any](ctx context.Context, store SnapshotReader[State], sessionID string) (*SessionSnapshot[State], error) {
	if mr, ok := store.(SnapshotMetadataReader[State]); ok {
		return mr.GetLatestSnapshotMetadata(ctx, sessionID)
	}
	return store.GetLatestSnapshot(ctx, sessionID)
}

// SnapshotWriter persists snapshots. The minimum any session store must
// implement to be used with [WithSessionStore].
type SnapshotWriter[State any] interface {
	// SaveSnapshot atomically reads the snapshot at id (if any), applies
	// fn, and persists the result largely verbatim. The store owns only
	// identity; the caller (fn) owns the lifecycle timestamps and status:
	//
	//   - SnapshotID: if id is empty, the store generates a fresh ID;
	//     otherwise the store uses id (any SnapshotID populated by fn is
	//     overridden).
	//   - SessionID: the ID of the session (chain of snapshots) the row
	//     belongs to: preserved from the existing row on update (a row's
	//     session never changes once set), otherwise taken from fn's row
	//     as-is. Stores never mint or infer session IDs.
	//   - CreatedAt / UpdatedAt: caller-managed. The store persists whatever fn
	//     returns and never stamps them. fn sets CreatedAt and UpdatedAt to the
	//     current time on a new row, preserves CreatedAt and advances UpdatedAt
	//     on a state-changing rewrite, and preserves both on a non-state write
	//     (e.g. a heartbeat refresh, which carries the existing snapshot through
	//     unchanged but for HeartbeatAt). Keeping timestamps with the caller is
	//     what lets a heartbeat advance liveness without registering as a state
	//     change - the store has no special heartbeat path.
	//   - Status: if the snapshot returned by fn has Status="", it is
	//     defaulted to [SnapshotStatusCompleted] (the common case for
	//     synchronous turn-end writes). Callers writing a pending row must
	//     set Status explicitly.
	//
	// fn receives the existing snapshot (or nil if id is empty or the
	// row does not exist) and returns the snapshot to commit, or
	// (nil, nil) to skip the write without changing the row.
	//
	// Under contention, stores that use optimistic concurrency or
	// transaction retries may call fn multiple times. fn must therefore
	// be a pure function of its input: no side effects (channel sends,
	// logging, external I/O) inside fn.
	//
	// Returns the snapshot as persisted (with the store-owned fields
	// populated), or nil if fn declined to write.
	SaveSnapshot(
		ctx context.Context,
		snapshotID string,
		fn func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error),
	) (*SessionSnapshot[State], error)
}

// SnapshotSubscriber is the optional capability layered on [SessionStore] that
// lets the agent runtime observe a snapshot's status changes without polling.
// It is what makes a detached invocation abortable: aborting is an ordinary
// [SnapshotWriter.SaveSnapshot] that flips a pending row to
// [SnapshotStatusAborting], and the runtime reacts to that flip through this
// subscription, promptly cancelling the background work context.
//
// A store that does not implement it cannot support detach (there is no way to
// signal the background work to stop); see the runtime's detach precondition
// check.
type SnapshotSubscriber interface {
	// OnSnapshotStatusChange returns a channel that yields the snapshot's
	// status whenever it changes. The first value (if any) reflects the
	// status at subscription time. The channel is closed when ctx is
	// cancelled. If the snapshot does not exist when the subscription is
	// established, the channel is closed without yielding a value.
	//
	// Implementations may push changes from a transaction log or CDC feed,
	// or poll internally.
	OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan SnapshotStatus
}

// SessionStore is the minimum store interface required by
// [WithSessionStore]. Status-change observation is layered as the optional
// [SnapshotSubscriber] capability and checked at runtime: a store wired
// into an agent that intends to support detach must also implement
// [SnapshotSubscriber], or the runtime will reject detach attempts.
type SessionStore[State any] interface {
	SnapshotReader[State]
	SnapshotWriter[State]
}

// jsonClone deep-copies v via JSON marshal/unmarshal. Returns nil if v
// is nil. Panics on marshal/unmarshal failure: callers use this for
// types we control (messages, artifacts) where serialization failure
// indicates a programmer error, not a runtime condition.
func jsonClone[T any](v *T) *T {
	if v == nil {
		return nil
	}
	bytes, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("agent: jsonClone marshal: %v", err))
	}
	var out T
	if err := json.Unmarshal(bytes, &out); err != nil {
		panic(fmt.Sprintf("agent: jsonClone unmarshal: %v", err))
	}
	return &out
}

// cloneArtifacts returns a deep copy of arts. Returns nil if arts is empty.
func cloneArtifacts(arts []*Artifact) []*Artifact {
	if len(arts) == 0 {
		return nil
	}
	out := make([]*Artifact, len(arts))
	for i, a := range arts {
		out[i] = jsonClone(a)
	}
	return out
}

// --- Snapshot companion actions ---

// readSnapshot resolves a snapshot by ID, or by the session's latest when
// snapshotID is empty, and returns a normalized copy shaped for a client:
// the documented defaults are applied (empty status means completed, zero
// UpdatedAt means CreatedAt), an in-flight row (pending, or aborting while
// its worker winds down) whose heartbeat has gone stale is surfaced as
// [SnapshotStatusExpired] (computed on read, never written back), and
// transform shapes the outbound state. It backs the getSnapshot and
// waitForSnapshot companion actions and the typed [Agent.GetSnapshot] /
// [Agent.GetLatestSnapshot], so Go callers, the Dev UI, and non-Go clients all
// observe identically shaped snapshots. At least one of snapshotID / sessionID
// must be non-empty; callers validate that before calling. op names the
// operation the read serves, so a failure reports the caller's own name rather
// than always the read's.
//
// With metadataOnly set the response carries the shaped metadata only: the
// shaping above needs nothing but the metadata (status defaulting and
// heartbeat expiry), so the state is dropped instead of cloned and
// transformed. The transform never runs, which matches a stateless row's full
// read: the transform shapes outbound state, and none goes out.
func readSnapshot[State any](
	ctx context.Context,
	store SnapshotReader[State],
	transform StateTransform[State],
	op, snapshotID, sessionID string,
	metadataOnly bool,
) (*SessionSnapshot[State], error) {
	// Resolve the snapshot. A snapshot ID fetches that exact row; a session ID
	// alone fetches the session's latest row (whatever its status). When both
	// are present the snapshot ID picks the row and the session ID asserts it
	// belongs to that session, mirroring AgentInit's combined-ID check. A
	// metadata-only read takes the store's metadata path when it has one
	// ([SnapshotMetadataReader]) and a full read otherwise; the state is
	// dropped below either way.
	var (
		snap *SessionSnapshot[State]
		err  error
	)
	if snapshotID != "" {
		if metadataOnly {
			snap, err = getSnapshotMetadata(ctx, store, snapshotID)
		} else {
			snap, err = store.GetSnapshot(ctx, snapshotID)
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		if snap == nil {
			return nil, status.PublicErrorf(ErrSnapshotNotFound, "%s: snapshot %q not found", op, snapshotID)
		}
		if sessionID != "" && snap.SessionID != sessionID {
			return nil, status.Errorf(status.ErrInvalidArgument,
				"%s: snapshot %q does not belong to session %q (snapshot's session: %q)", op, snapshotID, sessionID, snap.SessionID)
		}
	} else {
		if metadataOnly {
			snap, err = getLatestSnapshotMetadata(ctx, store, sessionID)
		} else {
			snap, err = store.GetLatestSnapshot(ctx, sessionID)
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		if snap == nil {
			return nil, status.PublicErrorf(ErrSnapshotNotFound, "%s: no snapshot found for session %q", op, sessionID)
		}
	}

	// Return a normalized copy: the documented defaults (empty status means
	// completed, zero UpdatedAt means CreatedAt) are resolved here so every
	// caller sees the same shaping, and the state transform shapes what leaves
	// the server. A failed snapshot's state is what its turn committed, so it
	// is returned like any other row's.
	resp := *snap
	// Surface an in-flight row (pending, or aborting while its worker drains
	// toward the finalize) whose heartbeat has gone stale as expired: its
	// detached background worker is presumed dead, so report the orphan
	// rather than leaving it in flight forever. Computed on read only, never
	// written back to the store, so the raw row keeps its status, which is
	// what the worker's abort subscription and resumeSessionFrom key on.
	// Checked before the empty-status default below, which applies only to a
	// row carrying no status at all.
	if isHeartbeatExpired(snap, defaultHeartbeatTimeout) {
		resp.Status = SnapshotStatusExpired
	}
	if resp.Status == "" {
		resp.Status = SnapshotStatusCompleted
	}
	if resp.UpdatedAt.IsZero() {
		resp.UpdatedAt = resp.CreatedAt
	}
	// A metadata-only read is done: shaping needed only the metadata. The
	// state is dropped here for both read paths, the capability's (which
	// never loaded it) and the fallback's (which did), on the copy rather
	// than the store's row.
	if metadataOnly {
		resp.State = nil
		return &resp, nil
	}
	// Clone before transforming: the [StateTransform] contract promises a fresh
	// deep copy the transform may mutate in place, and the store's row may share
	// memory with its internal copy, which neither the transform nor the SessionID
	// re-stamp below may write into. A transform error fails the read closed,
	// with the transform's own status (e.g. PERMISSION_DENIED) preserved.
	transformed, err := applyTransform(ctx, transform, jsonClone(snap.State))
	if err != nil {
		return nil, err
	}
	resp.State = transformed
	if resp.State != nil {
		// SessionID is framework identity, not user data: re-stamp it from the
		// row after the transform so outbound state always agrees with the
		// snapshot it came from.
		resp.State.SessionID = resp.SessionID
	}
	return &resp, nil
}

// Cadences of [waitSnapshot]'s re-reads. Package-level so tests can shorten
// them.
var (
	// snapshotWaitPollInterval is how often a wait re-reads a pending row when
	// the store cannot push status changes.
	snapshotWaitPollInterval = 2 * time.Second
	// snapshotWaitLivenessInterval is how often a subscribed wait re-reads a
	// pending row to notice a heartbeat that has gone stale. Expiry needs two
	// missed beats (defaultHeartbeatTimeout is twice the interval), so checking
	// once per beat cannot miss a stale row by more than one beat.
	snapshotWaitLivenessInterval = defaultHeartbeatInterval
)

const (
	// snapshotWaitProgressInterval is how often a still-blocked [waitSnapshot]
	// logs progress, so a long wait stays visible in the logs and in its span
	// without a line per check.
	snapshotWaitProgressInterval = 30 * time.Second
	// snapshotWaitReadTimeout bounds one store read inside a wait. A wait is
	// long by design and a read is not, and only the wait can tell the two
	// apart, so this is where a hung store is caught: without it an unbounded
	// wait would freeze on one rather than surface the read error. It is
	// generous, because it guards against a hang and not against a slow store.
	snapshotWaitReadTimeout = 30 * time.Second
	// snapshotWaitReadRetries is how many consecutive in-wait re-read failures
	// a wait rides out, at its own re-read cadence, before surfacing the error.
	// A wait runs for as long as the work does, so one store blip must not
	// fail it; dead ends (see waitReadDeadEnd) are surfaced at once.
	snapshotWaitReadRetries = 3
)

// waitReadDeadEnd reports whether an in-wait re-read failure cannot be helped
// by retrying: the row is gone or the request itself is rejected. Anything
// else (a store blip, a read that hit snapshotWaitReadTimeout) is presumed
// transient.
func waitReadDeadEnd(err error) bool {
	// One chain walk, and the set reads as the policy it is. Subtypes carry
	// their base's status (ErrSnapshotNotFound is a NOT_FOUND), so they land
	// here too; an unclassified failure is presumed transient.
	if s, ok := status.Classified(err); ok {
		return slices.Contains(waitReadDeadEndStatuses, s)
	}
	return false
}

// waitReadDeadEndStatuses are the read failures no retry can help.
var waitReadDeadEndStatuses = []status.Name{
	status.NotFound,
	status.InvalidArgument,
	status.FailedPrecondition,
}

// waitSnapshot resolves a snapshot exactly as [readSnapshot] does and then
// blocks until it settles, returning the terminal snapshot with the same
// shaping a read applies. A snapshot that is already terminal returns at once,
// so the wait costs one read in the common case.
//
// Where the store implements [SnapshotSubscriber] the wait is push-driven: it
// subscribes before it would re-read, so a settlement racing the subscription
// is still delivered (the channel yields the status at subscription time). It
// still re-reads on an interval, because expiry is not a write: a dead worker
// leaves the row pending and only its heartbeat goes stale, so no subscription
// can report it. Without a subscriber that same interval is the whole
// mechanism, so it is much shorter.
//
// A read that fails transiently is retried at that same cadence, up to
// snapshotWaitReadRetries consecutive failures, so a store blip does not fail
// a long wait; a dead end (the row is gone, the request is rejected) surfaces
// at once, including on the first read. The success path is unaffected: an
// already-terminal snapshot still costs exactly one read.
//
// Cancelling ctx ends the wait with ctx's error. Callers bound a wait with
// [context.WithTimeout] and re-read the row afterwards to learn where it stands.
func waitSnapshot[State any](
	ctx context.Context,
	store SessionStore[State],
	transform StateTransform[State],
	op, snapshotID, sessionID string,
) (*SessionSnapshot[State], error) {
	read := func(metadataOnly bool) (*SessionSnapshot[State], error) {
		readCtx, cancel := context.WithTimeout(ctx, snapshotWaitReadTimeout)
		defer cancel()
		return readSnapshot(readCtx, store, transform, op, snapshotID, sessionID, metadataOnly)
	}

	// retryRead decides what a failed read inside a wait means: a dead end or
	// exhausted retries surface the error, anything else is ridden out at the
	// wait's own re-read cadence (the next tick retries). It owns the policy
	// for the first read and every re-read alike, so a change to the budget or
	// to what counts as a dead end lands in one place.
	readFailures := 0
	retryRead := func(err error) error {
		if waitReadDeadEnd(err) || readFailures >= snapshotWaitReadRetries {
			return err
		}
		readFailures++
		logger.Debug(ctx, "snapshot read failed inside a wait; retrying",
			"snapshotId", snapshotID, "failures", readFailures, "error", err)
		return nil
	}

	// The first read prices the common already-terminal case at exactly one
	// read. A dead end reaches the caller unchanged (e.g. NOT_FOUND for an
	// unknown snapshot); a transient failure falls into the wait below and is
	// retried there on the wait's own cadence, because a store blip at the
	// moment a wait starts is no more fatal than one in the middle of it.
	snap, err := read(false)
	if err == nil && snap.Status.Terminal() {
		return snap, nil
	}
	if err != nil {
		if err := retryRead(err); err != nil {
			return nil, err
		}
	}

	var statusCh <-chan SnapshotStatus
	if sub, ok := store.(SnapshotSubscriber); ok {
		subCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		statusCh = sub.OnSnapshotStatusChange(subCtx, snapshotID)
	}
	_, metaOnly := store.(SnapshotMetadataReader[State])
	interval := snapshotWaitPollInterval
	if statusCh != nil {
		interval = snapshotWaitLivenessInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	start := time.Now()
	lastProgress := start
	logger.Debug(ctx, "waiting for snapshot to settle",
		"snapshotId", snapshotID, "subscribed", statusCh != nil)
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case snapStatus, ok := <-statusCh:
			if !ok {
				// The subscription ended under us (the row was removed, or the
				// store dropped it). Fall back to re-reading, which reports
				// whatever the row says now, at the unsubscribed cadence.
				statusCh = nil
				interval = snapshotWaitPollInterval
				ticker.Reset(interval)
				continue
			}
			// A notification carries a status, not the row, so it only means
			// "re-read now": the row is what the caller gets back, and the
			// shared re-read below is what decides the wait is over.
			if !snapStatus.Terminal() {
				continue
			}
			// A terminal notification fires once and has now been consumed, so
			// the re-read below is the only path left to the settled row. Drop
			// to the poll cadence for the rest of the wait: a store that
			// notifies ahead of read visibility would otherwise leave the
			// settled row unseen until the next liveness beat.
			interval = snapshotWaitPollInterval
			ticker.Reset(interval)
		case <-ticker.C:
		}

		// One re-read serves both wakeups. It keeps sessionID, so the
		// ownership assertion the caller made holds for the whole wait and not
		// just its first read, and it settles the wait only on a row that says
		// so: a store may notify before the write is visible to a reader, and
		// answering a wait with a pending row would break its contract.
		//
		// The re-read is metadata-only where the store can serve one: the
		// loop dispatches on status alone, and a full read of an in-flight
		// row would materialize a state the row does not carry yet (on a
		// store that reconstructs it, the whole parent chain), once per tick
		// for as long as the work runs. Once the metadata says the row
		// settled, one full read fetches the row the caller gets back, and
		// that row settles the wait on its own say-so: expiry is a verdict on
		// the heartbeat rather than a write, so a row that read as expired a
		// moment ago reads as in flight again if a slow worker beat in
		// between, and the wait then goes on. A store without the capability
		// would load the state on the fallback anyway, so it is read in full
		// once instead.
		cur, err := read(metaOnly)
		if err == nil && metaOnly && cur.Status.Terminal() {
			cur, err = read(false)
		}
		if err != nil {
			if err := retryRead(err); err != nil {
				return nil, err
			}
			// Retry soon rather than on the next liveness beat. A subscribed
			// wait's terminal notification fires once and has already been
			// consumed, so after a failure the re-read is the only path left
			// to the settled row and must not be 30s away.
			interval = snapshotWaitPollInterval
			ticker.Reset(interval)
			continue
		}
		if readFailures > 0 {
			readFailures = 0
			ticker.Reset(interval)
		}
		if cur.Status.Terminal() {
			return cur, nil
		}
		if time.Since(lastProgress) >= snapshotWaitProgressInterval {
			lastProgress = time.Now()
			logger.Debug(ctx, "still waiting for snapshot",
				"snapshotId", snapshotID, "elapsedMs", time.Since(start).Milliseconds())
		}
	}
}

// newSnapshotActions creates the agent's companion actions, without
// registering them, when the agent has a [SessionStore] configured:
//
//   - The agent's name under [api.ActionTypeAgentSnapshot] — getSnapshot,
//     the remote counterpart to [SnapshotReader.GetSnapshot] (by snapshot
//     ID) and [SnapshotReader.GetLatestSnapshot] (by session ID) for Dev UI
//     and non-Go clients. Local Go callers use the store reference directly.
//
//   - The agent's name under [api.ActionTypeAgentWait] — waitForSnapshot,
//     getSnapshot's blocking counterpart: it resolves the same request and
//     returns once the snapshot settles. It is how a caller that holds only
//     actions follows a detached invocation without re-dispatching a read per
//     tick, which is one call and one span instead of a stream of them.
//
//   - The agent's name under [api.ActionTypeAgentAbort] — abort,
//     created only when the store also implements [SnapshotSubscriber], so the
//     runtime can react to the abort it writes via SaveSnapshot.
//
// When the agent is client-managed (no store configured), no action is created:
// there is no server-side snapshot to fetch, follow, or abort. Surfacing
// actions only when the underlying capabilities exist keeps the reflected API
// aligned with what the agent can actually do.
//
// The [Agent] retains the returned actions (an absent one is nil) and
// registers them alongside its run action; see [Agent.Register],
// [Agent.GetSnapshotAction], [Agent.WaitForSnapshotAction], and
// [Agent.AbortAction].
func newSnapshotActions[State any](
	agentName string,
	store SessionStore[State],
	transform StateTransform[State],
) (getSnapshot, waitForSnapshot, abort api.Action) {
	if store == nil {
		return nil, nil, nil
	}
	getSnapshotAction := core.NewActionOf(api.ActionTypeAgentSnapshot, agentName, nil,
		func(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[State], error) {
			if req == nil || (req.SnapshotID == "" && req.SessionID == "") {
				return nil, status.Errorf(status.ErrInvalidArgument, "getSnapshot: snapshotId or sessionId is required")
			}

			return readSnapshot(ctx, store, transform, "getSnapshot", req.SnapshotID, req.SessionID, req.MetadataOnly)
		})

	// waitForSnapshot takes getSnapshot's request, so a caller switching from
	// one to the other keeps its payload, but it requires the snapshot ID: a
	// session's latest row is whichever one is latest at resolution time, and
	// waiting on that is a race with the session's next turn. A session ID may
	// still accompany the snapshot ID, where it asserts ownership exactly as it
	// does on a read.
	waitAction := core.NewActionOf(api.ActionTypeAgentWait, agentName, nil,
		func(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[State], error) {
			if req == nil || req.SnapshotID == "" {
				return nil, status.Errorf(status.ErrInvalidArgument, "waitForSnapshot: snapshotId is required")
			}
			return waitSnapshot(ctx, store, transform, "waitForSnapshot", req.SnapshotID, req.SessionID)
		})

	if _, ok := store.(SnapshotSubscriber); !ok {
		// Without a subscriber the runtime cannot react to an abort, so the
		// abort lifecycle is unsupported; don't surface the action.
		return getSnapshotAction, waitAction, nil
	}
	abortAction := core.NewActionOf(api.ActionTypeAgentAbort, agentName, nil,
		func(ctx context.Context, req *AgentAbortRequest) (*AgentAbortResponse, error) {
			if req == nil || req.SnapshotID == "" {
				return nil, status.Errorf(status.ErrInvalidArgument, "abort: snapshotId is required")
			}
			// Aborting is an ordinary SaveSnapshot that flips a pending row to
			// aborted; the store has no dedicated abort method.
			snapStatus, err := abortPendingSnapshot(ctx, store, req.SnapshotID)
			if err != nil {
				return nil, fmt.Errorf("abort: %w", err)
			}
			if snapStatus == "" {
				return nil, status.PublicErrorf(ErrSnapshotNotFound, "abort: snapshot %q not found", req.SnapshotID)
			}
			return &AgentAbortResponse{SnapshotID: req.SnapshotID, Status: snapStatus}, nil
		})
	return getSnapshotAction, waitAction, abortAction
}

// --- Session ---

// Session holds conversation state and provides thread-safe read/write
// access to messages, custom state, and artifacts.
type Session[State any] struct {
	mu    sync.RWMutex
	state SessionState[State]
	store SessionStore[State]

	// onCustomChange, when set by the agent runtime, is invoked after every
	// UpdateCustom mutation (outside the lock) so the runtime can emit a
	// customPatch chunk describing the delta. Nil for a standalone Session,
	// in which case UpdateCustom is silent.
	onCustomChange func()
}

// SessionID returns the ID of the session this conversation belongs to. The
// agent runtime settles it before the agent function runs and keeps it stable
// for the invocation's lifetime, stamping it on every snapshot persisted. It is
// safe to use as a key for external resources tied to the conversation,
// including from code that retrieves the session via [SessionFromContext].
func (s *Session[State]) SessionID() string {
	// Written once at construction, before fn runs and before the session
	// is shared, then never mutated; safe to read without holding mu.
	return s.state.SessionID
}

// State returns a copy of the current state.
func (s *Session[State]) State() *SessionState[State] {
	s.mu.RLock()
	defer s.mu.RUnlock()
	copied := s.copyStateLocked()
	return &copied
}

// Messages returns the current conversation history. The returned slice
// is a fresh copy, but its elements point at the live messages held by
// the session: treat them as read-only, or deep-copy before mutating.
// [Session.State] returns a fully independent copy.
func (s *Session[State]) Messages() []*ai.Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	msgs := make([]*ai.Message, len(s.state.Messages))
	copy(msgs, s.state.Messages)
	return msgs
}

// AddMessages appends messages to the conversation history.
func (s *Session[State]) AddMessages(messages ...*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = append(s.state.Messages, messages...)
}

// SetMessages replaces the conversation history with the given messages.
func (s *Session[State]) SetMessages(messages []*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = messages
}

// UpdateMessages atomically reads the current messages, applies the given
// function, and writes the result back. fn runs while the session's
// internal lock is held: it must not call other Session methods or send
// on a [Responder], or it will deadlock.
func (s *Session[State]) UpdateMessages(fn func([]*ai.Message) []*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = fn(s.state.Messages)
}

// Custom returns the current user-defined custom state.
func (s *Session[State]) Custom() State {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.Custom
}

// customJSON returns a deep, JSON-normalized copy (a map[string]any / []any /
// ... tree) of just the custom state, taken under the lock so it is safe to
// use after the lock is released. Unlike [Session.State] it does not copy the
// messages or artifacts, so the streaming patcher can diff custom on the hot
// path without re-serializing the whole conversation on every mutation.
func (s *Session[State]) customJSON() any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return normalizeJSON(s.state.Custom)
}

// UpdateCustom atomically reads the current custom state, applies the given
// function, and writes the result back. fn runs while the session's
// internal lock is held: it must not call other Session methods or send
// on a [Responder], or it will deadlock.
//
// When the session is driven by an agent invocation, the mutation is streamed
// to the client as an [AgentStreamChunk.CustomPatch] describing the delta (the
// runtime computes and emits it after fn returns). Agents therefore just mutate
// state; they never hand-craft patches.
func (s *Session[State]) UpdateCustom(fn func(State) State) {
	s.mu.Lock()
	s.state.Custom = fn(s.state.Custom)
	s.mu.Unlock()
	// Emit the customPatch delta after releasing the lock: the hook reads
	// session state (and may send on the wire), neither of which is safe to
	// do while holding s.mu.
	if s.onCustomChange != nil {
		s.onCustomChange()
	}
}

// Artifacts returns the current artifacts. The returned slice is a fresh
// copy, but its elements point at the live artifacts held by the
// session: treat them as read-only, or deep-copy before mutating.
// [Session.State] returns a fully independent copy.
func (s *Session[State]) Artifacts() []*Artifact {
	s.mu.RLock()
	defer s.mu.RUnlock()
	arts := make([]*Artifact, len(s.state.Artifacts))
	copy(arts, s.state.Artifacts)
	return arts
}

// AddArtifacts adds artifacts to the session. If an artifact with the same
// name already exists, it is replaced.
func (s *Session[State]) AddArtifacts(artifacts ...*Artifact) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, a := range artifacts {
		replaced := false
		if a.Name != "" {
			for i, existing := range s.state.Artifacts {
				if existing.Name == a.Name {
					s.state.Artifacts[i] = a
					replaced = true
					break
				}
			}
		}
		if !replaced {
			s.state.Artifacts = append(s.state.Artifacts, a)
		}
	}
}

// UpdateArtifacts atomically reads the current artifacts, applies the given
// function, and writes the result back. fn runs while the session's
// internal lock is held: it must not call other Session methods or send
// on a [Responder], or it will deadlock.
func (s *Session[State]) UpdateArtifacts(fn func([]*Artifact) []*Artifact) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Artifacts = fn(s.state.Artifacts)
}

// copyStateLocked returns a deep copy of the state. Caller must hold mu (read or write).
func (s *Session[State]) copyStateLocked() SessionState[State] {
	bytes, err := json.Marshal(s.state)
	if err != nil {
		panic(fmt.Sprintf("agent: failed to marshal state: %v", err))
	}
	var copied SessionState[State]
	if err := json.Unmarshal(bytes, &copied); err != nil {
		panic(fmt.Sprintf("agent: failed to unmarshal state: %v", err))
	}
	return copied
}

// --- Session context ---

var sessionCtxKey = base.NewContextKey[any]()

// NewSessionContext returns a new context with the session attached.
//
// It also publishes a type-erased view of the session's custom state so prompt
// rendering can inject it into templates as {{@state}}. go/ai cannot import this
// package (this package imports go/ai), so the custom state is exposed through a
// getter in internal/base, evaluated at render time so templates see the latest
// values.
func NewSessionContext[State any](ctx context.Context, s *Session[State]) context.Context {
	ctx = sessionCtxKey.NewContext(ctx, s)
	return base.WithPromptState(ctx, func() any { return s.customJSON() })
}

// SessionFromContext retrieves the current session from context.
// Returns nil if no session is in context or if the type doesn't match.
func SessionFromContext[State any](ctx context.Context) *Session[State] {
	session, _ := sessionCtxKey.FromContext(ctx).(*Session[State])
	return session
}

// ArtifactStore is the State-agnostic view of a session's artifact collection.
// Every [Session] satisfies it regardless of its State type, since artifact
// operations do not touch custom state. Middleware and tools that work with
// artifacts without knowing the agent's State type use it via
// [ArtifactStoreFromContext], where [SessionFromContext] cannot help because it
// requires the concrete State.
type ArtifactStore interface {
	// Artifacts returns a snapshot of the session's current artifacts.
	Artifacts() []*Artifact
	// AddArtifacts adds artifacts, replacing any existing artifact of the same
	// name.
	AddArtifacts(artifacts ...*Artifact)
}

// ArtifactStoreFromContext returns the active session's artifacts as a
// State-agnostic [ArtifactStore], or nil if there is no active session in ctx.
// Unlike [SessionFromContext] it does not require knowing the session's State
// type, so it is the accessor for middleware and tools.
func ArtifactStoreFromContext(ctx context.Context) ArtifactStore {
	store, _ := sessionCtxKey.FromContext(ctx).(ArtifactStore)
	return store
}
