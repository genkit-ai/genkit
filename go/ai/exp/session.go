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
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/uuid"
)

// --- Snapshot ---

// SnapshotContext provides context for snapshot decision callbacks.
type SnapshotContext[State any] struct {
	// State is the current state that will be snapshotted if the callback returns true.
	State *SessionState[State]
	// PrevState is the state at the last snapshot, or nil if none exists.
	PrevState *SessionState[State]
	// TurnIndex is the turn number in the current invocation.
	TurnIndex int
	// Event is what triggered this snapshot check.
	Event SnapshotEvent
}

// SnapshotCallback decides whether to create a snapshot.
// If not provided and a store is configured, snapshots are always created.
type SnapshotCallback[State any] = func(ctx context.Context, sc *SnapshotContext[State]) bool

// applyTransform returns the result of applying t to state, or state
// unchanged if t is nil. A nil state is returned as-is.
func applyTransform[State any](ctx context.Context, t StateTransform[State], state *SessionState[State]) *SessionState[State] {
	if t == nil || state == nil {
		return state
	}
	return t(ctx, state)
}

// --- Session store ---

// SnapshotReader retrieves snapshots. The minimum any session store must
// implement to be used with [WithSessionStore].
type SnapshotReader[State any] interface {
	// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
	GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error)
}

// SnapshotWriter persists snapshots. The minimum any session store must
// implement to be used with [WithSessionStore].
type SnapshotWriter[State any] interface {
	// SaveSnapshot atomically reads the snapshot at id (if any), applies
	// fn, and persists the result. The store owns identity and
	// lifecycle-timestamp fields:
	//
	//   - SnapshotID: if id is empty, the store generates a fresh ID;
	//     otherwise the store uses id (any SnapshotID populated by fn is
	//     overridden).
	//   - CreatedAt: stamped to the wall clock on first write; preserved
	//     from the existing row on update.
	//   - UpdatedAt: stamped to the wall clock on every commit.
	//   - Status: if the snapshot returned by fn has Status="", it is
	//     defaulted to [SnapshotStatusSucceeded] (the common case for
	//     synchronous turn-end and invocation-end writes). Callers
	//     writing a pending row must set Status explicitly.
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

// SnapshotAborter is the optional capability layered on [SessionStore]
// that lets an agent's invocations be aborted. It bundles the two
// methods that must be implemented together for the abort lifecycle to
// function:
//
//   - [SnapshotAborter.AbortSnapshot] flips a pending snapshot's status
//     to aborted (typically called by the abortSnapshot companion
//     action or directly by a Go caller holding the store).
//
//   - [SnapshotAborter.OnSnapshotStatusChange] lets the agent runtime
//     observe the flip without polling, so it can promptly cancel the
//     work context.
//
// They are bundled because neither is useful alone: flipping status
// with no observer means the running fn never learns it was aborted;
// observing without a way to trigger the flip means no abort can
// happen.
type SnapshotAborter interface {
	// AbortSnapshot atomically transitions a snapshot from
	// [SnapshotStatusPending] to [SnapshotStatusAborted] and returns the
	// resulting status. If the snapshot is in any other status the
	// operation is a no-op and the existing status is returned. Returns
	// an empty status with a nil error if the snapshot is not found, so
	// callers can distinguish "not found" from a real error.
	//
	// Implementations must perform the read-and-write atomically (e.g., a
	// transaction or a compare-and-swap). The agent's abortSnapshot
	// action and finalizer rely on this to avoid a pending row being
	// clobbered by a racing terminal write.
	AbortSnapshot(ctx context.Context, snapshotID string) (SnapshotStatus, error)

	// OnSnapshotStatusChange returns a channel that yields the snapshot's
	// status whenever it changes. The first value (if any) reflects the
	// status at subscription time. The channel is closed when ctx is
	// cancelled. If the snapshot does not exist when the subscription is
	// established, the channel is closed without yielding a value.
	//
	// Implementations may push changes from a transaction log, a CDC
	// feed, or fall back to polling internally; the contract just spares
	// callers the choice.
	OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan SnapshotStatus
}

// SessionStore is the minimum store interface required by
// [WithSessionStore]. The abort lifecycle is layered as the optional
// [SnapshotAborter] capability and checked at runtime: a store wired
// into a flow that intends to support detach must also implement
// [SnapshotAborter], or the runtime will reject detach attempts.
type SessionStore[State any] interface {
	SnapshotReader[State]
	SnapshotWriter[State]
}

// InMemorySessionStore provides a thread-safe in-memory snapshot store. It
// implements the full set of optional store interfaces (reader, writer,
// aborter, status subscriber).
type InMemorySessionStore[State any] struct {
	// mu is RWMutex so GetSnapshot (which JSON-marshals while holding the
	// lock) can run concurrently with other readers. All writers (Save,
	// Abort, OnSnapshotStatusChange, removeSub) take the full Lock().
	mu        sync.RWMutex
	snapshots map[string]*SessionSnapshot[State]
	subs      map[string][]chan SnapshotStatus
}

// NewInMemorySessionStore creates a new in-memory snapshot store.
func NewInMemorySessionStore[State any]() *InMemorySessionStore[State] {
	return &InMemorySessionStore[State]{
		snapshots: make(map[string]*SessionSnapshot[State]),
		subs:      make(map[string][]chan SnapshotStatus),
	}
}

// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
func (s *InMemorySessionStore[State]) GetSnapshot(_ context.Context, snapshotID string) (*SessionSnapshot[State], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snap, ok := s.snapshots[snapshotID]
	if !ok {
		return nil, nil
	}
	return copySnapshot(snap)
}

// AbortSnapshot atomically flips a pending snapshot to aborted. If the
// snapshot is already terminal the existing status is returned unchanged.
// Returns an empty status if the snapshot is not found.
func (s *InMemorySessionStore[State]) AbortSnapshot(_ context.Context, snapshotID string) (SnapshotStatus, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.snapshots[snapshotID]
	if !ok {
		return "", nil
	}
	if snap.Status == SnapshotStatusPending {
		snap.Status = SnapshotStatusAborted
		snap.UpdatedAt = time.Now()
		s.notifyLocked(snapshotID, snap.Status)
	}
	return snap.Status, nil
}

// SaveSnapshot atomically reads, applies fn, and persists. See the
// [SnapshotWriter] interface for the full contract; this implementation
// satisfies it by holding s.mu for the entire read-modify-write so fn
// is called exactly once per SaveSnapshot call.
func (s *InMemorySessionStore[State]) SaveSnapshot(
	_ context.Context,
	id string,
	fn func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error),
) (*SessionSnapshot[State], error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if id == "" {
		id = uuid.New().String()
	}

	var existing *SessionSnapshot[State]
	if stored, ok := s.snapshots[id]; ok {
		copied, err := copySnapshot(stored)
		if err != nil {
			return nil, err
		}
		existing = copied
	}

	next, err := fn(existing)
	if err != nil {
		return nil, err
	}
	if next == nil {
		return nil, nil
	}

	next.SnapshotID = id
	now := time.Now()
	if existing != nil {
		next.CreatedAt = existing.CreatedAt
	} else {
		next.CreatedAt = now
	}
	next.UpdatedAt = now
	if next.Status == "" {
		next.Status = SnapshotStatusSucceeded
	}

	copied, err := copySnapshot(next)
	if err != nil {
		return nil, err
	}
	s.snapshots[id] = copied
	if existing == nil || existing.Status != next.Status {
		s.notifyLocked(id, next.Status)
	}
	// Return next (the freshly-allocated struct from fn) rather than
	// copied: copied is the pointer the store retains, so returning it
	// would alias the caller's view with the stored row and let future
	// in-place mutations (e.g. AbortSnapshot updating UpdatedAt) leak
	// through.
	return next, nil
}

// OnSnapshotStatusChange subscribes to status changes for a snapshot. The
// returned channel yields the current status (if any) and any subsequent
// changes, until ctx is cancelled.
func (s *InMemorySessionStore[State]) OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan SnapshotStatus {
	ch := make(chan SnapshotStatus, 1)

	s.mu.Lock()
	snap, ok := s.snapshots[snapshotID]
	if !ok {
		s.mu.Unlock()
		close(ch)
		return ch
	}
	ch <- snap.Status
	s.subs[snapshotID] = append(s.subs[snapshotID], ch)
	s.mu.Unlock()

	context.AfterFunc(ctx, func() { s.removeSub(snapshotID, ch) })
	return ch
}

// removeSub detaches a subscriber and closes its channel.
func (s *InMemorySessionStore[State]) removeSub(snapshotID string, ch chan SnapshotStatus) {
	s.mu.Lock()
	defer s.mu.Unlock()
	subs := s.subs[snapshotID]
	i := slices.Index(subs, ch)
	if i < 0 {
		return
	}
	subs = slices.Delete(subs, i, i+1)
	if len(subs) == 0 {
		delete(s.subs, snapshotID)
	} else {
		s.subs[snapshotID] = subs
	}
	close(ch)
}

// notifyLocked publishes status to all live subscribers of snapshotID.
// Caller must hold s.mu. Sends are best-effort: a slow subscriber may miss
// intermediate values, but the store guarantees the latest value visible
// to the subscription is the one persisted at notify time.
func (s *InMemorySessionStore[State]) notifyLocked(snapshotID string, status SnapshotStatus) {
	for _, ch := range s.subs[snapshotID] {
		select {
		case ch <- status:
		default:
		}
	}
}

// copySnapshot creates a deep copy of a snapshot using JSON marshaling.
func copySnapshot[State any](snap *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
	if snap == nil {
		return nil, nil
	}
	bytes, err := json.Marshal(snap)
	if err != nil {
		return nil, fmt.Errorf("copy snapshot: marshal: %w", err)
	}
	var copied SessionSnapshot[State]
	if err := json.Unmarshal(bytes, &copied); err != nil {
		return nil, fmt.Errorf("copy snapshot: unmarshal: %w", err)
	}
	return &copied, nil
}

// --- Snapshot companion actions ---

// registerSnapshotActions registers the agent's companion actions when
// the agent has a [SessionStore] configured:
//
//   - The agent's name under [api.ActionTypeAgentSnapshot] — getSnapshot,
//     the remote counterpart to [SessionStore.GetSnapshot] for Dev UI and
//     non-Go clients. Local Go callers use the store reference directly.
//
//   - The agent's name under [api.ActionTypeAgentAbort] — abortSnapshot,
//     registered only when the store also implements [SnapshotAborter]
//     (which bundles both the abort trigger and the status-change
//     subscription needed for the runtime to react).
//
// When the agent is client-managed (no store configured), neither action
// is registered: there is no server-side snapshot to fetch or abort.
// Surfacing actions only when the underlying capabilities exist keeps the
// reflected API aligned with what the agent can actually do.
func registerSnapshotActions[State any](
	r api.Registry,
	agentName string,
	store SessionStore[State],
	transform StateTransform[State],
) {
	if store == nil {
		return
	}
	core.DefineAction(r, agentName, api.ActionTypeAgentSnapshot, nil, nil,
		func(ctx context.Context, req *GetSnapshotRequest) (*GetSnapshotResponse[State], error) {
			if req == nil || req.SnapshotID == "" {
				return nil, core.NewError(core.INVALID_ARGUMENT, "getSnapshot: snapshotId is required")
			}
			snap, err := store.GetSnapshot(ctx, req.SnapshotID)
			if err != nil {
				return nil, core.NewError(core.INTERNAL, "getSnapshot: %v", err)
			}
			if snap == nil {
				return nil, core.NewError(core.NOT_FOUND, "getSnapshot: snapshot %q not found", req.SnapshotID)
			}

			status := snap.Status
			if status == "" {
				status = SnapshotStatusSucceeded
			}
			updatedAt := snap.UpdatedAt
			if updatedAt.IsZero() {
				updatedAt = snap.CreatedAt
			}

			resp := &GetSnapshotResponse[State]{
				SnapshotID: snap.SnapshotID,
				CreatedAt:  snap.CreatedAt,
				UpdatedAt:  updatedAt,
				Status:     status,
				Error:      snap.Error,
			}
			if status != SnapshotStatusFailed && status != SnapshotStatusPending {
				resp.State = applyTransform(ctx, transform, snap.State)
			}
			return resp, nil
		})

	aborter, ok := store.(SnapshotAborter)
	if !ok {
		// Store doesn't support the abort lifecycle. Don't surface the
		// action.
		return
	}
	core.DefineAction(r, agentName, api.ActionTypeAgentAbort, nil, nil,
		func(ctx context.Context, req *AbortSnapshotRequest) (*AbortSnapshotResponse, error) {
			if req == nil || req.SnapshotID == "" {
				return nil, core.NewError(core.INVALID_ARGUMENT, "abortSnapshot: snapshotId is required")
			}
			status, err := aborter.AbortSnapshot(ctx, req.SnapshotID)
			if err != nil {
				return nil, core.NewError(core.INTERNAL, "abortSnapshot: %v", err)
			}
			if status == "" {
				return nil, core.NewError(core.NOT_FOUND, "abortSnapshot: snapshot %q not found", req.SnapshotID)
			}
			return &AbortSnapshotResponse{SnapshotID: req.SnapshotID, Status: status}, nil
		})
}

// --- Session ---

// Session holds conversation state and provides thread-safe read/write access to messages,
// input variables, custom state, and artifacts.
type Session[State any] struct {
	mu      sync.RWMutex
	state   SessionState[State]
	store   SessionStore[State]
	version uint64 // incremented on every mutation; used to skip redundant snapshots
}

// State returns a copy of the current state.
func (s *Session[State]) State() *SessionState[State] {
	s.mu.RLock()
	defer s.mu.RUnlock()
	copied := s.copyStateLocked()
	return &copied
}

// Messages returns the current conversation history.
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
	s.version++
}

// SetMessages replaces the conversation history with the given messages.
func (s *Session[State]) SetMessages(messages []*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = messages
	s.version++
}

// UpdateMessages atomically reads the current messages, applies the given
// function, and writes the result back.
func (s *Session[State]) UpdateMessages(fn func([]*ai.Message) []*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = fn(s.state.Messages)
	s.version++
}

// Custom returns the current user-defined custom state.
func (s *Session[State]) Custom() State {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.Custom
}

// UpdateCustom atomically reads the current custom state, applies the given
// function, and writes the result back.
func (s *Session[State]) UpdateCustom(fn func(State) State) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Custom = fn(s.state.Custom)
	s.version++
}

// Artifacts returns the current artifacts.
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
	s.version++
}

// UpdateArtifacts atomically reads the current artifacts, applies the given
// function, and writes the result back.
func (s *Session[State]) UpdateArtifacts(fn func([]*Artifact) []*Artifact) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Artifacts = fn(s.state.Artifacts)
	s.version++
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
func NewSessionContext[State any](ctx context.Context, s *Session[State]) context.Context {
	return sessionCtxKey.NewContext(ctx, s)
}

// SessionFromContext retrieves the current session from context.
// Returns nil if no session is in context or if the type doesn't match.
func SessionFromContext[State any](ctx context.Context) *Session[State] {
	session, _ := sessionCtxKey.FromContext(ctx).(*Session[State])
	return session
}
