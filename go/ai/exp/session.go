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
)

// --- Snapshot ---

// SnapshotStatus describes the lifecycle state of a snapshot. Snapshots
// written for synchronous turns or invocations are always [SnapshotStatusComplete]
// (an empty value is also treated as complete for backwards compatibility).
//
// When a client sets [SessionFlowInput.Detach], the server writes a single
// snapshot with [SnapshotStatusPending] capturing the queued inputs and
// returns its ID immediately. Background processing then either updates that
// snapshot to [SnapshotStatusComplete] / [SnapshotStatusError] when the flow
// finishes, or to [SnapshotStatusCanceled] if the client called
// abortSnapshot in the meantime.
type SnapshotStatus string

const (
	// SnapshotStatusPending indicates a detached invocation is still
	// processing the queued inputs. The snapshot will be rewritten with a
	// terminal status once the flow exits.
	SnapshotStatusPending SnapshotStatus = "pending"
	// SnapshotStatusComplete indicates the snapshot captures a settled state.
	SnapshotStatusComplete SnapshotStatus = "complete"
	// SnapshotStatusCanceled indicates the snapshot's invocation was
	// aborted via the abortSnapshot companion action while detached.
	SnapshotStatusCanceled SnapshotStatus = "canceled"
	// SnapshotStatusError indicates the invocation terminated with an error.
	// The snapshot's Error field describes the failure and resume is
	// rejected with that same error.
	SnapshotStatusError SnapshotStatus = "error"
)

// SessionSnapshot is a persisted point-in-time capture of session state.
type SessionSnapshot[State any] struct {
	// SnapshotID is the unique identifier for this snapshot (UUID).
	SnapshotID string `json:"snapshotId"`
	// ParentID is the ID of the previous snapshot in this timeline.
	ParentID string `json:"parentId,omitempty"`
	// CreatedAt is when the snapshot was created.
	CreatedAt time.Time `json:"createdAt"`
	// UpdatedAt is when the snapshot was last written. For pending snapshots
	// it equals CreatedAt; once the snapshot is finalized it reflects the
	// terminal write.
	UpdatedAt time.Time `json:"updatedAt,omitempty"`
	// Event is what triggered this snapshot.
	Event SnapshotEvent `json:"event"`
	// Status is the lifecycle state of this snapshot. Empty is treated as
	// [SnapshotStatusComplete] for backwards compatibility.
	Status SnapshotStatus `json:"status,omitempty"`
	// Error is the failure message for a snapshot in [SnapshotStatusError].
	// Empty otherwise.
	Error string `json:"error,omitempty"`
	// PendingInputs is the inputs captured at detach time, in FIFO order.
	// The first entry may be the input that was in flight when detach
	// landed (its turn was suppressed because snapshots were suspended in
	// the same atomic step that captured it); the rest were queued behind
	// it. Set only on snapshots in [SnapshotStatusPending]; cleared when
	// the snapshot is finalized.
	PendingInputs []*SessionFlowInput `json:"pendingInputs,omitempty"`
	// State is the actual conversation state. Empty on a pending snapshot
	// (the queued inputs are in [PendingInputs] and the live state is not
	// yet committed); populated on terminal snapshots.
	State SessionState[State] `json:"state"`
}

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

// applyTransform returns the result of applying t to *state, or state
// unchanged if t is nil. A nil state is returned as-is.
func applyTransform[State any](t SnapshotTransform[State], state *SessionState[State]) *SessionState[State] {
	if t == nil || state == nil {
		return state
	}
	transformed := t(*state)
	return &transformed
}

// --- Session store ---

// SnapshotMetadata is the metadata-only projection of a [SessionSnapshot]:
// identifying fields, lifecycle timestamps, and status. Returned by store
// operations that surface a snapshot's lifecycle state without paying for
// a full state read.
type SnapshotMetadata struct {
	// SnapshotID is the unique identifier for this snapshot.
	SnapshotID string `json:"snapshotId"`
	// ParentID is the ID of the previous snapshot in this timeline.
	ParentID string `json:"parentId,omitempty"`
	// CreatedAt is when the snapshot was first written.
	CreatedAt time.Time `json:"createdAt"`
	// UpdatedAt is when the snapshot was last written.
	UpdatedAt time.Time `json:"updatedAt,omitempty"`
	// Event is what triggered this snapshot.
	Event SnapshotEvent `json:"event"`
	// Status is the lifecycle state of this snapshot.
	Status SnapshotStatus `json:"status,omitempty"`
	// Error is the failure message for a snapshot in [SnapshotStatusError].
	Error string `json:"error,omitempty"`
}

// SnapshotReader retrieves snapshots. The minimum any session store must
// implement to be used with [WithSessionStore].
type SnapshotReader[State any] interface {
	// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
	GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error)
}

// SnapshotWriter persists snapshots. The minimum any session store must
// implement to be used with [WithSessionStore].
type SnapshotWriter[State any] interface {
	// SaveSnapshot persists a snapshot.
	SaveSnapshot(ctx context.Context, snapshot *SessionSnapshot[State]) error
}

// SnapshotAborter is the optional capability layered on [SessionStore]
// that lets a session flow's invocations be aborted. It bundles the two
// methods that must be implemented together for the abort lifecycle to
// function:
//
//   - [SnapshotAborter.AbortSnapshot] flips a pending snapshot's status
//     to canceled (typically called by the abortSnapshot companion
//     action or directly by a Go caller holding the store).
//
//   - [SnapshotAborter.OnSnapshotStatusChange] lets the session flow
//     runtime observe the flip without polling, so it can promptly
//     cancel the work context.
//
// They are bundled because neither is useful alone: flipping status
// with no observer means the running fn never learns it was aborted;
// observing without a way to trigger the flip means no abort can
// happen. Splitting them into separate interfaces made the
// "implemented one, not the other" footgun too easy to hit.
type SnapshotAborter interface {
	// AbortSnapshot atomically transitions a snapshot from
	// [SnapshotStatusPending] to [SnapshotStatusCanceled] and returns the
	// resulting metadata. If the snapshot is in any other status the
	// operation is a no-op and the existing metadata is returned. Returns
	// nil if the snapshot is not found.
	//
	// Implementations must perform the read-and-write atomically (e.g., a
	// transaction or a compare-and-swap). The session flow's abortSnapshot
	// action and finalizer rely on this to avoid a pending row being
	// clobbered by a racing terminal write.
	AbortSnapshot(ctx context.Context, snapshotID string) (*SnapshotMetadata, error)

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

// AbortSnapshot atomically flips a pending snapshot to canceled. If the
// snapshot is already terminal the existing metadata is returned unchanged.
// Returns nil if the snapshot is not found.
func (s *InMemorySessionStore[State]) AbortSnapshot(_ context.Context, snapshotID string) (*SnapshotMetadata, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	snap, ok := s.snapshots[snapshotID]
	if !ok {
		return nil, nil
	}
	if snap.Status == SnapshotStatusPending {
		snap.Status = SnapshotStatusCanceled
		snap.UpdatedAt = time.Now()
		s.notifyLocked(snapshotID, snap.Status)
	}
	return snapshotMetadata(snap), nil
}

// SaveSnapshot persists a snapshot.
func (s *InMemorySessionStore[State]) SaveSnapshot(_ context.Context, snapshot *SessionSnapshot[State]) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	copied, err := copySnapshot(snapshot)
	if err != nil {
		return err
	}
	prev, existed := s.snapshots[copied.SnapshotID]
	s.snapshots[copied.SnapshotID] = copied
	if !existed || prev.Status != copied.Status {
		s.notifyLocked(copied.SnapshotID, copied.Status)
	}
	return nil
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

// snapshotMetadata projects the metadata fields of a snapshot.
func snapshotMetadata[State any](snap *SessionSnapshot[State]) *SnapshotMetadata {
	return &SnapshotMetadata{
		SnapshotID: snap.SnapshotID,
		ParentID:   snap.ParentID,
		CreatedAt:  snap.CreatedAt,
		UpdatedAt:  snap.UpdatedAt,
		Event:      snap.Event,
		Status:     snap.Status,
		Error:      snap.Error,
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

// GetSnapshotRequest is the input for a session flow's getSnapshot companion
// action. The action is registered at `{flowName}/getSnapshot` when the flow
// is defined and is intended for Dev UI and client-side reconnect flows.
type GetSnapshotRequest struct {
	// SnapshotID identifies the snapshot to fetch.
	SnapshotID string `json:"snapshotId"`
}

// GetSnapshotResponse is the output of the getSnapshot companion action. It
// is a client-facing view of the stored snapshot: identifying metadata plus
// the session state, with [WithSnapshotTransform] applied if configured.
//
// Unlike the raw [SessionSnapshot], this response intentionally omits
// internal fields (parent ID, event) and does not leak the snapshot
// envelope beyond what callers need to repopulate a UI.
type GetSnapshotResponse[State any] struct {
	// SnapshotID echoes the requested snapshot ID.
	SnapshotID string `json:"snapshotId"`
	// CreatedAt is when the snapshot record was first written.
	CreatedAt time.Time `json:"createdAt,omitempty"`
	// UpdatedAt is when the snapshot record was last written. Equals
	// CreatedAt for snapshots that have not been rewritten.
	UpdatedAt time.Time `json:"updatedAt,omitempty"`
	// Status is the lifecycle state of the snapshot. See [SnapshotStatus].
	Status SnapshotStatus `json:"status,omitempty"`
	// Error is populated when Status is [SnapshotStatusError].
	Error string `json:"error,omitempty"`
	// PendingInputs is the queued inputs captured at detach time. Populated
	// only when Status is [SnapshotStatusPending].
	PendingInputs []*SessionFlowInput `json:"pendingInputs,omitempty"`
	// State is the session state captured by the snapshot, after any
	// configured transform. Empty when Status is pending or error.
	State *SessionState[State] `json:"state,omitempty"`
}

// AbortSnapshotRequest is the input for the abortSnapshot companion action.
type AbortSnapshotRequest struct {
	// SnapshotID identifies the snapshot whose invocation should be aborted.
	SnapshotID string `json:"snapshotId"`
}

// AbortSnapshotResponse is the output of the abortSnapshot companion action.
type AbortSnapshotResponse struct {
	// SnapshotID echoes the requested snapshot ID.
	SnapshotID string `json:"snapshotId"`
	// Status is the snapshot's status after the abort attempt. For a
	// pending snapshot this is [SnapshotStatusCanceled]. For an
	// already-terminal snapshot this is the existing terminal status (the
	// abort is a no-op).
	Status SnapshotStatus `json:"status,omitempty"`
}

// registerSnapshotActions registers the session flow's companion actions:
//
//   - The flow's name under [api.ActionTypeAgentSnapshot] — getSnapshot,
//     registered whenever a [SessionStore] is configured. The action is
//     the remote counterpart to [SessionStore.GetSnapshot] for Dev UI and
//     non-Go clients; local Go callers use the store reference directly.
//
//   - The flow's name under [api.ActionTypeAgentAbort] — abortSnapshot,
//     registered only when the store implements [SnapshotAborter]
//     (which bundles both the abort trigger and the status-change
//     subscription needed for the runtime to react). Surfacing the
//     action only when the capability is present keeps the reflected
//     API aligned with what the store can actually do.
func registerSnapshotActions[State any](
	r api.Registry,
	flowName string,
	store SessionStore[State],
	transform SnapshotTransform[State],
) {
	core.DefineAction(r, flowName, api.ActionTypeAgentSnapshot, nil, nil,
		func(ctx context.Context, req *GetSnapshotRequest) (*GetSnapshotResponse[State], error) {
			if store == nil {
				return nil, core.NewError(core.FAILED_PRECONDITION,
					"getSnapshot: session flow %q has no session store configured", flowName)
			}
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
				status = SnapshotStatusComplete
			}
			updatedAt := snap.UpdatedAt
			if updatedAt.IsZero() {
				updatedAt = snap.CreatedAt
			}

			resp := &GetSnapshotResponse[State]{
				SnapshotID:    snap.SnapshotID,
				CreatedAt:     snap.CreatedAt,
				UpdatedAt:     updatedAt,
				Status:        status,
				Error:         snap.Error,
				PendingInputs: snap.PendingInputs,
			}
			if status != SnapshotStatusError && status != SnapshotStatusPending {
				resp.State = applyTransform(transform, &snap.State)
			}
			return resp, nil
		})

	aborter, ok := store.(SnapshotAborter)
	if !ok {
		// Store doesn't support the abort lifecycle. Don't surface the
		// action.
		return
	}
	core.DefineAction(r, flowName, api.ActionTypeAgentAbort, nil, nil,
		func(ctx context.Context, req *AbortSnapshotRequest) (*AbortSnapshotResponse, error) {
			if req == nil || req.SnapshotID == "" {
				return nil, core.NewError(core.INVALID_ARGUMENT, "abortSnapshot: snapshotId is required")
			}
			meta, err := aborter.AbortSnapshot(ctx, req.SnapshotID)
			if err != nil {
				return nil, core.NewError(core.INTERNAL, "abortSnapshot: %v", err)
			}
			if meta == nil {
				return nil, core.NewError(core.NOT_FOUND, "abortSnapshot: snapshot %q not found", req.SnapshotID)
			}
			return &AbortSnapshotResponse{SnapshotID: meta.SnapshotID, Status: meta.Status}, nil
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

// InputVariables returns the prompt input stored in the session state.
func (s *Session[State]) InputVariables() any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.InputVariables
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
		panic(fmt.Sprintf("session flow: failed to marshal state: %v", err))
	}
	var copied SessionState[State]
	if err := json.Unmarshal(bytes, &copied); err != nil {
		panic(fmt.Sprintf("session flow: failed to unmarshal state: %v", err))
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
