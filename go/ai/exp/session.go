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
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

// --- Snapshot ---

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

	// GetLatestSnapshot returns the session's most recently updated
	// snapshot, whatever its status: a pending, failed, or aborted row is
	// returned like any other, and the caller applies its own policy.
	// Returns nil if the session has no rows, and an error if sessionID is
	// empty.
	//
	// "Most recently updated" means the greatest [SessionSnapshot.UpdatedAt],
	// falling back to CreatedAt on rows that lack one; break ties
	// deterministically (e.g. by SnapshotID). This is a plain max-timestamp
	// lookup, implementable as a single indexed query (e.g. WHERE sessionId = ?
	// ORDER BY updatedAt DESC LIMIT 1). ParentID is informational lineage and
	// plays no part in resolution: when history forks, the most recently
	// updated branch wins.
	GetLatestSnapshot(ctx context.Context, sessionID string) (*SessionSnapshot[State], error)
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
	//   - SessionID: the ID of the session (chain of snapshots) the row
	//     belongs to: preserved from the existing row on update (a row's
	//     session never changes once set), otherwise taken from fn's row
	//     as-is. Stores never mint or infer session IDs.
	//   - CreatedAt: stamped to the wall clock on first write; preserved
	//     from the existing row on update.
	//   - UpdatedAt: stamped to the wall clock on every commit.
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

// SnapshotAborter is the optional capability layered on [SessionStore] that
// lets an agent's invocations be aborted. The two methods work together:
// [SnapshotAborter.AbortSnapshot] flips a pending snapshot's status to aborted,
// and [SnapshotAborter.OnSnapshotStatusChange] lets the agent runtime observe
// the flip without polling so it can promptly cancel the work context. A store
// must implement both or neither.
type SnapshotAborter interface {
	// AbortSnapshot atomically transitions a snapshot from
	// [SnapshotStatusPending] to [SnapshotStatusAborted] and returns the
	// resulting status. If the snapshot is in any other status the
	// operation is a no-op and the existing status is returned. Returns
	// an empty status with a nil error if the snapshot is not found, so
	// callers can distinguish "not found" from a real error.
	//
	// Implementations must perform the read-and-write atomically (e.g., a
	// transaction or a compare-and-swap) so a racing terminal write cannot
	// clobber the pending row.
	AbortSnapshot(ctx context.Context, snapshotID string) (SnapshotStatus, error)

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
// [WithSessionStore]. The abort lifecycle is layered as the optional
// [SnapshotAborter] capability and checked at runtime: a store wired
// into an agent that intends to support detach must also implement
// [SnapshotAborter], or the runtime will reject detach attempts.
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

// newSnapshotActions creates the agent's companion actions, without
// registering them, when the agent has a [SessionStore] configured:
//
//   - The agent's name under [api.ActionTypeAgentSnapshot] — getSnapshot,
//     the remote counterpart to [SnapshotReader.GetSnapshot] (by snapshot
//     ID) and [SnapshotReader.GetLatestSnapshot] (by session ID) for Dev UI
//     and non-Go clients. Local Go callers use the store reference directly.
//
//   - The agent's name under [api.ActionTypeAgentAbort] — abortSnapshot,
//     created only when the store also implements [SnapshotAborter] (which
//     bundles both the abort trigger and the status-change subscription
//     needed for the runtime to react).
//
// When the agent is client-managed (no store configured), neither action
// is created: there is no server-side snapshot to fetch or abort.
// Surfacing actions only when the underlying capabilities exist keeps the
// reflected API aligned with what the agent can actually do.
//
// The [Agent] retains the returned actions (an absent one is nil) and
// registers them alongside its run action; see [Agent.Register],
// [Agent.GetSnapshotAction], and [Agent.AbortSnapshotAction].
func newSnapshotActions[State any](
	agentName string,
	store SessionStore[State],
	transform StateTransform[State],
) (getSnapshot, abortSnapshot api.Action) {
	if store == nil {
		return nil, nil
	}
	getSnapshotAction := core.NewAction(agentName, api.ActionTypeAgentSnapshot, nil, nil,
		func(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[State], error) {
			if req == nil || (req.SnapshotID == "" && req.SessionID == "") {
				return nil, core.NewError(core.INVALID_ARGUMENT, "getSnapshot: snapshotId or sessionId is required")
			}

			// Resolve the snapshot. A snapshot ID fetches that exact row; a
			// session ID alone fetches the session's latest row (whatever
			// its status). When both are present the snapshot ID picks the
			// row and the session ID asserts it belongs to that session,
			// mirroring AgentInit's combined-ID check.
			var (
				snap *SessionSnapshot[State]
				err  error
			)
			if req.SnapshotID != "" {
				snap, err = store.GetSnapshot(ctx, req.SnapshotID)
				if err != nil {
					return nil, core.NewError(core.INTERNAL, "getSnapshot: %v", err)
				}
				if snap == nil {
					return nil, core.NewError(core.NOT_FOUND, "getSnapshot: snapshot %q not found", req.SnapshotID)
				}
				if req.SessionID != "" && snap.SessionID != req.SessionID {
					return nil, core.NewError(core.INVALID_ARGUMENT,
						"getSnapshot: snapshot %q does not belong to session %q (snapshot's session: %q)", req.SnapshotID, req.SessionID, snap.SessionID)
				}
			} else {
				snap, err = store.GetLatestSnapshot(ctx, req.SessionID)
				if err != nil {
					return nil, core.NewError(core.INTERNAL, "getSnapshot: %v", err)
				}
				if snap == nil {
					return nil, core.NewError(core.NOT_FOUND, "getSnapshot: no snapshot found for session %q", req.SessionID)
				}
			}

			// Return a normalized copy: the documented defaults (empty
			// status means completed, zero UpdatedAt means CreatedAt) are
			// resolved server-side so remote clients don't reimplement
			// them, and the state transform shapes what leaves the server.
			// A failed snapshot's state is its last-good state, so it is
			// returned like any other.
			resp := *snap
			if resp.Status == "" {
				resp.Status = SnapshotStatusCompleted
			}
			if resp.UpdatedAt.IsZero() {
				resp.UpdatedAt = resp.CreatedAt
			}
			// Clone before transforming: the [StateTransform] contract
			// promises a fresh deep copy the transform may mutate in
			// place, and the store's row may share memory with its
			// internal copy, which neither the transform nor the
			// SessionID re-stamp below may write into.
			resp.State = applyTransform(ctx, transform, jsonClone(snap.State))
			if resp.State != nil {
				// SessionID is framework identity, not user data: re-stamp
				// it from the row after the transform so outbound state
				// always agrees with the snapshot it came from.
				resp.State.SessionID = resp.SessionID
			}
			return &resp, nil
		})

	aborter, ok := store.(SnapshotAborter)
	if !ok {
		// Store doesn't support the abort lifecycle. Don't surface the
		// action.
		return getSnapshotAction, nil
	}
	abortSnapshotAction := core.NewAction(agentName, api.ActionTypeAgentAbort, nil, nil,
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
			return &AbortSnapshotResponse{Status: status}, nil
		})
	return getSnapshotAction, abortSnapshotAction
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
