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

// Package localstore provides single-process [exp.SessionStore] implementations
// suitable for local development, tests, and single-instance apps (CLI tools,
// desktop apps, local web services). For multi-instance production deployments
// use a real database-backed store.
package localstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/google/uuid"

	"github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/status"
)

// InMemorySessionStore provides a thread-safe in-memory snapshot store. State
// is lost when the process exits; use [FileSessionStore] or a real backend
// when persistence is needed.
//
// It implements [exp.SessionStore] and [exp.SnapshotSubscriber].
type InMemorySessionStore[State any] struct {
	// mu is RWMutex so GetSnapshot (which JSON-marshals while holding the
	// lock) can run concurrently with other readers. All writers (Save,
	// Abort, OnSnapshotStatusChange, removeSub) take the full Lock().
	mu        sync.RWMutex
	snapshots map[string]*exp.SessionSnapshot[State]
	subs      map[string][]chan exp.SnapshotStatus
}

// NewInMemorySessionStore creates a new in-memory snapshot store.
func NewInMemorySessionStore[State any]() *InMemorySessionStore[State] {
	return &InMemorySessionStore[State]{
		snapshots: make(map[string]*exp.SessionSnapshot[State]),
		subs:      make(map[string][]chan exp.SnapshotStatus),
	}
}

// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
func (s *InMemorySessionStore[State]) GetSnapshot(_ context.Context, snapshotID string) (*exp.SessionSnapshot[State], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snap, ok := s.snapshots[snapshotID]
	if !ok {
		return nil, nil
	}
	return copySnapshot(snap)
}

// GetSnapshotMetadata retrieves a snapshot by ID without its state, per
// [exp.SnapshotMetadataReader]. Returns nil if not found. The returned row
// has State nil and, like a full read, owns everything else it carries.
func (s *InMemorySessionStore[State]) GetSnapshotMetadata(_ context.Context, snapshotID string) (*exp.SessionSnapshot[State], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snap, ok := s.snapshots[snapshotID]
	if !ok {
		return nil, nil
	}
	return snapshotMetadata(snap)
}

// GetLatestSnapshot returns the session's most recently created snapshot
// regardless of status, per the [exp.SnapshotReader.GetLatestSnapshot]
// contract. Ties on CreatedAt are broken by SnapshotID so resolution is
// deterministic. The returned snapshot is a deep copy.
func (s *InMemorySessionStore[State]) GetLatestSnapshot(_ context.Context, sessionID string) (*exp.SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("InMemorySessionStore: session ID is empty")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	latest := s.latestLocked(sessionID)
	if latest == nil {
		return nil, nil
	}
	return copySnapshot(latest)
}

// GetLatestSnapshotMetadata is [InMemorySessionStore.GetLatestSnapshot]
// without the state, per [exp.SnapshotMetadataReader]: the
// same resolution, and a row copied the way
// [InMemorySessionStore.GetSnapshotMetadata] copies one.
func (s *InMemorySessionStore[State]) GetLatestSnapshotMetadata(_ context.Context, sessionID string) (*exp.SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("InMemorySessionStore: session ID is empty")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	latest := s.latestLocked(sessionID)
	if latest == nil {
		return nil, nil
	}
	return snapshotMetadata(latest)
}

// latestLocked returns the session's most recently created row, ties broken
// by SnapshotID, or nil when the session has none. Caller must hold s.mu.
func (s *InMemorySessionStore[State]) latestLocked(sessionID string) *exp.SessionSnapshot[State] {
	var latest *exp.SessionSnapshot[State]
	for _, snap := range s.snapshots {
		if snap.SessionID != sessionID {
			continue
		}
		if latest == nil || snap.CreatedAt.After(latest.CreatedAt) ||
			(snap.CreatedAt.Equal(latest.CreatedAt) && snap.SnapshotID > latest.SnapshotID) {
			latest = snap
		}
	}
	return latest
}

// snapshotMetadata returns snap without its state, owning everything else it
// carries: a copy with State nil, its own HeartbeatAt value, and its Error
// copied the way a full read copies the whole row, so a caller that edits
// either result never reaches the store's row.
func snapshotMetadata[State any](snap *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
	meta := *snap
	meta.State = nil
	if snap.HeartbeatAt != nil {
		hb := *snap.HeartbeatAt
		meta.HeartbeatAt = &hb
	}
	if snap.Error != nil {
		bytes, err := json.Marshal(snap.Error)
		if err != nil {
			return nil, fmt.Errorf("copy snapshot error: marshal: %w", err)
		}
		var copied status.Error
		if err := json.Unmarshal(bytes, &copied); err != nil {
			return nil, fmt.Errorf("copy snapshot error: unmarshal: %w", err)
		}
		meta.Error = &copied
	}
	return &meta, nil
}

// SaveSnapshot atomically reads, applies fn, and persists. See
// [exp.SnapshotWriter] for the full contract; this implementation calls fn
// exactly once per call.
func (s *InMemorySessionStore[State]) SaveSnapshot(
	_ context.Context,
	id string,
	fn func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error),
) (*exp.SessionSnapshot[State], error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if id == "" {
		id = uuid.New().String()
	}

	var existing *exp.SessionSnapshot[State]
	var prevStatus exp.SnapshotStatus
	if stored, ok := s.snapshots[id]; ok {
		copied, err := copySnapshot(stored)
		if err != nil {
			return nil, err
		}
		existing = copied
		// Captured before fn runs: fn may edit existing in place and return
		// it, and comparing against the same object afterwards would hide the
		// status change from subscribers.
		prevStatus = existing.Status
	}

	next, err := fn(existing)
	if err != nil {
		return nil, err
	}
	if next == nil {
		return nil, nil
	}

	next.SnapshotID = id
	// SessionID is preserved (a row's session never changes); CreatedAt,
	// UpdatedAt, and HeartbeatAt are caller-managed and persisted verbatim.
	if existing != nil && existing.SessionID != "" {
		next.SessionID = existing.SessionID
	}
	if next.SessionID == "" {
		// A snapshot must belong to a session; stores never mint or infer one. The
		// runtime stamps a session ID on every row it writes, so an empty one
		// indicates misuse.
		return nil, status.Errorf(exp.ErrSessionIDRequired, "InMemorySessionStore requires sessionId to be set on the snapshot")
	}
	if next.Status == "" {
		next.Status = exp.SnapshotStatusCompleted
	}

	copied, err := copySnapshot(next)
	if err != nil {
		return nil, err
	}
	s.snapshots[id] = copied
	if existing == nil || prevStatus != next.Status {
		s.notifyLocked(id, next.Status)
	}
	// Return next (the freshly-allocated struct from fn) rather than
	// copied: copied is the pointer the store retains, so returning it
	// would alias the caller's view with the stored row.
	return next, nil
}

// OnSnapshotStatusChange subscribes to status changes for a snapshot. The
// returned channel yields the current status (if any) and any subsequent
// changes, until ctx is cancelled.
func (s *InMemorySessionStore[State]) OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan exp.SnapshotStatus {
	ch := make(chan exp.SnapshotStatus, 1)

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
func (s *InMemorySessionStore[State]) removeSub(snapshotID string, ch chan exp.SnapshotStatus) {
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
// Caller must hold s.mu. A slow subscriber may miss intermediate values, but
// the latest value is always delivered (see [coalesceSend]).
func (s *InMemorySessionStore[State]) notifyLocked(snapshotID string, status exp.SnapshotStatus) {
	for _, ch := range s.subs[snapshotID] {
		coalesceSend(ch, status)
	}
}

// coalesceSend delivers status on a size-1 buffered subscriber channel,
// guaranteeing the latest value stays observable even if an earlier value is
// still unread. Each channel is seeded at subscription time, so a plain
// non-blocking send would drop a newer status while the seed (or a prior
// status) sits in the buffer. Drop any stale unread value first; the caller
// holds the store mutex and is the only sender, so after the drain the send
// always has room. Shared by [InMemorySessionStore] and [FileSessionStore].
func coalesceSend(ch chan exp.SnapshotStatus, status exp.SnapshotStatus) {
	select {
	case <-ch:
	default:
	}
	select {
	case ch <- status:
	default:
	}
}

// copySnapshot creates a deep copy of a snapshot using JSON marshaling.
func copySnapshot[State any](snap *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
	if snap == nil {
		return nil, nil
	}
	bytes, err := json.Marshal(snap)
	if err != nil {
		return nil, fmt.Errorf("copy snapshot: marshal: %w", err)
	}
	var copied exp.SessionSnapshot[State]
	if err := json.Unmarshal(bytes, &copied); err != nil {
		return nil, fmt.Errorf("copy snapshot: unmarshal: %w", err)
	}
	return &copied, nil
}
