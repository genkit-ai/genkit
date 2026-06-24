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

// This file is a private session store fixture used only by the agent's
// internal tests (which need access to unexported package symbols and so
// must remain in [package exp]). The production in-memory and file stores
// live in [github.com/firebase/genkit/go/ai/exp/localstore]; importing that
// package here would create an import cycle, since localstore depends on
// exp.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/google/uuid"
)

// testInMemStore is a thread-safe in-memory snapshot store. Its semantics
// mirror localstore.InMemorySessionStore so the agent's internal tests
// exercise the same store behavior that production users see.
type testInMemStore[State any] struct {
	mu        sync.RWMutex
	snapshots map[string]*SessionSnapshot[State]
	subs      map[string][]chan SnapshotStatus
}

func newTestInMemStore[State any]() *testInMemStore[State] {
	return &testInMemStore[State]{
		snapshots: make(map[string]*SessionSnapshot[State]),
		subs:      make(map[string][]chan SnapshotStatus),
	}
}

// snapshotCount reports the number of stored snapshot rows.
func (s *testInMemStore[State]) snapshotCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.snapshots)
}

func (s *testInMemStore[State]) GetSnapshot(_ context.Context, snapshotID string) (*SessionSnapshot[State], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	snap, ok := s.snapshots[snapshotID]
	if !ok {
		return nil, nil
	}
	return testCopySnapshot(snap)
}

func (s *testInMemStore[State]) GetLatestSnapshot(_ context.Context, sessionID string) (*SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("testInMemStore: session ID is empty")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	var latest *SessionSnapshot[State]
	for _, snap := range s.snapshots {
		if snap.SessionID != sessionID {
			continue
		}
		if latest == nil || snap.CreatedAt.After(latest.CreatedAt) ||
			(snap.CreatedAt.Equal(latest.CreatedAt) && snap.SnapshotID > latest.SnapshotID) {
			latest = snap
		}
	}
	if latest == nil {
		return nil, nil
	}
	return testCopySnapshot(latest)
}

func (s *testInMemStore[State]) SaveSnapshot(
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
		copied, err := testCopySnapshot(stored)
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
	// SessionID is preserved (a row's session never changes); CreatedAt,
	// UpdatedAt, and HeartbeatAt are caller-managed and persisted verbatim.
	if existing != nil && existing.SessionID != "" {
		next.SessionID = existing.SessionID
	}
	if next.Status == "" {
		next.Status = SnapshotStatusCompleted
	}

	copied, err := testCopySnapshot(next)
	if err != nil {
		return nil, err
	}
	s.snapshots[id] = copied
	if existing == nil || existing.Status != next.Status {
		s.notifyLocked(id, next.Status)
	}
	return next, nil
}

func (s *testInMemStore[State]) OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan SnapshotStatus {
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

func (s *testInMemStore[State]) removeSub(snapshotID string, ch chan SnapshotStatus) {
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

func (s *testInMemStore[State]) notifyLocked(snapshotID string, status SnapshotStatus) {
	// Coalesce to the latest status, mirroring localstore's coalesceSend: drop
	// a stale unread value before sending so a newer status is never dropped
	// while the seed sits in the size-1 buffer. (Can't share that helper here:
	// localstore imports exp, so exp's test fixture can't import localstore.)
	for _, ch := range s.subs[snapshotID] {
		select {
		case <-ch:
		default:
		}
		select {
		case ch <- status:
		default:
		}
	}
}

func testCopySnapshot[State any](snap *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
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
