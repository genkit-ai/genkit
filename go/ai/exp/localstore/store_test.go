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

package localstore

import (
	"context"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai/exp"
)

// testState is the custom-state type used by store unit tests.
type testState struct {
	Counter int      `json:"counter"`
	Topics  []string `json:"topics,omitempty"`
}

// runSessionIDStoreTests exercises the store-owned SessionID settle rules
// and the GetLatestSnapshot contract against any [exp.SessionStore]
// implementation. Both the in-memory and file store test files invoke it
// so the two stores stay behaviorally aligned.
func runSessionIDStoreTests(t *testing.T, newStore func(t *testing.T) exp.SessionStore[testState]) {
	ctx := context.Background()

	saveRow := func(t *testing.T, store exp.SessionStore[testState], id, sessionID, parentID string, status exp.SnapshotStatus) *exp.SessionSnapshot[testState] {
		t.Helper()
		// Timestamps are caller-managed; a fresh row is created now.
		now := time.Now()
		saved, err := store.SaveSnapshot(ctx, id,
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					SessionID: sessionID,
					ParentID:  parentID,
					Status:    status,
					State:     &exp.SessionState[testState]{Custom: testState{Counter: 1}},
					CreatedAt: now,
					UpdatedAt: now,
				}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", id, err)
		}
		return saved
	}

	// tick spaces consecutive writes far enough apart that CreatedAt orders
	// them unambiguously even on coarse clocks.
	tick := func() { time.Sleep(2 * time.Millisecond) }

	t.Run("SessionIDKeptWhenProvided", func(t *testing.T) {
		store := newStore(t)
		saved := saveRow(t, store, "a", "sess-keep", "", exp.SnapshotStatusCompleted)
		if saved.SessionID != "sess-keep" {
			t.Errorf("SessionID = %q, want provided %q", saved.SessionID, "sess-keep")
		}
		stored, err := store.GetSnapshot(ctx, "a")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if stored.SessionID != "sess-keep" {
			t.Errorf("stored row SessionID = %q, want %q", stored.SessionID, "sess-keep")
		}
	})

	t.Run("SessionIDPreservedOnUpdate", func(t *testing.T) {
		store := newStore(t)
		saveRow(t, store, "a", "sess-orig", "", exp.SnapshotStatusPending)
		// Finalize-style rewrite that omits (or even contradicts) the
		// session ID: the existing row's ID wins, a row's session never
		// changes.
		for _, rewrite := range []string{"", "sess-other"} {
			updated, err := store.SaveSnapshot(ctx, "a",
				func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
					if existing == nil {
						t.Fatal("expected existing row on update")
					}
					return &exp.SessionSnapshot[testState]{
						SessionID: rewrite,
						ParentID:  existing.ParentID,
						Status:    exp.SnapshotStatusCompleted,
						State:     &exp.SessionState[testState]{Custom: testState{Counter: 2}},
					}, nil
				})
			if err != nil {
				t.Fatalf("SaveSnapshot update: %v", err)
			}
			if updated.SessionID != "sess-orig" {
				t.Errorf("updated SessionID = %q, want preserved %q", updated.SessionID, "sess-orig")
			}
		}
	})

	t.Run("NoSessionWithoutID", func(t *testing.T) {
		// Stores never mint or infer session IDs (the agent runtime assigns
		// them at invocation start and stamps every row it writes); a row
		// written without one is session-less even when its parent has one.
		store := newStore(t)
		saveRow(t, store, "parent", "sess-1", "", exp.SnapshotStatusCompleted)
		child := saveRow(t, store, "child", "", "parent", exp.SnapshotStatusCompleted)
		if child.SessionID != "" {
			t.Errorf("expected session-less row, got SessionID %q", child.SessionID)
		}
	})

	t.Run("GetLatestSnapshotPicksMostRecent", func(t *testing.T) {
		// IDs deliberately sort against write order so a recency bug (or an
		// accidental reliance on the tie-break) cannot pass by luck.
		store := newStore(t)
		saveRow(t, store, "z", "sess-1", "", exp.SnapshotStatusCompleted)
		tick()
		saveRow(t, store, "m", "sess-1", "z", exp.SnapshotStatusCompleted)
		tick()
		saveRow(t, store, "a", "sess-1", "m", exp.SnapshotStatusCompleted)
		tick()
		saveRow(t, store, "x", "sess-other", "", exp.SnapshotStatusCompleted)

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "a" {
			t.Fatalf("latest = %+v, want most recently written snapshot a", latest)
		}
		// The contract returns the full row, not a header: the runtime
		// loads its state to resume.
		if latest.State == nil || latest.State.Custom.Counter != 1 {
			t.Errorf("latest state = %+v, want full row with counter=1", latest.State)
		}
	})

	t.Run("GetLatestSnapshotByCreatedAt", func(t *testing.T) {
		// Recency is judged by CreatedAt: the newest-created leaf wins, and a
		// later rewrite of an older row (e.g. a detach finalize) does not move
		// it ahead, because the rewrite preserves CreatedAt.
		store := newStore(t)
		saveRow(t, store, "root", "sess-1", "", exp.SnapshotStatusCompleted)
		tick()
		saveRow(t, store, "b1", "sess-1", "root", exp.SnapshotStatusPending)
		tick()
		saveRow(t, store, "b2", "sess-1", "root", exp.SnapshotStatusCompleted)
		tick()
		// Finalize the older row b1; the copy preserves its CreatedAt.
		if _, err := store.SaveSnapshot(ctx, "b1",
			func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				rewritten := *existing
				rewritten.Status = exp.SnapshotStatusCompleted
				rewritten.State = &exp.SessionState[testState]{Custom: testState{Counter: 2}}
				rewritten.UpdatedAt = time.Now()
				return &rewritten, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot finalize: %v", err)
		}

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "b2" {
			t.Errorf("latest = %+v, want newest-created snapshot b2 (finalize must not move b1 ahead)", latest)
		}
	})

	t.Run("GetLatestSnapshotReturnsLatestAnyStatus", func(t *testing.T) {
		// The latest row is returned whatever its status: failed and aborted
		// tips are no longer skipped. Deciding a tip is a dead end is the
		// resume path's job, not the store's. Here the newest row is aborted.
		store := newStore(t)
		saveRow(t, store, "a", "sess-1", "", exp.SnapshotStatusCompleted)
		tick()
		saveRow(t, store, "b", "sess-1", "a", exp.SnapshotStatusFailed)
		tick()
		saveRow(t, store, "c", "sess-1", "a", exp.SnapshotStatusAborted)

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "c" || latest.Status != exp.SnapshotStatusAborted {
			t.Errorf("latest = %+v, want newest row c (aborted)", latest)
		}
	})

	t.Run("GetLatestSnapshotPendingReturned", func(t *testing.T) {
		// A pending row is returned like any other: it marks a detached
		// invocation that is still running, and the runtime needs to see it
		// to reject the resume instead of silently racing the background
		// work.
		store := newStore(t)
		saveRow(t, store, "a", "sess-1", "", exp.SnapshotStatusCompleted)
		tick()
		saveRow(t, store, "b", "sess-1", "a", exp.SnapshotStatusPending)

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "b" || latest.Status != exp.SnapshotStatusPending {
			t.Errorf("latest = %+v, want pending snapshot b", latest)
		}
	})

	t.Run("GetLatestSnapshotUnknownSession", func(t *testing.T) {
		store := newStore(t)
		saveRow(t, store, "a", "sess-1", "", exp.SnapshotStatusCompleted)
		latest, err := store.GetLatestSnapshot(ctx, "sess-unknown")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest != nil {
			t.Errorf("expected nil for unknown session, got %+v", latest)
		}
	})

	t.Run("GetLatestSnapshotEmptySessionID", func(t *testing.T) {
		store := newStore(t)
		if _, err := store.GetLatestSnapshot(ctx, ""); err == nil {
			t.Error("expected error for empty session ID")
		}
	})
}

// abortViaSave flips a pending snapshot to aborted via SaveSnapshot, mirroring
// the agent runtime's abort (the store has no dedicated abort method). Returns
// the resulting status: aborted when it was pending, the existing terminal
// status when already settled, or "" when the snapshot does not exist.
func abortViaSave(t *testing.T, store exp.SessionStore[testState], id string) exp.SnapshotStatus {
	t.Helper()
	now := time.Now()
	saved, err := store.SaveSnapshot(context.Background(), id,
		func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			if existing == nil {
				return nil, nil
			}
			if existing.Status != exp.SnapshotStatusPending {
				return existing, nil
			}
			updated := *existing
			updated.Status = exp.SnapshotStatusAborted
			updated.UpdatedAt = now
			return &updated, nil
		})
	if err != nil {
		t.Fatalf("abortViaSave(%q): %v", id, err)
	}
	if saved == nil {
		return ""
	}
	return saved.Status
}

// runHeartbeatStoreTests exercises a heartbeat refresh - an ordinary
// SaveSnapshot that touches only HeartbeatAt on a still-pending row - against
// any store, so the in-memory and file stores stay behaviorally aligned. The
// central property: a heartbeat is a liveness signal, not a state change, so it
// advances HeartbeatAt but touches neither UpdatedAt nor the store's recency
// ordering.
func runHeartbeatStoreTests(t *testing.T, newStore func(t *testing.T) exp.SessionStore[testState]) {
	ctx := context.Background()
	tick := func() { time.Sleep(2 * time.Millisecond) }

	// beat refreshes a pending snapshot's heartbeat the way the agent runtime
	// does: an ordinary SaveSnapshot carrying the existing row through unchanged
	// but for HeartbeatAt (so caller-managed timestamps are preserved), touching
	// only a still-pending row.
	beat := func(t *testing.T, store exp.SessionStore[testState], id string) {
		t.Helper()
		now := time.Now()
		if _, err := store.SaveSnapshot(ctx, id,
			func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				if existing == nil || existing.Status != exp.SnapshotStatusPending {
					return nil, nil
				}
				updated := *existing
				updated.HeartbeatAt = &now
				return &updated, nil
			}); err != nil {
			t.Fatalf("heartbeat SaveSnapshot: %v", err)
		}
	}
	savePending := func(t *testing.T, store exp.SessionStore[testState], id, sessionID, parentID string) {
		t.Helper()
		now := time.Now()
		if _, err := store.SaveSnapshot(ctx, id,
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{SessionID: sessionID, ParentID: parentID, Status: exp.SnapshotStatusPending, CreatedAt: now, UpdatedAt: now}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", id, err)
		}
	}

	t.Run("AdvancesHeartbeatNotUpdatedAt", func(t *testing.T) {
		store := newStore(t)
		savePending(t, store, "p", "sess", "")
		before, err := store.GetSnapshot(ctx, "p")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		tick()
		beat(t, store, "p")
		after, err := store.GetSnapshot(ctx, "p")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if after.HeartbeatAt == nil {
			t.Error("HeartbeatAt was not stamped")
		}
		if !after.UpdatedAt.Equal(before.UpdatedAt) {
			t.Errorf("heartbeat advanced UpdatedAt: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
		}
		if after.Status != exp.SnapshotStatusPending {
			t.Errorf("status = %q, want pending", after.Status)
		}
	})

	t.Run("NoopOnTerminal", func(t *testing.T) {
		store := newStore(t)
		now := time.Now()
		if _, err := store.SaveSnapshot(ctx, "c",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{SessionID: "sess", Status: exp.SnapshotStatusCompleted, CreatedAt: now, UpdatedAt: now}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		before, err := store.GetSnapshot(ctx, "c")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		beat(t, store, "c")
		after, err := store.GetSnapshot(ctx, "c")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if after.HeartbeatAt != nil {
			t.Errorf("beat stamped a heartbeat on a terminal row: %v", after.HeartbeatAt)
		}
		if !after.UpdatedAt.Equal(before.UpdatedAt) {
			t.Errorf("beat bumped UpdatedAt on a terminal row: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
		}
	})

	t.Run("DoesNotChangeRecency", func(t *testing.T) {
		// A heartbeat must not move a pending row ahead of a newer row in
		// GetLatestSnapshot: recency is by CreatedAt, and a beat must not touch
		// it (nor anything else but HeartbeatAt).
		store := newStore(t)
		savePending(t, store, "old", "sess", "")
		tick()
		now := time.Now()
		if _, err := store.SaveSnapshot(ctx, "new",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{SessionID: "sess", ParentID: "old", Status: exp.SnapshotStatusCompleted, State: &exp.SessionState[testState]{Custom: testState{Counter: 9}}, CreatedAt: now, UpdatedAt: now}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot(new): %v", err)
		}
		tick()
		for i := 0; i < 3; i++ {
			beat(t, store, "old")
			tick()
		}
		latest, err := store.GetLatestSnapshot(ctx, "sess")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "new" {
			t.Errorf("latest = %+v, want \"new\" (a heartbeat must not affect recency)", latest)
		}
	})
}
