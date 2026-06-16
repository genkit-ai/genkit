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
		saved, err := store.SaveSnapshot(ctx, id,
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					SessionID: sessionID,
					ParentID:  parentID,
					Event:     exp.SnapshotEventTurnEnd,
					Status:    status,
					State:     &exp.SessionState[testState]{Custom: testState{Counter: 1}},
				}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", id, err)
		}
		return saved
	}

	// tick spaces consecutive writes far enough apart that UpdatedAt (and
	// the file store's mtimes) order them unambiguously even on coarse
	// clocks.
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
						Event:     existing.Event,
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

	t.Run("GetLatestSnapshotUpdateWins", func(t *testing.T) {
		// Recency is judged by UpdatedAt, not creation order: rewriting a
		// row (e.g. a detach finalize landing after other branches were
		// written) moves it to the front.
		store := newStore(t)
		saveRow(t, store, "root", "sess-1", "", exp.SnapshotStatusCompleted)
		tick()
		saveRow(t, store, "b1", "sess-1", "root", exp.SnapshotStatusPending)
		tick()
		saveRow(t, store, "b2", "sess-1", "root", exp.SnapshotStatusCompleted)
		tick()
		if _, err := store.SaveSnapshot(ctx, "b1",
			func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				rewritten := *existing
				rewritten.Status = exp.SnapshotStatusCompleted
				rewritten.State = &exp.SessionState[testState]{Custom: testState{Counter: 2}}
				return &rewritten, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot finalize: %v", err)
		}

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "b1" {
			t.Errorf("latest = %+v, want freshly finalized snapshot b1", latest)
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
