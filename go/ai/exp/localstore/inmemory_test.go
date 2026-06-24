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

func TestInMemorySessionStore(t *testing.T) {
	t.Run("GetMissing", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		snap, err := store.GetSnapshot(context.Background(), "nonexistent")
		if err != nil {
			t.Fatalf("GetSnapshot failed: %v", err)
		}
		if snap != nil {
			t.Errorf("expected nil, got %v", snap)
		}
	})

	t.Run("SaveWithFixedID", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		now := time.Now()
		saved, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				if existing != nil {
					t.Errorf("expected nil existing on first save, got %+v", existing)
				}
				return &exp.SessionSnapshot[testState]{
					Status:    exp.SnapshotStatusCompleted,
					State:     &exp.SessionState[testState]{Custom: testState{Counter: 1}},
					CreatedAt: now,
					UpdatedAt: now,
				}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot failed: %v", err)
		}
		if saved.SnapshotID != "snap-1" {
			t.Errorf("saved SnapshotID = %q, want %q", saved.SnapshotID, "snap-1")
		}
		// Timestamps are caller-managed: the store persists them verbatim.
		if !saved.CreatedAt.Equal(now) || !saved.UpdatedAt.Equal(now) {
			t.Errorf("expected caller-set timestamps persisted, got created=%v updated=%v want %v",
				saved.CreatedAt, saved.UpdatedAt, now)
		}
	})

	t.Run("GetReturnsCopy", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					Status: exp.SnapshotStatusCompleted,
					State:  &exp.SessionState[testState]{Custom: testState{Counter: 1}},
				}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		retrieved, _ := store.GetSnapshot(context.Background(), "snap-1")
		retrieved.State.Custom.Counter = 999
		retrieved2, _ := store.GetSnapshot(context.Background(), "snap-1")
		if retrieved2.State.Custom.Counter != 1 {
			t.Errorf("expected counter=1 (isolation), got %d", retrieved2.State.Custom.Counter)
		}
	})

	t.Run("DefaultsEmptyStatusToComplete", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		saved, err := store.SaveSnapshot(context.Background(), "",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		if saved.SnapshotID == "" {
			t.Error("expected store to generate SnapshotID")
		}
		if saved.Status != exp.SnapshotStatusCompleted {
			t.Errorf("expected Status=complete by default, got %q", saved.Status)
		}
	})

	t.Run("NoopFnSkipsWrite", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusCompleted}, nil
			}); err != nil {
			t.Fatalf("seed: %v", err)
		}
		before, _ := store.GetSnapshot(context.Background(), "snap-1")
		noop, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return nil, nil
			})
		if err != nil {
			t.Fatalf("noop SaveSnapshot: %v", err)
		}
		if noop != nil {
			t.Errorf("expected nil return on noop, got %+v", noop)
		}
		after, _ := store.GetSnapshot(context.Background(), "snap-1")
		if before.UpdatedAt != after.UpdatedAt {
			t.Errorf("noop should not bump UpdatedAt: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
		}
	})

	t.Run("PersistsCallerTimestamps", func(t *testing.T) {
		// Timestamps are caller-managed: the store round-trips them verbatim
		// (it does not stamp). The caller preserves CreatedAt and advances
		// UpdatedAt across a rewrite.
		store := NewInMemorySessionStore[testState]()
		created := time.Now()
		saved, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusCompleted, CreatedAt: created, UpdatedAt: created}, nil
			})
		if err != nil {
			t.Fatalf("seed: %v", err)
		}
		time.Sleep(time.Millisecond) // ensure measurable UpdatedAt delta
		later := time.Now()
		updated, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				if existing == nil {
					t.Fatal("expected non-nil existing on update")
				}
				return &exp.SessionSnapshot[testState]{
					Status:    exp.SnapshotStatusCompleted,
					State:     &exp.SessionState[testState]{Custom: testState{Counter: 2}},
					CreatedAt: existing.CreatedAt,
					UpdatedAt: later,
				}, nil
			})
		if err != nil {
			t.Fatalf("update: %v", err)
		}
		if !updated.CreatedAt.Equal(saved.CreatedAt) {
			t.Errorf("CreatedAt not preserved: before=%v after=%v", saved.CreatedAt, updated.CreatedAt)
		}
		if !updated.UpdatedAt.After(saved.UpdatedAt) {
			t.Errorf("UpdatedAt did not advance: before=%v after=%v", saved.UpdatedAt, updated.UpdatedAt)
		}
	})

	t.Run("ImplementsSessionStoreAndSubscriber", func(t *testing.T) {
		var _ exp.SessionStore[testState] = (*InMemorySessionStore[testState])(nil)
		var _ exp.SnapshotSubscriber = (*InMemorySessionStore[testState])(nil)
	})
}

func TestInMemorySessionStore_Heartbeat(t *testing.T) {
	runHeartbeatStoreTests(t, func(t *testing.T) exp.SessionStore[testState] {
		return NewInMemorySessionStore[testState]()
	})
}

func TestInMemorySessionStore_SessionIDs(t *testing.T) {
	runSessionIDStoreTests(t, func(t *testing.T) exp.SessionStore[testState] {
		return NewInMemorySessionStore[testState]()
	})

	t.Run("GetLatestSnapshotReturnsCopy", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		if _, err := store.SaveSnapshot(context.Background(), "a",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					SessionID: "sess-1",
					Status:    exp.SnapshotStatusCompleted,
					State:     &exp.SessionState[testState]{Custom: testState{Counter: 1}},
				}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		first, err := store.GetLatestSnapshot(context.Background(), "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if first == nil || first.State == nil {
			t.Fatalf("expected full tip row, got %+v", first)
		}
		// Mutating the returned row must not leak into the store.
		first.State.Custom.Counter = 999
		second, err := store.GetLatestSnapshot(context.Background(), "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if second == nil || second.State == nil || second.State.Custom.Counter != 1 {
			t.Errorf("expected isolated copy with counter=1, got %+v", second)
		}
	})
}
