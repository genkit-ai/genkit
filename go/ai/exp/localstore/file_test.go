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
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai/exp"
)

func newFileStore(t *testing.T) *FileSessionStore[testState] {
	t.Helper()
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	return store
}

func TestFileSessionStore(t *testing.T) {
	t.Run("EmptyDirRejected", func(t *testing.T) {
		if _, err := NewFileSessionStore[testState](""); err == nil {
			t.Error("expected error for empty dir, got nil")
		}
	})

	t.Run("CreatesMissingDir", func(t *testing.T) {
		dir := filepath.Join(t.TempDir(), "nested", "subdir")
		if _, err := NewFileSessionStore[testState](dir); err != nil {
			t.Fatalf("NewFileSessionStore: %v", err)
		}
		if _, err := os.Stat(dir); err != nil {
			t.Errorf("expected dir to be created, stat: %v", err)
		}
	})

	t.Run("GetMissing", func(t *testing.T) {
		store := newFileStore(t)
		snap, err := store.GetSnapshot(context.Background(), "nonexistent")
		if err != nil {
			t.Fatalf("GetSnapshot failed: %v", err)
		}
		if snap != nil {
			t.Errorf("expected nil, got %v", snap)
		}
	})

	t.Run("SaveWithFixedID", func(t *testing.T) {
		store := newFileStore(t)
		saved, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				if existing != nil {
					t.Errorf("expected nil existing on first save, got %+v", existing)
				}
				return &exp.SessionSnapshot[testState]{
					Status: exp.SnapshotStatusSucceeded,
					State:  &exp.SessionState[testState]{Custom: testState{Counter: 1}},
				}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot failed: %v", err)
		}
		if saved.SnapshotID != "snap-1" {
			t.Errorf("saved SnapshotID = %q, want %q", saved.SnapshotID, "snap-1")
		}
		if saved.CreatedAt.IsZero() || saved.UpdatedAt.IsZero() {
			t.Errorf("expected CreatedAt/UpdatedAt stamped, got created=%v updated=%v",
				saved.CreatedAt, saved.UpdatedAt)
		}
	})

	t.Run("SaveWithEmptyIDGeneratesUUID", func(t *testing.T) {
		store := newFileStore(t)
		saved, err := store.SaveSnapshot(context.Background(), "",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusSucceeded}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		if saved.SnapshotID == "" {
			t.Error("expected store to generate SnapshotID")
		}
	})

	t.Run("GetReturnsCopy", func(t *testing.T) {
		store := newFileStore(t)
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					Status: exp.SnapshotStatusSucceeded,
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

	t.Run("DefaultsEmptyStatusToSucceeded", func(t *testing.T) {
		store := newFileStore(t)
		saved, err := store.SaveSnapshot(context.Background(), "",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		if saved.Status != exp.SnapshotStatusSucceeded {
			t.Errorf("expected Status=succeeded by default, got %q", saved.Status)
		}
	})

	t.Run("NoopFnSkipsWrite", func(t *testing.T) {
		store := newFileStore(t)
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusSucceeded}, nil
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
		if !before.UpdatedAt.Equal(after.UpdatedAt) {
			t.Errorf("noop should not bump UpdatedAt: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
		}
	})

	t.Run("PreservesCreatedAtOnUpdate", func(t *testing.T) {
		store := newFileStore(t)
		saved, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusSucceeded}, nil
			})
		if err != nil {
			t.Fatalf("seed: %v", err)
		}
		time.Sleep(time.Millisecond)
		updated, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				if existing == nil {
					t.Fatal("expected non-nil existing on update")
				}
				return &exp.SessionSnapshot[testState]{
					Status: exp.SnapshotStatusSucceeded,
					State:  &exp.SessionState[testState]{Custom: testState{Counter: 2}},
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

	t.Run("AbortPendingFlipsToAborted", func(t *testing.T) {
		store := newFileStore(t)
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusPending}, nil
			}); err != nil {
			t.Fatalf("seed: %v", err)
		}
		status, err := store.AbortSnapshot(context.Background(), "snap-1")
		if err != nil {
			t.Fatalf("AbortSnapshot: %v", err)
		}
		if status != exp.SnapshotStatusAborted {
			t.Errorf("status = %q, want %q", status, exp.SnapshotStatusAborted)
		}
		snap, _ := store.GetSnapshot(context.Background(), "snap-1")
		if snap.Status != exp.SnapshotStatusAborted {
			t.Errorf("persisted status = %q, want %q", snap.Status, exp.SnapshotStatusAborted)
		}
	})

	t.Run("AbortTerminalIsNoop", func(t *testing.T) {
		store := newFileStore(t)
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusSucceeded}, nil
			}); err != nil {
			t.Fatalf("seed: %v", err)
		}
		status, err := store.AbortSnapshot(context.Background(), "snap-1")
		if err != nil {
			t.Fatalf("AbortSnapshot: %v", err)
		}
		if status != exp.SnapshotStatusSucceeded {
			t.Errorf("status = %q, want %q (no-op on terminal)", status, exp.SnapshotStatusSucceeded)
		}
	})

	t.Run("AbortMissingReturnsEmpty", func(t *testing.T) {
		store := newFileStore(t)
		status, err := store.AbortSnapshot(context.Background(), "nonexistent")
		if err != nil {
			t.Fatalf("AbortSnapshot: %v", err)
		}
		if status != "" {
			t.Errorf("status = %q, want empty (not found)", status)
		}
	})

	t.Run("StatusSubscriptionYieldsCurrentAndChanges", func(t *testing.T) {
		store := newFileStore(t)
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusPending}, nil
			}); err != nil {
			t.Fatalf("seed: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		ch := store.OnSnapshotStatusChange(ctx, "snap-1")

		select {
		case s := <-ch:
			if s != exp.SnapshotStatusPending {
				t.Errorf("initial status = %q, want %q", s, exp.SnapshotStatusPending)
			}
		case <-time.After(time.Second):
			t.Fatal("timeout waiting for initial status")
		}

		if _, err := store.AbortSnapshot(context.Background(), "snap-1"); err != nil {
			t.Fatalf("AbortSnapshot: %v", err)
		}
		select {
		case s := <-ch:
			if s != exp.SnapshotStatusAborted {
				t.Errorf("post-abort status = %q, want %q", s, exp.SnapshotStatusAborted)
			}
		case <-time.After(time.Second):
			t.Fatal("timeout waiting for aborted status")
		}
	})

	t.Run("StatusSubscriptionOnMissingIsClosed", func(t *testing.T) {
		store := newFileStore(t)
		ch := store.OnSnapshotStatusChange(context.Background(), "nonexistent")
		select {
		case _, ok := <-ch:
			if ok {
				t.Error("expected closed channel for missing snapshot")
			}
		case <-time.After(time.Second):
			t.Fatal("timeout waiting on closed channel")
		}
	})

	t.Run("StatusSubscriptionClosesOnCtxCancel", func(t *testing.T) {
		store := newFileStore(t)
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusPending}, nil
			}); err != nil {
			t.Fatalf("seed: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		ch := store.OnSnapshotStatusChange(ctx, "snap-1")
		<-ch // drain initial
		cancel()
		select {
		case _, ok := <-ch:
			if ok {
				select {
				case _, ok2 := <-ch:
					if ok2 {
						t.Error("expected channel closed after ctx cancel")
					}
				case <-time.After(time.Second):
					t.Fatal("timeout waiting for channel close")
				}
			}
		case <-time.After(time.Second):
			t.Fatal("timeout waiting for channel close")
		}
	})

	t.Run("PersistsAcrossStoreInstances", func(t *testing.T) {
		dir := t.TempDir()
		store1, err := NewFileSessionStore[testState](dir)
		if err != nil {
			t.Fatalf("NewFileSessionStore: %v", err)
		}
		if _, err := store1.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					Status: exp.SnapshotStatusSucceeded,
					State:  &exp.SessionState[testState]{Custom: testState{Counter: 42}},
				}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}

		store2, err := NewFileSessionStore[testState](dir)
		if err != nil {
			t.Fatalf("NewFileSessionStore: %v", err)
		}
		got, err := store2.GetSnapshot(context.Background(), "snap-1")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if got == nil {
			t.Fatal("expected snapshot to persist across store instances")
		}
		if got.State.Custom.Counter != 42 {
			t.Errorf("counter = %d, want 42", got.State.Custom.Counter)
		}
	})

	t.Run("InvalidIDRejected", func(t *testing.T) {
		store := newFileStore(t)
		cases := []string{
			"../escape",
			"a/b",
			`a\b`,
			".hidden",
			"foo..bar",
		}
		for _, id := range cases {
			t.Run(id, func(t *testing.T) {
				if _, err := store.GetSnapshot(context.Background(), id); err == nil {
					t.Errorf("GetSnapshot(%q): expected error, got nil", id)
				}
				if _, err := store.SaveSnapshot(context.Background(), id,
					func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
						return &exp.SessionSnapshot[testState]{}, nil
					}); err == nil {
					t.Errorf("SaveSnapshot(%q): expected error, got nil", id)
				}
				if _, err := store.AbortSnapshot(context.Background(), id); err == nil {
					t.Errorf("AbortSnapshot(%q): expected error, got nil", id)
				}
			})
		}
	})

	t.Run("FileWrittenOnDisk", func(t *testing.T) {
		dir := t.TempDir()
		store, err := NewFileSessionStore[testState](dir)
		if err != nil {
			t.Fatalf("NewFileSessionStore: %v", err)
		}
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusSucceeded}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, "snap-1.json")); err != nil {
			t.Errorf("expected snap-1.json on disk: %v", err)
		}
	})

	t.Run("ImplementsSessionStoreAndAborter", func(t *testing.T) {
		var _ exp.SessionStore[testState] = (*FileSessionStore[testState])(nil)
		var _ exp.SnapshotAborter = (*FileSessionStore[testState])(nil)
	})

	t.Run("LatestSnapshotEmpty", func(t *testing.T) {
		store := newFileStore(t)
		latest, err := store.LatestSnapshot(context.Background())
		if err != nil {
			t.Fatalf("LatestSnapshot: %v", err)
		}
		if latest != nil {
			t.Errorf("expected nil on empty store, got %+v", latest)
		}
	})

	t.Run("LatestSnapshotMissingDir", func(t *testing.T) {
		// Construct a store but then remove the dir to simulate a
		// pre-use store. LatestSnapshot should treat "no dir" as
		// "no snapshots" rather than error.
		dir := filepath.Join(t.TempDir(), "nope")
		store, err := NewFileSessionStore[testState](dir)
		if err != nil {
			t.Fatalf("NewFileSessionStore: %v", err)
		}
		if err := os.RemoveAll(dir); err != nil {
			t.Fatalf("RemoveAll: %v", err)
		}
		latest, err := store.LatestSnapshot(context.Background())
		if err != nil {
			t.Fatalf("LatestSnapshot: %v", err)
		}
		if latest != nil {
			t.Errorf("expected nil on missing dir, got %+v", latest)
		}
	})

	t.Run("LatestSnapshotReturnsMostRecent", func(t *testing.T) {
		store := newFileStore(t)
		ctx := context.Background()
		for _, id := range []string{"snap-1", "snap-2", "snap-3"} {
			if _, err := store.SaveSnapshot(ctx, id,
				func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
					return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusSucceeded}, nil
				}); err != nil {
				t.Fatalf("SaveSnapshot %q: %v", id, err)
			}
			// Force monotonically increasing UpdatedAt — the store
			// stamps to wall-clock time and successive writes within
			// the same tick can tie.
			time.Sleep(2 * time.Millisecond)
		}
		latest, err := store.LatestSnapshot(ctx)
		if err != nil {
			t.Fatalf("LatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "snap-3" {
			t.Errorf("expected latest=snap-3, got %+v", latest)
		}
	})

	t.Run("LatestSnapshotSkipsCorruptedFiles", func(t *testing.T) {
		store := newFileStore(t)
		ctx := context.Background()
		good, err := store.SaveSnapshot(ctx, "good",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{Status: exp.SnapshotStatusSucceeded}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot good: %v", err)
		}
		// Drop an unparseable file alongside.
		if err := os.WriteFile(filepath.Join(store.dir, "broken.json"), []byte("not json"), 0o600); err != nil {
			t.Fatalf("seed broken file: %v", err)
		}
		latest, err := store.LatestSnapshot(ctx)
		if err != nil {
			t.Fatalf("LatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != good.SnapshotID {
			t.Errorf("expected to skip broken.json and return %q, got %+v", good.SnapshotID, latest)
		}
	})
}
