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
	"errors"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai/exp"
)

// runDeleteStoreTests exercises [exp.SessionDeleter] against any store that
// implements it, so the in-memory and file stores stay behaviorally aligned.
func runDeleteStoreTests(t *testing.T, newStore func(t *testing.T) exp.SessionStore[testState]) {
	ctx := context.Background()

	deleterOf := func(t *testing.T) (exp.SessionStore[testState], exp.SessionDeleter) {
		t.Helper()
		store := newStore(t)
		deleter, ok := store.(exp.SessionDeleter)
		if !ok {
			t.Fatalf("%T does not implement exp.SessionDeleter", store)
		}
		return store, deleter
	}

	save := func(t *testing.T, store exp.SessionStore[testState], id, sessionID, parentID string, st exp.SnapshotStatus) error {
		t.Helper()
		now := time.Now()
		_, err := store.SaveSnapshot(ctx, id,
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					SessionID: sessionID,
					ParentID:  parentID,
					Status:    st,
					State:     &exp.SessionState[testState]{Custom: testState{Counter: 1}},
					CreatedAt: now,
					UpdatedAt: now,
				}, nil
			})
		return err
	}

	mustSave := func(t *testing.T, store exp.SessionStore[testState], id, sessionID, parentID string, st exp.SnapshotStatus) {
		t.Helper()
		if err := save(t, store, id, sessionID, parentID, st); err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", id, err)
		}
	}

	t.Run("RemovesEverySnapshotOfTheSession", func(t *testing.T) {
		store, deleter := deleterOf(t)
		mustSave(t, store, "a1", "sess-a", "", exp.SnapshotStatusCompleted)
		mustSave(t, store, "a2", "sess-a", "a1", exp.SnapshotStatusCompleted)

		removed, err := deleter.DeleteSession(ctx, "sess-a")
		if err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}
		if !removed {
			t.Error("DeleteSession = false, want true")
		}
		for _, id := range []string{"a1", "a2"} {
			if got, err := store.GetSnapshot(ctx, id); err != nil || got != nil {
				t.Errorf("GetSnapshot(%q) = %v, %v; want nil, nil", id, got, err)
			}
		}
		if got, err := store.GetLatestSnapshot(ctx, "sess-a"); err != nil || got != nil {
			t.Errorf("GetLatestSnapshot = %v, %v; want nil, nil", got, err)
		}
	})

	t.Run("IsIdempotent", func(t *testing.T) {
		store, deleter := deleterOf(t)
		if removed, err := deleter.DeleteSession(ctx, "never-existed"); err != nil || removed {
			t.Errorf("DeleteSession(missing) = %v, %v; want false, nil", removed, err)
		}
		mustSave(t, store, "x1", "sess-x", "", exp.SnapshotStatusCompleted)
		if removed, err := deleter.DeleteSession(ctx, "sess-x"); err != nil || !removed {
			t.Fatalf("first DeleteSession = %v, %v; want true, nil", removed, err)
		}
		if removed, err := deleter.DeleteSession(ctx, "sess-x"); err != nil || removed {
			t.Errorf("second DeleteSession = %v, %v; want false, nil", removed, err)
		}
	})

	t.Run("LeavesOtherSessionsAlone", func(t *testing.T) {
		store, deleter := deleterOf(t)
		mustSave(t, store, "a1", "sess-a", "", exp.SnapshotStatusCompleted)
		mustSave(t, store, "b1", "sess-b", "", exp.SnapshotStatusCompleted)

		if _, err := deleter.DeleteSession(ctx, "sess-a"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}
		if got, err := store.GetLatestSnapshot(ctx, "sess-b"); err != nil || got == nil || got.SnapshotID != "b1" {
			t.Errorf("GetLatestSnapshot(sess-b) = %v, %v; want b1", got, err)
		}
	})

	t.Run("RejectsEmptySessionID", func(t *testing.T) {
		_, deleter := deleterOf(t)
		if _, err := deleter.DeleteSession(ctx, ""); !errors.Is(err, exp.ErrSessionIDRequired) {
			t.Errorf("DeleteSession(\"\") error = %v, want ErrSessionIDRequired", err)
		}
	})

	t.Run("RejectsWritersContinuingTheDeletedChain", func(t *testing.T) {
		store, deleter := deleterOf(t)
		mustSave(t, store, "a1", "sess-a", "", exp.SnapshotStatusCompleted)
		if _, err := deleter.DeleteSession(ctx, "sess-a"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}

		err := save(t, store, "a2", "sess-a", "a1", exp.SnapshotStatusCompleted)
		if !errors.Is(err, exp.ErrSessionDeleted) {
			t.Errorf("late child write error = %v, want ErrSessionDeleted", err)
		}
		if got, _ := store.GetLatestSnapshot(ctx, "sess-a"); got != nil {
			t.Errorf("session came back: %v", got)
		}
	})

	t.Run("AllowsAFreshRootSnapshot", func(t *testing.T) {
		store, deleter := deleterOf(t)
		mustSave(t, store, "a1", "sess-a", "", exp.SnapshotStatusCompleted)
		if _, err := deleter.DeleteSession(ctx, "sess-a"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}

		mustSave(t, store, "a-new", "sess-a", "", exp.SnapshotStatusCompleted)
		got, err := store.GetLatestSnapshot(ctx, "sess-a")
		if err != nil || got == nil || got.SnapshotID != "a-new" {
			t.Errorf("GetLatestSnapshot = %v, %v; want a-new", got, err)
		}
		// The tombstone is cleared, so the new session extends normally.
		mustSave(t, store, "a-new-2", "sess-a", "a-new", exp.SnapshotStatusCompleted)
	})

	t.Run("NotifiesSubscribersAsAborted", func(t *testing.T) {
		store, deleter := deleterOf(t)
		sub, ok := store.(exp.SnapshotSubscriber)
		if !ok {
			t.Skipf("%T does not implement exp.SnapshotSubscriber", store)
		}
		mustSave(t, store, "p1", "sess-p", "", exp.SnapshotStatusPending)

		subCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		ch := sub.OnSnapshotStatusChange(subCtx, "p1")
		if got := <-ch; got != exp.SnapshotStatusPending {
			t.Fatalf("initial status = %q, want pending", got)
		}

		if _, err := deleter.DeleteSession(ctx, "sess-p"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}
		select {
		case got, open := <-ch:
			if !open || got != exp.SnapshotStatusAborted {
				t.Fatalf("after delete got %q (open=%v), want aborted", got, open)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("subscriber was not notified of the delete")
		}
		select {
		case _, open := <-ch:
			if open {
				t.Error("channel still open after the aborted notification")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("channel was not closed after the delete")
		}
	})
}
