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

package sessionstoretest

import (
	"context"
	"errors"
	"testing"

	"github.com/firebase/genkit/go/ai/exp"
)

// testDelete checks the optional [exp.SessionDeleter] capability: a delete
// removes the whole session, is idempotent, and leaves a store that rejects a
// late writer from the deleted chain, whether or not a new session has since
// reused the ID.
func (s *suite[State]) testDelete(t *testing.T) {
	ctx := context.Background()
	if !s.deleter {
		t.Skip("the store does not implement exp.SessionDeleter")
	}
	newStore := func(t *testing.T) (exp.SessionStore[State], exp.SessionDeleter) {
		store := s.newStore(t)
		return store, store.(exp.SessionDeleter)
	}
	// try saves a row and returns the store's error, for the writes the suite
	// expects the store to refuse.
	try := func(store exp.SessionStore[State], id string, snap *exp.SessionSnapshot[State]) error {
		_, err := store.SaveSnapshot(ctx, id, func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			return cloneJSON(snap), nil
		})
		return err
	}
	del := func(t *testing.T, d exp.SessionDeleter, sessionID string) bool {
		t.Helper()
		removed, err := d.DeleteSession(ctx, sessionID)
		if err != nil {
			t.Fatalf("DeleteSession(%q): %v", sessionID, err)
		}
		return removed
	}

	t.Run("RemovesEverySnapshotOfTheSession", func(t *testing.T) {
		store, d := newStore(t)
		clk := newClock()
		put(t, ctx, store, "a1", s.row("sess-a", "", exp.SnapshotStatusCompleted, s.talk("one"), clk.now()))
		put(t, ctx, store, "a2", s.row("sess-a", "a1", exp.SnapshotStatusCompleted, s.talk("two"), clk.now()))
		if !del(t, d, "sess-a") {
			t.Error("DeleteSession = false, want true")
		}
		for _, id := range []string{"a1", "a2"} {
			if got := get(t, ctx, store, id); got != nil {
				t.Errorf("snapshot %q survived the delete: %s", id, describe(got))
			}
		}
		if got := getLatest(t, ctx, store, "sess-a"); got != nil {
			t.Errorf("latest after the delete = %s, want nil", describe(got))
		}
	})

	t.Run("IsIdempotent", func(t *testing.T) {
		store, d := newStore(t)
		if del(t, d, "never-existed") {
			t.Error("DeleteSession of an unknown session = true, want false")
		}
		put(t, ctx, store, "x1", s.row("sess-x", "", exp.SnapshotStatusCompleted, s.talk("x"), newClock().now()))
		if !del(t, d, "sess-x") {
			t.Fatal("first DeleteSession = false, want true")
		}
		if del(t, d, "sess-x") {
			t.Error("second DeleteSession = true, want false")
		}
	})

	t.Run("LeavesOtherSessionsAlone", func(t *testing.T) {
		store, d := newStore(t)
		clk := newClock()
		put(t, ctx, store, "a1", s.row("sess-a", "", exp.SnapshotStatusCompleted, s.talk("a"), clk.now()))
		put(t, ctx, store, "b1", s.row("sess-b", "", exp.SnapshotStatusCompleted, s.talk("b"), clk.now()))
		del(t, d, "sess-a")
		if got := getLatest(t, ctx, store, "sess-b"); got == nil || got.SnapshotID != "b1" {
			t.Errorf("latest of the other session = %s, want b1", describe(got))
		}
	})

	t.Run("RejectsEmptySessionID", func(t *testing.T) {
		_, d := newStore(t)
		if _, err := d.DeleteSession(ctx, ""); !errors.Is(err, exp.ErrSessionIDRequired) {
			t.Errorf("DeleteSession(\"\") error = %v, want ErrSessionIDRequired", err)
		}
	})

	t.Run("RejectsWritersContinuingTheDeletedChain", func(t *testing.T) {
		store, d := newStore(t)
		clk := newClock()
		put(t, ctx, store, "a1", s.row("sess-a", "", exp.SnapshotStatusCompleted, s.talk("one"), clk.now()))
		del(t, d, "sess-a")
		err := try(store, "a2", s.row("sess-a", "a1", exp.SnapshotStatusCompleted, s.talk("late"), clk.now()))
		if !errors.Is(err, exp.ErrSessionDeleted) {
			t.Errorf("late child write error = %v, want ErrSessionDeleted", err)
		}
		if got := getLatest(t, ctx, store, "sess-a"); got != nil {
			t.Errorf("session came back: %s", describe(got))
		}
	})

	t.Run("RejectsRewritingADeletedRow", func(t *testing.T) {
		store, d := newStore(t)
		clk := newClock()
		put(t, ctx, store, "p1", s.pendingRow("sess-p", "", clk.now()))
		del(t, d, "sess-p")
		err := try(store, "p1", s.row("sess-p", "", exp.SnapshotStatusCompleted, s.talk("done"), clk.now()))
		if !errors.Is(err, exp.ErrSessionDeleted) {
			t.Errorf("rewrite of a deleted row error = %v, want ErrSessionDeleted", err)
		}
	})

	t.Run("AllowsAFreshRootSnapshot", func(t *testing.T) {
		store, d := newStore(t)
		clk := newClock()
		put(t, ctx, store, "a1", s.row("sess-a", "", exp.SnapshotStatusCompleted, s.talk("one"), clk.now()))
		del(t, d, "sess-a")
		put(t, ctx, store, "a-new", s.row("sess-a", "", exp.SnapshotStatusCompleted, s.talk("fresh"), clk.now()))
		if got := getLatest(t, ctx, store, "sess-a"); got == nil || got.SnapshotID != "a-new" {
			t.Errorf("latest = %s, want a-new", describe(got))
		}
		put(t, ctx, store, "a-new-2", s.row("sess-a", "a-new", exp.SnapshotStatusCompleted, s.talk("next"), clk.now()))
	})

	t.Run("RejectsLateWritersAfterAFreshRoot", func(t *testing.T) {
		// A new session may reuse the ID, but that must not reopen the old chain:
		// a late write continuing it would otherwise land as the session's
		// latest and bring the deleted conversation back.
		store, d := newStore(t)
		clk := newClock()
		put(t, ctx, store, "a1", s.row("sess-a", "", exp.SnapshotStatusCompleted, s.talk("old"), clk.now()))
		del(t, d, "sess-a")
		put(t, ctx, store, "a-new", s.row("sess-a", "", exp.SnapshotStatusCompleted, s.talk("fresh"), clk.now()))
		err := try(store, "a2", s.row("sess-a", "a1", exp.SnapshotStatusCompleted, s.talk("late"), clk.now()))
		if !errors.Is(err, exp.ErrSessionDeleted) {
			t.Errorf("late write after a fresh root error = %v, want ErrSessionDeleted", err)
		}
		if got := getLatest(t, ctx, store, "sess-a"); got == nil || got.SnapshotID != "a-new" {
			t.Errorf("latest = %s, want a-new (the deleted chain must stay gone)", describe(got))
		}
	})

	t.Run("NotifiesSubscribersAsAborted", func(t *testing.T) {
		if !s.subscriber {
			t.Skip("the store does not implement exp.SnapshotSubscriber")
		}
		store, d := newStore(t)
		put(t, ctx, store, "p1", s.pendingRow("sess-p", "", newClock().now()))
		subCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		ch := store.(exp.SnapshotSubscriber).OnSnapshotStatusChange(subCtx, "p1")
		awaitStatus(t, ch, exp.SnapshotStatusPending)
		del(t, d, "sess-p")
		awaitStatus(t, ch, exp.SnapshotStatusAborted)
		awaitClose(t, ch, false)
	})
}
