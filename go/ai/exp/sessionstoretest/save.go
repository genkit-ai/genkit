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
	"fmt"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp"
)

func (s *suite[State]) testSave(t *testing.T) {
	ctx := context.Background()

	t.Run("GeneratesIDWhenEmpty", func(t *testing.T) {
		// The runtime mints its own IDs, but the contract lets a caller leave
		// the ID to the store, and the IDs it generates must not collide.
		store, clk := s.newStore(t), newClock()
		first := put(t, ctx, store, "", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("one"), clk.now()))
		second := put(t, ctx, store, "", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("two"), clk.now()))
		if first.SnapshotID == "" || second.SnapshotID == "" || first.SnapshotID == second.SnapshotID {
			t.Fatalf("saves with an empty ID got IDs %q and %q, want two distinct generated IDs", first.SnapshotID, second.SnapshotID)
		}
		checkState(t, "first generated row", mustGet(t, ctx, store, first.SnapshotID).State, s.talk("one"))
		checkState(t, "second generated row", mustGet(t, ctx, store, second.SnapshotID).State, s.talk("two"))
	})

	t.Run("UsesTheGivenID", func(t *testing.T) {
		// The store owns identity: an ID fn sets is overridden by the one the
		// caller passed.
		store, clk := s.newStore(t), newClock()
		snap := s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("hello"), clk.now())
		snap.SnapshotID = "from-fn"
		if saved := put(t, ctx, store, "given", snap); saved.SnapshotID != "given" {
			t.Errorf("saved SnapshotID = %q, want %q", saved.SnapshotID, "given")
		}
		if got := get(t, ctx, store, "given"); got == nil || got.SnapshotID != "given" {
			t.Errorf("GetSnapshot(given) = %s, want the row", describe(got))
		}
		if got := get(t, ctx, store, "from-fn"); got != nil {
			t.Errorf("GetSnapshot(from-fn) = %s, want nil: the ID fn sets must not name the row", describe(got))
		}
	})

	t.Run("PassesTheCurrentRowToFn", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		want := s.failedRow("sess", "elsewhere", s.talk("question", "partial answer"), clk.now())
		var called bool
		var first *exp.SessionSnapshot[State]
		if _, err := store.SaveSnapshot(ctx, "row", func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			called, first = true, existing
			return cloneJSON(want), nil
		}); err != nil {
			t.Fatalf("SaveSnapshot(row): %v", err)
		}
		if !called {
			t.Fatal("SaveSnapshot did not call fn")
		}
		if first != nil {
			t.Errorf("fn got %s for a missing row, want nil", describe(first))
		}

		var seen *exp.SessionSnapshot[State]
		if _, err := store.SaveSnapshot(ctx, "row", func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			seen = existing
			return nil, nil
		}); err != nil {
			t.Fatalf("SaveSnapshot(row) again: %v", err)
		}
		want.SnapshotID = "row"
		checkRow(t, "row passed to fn", seen, want)
	})

	t.Run("SkipsTheWriteWhenFnReturnsNil", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		saved, err := store.SaveSnapshot(ctx, "never", func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			return nil, nil
		})
		if err != nil || saved != nil {
			t.Errorf("SaveSnapshot with a declining fn = (%s, %v), want (nil, nil)", describe(saved), err)
		}
		if got := get(t, ctx, store, "never"); got != nil {
			t.Errorf("a declined write created a row: %s", describe(got))
		}

		before := put(t, ctx, store, "row", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("keep"), clk.now()))
		saved, err = store.SaveSnapshot(ctx, "row", func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			return nil, nil
		})
		if err != nil || saved != nil {
			t.Errorf("SaveSnapshot with a declining fn = (%s, %v), want (nil, nil)", describe(saved), err)
		}
		checkRow(t, "row after a declined write", mustGet(t, ctx, store, "row"), before)
	})

	t.Run("ReturnsTheFnError", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		errBoom := errors.New("boom")
		if _, err := store.SaveSnapshot(ctx, "never", func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			return nil, errBoom
		}); !errors.Is(err, errBoom) {
			t.Errorf("SaveSnapshot with a failing fn: err = %v, want one that wraps fn's error", err)
		}
		if got := get(t, ctx, store, "never"); got != nil {
			t.Errorf("a failed fn created a row: %s", describe(got))
		}

		// fn returns a row and an error: the error wins and nothing is written.
		before := put(t, ctx, store, "row", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("keep"), clk.now()))
		changed := s.row("sess", "", exp.SnapshotStatusFailed, s.talk("changed"), clk.now())
		if _, err := store.SaveSnapshot(ctx, "row", func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			return cloneJSON(changed), errBoom
		}); !errors.Is(err, errBoom) {
			t.Errorf("SaveSnapshot with a failing fn: err = %v, want one that wraps fn's error", err)
		}
		checkRow(t, "row after a failed fn", mustGet(t, ctx, store, "row"), before)
	})

	t.Run("DefaultsEmptyStatusToCompleted", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		saved := put(t, ctx, store, "row", s.row("sess", "", "", s.talk("hi"), clk.now()))
		if saved.Status != exp.SnapshotStatusCompleted {
			t.Errorf("returned Status = %q, want %q", saved.Status, exp.SnapshotStatusCompleted)
		}
		if got := mustGet(t, ctx, store, "row"); got.Status != exp.SnapshotStatusCompleted {
			t.Errorf("stored Status = %q, want %q", got.Status, exp.SnapshotStatusCompleted)
		}
	})

	t.Run("PersistsFieldsVerbatim", func(t *testing.T) {
		// The caller owns the lifecycle fields: the store stamps none of them,
		// and the parent need not exist (lineage is informational).
		store, clk := s.newStore(t), newClock()
		want := s.failedRow("sess", "not-a-row", s.talk("question"), clk.now())
		beat := clk.now()
		want.HeartbeatAt = &beat
		want.UpdatedAt = clk.now()
		saved := put(t, ctx, store, "row", want)
		want.SnapshotID = "row"
		checkRow(t, "returned row", saved, want)
		checkRow(t, "stored row", mustGet(t, ctx, store, "row"), want)
	})

	t.Run("KeepsTheGivenSessionID", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		if saved := put(t, ctx, store, "row", s.row("sess-keep", "", exp.SnapshotStatusCompleted, s.talk("hi"), clk.now())); saved.SessionID != "sess-keep" {
			t.Errorf("returned SessionID = %q, want %q", saved.SessionID, "sess-keep")
		}
		if got := mustGet(t, ctx, store, "row"); got.SessionID != "sess-keep" {
			t.Errorf("stored SessionID = %q, want %q", got.SessionID, "sess-keep")
		}
	})

	t.Run("PreservesSessionIDOnRewrite", func(t *testing.T) {
		// A row's session never changes once set: a rewrite that omits or
		// contradicts it keeps the original.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "row", s.pendingRow("sess-orig", "", clk.now()))
		for _, rewrite := range []string{"", "sess-other"} {
			at := clk.now()
			saved, err := store.SaveSnapshot(ctx, "row", func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
				if existing == nil {
					return nil, errors.New("row is missing")
				}
				next := cloneJSON(existing)
				next.SessionID = rewrite
				next.Status = exp.SnapshotStatusCompleted
				next.State = s.talk("done")
				next.UpdatedAt = at
				return next, nil
			})
			if err != nil {
				t.Fatalf("rewrite with session ID %q: %v", rewrite, err)
			}
			if saved.SessionID != "sess-orig" {
				t.Errorf("rewrite with session ID %q returned SessionID %q, want %q", rewrite, saved.SessionID, "sess-orig")
			}
			if got := mustGet(t, ctx, store, "row"); got.SessionID != "sess-orig" {
				t.Errorf("rewrite with session ID %q stored SessionID %q, want %q", rewrite, got.SessionID, "sess-orig")
			}
		}
	})

	t.Run("RejectsRowWithoutSessionID", func(t *testing.T) {
		// Stores never mint or infer a session ID, not even from the parent.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "parent", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("parent"), clk.now()))
		orphan := s.row("", "parent", exp.SnapshotStatusCompleted, s.talk("child"), clk.now())
		if _, err := store.SaveSnapshot(ctx, "child", func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			return cloneJSON(orphan), nil
		}); !errors.Is(err, exp.ErrSessionIDRequired) {
			t.Errorf("SaveSnapshot without a session ID: err = %v, want exp.ErrSessionIDRequired", err)
		}
		if got := get(t, ctx, store, "child"); got != nil {
			t.Errorf("a row without a session ID was stored: %s", describe(got))
		}
	})

	t.Run("KeepsNilAndEmptyStateApart", func(t *testing.T) {
		// A pending row has no state yet; a settled row with an empty
		// conversation has one. A reader tells them apart by nil.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "pending", s.pendingRow("sess", "", clk.now()))
		put(t, ctx, store, "empty", s.row("sess", "pending", exp.SnapshotStatusCompleted, &exp.SessionState[State]{}, clk.now()))
		if got := mustGet(t, ctx, store, "pending"); got.State != nil {
			t.Error("a nil state read back as non-nil")
		}
		if got := mustGet(t, ctx, store, "empty"); got.State == nil {
			t.Error("an empty state read back as nil")
		}
	})

	t.Run("RoundTripsRichState", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		rich := s.richState()
		put(t, ctx, store, "rich", s.row("sess", "", exp.SnapshotStatusCompleted, rich, clk.now()))
		checkState(t, "rich row", mustGet(t, ctx, store, "rich").State, rich)

		// A child that changes every kind of content, so a store that writes
		// diffs carries each kind of change.
		child := cloneJSON(rich)
		child.Messages[1].Metadata["source"] = "mobile"
		delete(child.Messages[1].Metadata, "nested")
		child.Messages = append(child.Messages, ai.NewUserTextMessage("follow-up \x00 with NUL"), ai.NewModelTextMessage("sure"))
		child.Artifacts = []*exp.Artifact{{Name: "summary.txt", Parts: []*ai.Part{ai.NewTextPart("short")}}}
		child.Usage.InputTokens += 100
		put(t, ctx, store, "child", s.row("sess", "rich", exp.SnapshotStatusCompleted, child, clk.now()))
		checkState(t, "child of the rich row", mustGet(t, ctx, store, "child").State, child)
		checkState(t, "rich row after its child", mustGet(t, ctx, store, "rich").State, rich)
	})

	t.Run("RoundTripsLargeState", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		big := s.largeState(largeStateSize)
		put(t, ctx, store, "big", s.row("sess", "", exp.SnapshotStatusCompleted, big, clk.now()))
		checkState(t, "large row", mustGet(t, ctx, store, "big").State, big)

		grown := cloneJSON(big)
		grown.Messages = append(grown.Messages, ai.NewModelTextMessage("one more"))
		leaf := put(t, ctx, store, "grown", s.row("sess", "big", exp.SnapshotStatusCompleted, grown, clk.now()))
		checkState(t, "child of the large row", mustGet(t, ctx, store, "grown").State, grown)

		// Rewrite the leaf with a much smaller state, the way compaction does.
		trimmed := s.talk("summary of a long conversation")
		rewrite(t, ctx, store, leaf, trimmed, clk.now())
		checkState(t, "rewritten child", mustGet(t, ctx, store, "grown").State, trimmed)
		checkState(t, "large row after its child's rewrite", mustGet(t, ctx, store, "big").State, big)
	})

	t.Run("DoesNotAliasCallerValues", func(t *testing.T) {
		// Values the store hands out, and values it was handed, belong to the
		// caller: editing them afterward must not change the stored row.
		store, clk := s.newStore(t), newClock()
		want := s.failedRow("sess", "", s.talk("original"), clk.now())
		saved := put(t, ctx, store, "row", want)
		want.SnapshotID = "row"
		tamper := func(snap *exp.SessionSnapshot[State]) {
			if snap.State != nil && len(snap.State.Messages) > 0 && len(snap.State.Messages[0].Content) > 0 {
				snap.State.Messages[0].Content[0].Text = "tampered"
			}
			if snap.Error != nil {
				snap.Error.Message = "tampered"
				if snap.Error.Details != nil {
					snap.Error.Details["step"] = "tampered"
				}
			}
		}

		tamper(saved)
		checkRow(t, "row after editing the returned row", mustGet(t, ctx, store, "row"), want)
		tamper(mustGet(t, ctx, store, "row"))
		checkRow(t, "row after editing a read", mustGet(t, ctx, store, "row"), want)
		if latest := getLatest(t, ctx, store, "sess"); latest != nil {
			tamper(latest)
		}
		checkRow(t, "row after editing a latest read", mustGet(t, ctx, store, "row"), want)

		if _, err := store.SaveSnapshot(ctx, "row", func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			if existing != nil {
				tamper(existing)
			}
			return nil, nil
		}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		checkRow(t, "row after fn edited its input and declined", mustGet(t, ctx, store, "row"), want)

		next := cloneJSON(want)
		if _, err := store.SaveSnapshot(ctx, "row", func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			return next, nil
		}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		tamper(next)
		checkRow(t, "row after editing the value fn returned", mustGet(t, ctx, store, "row"), want)
	})

	t.Run("SerializesConcurrentRewrites", func(t *testing.T) {
		// Concurrent read-modify-writes of one row each land exactly once: this
		// is what keeps an abort and a racing finalize from overwriting each
		// other.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "row", s.row("sess", "", exp.SnapshotStatusCompleted, &exp.SessionState[State]{}, clk.now()))
		errs := make(chan error, writers)
		var wg sync.WaitGroup
		for i := range writers {
			wg.Add(1)
			go func() {
				defer wg.Done()
				text := fmt.Sprintf("writer %d", i)
				errs <- retry(func() error {
					_, err := store.SaveSnapshot(ctx, "row", func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
						if existing == nil || existing.State == nil {
							return nil, errors.New("row or its state is missing")
						}
						next := cloneJSON(existing)
						next.State.Messages = append(next.State.Messages, ai.NewUserTextMessage(text))
						return next, nil
					})
					return err
				})
			}()
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatalf("a concurrent rewrite kept failing: %v", err)
			}
		}
		counts := map[string]int{}
		got := mustGet(t, ctx, store, "row")
		for _, m := range got.State.Messages {
			counts[m.Text()]++
		}
		for i := range writers {
			if n := counts[fmt.Sprintf("writer %d", i)]; n != 1 {
				t.Errorf("writer %d's message landed %d times, want exactly once (messages: %v)", i, n, counts)
			}
		}
	})

	t.Run("AcceptsConcurrentNewRows", func(t *testing.T) {
		// Concurrent turns of one session, such as forks off one parent, all
		// land, and the session resolves to the newest of them.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "root", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("root"), clk.now()))
		rows := make([]*exp.SessionSnapshot[State], writers)
		for i := range rows {
			rows[i] = s.row("sess", "root", exp.SnapshotStatusCompleted, s.talk("root", fmt.Sprintf("fork %d", i)), clk.now())
		}
		errs := make(chan error, writers)
		var wg sync.WaitGroup
		for i, snap := range rows {
			wg.Add(1)
			go func() {
				defer wg.Done()
				errs <- retry(func() error {
					_, err := store.SaveSnapshot(ctx, fmt.Sprintf("fork-%d", i), func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
						return cloneJSON(snap), nil
					})
					return err
				})
			}()
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatalf("a concurrent new row kept failing: %v", err)
			}
		}
		for i, snap := range rows {
			checkState(t, fmt.Sprintf("fork-%d", i), mustGet(t, ctx, store, fmt.Sprintf("fork-%d", i)).State, snap.State)
		}
		newest := fmt.Sprintf("fork-%d", writers-1)
		if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != newest {
			t.Errorf("latest = %s, want %s, the newest fork", describe(got), newest)
		}
	})
}
