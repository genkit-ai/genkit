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

// Package sessionstoretest checks [exp.SessionStore] implementations against
// the contract the agent runtime relies on.
//
// Run the suite from a test in the store's package:
//
//	func TestConformance(t *testing.T) {
//		sessionstoretest.Run(t, func(t *testing.T) aix.SessionStore[MyState] {
//			return newEmptyStore[MyState](t)
//		}, nil)
//	}
//
// The suite works only through the store's exported methods, so it suits any
// implementation, including one outside this module. It checks the optional
// capabilities ([exp.SnapshotMetadataReader] and [exp.SnapshotSubscriber]) when
// the store implements them, and [Options] turns on the checks that need more
// than one store: a second instance over the same storage, and a store that
// partitions its rows by a key from the context.
//
// The test data lives in the conversation (messages, artifacts, and usage) and
// the custom state keeps its zero value, so the suite runs with any State type,
// including against a store that supports only one State type.
//
// APIs in this package are under active development and may change in any
// minor version release.
package sessionstoretest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/status"
)

const (
	// eventTimeout bounds the wait for a subscription event the suite expects.
	// It is generous because a store may deliver through a remote listener or
	// a poll; a store that delivers at once is not slowed down by it.
	eventTimeout = 15 * time.Second
	// writers is how many goroutines a concurrency check runs at once.
	writers = 8
	// writeAttempts is how many times a concurrent writer tries one save. A
	// store with optimistic transactions may give up under contention, and
	// callers retry such a save, so the suite does too; what it checks is that
	// no committed write is lost or applied twice.
	writeAttempts = 25
	// chainLength is how many rows the linear chain check writes: enough to
	// cross several boundaries of a store that writes diffs with a full
	// checkpoint every few dozen rows.
	chainLength = 60
	// largeStateSize is the approximate JSON size of the large state, beyond
	// the single-document limits of common databases.
	largeStateSize = 5 << 19 // 2.5 MiB
)

// Options turns on the checks that need more than the store under test. A nil
// *Options runs the single-store checks only.
type Options[State any] struct {
	// Reopen returns a second store over the same storage as store, the way
	// another process would open it. With it, the suite checks that rows
	// written through one instance read back through the other, and that a
	// status change written through one reaches a subscriber on the other,
	// which is how one process aborts a detached run another process executes.
	// The suite subscribes through the reopened store and writes through the
	// original. Leave it nil when the storage is private to one instance.
	Reopen func(t *testing.T, store exp.SessionStore[State]) exp.SessionStore[State]

	// Scoped returns a fresh, empty store that partitions its rows by a key it
	// derives from each call's context, such as a tenant prefix, and two
	// contexts the store maps to different partitions. With it, the suite
	// checks that neither partition can read, change, or subscribe to the
	// other's rows, including under a snapshot ID both use. Leave it nil when
	// the store has no such partitioning.
	Scoped func(t *testing.T) (store exp.SessionStore[State], a, b context.Context)
}

// Run checks the stores newStore returns against the session store contract,
// as subtests of t grouped by area: Save, Lifecycle, Chains, Latest, Metadata,
// Subscriber, SecondInstance, and Scopes. A group whose capability the store
// lacks, or whose [Options] hook is nil, is skipped with the reason.
//
// newStore must return a fresh, empty store on every call, with its default
// behavior: retention that deletes old rows, for example, must be off, because
// the suite reads back every row it writes. Stores from separate calls may
// share storage as long as neither can see the other's rows.
//
// The suite expects timestamps to round-trip to at least millisecond precision
// (it compares instants, so the time zone may change) and the state to
// round-trip as JSON. Its snapshot and session IDs contain only ASCII letters,
// digits, and hyphens. A store that cannot hold one of the larger cases (about
// 2.5 MiB of state) can leave it out with go test -skip.
func Run[State any](t *testing.T, newStore func(t *testing.T) exp.SessionStore[State], opts *Options[State]) {
	t.Helper()
	s := &suite[State]{newStore: newStore}
	if opts != nil {
		s.opts = *opts
	}
	t.Run("Save", s.testSave)
	t.Run("Lifecycle", s.testLifecycle)
	t.Run("Chains", s.testChains)
	t.Run("Latest", s.testLatest)
	t.Run("Metadata", s.testMetadata)
	t.Run("Subscriber", s.testSubscriber)
	t.Run("SecondInstance", s.testSecondInstance)
	t.Run("Scopes", s.testScopes)
}

// suite holds what every check needs: the store factory and the options.
type suite[State any] struct {
	newStore func(t *testing.T) exp.SessionStore[State]
	opts     Options[State]
}

// --- Save ---

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

// --- Lifecycle ---

// testLifecycle drives a detached run's writes the way the runtime does: a
// pending row, heartbeats, an abort, and the finalize, each an ordinary
// SaveSnapshot with the runtime's own mutator.
func (s *suite[State]) testLifecycle(t *testing.T) {
	ctx := context.Background()

	t.Run("HeartbeatAdvancesOnlyHeartbeatAt", func(t *testing.T) {
		// A heartbeat is a liveness signal, not a state change: it moves
		// HeartbeatAt and nothing else, UpdatedAt included.
		store, clk := s.newStore(t), newClock()
		before := put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		at := clk.now()
		beat(t, ctx, store, "p", at)
		want := cloneJSON(before)
		want.HeartbeatAt = &at
		checkRow(t, "row after a heartbeat", mustGet(t, ctx, store, "p"), want)
	})

	t.Run("HeartbeatSkipsSettledRow", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		before := put(t, ctx, store, "c", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("done"), clk.now()))
		beat(t, ctx, store, "c", clk.now())
		checkRow(t, "settled row after a heartbeat", mustGet(t, ctx, store, "c"), before)
	})

	t.Run("HeartbeatKeepsRecency", func(t *testing.T) {
		// Beats on an older row, however recent, do not move it ahead of a
		// newer row: recency is CreatedAt.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "old", s.pendingRow("sess", "", clk.now()))
		put(t, ctx, store, "new", s.row("sess", "old", exp.SnapshotStatusCompleted, s.talk("new"), clk.now()))
		for range 3 {
			beat(t, ctx, store, "old", clk.now())
		}
		if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != "new" {
			t.Errorf("latest = %s, want new (a heartbeat must not affect recency)", describe(got))
		}
	})

	t.Run("AbortFlipsPendingToAborting", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		before := put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		at := clk.now()
		if got := abort(t, ctx, store, "p", at); got != exp.SnapshotStatusAborting {
			t.Errorf("abort returned %q, want %q", got, exp.SnapshotStatusAborting)
		}
		want := cloneJSON(before)
		want.Status = exp.SnapshotStatusAborting
		want.UpdatedAt = at
		checkRow(t, "row after an abort", mustGet(t, ctx, store, "p"), want)

		// A second abort rewrites the row verbatim.
		if got := abort(t, ctx, store, "p", clk.now()); got != exp.SnapshotStatusAborting {
			t.Errorf("second abort returned %q, want %q", got, exp.SnapshotStatusAborting)
		}
		checkRow(t, "row after a second abort", mustGet(t, ctx, store, "p"), want)
	})

	t.Run("AbortLeavesSettledRowAlone", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		before := put(t, ctx, store, "c", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("done"), clk.now()))
		if got := abort(t, ctx, store, "c", clk.now()); got != exp.SnapshotStatusCompleted {
			t.Errorf("abort of a settled row returned %q, want %q", got, exp.SnapshotStatusCompleted)
		}
		checkRow(t, "settled row after an abort", mustGet(t, ctx, store, "c"), before)
	})

	t.Run("AbortOfMissingRowWritesNothing", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		if got := abort(t, ctx, store, "missing", clk.now()); got != "" {
			t.Errorf("abort of a missing row returned %q, want \"\"", got)
		}
		if got := get(t, ctx, store, "missing"); got != nil {
			t.Errorf("abort of a missing row created %s", describe(got))
		}
	})

	t.Run("FinalizeLandsStateAndClearsHeartbeat", func(t *testing.T) {
		// The finalize replaces the row: the fields it leaves out, the
		// heartbeat among them, are gone afterward rather than merged in.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "parent", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("question"), clk.now()))
		pending := put(t, ctx, store, "p", s.pendingRow("sess", "parent", clk.now()))
		beat(t, ctx, store, "p", clk.now())
		abort(t, ctx, store, "p", clk.now())
		final := s.talk("question", "partial answer")
		at := clk.now()
		finalize(t, ctx, store, pending, exp.SnapshotStatusAborted, exp.AgentFinishReasonAborted, final, at)
		checkRow(t, "finalized row", mustGet(t, ctx, store, "p"), &exp.SessionSnapshot[State]{
			SnapshotID:   "p",
			SessionID:    "sess",
			ParentID:     "parent",
			Status:       exp.SnapshotStatusAborted,
			FinishReason: exp.AgentFinishReasonAborted,
			State:        final,
			CreatedAt:    pending.CreatedAt,
			UpdatedAt:    at,
		})
	})

	t.Run("FinalizeSkipsSettledRow", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		before := put(t, ctx, store, "c", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("done"), clk.now()))
		finalize(t, ctx, store, before, exp.SnapshotStatusFailed, exp.AgentFinishReasonFailed, s.talk("other"), clk.now())
		checkRow(t, "settled row after a finalize", mustGet(t, ctx, store, "c"), before)
	})
}

// --- Chains ---

func (s *suite[State]) testChains(t *testing.T) {
	ctx := context.Background()

	t.Run("LinearChainReadsBackEveryRow", func(t *testing.T) {
		// A long session with the edits a real one sees: the history is
		// compacted midway, artifacts come and go, and one turn is detached
		// (a pending row, then the finalize). Every row reads back as written.
		store, clk := s.newStore(t), newClock()
		var (
			ids    []string
			want   = map[string]*exp.SessionState[State]{}
			msgs   []*ai.Message
			arts   []*exp.Artifact
			parent string
		)
		for i := range chainLength {
			id := fmt.Sprintf("turn-%02d", i)
			switch i {
			case chainLength / 2:
				msgs = []*ai.Message{ai.NewSystemTextMessage(fmt.Sprintf("Summary of turns 0 to %d.", i-1))}
			case chainLength/2 + 5:
				arts = []*exp.Artifact{{Name: "draft.md", Parts: []*ai.Part{ai.NewTextPart("first draft")}}}
			case chainLength/2 + 10:
				arts = []*exp.Artifact{
					{Name: "draft.md", Parts: []*ai.Part{ai.NewTextPart("second draft")}},
					{Name: "plan.md", Parts: []*ai.Part{ai.NewTextPart("the plan")}},
				}
			}
			msgs = append(msgs, ai.NewUserTextMessage(fmt.Sprintf("question %d", i)), ai.NewModelTextMessage(fmt.Sprintf("answer %d", i)))
			st := cloneJSON(&exp.SessionState[State]{
				Messages:  msgs,
				Artifacts: arts,
				Usage:     &ai.GenerationUsage{InputTokens: 10 * (i + 1), OutputTokens: 5 * (i + 1)},
			})
			if i == chainLength/2+15 {
				pending := put(t, ctx, store, id, s.pendingRow("sess", parent, clk.now()))
				beat(t, ctx, store, id, clk.now())
				finalize(t, ctx, store, pending, exp.SnapshotStatusCompleted, exp.AgentFinishReasonStop, st, clk.now())
			} else {
				put(t, ctx, store, id, s.row("sess", parent, exp.SnapshotStatusCompleted, st, clk.now()))
			}
			ids = append(ids, id)
			want[id] = st
			parent = id
		}
		for _, id := range ids {
			checkState(t, id, mustGet(t, ctx, store, id).State, want[id])
		}
		last := ids[len(ids)-1]
		got := getLatest(t, ctx, store, "sess")
		if got == nil || got.SnapshotID != last {
			t.Fatalf("latest = %s, want %s", describe(got), last)
		}
		checkState(t, "latest", got.State, want[last])
	})

	t.Run("BranchesReadBackIndependently", func(t *testing.T) {
		// Regenerating a turn forks the chain: siblings share a parent and
		// each branch reads back its own history.
		store, clk := s.newStore(t), newClock()
		states := map[string]*exp.SessionState[State]{}
		add := func(id, parent, text string) {
			st := s.talk(text)
			if parent != "" {
				st = cloneJSON(states[parent])
				st.Messages = append(st.Messages, ai.NewModelTextMessage(text))
			}
			put(t, ctx, store, id, s.row("sess", parent, exp.SnapshotStatusCompleted, st, clk.now()))
			states[id] = st
		}
		add("root", "", "root")
		add("left", "root", "left")
		add("right", "root", "right")
		add("left-2", "left", "left again")
		add("right-2", "right", "right again")
		add("left-2b", "left", "left, regenerated")
		for id, st := range states {
			checkState(t, id, mustGet(t, ctx, store, id).State, st)
		}
		if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != "left-2b" {
			t.Errorf("latest = %s, want left-2b, the newest branch", describe(got))
		}
	})

	t.Run("LeafRewriteKeepsAncestors", func(t *testing.T) {
		// Rewriting the newest row of a branch (as a finalize does) changes
		// that row only, and a child written afterward builds on the new state.
		store, clk := s.newStore(t), newClock()
		root := s.talk("root")
		mid := s.talk("root", "mid")
		put(t, ctx, store, "root", s.row("sess", "", exp.SnapshotStatusCompleted, root, clk.now()))
		put(t, ctx, store, "mid", s.row("sess", "root", exp.SnapshotStatusCompleted, mid, clk.now()))
		leaf := put(t, ctx, store, "leaf", s.row("sess", "mid", exp.SnapshotStatusCompleted, s.talk("root", "mid", "leaf"), clk.now()))
		first := s.talk("root", "mid", "leaf, rewritten")
		rewrite(t, ctx, store, leaf, first, clk.now())
		second := s.talk("root", "mid", "leaf, rewritten twice", "with more")
		rewrite(t, ctx, store, leaf, second, clk.now())
		child := s.talk("root", "mid", "leaf, rewritten twice", "with more", "child")
		put(t, ctx, store, "child", s.row("sess", "leaf", exp.SnapshotStatusCompleted, child, clk.now()))

		checkState(t, "root", mustGet(t, ctx, store, "root").State, root)
		checkState(t, "mid", mustGet(t, ctx, store, "mid").State, mid)
		checkState(t, "leaf", mustGet(t, ctx, store, "leaf").State, second)
		checkState(t, "child", mustGet(t, ctx, store, "child").State, child)
	})
}

// --- Latest ---

func (s *suite[State]) testLatest(t *testing.T) {
	ctx := context.Background()

	t.Run("PicksMostRecentlyCreated", func(t *testing.T) {
		// IDs sort against write order, so neither ID order nor the tie-break
		// can pass this by luck. The latest row comes back in full.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "z", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("z"), clk.now()))
		put(t, ctx, store, "m", s.row("sess", "z", exp.SnapshotStatusCompleted, s.talk("m"), clk.now()))
		put(t, ctx, store, "a", s.row("sess", "m", exp.SnapshotStatusCompleted, s.talk("a"), clk.now()))
		got := getLatest(t, ctx, store, "sess")
		if got == nil || got.SnapshotID != "a" {
			t.Fatalf("latest = %s, want a", describe(got))
		}
		checkState(t, "latest", got.State, s.talk("a"))
	})

	t.Run("IgnoresRewritesOfOlderRows", func(t *testing.T) {
		// A rewrite keeps CreatedAt, so finalizing an older row does not move
		// it ahead of a newer one.
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "root", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("root"), clk.now()))
		pending := put(t, ctx, store, "b1", s.pendingRow("sess", "root", clk.now()))
		put(t, ctx, store, "b2", s.row("sess", "root", exp.SnapshotStatusCompleted, s.talk("b2"), clk.now()))
		finalize(t, ctx, store, pending, exp.SnapshotStatusCompleted, exp.AgentFinishReasonStop, s.talk("b1"), clk.now())
		if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != "b2" {
			t.Errorf("latest = %s, want b2 (the finalize of b1 must not move it ahead)", describe(got))
		}
	})

	t.Run("ReturnsAnyStatus", func(t *testing.T) {
		// Failed, aborted, and pending rows are returned like any other; the
		// caller decides what a dead-end or in-flight tip means.
		for _, st := range []exp.SnapshotStatus{exp.SnapshotStatusFailed, exp.SnapshotStatusAborted, exp.SnapshotStatusPending, exp.SnapshotStatusAborting} {
			t.Run(string(st), func(t *testing.T) {
				store, clk := s.newStore(t), newClock()
				put(t, ctx, store, "a", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("a"), clk.now()))
				tip := s.row("sess", "a", st, s.talk("a", "b"), clk.now())
				if st == exp.SnapshotStatusPending || st == exp.SnapshotStatusAborting {
					tip = s.pendingRow("sess", "a", tip.CreatedAt)
					tip.Status = st
				}
				put(t, ctx, store, "b", tip)
				if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != "b" || got.Status != st {
					t.Errorf("latest = %s, want b (%s)", describe(got), st)
				}
			})
		}
	})

	t.Run("BreaksCreatedAtTiesBySnapshotID", func(t *testing.T) {
		// Rows created at the same instant resolve to the greater snapshot ID,
		// whichever was written first.
		store, clk := s.newStore(t), newClock()
		at := clk.now()
		put(t, ctx, store, "b", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("b"), at))
		put(t, ctx, store, "a", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("a"), at))
		if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != "b" {
			t.Errorf("latest = %s, want b (equal CreatedAt resolves to the greater ID)", describe(got))
		}
		put(t, ctx, store, "c", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("c"), at))
		if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != "c" {
			t.Errorf("latest = %s, want c (equal CreatedAt resolves to the greater ID)", describe(got))
		}
	})

	t.Run("IgnoresBackdatedNewRow", func(t *testing.T) {
		// A new row created before the current latest does not replace it.
		store, clk := s.newStore(t), newClock()
		early, late := clk.now(), clk.now()
		put(t, ctx, store, "a", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("a"), late))
		put(t, ctx, store, "z", s.row("sess", "a", exp.SnapshotStatusCompleted, s.talk("z"), early))
		if got := getLatest(t, ctx, store, "sess"); got == nil || got.SnapshotID != "a" {
			t.Errorf("latest = %s, want a (a backdated row must not win)", describe(got))
		}
	})

	t.Run("KeepsSessionsApart", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "one-a", s.row("one", "", exp.SnapshotStatusCompleted, s.talk("one a"), clk.now()))
		put(t, ctx, store, "two-a", s.row("two", "", exp.SnapshotStatusCompleted, s.talk("two a"), clk.now()))
		put(t, ctx, store, "one-b", s.row("one", "one-a", exp.SnapshotStatusCompleted, s.talk("one b"), clk.now()))
		put(t, ctx, store, "two-b", s.row("two", "two-a", exp.SnapshotStatusCompleted, s.talk("two b"), clk.now()))
		for session, want := range map[string]string{"one": "one-b", "two": "two-b"} {
			if got := getLatest(t, ctx, store, session); got == nil || got.SnapshotID != want {
				t.Errorf("latest of %s = %s, want %s", session, describe(got), want)
			}
		}
	})

	t.Run("UnknownSessionIsNil", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		put(t, ctx, store, "a", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("a"), clk.now()))
		if got := getLatest(t, ctx, store, "other"); got != nil {
			t.Errorf("latest of an unknown session = %s, want nil", describe(got))
		}
	})

	t.Run("EmptySessionIDFails", func(t *testing.T) {
		store := s.newStore(t)
		if got, err := store.GetLatestSnapshot(ctx, ""); err == nil {
			t.Errorf("GetLatestSnapshot(\"\") = (%s, nil), want an error", describe(got))
		}
	})
}

// --- Metadata ---

func (s *suite[State]) testMetadata(t *testing.T) {
	ctx := context.Background()
	if _, ok := s.newStore(t).(exp.SnapshotMetadataReader[State]); !ok {
		t.Skip("the store does not implement exp.SnapshotMetadataReader")
	}
	newStore := func(t *testing.T) (exp.SessionStore[State], exp.SnapshotMetadataReader[State]) {
		store := s.newStore(t)
		return store, store.(exp.SnapshotMetadataReader[State])
	}
	// withoutState is the full row a metadata read must match.
	withoutState := func(snap *exp.SessionSnapshot[State]) *exp.SessionSnapshot[State] {
		meta := cloneJSON(snap)
		meta.State = nil
		return meta
	}

	t.Run("MatchesFullReadWithoutState", func(t *testing.T) {
		store, mr := newStore(t)
		clk := newClock()
		row := s.failedRow("sess", "", s.talk("question"), clk.now())
		beat := clk.now()
		row.HeartbeatAt = &beat
		put(t, ctx, store, "a", row)
		meta, err := mr.GetSnapshotMetadata(ctx, "a")
		if err != nil {
			t.Fatalf("GetSnapshotMetadata: %v", err)
		}
		checkRow(t, "metadata read", meta, withoutState(mustGet(t, ctx, store, "a")))
	})

	t.Run("LatestResolvesTheSameRow", func(t *testing.T) {
		store, mr := newStore(t)
		clk := newClock()
		put(t, ctx, store, "a", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("a"), clk.now()))
		put(t, ctx, store, "b", s.pendingRow("sess", "a", clk.now()))
		put(t, ctx, store, "other", s.row("sess-2", "", exp.SnapshotStatusCompleted, s.talk("other"), clk.now()))
		full := getLatest(t, ctx, store, "sess")
		if full == nil || full.SnapshotID != "b" {
			t.Fatalf("latest = %s, want b", describe(full))
		}
		meta, err := mr.GetLatestSnapshotMetadata(ctx, "sess")
		if err != nil {
			t.Fatalf("GetLatestSnapshotMetadata: %v", err)
		}
		checkRow(t, "latest metadata read", meta, withoutState(full))
	})

	t.Run("UnknownSnapshotIsNil", func(t *testing.T) {
		_, mr := newStore(t)
		if meta, err := mr.GetSnapshotMetadata(ctx, "missing"); err != nil || meta != nil {
			t.Errorf("GetSnapshotMetadata(missing) = (%s, %v), want (nil, nil)", describe(meta), err)
		}
	})

	t.Run("UnknownSessionIsNil", func(t *testing.T) {
		_, mr := newStore(t)
		if meta, err := mr.GetLatestSnapshotMetadata(ctx, "missing"); err != nil || meta != nil {
			t.Errorf("GetLatestSnapshotMetadata(missing) = (%s, %v), want (nil, nil)", describe(meta), err)
		}
	})

	t.Run("EmptySessionIDFails", func(t *testing.T) {
		_, mr := newStore(t)
		if meta, err := mr.GetLatestSnapshotMetadata(ctx, ""); err == nil {
			t.Errorf("GetLatestSnapshotMetadata(\"\") = (%s, nil), want an error", describe(meta))
		}
	})

	t.Run("DoesNotAliasTheStore", func(t *testing.T) {
		store, mr := newStore(t)
		clk := newClock()
		put(t, ctx, store, "a", s.failedRow("sess", "", s.talk("question"), clk.now()))
		want := withoutState(mustGet(t, ctx, store, "a"))
		reads := map[string]func() (*exp.SessionSnapshot[State], error){
			"GetSnapshotMetadata":       func() (*exp.SessionSnapshot[State], error) { return mr.GetSnapshotMetadata(ctx, "a") },
			"GetLatestSnapshotMetadata": func() (*exp.SessionSnapshot[State], error) { return mr.GetLatestSnapshotMetadata(ctx, "sess") },
		}
		for name, read := range reads {
			meta, err := read()
			if err != nil || meta == nil || meta.Error == nil {
				t.Fatalf("%s = (%s, %v), want the row with its error", name, describe(meta), err)
			}
			meta.Error.Message = "tampered"
			if meta.Error.Details != nil {
				meta.Error.Details["step"] = "tampered"
			}
			again, err := read()
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			checkRow(t, name+" after editing an earlier read", again, want)
		}
	})
}

// --- Subscriber ---

func (s *suite[State]) testSubscriber(t *testing.T) {
	ctx := context.Background()
	if _, ok := s.newStore(t).(exp.SnapshotSubscriber); !ok {
		t.Skip("the store does not implement exp.SnapshotSubscriber")
	}
	newStore := func(t *testing.T) (exp.SessionStore[State], exp.SnapshotSubscriber) {
		store := s.newStore(t)
		return store, store.(exp.SnapshotSubscriber)
	}
	subscribe := func(t *testing.T, sub exp.SnapshotSubscriber, id string) <-chan exp.SnapshotStatus {
		subCtx, cancel := context.WithCancel(ctx)
		t.Cleanup(cancel)
		return sub.OnSnapshotStatusChange(subCtx, id)
	}

	t.Run("YieldsCurrentStatusFirst", func(t *testing.T) {
		store, sub := newStore(t)
		clk := newClock()
		put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		put(t, ctx, store, "c", s.row("sess", "p", exp.SnapshotStatusCompleted, s.talk("done"), clk.now()))
		if got := nextStatus(t, subscribe(t, sub, "p")); got != exp.SnapshotStatusPending {
			t.Errorf("first status of a pending row = %q, want %q", got, exp.SnapshotStatusPending)
		}
		if got := nextStatus(t, subscribe(t, sub, "c")); got != exp.SnapshotStatusCompleted {
			t.Errorf("first status of a completed row = %q, want %q", got, exp.SnapshotStatusCompleted)
		}
	})

	t.Run("DeliversStatusChanges", func(t *testing.T) {
		store, sub := newStore(t)
		clk := newClock()
		pending := put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		ch := subscribe(t, sub, "p")
		if got := nextStatus(t, ch); got != exp.SnapshotStatusPending {
			t.Fatalf("first status = %q, want %q", got, exp.SnapshotStatusPending)
		}
		abort(t, ctx, store, "p", clk.now())
		awaitStatus(t, ch, exp.SnapshotStatusAborting)
		finalize(t, ctx, store, pending, exp.SnapshotStatusAborted, exp.AgentFinishReasonAborted, s.talk("partial"), clk.now())
		awaitStatus(t, ch, exp.SnapshotStatusAborted)
	})

	t.Run("DeliversInPlaceStatusChange", func(t *testing.T) {
		// A mutator may edit the row it is handed and return it; the status
		// change must still reach subscribers.
		store, sub := newStore(t)
		clk := newClock()
		put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		ch := subscribe(t, sub, "p")
		if got := nextStatus(t, ch); got != exp.SnapshotStatusPending {
			t.Fatalf("first status = %q, want %q", got, exp.SnapshotStatusPending)
		}
		if _, err := store.SaveSnapshot(ctx, "p", func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			if existing == nil {
				return nil, errors.New("row is missing")
			}
			existing.Status = exp.SnapshotStatusAborting
			return existing, nil
		}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		awaitStatus(t, ch, exp.SnapshotStatusAborting)
	})

	t.Run("FansOutToEverySubscriber", func(t *testing.T) {
		store, sub := newStore(t)
		clk := newClock()
		put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		first, second := subscribe(t, sub, "p"), subscribe(t, sub, "p")
		nextStatus(t, first)
		nextStatus(t, second)
		abort(t, ctx, store, "p", clk.now())
		awaitStatus(t, first, exp.SnapshotStatusAborting)
		awaitStatus(t, second, exp.SnapshotStatusAborting)
	})

	t.Run("MissingSnapshotClosesTheChannel", func(t *testing.T) {
		_, sub := newStore(t)
		awaitClose(t, subscribe(t, sub, "missing"), false)
	})

	t.Run("ClosesOnContextCancel", func(t *testing.T) {
		store, sub := newStore(t)
		clk := newClock()
		put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		subCtx, cancel := context.WithCancel(ctx)
		ch := sub.OnSnapshotStatusChange(subCtx, "p")
		nextStatus(t, ch)
		cancel()
		awaitClose(t, ch, true)
	})
}

// --- Second instance ---

func (s *suite[State]) testSecondInstance(t *testing.T) {
	ctx := context.Background()
	if s.opts.Reopen == nil {
		t.Skip("Options.Reopen is nil")
	}

	t.Run("ReadsRowsWrittenByTheOther", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		other := s.opts.Reopen(t, store)
		rows := map[string]*exp.SessionSnapshot[State]{
			"a": put(t, ctx, store, "a", s.row("sess", "", exp.SnapshotStatusCompleted, s.richState(), clk.now())),
			"b": put(t, ctx, store, "b", s.failedRow("sess", "a", s.talk("failed turn"), clk.now())),
			"c": put(t, ctx, store, "c", s.pendingRow("sess", "b", clk.now())),
		}
		rows["d"] = put(t, ctx, other, "d", s.row("sess", "c", exp.SnapshotStatusCompleted, s.talk("from the other"), clk.now()))
		for id, want := range rows {
			checkRow(t, id+" through the original", mustGet(t, ctx, store, id), want)
			checkRow(t, id+" through the reopened store", mustGet(t, ctx, other, id), want)
		}
		for name, st := range map[string]exp.SessionStore[State]{"original": store, "reopened": other} {
			if got := getLatest(t, ctx, st, "sess"); got == nil || got.SnapshotID != "d" {
				t.Errorf("latest through the %s store = %s, want d", name, describe(got))
			}
		}
	})

	t.Run("DeliversStatusChangesFromTheOther", func(t *testing.T) {
		store, clk := s.newStore(t), newClock()
		other := s.opts.Reopen(t, store)
		sub, ok := other.(exp.SnapshotSubscriber)
		if !ok {
			t.Skip("the store does not implement exp.SnapshotSubscriber")
		}
		pending := put(t, ctx, store, "p", s.pendingRow("sess", "", clk.now()))
		subCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		ch := sub.OnSnapshotStatusChange(subCtx, "p")
		if got := nextStatus(t, ch); got != exp.SnapshotStatusPending {
			t.Fatalf("first status = %q, want %q", got, exp.SnapshotStatusPending)
		}
		abort(t, ctx, store, "p", clk.now())
		awaitStatus(t, ch, exp.SnapshotStatusAborting)
		finalize(t, ctx, store, pending, exp.SnapshotStatusAborted, exp.AgentFinishReasonAborted, s.talk("partial"), clk.now())
		awaitStatus(t, ch, exp.SnapshotStatusAborted)
	})
}

// --- Scopes ---

func (s *suite[State]) testScopes(t *testing.T) {
	if s.opts.Scoped == nil {
		t.Skip("Options.Scoped is nil")
	}

	t.Run("IsolatesReads", func(t *testing.T) {
		store, a, b := s.opts.Scoped(t)
		clk := newClock()
		put(t, a, store, "shared", s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("a"), clk.now()))
		if got := get(t, b, store, "shared"); got != nil {
			t.Errorf("GetSnapshot in the other scope = %s, want nil", describe(got))
		}
		if got := getLatest(t, b, store, "sess"); got != nil {
			t.Errorf("GetLatestSnapshot in the other scope = %s, want nil", describe(got))
		}
		if mr, ok := store.(exp.SnapshotMetadataReader[State]); ok {
			if got, err := mr.GetSnapshotMetadata(b, "shared"); err != nil || got != nil {
				t.Errorf("GetSnapshotMetadata in the other scope = (%s, %v), want (nil, nil)", describe(got), err)
			}
			if got, err := mr.GetLatestSnapshotMetadata(b, "sess"); err != nil || got != nil {
				t.Errorf("GetLatestSnapshotMetadata in the other scope = (%s, %v), want (nil, nil)", describe(got), err)
			}
		}
		checkState(t, "row in its own scope", mustGet(t, a, store, "shared").State, s.talk("a"))
	})

	t.Run("IsolatesWrites", func(t *testing.T) {
		// Each scope has its own row under a shared ID: a write in one scope
		// neither sees nor changes the other's.
		store, a, b := s.opts.Scoped(t)
		clk := newClock()
		inA := put(t, a, store, "shared", s.pendingRow("sess", "", clk.now()))
		inB := s.row("sess", "", exp.SnapshotStatusCompleted, s.talk("b"), clk.now())
		var seen *exp.SessionSnapshot[State]
		if _, err := store.SaveSnapshot(b, "shared", func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
			seen = existing
			return cloneJSON(inB), nil
		}); err != nil {
			t.Fatalf("SaveSnapshot in the second scope: %v", err)
		}
		if seen != nil {
			t.Errorf("fn in the second scope got %s, want nil (the first scope's row)", describe(seen))
		}
		if got := abort(t, b, store, "shared", clk.now()); got != exp.SnapshotStatusCompleted {
			t.Errorf("abort in the second scope returned %q, want %q (its own row)", got, exp.SnapshotStatusCompleted)
		}
		checkRow(t, "first scope's row", mustGet(t, a, store, "shared"), inA)
		inB.SnapshotID = "shared"
		checkRow(t, "second scope's row", mustGet(t, b, store, "shared"), inB)
		for name, scope := range map[string]context.Context{"first": a, "second": b} {
			if got := getLatest(t, scope, store, "sess"); got == nil || got.SnapshotID != "shared" {
				t.Errorf("latest in the %s scope = %s, want its own row", name, describe(got))
			}
		}
	})

	t.Run("IsolatesSubscriptions", func(t *testing.T) {
		store, a, b := s.opts.Scoped(t)
		sub, ok := store.(exp.SnapshotSubscriber)
		if !ok {
			t.Skip("the store does not implement exp.SnapshotSubscriber")
		}
		put(t, a, store, "only-a", s.pendingRow("sess", "", newClock().now()))
		subCtx, cancel := context.WithCancel(b)
		defer cancel()
		awaitClose(t, sub.OnSnapshotStatusChange(subCtx, "only-a"), false)
	})
}

// --- Rows and states ---

// row returns a row with status st and both timestamps set to at.
func (s *suite[State]) row(sessionID, parentID string, st exp.SnapshotStatus, state *exp.SessionState[State], at time.Time) *exp.SessionSnapshot[State] {
	return &exp.SessionSnapshot[State]{
		SessionID: sessionID,
		ParentID:  parentID,
		Status:    st,
		State:     state,
		CreatedAt: at,
		UpdatedAt: at,
	}
}

// pendingRow returns the row a detach writes: pending, no state, and a first
// heartbeat.
func (s *suite[State]) pendingRow(sessionID, parentID string, at time.Time) *exp.SessionSnapshot[State] {
	snap := s.row(sessionID, parentID, exp.SnapshotStatusPending, nil, at)
	snap.HeartbeatAt = &at
	return snap
}

// failedRow returns a failed turn's row, with a finish reason and an error
// that carries details.
func (s *suite[State]) failedRow(sessionID, parentID string, state *exp.SessionState[State], at time.Time) *exp.SessionSnapshot[State] {
	snap := s.row(sessionID, parentID, exp.SnapshotStatusFailed, state, at)
	snap.FinishReason = exp.AgentFinishReasonFailed
	snap.Error = status.Errorf(status.ErrUnavailable, "model unavailable").WithDetails(map[string]any{"step": "generate", "attempt": 3})
	return snap
}

// talk returns a state whose messages carry texts, alternating user and model.
func (s *suite[State]) talk(texts ...string) *exp.SessionState[State] {
	msgs := make([]*ai.Message, len(texts))
	for i, text := range texts {
		if i%2 == 0 {
			msgs[i] = ai.NewUserTextMessage(text)
		} else {
			msgs[i] = ai.NewModelTextMessage(text)
		}
	}
	return &exp.SessionState[State]{Messages: msgs}
}

// richState returns a state that uses every kind of content a conversation can
// hold: each part type, nested metadata, artifacts, usage, and text a store
// can mangle (escapes, NUL, characters outside the Basic Multilingual Plane).
func (s *suite[State]) richState() *exp.SessionState[State] {
	tricky := "quotes \" and \\ backslash, <tag> & ampersand, tab\t, newline\n, NUL \x00, accents é, CJK 世界, emoji 🌍, astral 𝄞, RTL \u202e mark, BOM \ufeff"
	return &exp.SessionState[State]{
		Messages: []*ai.Message{
			ai.NewSystemTextMessage("Answer briefly."),
			{
				Role:    ai.RoleUser,
				Content: []*ai.Part{ai.NewTextPart(tricky)},
				Metadata: map[string]any{
					"source": "web",
					"tags":   []any{"a", "b"},
					"nested": map[string]any{"ok": true, "ratio": 1.5, "none": nil, "list": []any{1.0, "two", false}},
				},
			},
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewReasoningPart("thinking it over", []byte{0x00, 0x01, 0xfe, 0xff}),
					ai.NewTextPart("Let me look that up."),
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "lookup", Ref: "call-1", Input: map[string]any{"query": "weather", "days": 3.0, "units": []any{"c", "f"}, "strict": true, "extra": nil}}),
				},
			},
			{
				Role:    ai.RoleTool,
				Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{Name: "lookup", Ref: "call-1", Output: map[string]any{"temp": 21.5, "conditions": "sunny"}})},
			},
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="),
					ai.NewDataPart(map[string]any{"chart": []any{1.0, 2.0, 3.0}}),
					ai.NewCustomPart(map[string]any{"kind": "widget", "id": "w-1"}),
					ai.NewTextPart(""),
				},
			},
		},
		Artifacts: []*exp.Artifact{
			{Name: "notes.md", Parts: []*ai.Part{ai.NewTextPart("# Notes\n- one\n- two")}, Metadata: map[string]any{"version": 2.0}},
			{Name: "chart.png", Parts: []*ai.Part{ai.NewMediaPart("image/png", "https://example.com/chart.png")}},
		},
		Usage: &ai.GenerationUsage{
			InputTokens:         1200,
			OutputTokens:        340,
			ThoughtsTokens:      56,
			TotalTokens:         1596,
			CachedContentTokens: 800,
			Custom:              map[string]float64{"cost": 0.0123},
		},
	}
}

// largeState returns a state of about size bytes of JSON, in messages of 64 KiB
// of varied text.
func (s *suite[State]) largeState(size int) *exp.SessionState[State] {
	const chunk = 64 << 10
	const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-"
	seed := uint32(1)
	var msgs []*ai.Message
	for total := 0; total < size; total += chunk {
		b := make([]byte, chunk)
		for i := range b {
			seed = seed*1664525 + 1013904223
			b[i] = alphabet[int(seed>>24)%len(alphabet)]
		}
		if len(msgs)%2 == 0 {
			msgs = append(msgs, ai.NewUserTextMessage(string(b)))
		} else {
			msgs = append(msgs, ai.NewModelTextMessage(string(b)))
		}
	}
	return &exp.SessionState[State]{Messages: msgs}
}

// clock hands out increasing timestamps a millisecond apart, so the checks do
// not depend on the wall clock's resolution and fit any store that keeps at
// least millisecond precision. The zone is not UTC, so a store that drops the
// offset instead of converting the instant fails.
type clock struct{ next time.Time }

func newClock() *clock {
	return &clock{next: time.Date(2026, 3, 4, 5, 6, 7, 0, time.FixedZone("UTC+05:30", 5*3600+30*60))}
}

func (c *clock) now() time.Time {
	at := c.next
	c.next = c.next.Add(time.Millisecond)
	return at
}

// --- Store operations ---

// put saves snap at id and returns the row the store reports. fn hands the
// store a fresh copy on every call, since a store may call it more than once
// and may write into the value it returns.
func put[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string, snap *exp.SessionSnapshot[State]) *exp.SessionSnapshot[State] {
	t.Helper()
	saved, err := store.SaveSnapshot(ctx, id, func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		return cloneJSON(snap), nil
	})
	if err != nil {
		t.Fatalf("SaveSnapshot(%q): %v", id, err)
	}
	if saved == nil {
		t.Fatalf("SaveSnapshot(%q) returned nil for a write", id)
	}
	return saved
}

// rewrite replaces the state of an existing row the way a finalize does: it
// keeps CreatedAt and advances UpdatedAt.
func rewrite[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], row *exp.SessionSnapshot[State], state *exp.SessionState[State], at time.Time) {
	t.Helper()
	next := cloneJSON(row)
	next.State = state
	next.UpdatedAt = at
	put(t, ctx, store, row.SnapshotID, next)
}

// get reads a row by ID, failing t on an error. It returns nil for a miss.
func get[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string) *exp.SessionSnapshot[State] {
	t.Helper()
	snap, err := store.GetSnapshot(ctx, id)
	if err != nil {
		t.Fatalf("GetSnapshot(%q): %v", id, err)
	}
	return snap
}

// mustGet reads a row by ID, failing t on an error or a miss.
func mustGet[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string) *exp.SessionSnapshot[State] {
	t.Helper()
	snap := get(t, ctx, store, id)
	if snap == nil {
		t.Fatalf("GetSnapshot(%q) = nil, want the row", id)
	}
	return snap
}

// getLatest reads a session's latest row, failing t on an error. It returns
// nil for a miss.
func getLatest[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], sessionID string) *exp.SessionSnapshot[State] {
	t.Helper()
	snap, err := store.GetLatestSnapshot(ctx, sessionID)
	if err != nil {
		t.Fatalf("GetLatestSnapshot(%q): %v", sessionID, err)
	}
	return snap
}

// beat refreshes a row's heartbeat with the runtime's mutator: a pending or
// aborting row is carried through unchanged but for HeartbeatAt, and a settled
// row is left alone.
func beat[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string, at time.Time) {
	t.Helper()
	if _, err := store.SaveSnapshot(ctx, id, func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		if existing == nil || existing.Status.Terminal() {
			return nil, nil
		}
		updated := *existing
		updated.HeartbeatAt = &at
		return &updated, nil
	}); err != nil {
		t.Fatalf("heartbeat of %q: %v", id, err)
	}
}

// abort flips a pending row to aborting with the runtime's mutator, and
// returns the row's status afterward, or "" when the row does not exist. A
// row that is not pending is rewritten verbatim.
func abort[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string, at time.Time) exp.SnapshotStatus {
	t.Helper()
	saved, err := store.SaveSnapshot(ctx, id, func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		if existing == nil {
			return nil, nil
		}
		if existing.Status != exp.SnapshotStatusPending {
			return existing, nil
		}
		updated := *existing
		updated.Status = exp.SnapshotStatusAborting
		updated.UpdatedAt = at
		return &updated, nil
	})
	if err != nil {
		t.Fatalf("abort of %q: %v", id, err)
	}
	if saved == nil {
		return ""
	}
	return saved.Status
}

// finalize lands a detached run's outcome on its pending row with the
// runtime's mutator: a settled row is left alone, and anything else is
// replaced by a row built from the pending one, with no heartbeat.
func finalize[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], pending *exp.SessionSnapshot[State], st exp.SnapshotStatus, reason exp.AgentFinishReason, state *exp.SessionState[State], at time.Time) {
	t.Helper()
	if _, err := store.SaveSnapshot(ctx, pending.SnapshotID, func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		if existing != nil && existing.Status.Terminal() {
			return nil, nil
		}
		return &exp.SessionSnapshot[State]{
			SessionID:    pending.SessionID,
			ParentID:     pending.ParentID,
			Status:       st,
			FinishReason: reason,
			State:        cloneJSON(state),
			CreatedAt:    pending.CreatedAt,
			UpdatedAt:    at,
		}, nil
	}); err != nil {
		t.Fatalf("finalize of %q: %v", pending.SnapshotID, err)
	}
}

// retry calls save until it succeeds, up to writeAttempts times, and returns
// the last error.
func retry(save func() error) error {
	var err error
	for range writeAttempts {
		if err = save(); err == nil {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return err
}

// --- Subscriptions ---

// nextStatus returns the next status ch yields, failing t if ch closes or
// stays silent past eventTimeout.
func nextStatus(t *testing.T, ch <-chan exp.SnapshotStatus) exp.SnapshotStatus {
	t.Helper()
	select {
	case st, ok := <-ch:
		if !ok {
			t.Fatal("the subscription closed, want a status")
		}
		return st
	case <-time.After(eventTimeout):
		t.Fatalf("no status within %v", eventTimeout)
		return ""
	}
}

// awaitStatus reads ch until it yields want. Subscriptions are level-triggered:
// a store may skip a status that was overwritten before it was read, or repeat
// one, so values before want are passed over.
func awaitStatus(t *testing.T, ch <-chan exp.SnapshotStatus, want exp.SnapshotStatus) {
	t.Helper()
	var seen []exp.SnapshotStatus
	timeout := time.After(eventTimeout)
	for {
		select {
		case st, ok := <-ch:
			if !ok {
				t.Fatalf("the subscription closed before %q (saw %v)", want, seen)
			}
			if st == want {
				return
			}
			seen = append(seen, st)
		case <-timeout:
			t.Fatalf("no %q within %v (saw %v)", want, eventTimeout, seen)
		}
	}
}

// awaitClose reads ch until it closes, failing t past eventTimeout. Unless
// allowValues is set, a value before the close fails t too.
func awaitClose(t *testing.T, ch <-chan exp.SnapshotStatus, allowValues bool) {
	t.Helper()
	timeout := time.After(eventTimeout)
	for {
		select {
		case st, ok := <-ch:
			if !ok {
				return
			}
			if !allowValues {
				t.Fatalf("the subscription yielded %q, want it closed without a value", st)
			}
		case <-timeout:
			t.Fatalf("the subscription did not close within %v", eventTimeout)
		}
	}
}

// --- Comparisons ---

// checkRow fails t unless got carries want's fields: identity, lineage, and
// lifecycle exactly, timestamps as instants, the error by its wire form, and
// the state as JSON.
func checkRow[State any](t *testing.T, label string, got, want *exp.SessionSnapshot[State]) {
	t.Helper()
	if got == nil {
		t.Errorf("%s: row is nil, want %s", label, describe(want))
		return
	}
	if got.SnapshotID != want.SnapshotID || got.SessionID != want.SessionID || got.ParentID != want.ParentID ||
		got.Status != want.Status || got.FinishReason != want.FinishReason {
		t.Errorf("%s: got %s, want %s", label, describe(got), describe(want))
	}
	if !got.CreatedAt.Equal(want.CreatedAt) {
		t.Errorf("%s: CreatedAt = %v, want %v", label, got.CreatedAt, want.CreatedAt)
	}
	if !got.UpdatedAt.Equal(want.UpdatedAt) {
		t.Errorf("%s: UpdatedAt = %v, want %v", label, got.UpdatedAt, want.UpdatedAt)
	}
	switch {
	case got.HeartbeatAt == nil && want.HeartbeatAt == nil:
	case got.HeartbeatAt == nil || want.HeartbeatAt == nil || !got.HeartbeatAt.Equal(*want.HeartbeatAt):
		t.Errorf("%s: HeartbeatAt = %v, want %v", label, timeOrNil(got.HeartbeatAt), timeOrNil(want.HeartbeatAt))
	}
	if ok, diff := sameJSON(want.Error, got.Error); !ok {
		t.Errorf("%s: Error differs from the one written (JSON Patch from want to got): %s", label, diff)
	}
	checkState(t, label, got.State, want.State)
}

// checkState fails t unless got and want encode to the same JSON, nil included.
func checkState[State any](t *testing.T, label string, got, want *exp.SessionState[State]) {
	t.Helper()
	if (got == nil) != (want == nil) {
		t.Errorf("%s: state is %v, want %v", label, stateOrNil(got), stateOrNil(want))
		return
	}
	if ok, diff := sameJSON(want, got); !ok {
		t.Errorf("%s: state differs from the one written (JSON Patch from want to got): %s", label, diff)
	}
}

// sameJSON reports whether want and got encode to the same JSON. When they do
// not, it describes the difference as a JSON Patch from want to got.
func sameJSON(want, got any) (bool, string) {
	wantJSON, err := json.Marshal(want)
	if err != nil {
		return false, fmt.Sprintf("marshal want: %v", err)
	}
	gotJSON, err := json.Marshal(got)
	if err != nil {
		return false, fmt.Sprintf("marshal got: %v", err)
	}
	if bytes.Equal(wantJSON, gotJSON) {
		return true, ""
	}
	var w, g any
	if err := json.Unmarshal(wantJSON, &w); err != nil {
		return false, fmt.Sprintf("unmarshal want: %v", err)
	}
	if err := json.Unmarshal(gotJSON, &g); err != nil {
		return false, fmt.Sprintf("unmarshal got: %v", err)
	}
	patch, err := json.Marshal(exp.Diff(w, g))
	if err != nil {
		return false, fmt.Sprintf("marshal diff: %v", err)
	}
	return false, truncate(string(patch), 2000)
}

// describe renders a row's identity and lifecycle for a failure message.
func describe[State any](snap *exp.SessionSnapshot[State]) string {
	if snap == nil {
		return "<nil>"
	}
	return fmt.Sprintf("{id=%q session=%q parent=%q status=%q finish=%q}", snap.SnapshotID, snap.SessionID, snap.ParentID, snap.Status, snap.FinishReason)
}

func timeOrNil(at *time.Time) any {
	if at == nil {
		return "<nil>"
	}
	return *at
}

func stateOrNil[State any](st *exp.SessionState[State]) string {
	if st == nil {
		return "nil"
	}
	return "non-nil"
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "... (" + fmt.Sprint(len(s)-n) + " more bytes)"
}

// cloneJSON deep-copies v through JSON, the way stores copy rows. It returns
// nil for nil.
func cloneJSON[T any](v *T) *T {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("sessionstoretest: clone: %v", err))
	}
	var out T
	if err := json.Unmarshal(b, &out); err != nil {
		panic(fmt.Sprintf("sessionstoretest: clone: %v", err))
	}
	return &out
}
