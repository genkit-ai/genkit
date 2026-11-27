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

func (s *suite[State]) testMetadata(t *testing.T) {
	ctx := context.Background()
	if !s.metadata {
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
	if !s.subscriber {
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
