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
	"testing"

	"github.com/firebase/genkit/go/ai/exp"
)

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
		checkRow(t, "latest in the first scope", getLatest(t, a, store, "sess"), inA)
		checkRow(t, "latest in the second scope", getLatest(t, b, store, "sess"), inB)
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
