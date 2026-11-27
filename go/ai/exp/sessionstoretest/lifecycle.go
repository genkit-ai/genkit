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
