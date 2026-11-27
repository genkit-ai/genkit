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
	"fmt"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp"
)

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
