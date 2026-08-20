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

package exp

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// TestNewSessionContextPublishesPromptState verifies that attaching a session to
// a context also exposes its custom state through internal/base, which is how
// ai.prompt injects {{@state}} into templates without importing this package.
func TestNewSessionContextPublishesPromptState(t *testing.T) {
	s := &Session[map[string]any]{
		state: SessionState[map[string]any]{
			Custom: map[string]any{
				"name":        "Alice",
				"preferences": map[string]any{"theme": "dark"},
			},
		},
	}

	ctx := NewSessionContext(context.Background(), s)

	got := base.PromptStateFromContext(ctx)
	want := map[string]any{
		"name":        "Alice",
		"preferences": map[string]any{"theme": "dark"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("PromptStateFromContext() = %#v, want %#v", got, want)
	}
}

// TestPromptStateReflectsLatestCustom verifies the published state getter is
// evaluated lazily, so a template rendered later sees custom-state mutations
// made after the context was built.
func TestPromptStateReflectsLatestCustom(t *testing.T) {
	s := &Session[map[string]any]{
		state: SessionState[map[string]any]{Custom: map[string]any{"n": float64(1)}},
	}
	ctx := NewSessionContext(context.Background(), s)

	s.UpdateCustom(func(map[string]any) map[string]any {
		return map[string]any{"n": float64(2)}
	})

	got := base.PromptStateFromContext(ctx)
	want := map[string]any{"n": float64(2)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("PromptStateFromContext() = %#v, want %#v", got, want)
	}
}

// TestPromptStateNilWithoutSession verifies that no state is published when no
// session is attached to the context.
func TestPromptStateNilWithoutSession(t *testing.T) {
	if got := base.PromptStateFromContext(context.Background()); got != nil {
		t.Errorf("PromptStateFromContext() = %#v, want nil", got)
	}
}

func TestSnapshotStatus_Terminal(t *testing.T) {
	cases := []struct {
		status SnapshotStatus
		want   bool
	}{
		{SnapshotStatusPending, false},
		{SnapshotStatusCompleted, true},
		{SnapshotStatusAborted, true},
		{SnapshotStatusFailed, true},
		{SnapshotStatusExpired, true},
		// Empty counts as completed, matching the documented default.
		{SnapshotStatus(""), true},
	}
	for _, tc := range cases {
		if got := tc.status.Terminal(); got != tc.want {
			t.Errorf("SnapshotStatus(%q).Terminal() = %v, want %v", tc.status, got, tc.want)
		}
	}
}

func TestSessionState_LastModelMessage(t *testing.T) {
	toolOnly := &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "search"}),
	}}

	cases := []struct {
		name     string
		messages []*ai.Message
		want     string // "" means nil expected
	}{
		{name: "empty history"},
		{name: "no model messages", messages: []*ai.Message{ai.NewUserTextMessage("hi")}},
		{
			name: "latest text-bearing model message wins",
			messages: []*ai.Message{
				ai.NewUserTextMessage("q1"),
				ai.NewModelTextMessage("a1"),
				ai.NewUserTextMessage("q2"),
				ai.NewModelTextMessage("a2"),
			},
			want: "a2",
		},
		{
			name: "tool-request-only tip is skipped",
			messages: []*ai.Message{
				ai.NewModelTextMessage("spoken answer"),
				ai.NewUserTextMessage("follow-up"),
				toolOnly,
			},
			want: "spoken answer",
		},
		{
			name:     "only tool requests",
			messages: []*ai.Message{toolOnly},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			state := &SessionState[any]{Messages: tc.messages}
			got := state.LastModelMessage()
			if tc.want == "" {
				if got != nil {
					t.Fatalf("LastModelMessage() = %+v, want nil", got)
				}
				return
			}
			if got == nil || got.Text() != tc.want {
				t.Fatalf("LastModelMessage().Text() = %v, want %q", got, tc.want)
			}
		})
	}

	// A nil receiver (e.g. a pending snapshot's nil state) is tolerated.
	var nilState *SessionState[any]
	if got := nilState.LastModelMessage(); got != nil {
		t.Fatalf("nil receiver LastModelMessage() = %+v, want nil", got)
	}
}

// --- waitForSnapshot ---

// unsubscribableStore is a [SessionStore] that deliberately does not implement
// [SnapshotSubscriber], so a wait on it falls back to re-reading. It forwards
// to testInMemStore rather than embedding it, because embedding would promote
// the subscription method and defeat the point.
type unsubscribableStore[State any] struct{ inner *testInMemStore[State] }

func (s unsubscribableStore[State]) GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error) {
	return s.inner.GetSnapshot(ctx, snapshotID)
}

func (s unsubscribableStore[State]) GetLatestSnapshot(ctx context.Context, sessionID string) (*SessionSnapshot[State], error) {
	return s.inner.GetLatestSnapshot(ctx, sessionID)
}

func (s unsubscribableStore[State]) SaveSnapshot(
	ctx context.Context,
	snapshotID string,
	fn func(*SessionSnapshot[State]) (*SessionSnapshot[State], error),
) (*SessionSnapshot[State], error) {
	return s.inner.SaveSnapshot(ctx, snapshotID, fn)
}

// putSnapshot writes one row verbatim, so a test can stage a snapshot in any
// lifecycle state without running an agent.
func putSnapshot[State any](t *testing.T, store SessionStore[State], snap *SessionSnapshot[State]) {
	t.Helper()
	if _, err := store.SaveSnapshot(context.Background(), snap.SnapshotID,
		func(*SessionSnapshot[State]) (*SessionSnapshot[State], error) { return snap, nil }); err != nil {
		t.Fatalf("SaveSnapshot(%q): %v", snap.SnapshotID, err)
	}
}

// settleSnapshot flips a staged row to a terminal status, as a detached turn's
// finalize does.
func settleSnapshot[State any](t *testing.T, store SessionStore[State], snapshotID string, snapStatus SnapshotStatus) {
	t.Helper()
	if _, err := store.SaveSnapshot(context.Background(), snapshotID,
		func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
			settled := *existing
			settled.Status = snapStatus
			return &settled, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot(%q, %q): %v", snapshotID, snapStatus, err)
	}
}

func TestWaitSnapshot_TerminalReturnsWithoutWaiting(t *testing.T) {
	store := newTestInMemStore[any]()
	putSnapshot(t, store, &SessionSnapshot[any]{
		SnapshotID: "done", SessionID: "s1", Status: SnapshotStatusCompleted,
		State: &SessionState[any]{Messages: []*ai.Message{ai.NewModelTextMessage("finished")}},
	})

	// No deadline: a wait that did not return at once would hang the test.
	got, err := waitSnapshot(context.Background(), store, nil, "waitForSnapshot", "done", "")
	if err != nil {
		t.Fatalf("waitSnapshot: %v", err)
	}
	if got.Status != SnapshotStatusCompleted {
		t.Fatalf("status = %q, want %q", got.Status, SnapshotStatusCompleted)
	}
	if text := got.State.LastModelMessage().Text(); text != "finished" {
		t.Errorf("last model message = %q, want %q", text, "finished")
	}
}

func TestWaitSnapshot_SubscriptionDeliversSettlement(t *testing.T) {
	store := newTestInMemStore[any]()
	beat := time.Now()
	putSnapshot(t, store, &SessionSnapshot[any]{
		SnapshotID: "running", SessionID: "s1", Status: SnapshotStatusPending, HeartbeatAt: &beat,
	})

	// The liveness re-read is left at its production cadence, so a settlement
	// observed inside the test's timeout can only have arrived by subscription.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	done := make(chan *SessionSnapshot[any], 1)
	go func() {
		snap, err := waitSnapshot(ctx, store, nil, "waitForSnapshot", "running", "")
		if err != nil {
			t.Errorf("waitSnapshot: %v", err)
			close(done)
			return
		}
		done <- snap
	}()

	settleSnapshot(t, store, "running", SnapshotStatusCompleted)
	select {
	case snap := <-done:
		if snap == nil {
			t.Fatal("wait failed")
		}
		if snap.Status != SnapshotStatusCompleted {
			t.Fatalf("status = %q, want %q", snap.Status, SnapshotStatusCompleted)
		}
	case <-ctx.Done():
		t.Fatal("wait did not observe the settlement")
	}
}

func TestWaitSnapshot_RereadsWithoutSubscriber(t *testing.T) {
	restore := snapshotWaitPollInterval
	snapshotWaitPollInterval = 10 * time.Millisecond
	t.Cleanup(func() { snapshotWaitPollInterval = restore })

	inner := newTestInMemStore[any]()
	store := unsubscribableStore[any]{inner: inner}
	beat := time.Now()
	putSnapshot[any](t, store, &SessionSnapshot[any]{
		SnapshotID: "running", SessionID: "s1", Status: SnapshotStatusPending, HeartbeatAt: &beat,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	go func() {
		time.Sleep(30 * time.Millisecond)
		settleSnapshot[any](t, store, "running", SnapshotStatusFailed)
	}()

	got, err := waitSnapshot[any](ctx, store, nil, "waitForSnapshot", "running", "")
	if err != nil {
		t.Fatalf("waitSnapshot: %v", err)
	}
	if got.Status != SnapshotStatusFailed {
		t.Fatalf("status = %q, want %q", got.Status, SnapshotStatusFailed)
	}
}

func TestWaitSnapshot_ExpiredHeartbeatEndsTheWait(t *testing.T) {
	restore := snapshotWaitLivenessInterval
	snapshotWaitLivenessInterval = 10 * time.Millisecond
	t.Cleanup(func() { snapshotWaitLivenessInterval = restore })

	store := newTestInMemStore[any]()
	// A worker that died a heartbeat timeout ago: the row stays pending, so no
	// subscription can report it and only the liveness re-read notices.
	stale := time.Now().Add(-2 * defaultHeartbeatTimeout)
	putSnapshot(t, store, &SessionSnapshot[any]{
		SnapshotID: "orphan", SessionID: "s1", Status: SnapshotStatusPending, HeartbeatAt: &stale,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	got, err := waitSnapshot(ctx, store, nil, "waitForSnapshot", "orphan", "")
	if err != nil {
		t.Fatalf("waitSnapshot: %v", err)
	}
	if got.Status != SnapshotStatusExpired {
		t.Fatalf("status = %q, want %q", got.Status, SnapshotStatusExpired)
	}
}

func TestWaitSnapshot_ContextEndsTheWait(t *testing.T) {
	store := newTestInMemStore[any]()
	beat := time.Now()
	putSnapshot(t, store, &SessionSnapshot[any]{
		SnapshotID: "running", SessionID: "s1", Status: SnapshotStatusPending, HeartbeatAt: &beat,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := waitSnapshot(ctx, store, nil, "waitForSnapshot", "running", ""); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("waitSnapshot error = %v, want DeadlineExceeded", err)
	}
}

func TestWaitSnapshot_UnknownSnapshot(t *testing.T) {
	store := newTestInMemStore[any]()
	_, err := waitSnapshot(context.Background(), store, nil, "waitForSnapshot", "nope", "")
	if !errors.Is(err, ErrSnapshotNotFound) {
		t.Fatalf("waitSnapshot error = %v, want ErrSnapshotNotFound", err)
	}
	if got, want := err.Error(), "waitForSnapshot: "; len(got) < len(want) || got[:len(want)] != want {
		t.Errorf("error = %q, want it to name the operation that failed", got)
	}
}

// scriptedStore answers each GetSnapshot from a fixed script, so a test can
// stage read failures inside a wait deterministically: read n gets entry n-1,
// and the last entry repeats. It deliberately implements no [SnapshotSubscriber],
// and the other store methods are never reached by a wait on a snapshot ID.
type scriptedStore struct {
	mu     sync.Mutex
	reads  int
	script []scriptedRead
}

type scriptedRead struct {
	snap *SessionSnapshot[any]
	err  error
}

func (s *scriptedStore) GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[any], error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reads++
	entry := s.script[min(s.reads, len(s.script))-1]
	return entry.snap, entry.err
}

func (s *scriptedStore) GetLatestSnapshot(ctx context.Context, sessionID string) (*SessionSnapshot[any], error) {
	return nil, errors.New("scriptedStore: GetLatestSnapshot is not scripted")
}

func (s *scriptedStore) SaveSnapshot(ctx context.Context, snapshotID string, fn func(*SessionSnapshot[any]) (*SessionSnapshot[any], error)) (*SessionSnapshot[any], error) {
	return nil, errors.New("scriptedStore: SaveSnapshot is not scripted")
}

func (s *scriptedStore) readCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reads
}

// scriptedPending is a pending row with a live heartbeat, so a scripted read
// reports it as genuinely pending rather than expired.
func scriptedPending() *SessionSnapshot[any] {
	beat := time.Now()
	return &SessionSnapshot[any]{SnapshotID: "job", SessionID: "s1", Status: SnapshotStatusPending, HeartbeatAt: &beat}
}

func TestWaitSnapshot_TransientReadFailuresAreRetried(t *testing.T) {
	restore := snapshotWaitPollInterval
	snapshotWaitPollInterval = 10 * time.Millisecond
	t.Cleanup(func() { snapshotWaitPollInterval = restore })

	blip := errors.New("store blip")
	store := &scriptedStore{script: []scriptedRead{
		{snap: scriptedPending()}, // the wait's initial read
		{err: blip},               // two consecutive in-wait blips, both
		{err: blip},               // within the retry budget
		{snap: &SessionSnapshot[any]{SnapshotID: "job", SessionID: "s1", Status: SnapshotStatusCompleted}},
	}}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	got, err := waitSnapshot[any](ctx, store, nil, "waitForSnapshot", "job", "")
	if err != nil {
		t.Fatalf("waitSnapshot: %v", err)
	}
	if got.Status != SnapshotStatusCompleted {
		t.Fatalf("status = %q, want %q", got.Status, SnapshotStatusCompleted)
	}
	if got := store.readCount(); got != 4 {
		t.Errorf("reads = %d, want 4 (initial, two retried blips, settled)", got)
	}
}

func TestWaitSnapshot_PersistentReadFailureSurfaces(t *testing.T) {
	restore := snapshotWaitPollInterval
	snapshotWaitPollInterval = 10 * time.Millisecond
	t.Cleanup(func() { snapshotWaitPollInterval = restore })

	down := errors.New("store down")
	store := &scriptedStore{script: []scriptedRead{{snap: scriptedPending()}, {err: down}}}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := waitSnapshot[any](ctx, store, nil, "waitForSnapshot", "job", "")
	if !errors.Is(err, down) {
		t.Fatalf("waitSnapshot error = %v, want the store's own error", err)
	}
	// The initial read, the retried failures, and the one that surfaced.
	if got, want := store.readCount(), 2+snapshotWaitReadRetries; got != want {
		t.Errorf("reads = %d, want %d (retry budget exhausted)", got, want)
	}
}

func TestWaitSnapshot_DeadEndReadFailureFailsFast(t *testing.T) {
	restore := snapshotWaitPollInterval
	snapshotWaitPollInterval = 10 * time.Millisecond
	t.Cleanup(func() { snapshotWaitPollInterval = restore })

	// The row disappears mid-wait: the re-read reports NOT_FOUND, which no
	// retry can help, so it surfaces at once instead of burning the budget.
	store := &scriptedStore{script: []scriptedRead{{snap: scriptedPending()}, {}}}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := waitSnapshot[any](ctx, store, nil, "waitForSnapshot", "job", "")
	if !errors.Is(err, ErrSnapshotNotFound) {
		t.Fatalf("waitSnapshot error = %v, want ErrSnapshotNotFound", err)
	}
	if got := store.readCount(); got != 2 {
		t.Errorf("reads = %d, want 2 (dead ends are not retried)", got)
	}
}

func TestNewSnapshotActions_WaitAction(t *testing.T) {
	store := newTestInMemStore[any]()
	putSnapshot(t, store, &SessionSnapshot[any]{
		SnapshotID: "done", SessionID: "s1", Status: SnapshotStatusCompleted,
	})
	_, wait, _ := newSnapshotActions[any]("waiter", store, nil)
	if wait == nil {
		t.Fatal("newSnapshotActions returned no wait action for a store-backed agent")
	}
	if got := wait.Desc().Type; got != api.ActionTypeAgentWait {
		t.Errorf("wait action type = %q, want %q", got, api.ActionTypeAgentWait)
	}

	// A session ID alone resolves whichever row is latest at resolution time,
	// which is a race with the session's next turn, so the wait requires the
	// snapshot ID that a read would accept on its own.
	_, err := wait.RunJSON(context.Background(), json.RawMessage(`{"sessionId":"s1"}`), nil)
	if !errors.Is(err, status.ErrInvalidArgument) {
		t.Fatalf("wait by session ID error = %v, want INVALID_ARGUMENT", err)
	}

	raw, err := wait.RunJSON(context.Background(), json.RawMessage(`{"snapshotId":"done"}`), nil)
	if err != nil {
		t.Fatalf("wait action: %v", err)
	}
	var snap SessionSnapshot[any]
	if err := json.Unmarshal(raw, &snap); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}
	if snap.Status != SnapshotStatusCompleted {
		t.Errorf("status = %q, want %q", snap.Status, SnapshotStatusCompleted)
	}

	// A client-managed agent keeps no snapshots, so it gets no wait action.
	if _, clientWait, _ := newSnapshotActions[any]("clientManaged", nil, nil); clientWait != nil {
		t.Error("newSnapshotActions returned a wait action for a store-less agent")
	}
}
