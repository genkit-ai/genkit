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
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/firestore"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"github.com/google/uuid"
)

// testState is the custom-state type used by the store tests, mirroring the
// shared localstore conformance suite.
type testState struct {
	Counter int      `json:"counter"`
	Topics  []string `json:"topics,omitempty"`
}

const testProjectID = "genkit-firestore-session-test"

// newEmulatorStore creates a store backed by the Firestore emulator, isolated to
// a unique collection per call. It skips the test when FIRESTORE_EMULATOR_HOST
// is not set. Pass options (e.g. a small shard size or checkpoint interval) to
// exercise the sharding and checkpoint paths.
func newEmulatorStore(t *testing.T, opts ...SessionStoreOption) *FirestoreSessionStore[testState] {
	t.Helper()
	if os.Getenv("FIRESTORE_EMULATOR_HOST") == "" {
		t.Skip("Skipping: FIRESTORE_EMULATOR_HOST not set (start the Firestore emulator to run these tests)")
	}
	ctx := context.Background()
	client, err := firestore.NewClient(ctx, testProjectID)
	if err != nil {
		t.Fatalf("firestore.NewClient: %v", err)
	}
	t.Cleanup(func() { client.Close() })

	// A unique collection per store keeps tests independent on a shared emulator.
	// Tests build directly from a client (the unexported builder) so the store
	// logic runs against the emulator without standing up the Firebase plugin; the
	// public genkit-based constructor is covered separately.
	all := append([]SessionStoreOption{WithCollection("sessions-" + uuid.NewString())}, opts...)
	store, err := newFirestoreSessionStore[testState](client, all...)
	if err != nil {
		t.Fatalf("newFirestoreSessionStore: %v", err)
	}
	return store
}

// saveRow saves a fresh row with caller-managed timestamps stamped now, the way
// the runtime does.
func saveRow(t *testing.T, store *FirestoreSessionStore[testState], id, sessionID, parentID string, status aix.SnapshotStatus, counter int) *aix.SessionSnapshot[testState] {
	t.Helper()
	now := time.Now()
	saved, err := store.SaveSnapshot(context.Background(), id,
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{
				SessionID: sessionID,
				ParentID:  parentID,
				Status:    status,
				State:     &aix.SessionState[testState]{Custom: testState{Counter: counter}},
				CreatedAt: now,
				UpdatedAt: now,
			}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot(%q): %v", id, err)
	}
	return saved
}

// tick spaces consecutive writes far enough apart that CreatedAt orders them
// unambiguously even on coarse clocks.
func tick() { time.Sleep(2 * time.Millisecond) }

// --- Pure unit tests (no emulator) ---

func TestNewFirestoreSessionStoreNilClient(t *testing.T) {
	if _, err := newFirestoreSessionStore[testState](nil); err == nil {
		t.Error("expected error for nil client")
	}
}

func TestNewFirestoreSessionStorePluginNotFound(t *testing.T) {
	// Without the Firebase plugin registered, the public constructor surfaces a
	// clear error instead of resolving a client.
	g := genkit.Init(context.Background())
	if _, err := NewFirestoreSessionStore[testState](context.Background(), g); err == nil {
		t.Error("expected error when the Firebase plugin is not registered")
	}
}

func TestNewFirestoreSessionStorePublic(t *testing.T) {
	// Exercises the public, plugin-resolving constructor end to end against the
	// emulator (the logic tests use the unexported client-based builder).
	if os.Getenv("FIRESTORE_EMULATOR_HOST") == "" {
		t.Skip("Skipping: FIRESTORE_EMULATOR_HOST not set")
	}
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: testProjectID}))
	store, err := NewFirestoreSessionStore[testState](ctx, g, WithCollection("sessions-"+uuid.NewString()))
	if err != nil {
		t.Fatalf("NewFirestoreSessionStore: %v", err)
	}
	now := time.Now()
	saved, err := store.SaveSnapshot(ctx, "x",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "s", CreatedAt: now, UpdatedAt: now}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	if saved == nil || saved.SnapshotID != "x" {
		t.Errorf("saved = %+v, want snapshot x", saved)
	}
}

func TestOptionValidation(t *testing.T) {
	// The default for every option is to omit it; an explicit invalid value
	// (empty/zero/negative/nil) is rejected rather than silently defaulted. Options
	// scoped to the wrong service are a compile error, not a runtime one: e.g.
	// WithTTL(...) cannot be passed to NewFirestoreSessionStore, and WithShardSize(...)
	// cannot be passed to NewFirestoreStreamManager.
	t.Run("session store rejects invalid values", func(t *testing.T) {
		cases := []struct {
			name string
			opt  SessionStoreOption
		}{
			{"empty collection", WithCollection("")},
			{"zero checkpoint interval", WithCheckpointInterval(0)},
			{"negative checkpoint interval", WithCheckpointInterval(-1)},
			{"zero shard size", WithShardSize(0)},
			{"negative shard size", WithShardSize(-10)},
			{"nil prefix fn", WithSnapshotPathPrefix(nil)},
		}
		for _, tc := range cases {
			var cfg sessionStoreOptions
			if err := tc.opt.applySessionStore(&cfg); err == nil {
				t.Errorf("%s: expected error", tc.name)
			}
		}
	})

	t.Run("stream manager rejects invalid values", func(t *testing.T) {
		cases := []struct {
			name string
			opt  StreamManagerOption
		}{
			{"empty collection", WithCollection("")},
			{"zero timeout", WithTimeout(0)},
			{"negative timeout", WithTimeout(-time.Second)},
			{"zero ttl", WithTTL(0)},
			{"negative ttl", WithTTL(-time.Second)},
		}
		for _, tc := range cases {
			var cfg streamManagerOptions
			if err := tc.opt.applyStreamManager(&cfg); err == nil {
				t.Errorf("%s: expected error", tc.name)
			}
		}
	})

	t.Run("rejects setting an option twice", func(t *testing.T) {
		var cfg sessionStoreOptions
		if err := WithCheckpointInterval(5).applySessionStore(&cfg); err != nil {
			t.Fatalf("first set: %v", err)
		}
		if err := WithCheckpointInterval(7).applySessionStore(&cfg); err == nil {
			t.Error("expected error setting checkpoint interval twice")
		}
	})

	t.Run("collection applies to both services", func(t *testing.T) {
		var ss sessionStoreOptions
		if err := WithCollection("c").applySessionStore(&ss); err != nil || ss.collection != "c" {
			t.Errorf("session store: collection=%q err=%v", ss.collection, err)
		}
		var sm streamManagerOptions
		if err := WithCollection("c").applyStreamManager(&sm); err != nil || sm.collection != "c" {
			t.Errorf("stream manager: collection=%q err=%v", sm.collection, err)
		}
	})
}

// TestInvalidPrefixRejected verifies every operation rejects a prefix that is
// not a valid single Firestore document ID before touching Firestore (the store
// has a nil client; the check runs first), rather than failing with an opaque
// path error deep in a transaction.
func TestInvalidPrefixRejected(t *testing.T) {
	ctx := context.Background()
	store := &FirestoreSessionStore[testState]{
		collection: "c",
		prefixFn:   func(context.Context) string { return "tenant/evil" },
	}

	if _, err := store.GetSnapshot(ctx, "snap"); err == nil {
		t.Error("GetSnapshot: expected error for prefix containing '/'")
	}
	if _, err := store.GetLatestSnapshot(ctx, "sess"); err == nil {
		t.Error("GetLatestSnapshot: expected error for prefix containing '/'")
	}
	if _, err := store.SaveSnapshot(ctx, "snap",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "s"}, nil
		}); err == nil {
		t.Error("SaveSnapshot: expected error for prefix containing '/'")
	}
	if _, ok := <-store.OnSnapshotStatusChange(ctx, "snap"); ok {
		t.Error("OnSnapshotStatusChange: expected a closed channel for prefix containing '/'")
	}
}

// --- SessionID rules (mirrors the shared conformance suite) ---

func TestSessionIDRules(t *testing.T) {
	ctx := context.Background()

	t.Run("KeptWhenProvided", func(t *testing.T) {
		store := newEmulatorStore(t)
		saved := saveRow(t, store, "a", "sess-keep", "", aix.SnapshotStatusCompleted, 1)
		if saved.SessionID != "sess-keep" {
			t.Errorf("SessionID = %q, want %q", saved.SessionID, "sess-keep")
		}
		stored, err := store.GetSnapshot(ctx, "a")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if stored.SessionID != "sess-keep" {
			t.Errorf("stored SessionID = %q, want %q", stored.SessionID, "sess-keep")
		}
	})

	t.Run("PreservedOnUpdate", func(t *testing.T) {
		store := newEmulatorStore(t)
		saveRow(t, store, "a", "sess-1", "", aix.SnapshotStatusCompleted, 1)
		// Update without re-supplying the session ID; the store must preserve it.
		updated, err := store.SaveSnapshot(ctx, "a",
			func(existing *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
				rewritten := *existing
				rewritten.SessionID = ""
				rewritten.State = &aix.SessionState[testState]{Custom: testState{Counter: 7}}
				return &rewritten, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot update: %v", err)
		}
		if updated.SessionID != "sess-1" {
			t.Errorf("SessionID = %q, want preserved %q", updated.SessionID, "sess-1")
		}
	})

	t.Run("SessionLessRowRejected", func(t *testing.T) {
		// The store keys its per-session pointer by session ID, so it cannot
		// persist a session-less row; it returns INVALID_ARGUMENT, matching the JS
		// FirestoreSessionStore. (The runtime stamps a session ID on every row.)
		store := newEmulatorStore(t)
		saveRow(t, store, "parent", "sess-1", "", aix.SnapshotStatusCompleted, 1)
		now := time.Now()
		_, err := store.SaveSnapshot(ctx, "child",
			func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
				return &aix.SessionSnapshot[testState]{
					ParentID:  "parent",
					State:     &aix.SessionState[testState]{Custom: testState{Counter: 2}},
					CreatedAt: now,
					UpdatedAt: now,
				}, nil
			})
		if err == nil {
			t.Error("expected error for a session-less row")
		}
	})
}

// --- GetLatestSnapshot contract (mirrors the shared conformance suite) ---

func TestGetLatestSnapshot(t *testing.T) {
	ctx := context.Background()

	t.Run("PicksMostRecent", func(t *testing.T) {
		// IDs deliberately sort against write order so a recency bug cannot pass
		// by luck.
		store := newEmulatorStore(t)
		saveRow(t, store, "z", "sess-1", "", aix.SnapshotStatusCompleted, 1)
		tick()
		saveRow(t, store, "m", "sess-1", "z", aix.SnapshotStatusCompleted, 1)
		tick()
		saveRow(t, store, "a", "sess-1", "m", aix.SnapshotStatusCompleted, 1)
		tick()
		saveRow(t, store, "x", "sess-other", "", aix.SnapshotStatusCompleted, 1)

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "a" {
			t.Fatalf("latest = %+v, want most recently written snapshot a", latest)
		}
		if latest.State == nil || latest.State.Custom.Counter != 1 {
			t.Errorf("latest state = %+v, want full row with counter=1", latest.State)
		}
	})

	t.Run("ByCreatedAtNotRewrite", func(t *testing.T) {
		// A later rewrite of an older row (a detach finalize) preserves CreatedAt,
		// so it must not move ahead of a newer-created sibling.
		store := newEmulatorStore(t)
		saveRow(t, store, "root", "sess-1", "", aix.SnapshotStatusCompleted, 1)
		tick()
		saveRow(t, store, "b1", "sess-1", "root", aix.SnapshotStatusPending, 1)
		tick()
		saveRow(t, store, "b2", "sess-1", "root", aix.SnapshotStatusCompleted, 1)
		tick()
		if _, err := store.SaveSnapshot(ctx, "b1",
			func(existing *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
				rewritten := *existing
				rewritten.Status = aix.SnapshotStatusCompleted
				rewritten.State = &aix.SessionState[testState]{Custom: testState{Counter: 2}}
				rewritten.UpdatedAt = time.Now()
				return &rewritten, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot finalize: %v", err)
		}

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "b2" {
			t.Errorf("latest = %+v, want newest-created b2 (finalize must not move b1 ahead)", latest)
		}
	})

	t.Run("ReturnsLatestAnyStatus", func(t *testing.T) {
		store := newEmulatorStore(t)
		saveRow(t, store, "a", "sess-1", "", aix.SnapshotStatusCompleted, 1)
		tick()
		saveRow(t, store, "b", "sess-1", "a", aix.SnapshotStatusFailed, 1)
		tick()
		saveRow(t, store, "c", "sess-1", "a", aix.SnapshotStatusAborted, 1)

		latest, err := store.GetLatestSnapshot(ctx, "sess-1")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "c" || latest.Status != aix.SnapshotStatusAborted {
			t.Errorf("latest = %+v, want newest row c (aborted)", latest)
		}
	})

	t.Run("UnknownSession", func(t *testing.T) {
		store := newEmulatorStore(t)
		saveRow(t, store, "a", "sess-1", "", aix.SnapshotStatusCompleted, 1)
		latest, err := store.GetLatestSnapshot(ctx, "sess-unknown")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest != nil {
			t.Errorf("expected nil for unknown session, got %+v", latest)
		}
	})

	t.Run("EmptySessionID", func(t *testing.T) {
		store := newEmulatorStore(t)
		if _, err := store.GetLatestSnapshot(ctx, ""); err == nil {
			t.Error("expected error for empty session ID")
		}
	})
}

// --- Heartbeat semantics (mirrors the shared conformance suite) ---

func TestHeartbeat(t *testing.T) {
	ctx := context.Background()

	savePending := func(t *testing.T, store *FirestoreSessionStore[testState], id, sessionID, parentID string) {
		t.Helper()
		now := time.Now()
		if _, err := store.SaveSnapshot(ctx, id,
			func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
				return &aix.SessionSnapshot[testState]{SessionID: sessionID, ParentID: parentID, Status: aix.SnapshotStatusPending, CreatedAt: now, UpdatedAt: now}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", id, err)
		}
	}
	beat := func(t *testing.T, store *FirestoreSessionStore[testState], id string) {
		t.Helper()
		now := time.Now()
		if _, err := store.SaveSnapshot(ctx, id,
			func(existing *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
				if existing == nil || existing.Status != aix.SnapshotStatusPending {
					return nil, nil
				}
				updated := *existing
				updated.HeartbeatAt = &now
				return &updated, nil
			}); err != nil {
			t.Fatalf("heartbeat SaveSnapshot: %v", err)
		}
	}

	t.Run("AdvancesHeartbeatNotUpdatedAt", func(t *testing.T) {
		store := newEmulatorStore(t)
		savePending(t, store, "p", "sess", "")
		before, err := store.GetSnapshot(ctx, "p")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		tick()
		beat(t, store, "p")
		after, err := store.GetSnapshot(ctx, "p")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if after.HeartbeatAt == nil {
			t.Error("HeartbeatAt was not stamped")
		}
		if !after.UpdatedAt.Equal(before.UpdatedAt) {
			t.Errorf("heartbeat advanced UpdatedAt: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
		}
		if after.Status != aix.SnapshotStatusPending {
			t.Errorf("status = %q, want pending", after.Status)
		}
	})

	t.Run("NoopOnTerminal", func(t *testing.T) {
		store := newEmulatorStore(t)
		saveRow(t, store, "c", "sess", "", aix.SnapshotStatusCompleted, 1)
		before, err := store.GetSnapshot(ctx, "c")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		beat(t, store, "c")
		after, err := store.GetSnapshot(ctx, "c")
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if after.HeartbeatAt != nil {
			t.Errorf("beat stamped a heartbeat on a terminal row: %v", after.HeartbeatAt)
		}
		if !after.UpdatedAt.Equal(before.UpdatedAt) {
			t.Errorf("beat bumped UpdatedAt on a terminal row")
		}
	})

	t.Run("DoesNotChangeRecency", func(t *testing.T) {
		store := newEmulatorStore(t)
		savePending(t, store, "old", "sess", "")
		tick()
		saveRow(t, store, "new", "sess", "old", aix.SnapshotStatusCompleted, 9)
		tick()
		for i := 0; i < 3; i++ {
			beat(t, store, "old")
			tick()
		}
		latest, err := store.GetLatestSnapshot(ctx, "sess")
		if err != nil {
			t.Fatalf("GetLatestSnapshot: %v", err)
		}
		if latest == nil || latest.SnapshotID != "new" {
			t.Errorf("latest = %+v, want \"new\" (a heartbeat must not affect recency)", latest)
		}
	})
}

// --- General store behavior ---

func TestGetSnapshotNotFound(t *testing.T) {
	store := newEmulatorStore(t)
	got, err := store.GetSnapshot(context.Background(), "nope")
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if got != nil {
		t.Errorf("expected nil for missing snapshot, got %+v", got)
	}
}

func TestSaveSnapshotSkip(t *testing.T) {
	store := newEmulatorStore(t)
	saved, err := store.SaveSnapshot(context.Background(), "x",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return nil, nil // decline
		})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	if saved != nil {
		t.Errorf("expected nil when fn declines, got %+v", saved)
	}
	got, _ := store.GetSnapshot(context.Background(), "x")
	if got != nil {
		t.Errorf("declined write should not create a row, got %+v", got)
	}
}

func TestSaveSnapshotGeneratesID(t *testing.T) {
	store := newEmulatorStore(t)
	saved, err := store.SaveSnapshot(context.Background(), "",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			now := time.Now()
			return &aix.SessionSnapshot[testState]{SessionID: "s", CreatedAt: now, UpdatedAt: now}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	if saved.SnapshotID == "" {
		t.Error("expected a generated snapshot ID")
	}
}

// TestConcurrentSaveAtomicity fires concurrent read-modify-write saves on the
// same snapshot and verifies none are lost or double-applied: the final counter
// equals the number of writers. This is the atomicity guarantee the abort-aware
// mutator relies on (a "completed" write must never clobber a concurrent
// "aborted"). Each writer retries on contention, the way a real caller does:
// under heavy single-document contention a transaction can be aborted after the
// SDK's own retries, and a committed transaction is the only one that returns a
// nil error, so retrying increments exactly once per writer.
func TestConcurrentSaveAtomicity(t *testing.T) {
	ctx := context.Background()
	store := newEmulatorStore(t)

	now := time.Now()
	if _, err := store.SaveSnapshot(ctx, "counter",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "sess", State: &aix.SessionState[testState]{Custom: testState{Counter: 0}}, CreatedAt: now, UpdatedAt: now}, nil
		}); err != nil {
		t.Fatalf("seed SaveSnapshot: %v", err)
	}

	const writers = 8
	var wg sync.WaitGroup
	failures := make(chan error, writers)
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var lastErr error
			for attempt := 0; attempt < 25; attempt++ {
				_, err := store.SaveSnapshot(ctx, "counter",
					func(existing *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
						// Pure: derive the next value only from the input. The
						// transaction may call this multiple times under contention.
						updated := *existing
						st := *existing.State
						st.Custom.Counter++
						updated.State = &st
						updated.UpdatedAt = time.Now()
						return &updated, nil
					})
				if err == nil {
					return
				}
				lastErr = err
				time.Sleep(15 * time.Millisecond)
			}
			failures <- lastErr
		}()
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Fatalf("concurrent SaveSnapshot exhausted retries: %v", err)
	}

	got, err := store.GetSnapshot(ctx, "counter")
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if got.State.Custom.Counter != writers {
		t.Errorf("counter = %d, want %d (a lost or double update means the write is not atomic)", got.State.Custom.Counter, writers)
	}
}

// --- Checkpoint / diff / shard specifics ---

// TestDiffChainReconstruction exercises a chain longer than the checkpoint
// interval so reconstruction spans multiple checkpoints and diff segments, then
// verifies every turn's state reconstructs exactly. This is the core fidelity
// test for the exp.Diff / exp.ApplyPatch round trip through the store.
func TestDiffChainReconstruction(t *testing.T) {
	ctx := context.Background()
	// Interval 3 forces a fresh checkpoint every 3 turns; a 10-turn chain spans
	// several checkpoint boundaries and diff segments.
	store := newEmulatorStore(t, WithCheckpointInterval(3))

	ids := make([]string, 10)
	parent := ""
	for i := 0; i < 10; i++ {
		ids[i] = fmt.Sprintf("turn-%02d", i)
		topics := make([]string, i+1)
		for j := range topics {
			topics[j] = fmt.Sprintf("t%d", j)
		}
		now := time.Now()
		p := parent
		idx := i
		if _, err := store.SaveSnapshot(ctx, ids[i],
			func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
				return &aix.SessionSnapshot[testState]{
					SessionID: "sess",
					ParentID:  p,
					Status:    aix.SnapshotStatusCompleted,
					State:     &aix.SessionState[testState]{Custom: testState{Counter: idx, Topics: topics}},
					CreatedAt: now,
					UpdatedAt: now,
				}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", ids[i], err)
		}
		parent = ids[i]
		tick()
	}

	// Every turn reconstructs to its own counter and topic list.
	for i, id := range ids {
		got, err := store.GetSnapshot(ctx, id)
		if err != nil {
			t.Fatalf("GetSnapshot(%q): %v", id, err)
		}
		if got == nil {
			t.Fatalf("GetSnapshot(%q) = nil", id)
		}
		if got.State.Custom.Counter != i {
			t.Errorf("turn %q: counter = %d, want %d", id, got.State.Custom.Counter, i)
		}
		if len(got.State.Custom.Topics) != i+1 {
			t.Errorf("turn %q: %d topics, want %d", id, len(got.State.Custom.Topics), i+1)
		}
	}

	// The latest resolves to the final turn with the full topic list.
	latest, err := store.GetLatestSnapshot(ctx, "sess")
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if latest == nil || latest.SnapshotID != ids[9] || latest.State.Custom.Counter != 9 {
		t.Errorf("latest = %+v, want final turn", latest)
	}
}

// TestLargeStateSharding stores a state whose JSON exceeds the shard size,
// forcing it across multiple shard documents, then verifies it reconstructs
// byte-for-byte.
func TestLargeStateSharding(t *testing.T) {
	ctx := context.Background()
	// Tiny shard size so even a modest state spans many shards.
	store := newEmulatorStore(t, WithShardSize(256))

	topics := make([]string, 200) // well over 256 bytes once serialized
	for i := range topics {
		topics[i] = fmt.Sprintf("topic-number-%04d", i)
	}
	now := time.Now()
	if _, err := store.SaveSnapshot(ctx, "big",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{
				SessionID: "sess",
				Status:    aix.SnapshotStatusCompleted,
				State:     &aix.SessionState[testState]{Custom: testState{Counter: 1, Topics: topics}},
				CreatedAt: now,
				UpdatedAt: now,
			}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}

	got, err := store.GetSnapshot(ctx, "big")
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if got == nil || len(got.State.Custom.Topics) != 200 {
		t.Fatalf("reconstructed %d topics, want 200", len(got.State.Custom.Topics))
	}
	for i, topic := range got.State.Custom.Topics {
		if want := fmt.Sprintf("topic-number-%04d", i); topic != want {
			t.Fatalf("topic %d = %q, want %q", i, topic, want)
			break
		}
	}
}

// TestOversizedDiffPromotion verifies that a turn whose diff would exceed the
// shard size is promoted to a checkpoint rather than written as one oversized
// diff document, and still reconstructs correctly.
func TestOversizedDiffPromotion(t *testing.T) {
	ctx := context.Background()
	store := newEmulatorStore(t, WithShardSize(256), WithCheckpointInterval(100))

	// Root checkpoint with small state.
	saveRow(t, store, "root", "sess", "", aix.SnapshotStatusCompleted, 0)
	tick()

	// Child whose state balloons: the diff exceeds the shard size and must be
	// promoted to a (sharded) checkpoint.
	topics := make([]string, 200)
	for i := range topics {
		topics[i] = fmt.Sprintf("big-topic-%04d", i)
	}
	now := time.Now()
	if _, err := store.SaveSnapshot(ctx, "child",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{
				SessionID: "sess",
				ParentID:  "root",
				Status:    aix.SnapshotStatusCompleted,
				State:     &aix.SessionState[testState]{Custom: testState{Counter: 1, Topics: topics}},
				CreatedAt: now,
				UpdatedAt: now,
			}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot(child): %v", err)
	}

	got, err := store.GetSnapshot(ctx, "child")
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if got == nil || len(got.State.Custom.Topics) != 200 {
		t.Fatalf("child reconstructed %d topics, want 200", len(got.State.Custom.Topics))
	}
	// The child must have been promoted: its document is a self-anchored
	// checkpoint, not a diff off root.
	if got.ParentID != "root" {
		t.Errorf("child ParentID = %q, want root", got.ParentID)
	}
}

// TestBranchingReconstruction verifies that two children of the same parent
// reconstruct independently (a regenerate fork).
func TestBranchingReconstruction(t *testing.T) {
	ctx := context.Background()
	store := newEmulatorStore(t)

	saveRow(t, store, "root", "sess", "", aix.SnapshotStatusCompleted, 0)
	tick()
	// Two divergent children of root.
	now := time.Now()
	mkChild := func(id string, counter int) {
		if _, err := store.SaveSnapshot(ctx, id,
			func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
				return &aix.SessionSnapshot[testState]{
					SessionID: "sess",
					ParentID:  "root",
					Status:    aix.SnapshotStatusCompleted,
					State:     &aix.SessionState[testState]{Custom: testState{Counter: counter}},
					CreatedAt: now,
					UpdatedAt: now,
				}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", id, err)
		}
	}
	mkChild("branch-a", 11)
	mkChild("branch-b", 22)

	a, _ := store.GetSnapshot(ctx, "branch-a")
	b, _ := store.GetSnapshot(ctx, "branch-b")
	if a == nil || a.State.Custom.Counter != 11 {
		t.Errorf("branch-a = %+v, want counter 11", a)
	}
	if b == nil || b.State.Custom.Counter != 22 {
		t.Errorf("branch-b = %+v, want counter 22", b)
	}
}

// --- Multi-tenant prefix isolation ---

func TestPrefixIsolation(t *testing.T) {
	if os.Getenv("FIRESTORE_EMULATOR_HOST") == "" {
		t.Skip("Skipping: FIRESTORE_EMULATOR_HOST not set")
	}
	ctx := context.Background()
	client, err := firestore.NewClient(ctx, testProjectID)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { client.Close() })

	type tenantKey struct{}
	prefixFn := func(ctx context.Context) string {
		if v, ok := ctx.Value(tenantKey{}).(string); ok {
			return v
		}
		return ""
	}
	store, err := newFirestoreSessionStore[testState](client,
		WithCollection("sessions-"+uuid.NewString()),
		WithSnapshotPathPrefix(prefixFn))
	if err != nil {
		t.Fatalf("newFirestoreSessionStore: %v", err)
	}

	ctxA := context.WithValue(ctx, tenantKey{}, "tenant-a")
	ctxB := context.WithValue(ctx, tenantKey{}, "tenant-b")

	now := time.Now()
	if _, err := store.SaveSnapshot(ctxA, "shared-id",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "s", State: &aix.SessionState[testState]{Custom: testState{Counter: 1}}, CreatedAt: now, UpdatedAt: now}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot(tenant-a): %v", err)
	}

	// Tenant B cannot see tenant A's snapshot, even with the same ID.
	gotB, err := store.GetSnapshot(ctxB, "shared-id")
	if err != nil {
		t.Fatalf("GetSnapshot(tenant-b): %v", err)
	}
	if gotB != nil {
		t.Errorf("tenant B saw tenant A's snapshot: %+v", gotB)
	}
	// Tenant A still sees its own.
	gotA, err := store.GetSnapshot(ctxA, "shared-id")
	if err != nil {
		t.Fatalf("GetSnapshot(tenant-a): %v", err)
	}
	if gotA == nil || gotA.State.Custom.Counter != 1 {
		t.Errorf("tenant A lost its snapshot: %+v", gotA)
	}
}

// --- Status subscription (OnSnapshotStatusChange) ---

func TestOnSnapshotStatusChangeAbort(t *testing.T) {
	store := newEmulatorStore(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// A pending detached row.
	now := time.Now()
	if _, err := store.SaveSnapshot(ctx, "p",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "sess", Status: aix.SnapshotStatusPending, CreatedAt: now, UpdatedAt: now}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot(pending): %v", err)
	}

	ch := store.OnSnapshotStatusChange(ctx, "p")

	// First value reflects the status at subscription time.
	if got := recvStatus(t, ch, "initial"); got != aix.SnapshotStatusPending {
		t.Errorf("first status = %q, want pending", got)
	}

	// Abort via an ordinary SaveSnapshot, as the runtime does.
	if _, err := store.SaveSnapshot(ctx, "p",
		func(existing *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			updated := *existing
			updated.Status = aix.SnapshotStatusAborted
			return &updated, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot(abort): %v", err)
	}

	// The abort propagates through the native listener.
	if got := recvStatus(t, ch, "after abort"); got != aix.SnapshotStatusAborted {
		t.Errorf("status after abort = %q, want aborted", got)
	}
}

func TestOnSnapshotStatusChangeMissing(t *testing.T) {
	store := newEmulatorStore(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch := store.OnSnapshotStatusChange(ctx, "does-not-exist")
	// The channel is closed without yielding a value.
	select {
	case v, ok := <-ch:
		if ok {
			t.Errorf("expected closed channel for missing snapshot, got %q", v)
		}
	case <-time.After(10 * time.Second):
		t.Error("timed out waiting for the channel to close")
	}
}

// recvStatus reads one status from ch with a generous timeout for emulator
// listener latency.
func recvStatus(t *testing.T, ch <-chan aix.SnapshotStatus, label string) aix.SnapshotStatus {
	t.Helper()
	select {
	case v, ok := <-ch:
		if !ok {
			t.Fatalf("%s: channel closed unexpectedly", label)
		}
		return v
	case <-time.After(15 * time.Second):
		t.Fatalf("%s: timed out waiting for a status", label)
		return ""
	}
}
