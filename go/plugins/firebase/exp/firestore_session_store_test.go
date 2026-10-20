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
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"cloud.google.com/go/firestore"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/sessionstoretest"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"github.com/google/uuid"
)

// testState is the custom-state type the store tests use.
type testState struct {
	Counter int      `json:"counter"`
	Topics  []string `json:"topics,omitempty"`
}

const testProjectID = "genkit-firestore-session-test"

// newEmulatorClient returns a client for the Firestore emulator. It skips the
// test when FIRESTORE_EMULATOR_HOST is not set.
func newEmulatorClient(t *testing.T) *firestore.Client {
	t.Helper()
	if os.Getenv("FIRESTORE_EMULATOR_HOST") == "" {
		t.Skip("Skipping: FIRESTORE_EMULATOR_HOST not set (start the Firestore emulator to run these tests)")
	}
	client, err := firestore.NewClient(context.Background(), testProjectID)
	if err != nil {
		t.Fatalf("firestore.NewClient: %v", err)
	}
	t.Cleanup(func() { client.Close() })
	return client
}

// newEmulatorStore creates a store backed by the Firestore emulator, isolated to
// a unique collection per call. Pass options (e.g. a small shard size or
// checkpoint interval) to exercise the sharding and checkpoint paths.
func newEmulatorStore(t *testing.T, opts ...SessionStoreOption) *FirestoreSessionStore[testState] {
	t.Helper()
	client := newEmulatorClient(t)
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

type tenantKey struct{}

func tenantFromContext(ctx context.Context) string {
	tenant, _ := ctx.Value(tenantKey{}).(string)
	return tenant
}

func TestFirestoreSessionStore(t *testing.T) {
	sessionstoretest.Run(t, func(t *testing.T) aix.SessionStore[testState] {
		return newEmulatorStore(t)
	}, &sessionstoretest.Options[testState]{
		// A store on its own client over the same collection stands in for
		// another process.
		Reopen: func(t *testing.T, store aix.SessionStore[testState]) aix.SessionStore[testState] {
			reopened, err := newFirestoreSessionStore[testState](newEmulatorClient(t),
				WithCollection(store.(*FirestoreSessionStore[testState]).collection))
			if err != nil {
				t.Fatalf("newFirestoreSessionStore: %v", err)
			}
			return reopened
		},
		Scoped: func(t *testing.T) (aix.SessionStore[testState], context.Context, context.Context) {
			ctx := context.Background()
			return newEmulatorStore(t, WithSnapshotPathPrefix(tenantFromContext)),
				context.WithValue(ctx, tenantKey{}, "tenant-a"),
				context.WithValue(ctx, tenantKey{}, "tenant-b")
		},
	})
}

// TestFirestoreSessionStore_FrequentCheckpoints runs the suite with a checkpoint
// every three turns, so its chains cross many checkpoint boundaries and every
// kind of write lands on both sides of one.
func TestFirestoreSessionStore_FrequentCheckpoints(t *testing.T) {
	sessionstoretest.Run(t, func(t *testing.T) aix.SessionStore[testState] {
		return newEmulatorStore(t, WithCheckpointInterval(3))
	}, nil)
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

	t.Run("a repeated option replaces the earlier one", func(t *testing.T) {
		var ss sessionStoreOptions
		for _, opt := range []SessionStoreOption{
			WithCollection("a"), WithCollection("b"),
			WithCheckpointInterval(5), WithCheckpointInterval(7),
			WithShardSize(1), WithShardSize(2),
			WithSnapshotPathPrefix(func(context.Context) string { return "first" }),
			WithSnapshotPathPrefix(func(context.Context) string { return "second" }),
		} {
			if err := opt.applySessionStore(&ss); err != nil {
				t.Fatalf("applySessionStore: %v", err)
			}
		}
		if ss.collection != "b" || ss.checkpointInterval != 7 || ss.shardSize != 2 || ss.prefixFn(context.Background()) != "second" {
			t.Errorf("session store options = %+v, want the later of each", ss)
		}
		var sm streamManagerOptions
		for _, opt := range []StreamManagerOption{
			WithCollection("a"), WithCollection("b"),
			WithTimeout(time.Second), WithTimeout(2 * time.Second),
			WithTTL(time.Minute), WithTTL(2 * time.Minute),
		} {
			if err := opt.applyStreamManager(&sm); err != nil {
				t.Fatalf("applyStreamManager: %v", err)
			}
		}
		if sm.collection != "b" || sm.timeout != 2*time.Second || sm.ttl != 2*time.Minute {
			t.Errorf("stream manager options = %+v, want the later of each", sm)
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

// TestEmptyPrefixRejected verifies every operation rejects a configured prefix
// function that returns an empty value, before touching Firestore (the store has
// a nil client). The default "global" prefix is requested by omitting
// WithSnapshotPathPrefix, not by returning an empty value from it.
func TestEmptyPrefixRejected(t *testing.T) {
	ctx := context.Background()
	store := &FirestoreSessionStore[testState]{
		collection: "c",
		prefixFn:   func(context.Context) string { return "" },
	}

	if _, err := store.GetSnapshot(ctx, "snap"); err == nil {
		t.Error("GetSnapshot: expected error for empty prefix")
	}
	if _, err := store.GetLatestSnapshot(ctx, "sess"); err == nil {
		t.Error("GetLatestSnapshot: expected error for empty prefix")
	}
	if _, err := store.SaveSnapshot(ctx, "snap",
		func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "s"}, nil
		}); err == nil {
		t.Error("SaveSnapshot: expected error for empty prefix")
	}
	if _, ok := <-store.OnSnapshotStatusChange(ctx, "snap"); ok {
		t.Error("OnSnapshotStatusChange: expected a closed channel for empty prefix")
	}
}

// --- Sharding specifics ---

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

func countDocs(t *testing.T, store *FirestoreSessionStore[testState], collection, sub string) int {
	t.Helper()
	docs, err := store.client.Collection(collection).Doc(defaultPrefix).Collection(sub).Documents(context.Background()).GetAll()
	if err != nil {
		t.Fatalf("list %s/%s: %v", collection, sub, err)
	}
	return len(docs)
}

func TestDeleteSession(t *testing.T) {
	ctx := context.Background()
	// Small shards and a short checkpoint interval force several checkpoints,
	// each sharded across many documents.
	newStore := func(t *testing.T) *FirestoreSessionStore[testState] {
		return newEmulatorStore(t, WithShardSize(64), WithCheckpointInterval(2))
	}
	chain := func(t *testing.T, store *FirestoreSessionStore[testState], sessionID string, n int) {
		t.Helper()
		parent := ""
		for i := range n {
			id := fmt.Sprintf("%s-%d", sessionID, i)
			saveRow(t, store, id, sessionID, parent, aix.SnapshotStatusCompleted, i)
			parent = id
			tick()
		}
	}

	t.Run("removes snapshots and shards", func(t *testing.T) {
		store := newStore(t)
		chain(t, store, "sess-a", 5)
		if countDocs(t, store, store.collection+"-shards", "shards") == 0 {
			t.Fatal("test setup wrote no shards")
		}

		removed, err := store.DeleteSession(ctx, "sess-a")
		if err != nil || !removed {
			t.Fatalf("DeleteSession = %v, %v; want true, nil", removed, err)
		}
		if got, err := store.GetLatestSnapshot(ctx, "sess-a"); err != nil || got != nil {
			t.Errorf("GetLatestSnapshot = %v, %v; want nil, nil", got, err)
		}
		if n := countDocs(t, store, store.collection, "snapshots"); n != 0 {
			t.Errorf("%d snapshot docs left, want 0", n)
		}
		if n := countDocs(t, store, store.collection+"-shards", "shards"); n != 0 {
			t.Errorf("%d shard docs left, want 0", n)
		}
	})

	t.Run("is idempotent", func(t *testing.T) {
		store := newStore(t)
		if removed, err := store.DeleteSession(ctx, "never-existed"); err != nil || removed {
			t.Errorf("DeleteSession(missing) = %v, %v; want false, nil", removed, err)
		}
		chain(t, store, "sess-b", 2)
		if removed, err := store.DeleteSession(ctx, "sess-b"); err != nil || !removed {
			t.Fatalf("first DeleteSession = %v, %v; want true, nil", removed, err)
		}
		if removed, err := store.DeleteSession(ctx, "sess-b"); err != nil || removed {
			t.Errorf("second DeleteSession = %v, %v; want false, nil", removed, err)
		}
	})

	t.Run("leaves other sessions untouched", func(t *testing.T) {
		store := newStore(t)
		chain(t, store, "sess-keep", 3)
		chain(t, store, "sess-drop", 3)
		if _, err := store.DeleteSession(ctx, "sess-drop"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}
		got, err := store.GetLatestSnapshot(ctx, "sess-keep")
		if err != nil || got == nil || got.SnapshotID != "sess-keep-2" || got.State.Custom.Counter != 2 {
			t.Errorf("surviving session = %+v, %v", got, err)
		}
	})

	t.Run("deletes a session larger than one batch", func(t *testing.T) {
		store := newEmulatorStore(t, WithShardSize(2))
		now := time.Now()
		big := make([]string, 0, 400)
		for range 400 {
			big = append(big, "xxxxxxxx")
		}
		if _, err := store.SaveSnapshot(ctx, "big-0", func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{
				SessionID: "sess-big",
				State:     &aix.SessionState[testState]{Custom: testState{Topics: big}},
				CreatedAt: now, UpdatedAt: now,
			}, nil
		}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		if n := countDocs(t, store, store.collection+"-shards", "shards"); n <= 500 {
			t.Fatalf("test setup wrote only %d shards, want > 500", n)
		}
		if removed, err := store.DeleteSession(ctx, "sess-big"); err != nil || !removed {
			t.Fatalf("DeleteSession = %v, %v; want true, nil", removed, err)
		}
		if n := countDocs(t, store, store.collection+"-shards", "shards"); n != 0 {
			t.Errorf("%d shard docs left, want 0", n)
		}
	})

	t.Run("rejects writers continuing the deleted chain", func(t *testing.T) {
		store := newStore(t)
		chain(t, store, "sess-late", 1)
		if _, err := store.DeleteSession(ctx, "sess-late"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}
		now := time.Now()
		_, err := store.SaveSnapshot(ctx, "late", func(_ *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "sess-late", ParentID: "sess-late-0", CreatedAt: now, UpdatedAt: now}, nil
		})
		if !errors.Is(err, aix.ErrSessionDeleted) {
			t.Errorf("late write error = %v, want ErrSessionDeleted", err)
		}
		if got, _ := store.GetLatestSnapshot(ctx, "sess-late"); got != nil {
			t.Errorf("session came back: %+v", got)
		}
	})

	t.Run("a fresh root snapshot starts the session again", func(t *testing.T) {
		store := newStore(t)
		chain(t, store, "sess-again", 2)
		if _, err := store.DeleteSession(ctx, "sess-again"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}
		saveRow(t, store, "again-root", "sess-again", "", aix.SnapshotStatusCompleted, 9)
		got, err := store.GetLatestSnapshot(ctx, "sess-again")
		if err != nil || got == nil || got.SnapshotID != "again-root" || got.State.Custom.Counter != 9 {
			t.Errorf("GetLatestSnapshot = %+v, %v; want again-root", got, err)
		}
		saveRow(t, store, "again-child", "sess-again", "again-root", aix.SnapshotStatusCompleted, 10)
	})

	t.Run("can be re-run after an interrupted delete", func(t *testing.T) {
		store := newStore(t)
		chain(t, store, "sess-retry", 3)
		// Simulate a crash after the shards were removed but before the
		// snapshots were.
		shards, err := store.client.Collection(store.collection + "-shards").Doc(defaultPrefix).Collection("shards").Documents(ctx).GetAll()
		if err != nil {
			t.Fatalf("list shards: %v", err)
		}
		for _, d := range shards {
			if _, err := d.Ref.Delete(ctx); err != nil {
				t.Fatalf("delete shard: %v", err)
			}
		}
		now := time.Now()
		if _, err := store.pointerRef(defaultPrefix, "sess-retry").Set(ctx, pointerDoc{DeletedAt: &now}); err != nil {
			t.Fatalf("tombstone: %v", err)
		}

		if removed, err := store.DeleteSession(ctx, "sess-retry"); err != nil || !removed {
			t.Fatalf("DeleteSession = %v, %v; want true, nil", removed, err)
		}
		if n := countDocs(t, store, store.collection, "snapshots"); n != 0 {
			t.Errorf("%d snapshot docs left, want 0", n)
		}
	})

	t.Run("notifies a subscriber as aborted", func(t *testing.T) {
		store := newStore(t)
		saveRow(t, store, "watched", "sess-watch", "", aix.SnapshotStatusPending, 1)
		subCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		ch := store.OnSnapshotStatusChange(subCtx, "watched")
		select {
		case got := <-ch:
			if got != aix.SnapshotStatusPending {
				t.Fatalf("initial status = %q, want pending", got)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("no initial status")
		}

		if _, err := store.DeleteSession(ctx, "sess-watch"); err != nil {
			t.Fatalf("DeleteSession: %v", err)
		}
		select {
		case got := <-ch:
			if got != aix.SnapshotStatusAborted {
				t.Errorf("status after delete = %q, want aborted", got)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("subscriber was not notified of the delete")
		}
	})

	t.Run("empty session ID", func(t *testing.T) {
		store := newStore(t)
		if _, err := store.DeleteSession(ctx, ""); !errors.Is(err, aix.ErrSessionIDRequired) {
			t.Errorf("error = %v, want ErrSessionIDRequired", err)
		}
	})
}
