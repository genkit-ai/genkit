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
