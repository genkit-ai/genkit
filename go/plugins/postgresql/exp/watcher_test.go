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
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	aix "github.com/firebase/genkit/go/ai/exp"
)

// listenerPIDs returns the backends LISTENing on the store's channel.
func listenerPIDs(t *testing.T, pool *pgxpool.Pool, store *PostgresSessionStore[testState]) []int32 {
	t.Helper()
	rows, err := pool.Query(context.Background(), `SELECT pid FROM pg_stat_activity WHERE query = $1`,
		"LISTEN "+pgx.Identifier{store.watcher.channel}.Sanitize())
	if err != nil {
		t.Fatalf("query pg_stat_activity: %v", err)
	}
	pids, err := pgx.CollectRows(rows, pgx.RowTo[int32])
	if err != nil {
		t.Fatalf("collect pids: %v", err)
	}
	return pids
}

// savePending writes a pending row and subscribes to it, returning the
// subscription after its first value.
func savePending(t *testing.T, store *PostgresSessionStore[testState], id string) <-chan aix.SnapshotStatus {
	t.Helper()
	ctx := context.Background()
	now := time.Now()
	if _, err := store.SaveSnapshot(ctx, id, func(*aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
		return &aix.SessionSnapshot[testState]{SessionID: "sess", Status: aix.SnapshotStatusPending, CreatedAt: now, UpdatedAt: now, HeartbeatAt: &now}, nil
	}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	subCtx, cancel := context.WithCancel(ctx)
	t.Cleanup(cancel)
	ch := store.OnSnapshotStatusChange(subCtx, id)
	if got := waitFor(t, ch, func(aix.SnapshotStatus) bool { return true }); got != aix.SnapshotStatusPending {
		t.Fatalf("first status = %q, want pending", got)
	}
	return ch
}

// waitFor reads ch until a status satisfies ok, and returns it.
func waitFor(t *testing.T, ch <-chan aix.SnapshotStatus, ok func(aix.SnapshotStatus) bool) aix.SnapshotStatus {
	t.Helper()
	timeout := time.After(10 * time.Second)
	for {
		select {
		case st, open := <-ch:
			if !open {
				t.Fatal("the subscription closed")
			}
			if ok(st) {
				return st
			}
		case <-timeout:
			t.Fatal("no matching status within 10s")
			return ""
		}
	}
}

// setStatus changes a row's status with plain SQL, the way an operator would,
// so no notification is sent.
func setStatus(t *testing.T, pool *pgxpool.Pool, store *PostgresSessionStore[testState], id string, st aix.SnapshotStatus) {
	t.Helper()
	if _, err := pool.Exec(context.Background(), fmt.Sprintf(`UPDATE %s SET status = $1 WHERE snapshot_id = $2`, store.table), string(st), id); err != nil {
		t.Fatalf("UPDATE status: %v", err)
	}
}

// TestWatcherRecoversALostConnection checks that a subscription outlives its
// LISTEN connection: with the poll off, a change made while the watcher
// reconnects still arrives, through the read that follows every reconnect.
func TestWatcherRecoversALostConnection(t *testing.T) {
	pool := testPool(t)
	store := newTestStore(t, pool, testTable(t, pool), WithPollInterval(0))
	ch := savePending(t, store, "p")

	pids := listenerPIDs(t, pool, store)
	if len(pids) != 1 {
		t.Fatalf("%d backends LISTEN on the store's channel, want 1", len(pids))
	}
	if _, err := pool.Exec(context.Background(), `SELECT pg_terminate_backend($1)`, pids[0]); err != nil {
		t.Fatalf("pg_terminate_backend: %v", err)
	}
	setStatus(t, pool, store, "p", aix.SnapshotStatusAborting)
	waitFor(t, ch, func(st aix.SnapshotStatus) bool { return st == aix.SnapshotStatusAborting })
}

// TestWatcherPollsForUnnotifiedChanges checks that a status change written
// without a notification reaches subscribers through the poll.
func TestWatcherPollsForUnnotifiedChanges(t *testing.T) {
	pool := testPool(t)
	store := newTestStore(t, pool, testTable(t, pool), WithPollInterval(50*time.Millisecond))
	ch := savePending(t, store, "p")
	setStatus(t, pool, store, "p", aix.SnapshotStatusAborting)
	waitFor(t, ch, func(st aix.SnapshotStatus) bool { return st == aix.SnapshotStatusAborting })
}

// TestWatcherReleasesItsConnection checks that the LISTEN connection closes
// once the last subscription ends, so an idle store holds no connection.
func TestWatcherReleasesItsConnection(t *testing.T) {
	pool := testPool(t)
	store := newTestStore(t, pool, testTable(t, pool))
	now := time.Now()
	if _, err := store.SaveSnapshot(context.Background(), "p", func(*aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
		return &aix.SessionSnapshot[testState]{SessionID: "sess", Status: aix.SnapshotStatusPending, CreatedAt: now, UpdatedAt: now}, nil
	}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	subCtx, cancel := context.WithCancel(context.Background())
	ch := store.OnSnapshotStatusChange(subCtx, "p")
	<-ch
	if n := len(listenerPIDs(t, pool, store)); n != 1 {
		t.Fatalf("%d backends LISTEN while subscribed, want 1", n)
	}
	cancel()
	deadline := time.Now().Add(10 * time.Second)
	for len(listenerPIDs(t, pool, store)) != 0 {
		if time.Now().After(deadline) {
			t.Fatal("the LISTEN connection is still open 10s after the last subscription ended")
		}
		time.Sleep(20 * time.Millisecond)
	}
}
