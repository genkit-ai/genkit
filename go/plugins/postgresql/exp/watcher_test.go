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
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
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

// eventually waits up to 10s for cond to hold, and fails t with what if it
// does not.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("%s: not within 10s", what)
		}
		time.Sleep(20 * time.Millisecond)
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
// LISTEN connection, with the poll off and with a poll that never comes due:
// the watcher LISTENs on a new connection, and the read that follows every
// reconnect delivers a change that no notification announced.
func TestWatcherRecoversALostConnection(t *testing.T) {
	for _, poll := range []time.Duration{0, time.Hour} {
		t.Run(fmt.Sprintf("poll %v", poll), func(t *testing.T) {
			pool := testPool(t)
			store := newTestStore(t, pool, testTable(t, pool), WithPollInterval(poll))
			ch := savePending(t, store, "p")

			pids := listenerPIDs(t, pool, store)
			if len(pids) != 1 {
				t.Fatalf("%d backends LISTEN on the store's channel, want 1", len(pids))
			}
			setStatus(t, pool, store, "p", aix.SnapshotStatusAborting)
			if _, err := pool.Exec(context.Background(), `SELECT pg_terminate_backend($1)`, pids[0]); err != nil {
				t.Fatalf("pg_terminate_backend: %v", err)
			}
			waitFor(t, ch, func(st aix.SnapshotStatus) bool { return st == aix.SnapshotStatusAborting })
			eventually(t, "the watcher LISTENs on a new connection", func() bool {
				now := listenerPIDs(t, pool, store)
				return len(now) == 1 && now[0] != pids[0]
			})
		})
	}
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

// TestWatcherWorksWithANotificationHandler checks that status changes arrive
// on a pool whose connections set OnNotification: pgx hands every
// notification to that handler and returns the store no payload.
func TestWatcherWorksWithANotificationHandler(t *testing.T) {
	var handled atomic.Int32
	pool := testPool(t, func(cfg *pgxpool.Config) {
		cfg.ConnConfig.OnNotification = func(*pgconn.PgConn, *pgconn.Notification) { handled.Add(1) }
	})
	store := newTestStore(t, pool, testTable(t, pool), WithPollInterval(0))
	ch := savePending(t, store, "p")
	if _, err := store.SaveSnapshot(context.Background(), "p", func(s *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
		s.Status = aix.SnapshotStatusAborting
		return s, nil
	}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	waitFor(t, ch, func(st aix.SnapshotStatus) bool { return st == aix.SnapshotStatusAborting })
	if handled.Load() == 0 {
		t.Error("the pool's OnNotification handler received nothing, so the test did not cover it")
	}
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
	eventually(t, "the LISTEN connection closes after the last subscription ends", func() bool {
		return len(listenerPIDs(t, pool, store)) == 0
	})
}
