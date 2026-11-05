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
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	aix "github.com/firebase/genkit/go/ai/exp"
)

// listenerPIDs returns the backends LISTENing on the store's channel.
func listenerPIDs(t *testing.T, store *PostgresSessionStore[testState]) []int32 {
	t.Helper()
	rows, err := store.pool.Query(context.Background(), `SELECT pid FROM pg_stat_activity WHERE query = $1`,
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

// savePending writes a pending row and subscribes to it until ctx is cancelled
// or the test ends, returning the subscription after its first value.
func savePending(t *testing.T, ctx context.Context, store *PostgresSessionStore[testState], id string) <-chan aix.SnapshotStatus {
	t.Helper()
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
func setStatus(t *testing.T, store *PostgresSessionStore[testState], id string, st aix.SnapshotStatus) {
	t.Helper()
	if _, err := store.pool.Exec(context.Background(), fmt.Sprintf(`UPDATE %s SET status = $1 WHERE snapshot_id = $2`, store.table), string(st), id); err != nil {
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
			ch := savePending(t, context.Background(), store, "p")

			pids := listenerPIDs(t, store)
			if len(pids) != 1 {
				t.Fatalf("%d backends LISTEN on the store's channel, want 1", len(pids))
			}
			setStatus(t, store, "p", aix.SnapshotStatusAborting)
			if _, err := pool.Exec(context.Background(), `SELECT pg_terminate_backend($1)`, pids[0]); err != nil {
				t.Fatalf("pg_terminate_backend: %v", err)
			}
			waitFor(t, ch, func(st aix.SnapshotStatus) bool { return st == aix.SnapshotStatusAborting })
			eventually(t, "the watcher LISTENs on a new connection", func() bool {
				now := listenerPIDs(t, store)
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
	ch := savePending(t, context.Background(), store, "p")
	setStatus(t, store, "p", aix.SnapshotStatusAborting)
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
	ch := savePending(t, context.Background(), store, "p")
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

// TestWatcherReadsOnAnEmptyNotification checks that a status change whose
// notification payload is too large to send still arrives: the store sends an
// empty notification, and the watcher re-reads its rows.
func TestWatcherReadsOnAnEmptyNotification(t *testing.T) {
	pool := testPool(t)
	store := newTestStore(t, pool, testTable(t, pool), WithPollInterval(0))
	// The ID compresses well, so the row fits the index, but the payload that
	// names it does not fit a notification.
	id := strings.Repeat("x", maxNotifyPayload)
	ch := savePending(t, context.Background(), store, id)
	if _, err := store.SaveSnapshot(context.Background(), id, func(s *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
		s.Status = aix.SnapshotStatusAborting
		return s, nil
	}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	waitFor(t, ch, func(st aix.SnapshotStatus) bool { return st == aix.SnapshotStatusAborting })
}

// TestWatcherReleasesItsConnection checks that the LISTEN connection closes
// once the last subscription ends, so an idle store holds no connection.
func TestWatcherReleasesItsConnection(t *testing.T) {
	pool := testPool(t)
	store := newTestStore(t, pool, testTable(t, pool))
	ctx, cancel := context.WithCancel(context.Background())
	savePending(t, ctx, store, "p")
	if n := len(listenerPIDs(t, store)); n != 1 {
		t.Fatalf("%d backends LISTEN while subscribed, want 1", n)
	}
	cancel()
	eventually(t, "the LISTEN connection closes after the last subscription ends", func() bool {
		return len(listenerPIDs(t, store)) == 0
	})
}

// newDrivenWatcher returns a watcher whose LISTEN loop never runs, so a test
// drives its reads and notifications itself. read serves every read.
func newDrivenWatcher(read func(context.Context, []watchKey) (map[watchKey]aix.SnapshotStatus, error)) *watcher {
	w := newWatcher(nil, "test", 0, read)
	ready := make(chan struct{})
	close(ready)
	// A loop seems to run, so subscribe starts none.
	w.stop, w.ready = func() {}, ready
	return w
}

// fakeRows serves a driven watcher's reads from a map of row statuses, or
// fails them with err. A read whose context carries a gate reports what it
// found when it started, once the gate opens. It records the most reads that
// ran at once.
type fakeRows struct {
	mu      sync.Mutex
	rows    map[watchKey]aix.SnapshotStatus
	err     error
	running int
	most    int
}

func (f *fakeRows) set(k watchKey, st aix.SnapshotStatus) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.rows[k] = st
}

func (f *fakeRows) fail(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.err = err
}

func (f *fakeRows) read(ctx context.Context, keys []watchKey) (map[watchKey]aix.SnapshotStatus, error) {
	f.mu.Lock()
	f.running++
	f.most = max(f.most, f.running)
	found, err := make(map[watchKey]aix.SnapshotStatus), f.err
	for _, k := range keys {
		if st, ok := f.rows[k]; ok {
			found[k] = st
		}
	}
	f.mu.Unlock()
	defer func() {
		f.mu.Lock()
		f.running--
		f.mu.Unlock()
	}()
	if g, ok := ctx.Value(gateKey{}).(*gate); ok {
		close(g.started)
		<-g.release
	}
	if err != nil {
		return nil, err
	}
	return found, nil
}

// holdPoll starts a poll of w whose read sees the rows as they are now, and
// returns once that read started. The poll delivers what it read when
// release is called, which returns once the poll finished.
func holdPoll(ctx context.Context, w *watcher) (release func()) {
	g := newGate()
	polled := make(chan struct{})
	go func() {
		w.pollAll(g.on(ctx))
		close(polled)
	}()
	<-g.started
	return func() {
		close(g.release)
		<-polled
	}
}

type gateKey struct{}

// gate holds a read between its start and its result.
type gate struct{ started, release chan struct{} }

func newGate() *gate { return &gate{started: make(chan struct{}), release: make(chan struct{})} }

// on returns ctx carrying g, for the read made with it.
func (g *gate) on(ctx context.Context) context.Context { return context.WithValue(ctx, gateKey{}, g) }

// subscribeDuring subscribes to key in the background while a read holds w,
// and returns once the subscription is registered. The returned channel yields
// the subscription's channel once subscribe returns.
func subscribeDuring(t *testing.T, ctx context.Context, w *watcher, key watchKey) <-chan (<-chan aix.SnapshotStatus) {
	t.Helper()
	w.mu.Lock()
	before := len(w.watches[key])
	w.mu.Unlock()
	subscribed := make(chan (<-chan aix.SnapshotStatus), 1)
	go func() { subscribed <- w.subscribe(ctx, key) }()
	eventually(t, "the subscription registers", func() bool {
		w.mu.Lock()
		defer w.mu.Unlock()
		return len(w.watches[key]) > before
	})
	return subscribed
}

// isClosed reports whether ch is closed, without waiting, and fails t if ch
// holds a status.
func isClosed(t *testing.T, ch <-chan aix.SnapshotStatus) bool {
	t.Helper()
	select {
	case st, open := <-ch:
		if open {
			t.Fatalf("the subscription yielded %q, want none", st)
		}
		return true
	default:
		return false
	}
}

// TestWatcherNeverGoesBackToAnOlderStatus checks that a notification that
// arrives after a newer read, such as an abort's that waited unread on the
// connection while a read saw the run finish, does not bring back the older
// status: the watcher re-reads the row instead of trusting the notification.
func TestWatcherNeverGoesBackToAnOlderStatus(t *testing.T) {
	ctx := t.Context()
	key := watchKey{prefix: "p", id: "row"}
	rows := &fakeRows{rows: map[watchKey]aix.SnapshotStatus{key: aix.SnapshotStatusAborted}}
	w := newDrivenWatcher(rows.read)
	ch := w.subscribe(ctx, key)
	<-ch

	w.notified(ctx, &pgconn.Notification{Payload: `{"p":"p","id":"row","s":"aborting"}`})
	select {
	case st := <-ch:
		t.Errorf("the subscription got %q after aborted, want no change", st)
	default:
	}
}

// TestWatcherRunsOneReadAtATime checks that reads never overlap, whatever
// starts them, so each read sees the rows at least as new as the one before.
func TestWatcherRunsOneReadAtATime(t *testing.T) {
	ctx := t.Context()
	rows := &fakeRows{rows: map[watchKey]aix.SnapshotStatus{}}
	w := newDrivenWatcher(func(ctx context.Context, keys []watchKey) (map[watchKey]aix.SnapshotStatus, error) {
		found, err := rows.read(ctx, keys)
		time.Sleep(time.Millisecond) // widen the window for an overlap
		return found, err
	})
	var wg sync.WaitGroup
	for i := range 4 {
		key := watchKey{prefix: "p", id: fmt.Sprintf("row-%d", i)}
		rows.set(key, aix.SnapshotStatusPending)
		wg.Go(func() { w.subscribe(ctx, key) })
		wg.Go(func() { w.pollAll(ctx) })
		wg.Go(func() { w.notified(ctx, &pgconn.Notification{Payload: fmt.Sprintf(`{"p":"p","id":%q}`, key.id)}) })
	}
	wg.Wait()
	if rows.most != 1 {
		t.Errorf("%d reads ran at once, want 1", rows.most)
	}
}

// TestWatcherClosesASubscriptionToAMissingRow checks that a subscription to a
// row that does not exist closes even when its first read fails: the first
// read that works closes it, if that read started after the subscription.
func TestWatcherClosesASubscriptionToAMissingRow(t *testing.T) {
	ctx := t.Context()
	key := watchKey{prefix: "p", id: "row"}
	rows := &fakeRows{rows: map[watchKey]aix.SnapshotStatus{}}
	w := newDrivenWatcher(rows.read)
	refused := errors.New("connection refused")

	rows.fail(refused)
	early := w.subscribe(ctx, key)
	rows.fail(nil)
	// A poll starts while the row is missing. Then the row is created, and a
	// second subscription starts, whose first read fails.
	releasePoll := holdPoll(ctx, w)
	rows.set(key, aix.SnapshotStatusPending)
	rows.fail(refused)
	subscribed := subscribeDuring(t, ctx, w, key)
	releasePoll()
	late := <-subscribed

	if !isClosed(t, early) {
		t.Error("the subscription made while the row was missing is open after a read found no row")
	}
	if isClosed(t, late) {
		t.Error("a read that started before the row existed closed a subscription made after")
	}
}

// TestWatcherSkipsReadsOlderThanASubscription checks that a subscription gets
// no status from a read that started before it: here the row changed between
// that read and the subscription, whose own first read fails.
func TestWatcherSkipsReadsOlderThanASubscription(t *testing.T) {
	ctx := t.Context()
	key := watchKey{prefix: "p", id: "row"}
	rows := &fakeRows{rows: map[watchKey]aix.SnapshotStatus{key: aix.SnapshotStatusPending}}
	w := newDrivenWatcher(rows.read)
	<-w.subscribe(ctx, key)

	releasePoll := holdPoll(ctx, w)
	rows.set(key, aix.SnapshotStatusAborting)
	rows.fail(errors.New("connection refused"))
	subscribed := subscribeDuring(t, ctx, w, key)
	releasePoll()
	late := <-subscribed

	select {
	case st := <-late:
		t.Errorf("the subscription got %q from a read that started before it, want no status yet", st)
	default:
	}
}
