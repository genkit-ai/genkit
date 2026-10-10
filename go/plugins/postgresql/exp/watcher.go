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
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/logger"
)

const (
	// minRetryDelay and maxRetryDelay bound the wait before the watcher tries
	// to LISTEN again after a failure. It polls in the meantime.
	minRetryDelay = 500 * time.Millisecond
	maxRetryDelay = 30 * time.Second
)

// watcher serves the status subscriptions of one store. It runs while the
// store has subscribers: one connection LISTENs on the store's channel and
// delivers each notification to the subscribers of its row, and a poll
// re-reads every watched row as a fallback, after each (re)connect and then
// every poll interval.
type watcher struct {
	pool    *pgxpool.Pool
	channel string
	poll    time.Duration
	read    func(context.Context, []watchKey) (map[watchKey]aix.SnapshotStatus, error)

	mu      sync.Mutex
	watches map[watchKey]*watch
	// clock orders reads and notifications: each takes the next tick as it
	// starts.
	clock uint64
	// stop ends the running loop; nil while no loop runs.
	stop context.CancelFunc
	// ready is closed once the running loop's first LISTEN attempt finished.
	ready chan struct{}
}

// watchKey identifies a watched row.
type watchKey struct{ prefix, id string }

// watch holds the subscribers of one row.
type watch struct {
	subs []*subscriber
	// tick is the clock tick of the newest status delivered to the row's
	// subscribers. A status from a read that started earlier may be older, so
	// it is not delivered.
	tick uint64
}

// subscriber is one subscription's channel and the status it last received.
type subscriber struct {
	ch   chan aix.SnapshotStatus
	last aix.SnapshotStatus
	seen bool
	// since is the clock's tick when the subscription started. Only a read or
	// notification that starts later resolves the subscription: it delivers
	// a status, or, for a read that finds no row, closes the subscription if it
	// has received nothing. What started earlier may be older than the status
	// when the subscription started.
	since uint64
	// stop releases the hook that ends the subscription with its context.
	stop func() bool
}

// notification is the payload a save sends when it changes a row's status.
type notification struct {
	Prefix string             `json:"p"`
	ID     string             `json:"id"`
	Status aix.SnapshotStatus `json:"s"`
}

func newWatcher(pool *pgxpool.Pool, channel string, poll time.Duration, read func(context.Context, []watchKey) (map[watchKey]aix.SnapshotStatus, error)) *watcher {
	return &watcher{
		pool:    pool,
		channel: channel,
		poll:    poll,
		read:    read,
		watches: make(map[watchKey]*watch),
	}
}

// subscribe registers a subscription to the row key and returns its channel,
// which yields the row's status now and on every change until ctx is
// cancelled. If the row does not exist, the channel is closed at once, or, if
// the first read fails, by the first read that works.
func (w *watcher) subscribe(ctx context.Context, key watchKey) <-chan aix.SnapshotStatus {
	w.mu.Lock()
	sub := &subscriber{ch: make(chan aix.SnapshotStatus, 1), since: w.clock}
	wt := w.watches[key]
	if wt == nil {
		wt = &watch{}
		w.watches[key] = wt
	}
	wt.subs = append(wt.subs, sub)
	sub.stop = context.AfterFunc(ctx, func() { w.remove(key, sub) })
	ready := w.startLocked()
	w.mu.Unlock()

	// Read the row only once the loop listens (or failed to, leaving the
	// poll), so a change committed after the read is delivered too.
	select {
	case <-ready:
		w.refresh(ctx, []watchKey{key})
	case <-ctx.Done():
		w.remove(key, sub)
	}
	return sub.ch
}

// remove ends a subscription and closes its channel.
func (w *watcher) remove(key watchKey, sub *subscriber) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.removeLocked(key, sub)
}

// removeLocked ends a subscription, closes its channel, and stops the loop
// once nothing is watched. Removing a subscription twice is a no-op.
func (w *watcher) removeLocked(key watchKey, sub *subscriber) {
	wt := w.watches[key]
	if wt == nil {
		return
	}
	i := slices.Index(wt.subs, sub)
	if i < 0 {
		return
	}
	wt.subs = slices.Delete(wt.subs, i, i+1)
	sub.stop()
	close(sub.ch)
	if len(wt.subs) == 0 {
		delete(w.watches, key)
	}
	if len(w.watches) == 0 && w.stop != nil {
		w.stop()
		w.stop, w.ready = nil, nil
	}
}

// startLocked starts the loop unless it runs, and returns the channel closed
// once the loop's first LISTEN attempt finished.
func (w *watcher) startLocked() chan struct{} {
	if w.stop == nil {
		ctx, cancel := context.WithCancel(context.Background())
		w.stop, w.ready = cancel, make(chan struct{})
		go w.run(ctx, w.ready)
	}
	return w.ready
}

// nextTick advances the clock and returns its new tick.
func (w *watcher) nextTick() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.clock++
	return w.clock
}

// deliverLocked hands st, from a read or notification that started at tick, to
// every subscriber of wt that started earlier and does not hold it already,
// unless a status from a later start was delivered first.
func (w *watcher) deliverLocked(wt *watch, tick uint64, st aix.SnapshotStatus) {
	if tick <= wt.tick {
		return
	}
	wt.tick = tick
	for _, sub := range wt.subs {
		if tick <= sub.since || (sub.seen && sub.last == st) {
			continue
		}
		sub.last, sub.seen = st, true
		coalesceSend(sub.ch, st)
	}
}

// run LISTENs and delivers notifications until ctx is cancelled. After every
// reconnect it polls once, to deliver what changed while nothing listened, and
// while it cannot LISTEN it polls between attempts. The first connect needs no
// poll: each subscription reads its row once the first attempt finishes.
func (w *watcher) run(ctx context.Context, ready chan struct{}) {
	var once sync.Once
	markReady := func() { once.Do(func() { close(ready) }) }
	defer markReady()

	delay := minRetryDelay
	for first := true; ctx.Err() == nil; first = false {
		conn, err := w.listen(ctx)
		markReady()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			logger.Debug(ctx, "postgresql session store: cannot LISTEN for status changes; polling",
				"channel", w.channel, "error", err)
			w.pollAll(ctx)
			select {
			case <-ctx.Done():
				return
			case <-time.After(delay):
			}
			delay = min(2*delay, maxRetryDelay)
			continue
		}
		delay = minRetryDelay
		if !first {
			w.pollAll(ctx)
		}
		err = w.receive(ctx, conn)
		conn.Close(context.Background())
		if ctx.Err() != nil {
			return
		}
		logger.Debug(ctx, "postgresql session store: lost the LISTEN connection; reconnecting",
			"channel", w.channel, "error", err)
	}
}

// listen takes a connection out of the pool for good and LISTENs on it.
// Hijacking keeps the pool's connect hooks (such as IAM token refresh) and
// leaves the pool free to close without waiting for this connection.
func (w *watcher) listen(ctx context.Context) (*pgx.Conn, error) {
	pooled, err := w.pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	conn := pooled.Hijack()
	if _, err := conn.Exec(ctx, "LISTEN "+pgx.Identifier{w.channel}.Sanitize()); err != nil {
		conn.Close(context.Background())
		return nil, err
	}
	return conn, nil
}

// receive delivers notifications from conn, and polls every poll interval
// whether or not notifications arrive, until ctx is cancelled or the
// connection fails.
func (w *watcher) receive(ctx context.Context, conn *pgx.Conn) error {
	nextPoll := time.Now().Add(w.poll)
	for {
		waitCtx, cancel := ctx, context.CancelFunc(func() {})
		if w.poll > 0 {
			waitCtx, cancel = context.WithDeadline(ctx, nextPoll)
		}
		n, err := conn.WaitForNotification(waitCtx)
		cancel()
		switch {
		case err == nil:
			// With no payload (the pool's OnNotification handler took the
			// notification, or the payload was too large to send), read every
			// watched row instead.
			if n == nil || !w.dispatch(n.Payload) {
				w.pollAll(ctx)
			}
		case ctx.Err() != nil:
			return ctx.Err()
		case errors.Is(err, context.DeadlineExceeded):
			// The poll interval elapsed. The deadline interrupted the wait
			// without closing the connection.
			w.pollAll(ctx)
			nextPoll = time.Now().Add(w.poll)
		default:
			return err
		}
	}
}

// dispatch delivers a notification to the subscribers of its row, and reports
// whether the payload named a row.
func (w *watcher) dispatch(payload string) bool {
	var n notification
	if err := json.Unmarshal([]byte(payload), &n); err != nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.clock++
	if wt := w.watches[watchKey{prefix: n.Prefix, id: n.ID}]; wt != nil {
		w.deliverLocked(wt, w.clock, n.Status)
	}
	return true
}

// pollAll re-reads every watched row; see refresh.
func (w *watcher) pollAll(ctx context.Context) {
	w.mu.Lock()
	keys := slices.Collect(maps.Keys(w.watches))
	w.mu.Unlock()
	if len(keys) > 0 {
		w.refresh(ctx, keys)
	}
}

// refresh reads the rows keys and delivers their statuses. A row that does not
// exist closes the subscriptions to it that started before the read and have
// received nothing, since the row was missing when they were established. A
// subscription that received a status keeps waiting, as it would for a row
// deleted between notifications. A failed read changes nothing.
func (w *watcher) refresh(ctx context.Context, keys []watchKey) {
	tick := w.nextTick()
	found, err := w.read(ctx, keys)
	if err != nil {
		if ctx.Err() == nil {
			logger.Debug(ctx, "postgresql session store: status read failed", "channel", w.channel, "error", err)
		}
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, k := range keys {
		wt := w.watches[k]
		if wt == nil {
			continue
		}
		if st, ok := found[k]; ok {
			w.deliverLocked(wt, tick, st)
			continue
		}
		for _, sub := range slices.Clone(wt.subs) {
			if !sub.seen && sub.since < tick {
				w.removeLocked(k, sub)
			}
		}
	}
}

// coalesceSend delivers st on a size-1 buffered channel so the latest value
// stays observable even if an earlier one is still unread: it drops any unread
// value first, then sends. Only the watcher sends, under its lock, so after the
// drain the send always has room.
func coalesceSend(ch chan aix.SnapshotStatus, st aix.SnapshotStatus) {
	select {
	case <-ch:
	default:
	}
	select {
	case ch <- st:
	default:
	}
}
