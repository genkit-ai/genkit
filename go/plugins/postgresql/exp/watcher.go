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
	// gen counts the deliveries to the row's subscribers. A read that started
	// before a delivery may hold an older status than the one delivered, so a
	// read's result is delivered only if gen did not move while it ran.
	gen uint64
}

// subscriber is one subscription's channel and the status it last received.
type subscriber struct {
	ch   chan aix.SnapshotStatus
	last aix.SnapshotStatus
	seen bool
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
// cancelled. If the row does not exist, the channel is closed at once.
func (w *watcher) subscribe(ctx context.Context, key watchKey) <-chan aix.SnapshotStatus {
	sub := &subscriber{ch: make(chan aix.SnapshotStatus, 1)}
	w.mu.Lock()
	wt := w.watches[key]
	if wt == nil {
		wt = &watch{}
		w.watches[key] = wt
	}
	wt.subs = append(wt.subs, sub)
	ready := w.startLocked()
	gen := wt.gen
	w.mu.Unlock()

	// Read the row only once the loop listens (or failed to, leaving the
	// poll), so a change committed after the read is delivered too.
	select {
	case <-ready:
	case <-ctx.Done():
		w.remove(key, sub)
		return sub.ch
	}
	found, err := w.read(ctx, []watchKey{key})

	w.mu.Lock()
	switch st, ok := found[key]; {
	case err != nil:
		// Keep the subscription: the poll delivers the status once reads work.
	case wt.gen != gen:
		// A notification or the poll delivered a status at least as new.
	case ok:
		w.deliverLocked(wt, st)
	case !sub.seen:
		// The row did not exist when the subscription was established.
		w.removeLocked(key, sub)
		w.mu.Unlock()
		return sub.ch
	}
	w.mu.Unlock()
	context.AfterFunc(ctx, func() { w.remove(key, sub) })
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

// deliverLocked hands st to every subscriber of wt that does not hold it
// already.
func (w *watcher) deliverLocked(wt *watch, st aix.SnapshotStatus) {
	wt.gen++
	for _, sub := range wt.subs {
		if sub.seen && sub.last == st {
			continue
		}
		sub.last, sub.seen = st, true
		coalesceSend(sub.ch, st)
	}
}

// run LISTENs and delivers notifications until ctx is cancelled. After every
// (re)connect it polls once, to deliver what changed while nothing listened,
// and while it cannot LISTEN it polls between attempts.
func (w *watcher) run(ctx context.Context, ready chan struct{}) {
	var once sync.Once
	markReady := func() { once.Do(func() { close(ready) }) }
	defer markReady()

	delay := minRetryDelay
	for ctx.Err() == nil {
		conn, err := w.listen(ctx)
		markReady()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			logger.Debug(ctx, "postgresql session store: cannot LISTEN for status changes; polling",
				"channel", w.channel, "error", err)
			w.pollAll(ctx)
			if !sleep(ctx, delay) {
				return
			}
			delay = min(2*delay, maxRetryDelay)
			continue
		}
		delay = minRetryDelay
		w.pollAll(ctx)
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
		// cancel sets waitCtx's error too, so read whether the deadline passed
		// first.
		pollDue := waitCtx.Err() != nil
		cancel()
		switch {
		case err == nil && n == nil:
			// The pool's OnNotification handler took the notification, so pgx
			// returns no payload: read every watched row instead.
			w.pollAll(ctx)
		case err == nil:
			w.dispatch(n.Payload)
		case ctx.Err() != nil:
			return ctx.Err()
		case pollDue:
			// The poll interval elapsed. The deadline interrupted the wait
			// without closing the connection.
			w.pollAll(ctx)
			nextPoll = time.Now().Add(w.poll)
		default:
			return err
		}
	}
}

// dispatch delivers a notification to the subscribers of its row.
func (w *watcher) dispatch(payload string) {
	var n notification
	if err := json.Unmarshal([]byte(payload), &n); err != nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if wt := w.watches[watchKey{prefix: n.Prefix, id: n.ID}]; wt != nil {
		w.deliverLocked(wt, n.Status)
	}
}

// pollAll re-reads every watched row and delivers its status. A row that is
// gone is skipped: its subscribers keep waiting, as they would for a row
// deleted between notifications.
func (w *watcher) pollAll(ctx context.Context) {
	w.mu.Lock()
	keys := make([]watchKey, 0, len(w.watches))
	gens := make(map[watchKey]uint64, len(w.watches))
	for k, wt := range w.watches {
		keys = append(keys, k)
		gens[k] = wt.gen
	}
	w.mu.Unlock()
	if len(keys) == 0 {
		return
	}
	found, err := w.read(ctx, keys)
	if err != nil {
		if ctx.Err() == nil {
			logger.Debug(ctx, "postgresql session store: status poll failed", "channel", w.channel, "error", err)
		}
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	for k, st := range found {
		if wt := w.watches[k]; wt != nil && wt.gen == gens[k] {
			w.deliverLocked(wt, st)
		}
	}
}

// sleep waits for d or until ctx is cancelled, and reports whether d elapsed.
func sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
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
