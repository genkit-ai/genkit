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
	"github.com/jackc/pgx/v5/pgconn"
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
// store has subscribers: one connection LISTENs on the store's channel, and a
// notification makes it re-read the row it names; a poll re-reads every
// watched row as a fallback, after each reconnect and then every poll
// interval.
//
// Every status a subscriber receives comes from a read, and reads run one at
// a time, each with its delivery, so each read sees the rows at least as new
// as the one before it. A status therefore never goes back to an older one,
// as it could if a notification, which can wait unread on the connection,
// carried the status itself.
type watcher struct {
	pool    *pgxpool.Pool
	channel string
	poll    time.Duration
	read    func(context.Context, []watchKey) (map[watchKey]aix.SnapshotStatus, error)

	// reading holds a token while a read and its delivery run.
	reading chan struct{}

	mu      sync.Mutex
	watches map[watchKey][]*subscriber
	// reads counts the reads started so far.
	reads uint64
	// stop ends the running loop; nil while no loop runs.
	stop context.CancelFunc
	// ready is closed once the running loop's first LISTEN attempt finished.
	ready chan struct{}
}

// watchKey identifies a watched row.
type watchKey struct{ prefix, id string }

// subscriber is one subscription's channel and the status it last received.
type subscriber struct {
	ch   chan aix.SnapshotStatus
	last aix.SnapshotStatus
	seen bool
	// since is the number of reads started when the subscription started. A
	// read in progress then may be older than the row's status at that time,
	// so only a later read resolves the subscription: it delivers the status,
	// or, if it finds no row, closes a subscription that has received nothing.
	since uint64
	// stop releases the hook that ends the subscription with its context.
	stop func() bool
}

// notification is the payload a save sends when it changes a row's status. It
// names the row, and the watcher reads the status.
type notification struct {
	Prefix string `json:"p"`
	ID     string `json:"id"`
}

func newWatcher(pool *pgxpool.Pool, channel string, poll time.Duration, read func(context.Context, []watchKey) (map[watchKey]aix.SnapshotStatus, error)) *watcher {
	return &watcher{
		pool:    pool,
		channel: channel,
		poll:    poll,
		read:    read,
		reading: make(chan struct{}, 1),
		watches: make(map[watchKey][]*subscriber),
	}
}

// subscribe registers a subscription to the row key and returns its channel,
// which yields the row's status now and on every change until ctx is
// cancelled. If the row does not exist, the channel is closed at once, or, if
// the first read fails, by the first read that works.
func (w *watcher) subscribe(ctx context.Context, key watchKey) <-chan aix.SnapshotStatus {
	w.mu.Lock()
	sub := &subscriber{ch: make(chan aix.SnapshotStatus, 1), since: w.reads}
	w.watches[key] = append(w.watches[key], sub)
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
	subs := w.watches[key]
	i := slices.Index(subs, sub)
	if i < 0 {
		return
	}
	sub.stop()
	close(sub.ch)
	if subs = slices.Delete(subs, i, i+1); len(subs) == 0 {
		delete(w.watches, key)
	} else {
		w.watches[key] = subs
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
			w.notified(ctx, n)
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

// notified re-reads the row a notification names, if it is watched. With no
// usable payload (the pool's OnNotification handler took the notification, or
// the payload was too large to send), it re-reads every watched row.
func (w *watcher) notified(ctx context.Context, n *pgconn.Notification) {
	var payload notification
	if n == nil || json.Unmarshal([]byte(n.Payload), &payload) != nil {
		w.pollAll(ctx)
		return
	}
	key := watchKey{prefix: payload.Prefix, id: payload.ID}
	w.mu.Lock()
	_, watched := w.watches[key]
	w.mu.Unlock()
	if watched {
		w.refresh(ctx, []watchKey{key})
	}
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

// refresh reads the rows keys, after any read in progress, and resolves the
// subscriptions that started before it: each receives its row's status, and
// one whose row does not exist closes if it has received nothing, since the
// row was missing when it started. A subscription that received a status
// keeps waiting, as it would for a row deleted between changes. A failed read
// changes nothing.
func (w *watcher) refresh(ctx context.Context, keys []watchKey) {
	select {
	case w.reading <- struct{}{}:
	case <-ctx.Done():
		return
	}
	defer func() { <-w.reading }()
	w.mu.Lock()
	w.reads++
	read := w.reads
	w.mu.Unlock()

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
		st, ok := found[k]
		for _, sub := range slices.Clone(w.watches[k]) {
			switch {
			case sub.since >= read:
				// It started while this read ran; its own read follows.
			case ok:
				if !sub.seen || sub.last != st {
					sub.last, sub.seen = st, true
					coalesceSend(sub.ch, st)
				}
			case !sub.seen:
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
