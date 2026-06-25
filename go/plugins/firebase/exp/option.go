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
	"time"
)

const (
	// DefaultTTL is the default time-to-live for stream documents.
	DefaultTTL = 5 * time.Minute
	// defaultTimeout is how long a stream subscriber waits for new events by default.
	defaultTimeout = 60 * time.Second

	// defaultCollection is the root collection session snapshots live under.
	defaultCollection = "genkit-sessions"
	// defaultCheckpointInterval is the number of turns between full-state
	// checkpoints. Chosen to favor the common chat workload, where per-turn state
	// is small and read cost dominates; raise it for large per-turn state retained
	// long-term, lower it for small-state, read-heavy sessions.
	defaultCheckpointInterval = 25
	// defaultShardSize is the maximum size in bytes of a single shard or diff
	// document. Kept well under Firestore's 1 MiB per-document limit so no single
	// write can be rejected for being too large.
	defaultShardSize = 512 * 1024
	// defaultPrefix is the tenant prefix used when no [WithSnapshotPathPrefix] is
	// configured.
	defaultPrefix = "global"
)

// The options below follow the compile-time-checked pattern used by
// [github.com/firebase/genkit/go/ai] options: each option carrier implements only
// the apply method(s) for the services it is valid for, so passing an option to
// the wrong constructor is a compile error (e.g. WithTTL is a stream-manager-only
// option and cannot be passed to [NewFirestoreSessionStore]). Every option also
// rejects an invalid value: the way to request a default is to omit the option,
// not to pass a zero or empty value.

// --- Resolved option accumulators ---

// streamManagerOptions is the resolved configuration a [FirestoreStreamManager]
// is built from. Fields left zero take their default in the constructor.
type streamManagerOptions struct {
	collection string
	timeout    time.Duration
	ttl        time.Duration
}

// sessionStoreOptions is the resolved configuration a [FirestoreSessionStore] is
// built from. Fields left zero take their default in the constructor.
type sessionStoreOptions struct {
	collection         string
	checkpointInterval int
	shardSize          int
	prefixFn           func(context.Context) string
}

// --- Option interfaces ---

// StreamManagerOption configures a [FirestoreStreamManager].
type StreamManagerOption interface {
	applyStreamManager(*streamManagerOptions) error
}

// SessionStoreOption configures a [FirestoreSessionStore].
type SessionStoreOption interface {
	applySessionStore(*sessionStoreOptions) error
}

// CollectionOption is an option valid for both Firestore services in this
// package (the stream manager and the session store). Only [WithCollection]
// returns one.
type CollectionOption interface {
	StreamManagerOption
	SessionStoreOption
}

// --- Shared options ---

// collectionOption carries [WithCollection]. It applies to both services, so it
// implements both apply methods.
type collectionOption struct{ collection string }

func (o collectionOption) set(dst *string) error {
	if o.collection == "" {
		return errors.New("collection name must not be empty (WithCollection)")
	}
	if *dst != "" {
		return errors.New("cannot set collection more than once (WithCollection)")
	}
	*dst = o.collection
	return nil
}

func (o collectionOption) applyStreamManager(opts *streamManagerOptions) error {
	return o.set(&opts.collection)
}

func (o collectionOption) applySessionStore(opts *sessionStoreOptions) error {
	return o.set(&opts.collection)
}

// WithCollection sets the Firestore collection documents are stored under. For
// the stream manager this is the stream document collection (required); for the
// session store it is the root snapshot collection (two companion collections,
// "<collection>-shards" and "<collection>-pointers", are derived from it, and it
// defaults to "genkit-sessions" when omitted).
func WithCollection(collection string) CollectionOption {
	return collectionOption{collection}
}

// --- Stream manager options ---

// timeoutOption carries [WithTimeout]; it is a stream-manager-only option.
type timeoutOption struct{ timeout time.Duration }

func (o timeoutOption) applyStreamManager(opts *streamManagerOptions) error {
	if o.timeout <= 0 {
		return errors.New("timeout must be positive (WithTimeout)")
	}
	if opts.timeout != 0 {
		return errors.New("cannot set timeout more than once (WithTimeout)")
	}
	opts.timeout = o.timeout
	return nil
}

// WithTimeout sets how long a subscriber waits for new events before giving up.
// If no activity occurs within this duration, subscribers receive a
// DEADLINE_EXCEEDED error. Defaults to 60 seconds when omitted.
func WithTimeout(timeout time.Duration) StreamManagerOption {
	return timeoutOption{timeout}
}

// ttlOption carries [WithTTL]; it is a stream-manager-only option.
type ttlOption struct{ ttl time.Duration }

func (o ttlOption) applyStreamManager(opts *streamManagerOptions) error {
	if o.ttl <= 0 {
		return errors.New("TTL must be positive (WithTTL)")
	}
	if opts.ttl != 0 {
		return errors.New("cannot set TTL more than once (WithTTL)")
	}
	opts.ttl = o.ttl
	return nil
}

// WithTTL sets how long stream documents are retained before Firestore
// auto-deletes them. Requires a TTL policy on the collection for the "expiresAt"
// field. Defaults to 5 minutes when omitted.
// See: https://firebase.google.com/docs/firestore/ttl
func WithTTL(ttl time.Duration) StreamManagerOption {
	return ttlOption{ttl}
}

// --- Session store options ---

// checkpointIntervalOption carries [WithCheckpointInterval]; a session-store-only option.
type checkpointIntervalOption struct{ turns int }

func (o checkpointIntervalOption) applySessionStore(opts *sessionStoreOptions) error {
	if o.turns < 1 {
		return errors.New("checkpoint interval must be at least 1 (WithCheckpointInterval)")
	}
	if opts.checkpointInterval != 0 {
		return errors.New("cannot set checkpoint interval more than once (WithCheckpointInterval)")
	}
	opts.checkpointInterval = o.turns
	return nil
}

// WithCheckpointInterval sets the number of turns between full-state
// checkpoints. A larger value stores fewer (but reconstructs over more) diffs; a
// smaller value reconstructs faster at the cost of more frequent full-state
// writes. The number of diff documents read or written per turn is bounded by
// this value rather than by total session length. Must be at least 1; defaults
// to 25 when omitted.
func WithCheckpointInterval(turns int) SessionStoreOption {
	return checkpointIntervalOption{turns}
}

// shardSizeOption carries [WithShardSize]; a session-store-only option.
type shardSizeOption struct{ bytes int }

func (o shardSizeOption) applySessionStore(opts *sessionStoreOptions) error {
	if o.bytes < 1 {
		return errors.New("shard size must be at least 1 byte (WithShardSize)")
	}
	if opts.shardSize != 0 {
		return errors.New("cannot set shard size more than once (WithShardSize)")
	}
	opts.shardSize = o.bytes
	return nil
}

// WithShardSize sets the maximum size in bytes of a single shard or diff
// document. Checkpoint state is split into chunks of this size, and any diff
// exceeding it is promoted to a (sharded) checkpoint, so no document approaches
// Firestore's 1 MiB limit. Must be positive; defaults to 512 KiB when omitted.
func WithShardSize(bytes int) SessionStoreOption {
	return shardSizeOption{bytes}
}

// snapshotPathPrefixOption carries [WithSnapshotPathPrefix]; a session-store-only option.
type snapshotPathPrefixOption struct {
	fn func(context.Context) string
}

func (o snapshotPathPrefixOption) applySessionStore(opts *sessionStoreOptions) error {
	if o.fn == nil {
		return errors.New("snapshot path prefix function must not be nil (WithSnapshotPathPrefix)")
	}
	if opts.prefixFn != nil {
		return errors.New("cannot set snapshot path prefix more than once (WithSnapshotPathPrefix)")
	}
	opts.prefixFn = o.fn
	return nil
}

// WithSnapshotPathPrefix derives a per-call tenant prefix from the operation's
// context. When set, all snapshot, shard, and pointer documents are nested under
// a tenant-scoped subcollection keyed by this prefix, so reads and writes are
// isolated per tenant: one tenant can never address another's snapshots, even
// holding a snapshot ID, because resolving it still requires the matching,
// auth-derived prefix. A typical fn pulls a stable identity (e.g. an
// authenticated user or org ID) out of ctx.
//
// The value must be a valid Firestore document ID (no "/" separators) and stable
// for a given snapshot's lifetime, since every read recomputes it. An empty
// result falls back to the "global" prefix. Omitting the option entirely uses
// the "global" prefix for every call.
func WithSnapshotPathPrefix(fn func(ctx context.Context) string) SessionStoreOption {
	return snapshotPathPrefixOption{fn}
}
