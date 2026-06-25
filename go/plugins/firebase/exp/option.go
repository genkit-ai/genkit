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
	// DefaultTTL is the default time-to-live for Firestore documents (used by the
	// stream manager).
	DefaultTTL = 5 * time.Minute

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

// --- Shared Firestore options ---

// firestoreOptions holds configuration common to the Firestore-backed services
// in this package (the stream manager and the session store).
type firestoreOptions struct {
	Collection string
	TTL        time.Duration
}

// applyFirestore merges the common Firestore options, rejecting a setting
// applied more than once.
func (o *firestoreOptions) applyFirestore(opts *firestoreOptions) error {
	if o.Collection != "" {
		if opts.Collection != "" {
			return errors.New("cannot set collection more than once (WithCollection)")
		}
		opts.Collection = o.Collection
	}
	if o.TTL > 0 {
		if opts.TTL > 0 {
			return errors.New("cannot set TTL more than once (WithTTL)")
		}
		opts.TTL = o.TTL
	}
	return nil
}

// applyStreamManager implements StreamManagerOption for firestoreOptions.
func (o *firestoreOptions) applyStreamManager(opts *streamManagerOptions) error {
	return o.applyFirestore(&opts.firestoreOptions)
}

// applySessionStore implements FirestoreSessionStoreOption for firestoreOptions.
// Only the collection applies to the session store; TTL is ignored.
func (o *firestoreOptions) applySessionStore(opts *sessionStoreOptions) error {
	if o.Collection != "" {
		if opts.collection != "" {
			return errors.New("cannot set collection more than once (WithCollection)")
		}
		opts.collection = o.Collection
	}
	return nil
}

// WithCollection sets the Firestore collection documents are stored under. For
// the stream manager this is the stream document collection; for the session
// store it is the root snapshot collection (two companion collections,
// "<collection>-shards" and "<collection>-pointers", are derived from it).
// Defaults to "genkit-sessions" for the session store.
func WithCollection(collection string) *firestoreOptions {
	return &firestoreOptions{Collection: collection}
}

// WithTTL sets how long stream documents are retained before Firestore
// auto-deletes them. Requires a TTL policy on the collection for the "expiresAt"
// field. Default is 5 minutes. Applies to the stream manager only.
// See: https://firebase.google.com/docs/firestore/ttl
func WithTTL(ttl time.Duration) *firestoreOptions {
	return &firestoreOptions{TTL: ttl}
}

// --- Session store options ---

// sessionStoreOptions is the resolved configuration for a
// [FirestoreSessionStore].
type sessionStoreOptions struct {
	collection         string
	checkpointInterval int
	shardSize          int
	prefixFn           func(context.Context) string
}

// FirestoreSessionStoreOption configures a [FirestoreSessionStore]. It is
// implemented by firestoreOptions ([WithCollection]) and sessionStoreOptions
// ([WithCheckpointInterval], [WithShardSize], [WithSnapshotPathPrefix]).
type FirestoreSessionStoreOption interface {
	applySessionStore(*sessionStoreOptions) error
}

// applySessionStore merges the session-store options, validating values and
// rejecting a setting applied more than once. A zero value means "unset" (the
// constructor applies the default), matching the stream manager's option style.
func (o *sessionStoreOptions) applySessionStore(opts *sessionStoreOptions) error {
	if o.collection != "" {
		if opts.collection != "" {
			return errors.New("cannot set collection more than once (WithCollection)")
		}
		opts.collection = o.collection
	}
	if o.checkpointInterval != 0 {
		if o.checkpointInterval < 1 {
			return errors.New("checkpoint interval must be at least 1 (WithCheckpointInterval)")
		}
		if opts.checkpointInterval != 0 {
			return errors.New("cannot set checkpoint interval more than once (WithCheckpointInterval)")
		}
		opts.checkpointInterval = o.checkpointInterval
	}
	if o.shardSize != 0 {
		if o.shardSize < 0 {
			return errors.New("shard size must be positive (WithShardSize)")
		}
		if opts.shardSize != 0 {
			return errors.New("cannot set shard size more than once (WithShardSize)")
		}
		opts.shardSize = o.shardSize
	}
	if o.prefixFn != nil {
		if opts.prefixFn != nil {
			return errors.New("cannot set snapshot path prefix more than once (WithSnapshotPathPrefix)")
		}
		opts.prefixFn = o.prefixFn
	}
	return nil
}

// WithCheckpointInterval sets the number of turns between full-state
// checkpoints. A larger value stores fewer (but reconstructs over more) diffs; a
// smaller value reconstructs faster at the cost of more frequent full-state
// writes. The number of diff documents read or written per turn is bounded by
// this value rather than by total session length. Must be at least 1; defaults
// to 25.
func WithCheckpointInterval(turns int) *sessionStoreOptions {
	return &sessionStoreOptions{checkpointInterval: turns}
}

// WithShardSize sets the maximum size in bytes of a single shard or diff
// document. Checkpoint state is split into chunks of this size, and any diff
// exceeding it is promoted to a (sharded) checkpoint, so no document approaches
// Firestore's 1 MiB limit. Must be positive; defaults to 512 KiB.
func WithShardSize(bytes int) *sessionStoreOptions {
	return &sessionStoreOptions{shardSize: bytes}
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
// result falls back to the "global" prefix.
func WithSnapshotPathPrefix(fn func(ctx context.Context) string) *sessionStoreOptions {
	return &sessionStoreOptions{prefixFn: fn}
}
