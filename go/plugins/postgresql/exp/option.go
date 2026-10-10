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
	"time"
)

const (
	// defaultTableName is the table session snapshots live in.
	defaultTableName = "genkit_session_snapshots"
	// defaultCheckpointInterval is the number of turns between full-state
	// checkpoints. It matches the Firestore store: per-turn state is small in
	// the common chat workload, and a read applies at most this many diffs.
	defaultCheckpointInterval = 25
	// maxCheckpointInterval caps WithCheckpointInterval, so a read walks a
	// bounded chain.
	maxCheckpointInterval = 1000
	// defaultPrefix is the tenant prefix used when no [WithSnapshotPathPrefix]
	// is configured.
	defaultPrefix = "global"
	// defaultPollInterval is how often a status subscription re-reads the rows
	// it watches. LISTEN/NOTIFY delivers changes at once; the poll covers a
	// change written without a notification and a connection that cannot
	// LISTEN, such as one through a transaction-mode pooler.
	defaultPollInterval = 5 * time.Second
	// maxTableNameLength leaves room for the index-name suffixes within
	// PostgreSQL's 63-byte identifier limit.
	maxTableNameLength = maxIdentifierLength - len(sessionIndexSuffix)
	// maxIdentifierLength is PostgreSQL's identifier limit in bytes.
	maxIdentifierLength = 63
)

// The options below follow the pattern of the Firebase plugin's options: each
// option carrier implements the apply method of the service it configures, so
// an option passed to the wrong constructor is a compile error, and every
// option rejects an invalid value. The way to request a default is to omit the
// option, not to pass a zero or empty value.

// sessionStoreOptions is the resolved configuration a [PostgresSessionStore]
// is built from. Fields left zero take their default in the constructor.
type sessionStoreOptions struct {
	table              string
	schema             string
	checkpointInterval int
	prefixFn           func(context.Context) string
	pollInterval       *time.Duration
}

// SessionStoreOption configures a [PostgresSessionStore].
type SessionStoreOption interface {
	applySessionStore(*sessionStoreOptions) error
}

// tableNameOption carries [WithTableName].
type tableNameOption struct{ name string }

func (o tableNameOption) applySessionStore(opts *sessionStoreOptions) error {
	if o.name == "" {
		return errors.New("table name must not be empty (WithTableName)")
	}
	if len(o.name) > maxTableNameLength {
		return fmt.Errorf("table name %q is longer than %d bytes (WithTableName)", o.name, maxTableNameLength)
	}
	if opts.table != "" {
		return errors.New("cannot set table name more than once (WithTableName)")
	}
	opts.table = o.name
	return nil
}

// WithTableName sets the table session snapshots are stored in. The store
// creates it, with its indexes, when it does not exist. Defaults to
// "genkit_session_snapshots" when omitted; at most 51 bytes, so the index names
// derived from it fit PostgreSQL's identifier limit.
func WithTableName(name string) SessionStoreOption {
	return tableNameOption{name}
}

// schemaNameOption carries [WithSchemaName].
type schemaNameOption struct{ name string }

func (o schemaNameOption) applySessionStore(opts *sessionStoreOptions) error {
	if o.name == "" {
		return errors.New("schema name must not be empty (WithSchemaName)")
	}
	if len(o.name) > maxIdentifierLength {
		return fmt.Errorf("schema name %q is longer than %d bytes (WithSchemaName)", o.name, maxIdentifierLength)
	}
	if opts.schema != "" {
		return errors.New("cannot set schema name more than once (WithSchemaName)")
	}
	opts.schema = o.name
	return nil
}

// WithSchemaName sets the schema the snapshot table lives in. The schema must
// exist. When omitted, the table name resolves through the connection's
// search_path, the way an unqualified name does in SQL.
func WithSchemaName(name string) SessionStoreOption {
	return schemaNameOption{name}
}

// checkpointIntervalOption carries [WithCheckpointInterval].
type checkpointIntervalOption struct{ turns int }

func (o checkpointIntervalOption) applySessionStore(opts *sessionStoreOptions) error {
	if o.turns < 1 || o.turns > maxCheckpointInterval {
		return fmt.Errorf("checkpoint interval must be between 1 and %d (WithCheckpointInterval)", maxCheckpointInterval)
	}
	if opts.checkpointInterval != 0 {
		return errors.New("cannot set checkpoint interval more than once (WithCheckpointInterval)")
	}
	opts.checkpointInterval = o.turns
	return nil
}

// WithCheckpointInterval sets the number of turns between full-state
// checkpoints. Between checkpoints each row stores only the JSON Patch from its
// parent's state, so a larger value writes less per turn and reads apply more
// patches; a smaller value does the opposite. A read applies at most this many
// patches, however long the session. Must be between 1 and 1000; defaults to
// 25 when omitted.
func WithCheckpointInterval(turns int) SessionStoreOption {
	return checkpointIntervalOption{turns}
}

// snapshotPathPrefixOption carries [WithSnapshotPathPrefix].
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
// context. Every row is keyed by its prefix, so reads, writes, and status
// subscriptions are isolated per tenant: one tenant can never address another's
// snapshots, even holding a snapshot ID, because resolving it still requires
// the matching, auth-derived prefix. A typical fn pulls a stable identity (an
// authenticated user or org ID) out of ctx. The name matches the other session
// stores; here the prefix is a column, not a path.
//
// The value must be stable for a given snapshot's lifetime, since every read
// recomputes it. It must be non-empty: an empty result is rejected at call time,
// since the way to request the default "global" prefix is to omit this option.
func WithSnapshotPathPrefix(fn func(ctx context.Context) string) SessionStoreOption {
	return snapshotPathPrefixOption{fn}
}

// pollIntervalOption carries [WithPollInterval].
type pollIntervalOption struct{ d time.Duration }

func (o pollIntervalOption) applySessionStore(opts *sessionStoreOptions) error {
	if opts.pollInterval != nil {
		return errors.New("cannot set poll interval more than once (WithPollInterval)")
	}
	opts.pollInterval = &o.d
	return nil
}

// WithPollInterval sets how often a status subscription re-reads the rows it
// watches. Status changes arrive at once over LISTEN/NOTIFY; the poll is the
// fallback for a change written without a notification (a manual UPDATE) and
// for a connection that cannot receive one, such as a connection through a
// transaction-mode pooler. Defaults to 5 seconds when omitted; a value <= 0
// turns the poll off.
func WithPollInterval(d time.Duration) SessionStoreOption {
	return pollIntervalOption{d}
}
