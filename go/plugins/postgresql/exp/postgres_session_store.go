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

// Package exp provides experimental PostgreSQL integrations for Genkit's agent
// runtime (see [github.com/firebase/genkit/go/ai/exp]).
//
// The [PostgresSessionStore] persists agent session snapshots in PostgreSQL. It
// resolves its connection pool from the PostgreSQL plugin registered with the
// Genkit instance, then wires into an agent:
//
//	engine, err := postgresql.NewPostgresEngine(ctx,
//		postgresql.WithCloudSQLInstance("my-project", "us-central1", "my-instance"),
//		postgresql.WithDatabase("my-database"))
//	// handle err; postgresql.WithPool(pool) connects to any other PostgreSQL
//	g := genkit.Init(ctx,
//		genkit.WithExperimental(),
//		genkit.WithPlugins(&postgresql.Postgres{Engine: engine}))
//
//	store, err := exp.NewPostgresSessionStore[MyState](ctx, g)
//	// handle err
//
//	agent := genkitx.DefineAgent(g, "assistant",
//		aix.InlinePrompt{ai.WithModelName("googleai/gemini-flash-latest")},
//		aix.WithSessionStore(store))
//
// APIs in this package are under active development and may change in any minor
// version release. Use with caution in production environments.
package exp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/postgresql"
)

// Values of the kind column.
const (
	kindCheckpoint = "checkpoint"
	kindDiff       = "diff"
)

const (
	// pluginName is the name the PostgreSQL plugin registers under.
	pluginName = "postgres"
	// sessionIndexSuffix and parentIndexSuffix name the table's indexes after
	// the table.
	sessionIndexSuffix = "_session_idx"
	parentIndexSuffix  = "_parent_idx"
	// maxChainHops bounds the walk from a row to its checkpoint. A chain the
	// store writes is shorter than the checkpoint interval, which is at most
	// maxCheckpointInterval, so the bound only stops a walk through corrupted
	// rows, such as a cycle a manual UPDATE made, early.
	maxChainHops = maxCheckpointInterval
	// maxNotifyPayload keeps a notification under PostgreSQL's 8000-byte
	// payload limit. A larger one goes empty, and every watcher then re-reads
	// the rows it watches.
	maxNotifyPayload = 7900
	// layoutComment marks a table this store created, and the version of its
	// layout, so a later version can recognize and migrate it.
	layoutComment = "Genkit agent session snapshots, layout 1 (github.com/firebase/genkit/go/plugins/postgresql/exp)"
)

// txOptions runs the store's transactions at READ COMMITTED, whatever the
// database's default. The locks rely on it: each statement after a lock wait
// reads what committed during the wait. At a stricter level, a save that
// waited for another reads the row as it was before the wait, and fails.
var txOptions = pgx.TxOptions{IsoLevel: pgx.ReadCommitted}

// Compile-time checks that the store has the capabilities it documents.
var (
	_ aix.SessionStore[any]           = (*PostgresSessionStore[any])(nil)
	_ aix.SnapshotMetadataReader[any] = (*PostgresSessionStore[any])(nil)
	_ aix.SnapshotSubscriber          = (*PostgresSessionStore[any])(nil)
)

// PostgresSessionStore is a PostgreSQL-backed [aix.SessionStore] that persists
// session snapshots as JSON Patch diffs anchored to periodic full-state
// checkpoints, like the Firestore store, in one table:
//
//   - A checkpoint row holds the full state in its state column.
//   - A diff row holds the JSON Patch from its parent's state in its
//     state_patch column, and its depth: the number of diffs back to the
//     nearest checkpoint.
//   - A row without state (a pending detached run) is a checkpoint with a NULL
//     state.
//
// Every N turns (see [WithCheckpointInterval]) a row is written as a
// checkpoint, so a read applies at most N patches and each turn writes only its
// change, however long the session grows. A diff larger than half of the full
// state is written as a checkpoint instead. Large values need no sharding:
// PostgreSQL compresses them and stores them out of line, up to 1 GB each.
//
// Rows are keyed by (prefix, snapshot ID), where the prefix is the tenant key
// from [WithSnapshotPathPrefix], or "global". An index on (prefix, session ID,
// CreatedAt, snapshot ID) answers a session's latest row in one lookup, and a
// read walks from a row to its checkpoint in one recursive query, so it sees
// the whole chain at one point in time. State and patches are stored as json,
// which keeps the exact text the store wrote: jsonb would reject text that
// contains U+0000, which tool output can carry.
//
// SaveSnapshot runs in one READ COMMITTED transaction, whatever the database's
// default isolation level, that first takes an advisory lock on the snapshot
// ID, so concurrent saves of one snapshot wait for each other and fn runs once
// per save. It rejects, with FAILED_PRECONDITION, a change to the
// state of a row that another row stores a diff against, since the change
// would corrupt that row (see [aix.SnapshotWriter]).
//
// It implements [aix.SnapshotSubscriber] over LISTEN/NOTIFY: a save that
// changes the status of an existing row notifies, at commit, every store
// watching that row, so an abort committed by one process reaches the process
// running the detached turn at once. Only status changes notify, since
// PostgreSQL serializes the commits of transactions that do. Each store holds
// one connection for this while it has subscribers, and a poll re-reads the
// watched rows as a fallback (see [WithPollInterval]).
//
// The store creates its table and indexes when the table does not exist, which
// needs the CREATE privilege on the schema. Where the schema is managed
// elsewhere, construct the store once with a role that has it; from then on
// the store needs only SELECT, INSERT, and UPDATE on the table.
type PostgresSessionStore[State any] struct {
	pool               *pgxpool.Pool
	table              string // quoted, and qualified when a schema is set
	tableID            string // the table's OID, which names its locks and channel
	checkpointInterval int
	prefixFn           func(context.Context) string
	watcher            *watcher
}

// NewPostgresSessionStore creates a PostgreSQL-backed snapshot store. It
// resolves the connection pool from the PostgreSQL plugin registered with g
// (pass the plugin to genkit.Init before calling this), and creates the
// snapshot table when it does not exist.
//
// The State type parameter is the user-defined custom-state type carried in
// [aix.SessionState.Custom]; it must be JSON-serializable.
func NewPostgresSessionStore[State any](ctx context.Context, g *genkit.Genkit, opts ...SessionStoreOption) (*PostgresSessionStore[State], error) {
	pool, err := resolvePool(g)
	if err != nil {
		return nil, fmt.Errorf("postgresql.NewPostgresSessionStore: %w", err)
	}
	store, err := newPostgresSessionStore[State](ctx, pool, opts...)
	if err != nil {
		return nil, fmt.Errorf("postgresql.NewPostgresSessionStore: %w", err)
	}
	return store, nil
}

// newPostgresSessionStore builds the store on a resolved pool. It is separated
// from the public constructor so the store logic can be exercised against a
// database without standing up the plugin.
func newPostgresSessionStore[State any](ctx context.Context, pool *pgxpool.Pool, opts ...SessionStoreOption) (*PostgresSessionStore[State], error) {
	if pool == nil {
		return nil, errors.New("a connection pool is required")
	}
	var cfg sessionStoreOptions
	for _, o := range opts {
		if err := o.applySessionStore(&cfg); err != nil {
			return nil, err
		}
	}
	tableName := cfg.table
	if tableName == "" {
		tableName = defaultTableName
	}
	table := pgx.Identifier{tableName}
	if cfg.schema != "" {
		table = pgx.Identifier{cfg.schema, tableName}
	}
	checkpointInterval := cfg.checkpointInterval
	if checkpointInterval == 0 {
		checkpointInterval = defaultCheckpointInterval
	}
	pollInterval := defaultPollInterval
	if cfg.pollInterval != nil {
		pollInterval = *cfg.pollInterval
	}

	s := &PostgresSessionStore[State]{
		pool:               pool,
		table:              table.Sanitize(),
		checkpointInterval: checkpointInterval,
		prefixFn:           cfg.prefixFn,
	}
	oid, err := s.ensureTable(ctx, tableName)
	if err != nil {
		return nil, err
	}
	s.tableID = strconv.FormatUint(uint64(oid), 10)
	s.watcher = newWatcher(pool, "genkit_snapshots_"+s.tableID, pollInterval, s.readStatuses)
	return s, nil
}

// resolvePool resolves the connection pool from the PostgreSQL plugin
// registered with g, mirroring how the Firebase plugin's features share its
// Firestore client.
func resolvePool(g *genkit.Genkit) (*pgxpool.Pool, error) {
	plugin := genkit.LookupPlugin(g, pluginName)
	if plugin == nil {
		return nil, errors.New("PostgreSQL plugin not found.\n" +
			"  Pass the PostgreSQL plugin to genkit.Init():\n" +
			"    g := genkit.Init(ctx, genkit.WithPlugins(&postgresql.Postgres{Engine: engine}))")
	}
	p, ok := plugin.(*postgresql.Postgres)
	if !ok {
		return nil, fmt.Errorf("unexpected plugin type %T for provider %q", plugin, pluginName)
	}
	if p.Engine == nil || p.Engine.Pool == nil {
		return nil, errors.New("the PostgreSQL plugin has no connection pool; create its Engine with postgresql.NewPostgresEngine")
	}
	return p.Engine.Pool, nil
}

// --- Table ---

// rowColumns are the table's columns in the order the store reads them.
var rowColumns = []string{
	"snapshot_id", "session_id", "parent_id", "created_at", "updated_at", "heartbeat_at",
	"status", "finish_reason", "error", "kind", "depth", "state", "state_patch",
}

// metadataColumns are the columns a metadata read loads: everything but the
// state and the chain bookkeeping.
var metadataColumns = rowColumns[:9]

// Select lists of rowColumns and metadataColumns, and of rowColumns qualified
// by the alias p.
var (
	rowList      = strings.Join(rowColumns, ", ")
	metadataList = strings.Join(metadataColumns, ", ")
	parentList   = "p." + strings.Join(rowColumns, ", p.")
)

// Row selectors for reads, where $2 is the key: a snapshot ID, or a session ID
// for the session's latest row (the greatest CreatedAt, ties broken by the
// greater snapshot ID in byte order).
const (
	bySnapshotID    = `snapshot_id = $2`
	latestOfSession = `session_id = $2 ORDER BY created_at DESC, snapshot_id DESC LIMIT 1`
)

// ensureTable creates the snapshot table and its indexes when the table does
// not exist, checks that an existing table has the columns the store reads,
// and returns the table's OID. Creation runs under an advisory lock, so stores
// starting at once do not race, and a table that exists needs no DDL privilege.
func (s *PostgresSessionStore[State]) ensureTable(ctx context.Context, tableName string) (uint32, error) {
	oid, err := s.tableOID(ctx)
	if err != nil {
		return 0, err
	}
	if oid == 0 {
		quote := func(name string) string { return pgx.Identifier{name}.Sanitize() }
		ddl := []string{
			fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
				prefix        text COLLATE "C" NOT NULL,
				snapshot_id   text COLLATE "C" NOT NULL,
				session_id    text COLLATE "C" NOT NULL,
				parent_id     text COLLATE "C",
				created_at    timestamptz NOT NULL,
				updated_at    timestamptz NOT NULL,
				heartbeat_at  timestamptz,
				status        text NOT NULL,
				finish_reason text,
				error         json,
				kind          text NOT NULL CHECK (kind IN ('checkpoint', 'diff')),
				depth         integer NOT NULL,
				state         json,
				state_patch   json,
				PRIMARY KEY (prefix, snapshot_id)
			)`, s.table),
			// A session's latest row is the first entry of this index.
			fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s ON %s (prefix, session_id, created_at DESC, snapshot_id DESC)`,
				quote(tableName+sessionIndexSuffix), s.table),
			// The rows stored as a diff against a given row, which a change to
			// that row's state checks for.
			fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s ON %s (prefix, parent_id) WHERE kind = 'diff'`,
				quote(tableName+parentIndexSuffix), s.table),
			fmt.Sprintf(`COMMENT ON TABLE %s IS '%s'`, s.table, layoutComment),
		}
		err := pgx.BeginTxFunc(ctx, s.pool, txOptions, func(tx pgx.Tx) error {
			if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock($1)`, hashKey("genkit session store table", s.table)); err != nil {
				return err
			}
			// Another store may have created the table while this one waited
			// for the lock. Its DDL would then fail for a role that has CREATE
			// on the schema but does not own the table: COMMENT needs ownership.
			var exists bool
			if err := tx.QueryRow(ctx, `SELECT to_regclass($1) IS NOT NULL`, s.table).Scan(&exists); err != nil || exists {
				return err
			}
			for _, stmt := range ddl {
				if _, err := tx.Exec(ctx, stmt); err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil {
			return 0, fmt.Errorf("create table %s: %w", s.table, err)
		}
		if oid, err = s.tableOID(ctx); err != nil {
			return 0, err
		}
	}
	if _, err := s.pool.Exec(ctx, fmt.Sprintf(`SELECT %s FROM %s LIMIT 0`, rowList, s.table)); err != nil {
		return 0, fmt.Errorf("table %s exists but does not have the session snapshot layout: %w", s.table, err)
	}
	return oid, nil
}

// tableOID returns the OID of the snapshot table, or 0 when it does not exist.
func (s *PostgresSessionStore[State]) tableOID(ctx context.Context) (uint32, error) {
	var oid *uint32
	if err := s.pool.QueryRow(ctx, `SELECT to_regclass($1)::oid`, s.table).Scan(&oid); err != nil {
		return 0, fmt.Errorf("look up table %s: %w", s.table, err)
	}
	if oid == nil {
		return 0, nil
	}
	return *oid, nil
}

// --- Rows ---

// row is one row of the snapshot table.
type row struct {
	snapshotID   string
	sessionID    string
	parentID     *string
	createdAt    time.Time
	updatedAt    time.Time
	heartbeatAt  *time.Time
	status       string
	finishReason *string
	errJSON      []byte
	kind         string
	depth        int
	state        []byte
	patch        []byte
}

// fields returns the scan targets for rowColumns. Its first
// len(metadataColumns) entries are the targets for metadataColumns.
func (r *row) fields() []any {
	return []any{&r.snapshotID, &r.sessionID, &r.parentID, &r.createdAt, &r.updatedAt, &r.heartbeatAt,
		&r.status, &r.finishReason, &r.errJSON, &r.kind, &r.depth, &r.state, &r.patch}
}

// toSnapshot converts r's metadata into a snapshot carrying state.
func toSnapshot[State any](r *row, state *aix.SessionState[State]) (*aix.SessionSnapshot[State], error) {
	snap := &aix.SessionSnapshot[State]{
		SnapshotID:  r.snapshotID,
		SessionID:   r.sessionID,
		CreatedAt:   r.createdAt,
		UpdatedAt:   r.updatedAt,
		HeartbeatAt: r.heartbeatAt,
		Status:      aix.SnapshotStatus(r.status),
		State:       state,
	}
	if r.parentID != nil {
		snap.ParentID = *r.parentID
	}
	if r.finishReason != nil {
		snap.FinishReason = aix.AgentFinishReason(*r.finishReason)
	}
	if r.errJSON != nil {
		var e status.Error
		if err := json.Unmarshal(r.errJSON, &e); err != nil {
			return nil, fmt.Errorf("decode the error of snapshot %q: %w", r.snapshotID, err)
		}
		snap.Error = &e
	}
	return snap, nil
}

// querier is what the store reads through: the pool, or the transaction of a
// save.
type querier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

// readChain reads the row sel selects by key and the rows back to its
// checkpoint, the row first, in one query. It returns nil when there is no
// such row.
func (s *PostgresSessionStore[State]) readChain(ctx context.Context, q querier, prefix, sel, key string) ([]row, error) {
	rows, err := q.Query(ctx, fmt.Sprintf(`WITH RECURSIVE chain AS (
		(SELECT %[1]s, 0 AS hop FROM %[2]s WHERE prefix = $1 AND %[3]s)
		UNION ALL
		SELECT %[4]s, c.hop + 1 FROM chain c
		JOIN %[2]s p ON p.prefix = $1 AND p.snapshot_id = c.parent_id
		WHERE c.kind = 'diff' AND c.hop < %[5]d
	)
	SELECT %[1]s FROM chain ORDER BY hop`, rowList, s.table, sel, parentList, maxChainHops), prefix, key)
	if err != nil {
		return nil, err
	}
	return pgx.AppendRows([]row(nil), rows, func(cr pgx.CollectableRow) (row, error) {
		var r row
		err := cr.Scan(r.fields()...)
		return r, err
	})
}

// materialize returns the snapshot at the head of chain, with the state its
// checkpoint and diffs add up to. It returns nil for an empty chain.
func materialize[State any](chain []row) (*aix.SessionSnapshot[State], error) {
	if len(chain) == 0 {
		return nil, nil
	}
	head := &chain[0]
	stateJSON := head.state
	if head.kind != kindCheckpoint {
		doc, err := chainState(chain)
		if err != nil {
			return nil, err
		}
		if stateJSON, err = json.Marshal(doc); err != nil {
			return nil, fmt.Errorf("encode the state of snapshot %q: %w", head.snapshotID, err)
		}
	}
	var state *aix.SessionState[State]
	if stateJSON != nil {
		state = new(aix.SessionState[State])
		if err := json.Unmarshal(stateJSON, state); err != nil {
			return nil, fmt.Errorf("decode the state of snapshot %q: %w", head.snapshotID, err)
		}
	}
	return toSnapshot(head, state)
}

// chainState applies the diffs of chain, oldest first, onto the state of the
// checkpoint that ends it, and returns the head's state as a JSON value. The
// diffs apply as one patch, so the state is decoded once however long the
// chain is.
func chainState(chain []row) (any, error) {
	base := &chain[len(chain)-1]
	if base.kind != kindCheckpoint {
		return nil, fmt.Errorf("snapshot %q cannot be read: its chain ends at %q, whose parent is missing", chain[0].snapshotID, base.snapshotID)
	}
	var doc any
	if base.state != nil {
		doc = json.RawMessage(base.state)
	}
	var ops aix.JSONPatch
	for i := len(chain) - 2; i >= 0; i-- {
		var patch aix.JSONPatch
		// Numbers stay json.Number, so a large integer keeps its exact value.
		dec := json.NewDecoder(bytes.NewReader(chain[i].patch))
		dec.UseNumber()
		if err := dec.Decode(&patch); err != nil {
			return nil, fmt.Errorf("decode the patch of snapshot %q: %w", chain[i].snapshotID, err)
		}
		ops = append(ops, patch...)
	}
	doc, err := aix.ApplyPatch(doc, ops)
	if err != nil {
		return nil, fmt.Errorf("apply the patches of snapshot %q: %w", chain[0].snapshotID, err)
	}
	return doc, nil
}

// --- Reads ---

// GetSnapshot retrieves a snapshot by ID. Returns nil if not found. One query
// reads the row and its chain back to the checkpoint, at one point in time.
func (s *PostgresSessionStore[State]) GetSnapshot(ctx context.Context, snapshotID string) (*aix.SessionSnapshot[State], error) {
	if snapshotID == "" {
		return nil, nil
	}
	snap, err := s.read(ctx, bySnapshotID, snapshotID)
	if err != nil {
		return nil, fmt.Errorf("postgresql: PostgresSessionStore.GetSnapshot: %w", err)
	}
	return snap, nil
}

// GetLatestSnapshot returns the session's most recently created snapshot
// regardless of status, per the [aix.SnapshotReader.GetLatestSnapshot]
// contract: the greatest CreatedAt, ties broken by the greater snapshot ID.
func (s *PostgresSessionStore[State]) GetLatestSnapshot(ctx context.Context, sessionID string) (*aix.SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("postgresql: PostgresSessionStore.GetLatestSnapshot: session ID is empty")
	}
	snap, err := s.read(ctx, latestOfSession, sessionID)
	if err != nil {
		return nil, fmt.Errorf("postgresql: PostgresSessionStore.GetLatestSnapshot: %w", err)
	}
	return snap, nil
}

// GetSnapshotMetadata retrieves a snapshot's row without its state, per
// [aix.SnapshotMetadataReader]: one row, and none of its state or chain.
// Returns nil if not found.
func (s *PostgresSessionStore[State]) GetSnapshotMetadata(ctx context.Context, snapshotID string) (*aix.SessionSnapshot[State], error) {
	if snapshotID == "" {
		return nil, nil
	}
	snap, err := s.readMetadata(ctx, bySnapshotID, snapshotID)
	if err != nil {
		return nil, fmt.Errorf("postgresql: PostgresSessionStore.GetSnapshotMetadata: %w", err)
	}
	return snap, nil
}

// GetLatestSnapshotMetadata is [PostgresSessionStore.GetLatestSnapshot]
// without the state, per [aix.SnapshotMetadataReader].
func (s *PostgresSessionStore[State]) GetLatestSnapshotMetadata(ctx context.Context, sessionID string) (*aix.SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("postgresql: PostgresSessionStore.GetLatestSnapshotMetadata: session ID is empty")
	}
	snap, err := s.readMetadata(ctx, latestOfSession, sessionID)
	if err != nil {
		return nil, fmt.Errorf("postgresql: PostgresSessionStore.GetLatestSnapshotMetadata: %w", err)
	}
	return snap, nil
}

// read returns the snapshot sel selects by key, with its state, or nil when
// there is none.
func (s *PostgresSessionStore[State]) read(ctx context.Context, sel, key string) (*aix.SessionSnapshot[State], error) {
	prefix, err := s.prefixFor(ctx)
	if err != nil {
		return nil, err
	}
	chain, err := s.readChain(ctx, s.pool, prefix, sel, key)
	if err != nil {
		return nil, err
	}
	return materialize[State](chain)
}

// readMetadata returns the snapshot sel selects by key, without its state, or
// nil when there is none.
func (s *PostgresSessionStore[State]) readMetadata(ctx context.Context, sel, key string) (*aix.SessionSnapshot[State], error) {
	prefix, err := s.prefixFor(ctx)
	if err != nil {
		return nil, err
	}
	var r row
	err = s.pool.QueryRow(ctx, fmt.Sprintf(`SELECT %s FROM %s WHERE prefix = $1 AND %s`, metadataList, s.table, sel),
		prefix, key).Scan(r.fields()[:len(metadataColumns)]...)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return toSnapshot[State](&r, nil)
}

// --- Writes ---

// SaveSnapshot atomically reads the snapshot at id (if any), applies fn, and
// persists the result. See [aix.SnapshotWriter] for the full contract. The
// read-modify-write runs in one transaction under an advisory lock on the
// snapshot ID, so fn runs once per call and concurrent saves of one snapshot
// apply one after the other.
func (s *PostgresSessionStore[State]) SaveSnapshot(
	ctx context.Context,
	id string,
	fn func(existing *aix.SessionSnapshot[State]) (*aix.SessionSnapshot[State], error),
) (*aix.SessionSnapshot[State], error) {
	if id == "" {
		id = uuid.New().String()
	}
	prefix, err := s.prefixFor(ctx)
	if err != nil {
		return nil, fmt.Errorf("postgresql: PostgresSessionStore.SaveSnapshot: %w", err)
	}
	var persisted *aix.SessionSnapshot[State]
	err = pgx.BeginTxFunc(ctx, s.pool, txOptions, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `SELECT pg_advisory_xact_lock($1)`, hashKey(s.tableID, prefix, id)); err != nil {
			return err
		}
		chain, err := s.readChain(ctx, tx, prefix, bySnapshotID, id)
		if err != nil {
			return err
		}
		current, err := materialize[State](chain)
		if err != nil {
			return err
		}
		// fn may edit the snapshot it is handed and return it, so what the
		// row held is captured before fn runs.
		prev, err := storedOf(chain, current)
		if err != nil {
			return err
		}
		next, err := fn(current)
		if err != nil || next == nil {
			return err // a declined write leaves the row untouched
		}

		// The store owns identity; fn owns the lifecycle timestamps and status.
		next.SnapshotID = id
		if prev != nil && prev.sessionID != "" {
			// A row's session never changes once set.
			next.SessionID = prev.sessionID
		}
		if next.SessionID == "" {
			// A snapshot must belong to a session; stores never mint or infer
			// one. Matches the other session stores.
			return status.Errorf(aix.ErrSessionIDRequired, "PostgresSessionStore requires sessionId to be set on the snapshot")
		}
		if next.Status == "" {
			next.Status = aix.SnapshotStatusCompleted
		}

		if err := s.write(ctx, tx, prefix, prev, next); err != nil {
			return err
		}
		// Nobody can watch a row before it exists, so only a status change of
		// an existing row notifies.
		if prev != nil && prev.status != next.Status {
			if err := s.notify(ctx, tx, prefix, id); err != nil {
				return err
			}
		}
		persisted = next
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("postgresql: PostgresSessionStore.SaveSnapshot: %w", err)
	}
	return persisted, nil
}

// stored is what a row held before fn ran.
type stored struct {
	sessionID string
	parentID  string
	status    aix.SnapshotStatus
	kind      string
	state     []byte // the state as JSON; nil for none
	// parentChain is the parent's chain back to its checkpoint, already read
	// with the row's own when the row is a diff; nil otherwise.
	parentChain []row
}

// storedOf captures the head row of chain, whose materialized snapshot is
// current, or returns nil for a missing row.
func storedOf[State any](chain []row, current *aix.SessionSnapshot[State]) (*stored, error) {
	if current == nil {
		return nil, nil
	}
	state, err := marshalNullable(current.State)
	if err != nil {
		return nil, fmt.Errorf("encode state: %w", err)
	}
	st := &stored{
		sessionID: current.SessionID,
		parentID:  current.ParentID,
		status:    current.Status,
		kind:      chain[0].kind,
		state:     state,
	}
	if st.kind == kindDiff {
		st.parentChain = chain[1:]
	}
	return st, nil
}

// write persists next over the row prev describes (nil for a new row). A write
// that leaves the state, and for a diff the parent, as they were updates the
// metadata alone, so a heartbeat or an abort rewrites no state. Any other write
// plans the row afresh: a diff against its parent, or a checkpoint.
func (s *PostgresSessionStore[State]) write(ctx context.Context, tx pgx.Tx, prefix string, prev *stored, next *aix.SessionSnapshot[State]) error {
	stateJSON, err := marshalNullable(next.State)
	if err != nil {
		return fmt.Errorf("encode state: %w", err)
	}
	errJSON, err := marshalNullable(next.Error)
	if err != nil {
		return fmt.Errorf("encode error: %w", err)
	}
	meta := []any{prefix, next.SnapshotID, next.SessionID, nullable(next.ParentID), next.CreatedAt, next.UpdatedAt,
		next.HeartbeatAt, string(next.Status), nullable(string(next.FinishReason)), errJSON}

	if prev != nil {
		sameState := bytes.Equal(prev.state, stateJSON)
		if sameState && (prev.kind == kindCheckpoint || prev.parentID == next.ParentID) {
			_, err := tx.Exec(ctx, fmt.Sprintf(`UPDATE %s SET session_id = $3, parent_id = $4, created_at = $5,
				updated_at = $6, heartbeat_at = $7, status = $8, finish_reason = $9, error = $10
				WHERE prefix = $1 AND snapshot_id = $2`, s.table), meta...)
			return err
		}
		if !sameState {
			if err := s.checkNoDiffChildren(ctx, tx, prefix, next.SnapshotID); err != nil {
				return err
			}
		}
	}

	var parentChain []row
	if prev != nil && prev.parentID == next.ParentID {
		parentChain = prev.parentChain
	}
	p, err := s.plan(ctx, tx, prefix, next.SnapshotID, next.ParentID, stateJSON, parentChain)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, fmt.Sprintf(`INSERT INTO %s (prefix, snapshot_id, session_id, parent_id, created_at,
			updated_at, heartbeat_at, status, finish_reason, error, kind, depth, state, state_patch)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
		ON CONFLICT (prefix, snapshot_id) DO UPDATE SET session_id = EXCLUDED.session_id,
			parent_id = EXCLUDED.parent_id, created_at = EXCLUDED.created_at, updated_at = EXCLUDED.updated_at,
			heartbeat_at = EXCLUDED.heartbeat_at, status = EXCLUDED.status, finish_reason = EXCLUDED.finish_reason,
			error = EXCLUDED.error, kind = EXCLUDED.kind, depth = EXCLUDED.depth, state = EXCLUDED.state,
			state_patch = EXCLUDED.state_patch`, s.table),
		append(meta, p.kind, p.depth, p.state, p.patch)...)
	return err
}

// checkNoDiffChildren locks the row id against new children for the rest of
// the save, then fails if a row already stores a diff against it: changing the
// row's state would change what that row reads back.
func (s *PostgresSessionStore[State]) checkNoDiffChildren(ctx context.Context, tx pgx.Tx, prefix, id string) error {
	if _, err := tx.Exec(ctx, fmt.Sprintf(`SELECT 1 FROM %s WHERE prefix = $1 AND snapshot_id = $2 FOR UPDATE`, s.table), prefix, id); err != nil {
		return err
	}
	var child string
	err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT snapshot_id FROM %s WHERE prefix = $1 AND parent_id = $2 AND kind = 'diff' LIMIT 1`, s.table),
		prefix, id).Scan(&child)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return nil
	case err != nil:
		return err
	}
	return status.Errorf(status.ErrFailedPrecondition,
		"cannot change the state of snapshot %q: snapshot %q stores its state as a change against it", id, child)
}

// writePlan is how a row is stored: a checkpoint with its full state, or a
// diff with the patch from its parent's state.
type writePlan struct {
	kind  string
	depth int
	state json.RawMessage // checkpoint only; nil when the row has no state
	patch json.RawMessage // diff only
}

// plan decides how to store a row whose state is stateJSON (nil for no state)
// and whose parent is parentID. A row is a diff when its parent has a state, is
// fewer than checkpointInterval diffs from its checkpoint, and the patch is at
// most half the size of the state; any other row is a checkpoint. parentChain
// is the parent's chain when the save already read it, or nil.
func (s *PostgresSessionStore[State]) plan(ctx context.Context, tx pgx.Tx, prefix, id, parentID string, stateJSON json.RawMessage, parentChain []row) (writePlan, error) {
	checkpoint := writePlan{kind: kindCheckpoint, state: stateJSON}
	if stateJSON == nil || parentID == "" || parentID == id {
		return checkpoint, nil
	}
	// Lock the parent against a change of its state until this save commits:
	// the patch is computed against the state read below. Heartbeats and
	// aborts on the parent take a weaker lock and do not wait for this one.
	var parentDepth int
	var parentHasState bool
	err := tx.QueryRow(ctx, fmt.Sprintf(`SELECT depth, kind = 'diff' OR state IS NOT NULL FROM %s
		WHERE prefix = $1 AND snapshot_id = $2 FOR KEY SHARE`, s.table), prefix, parentID).Scan(&parentDepth, &parentHasState)
	if errors.Is(err, pgx.ErrNoRows) {
		return checkpoint, nil // lineage only: the parent is gone or never existed
	}
	if err != nil {
		return writePlan{}, err
	}
	if !parentHasState || parentDepth+1 >= s.checkpointInterval {
		return checkpoint, nil
	}
	// A chain the save read before taking the lock is still the parent's:
	// the parent's state cannot change while this row stores a diff against
	// it.
	chain := parentChain
	if chain == nil {
		if chain, err = s.readChain(ctx, tx, prefix, bySnapshotID, parentID); err != nil {
			return writePlan{}, err
		}
	}
	for _, r := range chain {
		if r.snapshotID == id {
			return checkpoint, nil // a diff here would make the chain a cycle
		}
	}
	if len(chain) == 0 {
		return checkpoint, nil
	}
	parentState, err := chainState(chain)
	if err != nil {
		// The parent cannot be read back, but this row holds its own full
		// state, so it can still be stored, as a checkpoint.
		return checkpoint, nil
	}
	patch, err := json.Marshal(aix.Diff(parentState, stateJSON))
	if err != nil {
		return writePlan{}, fmt.Errorf("encode patch: %w", err)
	}
	if len(patch) > len(stateJSON)/2 {
		return checkpoint, nil
	}
	return writePlan{kind: kindDiff, depth: parentDepth + 1, patch: patch}, nil
}

// notify queues a notification that id's status changed, for every store
// watching it to re-read the row. PostgreSQL delivers it when the transaction
// commits, and drops it when the transaction rolls back.
func (s *PostgresSessionStore[State]) notify(ctx context.Context, tx pgx.Tx, prefix, id string) error {
	payload, err := json.Marshal(notification{Prefix: prefix, ID: id})
	if err != nil {
		return err
	}
	if len(payload) > maxNotifyPayload {
		payload = nil // the watchers re-read every row they watch
	}
	_, err = tx.Exec(ctx, `SELECT pg_notify($1, $2)`, s.watcher.channel, string(payload))
	return err
}

// --- Status subscription ---

// OnSnapshotStatusChange returns a channel that yields the snapshot's status at
// subscription time and on every later change, until ctx is cancelled. Changes
// arrive over LISTEN/NOTIFY, so one written by any process reaches every
// process watching the row, and the poll covers one written without a
// notification. If the snapshot does not exist when the subscription is
// established, the channel is closed without yielding a value.
//
// Values are level-triggered: the latest status is always delivered, but a slow
// reader may skip intermediate values. Treat a received value as "the status is
// now X", not "X happened once".
func (s *PostgresSessionStore[State]) OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan aix.SnapshotStatus {
	if snapshotID != "" {
		if prefix, err := s.prefixFor(ctx); err == nil {
			return s.watcher.subscribe(ctx, watchKey{prefix: prefix, id: snapshotID})
		}
	}
	ch := make(chan aix.SnapshotStatus)
	close(ch)
	return ch
}

// readStatuses reads the status of each watched row that exists.
func (s *PostgresSessionStore[State]) readStatuses(ctx context.Context, keys []watchKey) (map[watchKey]aix.SnapshotStatus, error) {
	prefixes := make([]string, len(keys))
	ids := make([]string, len(keys))
	for i, k := range keys {
		prefixes[i], ids[i] = k.prefix, k.id
	}
	rows, err := s.pool.Query(ctx, fmt.Sprintf(`SELECT t.prefix, t.snapshot_id, t.status FROM %s t
		JOIN unnest($1::text[], $2::text[]) AS k(prefix, id)
		ON t.prefix = k.prefix COLLATE "C" AND t.snapshot_id = k.id COLLATE "C"`, s.table), prefixes, ids)
	if err != nil {
		return nil, err
	}
	found := make(map[watchKey]aix.SnapshotStatus, len(keys))
	var k watchKey
	var st string
	if _, err := pgx.ForEachRow(rows, []any{&k.prefix, &k.id, &st}, func() error {
		found[k] = aix.SnapshotStatus(st)
		return nil
	}); err != nil {
		return nil, err
	}
	return found, nil
}

// --- Helpers ---

func (s *PostgresSessionStore[State]) prefixFor(ctx context.Context) (string, error) {
	if s.prefixFn == nil {
		return defaultPrefix, nil
	}
	prefix := s.prefixFn(ctx)
	if prefix == "" {
		// As with every other option, the default is requested by omitting
		// WithSnapshotPathPrefix, not by returning an empty value from it.
		return "", fmt.Errorf("snapshot path prefix is empty; omit WithSnapshotPathPrefix to use the default %q prefix", defaultPrefix)
	}
	return prefix, nil
}

// hashKey returns a 64-bit advisory lock key for parts, which are joined with
// NUL so no two distinct part lists share an encoding.
func hashKey(parts ...string) int64 {
	h := fnv.New64a()
	for i, p := range parts {
		if i > 0 {
			h.Write([]byte{0})
		}
		h.Write([]byte(p))
	}
	return int64(h.Sum64())
}

// marshalNullable encodes v as JSON, or returns nil (SQL NULL) for a nil v. It
// returns a json.RawMessage, which pgx sends as JSON in every query mode: the
// modes that infer a parameter's type from its Go type (exec and simple
// protocol) send a []byte as bytea, which a json column rejects.
func marshalNullable[T any](v *T) (json.RawMessage, error) {
	if v == nil {
		return nil, nil
	}
	return json.Marshal(v)
}

// nullable returns nil (SQL NULL) for an empty string, and s otherwise.
func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}
