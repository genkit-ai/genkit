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
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/sessionstoretest"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/postgresql"
)

// The tests below that touch a database need a PostgreSQL server: set
// GENKIT_TEST_POSTGRES_URL to a connection URL for a database they may create
// tables in. They skip otherwise. Each test works in tables of its own and
// drops them when it ends.

// testState is the custom-state type the store tests use.
type testState struct {
	Counter int      `json:"counter"`
	Topics  []string `json:"topics,omitempty"`
}

// testPool returns a pool for the test database, closed when the test ends.
// Each non-nil configure edits the pool's config first.
func testPool(t *testing.T, configure ...func(*pgxpool.Config)) *pgxpool.Pool {
	t.Helper()
	url := os.Getenv("GENKIT_TEST_POSTGRES_URL")
	if url == "" {
		t.Skip("Skipping: GENKIT_TEST_POSTGRES_URL not set")
	}
	cfg, err := pgxpool.ParseConfig(url)
	if err != nil {
		t.Fatalf("pgxpool.ParseConfig: %v", err)
	}
	for _, c := range configure {
		if c != nil {
			c(cfg)
		}
	}
	pool, err := pgxpool.NewWithConfig(context.Background(), cfg)
	if err != nil {
		t.Fatalf("pgxpool.NewWithConfig: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// testTable returns a table name unique to the test, and drops the table when
// the test ends.
func testTable(t *testing.T, pool *pgxpool.Pool) string {
	t.Helper()
	name := "genkit_test_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:16]
	t.Cleanup(func() {
		if _, err := pool.Exec(context.Background(), "DROP TABLE IF EXISTS "+pgx.Identifier{name}.Sanitize()); err != nil {
			t.Errorf("drop %s: %v", name, err)
		}
	})
	return name
}

// newTestStore builds a store over pool in table.
func newTestStore(t *testing.T, pool *pgxpool.Pool, table string, opts ...SessionStoreOption) *PostgresSessionStore[testState] {
	t.Helper()
	store, err := newPostgresSessionStore[testState](context.Background(), pool, append([]SessionStoreOption{WithTableName(table)}, opts...)...)
	if err != nil {
		t.Fatalf("newPostgresSessionStore: %v", err)
	}
	return store
}

// fixedPrefix returns a prefix function that scopes a store to prefix.
func fixedPrefix(prefix string) func(context.Context) string {
	return func(context.Context) string { return prefix }
}

type tenantKey struct{}

func tenantFromContext(ctx context.Context) string {
	tenant, _ := ctx.Value(tenantKey{}).(string)
	return tenant
}

// runSuite runs the conformance suite with every store in one table, each
// scoped to a prefix of its own: the stores share storage but cannot see one
// another's rows, which is also how tenants share a production table.
func runSuite(t *testing.T, opts ...SessionStoreOption) {
	pool := testPool(t)
	table := testTable(t, pool)
	storeOpts := func(prefixFn func(context.Context) string) []SessionStoreOption {
		return append([]SessionStoreOption{WithSnapshotPathPrefix(prefixFn)}, opts...)
	}
	sessionstoretest.Run(t, func(t *testing.T) aix.SessionStore[testState] {
		return newTestStore(t, pool, table, storeOpts(fixedPrefix(uuid.NewString()))...)
	}, &sessionstoretest.Options[testState]{
		// A store on a pool of its own over the same rows stands in for
		// another process.
		Reopen: func(t *testing.T, store aix.SessionStore[testState]) aix.SessionStore[testState] {
			return newTestStore(t, testPool(t), table, storeOpts(store.(*PostgresSessionStore[testState]).prefixFn)...)
		},
		Scoped: func(t *testing.T) (aix.SessionStore[testState], context.Context, context.Context) {
			tenant := uuid.NewString()
			ctx := context.Background()
			return newTestStore(t, pool, table, storeOpts(tenantFromContext)...),
				context.WithValue(ctx, tenantKey{}, tenant+"-a"),
				context.WithValue(ctx, tenantKey{}, tenant+"-b")
		},
	})
}

func TestPostgresSessionStore(t *testing.T) {
	runSuite(t)
}

// TestPostgresSessionStore_FrequentCheckpoints runs the suite with a checkpoint
// every three turns, so its chains cross many checkpoint boundaries and every
// kind of write lands on both sides of one.
func TestPostgresSessionStore_FrequentCheckpoints(t *testing.T) {
	runSuite(t, WithCheckpointInterval(3))
}

func TestNewPostgresSessionStore(t *testing.T) {
	ctx := context.Background()

	t.Run("PluginNotFound", func(t *testing.T) {
		g := genkit.Init(ctx)
		if _, err := NewPostgresSessionStore[testState](ctx, g); err == nil || !strings.Contains(err.Error(), "plugin not found") {
			t.Errorf("NewPostgresSessionStore without the plugin: err = %v, want a plugin-not-found error", err)
		}
	})

	t.Run("SharesThePluginPool", func(t *testing.T) {
		pool := testPool(t)
		engine, err := postgresql.NewPostgresEngine(ctx, postgresql.WithPool(pool))
		if err != nil {
			t.Fatalf("NewPostgresEngine: %v", err)
		}
		g := genkit.Init(ctx, genkit.WithPlugins(&postgresql.Postgres{Engine: engine}))
		store, err := NewPostgresSessionStore[testState](ctx, g, WithTableName(testTable(t, pool)))
		if err != nil {
			t.Fatalf("NewPostgresSessionStore: %v", err)
		}
		if store.pool != pool {
			t.Error("the store does not use the plugin's pool")
		}
		now := time.Now()
		if _, err := store.SaveSnapshot(ctx, "x", func(*aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{SessionID: "s", CreatedAt: now, UpdatedAt: now}, nil
		}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		if got, err := store.GetSnapshot(ctx, "x"); err != nil || got == nil {
			t.Errorf("GetSnapshot = (%v, %v), want the row", got, err)
		}
	})
}

func TestOptionValidation(t *testing.T) {
	// The default for every option is to omit it; an explicit invalid value is
	// rejected rather than silently defaulted.
	cases := []struct {
		name string
		opt  SessionStoreOption
	}{
		{"empty table name", WithTableName("")},
		{"table name too long for its index names", WithTableName(strings.Repeat("t", maxTableNameLength+1))},
		{"empty schema name", WithSchemaName("")},
		{"schema name too long", WithSchemaName(strings.Repeat("s", maxIdentifierLength+1))},
		{"zero checkpoint interval", WithCheckpointInterval(0)},
		{"negative checkpoint interval", WithCheckpointInterval(-1)},
		{"nil prefix fn", WithSnapshotPathPrefix(nil)},
	}
	for _, tc := range cases {
		var cfg sessionStoreOptions
		if err := tc.opt.applySessionStore(&cfg); err == nil {
			t.Errorf("%s: expected an error", tc.name)
		}
	}

	twice := []struct {
		name        string
		first, then SessionStoreOption
	}{
		{"table name", WithTableName("a"), WithTableName("b")},
		{"schema name", WithSchemaName("a"), WithSchemaName("b")},
		{"checkpoint interval", WithCheckpointInterval(5), WithCheckpointInterval(7)},
		{"prefix", WithSnapshotPathPrefix(fixedPrefix("a")), WithSnapshotPathPrefix(fixedPrefix("b"))},
		{"poll interval", WithPollInterval(time.Second), WithPollInterval(0)},
	}
	for _, tc := range twice {
		var cfg sessionStoreOptions
		if err := tc.first.applySessionStore(&cfg); err != nil {
			t.Fatalf("%s: first set: %v", tc.name, err)
		}
		if err := tc.then.applySessionStore(&cfg); err == nil {
			t.Errorf("%s: expected an error setting it twice", tc.name)
		}
	}
}

// TestEmptyPrefixRejected verifies every operation rejects a configured prefix
// function that returns an empty value before touching the database (the store
// has no pool). The default "global" prefix is requested by omitting
// WithSnapshotPathPrefix, not by returning an empty value from it.
func TestEmptyPrefixRejected(t *testing.T) {
	ctx := context.Background()
	store := &PostgresSessionStore[testState]{prefixFn: fixedPrefix("")}
	if _, err := store.GetSnapshot(ctx, "snap"); err == nil {
		t.Error("GetSnapshot: expected an error for an empty prefix")
	}
	if _, err := store.GetLatestSnapshot(ctx, "sess"); err == nil {
		t.Error("GetLatestSnapshot: expected an error for an empty prefix")
	}
	if _, err := store.GetSnapshotMetadata(ctx, "snap"); err == nil {
		t.Error("GetSnapshotMetadata: expected an error for an empty prefix")
	}
	if _, err := store.SaveSnapshot(ctx, "snap", func(*aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
		return &aix.SessionSnapshot[testState]{SessionID: "s"}, nil
	}); err == nil {
		t.Error("SaveSnapshot: expected an error for an empty prefix")
	}
	if _, ok := <-store.OnSnapshotStatusChange(ctx, "snap"); ok {
		t.Error("OnSnapshotStatusChange: expected a closed channel for an empty prefix")
	}
}

func TestTableSetup(t *testing.T) {
	ctx := context.Background()
	pool := testPool(t)

	t.Run("ConcurrentStoresCreateOneTable", func(t *testing.T) {
		// Instances starting together all find or create the same table.
		table := testTable(t, pool)
		var wg sync.WaitGroup
		errs := make(chan error, 4)
		for range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, err := newPostgresSessionStore[testState](ctx, pool, WithTableName(table))
				errs <- err
			}()
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Errorf("newPostgresSessionStore: %v", err)
			}
		}
	})

	t.Run("InSchema", func(t *testing.T) {
		schema := testTable(t, pool) // a unique name; the schema replaces the table
		if _, err := pool.Exec(ctx, "CREATE SCHEMA "+pgx.Identifier{schema}.Sanitize()); err != nil {
			t.Fatalf("CREATE SCHEMA: %v", err)
		}
		t.Cleanup(func() {
			pool.Exec(context.Background(), "DROP SCHEMA "+pgx.Identifier{schema}.Sanitize()+" CASCADE")
		})
		store, err := newPostgresSessionStore[testState](ctx, pool, WithSchemaName(schema), WithTableName("snapshots"))
		if err != nil {
			t.Fatalf("newPostgresSessionStore: %v", err)
		}
		var exists bool
		if err := pool.QueryRow(ctx, `SELECT to_regclass($1) IS NOT NULL`, pgx.Identifier{schema, "snapshots"}.Sanitize()).Scan(&exists); err != nil || !exists {
			t.Fatalf("table %s.snapshots exists = %v (err %v), want true", schema, exists, err)
		}
		if want := (pgx.Identifier{schema, "snapshots"}).Sanitize(); store.table != want {
			t.Errorf("store table = %s, want %s", store.table, want)
		}
	})

	t.Run("RejectsATableWithAnotherLayout", func(t *testing.T) {
		table := testTable(t, pool)
		if _, err := pool.Exec(ctx, "CREATE TABLE "+pgx.Identifier{table}.Sanitize()+" (id text PRIMARY KEY)"); err != nil {
			t.Fatalf("CREATE TABLE: %v", err)
		}
		if _, err := newPostgresSessionStore[testState](ctx, pool, WithTableName(table)); err == nil || !strings.Contains(err.Error(), "layout") {
			t.Errorf("newPostgresSessionStore over a foreign table: err = %v, want a layout error", err)
		}
	})
}

// save writes a completed row with the given messages, created at at.
func save(t *testing.T, store *PostgresSessionStore[testState], id, parentID string, at time.Time, texts ...string) {
	t.Helper()
	topics := append([]string(nil), texts...)
	if _, err := store.SaveSnapshot(context.Background(), id, func(*aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
		return &aix.SessionSnapshot[testState]{
			SessionID: "sess",
			ParentID:  parentID,
			Status:    aix.SnapshotStatusCompleted,
			State:     &aix.SessionState[testState]{Custom: testState{Counter: len(topics), Topics: topics}},
			CreatedAt: at,
			UpdatedAt: at,
		}, nil
	}); err != nil {
		t.Fatalf("SaveSnapshot(%q): %v", id, err)
	}
}

// TestStorageLayout checks how rows are stored: a checkpoint every N rows of a
// chain with diffs between, and a row whose diff would be large stored as a
// checkpoint instead. A store that wrote every row as a checkpoint would read
// back correctly and still lose the point of the layout.
func TestStorageLayout(t *testing.T) {
	ctx := context.Background()
	pool := testPool(t)
	table := testTable(t, pool)
	store := newTestStore(t, pool, table, WithCheckpointInterval(3))

	// A long, stable history makes each turn's change small next to the state.
	base := make([]string, 50)
	for i := range base {
		base[i] = fmt.Sprintf("an early topic that stays in the history, number %02d", i)
	}
	at := time.Now()
	parent := ""
	for i := range 7 {
		id := fmt.Sprintf("turn-%d", i)
		save(t, store, id, parent, at.Add(time.Duration(i)*time.Millisecond), append(base, fmt.Sprintf("turn %d", i))...)
		parent = id
	}
	// The history is replaced: the change is larger than half the state.
	save(t, store, "compacted", parent, at.Add(time.Second), "summary")

	want := []struct {
		id    string
		kind  string
		depth int
	}{
		{"turn-0", kindCheckpoint, 0},
		{"turn-1", kindDiff, 1},
		{"turn-2", kindDiff, 2},
		{"turn-3", kindCheckpoint, 0},
		{"turn-4", kindDiff, 1},
		{"turn-5", kindDiff, 2},
		{"turn-6", kindCheckpoint, 0},
		{"compacted", kindCheckpoint, 0},
	}
	for _, w := range want {
		var kind string
		var depth int
		var hasState, hasPatch bool
		if err := pool.QueryRow(ctx, fmt.Sprintf(`SELECT kind, depth, state IS NOT NULL, state_patch IS NOT NULL FROM %s WHERE snapshot_id = $1`,
			pgx.Identifier{table}.Sanitize()), w.id).Scan(&kind, &depth, &hasState, &hasPatch); err != nil {
			t.Fatalf("read %s: %v", w.id, err)
		}
		if kind != w.kind || depth != w.depth {
			t.Errorf("%s stored as %s at depth %d, want %s at depth %d", w.id, kind, depth, w.kind, w.depth)
		}
		if hasState != (kind == kindCheckpoint) || hasPatch != (kind == kindDiff) {
			t.Errorf("%s (%s) has state %v and patch %v", w.id, kind, hasState, hasPatch)
		}
	}
}

// TestRejectsStateChangeUnderADiff checks that a row's state cannot change
// while another row stores a diff against it, since the change would corrupt
// that row, and that everything else about the row still can.
func TestRejectsStateChangeUnderADiff(t *testing.T) {
	ctx := context.Background()
	pool := testPool(t)
	store := newTestStore(t, pool, testTable(t, pool))

	topics := make([]string, 30)
	for i := range topics {
		topics[i] = fmt.Sprintf("a topic that keeps the state large, number %02d", i)
	}
	at := time.Now()
	save(t, store, "parent", "", at, topics...)
	save(t, store, "child", "parent", at.Add(time.Millisecond), append(topics, "one more")...)

	rewrite := func(edit func(*aix.SessionSnapshot[testState])) error {
		_, err := store.SaveSnapshot(ctx, "parent", func(existing *aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			edit(existing)
			return existing, nil
		})
		return err
	}
	if err := rewrite(func(s *aix.SessionSnapshot[testState]) { s.State.Custom.Counter = 99 }); !errors.Is(err, status.ErrFailedPrecondition) {
		t.Errorf("changing the state of a diff's parent: err = %v, want FAILED_PRECONDITION", err)
	}
	if err := rewrite(func(s *aix.SessionSnapshot[testState]) { s.FinishReason = aix.AgentFinishReasonStop }); err != nil {
		t.Errorf("changing the parent's metadata: %v", err)
	}
	child, err := store.GetSnapshot(ctx, "child")
	if err != nil || child == nil || len(child.State.Custom.Topics) != 31 {
		t.Errorf("child after the parent's rewrites = (%+v, %v), want its own 31 topics", child, err)
	}
}

// TestLatestBreaksTiesInByteOrder checks that equal CreatedAt values resolve to
// the greater snapshot ID in byte order, as every other store compares IDs,
// whatever the database's collation: in en_US.UTF-8, "a" sorts before "B".
func TestLatestBreaksTiesInByteOrder(t *testing.T) {
	ctx := context.Background()
	pool := testPool(t)
	store := newTestStore(t, pool, testTable(t, pool))
	at := time.Now()
	save(t, store, "a", "", at, "lower")
	save(t, store, "B", "", at, "upper")
	for name, read := range map[string]func() (*aix.SessionSnapshot[testState], error){
		"GetLatestSnapshot":         func() (*aix.SessionSnapshot[testState], error) { return store.GetLatestSnapshot(ctx, "sess") },
		"GetLatestSnapshotMetadata": func() (*aix.SessionSnapshot[testState], error) { return store.GetLatestSnapshotMetadata(ctx, "sess") },
	} {
		if got, err := read(); err != nil || got == nil || got.SnapshotID != "a" {
			t.Errorf("%s = (%v, %v), want a (\"a\" > \"B\" in byte order)", name, got, err)
		}
	}
}

// TestDiffWaitsForItsParentsChange checks that a new row computes its diff
// only after a change to its parent's state commits, so the diff is never taken
// against a state that is about to be replaced.
func TestDiffWaitsForItsParentsChange(t *testing.T) {
	ctx := context.Background()
	pool := testPool(t)
	table := testTable(t, pool)
	store := newTestStore(t, pool, table)

	topics := make([]string, 30)
	for i := range topics {
		topics[i] = fmt.Sprintf("a topic that keeps the state large, number %02d", i)
	}
	at := time.Now()
	save(t, store, "parent", "", at, topics...)

	// Change the parent's state the way a save does: under FOR UPDATE.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	defer tx.Rollback(ctx)
	changed := append(append([]string(nil), topics...), "changed")
	if _, err := tx.Exec(ctx, fmt.Sprintf(`SELECT 1 FROM %s WHERE snapshot_id = 'parent' FOR UPDATE`, store.table)); err != nil {
		t.Fatalf("lock the parent: %v", err)
	}
	if _, err := tx.Exec(ctx, fmt.Sprintf(`UPDATE %s SET state = $1 WHERE snapshot_id = 'parent'`, store.table),
		fmt.Sprintf(`{"custom":{"counter":%d,"topics":%s}}`, len(changed), mustJSON(t, changed))); err != nil {
		t.Fatalf("change the parent: %v", err)
	}

	child := append(append([]string(nil), changed...), "child")
	done := make(chan error, 1)
	go func() {
		_, err := store.SaveSnapshot(ctx, "child", func(*aix.SessionSnapshot[testState]) (*aix.SessionSnapshot[testState], error) {
			return &aix.SessionSnapshot[testState]{
				SessionID: "sess", ParentID: "parent", Status: aix.SnapshotStatusCompleted,
				State:     &aix.SessionState[testState]{Custom: testState{Counter: len(child), Topics: child}},
				CreatedAt: at.Add(time.Millisecond), UpdatedAt: at.Add(time.Millisecond),
			}, nil
		})
		done <- err
	}()
	select {
	case err := <-done:
		t.Fatalf("the child saved while its parent's state was changing (err: %v)", err)
	case <-time.After(300 * time.Millisecond):
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("SaveSnapshot(child): %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the child did not save after its parent's change committed")
	}
	got, err := store.GetSnapshot(ctx, "child")
	if err != nil || got == nil || len(got.State.Custom.Topics) != len(child) || got.State.Custom.Topics[len(child)-2] != "changed" {
		t.Errorf("child = (%+v, %v), want its own %d topics on top of the changed parent", got, err, len(child))
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	return string(b)
}
