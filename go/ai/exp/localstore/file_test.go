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

package localstore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/sessionstoretest"
)

// testState is the custom-state type the store tests use.
type testState struct {
	Counter int      `json:"counter"`
	Topics  []string `json:"topics,omitempty"`
}

func newFileStore(t *testing.T) *FileSessionStore[testState] {
	t.Helper()
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	return store
}

func TestFileSessionStore(t *testing.T) {
	sessionstoretest.Run(t, func(t *testing.T) exp.SessionStore[testState] {
		return newFileStore(t)
	}, &sessionstoretest.Options[testState]{
		// A second store over the same directory stands in for another
		// process. Its short poll interval keeps the cross-process status
		// check fast without changing what it checks.
		Reopen: func(t *testing.T, store exp.SessionStore[testState]) exp.SessionStore[testState] {
			reopened, err := NewFileSessionStore[testState](store.(*FileSessionStore[testState]).dir, WithPollInterval(5*time.Millisecond))
			if err != nil {
				t.Fatalf("NewFileSessionStore: %v", err)
			}
			return reopened
		},
		Scoped: func(t *testing.T) (exp.SessionStore[testState], context.Context, context.Context) {
			store, err := NewFileSessionStore[testState](t.TempDir(), WithSnapshotPathPrefix(prefixFromCtx))
			if err != nil {
				t.Fatalf("NewFileSessionStore: %v", err)
			}
			return store, ctxWithPrefix("tenant-a"), ctxWithPrefix("tenant-b")
		},
	})
}

func TestNewFileSessionStore(t *testing.T) {
	t.Run("EmptyDirRejected", func(t *testing.T) {
		if _, err := NewFileSessionStore[testState](""); err == nil {
			t.Error("expected error for empty dir, got nil")
		}
	})

	t.Run("CreatesMissingDir", func(t *testing.T) {
		dir := filepath.Join(t.TempDir(), "nested", "subdir")
		if _, err := NewFileSessionStore[testState](dir); err != nil {
			t.Fatalf("NewFileSessionStore: %v", err)
		}
		if _, err := os.Stat(dir); err != nil {
			t.Errorf("expected dir to be created, stat: %v", err)
		}
	})
}

func TestFileSessionStore_Files(t *testing.T) {
	t.Run("InvalidIDRejected", func(t *testing.T) {
		store := newFileStore(t)
		cases := []string{
			"../escape",
			"a/b",
			`a\b`,
			".hidden",
			"foo..bar",
		}
		for _, id := range cases {
			t.Run(id, func(t *testing.T) {
				if _, err := store.GetSnapshot(context.Background(), id); err == nil {
					t.Errorf("GetSnapshot(%q): expected error, got nil", id)
				}
				if _, err := store.SaveSnapshot(context.Background(), id,
					func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
						return &exp.SessionSnapshot[testState]{}, nil
					}); err == nil {
					t.Errorf("SaveSnapshot(%q): expected error, got nil", id)
				}
			})
		}
	})

	t.Run("FileWrittenOnDisk", func(t *testing.T) {
		dir := t.TempDir()
		store, err := NewFileSessionStore[testState](dir)
		if err != nil {
			t.Fatalf("NewFileSessionStore: %v", err)
		}
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{SessionID: "sess-1", Status: exp.SnapshotStatusCompleted}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		// With no prefix configured the snapshot defaults to the "global"
		// subdirectory, never the bare store root.
		if _, err := os.Stat(filepath.Join(dir, "global", "snap-1.json")); err != nil {
			t.Errorf("expected global/snap-1.json on disk: %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, "snap-1.json")); !os.IsNotExist(err) {
			t.Errorf("snapshot must not land in store root, stat err = %v", err)
		}
	})
}

// recvStatus waits up to timeout for the next status on ch, failing the test on
// a timeout or an unexpectedly closed channel.
func recvStatus(t *testing.T, ch <-chan exp.SnapshotStatus, timeout time.Duration) exp.SnapshotStatus {
	t.Helper()
	select {
	case s, ok := <-ch:
		if !ok {
			t.Fatal("status channel closed unexpectedly")
		}
		return s
	case <-time.After(timeout):
		t.Fatal("timeout waiting for status")
		return ""
	}
}

// abortViaSave flips a pending snapshot to aborted through SaveSnapshot, the
// way a writer in another process would. Returns the resulting status, or ""
// when the snapshot does not exist.
func abortViaSave(t *testing.T, store exp.SessionStore[testState], id string) exp.SnapshotStatus {
	t.Helper()
	saved, err := store.SaveSnapshot(context.Background(), id,
		func(existing *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			if existing == nil {
				return nil, nil
			}
			if existing.Status != exp.SnapshotStatusPending {
				return existing, nil
			}
			updated := *existing
			updated.Status = exp.SnapshotStatusAborted
			updated.UpdatedAt = time.Now()
			return &updated, nil
		})
	if err != nil {
		t.Fatalf("abortViaSave(%q): %v", id, err)
	}
	if saved == nil {
		return ""
	}
	return saved.Status
}

// TestFileSessionStore_PollIntervalDisabled verifies that WithPollInterval(0)
// turns off cross-process polling: the subscriber still gets the seed value but
// never sees a change written through another instance.
func TestFileSessionStore_PollIntervalDisabled(t *testing.T) {
	dir := t.TempDir()
	writer, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore (writer): %v", err)
	}
	watcher, err := NewFileSessionStore[testState](dir, WithPollInterval(0))
	if err != nil {
		t.Fatalf("NewFileSessionStore (watcher): %v", err)
	}

	if _, err := writer.SaveSnapshot(context.Background(), "snap-1",
		func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			return &exp.SessionSnapshot[testState]{SessionID: "sess-1", Status: exp.SnapshotStatusPending}, nil
		}); err != nil {
		t.Fatalf("seed pending: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch := watcher.OnSnapshotStatusChange(ctx, "snap-1")

	if got := recvStatus(t, ch, time.Second); got != exp.SnapshotStatusPending {
		t.Fatalf("initial status = %q, want %q", got, exp.SnapshotStatusPending)
	}

	abortViaSave(t, writer, "snap-1")
	select {
	case got := <-ch:
		t.Fatalf("with polling disabled, unexpectedly observed status %q", got)
	case <-time.After(150 * time.Millisecond):
	}
}

func TestFileSessionStore_GetLatestSnapshot_SkipsUnparseableFiles(t *testing.T) {
	// A stray unparseable .json file (crash artifact, partial copy,
	// hand-edited row) must not poison session resolution for the healthy
	// rows in the directory; it is skipped like the dead end it is.
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	ctx := context.Background()
	if _, err := store.SaveSnapshot(ctx, "a",
		func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			return &exp.SessionSnapshot[testState]{
				SessionID: "sess-1",
				Status:    exp.SnapshotStatusCompleted,
			}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	// Drop the junk file alongside the healthy row (under the default "global"
	// prefix, the directory GetLatestSnapshot scans) so it is genuinely in the
	// scan path rather than an unscanned sibling.
	if err := os.WriteFile(filepath.Join(dir, "global", "junk.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	tip, err := store.GetLatestSnapshot(ctx, "sess-1")
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if tip == nil || tip.SnapshotID != "a" {
		t.Errorf("expected the healthy row as tip, got %+v", tip)
	}
}

// TestFileSessionStore_MaxPersistedChainLength verifies that, with a retention
// window of n, each save unlinks the rows past the newest n in the snapshot's
// parentId chain while leaving the window (and session resolution) intact.
func TestFileSessionStore_MaxPersistedChainLength(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir, WithMaxPersistedChainLength(2))
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	ctx := context.Background()
	base := time.Now()

	// A linear chain s0 <- s1 <- s2 <- s3 <- s4, each created strictly after
	// the last so recency is unambiguous.
	ids := []string{"s0", "s1", "s2", "s3", "s4"}
	parent := ""
	for i, id := range ids {
		createdAt := base.Add(time.Duration(i) * time.Second)
		parentID := parent
		if _, err := store.SaveSnapshot(ctx, id,
			func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
				return &exp.SessionSnapshot[testState]{
					SessionID: "sess",
					ParentID:  parentID,
					Status:    exp.SnapshotStatusCompleted,
					State:     &exp.SessionState[testState]{Custom: testState{Counter: i}},
					CreatedAt: createdAt,
					UpdatedAt: createdAt,
				}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot(%q): %v", id, err)
		}
		parent = id
	}

	// Only the two newest rows survive; everything older is pruned.
	for _, gone := range []string{"s0", "s1", "s2"} {
		snap, err := store.GetSnapshot(ctx, gone)
		if err != nil {
			t.Fatalf("GetSnapshot(%q): %v", gone, err)
		}
		if snap != nil {
			t.Errorf("expected %q pruned, but it is still present", gone)
		}
	}
	for _, kept := range []string{"s3", "s4"} {
		snap, err := store.GetSnapshot(ctx, kept)
		if err != nil {
			t.Fatalf("GetSnapshot(%q): %v", kept, err)
		}
		if snap == nil {
			t.Errorf("expected %q retained, got nil", kept)
		}
	}

	// Session resolution still finds the newest survivor.
	latest, err := store.GetLatestSnapshot(ctx, "sess")
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if latest == nil || latest.SnapshotID != "s4" {
		t.Errorf("latest = %+v, want s4", latest)
	}
}

type prefixCtxKey struct{}

func ctxWithPrefix(prefix string) context.Context {
	return context.WithValue(context.Background(), prefixCtxKey{}, prefix)
}

func prefixFromCtx(ctx context.Context) string {
	v, _ := ctx.Value(prefixCtxKey{}).(string)
	return v
}

// TestFileSessionStore_PathPrefix verifies that a context-derived prefix names
// the subdirectory a snapshot lands in. Isolation between prefixes is part of
// the shared suite (see TestFileSessionStore).
func TestFileSessionStore_PathPrefix(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir, WithSnapshotPathPrefix(prefixFromCtx))
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	if _, err := store.SaveSnapshot(ctxWithPrefix("tenant-a"), "s1",
		func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			return &exp.SessionSnapshot[testState]{SessionID: "sess", Status: exp.SnapshotStatusCompleted}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}

	// Written under the tenant subdirectory, not the store root.
	if _, err := os.Stat(filepath.Join(dir, "tenant-a", "s1.json")); err != nil {
		t.Errorf("expected tenant-a/s1.json on disk: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "s1.json")); !os.IsNotExist(err) {
		t.Errorf("snapshot must not land in store root, stat err = %v", err)
	}
}

// TestFileSessionStore_PathPrefix_Nested verifies a prefix may nest multiple
// subdirectories via "/".
func TestFileSessionStore_PathPrefix_Nested(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir,
		WithSnapshotPathPrefix(func(context.Context) string { return "org-42/user-7" }))
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	ctx := context.Background()
	if _, err := store.SaveSnapshot(ctx, "s1",
		func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			return &exp.SessionSnapshot[testState]{SessionID: "sess", Status: exp.SnapshotStatusCompleted}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "org-42", "user-7", "s1.json")); err != nil {
		t.Errorf("expected org-42/user-7/s1.json on disk: %v", err)
	}
	got, err := store.GetSnapshot(ctx, "s1")
	if err != nil || got == nil {
		t.Errorf("GetSnapshot through nested prefix: got=%v err=%v", got, err)
	}
}

// TestFileSessionStore_PathPrefix_Rejected verifies a prefix that would escape
// the store directory is rejected at call time rather than silently writing
// outside it.
func TestFileSessionStore_PathPrefix_Rejected(t *testing.T) {
	for _, bad := range []string{"../escape", `a\b`, ".hidden", "ok/../escape"} {
		t.Run(bad, func(t *testing.T) {
			dir := t.TempDir()
			store, err := NewFileSessionStore[testState](dir,
				WithSnapshotPathPrefix(func(context.Context) string { return bad }))
			if err != nil {
				t.Fatalf("NewFileSessionStore: %v", err)
			}
			ctx := context.Background()
			if _, err := store.SaveSnapshot(ctx, "s1",
				func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
					return &exp.SessionSnapshot[testState]{SessionID: "sess"}, nil
				}); err == nil {
				t.Error("SaveSnapshot: expected error for escaping prefix, got nil")
			}
			if _, err := store.GetSnapshot(ctx, "s1"); err == nil {
				t.Error("GetSnapshot: expected error for escaping prefix, got nil")
			}
			if _, err := store.GetLatestSnapshot(ctx, "sess"); err == nil {
				t.Error("GetLatestSnapshot: expected error for escaping prefix, got nil")
			}
		})
	}
}

// TestFileSessionStore_PathPrefix_EmptyRejected verifies that a configured prefix
// function returning an empty or separator-only value is rejected at call time.
// The default "global" prefix is requested by omitting the option, not by
// returning an empty value from it.
func TestFileSessionStore_PathPrefix_EmptyRejected(t *testing.T) {
	cases := []struct{ name, prefix string }{
		{"empty", ""},
		{"single slash", "/"},
		{"multiple slashes", "///"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			store, err := NewFileSessionStore[testState](dir,
				WithSnapshotPathPrefix(func(context.Context) string { return tc.prefix }))
			if err != nil {
				t.Fatalf("NewFileSessionStore: %v", err)
			}
			ctx := context.Background()
			if _, err := store.SaveSnapshot(ctx, "s1",
				func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
					return &exp.SessionSnapshot[testState]{SessionID: "sess"}, nil
				}); err == nil {
				t.Error("SaveSnapshot: expected error for empty prefix, got nil")
			}
			if _, err := store.GetSnapshot(ctx, "s1"); err == nil {
				t.Error("GetSnapshot: expected error for empty prefix, got nil")
			}
			if _, err := store.GetLatestSnapshot(ctx, "sess"); err == nil {
				t.Error("GetLatestSnapshot: expected error for empty prefix, got nil")
			}
		})
	}
}

// TestFileSessionStore_OptionSetTwice verifies the construction options reject
// being set more than once, surfacing the error from NewFileSessionStore.
func TestFileSessionStore_OptionSetTwice(t *testing.T) {
	dir := t.TempDir()
	if _, err := NewFileSessionStore[testState](dir,
		WithMaxPersistedChainLength(2), WithMaxPersistedChainLength(3)); err == nil {
		t.Error("expected error setting max persisted chain length twice, got nil")
	}
	if _, err := NewFileSessionStore[testState](dir,
		WithSnapshotPathPrefix(prefixFromCtx), WithSnapshotPathPrefix(prefixFromCtx)); err == nil {
		t.Error("expected error setting snapshot path prefix twice, got nil")
	}
}

// TestFileSessionStore_MaxPersistedChainLength_Invalid verifies a retention
// window of 0 or less is rejected at construction rather than silently
// disabling pruning.
func TestFileSessionStore_MaxPersistedChainLength_Invalid(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []int{0, -1} {
		if _, err := NewFileSessionStore[testState](dir, WithMaxPersistedChainLength(n)); err == nil {
			t.Errorf("WithMaxPersistedChainLength(%d): expected error, got nil", n)
		}
	}
}

// TestFileSessionStore_MaxPersistedChainLength_One verifies a window of 1 is
// accepted and keeps only the latest snapshot, pruning each predecessor.
func TestFileSessionStore_MaxPersistedChainLength_One(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir, WithMaxPersistedChainLength(1))
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	ctx := context.Background()
	now := time.Now()
	if _, err := store.SaveSnapshot(ctx, "s0",
		func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			return &exp.SessionSnapshot[testState]{SessionID: "sess", Status: exp.SnapshotStatusCompleted, CreatedAt: now, UpdatedAt: now}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot(s0): %v", err)
	}
	later := now.Add(time.Second)
	if _, err := store.SaveSnapshot(ctx, "s1",
		func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			return &exp.SessionSnapshot[testState]{SessionID: "sess", ParentID: "s0", Status: exp.SnapshotStatusCompleted, CreatedAt: later, UpdatedAt: later}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot(s1): %v", err)
	}
	if snap, _ := store.GetSnapshot(ctx, "s0"); snap != nil {
		t.Error("expected predecessor s0 pruned with window 1")
	}
	if snap, _ := store.GetSnapshot(ctx, "s1"); snap == nil {
		t.Error("expected tip s1 retained with window 1")
	}
}

// saveCompletedAt saves a completed snapshot with an explicit CreatedAt so a
// test controls recency directly rather than racing the wall clock.
func saveCompletedAt(t *testing.T, store *FileSessionStore[testState], id, sessionID, parentID string, createdAt time.Time) *exp.SessionSnapshot[testState] {
	t.Helper()
	saved, err := store.SaveSnapshot(context.Background(), id,
		func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
			return &exp.SessionSnapshot[testState]{
				SessionID: sessionID,
				ParentID:  parentID,
				Status:    exp.SnapshotStatusCompleted,
				State:     &exp.SessionState[testState]{Custom: testState{Counter: 1}},
				CreatedAt: createdAt,
				UpdatedAt: createdAt,
			}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot(%q): %v", id, err)
	}
	return saved
}

// pointerFilePath is the on-disk path of a session's pointer file under the
// default "global" prefix.
func pointerFilePath(dir, sessionID string) string {
	return filepath.Join(dir, "global", pointersSubdir, sessionID+".json")
}

// readPointerFile reads and decodes a session's pointer file, failing the test
// if it is missing or unparseable.
func readPointerFile(t *testing.T, dir, sessionID string) pointerDoc {
	t.Helper()
	data, err := os.ReadFile(pointerFilePath(dir, sessionID))
	if err != nil {
		t.Fatalf("read pointer file: %v", err)
	}
	var doc pointerDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("decode pointer file: %v", err)
	}
	return doc
}

// TestFileSessionStore_SessionPointer_WrittenTracksLatest verifies that saving a
// chain advances the per-session pointer to the newest row, recording both its
// ID and CreatedAt.
func TestFileSessionStore_SessionPointer_WrittenTracksLatest(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	base := time.Now()
	saveCompletedAt(t, store, "first", "sess-1", "", base)
	second := saveCompletedAt(t, store, "second", "sess-1", "first", base.Add(time.Second))

	doc := readPointerFile(t, dir, "sess-1")
	if doc.CurrentSnapshotID != "second" {
		t.Errorf("pointer CurrentSnapshotID = %q, want %q", doc.CurrentSnapshotID, "second")
	}
	if !doc.CurrentCreatedAt.Equal(second.CreatedAt) {
		t.Errorf("pointer CurrentCreatedAt = %v, want %v", doc.CurrentCreatedAt, second.CreatedAt)
	}
}

// TestFileSessionStore_SessionPointer_FastPathTrustsPointer rigs the pointer and
// the scan to disagree, proving GetLatestSnapshot resolves through the pointer
// (the fast path) rather than scanning. A newer same-session row is written
// straight to disk, bypassing SaveSnapshot so the pointer is never advanced to
// it; the fast path must still return the pointer's (older) target. This is the
// documented best-effort trade-off: a pointer left behind by a lost update
// resolves to a valid-but-older row until a save advances it or a pointer-less
// lookup rescans.
func TestFileSessionStore_SessionPointer_FastPathTrustsPointer(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	base := time.Now()
	saveCompletedAt(t, store, "pointed", "sess-1", "", base)

	// A newer row written out-of-band: the scan would prefer it (greater
	// CreatedAt), but the pointer does not know about it.
	newer := &exp.SessionSnapshot[testState]{
		SnapshotID: "newer",
		SessionID:  "sess-1",
		ParentID:   "pointed",
		Status:     exp.SnapshotStatusCompleted,
		State:      &exp.SessionState[testState]{Custom: testState{Counter: 2}},
		CreatedAt:  base.Add(time.Hour),
		UpdatedAt:  base.Add(time.Hour),
	}
	data, err := json.MarshalIndent(newer, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "global", "newer.json"), data, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	latest, err := store.GetLatestSnapshot(context.Background(), "sess-1")
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if latest == nil || latest.SnapshotID != "pointed" {
		t.Errorf("latest = %+v, want pointer target %q (fast path must not scan)", latest, "pointed")
	}
}

// TestFileSessionStore_SessionPointer_SelfHealsMissingPointer verifies that a
// lookup with no pointer file (e.g. a legacy store) still resolves the latest
// row via the scan and rewrites the pointer for next time.
func TestFileSessionStore_SessionPointer_SelfHealsMissingPointer(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	base := time.Now()
	saveCompletedAt(t, store, "first", "sess-1", "", base)
	saveCompletedAt(t, store, "second", "sess-1", "first", base.Add(time.Second))

	if err := os.Remove(pointerFilePath(dir, "sess-1")); err != nil {
		t.Fatalf("remove pointer: %v", err)
	}

	latest, err := store.GetLatestSnapshot(context.Background(), "sess-1")
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if latest == nil || latest.SnapshotID != "second" {
		t.Fatalf("latest = %+v, want second (resolved via scan)", latest)
	}
	// The scan rewrote the pointer so the next lookup is fast again.
	if doc := readPointerFile(t, dir, "sess-1"); doc.CurrentSnapshotID != "second" {
		t.Errorf("rewritten pointer CurrentSnapshotID = %q, want %q", doc.CurrentSnapshotID, "second")
	}
}

// TestFileSessionStore_SessionPointer_StaleFallsBackToScan verifies that a
// pointer naming a snapshot that no longer exists falls back to the scan and is
// refreshed to the real latest row.
func TestFileSessionStore_SessionPointer_StaleFallsBackToScan(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	base := time.Now()
	saveCompletedAt(t, store, "first", "sess-1", "", base)
	saveCompletedAt(t, store, "second", "sess-1", "first", base.Add(time.Second))

	// Point at a snapshot that does not exist.
	stale, err := json.MarshalIndent(&pointerDoc{
		CurrentSnapshotID: "does-not-exist",
		CurrentCreatedAt:  base.Add(time.Hour),
		UpdatedAt:         base.Add(time.Hour),
	}, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(pointerFilePath(dir, "sess-1"), stale, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	latest, err := store.GetLatestSnapshot(context.Background(), "sess-1")
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if latest == nil || latest.SnapshotID != "second" {
		t.Fatalf("latest = %+v, want second (resolved via scan)", latest)
	}
	if doc := readPointerFile(t, dir, "sess-1"); doc.CurrentSnapshotID != "second" {
		t.Errorf("refreshed pointer CurrentSnapshotID = %q, want %q", doc.CurrentSnapshotID, "second")
	}
}

// TestFileSessionStore_SessionPointer_NotInScanSpace verifies the pointer lives
// in the hidden ".pointers" sub-directory, so the snapshot scan (which reads
// only *.json files directly under the prefix) never sees it.
func TestFileSessionStore_SessionPointer_NotInScanSpace(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	saveCompletedAt(t, store, "only", "sess-1", "", time.Now())

	entries, err := os.ReadDir(filepath.Join(dir, "global"))
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	var jsonFiles []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			jsonFiles = append(jsonFiles, e.Name())
		}
	}
	if len(jsonFiles) != 1 || jsonFiles[0] != "only.json" {
		t.Errorf("prefix-dir *.json files = %v, want [only.json] (pointer must live under %q)", jsonFiles, pointersSubdir)
	}
}

// TestFileSessionStore_SessionPointer_BackdatedNewRowDoesNotAdvance verifies the
// Go-specific advance rule: a brand-new row created *before* the current pointer
// target does not move the pointer, so the fast path keeps agreeing with the
// scan's greatest-CreatedAt ordering. (A naive "advance on every new row" port
// from JS would regress this.)
func TestFileSessionStore_SessionPointer_BackdatedNewRowDoesNotAdvance(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore[testState](dir)
	if err != nil {
		t.Fatalf("NewFileSessionStore: %v", err)
	}
	base := time.Now()
	saveCompletedAt(t, store, "newest", "sess-1", "", base.Add(time.Hour))
	// A second brand-new row for the same session, but backdated well before the
	// first - so it is the newest *save*, yet not the greatest-CreatedAt row.
	saveCompletedAt(t, store, "backdated", "sess-1", "", base)

	if doc := readPointerFile(t, dir, "sess-1"); doc.CurrentSnapshotID != "newest" {
		t.Errorf("pointer CurrentSnapshotID = %q, want %q (backdated row must not advance it)", doc.CurrentSnapshotID, "newest")
	}
	latest, err := store.GetLatestSnapshot(context.Background(), "sess-1")
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if latest == nil || latest.SnapshotID != "newest" {
		t.Errorf("latest = %+v, want newest (greatest CreatedAt)", latest)
	}
}

// TestFileSessionStore_PathUnsafeSessionIDRejected verifies that a session ID
// which cannot be a safe path segment is rejected on both the save and the
// lookup path - the session ID names the pointer file, so it is held to the same
// rule as snapshot IDs and prefixes. The save is rejected before any file is
// written, so a bad ID never leaves a half-written row.
func TestFileSessionStore_PathUnsafeSessionIDRejected(t *testing.T) {
	store := newFileStore(t)
	ctx := context.Background()

	for _, sessionID := range []string{"a/b", "../evil", ".hidden", `a\b`, "foo..bar"} {
		t.Run(sessionID, func(t *testing.T) {
			// SaveSnapshot rejects the path-unsafe session ID...
			if _, err := store.SaveSnapshot(ctx, "fixed-id",
				func(_ *exp.SessionSnapshot[testState]) (*exp.SessionSnapshot[testState], error) {
					return &exp.SessionSnapshot[testState]{
						SessionID: sessionID,
						Status:    exp.SnapshotStatusCompleted,
						State:     &exp.SessionState[testState]{Custom: testState{Counter: 7}},
						CreatedAt: time.Now(),
						UpdatedAt: time.Now(),
					}, nil
				}); err == nil {
				t.Errorf("SaveSnapshot(sessionID=%q): expected error, got nil", sessionID)
			}
			// ...before writing anything: no partial row is left behind.
			if snap, _ := store.GetSnapshot(ctx, "fixed-id"); snap != nil {
				t.Errorf("rejected save left a row on disk: %+v", snap)
			}
			// GetLatestSnapshot rejects it too.
			if _, err := store.GetLatestSnapshot(ctx, sessionID); err == nil {
				t.Errorf("GetLatestSnapshot(%q): expected error, got nil", sessionID)
			}
		})
	}
}
