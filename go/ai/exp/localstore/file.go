// Copyright 2025 Google LLC
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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai/exp"
	"github.com/google/uuid"
)

// FileSessionStore is a snapshot store that persists snapshots as JSON files on
// the local filesystem. Each snapshot is written to its own file named
// "<snapshotID>.json", under an optional per-call subdirectory ("prefix"):
//
//	<dir>/<prefix>/<snapshotID>.json
//
// The snapshot ID is the primary key: GetSnapshot, the by-ID SaveSnapshot
// (heartbeat, abort, finalize), and OnSnapshotStatusChange all open that file
// directly. GetLatestSnapshot, the only by-session lookup, scans the prefix
// directory and selects the most-recently-created row for the session. The
// prefix is derived from each call's context (see [WithSnapshotPathPrefix]), so
// it is always known on a by-ID call - unlike the session ID, which a by-ID
// caller does not have. That is why snapshots are grouped by prefix and kept
// flat within it rather than nested under a per-session directory.
//
// The store is safe for concurrent use within a single process, but does NOT
// coordinate with other processes sharing the directory: the last successful
// rename wins, and a reader may briefly observe a snapshot another process is
// still writing. [FileSessionStore.OnSnapshotStatusChange] likewise reflects
// only status changes made through this instance.
type FileSessionStore[State any] struct {
	// mu serializes the read-modify-write paths and the subscriber bookkeeping.
	// File I/O happens under the lock; this matches the simplicity of
	// [InMemorySessionStore] and is adequate when writes are infrequent
	// (typically once per turn).
	mu  sync.Mutex
	dir string
	// maxChain, when > 0, bounds how many snapshots a single conversation chain
	// retains on disk; see [WithMaxPersistedChainLength].
	maxChain int
	// prefixFn, when set, derives the per-call subdirectory from context; see
	// [WithSnapshotPathPrefix].
	prefixFn func(context.Context) string
	subs     map[string][]chan exp.SnapshotStatus
}

// NewFileSessionStore creates a file-based snapshot store rooted at dir.
// The directory is created (mode 0o700) if it does not already exist.
// Returns an error if dir is empty, cannot be created, or an option is set
// more than once. See [WithMaxPersistedChainLength] and
// [WithSnapshotPathPrefix].
func NewFileSessionStore[State any](dir string, opts ...FileStoreOption) (*FileSessionStore[State], error) {
	if dir == "" {
		return nil, errors.New("FileSessionStore: dir is required")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("FileSessionStore: create dir %q: %w", dir, err)
	}
	var resolved fileStoreOptions
	for _, o := range opts {
		if err := o.applyFileStore(&resolved); err != nil {
			return nil, err
		}
	}
	maxChain := 0
	if resolved.maxChain != nil {
		maxChain = *resolved.maxChain
	}
	return &FileSessionStore[State]{
		dir:      dir,
		maxChain: maxChain,
		prefixFn: resolved.prefixFn,
		subs:     make(map[string][]chan exp.SnapshotStatus),
	}, nil
}

// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
func (s *FileSessionStore[State]) GetSnapshot(ctx context.Context, snapshotID string) (*exp.SessionSnapshot[State], error) {
	if err := validateSnapshotID(snapshotID); err != nil {
		return nil, err
	}
	prefix, err := s.derivePrefix(ctx)
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readAt(s.pathFor(prefix, snapshotID))
}

// SaveSnapshot atomically reads, applies fn, and persists. See
// [exp.SnapshotWriter] for the full contract; this implementation calls fn
// exactly once per call.
func (s *FileSessionStore[State]) SaveSnapshot(
	ctx context.Context,
	id string,
	fn func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error),
) (*exp.SessionSnapshot[State], error) {
	if id == "" {
		id = uuid.New().String()
	} else if err := validateSnapshotID(id); err != nil {
		return nil, err
	}
	prefix, err := s.derivePrefix(ctx)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	existing, err := s.readAt(s.pathFor(prefix, id))
	if err != nil {
		return nil, err
	}

	next, err := fn(existing)
	if err != nil {
		return nil, err
	}
	if next == nil {
		return nil, nil
	}

	next.SnapshotID = id
	// SessionID is preserved (a row's session never changes); CreatedAt,
	// UpdatedAt, and HeartbeatAt are caller-managed and persisted verbatim.
	if existing != nil && existing.SessionID != "" {
		next.SessionID = existing.SessionID
	}
	if next.Status == "" {
		next.Status = exp.SnapshotStatusCompleted
	}

	if err := s.writeAt(prefix, next); err != nil {
		return nil, err
	}
	if existing == nil || existing.Status != next.Status {
		s.notifyLocked(id, next.Status)
	}
	if s.maxChain > 0 {
		s.pruneLocked(prefix, next)
	}
	return next, nil
}

// snapshotHeader is the subset of snapshot fields needed to pick a session's
// latest row during the scan. Decoding only these avoids materializing every
// row's full conversation state; only the winning row is fully decoded.
type snapshotHeader struct {
	SessionID string    `json:"sessionId"`
	CreatedAt time.Time `json:"createdAt"`
}

// GetLatestSnapshot returns the session's most recently created snapshot
// regardless of status, per the [exp.SnapshotReader.GetLatestSnapshot]
// contract. It scans the call's prefix directory (see [WithSnapshotPathPrefix]),
// so a session is only resolvable under the prefix it was written with.
//
// Recency is judged by the [exp.SessionSnapshot.CreatedAt] field (not file
// mtime), so a later rewrite of an older row - which preserves CreatedAt - does
// not move it ahead of a newer-created sibling. Ties are broken by snapshot ID.
// A file that fails to parse or vanishes mid-scan is skipped, so one corrupted
// row cannot hide every other session.
func (s *FileSessionStore[State]) GetLatestSnapshot(ctx context.Context, sessionID string) (*exp.SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("FileSessionStore: session ID is empty")
	}
	prefix, err := s.derivePrefix(ctx)
	if err != nil {
		return nil, err
	}
	dir := s.prefixDir(prefix)
	names, err := s.snapshotFileNames(dir)
	if err != nil {
		return nil, err
	}
	var (
		bestName string
		bestAt   time.Time
		found    bool
	)
	for _, name := range names {
		s.mu.Lock()
		data, err := os.ReadFile(filepath.Join(dir, name))
		s.mu.Unlock()
		if err != nil {
			continue
		}
		var h snapshotHeader
		if err := json.Unmarshal(data, &h); err != nil {
			continue
		}
		if h.SessionID != sessionID {
			continue
		}
		// Most recently created wins; the file name is "<snapshotId>.json", so
		// a name compare is a deterministic SnapshotID tie-break.
		if !found || h.CreatedAt.After(bestAt) ||
			(h.CreatedAt.Equal(bestAt) && name > bestName) {
			bestName, bestAt, found = name, h.CreatedAt, true
		}
	}
	if !found {
		return nil, nil
	}
	// Fully decode only the winner. CreatedAt is preserved across rewrites, so a
	// concurrent rewrite of this row between scan and read still yields the
	// right snapshot (with possibly fresher state). A parse failure here is
	// treated like a vanished row: report no tip rather than erroring.
	s.mu.Lock()
	snap, _ := s.readAt(filepath.Join(dir, bestName))
	s.mu.Unlock()
	if snap == nil {
		return nil, nil
	}
	return snap, nil
}

// snapshotFileNames returns the names of dir's snapshot files (non-directory
// *.json entries; writeAt's "<id>.*.tmp" temp files never match). Returns nil if
// the directory does not exist. The listing is not atomic with respect to
// concurrent writes; a snapshot that appears or disappears mid-scan may or may
// not be observed.
func (s *FileSessionStore[State]) snapshotFileNames(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("FileSessionStore: list dir: %w", err)
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		names = append(names, e.Name())
	}
	return names, nil
}

// OnSnapshotStatusChange subscribes to status changes for a snapshot. The
// returned channel yields the current status (if any) and any subsequent
// changes triggered by calls on this store instance, until ctx is cancelled.
// Changes made by other processes writing to the same directory are not
// observed.
func (s *FileSessionStore[State]) OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan exp.SnapshotStatus {
	ch := make(chan exp.SnapshotStatus, 1)
	if err := validateSnapshotID(snapshotID); err != nil {
		close(ch)
		return ch
	}
	prefix, err := s.derivePrefix(ctx)
	if err != nil {
		close(ch)
		return ch
	}

	s.mu.Lock()
	snap, err := s.readAt(s.pathFor(prefix, snapshotID))
	if err != nil || snap == nil {
		s.mu.Unlock()
		close(ch)
		return ch
	}
	ch <- snap.Status
	s.subs[snapshotID] = append(s.subs[snapshotID], ch)
	s.mu.Unlock()

	context.AfterFunc(ctx, func() { s.removeSub(snapshotID, ch) })
	return ch
}

// derivePrefix resolves the per-call subdirectory snapshots live under by
// invoking the configured prefix function (if any) and sanitizing its result.
// Returns "" (the store root) when no function is configured or it yields an
// empty value.
func (s *FileSessionStore[State]) derivePrefix(ctx context.Context) (string, error) {
	if s.prefixFn == nil {
		return "", nil
	}
	return sanitizePrefix(s.prefixFn(ctx))
}

// readAt reads and parses the snapshot file at path. Returns (nil, nil) if the
// file does not exist. Caller must hold s.mu.
func (s *FileSessionStore[State]) readAt(path string) (*exp.SessionSnapshot[State], error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("FileSessionStore: read %s: %w", path, err)
	}
	var snap exp.SessionSnapshot[State]
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, fmt.Errorf("FileSessionStore: unmarshal %s: %w", path, err)
	}
	return &snap, nil
}

// writeAt atomically writes snap to <prefix>/<id>.json via a temp file +
// rename, creating the prefix directory as needed. Caller must hold s.mu.
func (s *FileSessionStore[State]) writeAt(prefix string, snap *exp.SessionSnapshot[State]) error {
	dir := s.prefixDir(prefix)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("FileSessionStore: create dir: %w", err)
	}
	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return fmt.Errorf("FileSessionStore: marshal: %w", err)
	}
	f, err := os.CreateTemp(dir, snap.SnapshotID+".*.tmp")
	if err != nil {
		return fmt.Errorf("FileSessionStore: create temp: %w", err)
	}
	tmpName := f.Name()
	// Best-effort cleanup if anything fails before the rename succeeds.
	// Once renamed, the temp file no longer exists so Remove is a no-op.
	defer os.Remove(tmpName)

	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("FileSessionStore: write: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("FileSessionStore: sync: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("FileSessionStore: close: %w", err)
	}
	if err := os.Rename(tmpName, s.pathFor(prefix, snap.SnapshotID)); err != nil {
		return fmt.Errorf("FileSessionStore: rename: %w", err)
	}
	return nil
}

// pruneLocked enforces maxChain by walking the parentId chain back from the
// just-written snapshot and unlinking every row past the newest maxChain
// entries. Parents live in the same prefix directory, addressed directly by ID,
// so no scan is needed. A broken or cyclic chain stops the walk early (a visited
// set guards against cycles). Deletion is best-effort: a failed unlink leaves a
// stale row rather than failing the already-committed save. Caller holds s.mu
// and has checked maxChain > 0.
func (s *FileSessionStore[State]) pruneLocked(prefix string, tip *exp.SessionSnapshot[State]) {
	chain := []string{tip.SnapshotID}
	seen := map[string]bool{tip.SnapshotID: true}
	for cur := tip; cur.ParentID != "" && !seen[cur.ParentID]; {
		parent, err := s.readAt(s.pathFor(prefix, cur.ParentID))
		if err != nil || parent == nil {
			break
		}
		seen[cur.ParentID] = true
		chain = append(chain, cur.ParentID)
		cur = parent
	}
	for _, oldID := range chain[min(s.maxChain, len(chain)):] {
		_ = os.Remove(s.pathFor(prefix, oldID))
	}
}

// prefixDir returns the on-disk directory snapshots under prefix are stored in.
func (s *FileSessionStore[State]) prefixDir(prefix string) string {
	return filepath.Join(s.dir, prefix)
}

// pathFor returns the on-disk path for a snapshot ID under prefix. Both are
// assumed validated (by validateSnapshotID and sanitizePrefix).
func (s *FileSessionStore[State]) pathFor(prefix, snapshotID string) string {
	return filepath.Join(s.dir, prefix, snapshotID+".json")
}

// removeSub detaches a subscriber and closes its channel.
func (s *FileSessionStore[State]) removeSub(snapshotID string, ch chan exp.SnapshotStatus) {
	s.mu.Lock()
	defer s.mu.Unlock()
	subs := s.subs[snapshotID]
	i := slices.Index(subs, ch)
	if i < 0 {
		return
	}
	subs = slices.Delete(subs, i, i+1)
	if len(subs) == 0 {
		delete(s.subs, snapshotID)
	} else {
		s.subs[snapshotID] = subs
	}
	close(ch)
}

// notifyLocked publishes status to all live subscribers of snapshotID.
// Caller must hold s.mu. A slow subscriber may miss intermediate values, but
// the latest value is always delivered (see [coalesceSend]).
func (s *FileSessionStore[State]) notifyLocked(snapshotID string, status exp.SnapshotStatus) {
	for _, ch := range s.subs[snapshotID] {
		coalesceSend(ch, status)
	}
}

// sanitizePrefix turns a raw prefix (which may contain "/" to nest
// subdirectories) into a cleaned relative path under the store root, rejecting
// any value that could escape it. Empty and separator-only inputs yield "".
func sanitizePrefix(raw string) (string, error) {
	if strings.Contains(raw, `\`) {
		return "", fmt.Errorf("FileSessionStore: path prefix %q must use '/' separators", raw)
	}
	var segs []string
	for _, seg := range strings.Split(raw, "/") {
		if seg == "" {
			continue // collapse empty segments (leading/trailing/double slash)
		}
		if err := validatePathSegment(seg); err != nil {
			return "", fmt.Errorf("FileSessionStore: invalid path prefix %q: %w", raw, err)
		}
		segs = append(segs, seg)
	}
	return filepath.Join(segs...), nil
}

// validateSnapshotID rejects IDs that would escape the store directory or
// collide with reserved filenames; the ID is used directly as a file name.
// UUIDs (the default produced by an empty id) pass trivially.
func validateSnapshotID(id string) error {
	if err := validatePathSegment(id); err != nil {
		return fmt.Errorf("FileSessionStore: invalid snapshot ID %q: %w", id, err)
	}
	return nil
}

// validatePathSegment rejects a value that cannot serve as a single on-disk path
// component without risking directory escape or hidden/reserved-name collisions.
func validatePathSegment(s string) error {
	if s == "" {
		return errors.New("empty")
	}
	if strings.ContainsAny(s, `/\`) || strings.Contains(s, "..") {
		return errors.New("contains path separators")
	}
	if strings.HasPrefix(s, ".") {
		return errors.New("must not start with '.'")
	}
	// Disallow NUL and control characters that some filesystems reject.
	for _, r := range s {
		if r < 0x20 {
			return errors.New("contains control characters")
		}
	}
	return nil
}
