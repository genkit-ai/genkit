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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core"
	"github.com/google/uuid"
)

// FileSessionStore is a snapshot store that persists snapshots as JSON files on
// the local filesystem. Each snapshot is written to its own file named
// "<snapshotID>.json", under a per-call subdirectory ("prefix"):
//
//	<dir>/<prefix>/<snapshotID>.json
//
// The prefix is derived from each call's context (see [WithSnapshotPathPrefix])
// and defaults to "global" when none is configured; snapshots therefore always
// live under a subdirectory, never directly in the store root. That default
// places every session in one shared "global" directory, so pass
// [WithSnapshotPathPrefix] to scope them per tenant when identifiers could
// repeat across users (e.g. per-user session IDs).
//
// The snapshot ID is the primary key: GetSnapshot, the by-ID SaveSnapshot
// (heartbeat, abort, finalize), and OnSnapshotStatusChange all open that file
// directly. GetLatestSnapshot, the only by-session lookup, scans the prefix
// directory and selects the most-recently-created row for the session. The
// prefix is always known on a by-ID call - unlike the session ID, which a by-ID
// caller does not have. That is why snapshots are grouped by prefix and kept
// flat within it rather than nested under a per-session directory.
//
// The store is safe for concurrent use, and [FileSessionStore.OnSnapshotStatusChange]
// surfaces status changes written by other processes (or other store instances)
// sharing the directory by polling the snapshot files on an interval (see
// [WithPollInterval]); that cross-process visibility is what lets one process
// abort a detached turn another process is running. The store still does not
// provide cross-process transactions: the last successful rename wins. Each
// write is atomic (temp file + rename), so a concurrent reader sees either the
// old file or the new one, never a torn write.
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
	// poll is the interval at which the background poller re-reads subscribed
	// snapshot files to detect cross-process status changes; <= 0 disables it.
	poll time.Duration
	subs map[string]*snapshotSubs
	// pollCancel stops the background poller. It is non-nil exactly while the
	// poller runs: while at least one subscription is active and poll > 0.
	pollCancel context.CancelFunc
}

// snapshotSubs holds the live subscribers to one snapshot plus the state the
// poller needs to surface its status changes.
type snapshotSubs struct {
	chans []chan exp.SnapshotStatus
	// path is the file the poller re-reads, captured at subscription time
	// because the snapshot's prefix is derived from the subscriber's context
	// and is not otherwise known on a by-poll re-read.
	path string
	// last is the status most recently delivered to chans (seeded at the first
	// subscription). It is the single dedup gate shared by the in-process write
	// path and the poller, so a change is delivered once regardless of which
	// observes it first.
	last exp.SnapshotStatus
}

// defaultPollInterval is how often the poller re-reads subscribed snapshot
// files when no interval is configured. It sits well below the agent heartbeat
// interval so an operator-driven abort propagates promptly while the idle I/O
// cost stays negligible.
const defaultPollInterval = time.Second * 2

// defaultPrefix is the per-call subdirectory snapshots are written under when no
// [WithSnapshotPathPrefix] is configured, so snapshots always live at least one
// level below the store root rather than directly in it. A configured function
// must return a non-empty value: the default is requested by omitting the
// option, not by returning an empty prefix.
const defaultPrefix = "global"

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
	poll := defaultPollInterval
	if resolved.poll != nil {
		poll = *resolved.poll
	}
	return &FileSessionStore[State]{
		dir:      dir,
		maxChain: maxChain,
		prefixFn: resolved.prefixFn,
		poll:     poll,
		subs:     make(map[string]*snapshotSubs),
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
	if next.SessionID == "" {
		// A snapshot must belong to a session; stores never mint or infer one. The
		// runtime stamps a session ID on every row it writes, so an empty one
		// indicates misuse.
		return nil, core.NewError(core.INVALID_ARGUMENT, "FileSessionStore requires sessionId to be set on the snapshot")
	}
	if next.Status == "" {
		next.Status = exp.SnapshotStatusCompleted
	}

	if err := s.writeAt(prefix, next); err != nil {
		return nil, err
	}
	s.maybeNotifyLocked(id, next.Status)
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
// returned channel yields the status at subscription time and every subsequent
// change until ctx is cancelled. A change is surfaced immediately when written
// through this store instance, and within one poll interval when written by
// another process sharing the directory (see [WithPollInterval]); polling
// re-reads the file the snapshot was found under at subscription time. If the
// snapshot does not exist at subscription time, the channel is closed without
// yielding a value.
//
// Values are level-triggered: the latest status is always delivered, but a slow
// reader may skip intermediate values, and a subscriber that joins concurrently
// with a change may observe that status twice. Treat a received value as "the
// status is now X", not "X just happened once".
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
	path := s.pathFor(prefix, snapshotID)

	s.mu.Lock()
	snap, err := s.readAt(path)
	if err != nil || snap == nil {
		s.mu.Unlock()
		close(ch)
		return ch
	}
	ch <- snap.Status
	sub := s.subs[snapshotID]
	if sub == nil {
		// First subscriber: seed the dedup baseline with the current status and
		// remember the path so the poller can re-read it.
		sub = &snapshotSubs{path: path, last: snap.Status}
		s.subs[snapshotID] = sub
	}
	sub.chans = append(sub.chans, ch)
	s.startPollerLocked()
	s.mu.Unlock()

	context.AfterFunc(ctx, func() { s.removeSub(snapshotID, ch) })
	return ch
}

// derivePrefix resolves the per-call subdirectory snapshots live under. With no
// prefix function configured it returns [defaultPrefix] ("global"); otherwise it
// sanitizes the function's result and requires it to be non-empty. As with every
// other option, the default is requested by omitting [WithSnapshotPathPrefix],
// not by returning an empty value from it, so an empty or separator-only result
// is rejected rather than silently mapped to the default.
func (s *FileSessionStore[State]) derivePrefix(ctx context.Context) (string, error) {
	if s.prefixFn == nil {
		return defaultPrefix, nil
	}
	raw := s.prefixFn(ctx)
	prefix, err := sanitizePrefix(raw)
	if err != nil {
		return "", err
	}
	if prefix == "" {
		return "", fmt.Errorf("FileSessionStore: WithSnapshotPathPrefix returned %q, which is empty after normalization; omit the option to use the default %q prefix", raw, defaultPrefix)
	}
	return prefix, nil
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

// removeSub detaches a subscriber and closes its channel, dropping the
// snapshot's bookkeeping (and stopping the poller) once no subscribers remain.
func (s *FileSessionStore[State]) removeSub(snapshotID string, ch chan exp.SnapshotStatus) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sub := s.subs[snapshotID]
	if sub == nil {
		return
	}
	i := slices.Index(sub.chans, ch)
	if i < 0 {
		return
	}
	sub.chans = slices.Delete(sub.chans, i, i+1)
	if len(sub.chans) == 0 {
		delete(s.subs, snapshotID)
	}
	if len(s.subs) == 0 {
		s.stopPollerLocked()
	}
	close(ch)
}

// maybeNotifyLocked fans status out to snapshotID's subscribers, but only when
// it differs from the value they last saw. It is the shared dedup gate for both
// the in-process write path ([SaveSnapshot]) and the cross-process poller, so
// whichever observes a change first delivers it once and the other is a no-op.
// Caller must hold s.mu. A slow subscriber may miss intermediate values, but the
// latest is always delivered (see [coalesceSend]).
func (s *FileSessionStore[State]) maybeNotifyLocked(snapshotID string, status exp.SnapshotStatus) {
	sub := s.subs[snapshotID]
	if sub == nil || sub.last == status {
		return
	}
	sub.last = status
	for _, ch := range sub.chans {
		coalesceSend(ch, status)
	}
}

// startPollerLocked launches the background poller if it is not already running
// and polling is enabled. Caller must hold s.mu.
func (s *FileSessionStore[State]) startPollerLocked() {
	if s.pollCancel != nil || s.poll <= 0 {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	s.pollCancel = cancel
	go s.pollLoop(ctx, s.poll)
}

// stopPollerLocked signals the poller to exit. Caller must hold s.mu. It does
// not wait: the goroutine observes the cancellation and returns promptly, and
// the shared dedup gate keeps a briefly-overlapping successor poller correct.
func (s *FileSessionStore[State]) stopPollerLocked() {
	if s.pollCancel == nil {
		return
	}
	s.pollCancel()
	s.pollCancel = nil
}

// pollLoop re-reads every subscribed snapshot on each tick until ctx is
// cancelled, delivering status changes written by other processes (or other
// store instances) sharing the directory - the only way such changes reach
// subscribers, since a cross-process write never runs this instance's
// in-process notification.
func (s *FileSessionStore[State]) pollLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.pollOnce()
		}
	}
}

// pollOnce re-reads each subscribed snapshot's file and delivers any status
// change through the shared dedup gate. It snapshots the subscription set under
// the lock, then reads each file under its own lock acquisition, so a tick never
// blocks a write for longer than a single file read and the read pairs
// atomically with the dedup check (ruling out delivering a status the file no
// longer holds). A read error or vanished file is skipped.
func (s *FileSessionStore[State]) pollOnce() {
	type target struct{ id, path string }
	s.mu.Lock()
	targets := make([]target, 0, len(s.subs))
	for id, sub := range s.subs {
		targets = append(targets, target{id: id, path: sub.path})
	}
	s.mu.Unlock()

	for _, t := range targets {
		s.mu.Lock()
		if snap, err := s.readAt(t.path); err == nil && snap != nil {
			s.maybeNotifyLocked(t.id, snap.Status)
		}
		s.mu.Unlock()
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
