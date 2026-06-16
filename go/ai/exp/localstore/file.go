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

// FileSessionStore is a snapshot store that persists snapshots as JSON files
// on the local filesystem. Each snapshot is written to its own file named
// "<snapshotID>.json" in the configured directory.
//
// The store is safe for concurrent use within a single process. It does NOT
// coordinate writes with other processes that may share the same directory:
// the only synchronization is the per-instance mutex. If multiple processes
// write to the same directory the last successful rename wins; readers may
// also observe a brief window during which a snapshot is still being written
// by another process (the rename itself is atomic, but cross-process
// linearization is not guaranteed).
//
// [FileSessionStore.OnSnapshotStatusChange] uses in-process channels and only
// reflects status transitions caused by calls on this store instance.
// External writes to the directory and writes from other processes are not
// observed.
type FileSessionStore[State any] struct {
	// mu serializes the read-modify-write paths and the subscriber bookkeeping.
	// File I/O happens under the lock; this matches the simplicity of
	// [InMemorySessionStore] and is adequate when writes are infrequent
	// (typically once per turn).
	mu   sync.Mutex
	dir  string
	subs map[string][]chan exp.SnapshotStatus
}

// NewFileSessionStore creates a file-based snapshot store rooted at dir.
// The directory is created (mode 0o700) if it does not already exist.
// Returns an error if dir is empty or cannot be created.
func NewFileSessionStore[State any](dir string) (*FileSessionStore[State], error) {
	if dir == "" {
		return nil, errors.New("FileSessionStore: dir is required")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("FileSessionStore: create dir %q: %w", dir, err)
	}
	return &FileSessionStore[State]{
		dir:  dir,
		subs: make(map[string][]chan exp.SnapshotStatus),
	}, nil
}

// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
func (s *FileSessionStore[State]) GetSnapshot(_ context.Context, snapshotID string) (*exp.SessionSnapshot[State], error) {
	if err := validateSnapshotID(snapshotID); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.readLocked(snapshotID)
}

// SaveSnapshot atomically reads, applies fn, and persists. See the
// [exp.SnapshotWriter] interface for the full contract; this implementation
// satisfies it by holding s.mu for the entire read-modify-write so fn is
// called exactly once per SaveSnapshot call.
func (s *FileSessionStore[State]) SaveSnapshot(
	_ context.Context,
	id string,
	fn func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error),
) (*exp.SessionSnapshot[State], error) {
	if id == "" {
		id = uuid.New().String()
	} else if err := validateSnapshotID(id); err != nil {
		return nil, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	existing, err := s.readLocked(id)
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
	now := time.Now()
	if existing != nil {
		next.CreatedAt = existing.CreatedAt
		if existing.SessionID != "" {
			next.SessionID = existing.SessionID // a row's session never changes
		}
	} else {
		next.CreatedAt = now
	}
	next.UpdatedAt = now
	if next.Status == "" {
		next.Status = exp.SnapshotStatusSucceeded
	}

	if err := s.writeLocked(next); err != nil {
		return nil, err
	}
	if existing == nil || existing.Status != next.Status {
		s.notifyLocked(id, next.Status)
	}
	return next, nil
}

// snapshotHeader is the subset of snapshot fields needed to decide
// whether a row resolves a session resume. Decoding only these avoids
// materializing every row's full conversation state during the scan.
type snapshotHeader struct {
	SessionID string             `json:"sessionId"`
	Status    exp.SnapshotStatus `json:"status"`
}

// GetLatestSnapshot returns the session's most recently updated snapshot
// that is not a failed/aborted dead end, per the
// [exp.SnapshotReader.GetLatestSnapshot] contract.
//
// Recency is judged by file mtime, which for snapshots written by this
// package advances with [exp.SessionSnapshot.UpdatedAt] (each save
// creates a fresh temp file and renames it into place); if a file is
// touched externally, mtime wins. The scan walks files newest first and
// stops at the first row that matches, so resolving the most recently
// active session costs one read in the common case. Only header fields
// are decoded per candidate (the winner is the only full parse), the
// store lock is held per file rather than across the whole scan, and a
// file that vanishes mid-scan or fails to parse is skipped so one
// corrupted row cannot poison every session in the directory.
func (s *FileSessionStore[State]) GetLatestSnapshot(_ context.Context, sessionID string) (*exp.SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("FileSessionStore: session ID is empty")
	}
	names, err := s.snapshotFilesNewestFirst()
	if err != nil {
		return nil, err
	}
	for _, name := range names {
		s.mu.Lock()
		data, err := os.ReadFile(filepath.Join(s.dir, name))
		s.mu.Unlock()
		if err != nil {
			continue
		}
		var h snapshotHeader
		if err := json.Unmarshal(data, &h); err != nil {
			continue
		}
		if h.SessionID != sessionID ||
			h.Status == exp.SnapshotStatusFailed || h.Status == exp.SnapshotStatusAborted {
			continue
		}
		var snap exp.SessionSnapshot[State]
		if err := json.Unmarshal(data, &snap); err != nil {
			continue
		}
		return &snap, nil
	}
	return nil, nil
}

// snapshotFilesNewestFirst returns the names of the directory's snapshot
// files (non-directory *.json entries; writeLocked's "<id>.*.tmp" temp
// files never match) sorted by modification time, newest first, with
// name as a deterministic tie-break. Entries that vanish between the
// directory read and the stat are skipped. Returns nil if the directory
// does not exist. The listing is not atomic with respect to concurrent
// writes; a snapshot that appears or disappears mid-scan may or may not
// be observed.
func (s *FileSessionStore[State]) snapshotFilesNewestFirst() ([]string, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("FileSessionStore: list dir: %w", err)
	}
	type candidate struct {
		name    string
		modTime time.Time
	}
	var cands []candidate
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		cands = append(cands, candidate{e.Name(), info.ModTime()})
	}
	slices.SortFunc(cands, func(a, b candidate) int {
		if c := b.modTime.Compare(a.modTime); c != 0 { // newest first
			return c
		}
		return strings.Compare(b.name, a.name)
	})
	names := make([]string, len(cands))
	for i, c := range cands {
		names[i] = c.name
	}
	return names, nil
}

// AbortSnapshot atomically flips a pending snapshot to aborted. If the
// snapshot is already terminal the existing status is returned unchanged.
// Returns an empty status if the snapshot is not found.
func (s *FileSessionStore[State]) AbortSnapshot(_ context.Context, snapshotID string) (exp.SnapshotStatus, error) {
	if err := validateSnapshotID(snapshotID); err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	snap, err := s.readLocked(snapshotID)
	if err != nil {
		return "", err
	}
	if snap == nil {
		return "", nil
	}
	if snap.Status == exp.SnapshotStatusPending {
		snap.Status = exp.SnapshotStatusAborted
		snap.UpdatedAt = time.Now()
		if err := s.writeLocked(snap); err != nil {
			return "", err
		}
		s.notifyLocked(snapshotID, snap.Status)
	}
	return snap.Status, nil
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

	s.mu.Lock()
	snap, err := s.readLocked(snapshotID)
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

// readLocked reads and parses the snapshot file. Returns (nil, nil) if the
// file does not exist. Caller must hold s.mu.
func (s *FileSessionStore[State]) readLocked(snapshotID string) (*exp.SessionSnapshot[State], error) {
	data, err := os.ReadFile(s.path(snapshotID))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("FileSessionStore: read %s: %w", snapshotID, err)
	}
	var snap exp.SessionSnapshot[State]
	if err := json.Unmarshal(data, &snap); err != nil {
		return nil, fmt.Errorf("FileSessionStore: unmarshal %s: %w", snapshotID, err)
	}
	return &snap, nil
}

// writeLocked atomically writes the snapshot to disk via a temp file +
// rename. Caller must hold s.mu.
func (s *FileSessionStore[State]) writeLocked(snap *exp.SessionSnapshot[State]) error {
	data, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return fmt.Errorf("FileSessionStore: marshal: %w", err)
	}
	f, err := os.CreateTemp(s.dir, snap.SnapshotID+".*.tmp")
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
	if err := os.Rename(tmpName, s.path(snap.SnapshotID)); err != nil {
		return fmt.Errorf("FileSessionStore: rename: %w", err)
	}
	return nil
}

// path returns the on-disk path for a snapshot ID. The ID is assumed to have
// been validated by validateSnapshotID.
func (s *FileSessionStore[State]) path(snapshotID string) string {
	return filepath.Join(s.dir, snapshotID+".json")
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

// validateSnapshotID rejects IDs that would escape the store directory or
// collide with reserved filenames. UUIDs (the default produced by an empty
// id) pass trivially.
func validateSnapshotID(id string) error {
	if id == "" {
		return errors.New("FileSessionStore: snapshot ID is empty")
	}
	if strings.ContainsAny(id, `/\`) || strings.Contains(id, "..") {
		return fmt.Errorf("FileSessionStore: snapshot ID %q contains path separators", id)
	}
	if strings.HasPrefix(id, ".") {
		return fmt.Errorf("FileSessionStore: snapshot ID %q must not start with '.'", id)
	}
	// Disallow NUL and control characters that some filesystems reject.
	for _, r := range id {
		if r < 0x20 {
			return fmt.Errorf("FileSessionStore: snapshot ID %q contains control characters", id)
		}
	}
	return nil
}
