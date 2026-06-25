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

// Package exp provides experimental Firebase integrations for Genkit's agent
// runtime (see [github.com/firebase/genkit/go/ai/exp]).
//
// The [FirestoreSessionStore] persists agent session snapshots in Cloud
// Firestore. It resolves its Firestore client from the Firebase plugin
// registered with the Genkit instance, then wires into an agent:
//
//	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: "my-project"}))
//
//	store, err := exp.NewFirestoreSessionStore[MyState](ctx, g)
//	// handle err
//
//	agent := aix.DefineAgent(g, "assistant", run, aix.WithSessionStore(store))
//
// APIs in this package are under active development and may change in any minor
// version release. Use with caution in production environments.
package exp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/firestore"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"github.com/google/uuid"
)

// Document "kind" values for a snapshot document.
const (
	kindDiff       = "diff"
	kindCheckpoint = "checkpoint"
)

// Compile-time checks that the store satisfies the session-store interfaces.
var (
	_ aix.SessionStore[any]  = (*FirestoreSessionStore[any])(nil)
	_ aix.SnapshotSubscriber = (*FirestoreSessionStore[any])(nil)
)

// FirestoreSessionStore is a Firestore-backed [aix.SessionStore] that persists
// session snapshots as incremental JSON Patch diffs anchored to periodic,
// sharded full-state checkpoints.
//
// Storage layout (the <prefix> segment is the per-tenant prefix returned by
// [WithSnapshotPathPrefix], or "global" when none is configured):
//
//   - <collection>/<prefix>/snapshots/<snapshotID> - one document per snapshot.
//     A "diff" document holds the JSON Patch from its parent (statePatch); a
//     "checkpoint" document holds a full-state materialization, sharded out of
//     band.
//   - <collection>-shards/<prefix>/shards/<checkpointID>_<index> - the sharded
//     full state for a checkpoint.
//   - <collection>-pointers/<prefix>/pointers/<sessionID> - one document per
//     session pointing at its latest snapshot and the metadata needed to
//     reconstruct it.
//
// Reconstruction uses only document-ID lookups (GetAll), so it needs no
// secondary indexes and is strongly consistent. No single document approaches
// the 1 MiB limit (state is sharded by shard size), and the number of diff
// documents touched per read or write is bounded by the checkpoint interval
// rather than total session length, so the store scales to arbitrarily long
// sessions. Checkpoints still store the full accumulated state, so checkpoint
// shard count grows with the state's size; tune [WithCheckpointInterval] to
// trade per-save diff reads against checkpoint write amplification.
//
// It implements [aix.SessionStore] and [aix.SnapshotSubscriber]; the latter
// uses Firestore's native real-time listener, so an abort committed by one
// process is observed by the process running the detached turn even across
// instances.
type FirestoreSessionStore[State any] struct {
	client             *firestore.Client
	collection         string
	checkpointInterval int
	shardSize          int
	prefixFn           func(context.Context) string
}

// NewFirestoreSessionStore creates a Firestore-backed snapshot store. It
// resolves the Firestore client from the Firebase plugin registered with g (the
// Firebase plugin must be passed to genkit.Init before calling this), mirroring
// [github.com/firebase/genkit/go/plugins/firebase/exp.NewFirestoreStreamManager].
//
// The State type parameter is the user-defined custom-state type carried in
// [aix.SessionState.Custom]; it must be JSON-serializable.
func NewFirestoreSessionStore[State any](ctx context.Context, g *genkit.Genkit, opts ...SessionStoreOption) (*FirestoreSessionStore[State], error) {
	client, err := resolveFirestoreClient(ctx, g)
	if err != nil {
		return nil, fmt.Errorf("firebase.NewFirestoreSessionStore: %w", err)
	}
	store, err := newFirestoreSessionStore[State](client, opts...)
	if err != nil {
		return nil, fmt.Errorf("firebase.NewFirestoreSessionStore: %w", err)
	}
	return store, nil
}

// newFirestoreSessionStore builds the store from a resolved Firestore client. It
// is separated from the public constructor so the store logic can be exercised
// against the Firestore emulator without standing up the full Firebase plugin.
func newFirestoreSessionStore[State any](client *firestore.Client, opts ...SessionStoreOption) (*FirestoreSessionStore[State], error) {
	if client == nil {
		return nil, errors.New("a Firestore client is required")
	}
	var cfg sessionStoreOptions
	for _, o := range opts {
		if err := o.applySessionStore(&cfg); err != nil {
			return nil, err
		}
	}
	collection := cfg.collection
	if collection == "" {
		collection = defaultCollection
	}
	checkpointInterval := cfg.checkpointInterval
	if checkpointInterval == 0 {
		checkpointInterval = defaultCheckpointInterval
	}
	shardSize := cfg.shardSize
	if shardSize == 0 {
		shardSize = defaultShardSize
	}
	return &FirestoreSessionStore[State]{
		client:             client,
		collection:         collection,
		checkpointInterval: checkpointInterval,
		shardSize:          shardSize,
		prefixFn:           cfg.prefixFn,
	}, nil
}

// resolveFirestoreClient resolves the Firestore client from the Firebase plugin
// registered with g, mirroring NewFirestoreStreamManager's plugin resolution.
func resolveFirestoreClient(ctx context.Context, g *genkit.Genkit) (*firestore.Client, error) {
	plugin := genkit.LookupPlugin(g, "firebase")
	if plugin == nil {
		return nil, errors.New("Firebase plugin not found.\n" +
			"  Pass the Firebase plugin to genkit.Init():\n" +
			"    g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: \"your-project\"}))")
	}
	f, ok := plugin.(*firebase.Firebase)
	if !ok {
		return nil, fmt.Errorf("unexpected plugin type %T for provider \"firebase\"", plugin)
	}
	// The Firestore client is long-lived and cached by the plugin, so it must not
	// be bound to this (possibly request-scoped) construction context: a later
	// token refresh or connection maintenance would fail once it is cancelled.
	return f.Firestore(context.WithoutCancel(ctx))
}

// --- Persisted document shapes ---

// snapshotDoc is the persisted form of a snapshot. Metadata is stored in native
// fields; the (potentially large) conversation state is stored out of band: a
// checkpoint shards it across the shards collection, while a diff carries only
// the JSON Patch from its parent. statePatch and error are stored as opaque JSON
// bytes rather than native nested data, which sidesteps Firestore's
// nested-array restriction and matches Genkit's own JSON serialization exactly.
type snapshotDoc struct {
	SnapshotID   string     `firestore:"snapshotId"`
	SessionID    string     `firestore:"sessionId,omitempty"`
	ParentID     string     `firestore:"parentId,omitempty"`
	CreatedAt    time.Time  `firestore:"createdAt"`
	UpdatedAt    time.Time  `firestore:"updatedAt"`
	Status       string     `firestore:"status,omitempty"`
	HeartbeatAt  *time.Time `firestore:"heartbeatAt,omitempty"`
	FinishReason string     `firestore:"finishReason,omitempty"`
	// Error is a JSON-encoded *core.GenkitError, or nil.
	Error []byte `firestore:"error,omitempty"`
	// Kind is "diff" or "checkpoint".
	Kind string `firestore:"kind"`
	// CheckpointID is the nearest checkpoint ancestor (equals SnapshotID when
	// this document is itself a checkpoint).
	CheckpointID string `firestore:"checkpointId"`
	// CheckpointShardCount is the shard count of the checkpoint identified by
	// CheckpointID.
	CheckpointShardCount int `firestore:"checkpointShardCount"`
	// SegmentPath is the ordered diff IDs from the checkpoint (exclusive) to this
	// document (inclusive); empty for a checkpoint. Applying these patches in
	// order onto the checkpoint's state materializes this document's state.
	SegmentPath []string `firestore:"segmentPath,omitempty"`
	// StatePatch is a JSON-encoded [aix.JSONPatch], set only for kind "diff".
	StatePatch []byte `firestore:"statePatch,omitempty"`
}

// shardDoc is one byte-bounded chunk of a checkpoint's JSON-serialized state.
// Concatenating a checkpoint's chunks in index order and parsing the result
// yields the original state.
type shardDoc struct {
	Chunk []byte `firestore:"chunk"`
}

// pointerDoc tracks a session's latest snapshot and the metadata needed to
// reconstruct it, so a by-session lookup is one pointer read plus one batched
// GetAll. It deliberately caches no state, so it can never approach the 1 MiB
// limit no matter how long the session grows. "Latest" is the greatest
// CreatedAt (ties broken by snapshot ID), matching the [aix.SnapshotReader]
// contract.
type pointerDoc struct {
	CurrentSnapshotID    string    `firestore:"currentSnapshotId"`
	CurrentCreatedAt     time.Time `firestore:"currentCreatedAt"`
	CheckpointID         string    `firestore:"checkpointId"`
	CheckpointShardCount int       `firestore:"checkpointShardCount"`
	SegmentPath          []string  `firestore:"segmentPath,omitempty"`
	UpdatedAt            time.Time `firestore:"updatedAt"`
}

// chainMeta is a parent snapshot's reconstruction metadata, resolved without
// materializing its (potentially large) state, so SaveSnapshot can decide
// diff-vs-checkpoint before paying for a full reconstruction.
type chainMeta struct {
	checkpointID string
	shardCount   int
	segmentPath  []string
}

// writePlan is the decision of how to persist a snapshot: a diff carrying a
// JSON Patch, or a checkpoint whose serialized state is sharded.
type writePlan struct {
	kind          string
	checkpointID  string
	shardCount    int
	segmentPath   []string
	statePatch    []byte // diff only: the JSON-encoded patch (marshaled once)
	stateBytes    []byte // checkpoint only: serialized state to shard
	oldShardCount int    // checkpoint only: prior shard count, to prune stale shards
}

// --- Document references ---

func (s *FirestoreSessionStore[State]) prefixFor(ctx context.Context) (string, error) {
	prefix := defaultPrefix
	if s.prefixFn != nil {
		if p := s.prefixFn(ctx); p != "" {
			prefix = p
		}
	}
	// The prefix is used directly as a Firestore document ID (see snapshotRef et
	// al.), so it must be a single valid path segment. Reject the realistic
	// mistakes up front rather than letting them surface as an opaque Firestore
	// path error deep in a transaction; for a tenant-isolation prefix, failing
	// fast and loud is the safe default.
	if strings.ContainsRune(prefix, '/') || prefix == "." || prefix == ".." {
		return "", fmt.Errorf("invalid snapshot path prefix %q: must be a single Firestore document ID with no '/' separators", prefix)
	}
	return prefix, nil
}

func (s *FirestoreSessionStore[State]) snapshotRef(prefix, id string) *firestore.DocumentRef {
	return s.client.Collection(s.collection).Doc(prefix).Collection("snapshots").Doc(id)
}

func (s *FirestoreSessionStore[State]) shardRef(prefix, checkpointID string, i int) *firestore.DocumentRef {
	return s.client.Collection(s.collection + "-shards").Doc(prefix).Collection("shards").Doc(checkpointID + "_" + strconv.Itoa(i))
}

func (s *FirestoreSessionStore[State]) pointerRef(prefix, sessionID string) *firestore.DocumentRef {
	return s.client.Collection(s.collection + "-pointers").Doc(prefix).Collection("pointers").Doc(sessionID)
}

// --- Reads ---

// GetSnapshot retrieves a snapshot by ID. Returns nil if not found. The
// reconstruction runs inside a read-only transaction so the snapshot's
// checkpoint shards and diff segment are read at one consistent point, never a
// mix of pre- and post-checkpoint-rewrite chunks.
func (s *FirestoreSessionStore[State]) GetSnapshot(ctx context.Context, snapshotID string) (*aix.SessionSnapshot[State], error) {
	if snapshotID == "" {
		return nil, nil
	}
	prefix, err := s.prefixFor(ctx)
	if err != nil {
		return nil, fmt.Errorf("firebase: FirestoreSessionStore.GetSnapshot: %w", err)
	}
	var result *aix.SessionSnapshot[State]
	err = s.client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		result = nil
		doc, state, ok, err := s.reconstruct(tx, prefix, snapshotID)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		snap, err := s.toSnapshot(doc, state)
		if err != nil {
			return err
		}
		result = snap
		return nil
	}, firestore.ReadOnly)
	if err != nil {
		return nil, fmt.Errorf("firebase: FirestoreSessionStore.GetSnapshot: %w", err)
	}
	return result, nil
}

// GetLatestSnapshot returns the session's most recently created snapshot
// regardless of status, per the [aix.SnapshotReader.GetLatestSnapshot]
// contract. It reads the session's pointer document (which tracks the
// greatest-CreatedAt snapshot, ties broken by snapshot ID) and reconstructs from
// the pointer's cached checkpoint metadata.
func (s *FirestoreSessionStore[State]) GetLatestSnapshot(ctx context.Context, sessionID string) (*aix.SessionSnapshot[State], error) {
	if sessionID == "" {
		return nil, errors.New("firebase: FirestoreSessionStore.GetLatestSnapshot: session ID is empty")
	}
	prefix, err := s.prefixFor(ctx)
	if err != nil {
		return nil, fmt.Errorf("firebase: FirestoreSessionStore.GetLatestSnapshot: %w", err)
	}
	var result *aix.SessionSnapshot[State]
	err = s.client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		result = nil
		pointer, err := s.readPointer(tx, prefix, sessionID)
		if err != nil {
			return err
		}
		if pointer == nil {
			return nil
		}
		doc, state, ok, err := s.reconstructFrom(tx, prefix, pointer.CheckpointID, pointer.CheckpointShardCount, pointer.SegmentPath, pointer.CurrentSnapshotID)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		snap, err := s.toSnapshot(doc, state)
		if err != nil {
			return err
		}
		result = snap
		return nil
	}, firestore.ReadOnly)
	if err != nil {
		return nil, fmt.Errorf("firebase: FirestoreSessionStore.GetLatestSnapshot: %w", err)
	}
	return result, nil
}

// reconstruct reads the snapshot document at id to learn its checkpoint and
// segment path, then materializes its state. Returns ok=false when the snapshot
// does not exist.
func (s *FirestoreSessionStore[State]) reconstruct(tx *firestore.Transaction, prefix, id string) (snapshotDoc, any, bool, error) {
	snap, err := tx.Get(s.snapshotRef(prefix, id))
	if err != nil {
		if isNotFound(err) {
			return snapshotDoc{}, nil, false, nil
		}
		return snapshotDoc{}, nil, false, err
	}
	if !snap.Exists() {
		return snapshotDoc{}, nil, false, nil
	}
	var doc snapshotDoc
	if err := snap.DataTo(&doc); err != nil {
		return snapshotDoc{}, nil, false, fmt.Errorf("decode snapshot %q: %w", id, err)
	}
	return s.reconstructFrom(tx, prefix, doc.CheckpointID, doc.CheckpointShardCount, doc.SegmentPath, id)
}

// reconstructFrom materializes the state of targetID from a known checkpoint
// using a single batched GetAll: the checkpoint's shards, the (bounded) segment
// of diff documents along segmentPath, and - only when the target is itself the
// checkpoint - the checkpoint document. The diffs are applied in order onto the
// checkpoint's state. Returns the target's document and state, or ok=false when
// the target does not exist. Cost is bounded by the checkpoint interval plus
// shard count, independent of total session length.
func (s *FirestoreSessionStore[State]) reconstructFrom(tx *firestore.Transaction, prefix, checkpointID string, shardCount int, segmentPath []string, targetID string) (snapshotDoc, any, bool, error) {
	targetIsCheckpoint := len(segmentPath) == 0

	// The checkpoint document is only needed when it is itself the target;
	// otherwise the last segment document carries the target's metadata.
	var refs []*firestore.DocumentRef
	checkpointIdx := -1
	if targetIsCheckpoint {
		checkpointIdx = len(refs)
		refs = append(refs, s.snapshotRef(prefix, checkpointID))
	}
	shardStart := len(refs)
	for i := 0; i < shardCount; i++ {
		refs = append(refs, s.shardRef(prefix, checkpointID, i))
	}
	segStart := len(refs)
	for _, sid := range segmentPath {
		refs = append(refs, s.snapshotRef(prefix, sid))
	}

	snaps, err := tx.GetAll(refs)
	if err != nil {
		return snapshotDoc{}, nil, false, fmt.Errorf("batch read: %w", err)
	}

	// Stitch the checkpoint's shards into the base state. tx.GetAll preserves
	// request order, so the shards are already in index order.
	state, err := stitch(snaps[shardStart : shardStart+shardCount])
	if err != nil {
		return snapshotDoc{}, nil, false, err
	}

	if targetIsCheckpoint {
		cs := snaps[checkpointIdx]
		if !cs.Exists() {
			return snapshotDoc{}, nil, false, nil
		}
		var doc snapshotDoc
		if err := cs.DataTo(&doc); err != nil {
			return snapshotDoc{}, nil, false, fmt.Errorf("decode checkpoint %q: %w", checkpointID, err)
		}
		if doc.SnapshotID != targetID {
			return snapshotDoc{}, nil, false, nil
		}
		return doc, state, true, nil
	}

	var targetDoc snapshotDoc
	for _, segSnap := range snaps[segStart:] {
		if !segSnap.Exists() {
			return snapshotDoc{}, nil, false, nil // missing diff: corrupt chain
		}
		var d snapshotDoc
		if err := segSnap.DataTo(&d); err != nil {
			return snapshotDoc{}, nil, false, fmt.Errorf("decode diff %q: %w", segSnap.Ref.ID, err)
		}
		var patch aix.JSONPatch
		if len(d.StatePatch) > 0 {
			if err := json.Unmarshal(d.StatePatch, &patch); err != nil {
				return snapshotDoc{}, nil, false, fmt.Errorf("unmarshal patch %q: %w", d.SnapshotID, err)
			}
		}
		state, err = aix.ApplyPatch(state, patch)
		if err != nil {
			return snapshotDoc{}, nil, false, fmt.Errorf("apply patch %q: %w", d.SnapshotID, err)
		}
		targetDoc = d
	}
	if targetDoc.SnapshotID != targetID {
		return snapshotDoc{}, nil, false, nil
	}
	return targetDoc, state, true, nil
}

// stitch concatenates ordered shard documents and parses the materialized state
// into a JSON value. Returns nil when there are no shards.
func stitch(shardSnaps []*firestore.DocumentSnapshot) (any, error) {
	if len(shardSnaps) == 0 {
		return nil, nil
	}
	var buf []byte
	for _, ss := range shardSnaps {
		if !ss.Exists() {
			return nil, fmt.Errorf("missing checkpoint shard %q", ss.Ref.ID)
		}
		var sd shardDoc
		if err := ss.DataTo(&sd); err != nil {
			return nil, fmt.Errorf("decode shard %q: %w", ss.Ref.ID, err)
		}
		buf = append(buf, sd.Chunk...)
	}
	var state any
	if err := json.Unmarshal(buf, &state); err != nil {
		return nil, fmt.Errorf("unmarshal checkpoint state: %w", err)
	}
	return state, nil
}

// readPointer reads a session's pointer document. Returns nil when absent.
func (s *FirestoreSessionStore[State]) readPointer(tx *firestore.Transaction, prefix, sessionID string) (*pointerDoc, error) {
	snap, err := tx.Get(s.pointerRef(prefix, sessionID))
	if err != nil {
		if isNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	if !snap.Exists() {
		return nil, nil
	}
	var p pointerDoc
	if err := snap.DataTo(&p); err != nil {
		return nil, fmt.Errorf("decode pointer %q: %w", sessionID, err)
	}
	return &p, nil
}

// --- Writes ---

// SaveSnapshot atomically reads the snapshot at id (if any), applies fn, and
// persists the result. See [aix.SnapshotWriter] for the full contract. The
// read-modify-write runs inside a Firestore transaction, which may re-run fn on
// contention; fn must therefore be free of side effects, as the contract
// requires.
func (s *FirestoreSessionStore[State]) SaveSnapshot(
	ctx context.Context,
	id string,
	fn func(existing *aix.SessionSnapshot[State]) (*aix.SessionSnapshot[State], error),
) (*aix.SessionSnapshot[State], error) {
	if id == "" {
		id = uuid.New().String()
	}
	prefix, err := s.prefixFor(ctx)
	if err != nil {
		return nil, fmt.Errorf("firebase: FirestoreSessionStore.SaveSnapshot: %w", err)
	}

	var persisted *aix.SessionSnapshot[State]
	err = s.client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		// Reset on every (re-)run: a prior attempt may have set this before the
		// transaction was retried under contention.
		persisted = nil

		// Reads phase 1: the existing snapshot (if any) so fn can inspect the
		// current full state.
		existingDoc, existingState, existingOK, err := s.reconstruct(tx, prefix, id)
		if err != nil {
			return err
		}
		var current *aix.SessionSnapshot[State]
		if existingOK {
			current, err = s.toSnapshot(existingDoc, existingState)
			if err != nil {
				return err
			}
		}

		next, err := fn(current)
		if err != nil {
			return err
		}
		if next == nil {
			return nil // fn declined: leave the row untouched.
		}

		// Apply the store-owned fields. The store owns identity; fn owns the
		// lifecycle timestamps and status.
		next.SnapshotID = id
		sessionID := next.SessionID
		if existingOK && existingDoc.SessionID != "" {
			// A row's session never changes once set.
			sessionID = existingDoc.SessionID
		}
		if sessionID == "" {
			// A snapshot must belong to a session; the store also keys its
			// per-session pointer by session ID and cannot persist a session-less
			// row. The runtime stamps a session ID on every row it writes, so an
			// empty one indicates misuse. Matches the other session stores.
			return core.NewError(core.INVALID_ARGUMENT, "FirestoreSessionStore requires sessionId to be set on the snapshot")
		}
		next.SessionID = sessionID
		if next.Status == "" {
			next.Status = aix.SnapshotStatusCompleted
		}

		// Reads phase 2: the per-session pointer.
		pointer, err := s.readPointer(tx, prefix, sessionID)
		if err != nil {
			return err
		}

		// Reads phase 3 + decision: resolve diff-vs-checkpoint, reconstructing the
		// parent state only when a diff is actually a candidate.
		var existingDocPtr *snapshotDoc
		if existingOK {
			existingDocPtr = &existingDoc
		}
		plan, err := s.planWrite(tx, prefix, id, existingDocPtr, next, pointer)
		if err != nil {
			return err
		}

		// Writes phase.
		if err := s.commitPlan(tx, prefix, id, next, plan, pointer, !existingOK); err != nil {
			return err
		}

		persisted = next
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("firebase: FirestoreSessionStore.SaveSnapshot: %w", err)
	}
	return persisted, nil
}

// planWrite decides how to persist next and computes the document metadata. It
// performs only reads (it may reconstruct the parent's state to compute a diff);
// the actual document writes happen in commitPlan, preserving Firestore's
// reads-before-writes ordering within a transaction.
func (s *FirestoreSessionStore[State]) planWrite(tx *firestore.Transaction, prefix, id string, existingDoc *snapshotDoc, next *aix.SessionSnapshot[State], pointer *pointerDoc) (writePlan, error) {
	newState := next.State

	if existingDoc != nil {
		// Upsert: preserve the document's role and chain position; only the state
		// and metadata change. Callers must only upsert the leaf - rewriting a
		// non-leaf's state would invalidate its descendants' diffs.
		if existingDoc.Kind == kindCheckpoint {
			return s.planCheckpoint(id, newState, existingDoc.CheckpointShardCount)
		}
		// Diff upsert: resolve the parent's state to recompute the patch.
		var parentState any
		if existingDoc.ParentID != "" {
			_, ps, ok, err := s.reconstruct(tx, prefix, existingDoc.ParentID)
			if err != nil {
				return writePlan{}, err
			}
			if ok {
				parentState = ps
			}
		}
		return s.planDiff(id, existingDoc.CheckpointID, existingDoc.CheckpointShardCount, existingDoc.SegmentPath, parentState, newState)
	}

	// New snapshot: resolve the parent's chain metadata (no state) to decide
	// diff-vs-checkpoint before paying for a full reconstruction.
	var parentMeta *chainMeta
	if next.ParentID != "" {
		pm, err := s.loadParentChainMeta(tx, prefix, next.ParentID, pointer)
		if err != nil {
			return writePlan{}, err
		}
		parentMeta = pm
	}

	if next.ParentID == "" || parentMeta == nil || len(parentMeta.segmentPath)+1 >= s.checkpointInterval {
		// A session root, an orphaned parent, or a checkpoint-interval boundary:
		// write a full checkpoint without reconstructing the parent's state.
		return s.planCheckpoint(id, newState, 0)
	}

	// Diff candidate: materialize the parent's state to compute the patch.
	var parentState any
	if _, ps, ok, err := s.reconstructFrom(tx, prefix, parentMeta.checkpointID, parentMeta.shardCount, parentMeta.segmentPath, next.ParentID); err != nil {
		return writePlan{}, err
	} else if ok {
		parentState = ps
	}
	segment := make([]string, 0, len(parentMeta.segmentPath)+1)
	segment = append(segment, parentMeta.segmentPath...)
	segment = append(segment, id)
	return s.planDiff(id, parentMeta.checkpointID, parentMeta.shardCount, segment, parentState, newState)
}

// planDiff builds a diff plan from parentState to newState, marshaling the patch
// once. If the patch exceeds the shard size it is promoted to a sharded
// checkpoint so even an in-place leaf rewrite can never push the document past
// the 1 MiB limit.
func (s *FirestoreSessionStore[State]) planDiff(id, checkpointID string, shardCount int, segmentPath []string, parentState any, newState *aix.SessionState[State]) (writePlan, error) {
	patch, err := json.Marshal(aix.Diff(parentState, newState))
	if err != nil {
		return writePlan{}, fmt.Errorf("marshal patch: %w", err)
	}
	if len(patch) > s.shardSize {
		return s.planCheckpoint(id, newState, 0)
	}
	return writePlan{
		kind:         kindDiff,
		checkpointID: checkpointID,
		shardCount:   shardCount,
		segmentPath:  segmentPath,
		statePatch:   patch,
	}, nil
}

// planCheckpoint builds a checkpoint plan: a checkpoint anchors itself
// (checkpointID == id), has an empty segment path, and carries no patch. The
// state is serialized now (a read-only step) and sharded later in commitPlan.
func (s *FirestoreSessionStore[State]) planCheckpoint(id string, st *aix.SessionState[State], oldShardCount int) (writePlan, error) {
	b, err := json.Marshal(st)
	if err != nil {
		return writePlan{}, fmt.Errorf("marshal state: %w", err)
	}
	count := (len(b) + s.shardSize - 1) / s.shardSize
	if count == 0 {
		count = 1
	}
	return writePlan{
		kind:          kindCheckpoint,
		checkpointID:  id,
		shardCount:    count,
		stateBytes:    b,
		oldShardCount: oldShardCount,
	}, nil
}

// commitPlan writes the snapshot document, its shards (for a checkpoint), and
// advances or refreshes the session pointer. All writes; no reads.
func (s *FirestoreSessionStore[State]) commitPlan(tx *firestore.Transaction, prefix, id string, next *aix.SessionSnapshot[State], plan writePlan, pointer *pointerDoc, isNew bool) error {
	if plan.kind == kindCheckpoint {
		if err := s.writeShards(tx, prefix, id, plan.stateBytes, plan.shardCount, plan.oldShardCount); err != nil {
			return err
		}
	}

	doc, err := s.toDoc(id, next, plan)
	if err != nil {
		return err
	}
	if err := tx.Set(s.snapshotRef(prefix, id), doc); err != nil {
		return err
	}

	return s.updatePointer(tx, prefix, id, next, plan, pointer, isNew)
}

// writeShards splits stateBytes into shardSize-byte chunks written at
// <checkpointID>_<index>, and deletes any stale trailing shards from a prior,
// larger checkpoint at the same ID.
func (s *FirestoreSessionStore[State]) writeShards(tx *firestore.Transaction, prefix, checkpointID string, stateBytes []byte, count, oldShardCount int) error {
	for i := 0; i < count; i++ {
		start := i * s.shardSize
		end := start + s.shardSize
		if end > len(stateBytes) {
			end = len(stateBytes)
		}
		// A sub-slice is sufficient: Firestore serializes only the slice's bytes
		// (not its backing array), and stateBytes is never mutated for the
		// transaction's lifetime, so each shard document persists just its chunk.
		if err := tx.Set(s.shardRef(prefix, checkpointID, i), shardDoc{Chunk: stateBytes[start:end]}); err != nil {
			return err
		}
	}
	for i := count; i < oldShardCount; i++ {
		if err := tx.Delete(s.shardRef(prefix, checkpointID, i)); err != nil {
			return err
		}
	}
	return nil
}

// updatePointer advances the session pointer to id when id is the new
// greatest-CreatedAt snapshot, or refreshes the pointer's cached reconstruction
// metadata when id is the current pointer target (e.g. a leaf rewrite). An
// upsert of an older, non-latest snapshot leaves the pointer untouched.
func (s *FirestoreSessionStore[State]) updatePointer(tx *firestore.Transaction, prefix, id string, next *aix.SessionSnapshot[State], plan writePlan, pointer *pointerDoc, isNew bool) error {
	advance := false
	if isNew {
		// A new row becomes the latest only if it sorts ahead of the current
		// pointer by (CreatedAt, snapshotID). A backdated new row does not move the
		// pointer, matching the greatest-CreatedAt contract.
		advance = pointer == nil ||
			next.CreatedAt.After(pointer.CurrentCreatedAt) ||
			(next.CreatedAt.Equal(pointer.CurrentCreatedAt) && id > pointer.CurrentSnapshotID)
	} else {
		// Upsert: refresh only when rewriting the current leaf, whose checkpoint
		// metadata may have changed (e.g. a diff promoted to a checkpoint).
		advance = pointer != nil && pointer.CurrentSnapshotID == id
	}
	if !advance {
		return nil
	}
	return tx.Set(s.pointerRef(prefix, next.SessionID), pointerDoc{
		CurrentSnapshotID:    id,
		CurrentCreatedAt:     next.CreatedAt,
		CheckpointID:         plan.checkpointID,
		CheckpointShardCount: plan.shardCount,
		SegmentPath:          plan.segmentPath,
		UpdatedAt:            time.Now(),
	})
}

// loadParentChainMeta resolves the parent's reconstruction metadata without
// materializing its state. In the common linear case the parent is the
// session's current pointer target, so the metadata is read straight off the
// pointer with zero extra document reads; otherwise it reads the single parent
// document. Returns nil when the parent does not exist.
func (s *FirestoreSessionStore[State]) loadParentChainMeta(tx *firestore.Transaction, prefix, parentID string, pointer *pointerDoc) (*chainMeta, error) {
	if pointer != nil && pointer.CurrentSnapshotID == parentID {
		return &chainMeta{
			checkpointID: pointer.CheckpointID,
			shardCount:   pointer.CheckpointShardCount,
			segmentPath:  pointer.SegmentPath,
		}, nil
	}
	snap, err := tx.Get(s.snapshotRef(prefix, parentID))
	if err != nil {
		if isNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	if !snap.Exists() {
		return nil, nil
	}
	var d snapshotDoc
	if err := snap.DataTo(&d); err != nil {
		return nil, fmt.Errorf("decode parent %q: %w", parentID, err)
	}
	return &chainMeta{
		checkpointID: d.CheckpointID,
		shardCount:   d.CheckpointShardCount,
		segmentPath:  d.SegmentPath,
	}, nil
}

// --- Status subscription ---

// OnSnapshotStatusChange returns a channel that yields the snapshot's status at
// subscription time and on every subsequent change, until ctx is cancelled. It
// is backed by Firestore's native document listener, so a status change written
// by any process (e.g. an abort committed by the request handler while a
// detached worker watches) propagates across instances without polling. If the
// snapshot does not exist when the subscription is established, the channel is
// closed without yielding a value.
//
// Values are level-triggered: the latest status is always delivered, but a slow
// reader may skip intermediate values. Treat a received value as "the status is
// now X", not "X happened once".
func (s *FirestoreSessionStore[State]) OnSnapshotStatusChange(ctx context.Context, snapshotID string) <-chan aix.SnapshotStatus {
	ch := make(chan aix.SnapshotStatus, 1)
	if snapshotID == "" {
		close(ch)
		return ch
	}
	prefix, err := s.prefixFor(ctx)
	if err != nil {
		close(ch)
		return ch
	}
	ref := s.snapshotRef(prefix, snapshotID)

	go func() {
		defer close(ch)
		it := ref.Snapshots(ctx)
		defer it.Stop()

		firstEvent := true
		var last aix.SnapshotStatus
		delivered := false
		for {
			snap, err := it.Next()
			if err != nil {
				return // ctx cancelled, or a terminal listener error.
			}
			exists := snap.Exists()
			if firstEvent {
				firstEvent = false
				if !exists {
					return // not present at subscription time.
				}
			}
			if !exists {
				continue // deleted after we began watching; keep listening.
			}
			var sd struct {
				Status string `firestore:"status"`
			}
			if err := snap.DataTo(&sd); err != nil {
				continue
			}
			st := aix.SnapshotStatus(sd.Status)
			if st == "" {
				st = aix.SnapshotStatusCompleted
			}
			if delivered && st == last {
				continue // status unchanged (e.g. a heartbeat-only write).
			}
			last, delivered = st, true
			coalesceSend(ch, st)
		}
	}()
	return ch
}

// --- Conversions ---

// toSnapshot assembles a [aix.SessionSnapshot] from a persisted document and
// its reconstructed (JSON-shaped) state.
func (s *FirestoreSessionStore[State]) toSnapshot(doc snapshotDoc, stateAny any) (*aix.SessionSnapshot[State], error) {
	state, err := s.stateFromAny(stateAny)
	if err != nil {
		return nil, err
	}
	snap := &aix.SessionSnapshot[State]{
		SnapshotID:   doc.SnapshotID,
		SessionID:    doc.SessionID,
		ParentID:     doc.ParentID,
		CreatedAt:    doc.CreatedAt,
		UpdatedAt:    doc.UpdatedAt,
		Status:       aix.SnapshotStatus(doc.Status),
		FinishReason: aix.AgentFinishReason(doc.FinishReason),
		HeartbeatAt:  doc.HeartbeatAt,
		State:        state,
	}
	if len(doc.Error) > 0 {
		var ge core.GenkitError
		if err := json.Unmarshal(doc.Error, &ge); err != nil {
			return nil, fmt.Errorf("unmarshal error: %w", err)
		}
		snap.Error = &ge
	}
	return snap, nil
}

// stateFromAny converts a reconstructed JSON value into a typed session state.
// Returns nil for a nil value (a pending snapshot's state is nil).
func (s *FirestoreSessionStore[State]) stateFromAny(v any) (*aix.SessionState[State], error) {
	if v == nil {
		return nil, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("marshal reconstructed state: %w", err)
	}
	var st aix.SessionState[State]
	if err := json.Unmarshal(b, &st); err != nil {
		return nil, fmt.Errorf("unmarshal reconstructed state: %w", err)
	}
	return &st, nil
}

// toDoc builds the persisted snapshot document from the snapshot and its write
// plan. Caller-managed timestamps and status are persisted verbatim.
func (s *FirestoreSessionStore[State]) toDoc(id string, next *aix.SessionSnapshot[State], plan writePlan) (snapshotDoc, error) {
	doc := snapshotDoc{
		SnapshotID:           id,
		SessionID:            next.SessionID,
		ParentID:             next.ParentID,
		CreatedAt:            next.CreatedAt,
		UpdatedAt:            next.UpdatedAt,
		Status:               string(next.Status),
		HeartbeatAt:          next.HeartbeatAt,
		FinishReason:         string(next.FinishReason),
		Kind:                 plan.kind,
		CheckpointID:         plan.checkpointID,
		CheckpointShardCount: plan.shardCount,
		SegmentPath:          plan.segmentPath,
	}
	if next.Error != nil {
		b, err := json.Marshal(next.Error)
		if err != nil {
			return snapshotDoc{}, fmt.Errorf("marshal error: %w", err)
		}
		doc.Error = b
	}
	if plan.kind == kindDiff {
		doc.StatePatch = plan.statePatch
	}
	return doc, nil
}

// --- Helpers ---

// coalesceSend delivers status on a size-1 buffered channel, guaranteeing the
// latest value stays observable even if an earlier value is still unread: it
// drops any stale unread value first, then sends. The subscriber goroutine is
// the only sender, so after the drain the send always has room.
func coalesceSend(ch chan aix.SnapshotStatus, status aix.SnapshotStatus) {
	select {
	case <-ch:
	default:
	}
	select {
	case ch <- status:
	default:
	}
}
