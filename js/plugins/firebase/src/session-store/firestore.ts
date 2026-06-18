/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { App } from 'firebase-admin/app';
import {
  getFirestore,
  type CollectionReference,
  type DocumentReference,
  type DocumentSnapshot,
  type Firestore,
  type Transaction,
} from 'firebase-admin/firestore';
import {
  GenkitError,
  applyPatch,
  diff,
  type JsonPatch,
  type SessionSnapshot,
  type SessionState,
  type SessionStore,
  type SessionStoreOptions,
  type SnapshotMutator,
} from 'genkit/beta';

/**
 * Default number of turns between full-state checkpoints.
 *
 * Chosen to favor the common chat / `useChat` workload, where per-turn state is
 * small and read cost dominates. Per-save reconstruction reads grow ~linearly
 * with the interval while checkpoint write/storage cost shrinks with it, so the
 * op-cost optimum is roughly `sqrt(6 * checkpointShardCount)` (≈small for tiny
 * state); 25 sits near that optimum for chat while staying conservative for
 * larger states. Raise it (e.g. 50-100) for large per-turn state retained
 * long-term; lower it (e.g. 10) for small-state, read-heavy sessions.
 */
const DEFAULT_CHECKPOINT_INTERVAL = 25;

/**
 * Default maximum size (in bytes) of a single shard / diff document. Kept well

 * under Firestore's 1 MiB per-document limit so that no individual write can be
 * rejected for being too large.
 */
const DEFAULT_SHARD_SIZE = 512 * 1024;

/**
 * Options for {@link FirestoreSessionStore}.
 */
export interface FirestoreSessionStoreOptions {
  /** A Firebase app to derive the Firestore instance from. */
  firebaseApp?: App;
  /** An explicit Firestore instance. Takes precedence over `firebaseApp`. */
  db?: Firestore;
  /**
   * The collection where snapshot documents are stored (keyed by
   * `snapshotId`). Defaults to `"genkit-sessions"`. Two companion collections
   * are derived from it: `"<collection>-pointers"` holds one pointer document
   * per session and `"<collection>-shards"` holds the sharded checkpoint
   * state.
   */
  collection?: string;
  /**
   * Number of turns between full-state checkpoints. A larger value stores
   * fewer (but reconstructs over more) diffs; a smaller value reconstructs
   * faster at the cost of more frequent full-state writes. Defaults to
   * {@link DEFAULT_CHECKPOINT_INTERVAL}.
   *
   * Cost tuning: per-save reconstruction reads grow ~linearly with this value
   * (so per-interval read work is ~quadratic in it), while checkpoint write and
   * storage cost shrink with it. Lower it (e.g. 10) for small-state, read-heavy
   * sessions; raise it (e.g. 50-100) for large per-turn state retained for a
   * long time, where checkpoint write/storage amplification dominates.
   */
  checkpointInterval?: number;

  /**
   * Maximum size in bytes of a single shard / diff document. Checkpoint state
   * is split into chunks of this size, and any diff exceeding it is promoted to
   * a (sharded) checkpoint so that no document approaches Firestore's 1 MiB
   * limit. Defaults to {@link DEFAULT_SHARD_SIZE}.
   */
  shardSize?: number;
}

/**
 * The persisted shape of a snapshot document.
 *
 * A session's history is stored as a chain of per-turn documents. To keep
 * reads and document sizes bounded regardless of how long a session grows,
 * documents come in two `kind`s:
 *
 * - `checkpoint` - a full materialization of the session state at that turn.
 *   The state itself is stored *out of band*, sharded across the shards
 *   collection (see {@link ShardDoc}), so a checkpoint never approaches the
 *   1 MiB document limit. Written for the session root, every
 *   `checkpointInterval` turns, and whenever a single turn's diff would be too
 *   large.
 * - `diff` - only the {@link JsonPatch} (`statePatch`) that transforms its
 *   parent's state into its own.
 *
 * Every document carries the metadata needed to reconstruct it with a single
 * batched, strongly-consistent `getAll` (no queries / secondary indexes):
 * `checkpointId` (the nearest checkpoint ancestor), `checkpointShardCount`, and
 * `segmentPath` (the ordered diff IDs from that checkpoint down to this
 * document). Because `segmentPath` resets at every checkpoint, the *number of
 * diff documents* read per reconstruction is bounded by `checkpointInterval`,
 * not by total session length. (The number of shard documents still scales
 * with the state's size - i.e. with session length - since each checkpoint
 * stores the full accumulated state.)
 */
interface SnapshotDoc {
  snapshotId: string;
  sessionId: string;
  parentId?: string;
  createdAt: string;
  updatedAt?: string;
  event: SessionSnapshot['event'];
  status?: SessionSnapshot['status'];
  finishReason?: SessionSnapshot['finishReason'];
  error?: SessionSnapshot['error'];
  /** `checkpoint` stores full state in shards; `diff` stores `statePatch`. */
  kind: 'diff' | 'checkpoint';
  /** Nearest checkpoint ancestor (equals `snapshotId` when a checkpoint). */
  checkpointId: string;
  /** Shard count of the checkpoint identified by `checkpointId`. */
  checkpointShardCount: number;
  /**
   * Ordered diff IDs from the checkpoint (exclusive) to this document
   * (inclusive). Empty for a checkpoint. Applying these patches in order onto
   * the checkpoint's state materializes this document's state.
   */
  segmentPath: string[];
  /** RFC 6902 patch from the parent's state. Only set for `kind: 'diff'`. */
  statePatch?: JsonPatch;
}

/**
 * One shard of a checkpoint's materialized state. The full state is
 * JSON-serialized to UTF-8 and split into byte-bounded chunks stored at
 * `<checkpointId>_<index>`; concatenating the chunks and parsing the result
 * yields the original state.
 */
interface ShardDoc {
  chunk: Buffer;
}

/**
 * The per-session pointer document. Tracks the current leaf snapshot and the
 * metadata needed to reconstruct it (its checkpoint, shard count and segment
 * path) so the common `sessionId` lookup is a single pointer read followed by
 * one batched `getAll`. It deliberately does *not* cache the full state, so it
 * can never approach the 1 MiB document limit no matter how long the session
 * grows.
 */
interface PointerDoc {
  currentSnapshotId: string;
  checkpointId: string;
  checkpointShardCount: number;
  segmentPath: string[];
  updatedAt: string;
}

/**
 * Chain metadata about a parent snapshot needed to extend the chain - the
 * nearest checkpoint, its shard count and the diff segment leading to the
 * parent. Deliberately excludes the parent's (potentially large) state so it
 * can be resolved without a full reconstruction.
 */
interface ParentChainMeta {
  checkpointId: string;
  checkpointShardCount: number;
  segmentPath: string[];
}

/**
 * A minimal batched read interface so reconstruction can run identically
 * against a transaction or the bare Firestore instance. Both `get` and
 * `getAll` are document-ID lookups, which Firestore serves with strong
 * consistency (unlike queries), keeping reconstruction deterministic.
 */
interface Reader {
  get(ref: DocumentReference): Promise<DocumentSnapshot>;
  getAll(refs: DocumentReference[]): Promise<DocumentSnapshot[]>;
}

/**
 * Strips `undefined` members (Firestore rejects them) while preserving JSON
 * semantics - matching how snapshot state is diffed and reconstructed.
 */
function sanitize<T>(value: T): T {
  return JSON.parse(JSON.stringify(value ?? null));
}

/**
 * A Firestore-backed {@link SessionStore} that persists session snapshots as
 * incremental JSON Patch diffs anchored to periodic, sharded full-state
 * checkpoints.
 *
 * Storage layout:
 *
 * - `<collection>/<snapshotId>` - one document per snapshot. A `diff` document
 *   holds the patch from its parent (`statePatch`); a `checkpoint` document
 *   holds a full-state materialization (sharded out of band).
 * - `<collection>-shards/<checkpointId>_<index>` - the sharded full state for a
 *   checkpoint.
 * - `<collection>-pointers/<sessionId>` - one document per session pointing at
 *   the latest leaf snapshot and the metadata needed to reconstruct it.
 *
 * Reconstruction uses only document-ID lookups (`getAll`), so it needs no
 * secondary indexes and is strongly consistent. No single document approaches
 * the 1 MiB limit (state is sharded by `shardSize`), and the number of *diff*
 * documents touched per read/write is bounded by `checkpointInterval` rather
 * than total session length - so the store scales to arbitrarily long sessions
 * (e.g. coding agents, long-lived chatbots). Note that checkpoints still store
 * the full accumulated state, so checkpoint shard count (and the bytes written
 * per checkpoint) grow with the state's size; tune `checkpointInterval` to
 * trade per-save diff reads against checkpoint write amplification.
 */
export class FirestoreSessionStore<S = unknown> implements SessionStore<S> {
  private db: Firestore;
  private snapshots: CollectionReference;
  private pointers: CollectionReference;
  private shards: CollectionReference;
  private checkpointInterval: number;
  private shardSize: number;

  constructor(opts?: FirestoreSessionStoreOptions) {
    const collection = opts?.collection ?? 'genkit-sessions';
    this.db =
      opts?.db ??
      (opts?.firebaseApp ? getFirestore(opts.firebaseApp) : getFirestore());
    this.snapshots = this.db.collection(collection);
    this.pointers = this.db.collection(`${collection}-pointers`);
    this.shards = this.db.collection(`${collection}-shards`);
    this.checkpointInterval =
      opts?.checkpointInterval ?? DEFAULT_CHECKPOINT_INTERVAL;
    this.shardSize = opts?.shardSize ?? DEFAULT_SHARD_SIZE;
  }

  async getSnapshot(opts: {
    snapshotId?: string;
    sessionId?: string;
  }): Promise<SessionSnapshot<S> | undefined> {
    const { snapshotId, sessionId } = this.normalize(opts);
    const reader = this.reader();

    if (sessionId) {
      const pointerSnap = await this.pointers.doc(sessionId).get();
      if (!pointerSnap.exists) return undefined;
      const pointer = pointerSnap.data() as PointerDoc;
      // Reconstruct straight from the pointer's checkpoint metadata - one
      // batched round-trip, no extra read of the leaf document.
      const reconstructed = await this.reconstructFrom(
        reader,
        pointer.checkpointId,
        pointer.checkpointShardCount,
        pointer.segmentPath,
        pointer.currentSnapshotId
      );
      if (!reconstructed) return undefined;
      return this.toSnapshot(reconstructed.doc, reconstructed.state);
    }

    const reconstructed = await this.reconstruct(reader, snapshotId!);
    if (!reconstructed) return undefined;
    return this.toSnapshot(reconstructed.doc, reconstructed.state);
  }

  async saveSnapshot(
    snapshotId: string | undefined,
    mutator: SnapshotMutator<S>,
    _options?: SessionStoreOptions
  ): Promise<string | null> {
    return this.db.runTransaction(async (tx) => {
      const reader = this.reader(tx);

      // Reads phase 1: load the existing snapshot (if any) so the mutator can
      // inspect the current full state.
      let existing: { doc: SnapshotDoc; state: SessionState<S> } | undefined;
      if (snapshotId) {
        existing = await this.reconstruct(reader, snapshotId);
      }
      const current = existing
        ? this.toSnapshot(existing.doc, existing.state)
        : undefined;

      const result = mutator(current);
      if (result === null) return null;

      const id =
        snapshotId || result.snapshotId || globalThis.crypto.randomUUID();
      const sessionId = result.state?.sessionId;
      if (!sessionId) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message: `FirestoreSessionStore requires 'state.sessionId' to be set on the snapshot.`,
        });
      }
      const newState = (result.state ?? {}) as SessionState<S>;

      // Reads phase 2: the per-session pointer (current leaf metadata).
      const pointerRef = this.pointers.doc(sessionId);
      const pointerSnap = await tx.get(pointerRef);
      const pointer = pointerSnap.exists
        ? (pointerSnap.data() as PointerDoc)
        : undefined;

      let kind: 'diff' | 'checkpoint';
      let checkpointId: string;
      let checkpointShardCount: number;
      let segmentPath: string[];
      let statePatch: JsonPatch | undefined;

      if (existing) {
        // Upsert: preserve the document's role and chain position; only the
        // state/metadata change. Callers must only upsert the *leaf* -
        // rewriting a non-leaf snapshot's state would invalidate its
        // descendants' diffs.
        kind = existing.doc.kind;
        if (kind === 'checkpoint') {
          checkpointId = id;
          segmentPath = [];
          checkpointShardCount = this.writeShards(
            tx,
            id,
            newState,
            existing.doc.checkpointShardCount
          );
        } else {
          checkpointId = existing.doc.checkpointId;
          checkpointShardCount = existing.doc.checkpointShardCount;
          segmentPath = existing.doc.segmentPath;
          // Reads phase 3 (diff upsert): resolve parent state for the patch.
          const parentState = existing.doc.parentId
            ? (await this.reconstruct(reader, existing.doc.parentId))?.state
            : undefined;
          statePatch = diff(parentState, newState);
        }
      } else {
        // New snapshot: resolve the parent's *chain metadata* (no state) to
        // decide diff vs checkpoint. Materializing the parent's full state is
        // deferred until we know we actually need a diff - so the expensive
        // reconstruction is skipped on every checkpoint-boundary turn (which
        // would rewrite the whole state regardless).
        let parentMeta: ParentChainMeta | undefined;
        if (result.parentId) {
          parentMeta = await this.loadParentChainMeta(
            reader,
            result.parentId,
            pointer
          );
        }

        if (!result.parentId || !parentMeta) {
          // Session root (or an orphaned parent) starts a fresh checkpoint.
          kind = 'checkpoint';
          checkpointId = id;
          segmentPath = [];
          checkpointShardCount = this.writeShards(tx, id, newState);
        } else if (
          parentMeta.segmentPath.length + 1 >=
          this.checkpointInterval
        ) {
          // Reached the checkpoint interval: write a full checkpoint without
          // ever reconstructing the parent's state (the longest, costliest
          // segment is exactly the one we'd otherwise pay for here).
          kind = 'checkpoint';
          checkpointId = id;
          segmentPath = [];
          checkpointShardCount = this.writeShards(tx, id, newState);
        } else {
          // Diff candidate: now we must materialize the parent's state to
          // compute the patch.
          const parentState = (
            await this.reconstructFrom(
              reader,
              parentMeta.checkpointId,
              parentMeta.checkpointShardCount,
              parentMeta.segmentPath,
              result.parentId
            )
          )?.state;
          const candidatePatch = diff(parentState, newState);
          // Promote oversized diffs to checkpoints so a single large turn is
          // sharded rather than rejected by the 1 MiB limit.
          const diffTooLarge = this.byteLength(candidatePatch) > this.shardSize;

          if (diffTooLarge) {
            kind = 'checkpoint';
            checkpointId = id;
            segmentPath = [];
            checkpointShardCount = this.writeShards(tx, id, newState);
          } else {
            kind = 'diff';
            checkpointId = parentMeta.checkpointId;
            checkpointShardCount = parentMeta.checkpointShardCount;
            segmentPath = [...parentMeta.segmentPath, id];
            statePatch = candidatePatch;
          }
        }
      }

      // Writes phase.
      const doc: SnapshotDoc = {
        snapshotId: id,
        sessionId,
        parentId: result.parentId,
        createdAt: result.createdAt,
        updatedAt: result.updatedAt ?? result.createdAt,
        event: result.event,
        status: result.status,
        finishReason: result.finishReason,
        error: result.error,
        kind,
        checkpointId,
        checkpointShardCount,
        segmentPath,
        statePatch,
      };
      tx.set(this.snapshots.doc(id), sanitize(doc));

      // Advance the pointer when this is a new leaf, or refresh it when we just
      // rewrote the current leaf. Upserts of older, non-leaf snapshots leave
      // the pointer untouched.
      const isNew = !existing;
      if (isNew || !pointer || pointer.currentSnapshotId === id) {
        tx.set(
          pointerRef,
          sanitize<PointerDoc>({
            currentSnapshotId:
              isNew || !pointer ? id : pointer.currentSnapshotId,
            checkpointId,
            checkpointShardCount,
            segmentPath,
            updatedAt: new Date().toISOString(),
          })
        );
      }

      return id;
    });
  }

  onSnapshotStateChange(
    snapshotId: string,
    callback: (snapshot: SessionSnapshot<S>) => void,
    _options?: SessionStoreOptions
  ): void | (() => void) {
    const ref = this.snapshots.doc(snapshotId);
    return ref.onSnapshot(async (docSnap) => {
      if (!docSnap.exists) return;
      const snapshot = await this.getSnapshot({ snapshotId });
      if (snapshot) callback(snapshot);
    });
  }

  /**
   * Validates that exactly one of `snapshotId` / `sessionId` is provided.
   */
  private normalize(opts: { snapshotId?: string; sessionId?: string }): {
    snapshotId?: string;
    sessionId?: string;
  } {
    const { snapshotId, sessionId } = opts;
    if (!!snapshotId === !!sessionId) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          `getSnapshot requires exactly one of 'snapshotId' or 'sessionId' ` +
          `(got ${snapshotId ? 'snapshotId' : 'neither'}${sessionId ? ' and sessionId' : ''}).`,
      });
    }
    return { snapshotId, sessionId };
  }

  /** Builds a {@link Reader} bound to a transaction or the bare instance. */
  private reader(tx?: Transaction): Reader {
    if (tx) {
      return {
        get: (ref) => tx.get(ref),
        getAll: (refs) =>
          refs.length ? tx.getAll(...refs) : Promise.resolve([]),
      };
    }
    return {
      get: (ref) => ref.get(),
      getAll: (refs) =>
        refs.length ? this.db.getAll(...refs) : Promise.resolve([]),
    };
  }

  /**
   * Resolves a parent's chain metadata (nearest checkpoint, shard count and
   * segment path) *without* materializing its - potentially large - state.
   *
   * In the common linear case the parent is the session's current leaf, so the
   * metadata is read straight off the pointer and this performs *zero*
   * document reads. Otherwise it reads the single parent document. Crucially,
   * resolving only the metadata lets `saveSnapshot` decide diff-vs-checkpoint
   * (which only needs `segmentPath.length`) before paying for a full state
   * reconstruction - so checkpoint-boundary turns, which would rewrite the
   * whole state anyway, skip reconstruction entirely.
   */
  private async loadParentChainMeta(
    reader: Reader,
    parentId: string,
    pointer: PointerDoc | undefined
  ): Promise<ParentChainMeta | undefined> {
    if (pointer && pointer.currentSnapshotId === parentId) {
      return {
        checkpointId: pointer.checkpointId,
        checkpointShardCount: pointer.checkpointShardCount,
        segmentPath: pointer.segmentPath,
      };
    }
    const snap = await reader.get(this.snapshots.doc(parentId));
    if (!snap.exists) return undefined;
    const d = snap.data() as SnapshotDoc;
    return {
      checkpointId: d.checkpointId,
      checkpointShardCount: d.checkpointShardCount,
      segmentPath: d.segmentPath,
    };
  }

  /**
   * Reconstructs the state of `id` by reading its document to learn its
   * checkpoint and segment path, then materializing from that checkpoint.
   * Returns `undefined` when the snapshot does not exist.
   */
  private async reconstruct(
    reader: Reader,
    id: string
  ): Promise<{ doc: SnapshotDoc; state: SessionState<S> } | undefined> {
    const snap = await reader.get(this.snapshots.doc(id));
    if (!snap.exists) return undefined;
    const d = snap.data() as SnapshotDoc;
    return this.reconstructFrom(
      reader,
      d.checkpointId,
      d.checkpointShardCount,
      d.segmentPath,
      id
    );
  }

  /**
   * Materializes the state of `targetId` from a known checkpoint using a single
   * batched, strongly-consistent `getAll`: the checkpoint's shards, the
   * (bounded) segment of diff documents along `segmentPath`, and - only when
   * the target *is* the checkpoint - the checkpoint document itself. The diffs
   * are then applied in order onto the checkpoint's state. Cost is bounded by
   * `checkpointInterval` + shard count, independent of total session length.
   *
   * Note: when `segmentPath` is non-empty the state comes entirely from the
   * shards and the target's metadata from the last segment document, so the
   * checkpoint *document* is not read - saving one read on the common path.
   */
  private async reconstructFrom(
    reader: Reader,
    checkpointId: string,
    shardCount: number,
    segmentPath: string[],
    targetId: string
  ): Promise<{ doc: SnapshotDoc; state: SessionState<S> } | undefined> {
    const targetIsCheckpoint = segmentPath.length === 0;
    const checkpointRef = this.snapshots.doc(checkpointId);
    const shardRefs = Array.from({ length: shardCount }, (_, i) =>
      this.shards.doc(`${checkpointId}_${i}`)
    );
    const segRefs = segmentPath.map((sid) => this.snapshots.doc(sid));

    const snaps = await reader.getAll([
      // The checkpoint document is only needed when it is itself the target;
      // otherwise the last segment document carries the target metadata.
      ...(targetIsCheckpoint ? [checkpointRef] : []),
      ...shardRefs,
      ...segRefs,
    ]);

    // `getAll` does not guarantee result order matches request order, so index
    // the snapshots by their (fully-qualified) path and look each up explicitly.
    const byPath = new Map<string, DocumentSnapshot>();
    for (const s of snaps) byPath.set(s.ref.path, s);

    const shardSnaps = shardRefs.map((ref) => byPath.get(ref.path)!);
    let state = this.stitch(shardSnaps) as SessionState<S> | undefined;

    if (targetIsCheckpoint) {
      const checkpointSnap = byPath.get(checkpointRef.path);
      if (!checkpointSnap?.exists) return undefined;
      const checkpointDoc = checkpointSnap.data() as SnapshotDoc;
      if (checkpointDoc.snapshotId !== targetId) return undefined;
      return { doc: checkpointDoc, state: (state ?? {}) as SessionState<S> };
    }

    let targetDoc: SnapshotDoc | undefined;
    for (const ref of segRefs) {
      const segSnap = byPath.get(ref.path);
      if (!segSnap?.exists) return undefined; // Missing diff: corrupt chain.
      const segDoc = segSnap.data() as SnapshotDoc;
      state = applyPatch(state, segDoc.statePatch ?? []);
      targetDoc = segDoc;
    }

    if (!targetDoc || targetDoc.snapshotId !== targetId) return undefined;
    return { doc: targetDoc, state: (state ?? {}) as SessionState<S> };
  }

  /**
   * Serializes `state` to UTF-8, splits it into `shardSize`-byte chunks, and
   * writes them at `<checkpointId>_<index>`. When `oldShardCount` exceeds the
   * new count (a shrinking re-checkpoint), the now-stale trailing shards are
   * deleted. Returns the number of shards written.
   */
  private writeShards(
    tx: Transaction,
    checkpointId: string,
    state: SessionState<S>,
    oldShardCount = 0
  ): number {
    // `JSON.stringify` already drops `undefined` members, so it produces the
    // same bytes as `sanitize(state)` without the extra parse+stringify round
    // trip - a meaningful saving when checkpointing large states.
    const buf = Buffer.from(JSON.stringify(state ?? null), 'utf8');
    const count = Math.max(1, Math.ceil(buf.length / this.shardSize));
    for (let i = 0; i < count; i++) {
      const chunk = buf.subarray(i * this.shardSize, (i + 1) * this.shardSize);
      tx.set(this.shards.doc(`${checkpointId}_${i}`), {
        chunk,
      } satisfies ShardDoc);
    }
    for (let i = count; i < oldShardCount; i++) {
      tx.delete(this.shards.doc(`${checkpointId}_${i}`));
    }
    return count;
  }

  /** Concatenates ordered shard documents and parses the materialized state. */
  private stitch(shardSnaps: DocumentSnapshot[]): unknown {
    if (shardSnaps.length === 0) return undefined;
    const buffers: Buffer[] = [];
    for (const s of shardSnaps) {
      if (!s.exists) {
        throw new GenkitError({
          status: 'DATA_LOSS',
          message: `FirestoreSessionStore: missing checkpoint shard '${s.id}'.`,
        });
      }
      buffers.push((s.data() as ShardDoc).chunk);
    }
    return JSON.parse(Buffer.concat(buffers).toString('utf8'));
  }

  /** UTF-8 byte length of a JSON-serializable value. */
  private byteLength(value: unknown): number {
    return Buffer.byteLength(JSON.stringify(value ?? null), 'utf8');
  }

  /** Assembles a {@link SessionSnapshot} from a document and its state. */
  private toSnapshot(
    doc: SnapshotDoc,
    state: SessionState<S>
  ): SessionSnapshot<S> {
    const snapshot: SessionSnapshot<S> = {
      snapshotId: doc.snapshotId,
      createdAt: doc.createdAt,
      event: doc.event,
      // Normalize to plain objects: values reconstructed from Firestore
      // documents (e.g. patch operands) can carry non-plain prototypes.
      state: sanitize(state),
    };
    if (doc.parentId !== undefined) snapshot.parentId = doc.parentId;
    if (doc.updatedAt !== undefined) snapshot.updatedAt = doc.updatedAt;
    if (doc.status !== undefined) snapshot.status = doc.status;
    if (doc.finishReason !== undefined)
      snapshot.finishReason = doc.finishReason;
    if (doc.error !== undefined) snapshot.error = doc.error;
    return snapshot;
  }
}
