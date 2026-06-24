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

import {
  Firestore,
  type CollectionReference,
  type DocumentReference,
  type DocumentSnapshot,
  type Transaction,
} from '@google-cloud/firestore';
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
import { logger } from 'genkit/logging';

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
 * Fallback prefix used when no {@link FirestoreSessionStoreOptions.snapshotPathPrefix}
 * is configured.
 */
const DEFAULT_PREFIX = 'global';

/**
 * Options for {@link FirestoreSessionStore}.
 */
export interface FirestoreSessionStoreOptions {
  /**
   * An explicit Firestore instance. Defaults to a new {@link Firestore}
   * instance (which picks up Application Default Credentials and the
   * `FIRESTORE_EMULATOR_HOST` environment variable).
   */
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

  /**
   * Returns a per-tenant prefix derived from the call's
   * {@link SessionStoreOptions} (e.g. the authenticated user id from
   * `options.context`). When set, all snapshot, pointer and shard documents are
   * nested under a tenant-scoped subcollection keyed by this prefix, so reads
   * and writes are isolated per tenant: one tenant can never see (or even
   * address) another's snapshots, even if they get hold of a `snapshotId` -
   * resolving it still requires the matching, auth-derived prefix. Defaults to
   * `"global"`.
   *
   * The prefix is used only to build document paths; the stored ids
   * (`snapshotId`, `parentId`, `checkpointId`, ...) remain prefix-agnostic.
   */
  snapshotPathPrefix?: (options?: SessionStoreOptions) => string;
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
  status?: SessionSnapshot['status'];
  /** Heartbeat timestamp (RFC 3339) for an in-flight detached turn. */
  heartbeatAt?: SessionSnapshot['heartbeatAt'];
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
 * Storage layout (the `<prefix>` segment is the per-tenant prefix returned by
 * {@link FirestoreSessionStoreOptions.snapshotPathPrefix}, or `"global"` when
 * none is configured):
 *
 * - `<collection>/<prefix>/snapshots/<snapshotId>` - one document per snapshot.
 *   A `diff` document holds the patch from its parent (`statePatch`); a
 *   `checkpoint` document holds a full-state materialization (sharded out of
 *   band).
 * - `<collection>-shards/<prefix>/shards/<checkpointId>_<index>` - the sharded
 *   full state for a checkpoint.
 * - `<collection>-pointers/<prefix>/pointers/<sessionId>` - one document per
 *   session pointing at the latest leaf snapshot and the metadata needed to
 *   reconstruct it.
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
  protected db: Firestore;
  protected collection: string;
  protected checkpointInterval: number;
  protected shardSize: number;
  protected snapshotPathPrefix?: (options?: SessionStoreOptions) => string;

  constructor(opts?: FirestoreSessionStoreOptions) {
    this.db = opts?.db ?? new Firestore();
    this.collection = opts?.collection ?? 'genkit-sessions';
    this.checkpointInterval =
      opts?.checkpointInterval ?? DEFAULT_CHECKPOINT_INTERVAL;
    this.shardSize = opts?.shardSize ?? DEFAULT_SHARD_SIZE;
    this.snapshotPathPrefix = opts?.snapshotPathPrefix;
  }

  /** Resolves the (per-tenant) prefix for the given call options. */
  private prefixFor(options?: SessionStoreOptions): string {
    return this.snapshotPathPrefix
      ? this.snapshotPathPrefix(options)
      : DEFAULT_PREFIX;
  }

  /** The (per-tenant) snapshots subcollection. */
  private snapshotsCol(options?: SessionStoreOptions): CollectionReference {
    return this.db
      .collection(this.collection)
      .doc(this.prefixFor(options))
      .collection('snapshots');
  }

  /** The (per-tenant) pointers subcollection. */
  private pointersCol(options?: SessionStoreOptions): CollectionReference {
    return this.db
      .collection(`${this.collection}-pointers`)
      .doc(this.prefixFor(options))
      .collection('pointers');
  }

  /** The (per-tenant) shards subcollection. */
  private shardsCol(options?: SessionStoreOptions): CollectionReference {
    return this.db
      .collection(`${this.collection}-shards`)
      .doc(this.prefixFor(options))
      .collection('shards');
  }

  async getSnapshot(opts: {
    snapshotId?: string;
    sessionId?: string;
    context?: SessionStoreOptions['context'];
  }): Promise<SessionSnapshot<S> | undefined> {
    const { snapshotId, sessionId } = this.normalize(opts);
    const options: SessionStoreOptions = { context: opts.context };

    // Reconstruct inside a read-only transaction so the pointer read and the
    // batched shard/diff reads all observe a single, consistent point in time.
    // Without this, a concurrent checkpoint write - which overwrites a
    // checkpoint's shards in place and may delete now-stale trailing shards
    // (see `writeShards`) - could let a reader stitch together a mix of old and
    // new chunks, yielding a `DATA_LOSS` (missing shard) error or a corrupt
    // `JSON.parse`. A read-only transaction also avoids the contention/retries
    // of a read-write one.
    return this.db.runTransaction(
      async (tx) => {
        const reader = this.reader(tx);

        if (sessionId) {
          const pointerSnap = await tx.get(
            this.pointersCol(options).doc(sessionId)
          );
          if (!pointerSnap.exists) return undefined;
          const pointer = pointerSnap.data() as PointerDoc;
          // Reconstruct straight from the pointer's checkpoint metadata - one
          // batched round-trip, no extra read of the leaf document.
          const reconstructed = await this.reconstructFrom(
            reader,
            pointer.checkpointId,
            pointer.checkpointShardCount,
            pointer.segmentPath,
            pointer.currentSnapshotId,
            options
          );
          if (!reconstructed) return undefined;
          return this.toSnapshot(reconstructed.doc, reconstructed.state);
        }

        const reconstructed = await this.reconstruct(
          reader,
          snapshotId!,
          options
        );
        if (!reconstructed) return undefined;
        return this.toSnapshot(reconstructed.doc, reconstructed.state);
      },
      { readOnly: true }
    );
  }

  async saveSnapshot(
    snapshotId: string | undefined,
    mutator: SnapshotMutator<S>,
    options?: SessionStoreOptions
  ): Promise<string | null> {
    return this.db.runTransaction(async (tx) => {
      const reader = this.reader(tx);

      // Reads phase 1: load the existing snapshot (if any) so the mutator can
      // inspect the current full state.
      let existing: { doc: SnapshotDoc; state: SessionState<S> } | undefined;
      if (snapshotId) {
        existing = await this.reconstruct(reader, snapshotId, options);
      }
      const current = existing
        ? this.toSnapshot(existing.doc, existing.state)
        : undefined;

      const result = mutator(current);
      if (result === null) return null;

      const id =
        snapshotId || result.snapshotId || globalThis.crypto.randomUUID();
      // Prefer the snapshot's top-level `sessionId`; fall back to the id carried
      // in its state for rows written before snapshot-level ids existed.
      const sessionId = result.sessionId ?? result.state?.sessionId;
      if (!sessionId) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message: `FirestoreSessionStore requires 'sessionId' to be set on the snapshot.`,
        });
      }
      const newState = (result.state ?? {}) as SessionState<S>;

      // Reads phase 2: the per-session pointer (current leaf metadata).
      const pointerRef = this.pointersCol(options).doc(sessionId);
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
        if (existing.doc.kind === 'checkpoint') {
          ({
            kind,
            checkpointId,
            checkpointShardCount,
            segmentPath,
            statePatch,
          } = this.writeCheckpoint(
            tx,
            id,
            newState,
            options,
            existing.doc.checkpointShardCount
          ));
        } else {
          // Reads phase 3 (diff upsert): resolve parent state for the patch.
          const parentState = existing.doc.parentId
            ? (await this.reconstruct(reader, existing.doc.parentId, options))
                ?.state
            : undefined;
          const candidatePatch = diff(parentState, newState);
          // Promote an oversized diff to a (sharded) checkpoint so even an
          // in-place leaf rewrite can never push the document past the 1 MiB
          // limit. Safe because callers only upsert the leaf, which has no
          // descendants depending on its chain position.
          if (this.byteLength(candidatePatch) > this.shardSize) {
            ({
              kind,
              checkpointId,
              checkpointShardCount,
              segmentPath,
              statePatch,
            } = this.writeCheckpoint(tx, id, newState, options));
          } else {
            kind = 'diff';
            checkpointId = existing.doc.checkpointId;
            checkpointShardCount = existing.doc.checkpointShardCount;
            segmentPath = existing.doc.segmentPath;
            statePatch = candidatePatch;
          }
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
            pointer,
            options
          );
        }

        if (
          !result.parentId ||
          !parentMeta ||
          parentMeta.segmentPath.length + 1 >= this.checkpointInterval
        ) {
          // Write a full checkpoint without ever reconstructing the parent's
          // state, for any of: a session root, an orphaned parent, or reaching
          // the checkpoint interval (whose final segment is exactly the longest,
          // costliest one we'd otherwise pay to reconstruct here).
          ({
            kind,
            checkpointId,
            checkpointShardCount,
            segmentPath,
            statePatch,
          } = this.writeCheckpoint(tx, id, newState, options));
        } else {
          // Diff candidate: now we must materialize the parent's state to
          // compute the patch.
          const parentState = (
            await this.reconstructFrom(
              reader,
              parentMeta.checkpointId,
              parentMeta.checkpointShardCount,
              parentMeta.segmentPath,
              result.parentId,
              options
            )
          )?.state;
          const candidatePatch = diff(parentState, newState);
          // Promote oversized diffs to checkpoints so a single large turn is
          // sharded rather than rejected by the 1 MiB limit.
          if (this.byteLength(candidatePatch) > this.shardSize) {
            ({
              kind,
              checkpointId,
              checkpointShardCount,
              segmentPath,
              statePatch,
            } = this.writeCheckpoint(tx, id, newState, options));
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
        status: result.status,
        heartbeatAt: result.heartbeatAt,
        finishReason: result.finishReason,
        error: result.error,
        kind,
        checkpointId,
        checkpointShardCount,
        segmentPath,
        statePatch,
      };
      tx.set(this.snapshotsCol(options).doc(id), sanitize(doc));

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
    options?: SessionStoreOptions
  ): void | (() => void) {
    const ref = this.snapshotsCol(options).doc(snapshotId);
    return ref.onSnapshot(async (docSnap) => {
      if (!docSnap.exists) return;
      try {
        const snapshot = await this.getSnapshot({
          snapshotId,
          context: options?.context,
        });
        if (snapshot) callback(snapshot);
      } catch (err) {
        // Swallow errors so a transient read failure (network / permission)
        // doesn't surface as an unhandled promise rejection and crash the
        // process. The next snapshot event will retry.
        logger.error(
          `FirestoreSessionStore.watch failed to load snapshot ${snapshotId}`,
          err
        );
      }
    });
  }

  /**
   * Validates that exactly one of `snapshotId` / `sessionId` is provided, and
   * that the provided one is a non-blank string. Blank / whitespace-only ids
   * are rejected up front (rather than silently treated as "absent") so callers
   * get a clear `INVALID_ARGUMENT` instead of an unusable document key.
   */
  private normalize(opts: { snapshotId?: string; sessionId?: string }): {
    snapshotId?: string;
    sessionId?: string;
  } {
    const snapshotId = opts.snapshotId?.trim() ? opts.snapshotId : undefined;
    const sessionId = opts.sessionId?.trim() ? opts.sessionId : undefined;
    if (!!snapshotId === !!sessionId) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          `getSnapshot requires exactly one non-empty 'snapshotId' or ` +
          `'sessionId' (got ${snapshotId ? 'snapshotId' : opts.snapshotId !== undefined ? 'blank snapshotId' : 'neither'}` +
          `${sessionId ? ' and sessionId' : opts.sessionId !== undefined ? ' and blank sessionId' : ''}).`,
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
    pointer: PointerDoc | undefined,
    options?: SessionStoreOptions
  ): Promise<ParentChainMeta | undefined> {
    if (pointer && pointer.currentSnapshotId === parentId) {
      return {
        checkpointId: pointer.checkpointId,
        checkpointShardCount: pointer.checkpointShardCount,
        segmentPath: pointer.segmentPath,
      };
    }
    const snap = await reader.get(this.snapshotsCol(options).doc(parentId));
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
    id: string,
    options?: SessionStoreOptions
  ): Promise<{ doc: SnapshotDoc; state: SessionState<S> } | undefined> {
    const snap = await reader.get(this.snapshotsCol(options).doc(id));
    if (!snap.exists) return undefined;
    const d = snap.data() as SnapshotDoc;
    return this.reconstructFrom(
      reader,
      d.checkpointId,
      d.checkpointShardCount,
      d.segmentPath,
      id,
      options
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
    targetId: string,
    options?: SessionStoreOptions
  ): Promise<{ doc: SnapshotDoc; state: SessionState<S> } | undefined> {
    const targetIsCheckpoint = segmentPath.length === 0;
    const snapshotsCol = this.snapshotsCol(options);
    const shardsCol = this.shardsCol(options);
    const checkpointRef = snapshotsCol.doc(checkpointId);
    const shardRefs = Array.from({ length: shardCount }, (_, i) =>
      shardsCol.doc(`${checkpointId}_${i}`)
    );
    const segRefs = segmentPath.map((sid) => snapshotsCol.doc(sid));

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
    options?: SessionStoreOptions,
    oldShardCount = 0
  ): number {
    const shardsCol = this.shardsCol(options);
    // `JSON.stringify` already drops `undefined` members, so it produces the
    // same bytes as `sanitize(state)` without the extra parse+stringify round
    // trip - a meaningful saving when checkpointing large states.
    const buf = Buffer.from(JSON.stringify(state ?? null), 'utf8');
    const count = Math.max(1, Math.ceil(buf.length / this.shardSize));
    for (let i = 0; i < count; i++) {
      // Copy the slice into its own buffer. `subarray` returns a view sharing
      // the parent's underlying ArrayBuffer, which the Firestore serializer can
      // persist in full rather than just the sliced range.
      const chunk = Buffer.from(
        buf.subarray(i * this.shardSize, (i + 1) * this.shardSize)
      );
      tx.set(shardsCol.doc(`${checkpointId}_${i}`), {
        chunk,
      } satisfies ShardDoc);
    }
    for (let i = count; i < oldShardCount; i++) {
      tx.delete(shardsCol.doc(`${checkpointId}_${i}`));
    }
    return count;
  }

  /**
   * Writes a full-state checkpoint at `id` (sharding the state via
   * {@link writeShards}) and returns the snapshot metadata describing it: a
   * checkpoint anchors itself (`checkpointId === id`), has an empty
   * `segmentPath`, and carries no `statePatch`.
   *
   * This is the shared promotion path used whenever a snapshot must be a
   * checkpoint rather than a diff - the session root, an orphaned parent, a
   * checkpoint-interval boundary, an in-place checkpoint rewrite, and the
   * promotion of an oversized diff (whether new turn or leaf upsert) so that no
   * single document approaches Firestore's 1 MiB limit. Pass `oldShardCount`
   * when re-checkpointing an existing checkpoint so stale trailing shards are
   * pruned.
   */
  private writeCheckpoint(
    tx: Transaction,
    id: string,
    state: SessionState<S>,
    options?: SessionStoreOptions,
    oldShardCount = 0
  ): {
    kind: 'checkpoint';
    checkpointId: string;
    checkpointShardCount: number;
    segmentPath: string[];
    statePatch: undefined;
  } {
    return {
      kind: 'checkpoint',
      checkpointId: id,
      checkpointShardCount: this.writeShards(
        tx,
        id,
        state,
        options,
        oldShardCount
      ),
      segmentPath: [],
      statePatch: undefined,
    };
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
      sessionId: doc.sessionId,
      createdAt: doc.createdAt,
      // Normalize to plain objects: values reconstructed from Firestore
      // documents (e.g. patch operands) can carry non-plain prototypes.
      state: sanitize(state),
    };

    if (doc.parentId !== undefined) snapshot.parentId = doc.parentId;
    if (doc.updatedAt !== undefined) snapshot.updatedAt = doc.updatedAt;
    if (doc.status !== undefined) snapshot.status = doc.status;
    if (doc.heartbeatAt !== undefined) snapshot.heartbeatAt = doc.heartbeatAt;
    if (doc.finishReason !== undefined)
      snapshot.finishReason = doc.finishReason;
    if (doc.error !== undefined) snapshot.error = doc.error;
    return snapshot;
  }
}
