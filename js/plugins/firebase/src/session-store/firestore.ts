/**
 * Copyright 2025 Google LLC
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
  type DocumentData,
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
 * Options for {@link FirestoreSessionStore}.
 */
export interface FirestoreSessionStoreOptions {
  /** A Firebase app to derive the Firestore instance from. */
  firebaseApp?: App;
  /** An explicit Firestore instance. Takes precedence over `firebaseApp`. */
  db?: Firestore;
  /**
   * The collection where snapshot documents are stored (keyed by
   * `snapshotId`). Defaults to `"genkit-sessions"`. A companion collection
   * `"<collection>-pointers"` holds one pointer document per session.
   */
  collection?: string;
}

/**
 * The persisted shape of a snapshot document. Instead of storing the full
 * session state on every turn, each document persists only the
 * {@link JsonPatch} (`statePatch`) that transforms its parent's state into its
 * own. The root snapshot's patch is computed against `undefined`, so a chain
 * can always be reconstructed uniformly from the root.
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
  /** RFC 6902 patch from the parent's state to this snapshot's state. */
  statePatch: JsonPatch;
}

/**
 * The per-session pointer document. Tracks the current leaf snapshot and
 * caches its fully-materialized state so the common (linear) save and
 * `sessionId` lookup paths are a single read - no chain walk.
 */
interface PointerDoc<S> {
  currentSnapshotId: string;
  currentState: SessionState<S>;
  updatedAt: string;
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
 * incremental JSON Patch diffs.
 *
 * Compared to the file-system store this is intentionally simpler (no chain
 * pruning or per-tenant prefixes). Storage layout:
 *
 * - `<collection>/<snapshotId>` - one document per snapshot holding the patch
 *   from its parent's state (`statePatch`) plus snapshot metadata.
 * - `<collection>-pointers/<sessionId>` - one document per session pointing at
 *   the latest leaf snapshot and caching its materialized state.
 */
export class FirestoreSessionStore<S = unknown> implements SessionStore<S> {
  private db: Firestore;
  private snapshots: CollectionReference;
  private pointers: CollectionReference;

  constructor(opts?: FirestoreSessionStoreOptions) {
    const collection = opts?.collection ?? 'genkit-sessions';
    this.db =
      opts?.db ??
      (opts?.firebaseApp ? getFirestore(opts.firebaseApp) : getFirestore());
    this.snapshots = this.db.collection(collection);
    this.pointers = this.db.collection(`${collection}-pointers`);
  }

  async getSnapshot(opts: {
    snapshotId?: string;
    sessionId?: string;
  }): Promise<SessionSnapshot<S> | undefined> {
    const { snapshotId, sessionId } = this.normalize(opts);

    if (sessionId) {
      const pointerSnap = await this.pointers.doc(sessionId).get();
      if (!pointerSnap.exists) return undefined;
      const pointer = pointerSnap.data() as PointerDoc<S>;
      const docSnap = await this.snapshots.doc(pointer.currentSnapshotId).get();
      if (!docSnap.exists) return undefined;
      return this.toSnapshot(
        docSnap.data() as SnapshotDoc,
        pointer.currentState
      );
    }

    // snapshotId lookup: reconstruct full state from the parent chain.
    const reconstructed = await this.reconstruct(snapshotId!);
    if (!reconstructed) return undefined;
    return this.toSnapshot(reconstructed.doc, reconstructed.state);
  }

  async saveSnapshot(
    snapshotId: string | undefined,
    mutator: SnapshotMutator<S>,
    _options?: SessionStoreOptions
  ): Promise<string | null> {
    return this.db.runTransaction(async (tx) => {
      // Reads phase 1: load the existing snapshot (if any) so the mutator can
      // inspect the current full state.
      let existing: { doc: SnapshotDoc; state: SessionState<S> } | undefined;
      if (snapshotId) {
        existing = await this.reconstructTx(tx, snapshotId);
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

      // Reads phase 2: resolve the parent state needed to compute the patch.
      const pointerRef = this.pointers.doc(sessionId);
      const pointerSnap = await tx.get(pointerRef);
      const pointer = pointerSnap.exists
        ? (pointerSnap.data() as PointerDoc<S>)
        : undefined;

      let parentState: SessionState<S> | undefined;
      if (result.parentId) {
        if (pointer && pointer.currentSnapshotId === result.parentId) {
          parentState = pointer.currentState; // fast path (linear chain)
        } else {
          parentState = (await this.reconstructTx(tx, result.parentId))?.state;
        }
      }

      const statePatch = diff(parentState, newState);

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
        statePatch,
      };
      tx.set(this.snapshots.doc(id), sanitize(doc));

      // Advance the pointer when this is a new leaf, or refresh the cached
      // state when we just rewrote the current leaf (e.g. an abort). Upserts
      // of older, non-leaf snapshots leave the pointer untouched.
      const isNew = !existing;
      if (isNew || !pointer || pointer.currentSnapshotId === id) {
        tx.set(
          pointerRef,
          sanitize<PointerDoc<S>>({
            currentSnapshotId:
              isNew || !pointer ? id : pointer.currentSnapshotId,
            currentState: newState,
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

  /**
   * Walks the parent chain from `id` to the root and applies each patch in
   * order to materialize the full state. Returns `undefined` when the snapshot
   * does not exist.
   */
  private async reconstruct(
    id: string
  ): Promise<{ doc: SnapshotDoc; state: SessionState<S> } | undefined> {
    const chain: SnapshotDoc[] = [];
    let cur: string | undefined = id;
    while (cur) {
      const snap = await this.snapshots.doc(cur).get();
      if (!snap.exists) {
        // A missing leaf means the snapshot doesn't exist; a missing ancestor
        // means the chain is corrupt.
        if (chain.length === 0) return undefined;
        break;
      }
      const data = snap.data() as SnapshotDoc;
      chain.unshift(data);
      cur = data.parentId;
    }
    return { doc: chain[chain.length - 1], state: this.applyChain(chain) };
  }

  /** Transaction-scoped variant of {@link reconstruct}. */
  private async reconstructTx(
    tx: Transaction,
    id: string
  ): Promise<{ doc: SnapshotDoc; state: SessionState<S> } | undefined> {
    const chain: SnapshotDoc[] = [];
    let cur: string | undefined = id;
    while (cur) {
      const snap = await tx.get(this.snapshots.doc(cur));
      if (!snap.exists) {
        if (chain.length === 0) return undefined;
        break;
      }
      const data = snap.data() as SnapshotDoc;
      chain.unshift(data);
      cur = data.parentId;
    }
    return { doc: chain[chain.length - 1], state: this.applyChain(chain) };
  }

  /** Applies a root-first chain of patches to materialize the leaf state. */
  private applyChain(chain: SnapshotDoc[]): SessionState<S> {
    let state: SessionState<S> | undefined;
    for (const link of chain) {
      state = applyPatch(state, link.statePatch);
    }
    return (state ?? {}) as SessionState<S>;
  }

  /** Assembles a {@link SessionSnapshot} from a document and its state. */
  private toSnapshot(
    doc: SnapshotDoc | DocumentData,
    state: SessionState<S>
  ): SessionSnapshot<S> {
    const d = doc as SnapshotDoc;
    const snapshot: SessionSnapshot<S> = {
      snapshotId: d.snapshotId,
      createdAt: d.createdAt,
      event: d.event,
      // Normalize to plain objects: values reconstructed from Firestore
      // documents can carry non-plain prototypes.
      state: sanitize(state),
    };
    if (d.parentId !== undefined) snapshot.parentId = d.parentId;
    if (d.updatedAt !== undefined) snapshot.updatedAt = d.updatedAt;

    if (d.status !== undefined) snapshot.status = d.status;
    if (d.finishReason !== undefined) snapshot.finishReason = d.finishReason;
    if (d.error !== undefined) snapshot.error = d.error;
    return snapshot;
  }
}
