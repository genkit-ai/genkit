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

import { GenkitError } from '@genkit-ai/core';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';
import {
  assertValidSessionId,
  type GetSnapshotOptions,
  type SessionSnapshot,
  type SessionStore,
  type SessionStoreOptions,
  type SnapshotMutator,
} from './session.js';

/**
 * Normalizes and validates {@link GetSnapshotOptions}.
 *
 * Enforces that exactly one of `snapshotId` / `sessionId` is provided and,
 * when a `sessionId` is given, that it is a valid UUID.
 *
 * @throws `INVALID_ARGUMENT` when neither or both are provided.
 */
function normalizeGetSnapshotOptions(opts: GetSnapshotOptions): {
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
  if (sessionId) {
    assertValidSessionId(sessionId);
  }
  return { snapshotId, sessionId };
}

/**
 * Selects the latest leaf snapshot from a set belonging to one session.
 *
 * A "leaf" is a snapshot that no other snapshot points to as its `parentId`.
 * A healthy linear session has exactly one leaf - the latest turn.
 *
 * - Returns `undefined` when `snapshots` is empty.
 * - Returns the single leaf when the history is linear.
 * - When the history has branched (more than one leaf, e.g. after a
 *   regenerate) the behavior depends on `rejectBranching`:
 *   - `false` (default): returns the most-recently created leaf (by
 *     `createdAt`). This keeps `sessionId` lookups cheap and forgiving.
 *   - `true`: throws `FAILED_PRECONDITION`, since there is no unambiguous
 *     "latest". Opt into this in dev to surface accidental branching early.
 */
function selectLeafSnapshot<S>(
  snapshots: SessionSnapshot<S>[],
  sessionId: string,
  rejectBranching = false
): SessionSnapshot<S> | undefined {
  if (snapshots.length === 0) return undefined;

  const parentIds = new Set<string>();
  for (const snap of snapshots) {
    if (snap.parentId) parentIds.add(snap.parentId);
  }
  const leaves = snapshots.filter((s) => !parentIds.has(s.snapshotId));

  // A single-snapshot session, or any chain, collapses to one leaf.
  if (leaves.length === 1) return leaves[0];

  if (leaves.length === 0) {
    // Cyclic / corrupt history - every snapshot is someone's parent.
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        `Session '${sessionId}' has no leaf snapshot (corrupt or cyclic ` +
        `history). Resume by snapshotId instead.`,
    });
  }

  if (rejectBranching) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        `Session '${sessionId}' has branching snapshots (${leaves.length} ` +
        `leaves), so there is no single latest snapshot. This happens when a ` +
        `conversation is branched (e.g. regenerate). Resume by snapshotId instead.`,
    });
  }

  // Default: pick the most-recently created leaf. `createdAt` is an ISO-8601
  // timestamp, so lexicographic comparison matches chronological order.
  return leaves.reduce((latest, snap) =>
    snap.createdAt > latest.createdAt ? snap : latest
  );
}

/**
 * In-memory implementation of persistent Session Store.
 */
export class InMemorySessionStore<S = unknown> implements SessionStore<S> {
  private snapshots = new Map<string, SessionSnapshot<S>>();
  private listeners = new Map<
    string,
    Array<(snapshot: SessionSnapshot<S>) => void>
  >();
  private rejectBranchingSessions: boolean;

  /**
   * @param options.rejectBranchingSessions When `true`, a `sessionId` lookup
   *   that resolves to a branched history (more than one leaf) throws
   *   `FAILED_PRECONDITION` instead of returning the latest leaf. Defaults to
   *   `false`; opt in (e.g. in dev) to surface accidental branching early.
   */
  constructor(options?: { rejectBranchingSessions?: boolean }) {
    this.rejectBranchingSessions = options?.rejectBranchingSessions ?? false;
  }

  async getSnapshot(
    opts: GetSnapshotOptions
  ): Promise<SessionSnapshot<S> | undefined> {
    const { snapshotId, sessionId } = normalizeGetSnapshotOptions(opts);

    if (snapshotId) {
      const snap = this.snapshots.get(snapshotId);
      if (!snap) return undefined;
      return structuredClone(snap);
    }

    // sessionId lookup: gather every snapshot belonging to this session and
    // resolve the single leaf (latest) snapshot.
    const owned: SessionSnapshot<S>[] = [];
    for (const snap of this.snapshots.values()) {
      if (snap.state?.sessionId === sessionId) {
        owned.push(snap);
      }
    }
    const leaf = selectLeafSnapshot(
      owned,
      sessionId!,
      this.rejectBranchingSessions
    );
    return leaf ? structuredClone(leaf) : undefined;
  }

  async saveSnapshot(
    snapshotId: string | undefined,
    mutator: SnapshotMutator<S>,
    options?: SessionStoreOptions
  ): Promise<string | null> {
    const current = snapshotId ? this.snapshots.get(snapshotId) : undefined;

    const result = mutator(current ? structuredClone(current) : undefined);
    if (result === null) return null;

    // Determine the final ID. The runtime normally supplies a snapshotId, but
    // fall back to a fresh UUID for direct store users who omit it.
    const id =
      snapshotId || result.snapshotId || globalThis.crypto.randomUUID();
    const full: SessionSnapshot<S> = {
      ...result,
      snapshotId: id,
    };

    this.snapshots.set(id, structuredClone(full));
    const snapshotListeners = this.listeners.get(id);
    if (snapshotListeners) {
      for (const listener of snapshotListeners) {
        listener(structuredClone(full));
      }
    }
    return id;
  }

  onSnapshotStateChange(
    snapshotId: string,
    callback: (snapshot: SessionSnapshot<S>) => void,
    options?: SessionStoreOptions
  ): void | (() => void) {
    if (!this.listeners.has(snapshotId)) {
      this.listeners.set(snapshotId, []);
    }
    this.listeners.get(snapshotId)!.push(callback);
    return () => {
      const list = this.listeners.get(snapshotId);
      if (list) {
        const index = list.indexOf(callback);
        if (index >= 0) list.splice(index, 1);
      }
    };
  }
}

/**
 * A Node.js file-system backed session snapshot store.
 *
 * Snapshots are stored as flat JSON files keyed by their `snapshotId`, under an
 * optional per-tenant sub-directory `prefix`:
 *
 * File layout: `dirPath/<prefix>/<snapshotId>.json`
 *
 * `getSnapshot({ sessionId })` scans the prefix directory and selects the
 * single leaf snapshot whose `state.sessionId` matches - there is no separate
 * grouping directory, the `sessionId` lives in each snapshot's state.
 */
export class FileSessionStore<S = unknown> implements SessionStore<S> {
  private dirPath: string;
  private maxPersistedChainLength?: number;
  private snapshotPathPrefix?: (options?: SessionStoreOptions) => string;
  private rejectBranchingSessions: boolean;

  /**
   * @param dirPath Directory where snapshot JSON files are stored.
   * @param options.maxPersistedChainLength When set, snapshots older than this
   *   many entries in a chain are automatically deleted on each save.
   * @param options.snapshotPathPrefix Returns a sub-directory prefix derived
   *   from the call's {@link SessionStoreOptions} (e.g. the authenticated user
   *   id from `options.context`), useful for multi-tenant isolation: all reads
   *   and writes are scoped to that prefix, so one tenant can never see
   *   another's snapshots. Defaults to `"global"`.
   * @param options.rejectBranchingSessions When `true`, a `sessionId` lookup
   *   that resolves to a branched history (more than one leaf) throws
   *   `FAILED_PRECONDITION` instead of returning the latest leaf. Defaults to
   *   `false`; opt in (e.g. in dev) to surface accidental branching early.
   */
  constructor(
    dirPath: string,
    options?: {
      maxPersistedChainLength?: number;
      snapshotPathPrefix?: (options?: SessionStoreOptions) => string;
      rejectBranchingSessions?: boolean;
    }
  ) {
    this.dirPath = path.resolve(dirPath);
    fs.mkdirSync(this.dirPath, { recursive: true });
    this.maxPersistedChainLength = options?.maxPersistedChainLength;
    this.snapshotPathPrefix = options?.snapshotPathPrefix;
    this.rejectBranchingSessions = options?.rejectBranchingSessions ?? false;
  }

  private async ensureDir(dir: string): Promise<void> {
    await fsp.mkdir(dir, { recursive: true });
  }

  /** Resolves the (per-tenant) directory snapshots are stored under. */
  private prefixDir(options?: SessionStoreOptions): string {
    const prefix = this.snapshotPathPrefix
      ? this.snapshotPathPrefix(options)
      : 'global';
    return path.join(this.dirPath, prefix);
  }

  /**
   * Resolves the file path for a given snapshotId: `<prefix>/<snapshotId>.json`.
   */
  private async getFilePath(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<string> {
    const dir = this.prefixDir(options);
    await this.ensureDir(dir);
    return path.join(dir, `${snapshotId}.json`);
  }

  async getSnapshot(
    opts: GetSnapshotOptions
  ): Promise<SessionSnapshot<S> | undefined> {
    const { snapshotId, sessionId } = normalizeGetSnapshotOptions(opts);

    if (sessionId) {
      return this.getLatestSnapshotForSession(sessionId, opts);
    }

    const filePath = await this.getFilePath(snapshotId!, opts);
    try {
      const fileContents = await fsp.readFile(filePath, 'utf-8');
      return JSON.parse(fileContents) as SessionSnapshot<S>;
    } catch (e: unknown) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw e;
    }
  }

  /**
   * Loads a single snapshot file by its id (no sessionId branch). Used by
   * internal traversal (parent chains) where we always have a concrete id.
   */
  private async getSnapshotById(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S> | undefined> {
    const filePath = await this.getFilePath(snapshotId, options);
    try {
      const fileContents = await fsp.readFile(filePath, 'utf-8');
      return JSON.parse(fileContents) as SessionSnapshot<S>;
    } catch (e: unknown) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw e;
    }
  }

  /**
   * Resolves the latest (leaf) snapshot for a session by scanning every
   * snapshot file in the (per-tenant) prefix directory, keeping those whose
   * `state.sessionId` matches, and selecting the single leaf.
   */
  private async getLatestSnapshotForSession(
    sessionId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S> | undefined> {
    const dir = this.prefixDir(options);

    let files: string[];
    try {
      files = await fsp.readdir(dir);
    } catch (e: unknown) {
      if ((e as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw e;
    }

    const snapshots: SessionSnapshot<S>[] = [];
    for (const file of files) {
      if (!file.endsWith('.json')) continue;
      try {
        const contents = await fsp.readFile(path.join(dir, file), 'utf-8');
        const snap = JSON.parse(contents) as SessionSnapshot<S>;
        if (snap.state?.sessionId === sessionId) {
          snapshots.push(snap);
        }
      } catch (e: unknown) {
        if ((e as NodeJS.ErrnoException).code === 'ENOENT') continue;
        throw e;
      }
    }

    return selectLeafSnapshot(
      snapshots,
      sessionId,
      this.rejectBranchingSessions
    );
  }

  async saveSnapshot(
    snapshotId: string | undefined,
    mutator: SnapshotMutator<S>,
    options?: SessionStoreOptions
  ): Promise<string | null> {
    // Read the current snapshot when an ID is provided.
    const current = snapshotId
      ? await this.getSnapshotById(snapshotId, options)
      : undefined;

    const snapshot = mutator(current);
    if (snapshot === null) return null;

    // Determine the final ID. The runtime normally supplies a snapshotId, but
    // fall back to a fresh UUID for direct store users who omit it.
    const id =
      snapshotId || snapshot.snapshotId || globalThis.crypto.randomUUID();

    const full: SessionSnapshot<S> = {
      ...snapshot,
      snapshotId: id,
    };
    const filePath = await this.getFilePath(id, options);
    await fsp.writeFile(filePath, JSON.stringify(full, null, 2), 'utf-8');

    if (this.maxPersistedChainLength && this.maxPersistedChainLength > 0) {
      let cur: SessionSnapshot<S> | undefined = full;
      const chain: string[] = [];

      while (cur) {
        chain.push(cur.snapshotId);
        if (cur.parentId) {
          cur = await this.getSnapshotById(cur.parentId, options);
        } else {
          break;
        }
      }

      if (chain.length > this.maxPersistedChainLength) {
        for (let i = this.maxPersistedChainLength; i < chain.length; i++) {
          const pathToDelete = await this.getFilePath(chain[i], options);
          await fsp.unlink(pathToDelete).catch((e: unknown) => {
            if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e;
          });
        }
      }
    }

    return id;
  }
}
