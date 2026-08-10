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
 * The leaf is returned regardless of `status`; resumability (only `done`
 * snapshots are resumable) is enforced by the agent, which walks back over a
 * non-resumable leaf (e.g. a failed turn) to the last-good snapshot.
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
      if ((snap.sessionId ?? snap.state?.sessionId) === sessionId) {
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
 * Default interval (ms) for the polling fallback used by
 * {@link FileSessionStore.onSnapshotStateChange}.
 */
const DEFAULT_SNAPSHOT_WATCH_POLL_INTERVAL_MS = 2000;

/**
 * Validates that an id is a plain file basename and not a path that could
 * escape the (per-tenant) prefix directory.
 *
 * Ids (`snapshotId`, `sessionId`) can arrive straight off the wire (the
 * abort/getSnapshot actions accept a bare string), so without this an id like
 * `../../foo` would let a caller read or write outside the prefix and break
 * per-tenant isolation.
 */
function assertSafeId(id: string, label: string): void {
  if (
    !id ||
    id.includes('/') ||
    id.includes('\\') ||
    id.includes('\0') ||
    id === '.' ||
    id === '..' ||
    path.basename(id) !== id
  ) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `Invalid ${label}: "${id}". A ${label} must be a plain file name (no path separators or "..").`,
    });
  }
}

/**
 * Validates that a snapshotId is a plain file basename and not a path that
 * could escape the (per-tenant) prefix directory.
 */
function assertSafeSnapshotId(snapshotId: string): void {
  assertSafeId(snapshotId, 'snapshotId');
}

/**
 * The per-session pointer file. Records the current leaf snapshot id for a
 * session so a `getSnapshot({ sessionId })` lookup is a single pointer read
 * followed by one snapshot read, instead of scanning and parsing every snapshot
 * file in the prefix directory.
 */
interface PointerDoc {
  currentSnapshotId: string;
  updatedAt: string;
}

/**
 * Hidden sub-directory (within each prefix) holding the per-session
 * {@link PointerDoc} files. It is a directory, so the snapshot scan in
 * {@link FileSessionStore.getLatestSnapshotForSession} - which only considers
 * `*.json` *files* - naturally skips it.
 */
const POINTERS_SUBDIR = '.pointers';

/**
 * A Node.js file-system backed session snapshot store.
 *
 * Snapshots are stored as flat JSON files keyed by their `snapshotId`, under an
 * optional per-tenant sub-directory `prefix`:
 *
 * File layout: `dirPath/<prefix>/<snapshotId>.json`
 *
 * `getSnapshot({ sessionId })` resolves the session's current leaf via a tiny
 * per-session pointer file (`<prefix>/.pointers/<sessionId>.json`, see
 * {@link PointerDoc}) - one pointer read plus one snapshot read. When the
 * pointer is missing (e.g. a legacy store) or stale it transparently falls back
 * to scanning the prefix directory and selecting the single leaf whose
 * `sessionId` matches, then rewrites the pointer so subsequent lookups are fast
 * again.
 */
export class FileSessionStore<S = unknown> implements SessionStore<S> {
  private dirPath: string;
  private maxPersistedChainLength?: number;
  private snapshotPathPrefix?: (options?: SessionStoreOptions) => string;
  private rejectBranchingSessions: boolean;
  private snapshotWatchPollIntervalMs: number;

  /**
   * Per-file write locks. The {@link SessionStore} contract (and the
   * abort-aware mutator that branches on `current.status`) assumes
   * read-modify-write is atomic, but on the file system a read and the
   * `writeFile` below it are not. Without a lock two concurrent saves can
   * read the same `current` and the later write clobbers the earlier one
   * (e.g. a `completed` write overwriting a concurrent `aborted`). We
   * serialize saves per resolved file path with a simple promise chain.
   */
  private writeLocks = new Map<string, Promise<unknown>>();

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
   * @param options.snapshotWatchPollIntervalMs Polling interval (ms) for the
   *   {@link FileSessionStore.onSnapshotStateChange} fallback that backstops
   *   `fs.watch` (which can miss events on some filesystems, e.g. network
   *   mounts). Defaults to {@link DEFAULT_SNAPSHOT_WATCH_POLL_INTERVAL_MS}.
   */
  constructor(
    dirPath: string,
    options?: {
      maxPersistedChainLength?: number;
      snapshotPathPrefix?: (options?: SessionStoreOptions) => string;
      rejectBranchingSessions?: boolean;
      snapshotWatchPollIntervalMs?: number;
    }
  ) {
    this.dirPath = path.resolve(dirPath);
    fs.mkdirSync(this.dirPath, { recursive: true });
    this.maxPersistedChainLength = options?.maxPersistedChainLength;
    this.snapshotPathPrefix = options?.snapshotPathPrefix;
    this.rejectBranchingSessions = options?.rejectBranchingSessions ?? false;
    this.snapshotWatchPollIntervalMs =
      options?.snapshotWatchPollIntervalMs ??
      DEFAULT_SNAPSHOT_WATCH_POLL_INTERVAL_MS;
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
    assertSafeSnapshotId(snapshotId);
    const dir = this.prefixDir(options);
    await this.ensureDir(dir);
    // Defense in depth: even after the basename check, confirm the resolved
    // path stays inside the prefix directory.
    const filePath = path.join(dir, `${snapshotId}.json`);
    const resolvedDir = path.resolve(dir);
    const resolved = path.resolve(filePath);
    if (
      resolved !== path.join(resolvedDir, `${snapshotId}.json`) ||
      !resolved.startsWith(resolvedDir + path.sep)
    ) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Invalid snapshotId: "${snapshotId}". Resolved path escapes the snapshot directory.`,
      });
    }
    return filePath;
  }

  /** Resolves the (per-tenant) directory holding per-session pointer files. */
  private pointersDir(options?: SessionStoreOptions): string {
    return path.join(this.prefixDir(options), POINTERS_SUBDIR);
  }

  /**
   * Resolves the pointer file path for a session, validating `sessionId` is a
   * plain basename so it can never escape the pointers directory. Pure: it does
   * not create the directory, so the read path stays side-effect free. The
   * write path calls {@link ensureDir} before writing.
   */
  private getPointerPath(
    sessionId: string,
    options?: SessionStoreOptions
  ): string {
    assertSafeId(sessionId, 'sessionId');
    return path.join(this.pointersDir(options), `${sessionId}.json`);
  }

  /**
   * Reads the per-session {@link PointerDoc}, or `undefined` when it is missing
   * (legacy store / not yet written) or unreadable / corrupt - callers fall
   * back to a full directory scan in that case. Best-effort: any IO/parse error
   * resolves to `undefined` so the optimization can never make a lookup (or
   * save) fail where the scan-only baseline would have succeeded. An invalid
   * `sessionId` still throws (path validation is resolved outside the try) so it
   * fails fast rather than silently being ignored.
   */
  private async readPointer(
    sessionId: string,
    options?: SessionStoreOptions
  ): Promise<PointerDoc | undefined> {
    const filePath = this.getPointerPath(sessionId, options);
    let contents: string;
    try {
      contents = await fsp.readFile(filePath, 'utf-8');
    } catch {
      // Missing / unreadable pointer: fall back to the scan.
      return undefined;
    }
    try {
      const parsed = JSON.parse(contents) as PointerDoc;
      // Guard the type too: a non-string id would later throw in `assertSafeId`
      // on the fast path instead of falling back to the scan.
      return typeof parsed?.currentSnapshotId === 'string' ? parsed : undefined;
    } catch {
      // Partially written / corrupt pointer: treat as absent and rescan.
      return undefined;
    }
  }

  /**
   * Atomically writes the per-session {@link PointerDoc}. Best-effort: a
   * pointer write failure is swallowed since the pointer is only an
   * optimization - `sessionId` lookups still self-heal via the full scan. An
   * invalid `sessionId` still throws (path validation is resolved outside the
   * try) so it fails fast rather than silently being ignored.
   */
  private async writePointer(
    sessionId: string,
    currentSnapshotId: string,
    options?: SessionStoreOptions
  ): Promise<void> {
    const filePath = this.getPointerPath(sessionId, options);
    const dir = this.pointersDir(options);
    try {
      await this.ensureDir(dir);
      const pointer: PointerDoc = {
        currentSnapshotId,
        updatedAt: new Date().toISOString(),
      };
      await this.atomicWrite(filePath, JSON.stringify(pointer, null, 2));
    } catch {
      // Ignore: the scan fallback keeps `sessionId` lookups correct.
    }
  }

  /**
   * Serializes async work per resolved file path so a read-modify-write in
   * {@link saveSnapshot} is not interleaved with a concurrent one for the same
   * snapshot (see {@link writeLocks}).
   */
  private async withFileLock<T>(
    filePath: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const prev = this.writeLocks.get(filePath) ?? Promise.resolve();
    // Wait for any in-flight save of this file, ignoring its result/error so a
    // prior failure doesn't poison the lock for subsequent callers.
    const gate = prev.then(
      () => undefined,
      () => undefined
    );
    this.writeLocks.set(filePath, gate);
    await gate;
    try {
      return await fn();
    } finally {
      // If no one chained after us we are the tail; drop the entry to avoid
      // leaking a map entry per snapshotId.
      if (this.writeLocks.get(filePath) === gate) {
        this.writeLocks.delete(filePath);
      }
    }
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
   * Resolves the latest (leaf) snapshot for a session.
   *
   * Fast path: read the per-session pointer file and load the leaf it names -
   * one pointer read plus one snapshot read, independent of session count /
   * length. The pointer is skipped (and the scan used) when
   * `rejectBranchingSessions` is set, since detecting branches requires seeing
   * every leaf.
   *
   * Fallback (no/stale/corrupt pointer, or branch detection): scan every
   * snapshot file in the prefix directory, keep those whose `sessionId`
   * matches, select the single leaf, and refresh the pointer so later lookups
   * take the fast path.
   *
   * Known limitation: the fast path trusts the pointer when the snapshot it
   * names still exists and belongs to the session - it does not re-verify that
   * it is the actual leaf. So if a save succeeds but the subsequent (best-effort)
   * `writePointer` does not (crash/disk error), or two new saves for the same
   * session race and the older one writes the pointer last, the pointer can
   * linger on a valid-but-older same-session snapshot and lookups return it
   * until the next save advances the pointer. This is the accepted trade-off for
   * a best-effort cache: verifying leaf-ness on every read would reintroduce the
   * full scan the pointer exists to avoid. Callers needing strict guarantees can
   * resume by `snapshotId`, or set `rejectBranchingSessions` (which always
   * scans).
   */
  private async getLatestSnapshotForSession(
    sessionId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S> | undefined> {
    // Fast path via the pointer file (skipped when we must detect branching).
    if (!this.rejectBranchingSessions) {
      const pointer = await this.readPointer(sessionId, options);
      if (pointer) {
        const snap = await this.getSnapshotById(
          pointer.currentSnapshotId,
          options
        );
        // Honor the pointer only when the leaf still exists and belongs to this
        // session; otherwise it is stale - fall through to the scan, which also
        // rewrites the pointer.
        if (snap && (snap.sessionId ?? snap.state?.sessionId) === sessionId) {
          return snap;
        }
      }
    }

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
        if ((snap.sessionId ?? snap.state?.sessionId) === sessionId) {
          snapshots.push(snap);
        }
      } catch (e: unknown) {
        if ((e as NodeJS.ErrnoException).code === 'ENOENT') continue;
        throw e;
      }
    }

    const leaf = selectLeafSnapshot(
      snapshots,
      sessionId,
      this.rejectBranchingSessions
    );
    if (!leaf) return undefined;

    // Refresh the pointer so subsequent lookups take the fast path.
    // Best-effort.
    await this.writePointer(sessionId, leaf.snapshotId, options);
    return leaf;
  }

  async saveSnapshot(
    snapshotId: string | undefined,
    mutator: SnapshotMutator<S>,
    options?: SessionStoreOptions
  ): Promise<string | null> {
    // When an ID is supplied the read-modify-write below must be serialized
    // against concurrent saves of the same snapshot, otherwise a later write
    // (e.g. `completed`) can clobber an earlier concurrent one (e.g.
    // `aborted`). New (UUID) snapshots have no contender, so skip the lock.
    if (snapshotId) {
      const lockPath = await this.getFilePath(snapshotId, options);
      return this.withFileLock(lockPath, () =>
        this.saveSnapshotUnlocked(snapshotId, mutator, options)
      );
    }
    return this.saveSnapshotUnlocked(snapshotId, mutator, options);
  }

  private async saveSnapshotUnlocked(
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
    await this.atomicWrite(filePath, JSON.stringify(full, null, 2));

    // Maintain the per-session pointer so `sessionId` lookups stay fast (one
    // pointer read plus one snapshot read). Only a brand-new snapshot (`!current`)
    // is a new leaf, so only then do we advance the pointer; rewriting an
    // existing snapshot (heartbeat / status update) never changes leaf-ness, so
    // we leave the pointer alone. Snapshots without a `sessionId` are not
    // addressable by session, so skip the pointer.
    const sessionId = full.sessionId ?? full.state?.sessionId;
    if (sessionId && !current) {
      await this.writePointer(sessionId, id, options);
    }

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

  /**
   * Writes `contents` to `filePath` atomically: write to a temp file in the
   * same directory, then rename over the target. `rename` is atomic on POSIX
   * and Windows, so a concurrent reader in {@link getSnapshot} never observes a
   * half-written (torn) file.
   */
  private async atomicWrite(filePath: string, contents: string): Promise<void> {
    const tmpPath = `${filePath}.${process.pid}.${globalThis.crypto.randomUUID()}.tmp`;
    try {
      await fsp.writeFile(tmpPath, contents, 'utf-8');
      await fsp.rename(tmpPath, filePath);
    } catch (e) {
      await fsp.unlink(tmpPath).catch(() => {});
      throw e;
    }
  }

  /**
   * Watches a single snapshot file for changes and invokes `callback` with the
   * parsed snapshot whenever it changes.
   *
   * Unlike {@link InMemorySessionStore}, file-backed snapshots are frequently
   * mutated by a *different* process (e.g. the request handler that received an
   * abort writes `status: 'aborted'`, while a detached background worker is the
   * one watching). Detecting that requires observing the filesystem rather than
   * in-process `saveSnapshot` calls.
   *
   * Reliability comes from two layers:
   * - `fs.watch` on the (per-tenant) prefix directory, filtered to the target
   *   `<snapshotId>.json`. This is low latency but can miss events on some
   *   filesystems (network mounts, certain container volumes).
   * - A polling fallback (`snapshotWatchPollIntervalMs`) that re-reads the file
   *   on an interval, backstopping any events `fs.watch` drops. Its timer is
   *   `unref`'d so it never keeps the process alive on its own.
   *
   * Callbacks are de-duplicated by serialized content, so the noisy/duplicate
   * events `fs.watch` emits collapse into one callback per real change.
   * Transient read errors (e.g. a partially written file mid-rewrite, or a
   * not-yet-created file) are swallowed; the next event/poll re-reads.
   *
   * @returns An unsubscribe function that stops watching and polling.
   */
  onSnapshotStateChange(
    snapshotId: string,
    callback: (snapshot: SessionSnapshot<S>) => void,
    options?: SessionStoreOptions
  ): void | (() => void) {
    const dir = this.prefixDir(options);
    fs.mkdirSync(dir, { recursive: true });
    const fileName = `${snapshotId}.json`;
    const filePath = path.join(dir, fileName);

    let closed = false;
    let lastSerialized: string | undefined;

    // Re-read the file and fire the callback only when the content actually
    // changed. `fs.watch` fires multiple events per write, so dedupe by
    // serialized content.
    const emitIfChanged = async () => {
      if (closed) return;
      let contents: string;
      try {
        contents = await fsp.readFile(filePath, 'utf-8');
      } catch (e: unknown) {
        // Missing file (not yet created) or a transient read error during a
        // concurrent rewrite: ignore and wait for the next event/poll.
        return;
      }
      if (closed || contents === lastSerialized) return;
      let snapshot: SessionSnapshot<S>;
      try {
        snapshot = JSON.parse(contents) as SessionSnapshot<S>;
      } catch {
        // Partially written file mid-rewrite: skip without updating
        // lastSerialized so the next event/poll re-reads the complete file.
        return;
      }
      lastSerialized = contents;
      callback(snapshot);
    };

    // Watch the directory (not the file) so this still works before the file
    // exists and survives atomic rename-replace writes that swap the inode.
    let watcher: fs.FSWatcher | undefined;
    try {
      watcher = fs.watch(dir, (_event, changed) => {
        // `changed` can be null on some platforms; re-check in that case.
        if (!changed || changed === fileName) void emitIfChanged();
      });
    } catch {
      // Some environments disallow fs.watch; the polling fallback covers us.
    }

    // Polling fallback: backstops events fs.watch may drop. `unref` so the
    // timer never keeps the process alive on its own.
    const pollTimer = setInterval(() => {
      void emitIfChanged();
    }, this.snapshotWatchPollIntervalMs);
    pollTimer.unref?.();

    // Surface the current state immediately (if the file already exists).
    void emitIfChanged();

    return () => {
      closed = true;
      watcher?.close();
      clearInterval(pollTimer);
    };
  }
}
