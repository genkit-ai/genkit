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
  GenkitError,
  getAsyncContext,
  z,
  type ActionContext,
} from '@genkit-ai/core';

import { EventEmitter } from '@genkit-ai/core/async';
import type { Registry } from '@genkit-ai/core/registry';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';
import { MessageData, MessageSchema } from './model-types.js';

import { PartSchema } from './model-types.js';

/**
 * Schema for tracking persistent artifacts generated during a session turn.
 */
export const ArtifactSchema = z.object({
  name: z.string().optional(),
  parts: z.array(PartSchema),
  metadata: z.record(z.any()).optional(),
});

/**
 * Artifact generated during a session turn.
 */
export type Artifact = z.infer<typeof ArtifactSchema>;

/**
 * Events signifying a session snapshot persistence point.
 */
export const SnapshotEventSchema = z.enum(['turnEnd', 'invocationEnd']);

/**
 * Event signifying a session snapshot persistence point.
 */
export type SnapshotEvent = z.infer<typeof SnapshotEventSchema>;

/**
 * Reason an agent turn (or whole invocation) finished.
 *
 * Mirrors the generate `FinishReason` enum and adds two agent-specific
 * states: `detached` (the turn was moved to the background) and `failed`
 * (the turn ended in an error).
 */
export const AgentFinishReasonSchema = z.enum([
  // From generate's FinishReason:
  'stop',
  'length',
  'blocked',
  'aborted',
  'interrupted',
  'other',
  'unknown',
  // Agent additions:
  'detached',
  'failed',
]);

/**
 * Reason an agent turn (or whole invocation) finished.
 */
export type AgentFinishReason = z.infer<typeof AgentFinishReasonSchema>;

/**
 * Schema for session execution state.

 */
export const SessionStateSchema = z.object({
  sessionId: z.string().optional(),
  messages: z.array(MessageSchema).optional(),
  custom: z.any().optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * State persisted for a session across turns.
 */
export interface SessionState<S = unknown> {
  sessionId?: string;
  messages?: MessageData[];
  custom?: S;
  artifacts?: Artifact[];
}

/**
 * The execution context provided to a snapshot callback.
 */
export interface SnapshotContext<S = unknown> {
  state: SessionState<S>;
  prevState?: SessionState<S>;
  turnIndex: number;
  event: 'turnEnd' | 'invocationEnd';
}

/**
 * Callback triggered before a snapshot is saved. Return false to reject persistence.
 */
export type SnapshotCallback<S = unknown> = (
  ctx: SnapshotContext<S>
) => boolean;

/**
 * Saved snapshot of a session's state at a given event point.
 */
export interface SessionSnapshot<S = unknown> {
  snapshotId: string;
  parentId?: string;
  createdAt: string;
  event: 'turnEnd' | 'invocationEnd';
  state: SessionState<S>;
  status?: 'pending' | 'done' | 'failed' | 'aborted';

  /**
   * Semantic reason the turn/invocation finished (e.g. `interrupted`,
   * `stop`). Distinct from `status`, which tracks the persistence lifecycle.
   */
  finishReason?: AgentFinishReason;

  error?: {
    status: string;

    message: string;
    details?: any;
  };
}

/**
 * Input type for {@link SessionStore.saveSnapshot}.
 *
 * Identical to {@link SessionSnapshot} except that `snapshotId` is optional.
 * When omitted the store is responsible for assigning a new identifier
 * (enabling stores to encode grouping or routing information in the ID).
 * When provided the store performs an upsert - updating the existing snapshot.
 */
export type SessionSnapshotInput<S = unknown> = Omit<
  SessionSnapshot<S>,
  'snapshotId'
> & {
  snapshotId?: string;
};

/**
 * Options provided to the session store methods.
 */
export interface SessionStoreOptions {
  context?: ActionContext;
}

/**
 * Lookup options for {@link SessionStore.getSnapshot}.
 *
 * Exactly one of `snapshotId` or `sessionId` must be provided:
 *
 * - `snapshotId` loads that specific snapshot.
 * - `sessionId` loads the *latest* (leaf) snapshot of the session - the most
 *   recent snapshot that no other snapshot points to as its parent. This is
 *   the common case for simple session storage (e.g. `useChat`) where the
 *   client only tracks a stable session id and lets the server remember the
 *   conversation. A session with *branching* snapshots (more than one leaf,
 *   e.g. after a regenerate) has no single "latest", so the store rejects the
 *   lookup with `FAILED_PRECONDITION` - resume by `snapshotId` instead.
 */
export interface GetSnapshotOptions {
  snapshotId?: string;
  sessionId?: string;
  context?: ActionContext;
}

/**
 * A function that receives the current snapshot and returns the updated
 * snapshot to persist.
 *
 * - Return the mutated snapshot to save it.
 * - Return `null` to silently skip the update (no-op).
 * - Throw to abort with an error (e.g. precondition failure).
 */
export type SnapshotMutator<S = unknown> = (
  current: SessionSnapshot<S> | undefined
) => SessionSnapshotInput<S> | null;

/**
 * Interface for persistent session snapshot storage.
 */
export interface SessionStore<S = unknown> {
  /**
   * Loads a snapshot either by its `snapshotId` or by `sessionId`.
   *
   * See {@link GetSnapshotOptions} for the lookup semantics (exactly one of
   * `snapshotId` / `sessionId`; `sessionId` resolves to the session's latest
   * leaf snapshot and rejects branching histories).
   */
  getSnapshot(
    opts: GetSnapshotOptions
  ): Promise<SessionSnapshot<S> | undefined>;

  /**
   * Atomically reads the current snapshot (if `snapshotId` is provided),
   * passes it to `mutator`, and persists the result.
   *
   * - When `snapshotId` is provided the store reads the existing snapshot
   *   and passes it to the mutator.  The mutator can inspect the current
   *   state (e.g. to check for concurrent status changes) and return the
   *   updated snapshot to save, or `null` to skip the write.
   * - When `snapshotId` is `undefined` the store passes `undefined` to
   *   the mutator (signaling a new snapshot).  The store assigns a new
   *   identifier.
   *
   * Implementations should ensure the read→mutate→write cycle is atomic
   * to prevent race conditions (e.g. a "done" write overwriting a
   * concurrent "aborted" status).
   *
   * The mutator can:
   *
   * - Return a snapshot to save it.
   * - Return `null` to silently skip the write.
   * - Throw to abort with an error.
   *
   * @returns The `snapshotId` that was used, or `null` when the mutator
   *   returned `null`.
   */
  saveSnapshot(
    snapshotId: string | undefined,
    mutator: SnapshotMutator<S>,
    options?: SessionStoreOptions
  ): Promise<string | null>;

  onSnapshotStateChange?(
    snapshotId: string,
    callback: (snapshot: SessionSnapshot<S>) => void,
    options?: SessionStoreOptions
  ): void | (() => void);
}

/**
 * State manager for a session turn, tracking messages, custom state, and artifacts.
 */
export class Session<S = unknown> extends EventEmitter {
  private state: SessionState<S>;
  private version: number = 0;

  /** Stable identifier that correlates traces across agent turns. */
  readonly sessionId: string;

  constructor(initialState: SessionState<S>) {
    super();
    this.sessionId = initialState.sessionId || globalThis.crypto.randomUUID();
    initialState.sessionId = this.sessionId;
    this.state = initialState;
  }

  /**
   * Returns a deep copy of the current session state.
   */
  getState(): SessionState<S> {
    return structuredClone(this.state);
  }

  /**
   * Retrieves all messages associated with the session.
   */
  getMessages(): MessageData[] {
    return this.state.messages || [];
  }

  /**
   * Appends a list of messages to the session.
   */
  addMessages(messages: MessageData[]) {
    this.state.messages = [...(this.state.messages || []), ...messages];
    this.version++;
  }

  /**
   * Overwrites the session messages.
   */
  setMessages(messages: MessageData[]) {
    this.state.messages = messages;
    this.version++;
  }

  /**
   * Retrieves the custom state of the session.
   */
  getCustom(): S | undefined {
    return this.state.custom;
  }

  /**
   * Updates the custom state of the session using a mutator function.
   */
  updateCustom(fn: (custom?: S) => S) {
    this.state.custom = fn(this.state.custom);
    this.version++;
    this.emit('customChanged');
  }

  /**
   * Retrieves the list of artifacts generated during the session.
   */
  getArtifacts(): Artifact[] {
    return this.state.artifacts || [];
  }

  /**
   * Adds artifacts to the session, deduplicating items by name.
   * Emits 'artifactAdded' for new artifacts and 'artifactUpdated' for replacements.
   */
  addArtifacts(artifacts: Artifact[]) {
    const existing = this.state.artifacts || [];
    const added: Artifact[] = [];
    const updated: Artifact[] = [];

    for (const a of artifacts) {
      if (a.name) {
        const idx = existing.findIndex((e) => e.name === a.name);
        if (idx >= 0) {
          existing[idx] = a;
          updated.push(a);
          continue;
        }
      }
      existing.push(a);
      added.push(a);
    }

    this.state.artifacts = existing;
    if (added.length + updated.length > 0) {
      this.version++;
    }
    for (const a of added) {
      this.emit('artifactAdded', a);
    }
    for (const a of updated) {
      this.emit('artifactUpdated', a);
    }
  }

  /**
   * Runs the provided function inside the session's context.
   */
  run<O>(fn: () => O) {
    return getAsyncContext().run('ai.session', this, fn);
  }

  /**
   * Gets the current mutation version of the session state.
   */
  getVersion(): number {
    return this.version;
  }
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
    const leaf = selectLeafSnapshot(owned, sessionId!);
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

    // Determine the final ID. For new snapshots compose a `s_{convoId}_{suffix}`
    // id where convoId is derived from the session id (so all snapshots of a
    // session share a grouping key for `getSnapshot({ sessionId })`).
    const id =
      snapshotId ||
      result.snapshotId ||
      composeSnapshotId(deriveConvoId(result), generateSnapshotSuffix());
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
 * Utility to execute a function bound to a Session instance context.
 */
export function runWithSession<S = any, O = any>(
  registry: Registry,
  session: Session<S>,
  fn: () => O
): O {
  return getAsyncContext().run('ai.session', session, fn);
}

/**
 * Returns the Session instance active in the current context.
 */
export function getCurrentSession<S = any>(
  registry: Registry
): Session<S> | undefined {
  return getAsyncContext().getStore('ai.session');
}

/**
 * Error thrown during session execution.
 */
export class SessionError extends Error {
  constructor(msg: string) {
    super(msg);
  }
}

// Only UUID-shaped strings are accepted for the convoId / sessionId component.
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// The suffix part (after the convoId) must be alphanumeric / hyphens / underscores.
const SAFE_SUFFIX_PATTERN = /^[0-9a-zA-Z_-]+$/;

/**
 * Prefix that visually distinguishes a `snapshotId` from a bare `sessionId`.
 *
 * A `sessionId` is a plain UUID (e.g. `8f3c…`); a `snapshotId` is
 * `s_{sessionId}_{epochMs}_{random}`. The `s_` prefix lets both humans and the
 * runtime tell at a glance (and programmatically) which kind of id they hold.
 */
const SNAPSHOT_ID_PREFIX = 's_';

/**
 * Returns `true` when `id` looks like a snapshotId (carries the `s_` prefix).
 */
export function isSnapshotId(id: string): boolean {
  return id.startsWith(SNAPSHOT_ID_PREFIX);
}

/**
 * Validates that `sessionId` is a UUID, throwing a descriptive error otherwise.
 *
 * Session ids double as the grouping key (`convoId`) embedded in every
 * `snapshotId`, so they must be UUID-shaped for the id format to round-trip.
 */
export function assertValidSessionId(sessionId: string): void {
  if (!UUID_PATTERN.test(sessionId)) {
    throw new Error(
      `Invalid sessionId: expected a UUID, got "${sessionId}". ` +
        `Session ids must be UUIDs (e.g. crypto.randomUUID()).`
    );
  }
}

/**
 * Generates a short, unique suffix for a snapshot ID.
 *
 * Format: `{epochMs}_{random4}` - e.g. `1747000878123_k9m2`
 */
function generateSnapshotSuffix(): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).slice(2, 6);
  return `${timestamp}_${random}`;
}

/**
 * Composes a snapshot ID from a conversation ID and a short suffix.
 *
 * Format: `s_{convoId}_{epochMs}_{random}`
 */
function composeSnapshotId(convoId: string, suffix: string): string {
  return `${SNAPSHOT_ID_PREFIX}${convoId}_${suffix}`;
}

/**
 * Parses a composite snapshot ID into its conversation ID and file-name suffix.
 *
 * The id is `s_{convoId}_{suffix}` where convoId is a UUID (36 chars with
 * hyphens). Since UUIDs never contain underscores, the first `_` after the
 * UUID boundary reliably separates the two parts.
 *
 * @throws If the ID cannot be parsed or the convoId is not a valid UUID.
 */
function parseSnapshotId(snapshotId: string): {
  convoId: string;
  suffix: string;
} {
  if (!snapshotId.startsWith(SNAPSHOT_ID_PREFIX)) {
    throw new Error(
      `Invalid snapshotId: expected format "s_{uuid}_{suffix}", got ` +
        `"${snapshotId}". (A bare UUID is a sessionId, not a snapshotId.)`
    );
  }
  const body = snapshotId.slice(SNAPSHOT_ID_PREFIX.length);
  // UUID is always 36 chars (8-4-4-4-12).  The separator `_` follows at index 36.
  if (body.length < 38 || body[36] !== '_') {
    throw new Error(
      `Invalid snapshotId: expected format "s_{uuid}_{suffix}", got "${snapshotId}"`
    );
  }
  const convoId = body.slice(0, 36);
  const suffix = body.slice(37);
  if (!UUID_PATTERN.test(convoId)) {
    throw new Error(
      `Invalid snapshotId: convoId component is not a valid UUID ("${convoId}")`
    );
  }
  if (!suffix || !SAFE_SUFFIX_PATTERN.test(suffix)) {
    throw new Error(
      `Invalid snapshotId: suffix component is invalid ("${suffix}")`
    );
  }
  return { convoId, suffix };
}

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
 * Derives the convoId to embed in a *new* snapshot's id.
 *
 * Prefers the snapshot's own `state.sessionId` (so all snapshots of a session
 * share a convoId and `getSnapshot({ sessionId })` can group them), falling
 * back to the parent's convoId, then to a fresh random UUID.
 */
function deriveConvoId<S>(snapshot: SessionSnapshotInput<S>): string {
  const sessionId = snapshot.state?.sessionId;
  if (sessionId) {
    assertValidSessionId(sessionId);
    return sessionId;
  }
  if (snapshot.parentId) {
    return parseSnapshotId(snapshot.parentId).convoId;
  }
  return globalThis.crypto.randomUUID();
}

/**
 * Selects the single leaf (latest) snapshot from a set belonging to one
 * session.
 *
 * A "leaf" is a snapshot that no other snapshot points to as its `parentId`.
 * A healthy linear session has exactly one leaf - the latest turn.
 *
 * - Returns `undefined` when `snapshots` is empty.
 * - Returns the single leaf when the history is linear.
 * - Throws `FAILED_PRECONDITION` when the history has branched (more than one
 *   leaf, e.g. after a regenerate), since there is no unambiguous "latest".
 */
function selectLeafSnapshot<S>(
  snapshots: SessionSnapshot<S>[],
  sessionId: string
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

  throw new GenkitError({
    status: 'FAILED_PRECONDITION',
    message:
      `Session '${sessionId}' has branching snapshots (${leaves.length} ` +
      `leaves), so there is no single latest snapshot. This happens when a ` +
      `conversation is branched (e.g. regenerate). Resume by snapshotId instead.`,
  });
}

/**
 * A Node.js file-system backed session snapshot store.
 *
 * Snapshots belonging to the same conversation are grouped in a shared
 * sub-directory keyed by a conversation ID that is embedded in the
 * `snapshotId` itself.
 *
 * ID format: `{convoId}_{epochMs}_{random}`
 *
 * File layout: `dirPath/<prefix>/<convoId>/<epochMs>_<random>.json`
 */
export class FileSessionStore<S = unknown> implements SessionStore<S> {
  private dirPath: string;
  private maxPersistedChainLength?: number;
  private snapshotPathPrefix?: (
    snapshotId: string,
    options?: SessionStoreOptions
  ) => string;

  /**
   * @param dirPath Directory where snapshot JSON files are stored.
   * @param options.maxPersistedChainLength When set, snapshots older than this
   *   many entries in a chain are automatically deleted on each save.
   * @param options.snapshotPathPrefix Returns a sub-directory prefix per
   *   snapshot, useful for multi-tenant isolation. Defaults to `"global"`.
   */
  constructor(
    dirPath: string,
    options?: {
      maxPersistedChainLength?: number;
      snapshotPathPrefix?: (
        snapshotId: string,
        options?: SessionStoreOptions
      ) => string;
    }
  ) {
    this.dirPath = path.resolve(dirPath);
    fs.mkdirSync(this.dirPath, { recursive: true });
    this.maxPersistedChainLength = options?.maxPersistedChainLength;
    this.snapshotPathPrefix = options?.snapshotPathPrefix;
  }

  private async ensureDir(dir: string): Promise<void> {
    await fsp.mkdir(dir, { recursive: true });
  }

  /**
   * Resolves the file path for a given composite snapshotId.
   */
  private async getFilePath(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<string> {
    const { convoId, suffix } = parseSnapshotId(snapshotId);
    const prefix = this.snapshotPathPrefix
      ? this.snapshotPathPrefix(snapshotId, options)
      : 'global';
    const dir = path.join(this.dirPath, prefix, convoId);
    await this.ensureDir(dir);
    return path.join(dir, `${suffix}.json`);
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
   * Resolves the latest (leaf) snapshot for a session by reading all snapshot
   * files in the session's directory and selecting the single leaf.
   *
   * The session's snapshots live under `<prefix>/<sessionId>/` because the
   * convoId embedded in each snapshotId equals the sessionId.
   */
  private async getLatestSnapshotForSession(
    sessionId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S> | undefined> {
    // Reuse getFilePath's prefix resolution by composing a representative id.
    const probeId = composeSnapshotId(sessionId, 'probe_probe');
    const prefix = this.snapshotPathPrefix
      ? this.snapshotPathPrefix(probeId, options)
      : 'global';
    const dir = path.join(this.dirPath, prefix, sessionId);

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
        snapshots.push(JSON.parse(contents) as SessionSnapshot<S>);
      } catch (e: unknown) {
        if ((e as NodeJS.ErrnoException).code === 'ENOENT') continue;
        throw e;
      }
    }

    return selectLeafSnapshot(snapshots, sessionId);
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

    // Determine the final ID.
    let id: string;
    if (snapshotId) {
      // Upsert - the caller supplied an ID.
      id = snapshotId;
    } else if (snapshot.snapshotId) {
      id = snapshot.snapshotId;
    } else {
      // New snapshot - derive the convoId from the session id (so all
      // snapshots of a session group under one directory), falling back to
      // the parent's convoId or a fresh conversation.
      id = composeSnapshotId(deriveConvoId(snapshot), generateSnapshotSuffix());
    }

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
