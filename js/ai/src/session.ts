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

import { getAsyncContext, z, type ActionContext } from '@genkit-ai/core';
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
 * Schema for session execution state.
 */
export const SessionStateSchema = z.object({
  messages: z.array(MessageSchema).optional(),
  custom: z.any().optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * State persisted for a session across turns.
 */
export interface SessionState<S = unknown> {
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

  error?: {
    status: string;
    message: string;
    details?: any;
  };
}

/**
 * Options provided to the session store methods.
 */
export interface SessionStoreOptions {
  context?: ActionContext;
}

/**
 * Interface for persistent session snapshot storage.
 */
export interface SessionStore<S = unknown> {
  getSnapshot(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S> | undefined>;
  saveSnapshot(
    snapshot: SessionSnapshot<S>,
    options?: SessionStoreOptions
  ): Promise<void>;
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

  constructor(initialState: SessionState<S>) {
    super();
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
export class InMemorySessionStore<S = unknown>
  implements SessionStore<S>
{
  private snapshots = new Map<string, SessionSnapshot<S>>();
  private listeners = new Map<
    string,
    Array<(snapshot: SessionSnapshot<S>) => void>
  >();

  async getSnapshot(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S> | undefined> {
    const snap = this.snapshots.get(snapshotId);
    if (!snap) return undefined;
    return structuredClone(snap);
  }

  async saveSnapshot(
    snapshot: SessionSnapshot<S>,
    options?: SessionStoreOptions
  ): Promise<void> {
    this.snapshots.set(snapshot.snapshotId, structuredClone(snapshot));
    const snapshotListeners = this.listeners.get(snapshot.snapshotId);
    if (snapshotListeners) {
      for (const listener of snapshotListeners) {
        listener(structuredClone(snapshot));
      }
    }
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

// Only UUID-shaped IDs are accepted to prevent path traversal.
const SAFE_ID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * A Node.js file-system backed session snapshot store.
 *
 * Each snapshot is persisted as a JSON file under `dirPath/<prefix>/<snapshotId>.json`.
 * Only UUID-formatted snapshot IDs are accepted to prevent path traversal.
 */
export class FileSessionStore<S = unknown>
  implements SessionStore<S>
{
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

  private validateId(snapshotId: string): void {
    if (!SAFE_ID_PATTERN.test(snapshotId)) {
      throw new Error(`Invalid snapshotId: "${snapshotId}"`);
    }
  }

  private async ensureDir(dir: string): Promise<void> {
    await fsp.mkdir(dir, { recursive: true });
  }

  private async getFilePath(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<string> {
    this.validateId(snapshotId);
    const prefix = this.snapshotPathPrefix
      ? this.snapshotPathPrefix(snapshotId, options)
      : 'global';
    const dir = path.join(this.dirPath, prefix);
    await this.ensureDir(dir);
    return path.join(dir, `${snapshotId}.json`);
  }

  async getSnapshot(
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

  async saveSnapshot(
    snapshot: SessionSnapshot<S>,
    options?: SessionStoreOptions
  ): Promise<void> {
    const filePath = await this.getFilePath(snapshot.snapshotId, options);
    await fsp.writeFile(filePath, JSON.stringify(snapshot, null, 2), 'utf-8');

    if (this.maxPersistedChainLength && this.maxPersistedChainLength > 0) {
      let current: SessionSnapshot<S> | undefined = snapshot;
      const chain: string[] = [];

      while (current) {
        chain.push(current.snapshotId);
        if (current.parentId) {
          current = await this.getSnapshot(current.parentId, options);
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
  }
}
