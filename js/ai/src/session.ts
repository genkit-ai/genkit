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
  inputVariables: z.any().optional(),
});

/**
 * State persisted for a session across turns.
 */
export interface SessionState<S = unknown, I = unknown> {
  messages?: MessageData[];
  custom?: S;
  artifacts?: Artifact[];
  inputVariables?: I;
}

/**
 * The execution context provided to a snapshot callback.
 */
export interface SnapshotContext<S = unknown, I = unknown> {
  state: SessionState<S, I>;
  prevState?: SessionState<S, I>;
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
export interface SessionSnapshot<S = unknown, I = unknown> {
  snapshotId: string;
  parentId?: string;
  createdAt: string;
  event: 'turnEnd' | 'invocationEnd';
  state: SessionState<S, I>;
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
export interface SessionStore<S = unknown, I = unknown> {
  getSnapshot(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S, I> | undefined>;
  saveSnapshot(
    snapshot: SessionSnapshot<S, I>,
    options?: SessionStoreOptions
  ): Promise<void>;
  onSnapshotStateChange?(
    snapshotId: string,
    callback: (snapshot: SessionSnapshot<S, I>) => void,
    options?: SessionStoreOptions
  ): void | (() => void);
}

/**
 * State manager for a session turn, tracking messages, custom state, and artifacts.
 */
export class Session<S = unknown, I = unknown> extends EventEmitter {
  private state: SessionState<S, I>;
  private version: number = 0;

  constructor(initialState: SessionState<S, I>) {
    super();
    this.state = initialState;
  }

  /**
   * Returns a deep copy of the current session state.
   */
  getState(): SessionState<S, I> {
    return JSON.parse(JSON.stringify(this.state));
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
   */
  addArtifacts(artifacts: Artifact[]) {
    const existing = this.state.artifacts || [];
    for (const a of artifacts) {
      let replaced = false;
      if (a.name) {
        const idx = existing.findIndex((e) => e.name === a.name);
        if (idx >= 0) {
          existing[idx] = a;
          replaced = true;
          break;
        }
      }
      if (!replaced) {
        existing.push(a);
      }
    }
    this.state.artifacts = existing;
    this.version++;
    for (const a of artifacts) {
      this.emit('artifactAdded', a);
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
export class InMemorySessionStore<S = unknown, I = unknown>
  implements SessionStore<S, I>
{
  private snapshots = new Map<string, SessionSnapshot<S, I>>();
  private listeners = new Map<
    string,
    Array<(snapshot: SessionSnapshot<S, I>) => void>
  >();

  async getSnapshot(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S, I> | undefined> {
    const snap = this.snapshots.get(snapshotId);
    if (!snap) return undefined;
    return JSON.parse(JSON.stringify(snap));
  }

  async saveSnapshot(
    snapshot: SessionSnapshot<S, I>,
    options?: SessionStoreOptions
  ): Promise<void> {
    this.snapshots.set(
      snapshot.snapshotId,
      JSON.parse(JSON.stringify(snapshot))
    );
    const snapshotListeners = this.listeners.get(snapshot.snapshotId);
    if (snapshotListeners) {
      for (const listener of snapshotListeners) {
        listener(JSON.parse(JSON.stringify(snapshot)));
      }
    }
  }

  onSnapshotStateChange(
    snapshotId: string,
    callback: (snapshot: SessionSnapshot<S, I>) => void,
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
