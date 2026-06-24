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
 * Saved snapshot of a session's state at a given event point.
 */
export interface SessionSnapshot<S = unknown> {
  snapshotId: string;
  parentId?: string;
  createdAt: string;
  /** When the snapshot was last written (RFC 3339). Equals `createdAt` until rewritten. */
  updatedAt?: string;
  state: SessionState<S>;
  status?: 'pending' | 'completed' | 'failed' | 'aborted' | 'expired';

  /**
   * Heartbeat timestamp (RFC 3339) refreshed periodically while a detached
   * (background) turn is in flight. Used to detect a dead background worker:
   * if a `pending` snapshot's heartbeat goes stale (older than the configured
   * timeout), reads surface its status as `expired` (the dead process can no
   * longer persist a terminal status itself).
   */
  heartbeatAt?: string;

  /**
   * Semantic reason the turn/invocation finished (e.g. `interrupted`,
   * `stop`). Distinct from `status`, which tracks the persistence lifecycle.
   */
  finishReason?: AgentFinishReason;

  /**
   * Structured failure information (RuntimeError shape). `status` is the
   * canonical error category (e.g. `INTERNAL`, `FAILED_PRECONDITION`).
   */
  error?: {
    status?: string;
    message: string;
    details?: any;
  };
}

/**
 * Zod schema mirroring {@link SessionSnapshot}. Used as the output schema for
 * the `getSnapshot` companion action so the snapshot shape is discoverable in
 * the registry (rather than an opaque `z.any()`).
 */
export const SessionSnapshotSchema = z.object({
  snapshotId: z.string(),
  parentId: z.string().optional(),
  createdAt: z.string(),
  updatedAt: z.string().optional(),
  state: SessionStateSchema,
  status: z
    .enum(['pending', 'completed', 'failed', 'aborted', 'expired'])
    .optional(),
  heartbeatAt: z.string().optional(),
  finishReason: AgentFinishReasonSchema.optional(),
  error: z
    .object({
      status: z.string().optional(),
      message: z.string(),
      details: z.any().optional(),
    })
    .optional(),
});

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
 *   e.g. after a regenerate) has no single "latest"; by default the store
 *   returns the most-recent leaf, but it can be configured to reject the
 *   lookup with `FAILED_PRECONDITION` - in which case, resume by `snapshotId`.
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
   * leaf snapshot, optionally rejecting branching histories).
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
    // Clone so we never alias (or mutate) the caller's object: the session
    // owns its state, and a handler mutating it must not reach back into the
    // caller's / chat's state.
    const state = structuredClone(initialState);
    this.sessionId = state.sessionId || globalThis.crypto.randomUUID();
    state.sessionId = this.sessionId;
    this.state = state;
  }

  /**
   * Returns a deep copy of the current session state.
   */
  getState(): SessionState<S> {
    return structuredClone(this.state);
  }

  /**
   * Retrieves all messages associated with the session.
   *
   * Returns a copy so callers cannot mutate the session's internal message
   * array in place.
   */
  getMessages(): MessageData[] {
    return structuredClone(this.state.messages || []);
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

/**
 * Validates that `sessionId` is a non-empty string, throwing a descriptive
 * error otherwise.
 *
 * Session ids can be minted by the client and can be any non-empty string
 * (e.g. a UUID, or an application-specific identifier). We only reject empty /
 * blank values so the id stays usable as a key (and as a directory name in
 * {@link FileSessionStore}).
 */
export function assertValidSessionId(sessionId: string): void {
  if (typeof sessionId !== 'string' || sessionId.trim() === '') {
    throw new Error(
      `Invalid sessionId: expected a non-empty string, got "${sessionId}".`
    );
  }
}

/**
 * Mints a new `snapshotId` (a plain random UUID).
 *
 * The runtime normally supplies the snapshotId to the store at save time, but
 * some flows need the id *ahead of time* - e.g. an agent turn that wants to
 * know the snapshotId at turn *start* (to name a git branch / worktree after
 * it) and have the snapshot persisted at turn end reuse that very id, or the
 * detach path which pre-reserves the in-flight snapshot's id.
 */
export function reserveSnapshotId(): string {
  return globalThis.crypto.randomUUID();
}
