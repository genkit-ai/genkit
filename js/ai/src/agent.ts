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
  deepEqual,
  defineAction,
  defineBidiAction,
  getContext,
  run,
  z,
  type Action,
  type ActionContext,
  type ActionFnArg,
  type BidiAction,
} from '@genkit-ai/core';
import { Channel } from '@genkit-ai/core/async';
import type { Registry } from '@genkit-ai/core/registry';
import {
  createAgentAPI,
  type AgentAPI,
  type AgentTransport,
  type SnapshotLookup,
} from './agent-core.js';

import { parseSchema, toJsonSchema } from '@genkit-ai/core/schema';
import {
  setCustomMetadataAttribute,
  setCustomMetadataAttributes,
} from '@genkit-ai/core/tracing';
import {
  AgentAbortRequestSchema,
  AgentAbortResponseSchema,
  AgentInitSchema,
  AgentInputSchema,
  AgentOutputSchema,
  AgentStreamChunkSchema,
  GetSnapshotRequestSchema,
  type AgentInit,
  type AgentInput,
  type AgentResult,
  type AgentStreamChunk,
} from './agent-types.js';
import { generateStream } from './generate.js';
import { diff, type JsonPatch } from './json-patch.js';
import { MessageData } from './model-types.js';
import { type ToolRequestPart, type ToolResponsePart } from './parts.js';
import {
  definePrompt,
  type PromptAction,
  type PromptConfig,
} from './prompt.js';
import { InMemorySessionStore } from './session-stores.js';
import {
  Session,
  SessionSnapshot,
  SessionSnapshotSchema,
  SessionState,
  SessionStore,
  reserveSnapshotId,
  runWithSession,
  type AgentFinishReason,
  type Artifact,
  type SessionSnapshotInput,
  type SessionStoreOptions,
} from './session.js';

// Re-export the shared agent/session wire schemas + types from their canonical
// home (./agent-types.ts) so existing imports from './agent.js' (and the
// package barrel) keep working.
export {
  AgentAbortRequestSchema,
  AgentAbortResponseSchema,
  AgentInitSchema,
  AgentInputSchema,
  AgentOutputSchema,
  AgentResultSchema,
  AgentStreamChunkSchema,
  GetSnapshotRequestSchema,
  JsonPatchOperationSchema,
  JsonPatchSchema,
  TurnEndSchema,
  type AgentInit,
  type AgentInput,
  type AgentResult,
  type AgentStreamChunk,
  type TurnEnd,
} from './agent-types.js';

/**
 * Default interval (ms) at which a detached (background) turn refreshes its
 * pending snapshot's heartbeat. Each beat is a write to the session store.
 */
const DEFAULT_HEARTBEAT_INTERVAL_MS = 30_000;

/**
 * Default staleness threshold (ms) after which a `pending` snapshot whose
 * heartbeat has not advanced is reported as `expired` on read. Should be
 * comfortably larger than {@link DEFAULT_HEARTBEAT_INTERVAL_MS} so a single
 * missed beat does not trip expiry.
 */
const DEFAULT_HEARTBEAT_TIMEOUT_MS = 60_000;

/**
 * Returns `true` when a snapshot is a `pending` (detached, in-flight) snapshot
 * whose heartbeat is older than `timeoutMs` - i.e. its background worker is
 * presumed dead. A pending snapshot that has not yet written a first heartbeat
 * is not considered expired (the beat may simply not have fired yet).
 */
function isHeartbeatExpired(
  snapshot: SessionSnapshot,
  timeoutMs: number = DEFAULT_HEARTBEAT_TIMEOUT_MS
): boolean {
  if (snapshot.status !== 'pending' || !snapshot.heartbeatAt) {
    return false;
  }
  const last = Date.parse(snapshot.heartbeatAt);
  if (Number.isNaN(last)) {
    return false;
  }
  return Date.now() - last > timeoutMs;
}

/**
 * Result returned by a single turn handler passed to {@link SessionRunner.run}.
 *
 * Returning a `finishReason` lets a custom agent explicitly state why the turn
 * ended (e.g. `interrupted`, `length`). When omitted, no per-turn reason is
 * reported.
 */
export interface TurnResult {
  finishReason?: AgentFinishReason;
}

/**
 * Per-turn context handed to the handler passed to {@link SessionRunner.run}.
 *
 * The `snapshotId` is *reserved at turn start* (before the handler runs) and is
 * the id the snapshot persisted at turn end will reuse. This lets a handler
 * name external, snapshot-correlated resources - e.g. a git branch / worktree
 * named after the snapshot - up front, then commit them under that id, so a
 * later rollback to the snapshot can restore the external state too.
 */
export interface TurnContext {
  /**
   * The id the snapshot produced by this turn will be saved under (reserved
   * ahead of time so it is known before the turn runs).
   */
  snapshotId: string;
  /**
   * The id of the parent snapshot this turn continues from, or `undefined` on
   * the first turn of a fresh session.
   */
  parentSnapshotId?: string;
  /** Zero-based index of this turn within the current invocation. */
  turnIndex: number;
}

/**
 * Output returned at turn completion.
 */
export interface AgentOutput<S = unknown> {
  sessionId?: string;
  artifacts?: Artifact[];
  message?: MessageData;
  snapshotId?: string;
  state?: SessionState<S>;
  finishReason?: AgentFinishReason;
  /**
   * Present when `finishReason` is `failed`. Carries the original error
   * details (RuntimeError shape); `state`/`snapshotId` hold the last-good state.
   */
  error?: {
    status?: string;
    message: string;
    details?: any;
  };
}

/**
 * Structured error details surfaced on the failure path.
 */
interface AgentErrorDetails {
  status: string;
  message: string;
  details?: any;
}

/**
 * Normalizes a thrown value into the structured error shape used across the
 * agent (in `AgentOutput.error` and `SessionRunner.lastTurnError`).
 */
function toErrorDetails(e: any): AgentErrorDetails {
  return {
    status: e?.status || 'INTERNAL',
    message: e?.message || 'Internal failure',
    // Only surface explicitly-provided structured details. Never fall back to
    // the raw thrown value: it is serialized over the wire (AgentOutput.error)
    // and persisted into snapshots, so leaking it could expose stack traces /
    // internal state, or break JSON.stringify on circular error objects.
    details: e?.detail ?? e?.details,
  };
}

/**
 * Builds an abort-aware `saveSnapshot` mutator: it skips the write (returns
 * `null`) when the current snapshot was concurrently aborted, otherwise writes
 * `input`. This prevents a "done"/"failed" write from clobbering an "aborted"
 * status set by a concurrent abort.
 */
function abortAwareMutator<S>(input: SessionSnapshotInput<S>) {
  return (current: SessionSnapshot<S> | undefined) =>
    current?.status === 'aborted' ? null : input;
}

/**
 * Asserts that an operation requiring a persistent store is not being invoked
 * on a store-less (client-managed) agent.
 */
function requireStore<S>(
  store: SessionStore<S> | undefined,
  operation: string,
  agentName: string
): asserts store is SessionStore<S> {
  if (!store) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `${operation} requires a persistent store. Provide a 'store' when defining '${agentName}'.`,
    });
  }
}

/**
 * Sets a snapshot's status to `aborted` (unless it already reached a terminal
 * state) and returns its previous status, or `undefined` when the snapshot
 * does not exist.
 */
async function abortSnapshotInStore<S>(
  store: SessionStore<S>,
  snapshotId: string,
  options?: SessionStoreOptions
): Promise<SessionSnapshot['status'] | undefined> {
  let previousStatus: SessionSnapshot['status'] | undefined;
  await store.saveSnapshot(
    snapshotId,
    (current) => {
      if (!current) return null;
      previousStatus = current.status;
      if (
        current.status === 'completed' ||
        current.status === 'failed' ||
        current.status === 'aborted'
      ) {
        return null; // Already terminal - don't override.
      }
      return { ...current, status: 'aborted' };
    },
    options
  );
  return previousStatus;
}

/**
 * Executor responsible for running turns over input streams and persisting state.
 */
export class SessionRunner<State = unknown> {
  readonly session: Session<State>;
  readonly inputCh: AsyncIterable<AgentInput>;

  turnIndex: number = 0;
  public onEndTurn?: (
    snapshotId?: string,
    finishReason?: AgentFinishReason
  ) => void;
  public onDetach?: (snapshotId: string) => void;
  public newSnapshotId?: string;
  /** The finish reason of the most recently completed turn. */
  public lastTurnFinishReason?: AgentFinishReason;
  /**
   * Error details of the most recent failed turn. Set when a turn throws and
   * the runner resolves gracefully instead of propagating the exception.
   */
  public lastTurnError?: AgentErrorDetails;
  /**
   * The state the most recently *successful* turn left behind. On a failed
   * turn this is the state the failed turn started with - the last-good state
   * returned to the caller (for client-managed agents).
   */
  public lastGoodState?: SessionState<State>;
  /**
   * The snapshotId of the most recently *successful* (persisted, `done`) turn.
   * On a failed turn this is the last-good snapshot the caller resumes from;
   * `undefined` when no turn has succeeded yet (e.g. a first-turn failure).
   */
  public lastGoodSnapshotId?: string;
  private lastSnapshot?: SessionSnapshot<State>;

  private lastSnapshotVersion: number = 0;

  private store?: SessionStore<State>;
  public isDetached: boolean = false;
  /**
   * Aborts in-flight turns. When set and aborted, a turn that rejects out of
   * `generate` is reported as `aborted` (not `failed`) and its failed snapshot
   * write is skipped (the abort path already persisted the `aborted` status).
   */
  private abortSignal?: AbortSignal;

  /**
   * True until the first `customPatch` chunk of the current turn has been
   * emitted. The first patch of every turn is a whole-document replace
   * (re-basing clients that may not share the server's baseline); reset to
   * `true` at the start of each turn.
   */
  public firstCustomPatchInTurn: boolean = true;

  constructor(
    session: Session<State>,
    inputCh: AsyncIterable<AgentInput>,
    options?: {
      lastSnapshot?: SessionSnapshot<State>;
      store?: SessionStore<State>;
      abortSignal?: AbortSignal;
      onEndTurn?: (
        snapshotId?: string,
        finishReason?: AgentFinishReason
      ) => void;
      onDetach?: (snapshotId: string) => void;
    }
  ) {
    this.session = session;
    this.inputCh = inputCh;

    this.lastSnapshot = options?.lastSnapshot;
    this.store = options?.store;
    this.abortSignal = options?.abortSignal;
    this.onEndTurn = options?.onEndTurn;
    this.onDetach = options?.onDetach;

    // Seed the last-good state with the initial session state so that a
    // failure on the very first turn still has a valid state to fall back to
    // (the seed/loaded state, excluding the failed turn's mutations). The
    // last-good snapshotId is undefined until a turn successfully persists.
    this.lastGoodState = this.session.getState();
    this.lastGoodSnapshotId = options?.lastSnapshot?.snapshotId;
  }

  // ── Session delegate methods ────────────────────────────────────────
  // These forward to `this.session` so callers can write `sess.addMessages()`
  // instead of the verbose `sess.session.addMessages()`.

  /** Returns a deep copy of the current session state. */
  getState(): SessionState<State> {
    return this.session.getState();
  }

  /** Retrieves all messages associated with the session. */
  getMessages(): MessageData[] {
    return this.session.getMessages();
  }

  /** Appends messages to the session. */
  addMessages(messages: MessageData[]): void {
    this.session.addMessages(messages);
  }

  /** Overwrites the session messages. */
  setMessages(messages: MessageData[]): void {
    this.session.setMessages(messages);
  }

  /** Retrieves the custom state of the session. */
  getCustom(): State | undefined {
    return this.session.getCustom();
  }

  /** Updates the custom state using a mutator function. */
  updateCustom(fn: (custom?: State) => State): void {
    this.session.updateCustom(fn);
  }

  /** Retrieves the list of artifacts generated during the session. */
  getArtifacts(): Artifact[] {
    return this.session.getArtifacts();
  }

  /** Adds artifacts to the session, deduplicating by name. */
  addArtifacts(artifacts: Artifact[]): void {
    this.session.addArtifacts(artifacts);
  }

  /** Invokes the end-of-turn callback, absorbing errors from a closed stream. */
  private notifyEndTurn(
    snapshotId: string | undefined,
    finishReason?: AgentFinishReason
  ): void {
    try {
      this.onEndTurn?.(snapshotId, finishReason);
    } catch {
      // Stream was closed, absorb exception.
    }
  }

  /**
   * Executes the flow handler against incoming input messages sequentially.
   *
   * The handler receives the turn's {@link AgentInput} and a {@link TurnContext}
   * whose `snapshotId` is *reserved up front* - it is the id the snapshot
   * persisted at turn end will reuse. This lets a handler set up external,
   * snapshot-correlated state (e.g. a git branch/worktree named after the
   * snapshot) before generating, then commit it under that id.
   *
   * The handler may return a {@link TurnResult} carrying an explicit
   * `finishReason` for the just-completed turn. When omitted, no per-turn
   * reason is reported. Failures always report `failed`.
   */
  async run(
    fn: (input: AgentInput, ctx: TurnContext) => Promise<TurnResult | void>
  ): Promise<void> {
    for await (const input of this.inputCh) {
      if (input.message) {
        this.session.addMessages([input.message]);
      }

      // The first customPatch of every turn is a whole-document replace that
      // re-bases clients which may not share the server's baseline.
      this.firstCustomPatchInTurn = true;

      const parentSnapshotId = this.lastSnapshot?.snapshotId;

      // Reserve the turn's snapshotId up front (when a store is configured) so
      // the handler can name snapshot-correlated external resources before the
      // turn runs. The detach path may have already reserved one; reuse it.
      // The persisted snapshot at turn end reuses this id (maybeSnapshot
      // prefers `newSnapshotId`).
      if (this.store && !this.newSnapshotId) {
        this.newSnapshotId = reserveSnapshotId();
      }

      const turnSnapshotId = this.newSnapshotId;
      this.newSnapshotId = undefined;

      const turnContext: TurnContext = {
        snapshotId: turnSnapshotId!,
        parentSnapshotId,
        turnIndex: this.turnIndex,
      };

      try {
        await run(`runTurn-${this.turnIndex + 1}`, input, async () => {
          const turnResult = await fn(input, turnContext);
          const finishReason = turnResult?.finishReason;
          this.lastTurnFinishReason = finishReason;
          this.lastTurnError = undefined;

          const snapshotId = await this.maybeSnapshot(
            'completed',
            undefined,
            turnSnapshotId,
            finishReason
          );

          // Capture the state this successful turn produced. This becomes the
          // last-good state to fall back to if a later turn fails, and its
          // snapshotId is the last-good snapshot a failed turn resumes from.
          this.lastGoodState = this.session.getState();
          this.lastGoodSnapshotId = snapshotId;

          // Tag the turn span with the snapshotId this turn persisted under, so
          // a trace can correlate the turn with its snapshot (server-managed
          // agents only; client-managed turns have no snapshotId).
          if (snapshotId) {
            setCustomMetadataAttribute('agent:snapshotId', snapshotId);
          }

          this.notifyEndTurn(snapshotId, finishReason);

          // The turn span's output is the session state this turn produced -
          // applies to both client- and server-managed agents.
          return { state: this.session.getState() };
        });
        this.turnIndex++;
      } catch (e: any) {
        // An aborted turn rejects out of `generate` and lands here. Treat it as
        // `aborted` rather than `failed`: the abort path already persisted the
        // `aborted` status (the abort-aware mutator would skip a `failed` write
        // anyway), so we record the finish reason and skip the failed snapshot
        // write entirely instead of reporting a spurious error.
        if (this.abortSignal?.aborted) {
          this.lastTurnFinishReason = 'aborted';
          this.lastTurnError = undefined;
          this.notifyEndTurn(this.lastSnapshot?.snapshotId, 'aborted');
          break;
        }

        this.lastTurnFinishReason = 'failed';
        this.lastTurnError = toErrorDetails(e);
        const snapshotId = await this.maybeSnapshot(
          'failed',
          this.lastTurnError,
          turnSnapshotId,
          'failed'
        );
        this.notifyEndTurn(snapshotId, 'failed');

        // Graceful failure: rather than propagating the exception (which would
        // discard the action's final return - and with it the last-good state
        // and all prior successful turns), stop processing further inputs and
        // let the invocation resolve with `finishReason: 'failed'`. The caller
        // recovers the last-good state from the returned AgentOutput.
        break;
      }
    }
  }

  /**
   * Saves a snapshot of the current session state to the persistent store.
   *
   * When a store is configured every turn is persisted (snapshotting is no
   * longer opt-out). Uses the mutator-based `saveSnapshot` to atomically check
   * that the snapshot has not been concurrently aborted before writing -
   * preventing a race where a "done" write could overwrite a concurrent
   * "aborted" status.
   */
  async maybeSnapshot(
    status?: 'pending' | 'completed' | 'failed',
    error?: { status?: string; message: string; details?: any },
    snapshotId?: string,
    finishReason?: AgentFinishReason
  ): Promise<string | undefined> {
    if (
      !this.store ||
      (this.isDetached && snapshotId !== this.lastSnapshot?.snapshotId)
    )
      return this.lastSnapshot?.snapshotId;

    const currentVersion = this.session.getVersion();
    if (currentVersion === this.lastSnapshotVersion && !status) {
      return this.lastSnapshot?.snapshotId;
    }

    const currentState = this.session.getState();

    const snapshotInput: SessionSnapshotInput<State> = {
      ...(snapshotId || this.newSnapshotId
        ? { snapshotId: (snapshotId || this.newSnapshotId)! }
        : {}),
      // Stamp the session id onto every snapshot in the chain so callers can
      // resolve a snapshot's session without reaching into its state.
      sessionId: this.session.sessionId,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      state: currentState as SessionState<State>,
      parentId: this.lastSnapshot?.snapshotId,
      // Default to a resumable `completed` status. The only caller that omits a
      // status is the post-invocation write (which fires when the handler
      // mutates state after the last turn); persisting it as `completed` keeps
      // it a valid resume target under the "only `completed` is resumable" rule.
      status: status ?? 'completed',
      // Stamp an initial heartbeat on a `pending` (detached, in-flight)
      // snapshot. A background heartbeat loop refreshes it; if it goes stale the
      // snapshot is reported as `expired` on read (the worker is presumed dead).
      ...(status === 'pending' && { heartbeatAt: new Date().toISOString() }),
      ...(finishReason && { finishReason }),
      error,
    };

    const effectiveId = snapshotId || this.newSnapshotId;

    // Use the mutator-based saveSnapshot to atomically check the current
    // status before writing.  If the snapshot was concurrently aborted,
    // the mutator returns null and the write is skipped.
    const assignedId = await this.store.saveSnapshot(
      effectiveId,
      abortAwareMutator(snapshotInput),
      { context: getContext() }
    );
    if (assignedId === null) {
      // Snapshot was aborted concurrently; preserve the existing ID
      // without overwriting.
      return effectiveId;
    }

    this.lastSnapshot = { ...snapshotInput, snapshotId: assignedId };
    this.lastSnapshotVersion = currentVersion;

    return assignedId;
  }
}

/**
 * Projects an agent's server-side data onto the view a client should see.
 *
 * Every member is optional; an omitted member passes the corresponding data
 * through unchanged. Use this to redact sensitive fields or reshape data
 * before it leaves the server - covering both data at rest and data in flight:
 *
 * - `state` reshapes/redacts session state at rest. Applied to
 *   `AgentOutput.state` (client-managed agents), to snapshots returned by
 *   `getSnapshotData`, and as the baseline for streamed `customPatch` diffs
 *   (so streamed custom-state deltas stay consistent with the transformed
 *   full state). Note: `state.artifacts` is part of session state, so artifact
 *   redaction at rest happens here too.
 * - `chunk` reshapes/redacts each stream chunk in flight (`modelChunk`,
 *   `artifact`, `customPatch`, `turnEnd`) - e.g. filtering "internal" tool
 *   request/response parts out of model chunks, or redacting streamed
 *   artifacts. Return `null`/`undefined` to drop the chunk entirely.
 *
 * When both `state` and `chunk` touch the same data (e.g. artifacts), keeping
 * the two projections consistent is the author's responsibility.
 */
export interface ClientTransform<S = unknown> {
  /**
   * Reshapes/redacts session state before it is exposed to the client (at
   * rest: `AgentOutput.state`, snapshots, and the streamed `customPatch`
   * baseline).
   */
  state?: (state: SessionState<S>) => SessionState;
  /**
   * Reshapes/redacts each stream chunk before it is sent to the client.
   * Return `null`/`undefined` to drop the chunk entirely.
   */
  chunk?: (chunk: AgentStreamChunk) => AgentStreamChunk | null | undefined;
}

/**
 * Function handler definition for custom agent actions.
 */
export type AgentFn<State> = (
  sess: SessionRunner<State>,
  options: {
    sendChunk: (chunk: AgentStreamChunk) => void;
    abortSignal?: AbortSignal;
    context?: ActionContext;
  }
) => Promise<AgentResult>;

/**
 * Lookup input for the `getSnapshotData` action / method.
 *
 * Mirrors {@link GetSnapshotOptions}: provide exactly one of `snapshotId`
 * (an exact snapshot) or `sessionId` (the session's latest leaf snapshot).
 */
export const GetSnapshotDataInputSchema = z.object({
  snapshotId: z.string().optional(),
  sessionId: z.string().optional(),
});

/**
 * Lookup input for `getSnapshotData`.
 */
export interface GetSnapshotDataInput {
  snapshotId?: string;
  sessionId?: string;
  context?: ActionContext;
}

export type GetSnapshotDataAction<S = unknown> = Action<
  typeof GetSnapshotDataInputSchema,
  z.ZodType<SessionSnapshot<S>>
>;

/**
 * Represents a configured, registered Agent.
 *
 * An `Agent` exposes two surfaces:
 *
 * 1. The ergonomic, transport-agnostic {@link AgentAPI} (`chat`, `loadChat`,
 *    `getSnapshot`, `abort`) - the same surface returned by `remoteAgent` on
 *    the client, so server- and client-side code share one interface.
 * 2. The lower-level {@link BidiAction} surface (`run`, `streamBidi`, …) for
 *    advanced use and for serving over HTTP.
 */
export interface Agent<State = unknown>
  extends BidiAction<
      typeof AgentInputSchema,
      typeof AgentOutputSchema,
      typeof AgentStreamChunkSchema,
      typeof AgentInitSchema
    >,
    AgentAPI<State> {
  getSnapshotData(
    opts: GetSnapshotDataInput
  ): Promise<SessionSnapshot<State> | undefined>;

  abort(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot['status'] | undefined>;

  readonly getSnapshotDataAction: GetSnapshotDataAction<State>;
  readonly abortAgentAction: Action<
    typeof AgentAbortRequestSchema,
    typeof AgentAbortResponseSchema
  >;
}

/**
 * Error thrown for agent init *API misuse* that should surface to the caller as
 * a real, thrown error (mapped to an HTTP status by the server handler) rather
 * than being absorbed into a graceful `finishReason: 'failed'` result.
 *
 * Covers calling an agent with an init that does not match its state-management
 * mode (e.g. sending `state` to a server-managed agent, or `snapshotId`/
 * `sessionId` to a client-managed one) and the snapshot/session ownership
 * guard. Other pre-turn failures (missing snapshot, non-resumable snapshot,
 * invalid custom state) remain graceful.
 */
export class AgentInitError extends GenkitError {}

/**
 * Asserts that the init strategy matches the agent's state-management mode,
 * throwing an {@link AgentInitError} on a mismatch.
 *
 * Server-managed agents (with a store) resume via a `snapshotId` / `sessionId`;
 * client-managed agents (no store) supply the full `state` blob. This is API
 * misuse, so it propagates as a thrown error rather than a graceful failure.
 */
function assertInitMatchesStateManagement(
  config: { name: string; store?: SessionStore<unknown> },
  init: AgentInit | undefined
): void {
  if ((init?.snapshotId || init?.sessionId) && !config.store) {
    throw new AgentInitError({
      status: 'FAILED_PRECONDITION',
      message:
        `Cannot use '${init.snapshotId ? 'snapshotId' : 'sessionId'}' with ` +
        `agent '${config.name}': this agent has no store configured ` +
        `(client-managed state). Send 'state' instead.`,
    });
  }
  if (init?.state && config.store) {
    throw new AgentInitError({
      status: 'FAILED_PRECONDITION',
      message:
        `Cannot send 'state' to agent '${config.name}': this agent uses ` +
        `a server-managed store. Send 'snapshotId' or 'sessionId' instead.`,
    });
  }
}

/**
 * Resolves the {@link Session} (and originating snapshot, if any) for an agent
 * turn from its {@link AgentInit}.
 *
 * Server-managed agents (with a store) resume via a `snapshotId` (an exact
 * snapshot) or a `sessionId` (the session's latest snapshot); client-managed
 * agents (no store) supply the full `state` blob. Throws a {@link GenkitError}
 * on a missing snapshot, non-resumable snapshot, or invalid custom state - the
 * caller is expected to translate that into a graceful `finishReason: 'failed'`
 * result. The state-management mismatch checks are performed up front by
 * {@link assertInitMatchesStateManagement} and throw {@link AgentInitError}.
 */
async function resolveSession<State>(
  config: { name: string; store?: SessionStore<State> },
  store: SessionStore<State>,
  init: AgentInit | undefined,
  validateCustomState: (custom: unknown) => void
): Promise<{ session: Session<State>; snapshot?: SessionSnapshot<State> }> {
  if (init?.snapshotId) {
    const snapshot = await store.getSnapshot({
      snapshotId: init.snapshotId,
      context: getContext(),
    });
    if (!snapshot) {
      throw new GenkitError({
        status: 'NOT_FOUND',
        message: `Snapshot ${init.snapshotId} not found`,
      });
    }
    // When both `snapshotId` and `sessionId` are supplied, `snapshotId` selects
    // the exact snapshot to resume and `sessionId` acts as an ownership guard:
    // the snapshot must belong to that session. A mismatch is API misuse, so it
    // propagates as a thrown error (AgentInitError) rather than being absorbed
    // into a graceful failure.
    // Prefer the snapshot's top-level `sessionId`; fall back to the id carried
    // in its state for rows written before snapshot-level ids existed.
    const snapshotSessionId = snapshot.sessionId ?? snapshot.state?.sessionId;
    if (init.sessionId && snapshotSessionId !== init.sessionId) {
      throw new AgentInitError({
        status: 'INVALID_ARGUMENT',
        message:
          `Snapshot ${init.snapshotId} does not belong to session ` +
          `${init.sessionId} (it belongs to ` +
          `${snapshotSessionId ?? 'an unknown session'}).`,
      });
    }

    // Only `completed` snapshots are resumable. A failed/aborted/pending
    // snapshot is persisted for inspection but is not a valid resume target.
    if (snapshot.status !== 'completed') {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          `Snapshot ${init.snapshotId} is not resumable (status: ` +
          `${snapshot.status ?? 'unknown'}). Only 'completed' snapshots can ` +
          `be resumed.`,
      });
    }

    validateCustomState(snapshot.state?.custom);
    return {
      snapshot,
      session: new Session<State>(snapshot.state as SessionState<State>),
    };
  }

  if (init?.sessionId) {
    // Resume the session's latest snapshot. The store returns the latest leaf
    // regardless of status, but only `completed` snapshots are resumable - so
    // if the leaf is a non-resumable turn (e.g. a `failed`/`aborted`/`pending`
    // turn) walk back over its parent chain to the last-good (`completed`)
    // snapshot. When the session has no resumable snapshot (e.g. a first-turn
    // failure) seed a fresh session bound to the requested sessionId so
    // subsequent turns can find it.
    let snapshot = await store.getSnapshot({
      sessionId: init.sessionId,
      context: getContext(),
    });
    // Walk back over non-resumable leaves to the last-good (`completed`)
    // snapshot. Guard against a self-referential or cyclic `parentId` chain
    // (corrupt history) with a visited set so we fail fast with
    // `FAILED_PRECONDITION` instead of looping forever on store reads.
    const visited = new Set<string>();
    while (snapshot && snapshot.status !== 'completed') {
      if (visited.has(snapshot.snapshotId)) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message:
            `Session '${init.sessionId}' has a cyclic snapshot parent chain ` +
            `(snapshot '${snapshot.snapshotId}' was visited twice). Resume by ` +
            `snapshotId instead.`,
        });
      }
      visited.add(snapshot.snapshotId);
      snapshot = snapshot.parentId
        ? await store.getSnapshot({
            snapshotId: snapshot.parentId,
            context: getContext(),
          })
        : undefined;
    }
    if (snapshot) {
      validateCustomState(snapshot.state?.custom);
      return {
        snapshot,
        session: new Session<State>(snapshot.state as SessionState<State>),
      };
    }
    return {
      session: new Session<State>({
        custom: undefined,
        artifacts: [],
        messages: [],
        sessionId: init.sessionId,
      }),
    };
  }

  if (init?.state && !config.store) {
    validateCustomState(init.state.custom);
    return {
      session: new Session<State>(init.state as SessionState<State>),
    };
  }

  return {
    session: new Session<State>({
      custom: undefined,
      artifacts: [],
      messages: [],
    }),
  };
}

/**
 * Pumps the action's raw input stream into the runner's input channel while
 * intercepting `detach: true` directives.
 *
 * Running this proxy concurrently lets a detach directive take effect
 * immediately rather than waiting for the runner to drain a backlog of
 * pre-queued inputs. A detach-only message (no payload) is consumed here and
 * not forwarded, since it has no turn to process.
 */
function pipeInputWithDetach<State>(
  inputStream: AsyncIterable<AgentInput>,
  target: Channel<AgentInput>,
  getRunner: () => SessionRunner<State>,
  storeEnabled: boolean,
  rejectDetach: (reason: any) => void
): void {
  (async () => {
    try {
      for await (const input of inputStream) {
        if (input.detach) {
          if (!storeEnabled) {
            rejectDetach(
              new GenkitError({
                status: 'FAILED_PRECONDITION',
                message:
                  'Detach is only supported when a session store is provided.',
              })
            );
          } else {
            const runner = getRunner();
            // Reserve the in-flight snapshot's id up front so the detached
            // snapshot and any handler-named external resources share one id.
            const turnSnapshotId = runner.newSnapshotId || reserveSnapshotId();
            runner.newSnapshotId = turnSnapshotId;
            await runner.maybeSnapshot('pending', undefined, turnSnapshotId);
            runner.isDetached = true;

            if (runner.onDetach) {
              runner.onDetach(turnSnapshotId);
            }
          }
          // Only forward to the runner if the input carries a payload beyond
          // the detach directive; a detach-only message has no turn to process.
          const hasPayload = !!(
            input.message ||
            input.resume?.restart?.length ||
            input.resume?.respond?.length
          );
          if (hasPayload) {
            target.send(input);
          }
        } else {
          target.send(input);
        }
      }
      target.close();
    } catch (e) {
      target.error(e);
    }
  })();
}

/**
 * Registers a multi-turn custom agent action capable of maintaining persistent state.
 *
 * When `stateSchema` is provided the custom state is validated at load time
 * (from a snapshot store or from the client-supplied `init.state`) and the
 * JSON Schema representation is included in the action metadata so that
 * tooling (e.g. the Dev UI) can inspect / validate the state shape.
 */
export function defineCustomAgent<State = unknown>(
  registry: Registry,
  config: {
    name: string;
    description?: string;
    stateSchema?: z.ZodType<State>;
    store?: SessionStore<State>;
    clientTransform?: ClientTransform<State>;
  },
  fn: AgentFn<State>
): Agent<State> {
  // Helper that applies the optional state transform before exposing state to
  // the client.  When no transform is configured it returns the raw state.
  const toClientState = (
    state: SessionState<State>
  ): SessionState | undefined => {
    if (config.clientTransform?.state) {
      return config.clientTransform.state(state);
    }
    return state as SessionState;
  };

  // If a state schema was provided, pre-compute the JSON schema once so it
  // can be embedded in metadata and reused for validation.

  const stateJsonSchema = config.stateSchema
    ? toJsonSchema({ schema: config.stateSchema })
    : undefined;

  /**
   * Validates the `custom` field of a session state against the configured
   * `stateSchema`.  No-ops when no schema was provided.
   */
  const validateCustomState = (custom: unknown): void => {
    if (config.stateSchema && custom !== undefined) {
      parseSchema(custom, { schema: config.stateSchema });
    }
  };

  const primaryAction = defineBidiAction(
    registry,
    {
      name: config.name,
      description: config.description,
      actionType: 'agent',
      inputSchema: AgentInputSchema,
      outputSchema: AgentOutputSchema,
      streamSchema: AgentStreamChunkSchema,
      initSchema: AgentInitSchema,
      metadata: {
        agent: {
          stateManagement: config.store ? 'server' : 'client',
          abortable: !!config.store?.onSnapshotStateChange,
          ...(stateJsonSchema && { stateSchema: stateJsonSchema }),
        },
      },
    },
    async function* (
      arg: ActionFnArg<AgentStreamChunk, AgentInput, AgentInit>
    ) {
      const init = arg.init;
      const store = config.store || new InMemorySessionStore<State>();

      // API-misuse checks (init does not match the agent's state-management
      // mode) throw out of the generator so the server handler maps them to a
      // proper HTTP status, rather than being absorbed into a graceful
      // `finishReason: 'failed'` result below.
      assertInitMatchesStateManagement(config, init);

      let session!: Session<State>;
      let snapshot: SessionSnapshot<State> | undefined;

      try {
        ({ session, snapshot } = await resolveSession<State>(
          config,
          store,
          init,
          validateCustomState
        ));
      } catch (e: any) {
        // An AgentInitError signals API misuse (e.g. the snapshot/session
        // ownership guard) that must surface as a thrown error; re-throw it so
        // the server handler maps it to a proper HTTP status.
        if (e instanceof AgentInitError) {
          throw e;
        }
        // Other pre-turn / setup failures (missing snapshot, non-resumable
        // snapshot, invalid client state). Resolve gracefully with
        // `finishReason: 'failed'` - preserving the original `error.status` -
        // rather than throwing, so the caller gets a structured, inspectable
        // result. There is no last-good turn yet; echo back the
        // client-supplied state when present.
        return {
          finishReason: 'failed' as AgentFinishReason,
          error: toErrorDetails(e),
          ...(!config.store &&
            init?.state && { state: init.state as SessionState }),
        };
      }

      // Tag the current trace span with the sessionId so that traces
      // belonging to the same agent conversation can be correlated.
      setCustomMetadataAttributes({
        'agent:sessionId': session.sessionId,
      });

      let detachedSnapshotId: string | undefined;
      let resolveDetach:
        | ((value: void | PromiseLike<void>) => void)
        | undefined;
      let rejectDetach: ((reason: any) => void) | undefined;
      const detachPromise = new Promise<void>((resolve, reject) => {
        resolveDetach = resolve;
        rejectDetach = reject;
      });

      const abortController = new AbortController();
      let unsubscribe: any = undefined;
      // Background heartbeat timer for the detached snapshot. Started in
      // `onDetach`, cleared when the flow settles (or on abort).
      let heartbeatTimer: ReturnType<typeof setInterval> | undefined;
      const stopHeartbeat = () => {
        if (heartbeatTimer) {
          clearInterval(heartbeatTimer);
          heartbeatTimer = undefined;
        }
      };

      let runner!: SessionRunner<State>;

      // Centralized chunk emitter: every stream chunk passes through here so
      // the optional `clientTransform.chunk` can reshape/redact it (or drop it
      // by returning a nullish value) before it reaches the client.
      //
      // The actual dispatch is failure-isolated (like `notifyEndTurn`): the
      // artifact/customPatch emitters fire synchronously from inside
      // `Session.updateCustom`/`addArtifacts` (i.e. from the user's handler).
      // If the client stream is already closed, `sendChunk` throws; absorbing
      // it here prevents that from propagating out of the handler and turning
      // a normal turn into a `failed` one.
      const emitChunk = (chunk: AgentStreamChunk) => {
        try {
          let toSend: AgentStreamChunk | null | undefined = chunk;
          if (config.clientTransform?.chunk) {
            toSend = config.clientTransform.chunk(chunk);
          }
          if (!toSend) return;
          arg.sendChunk(toSend);
        } catch {
          // Stream was closed (or the transform threw); absorb the exception.
        }
      };

      // We construct an asynchronous proxy channel over the inputStream.
      // This enables immediate interception of `detach: true` directives. Without this proxy,
      // a backlog of pre-queued inputs would have to be resolved sequentially by the runner first.
      const runnerInputChannel = new Channel<AgentInput>();

      pipeInputWithDetach(
        arg.inputStream,
        runnerInputChannel,
        () => runner,
        !!config.store,
        (reason) => rejectDetach?.(reason)
      );

      runner = new SessionRunner<State>(session, runnerInputChannel, {
        store,
        lastSnapshot: snapshot,
        abortSignal: abortController.signal,

        onDetach: (snapshotId) => {
          detachedSnapshotId = snapshotId;
          if (resolveDetach) {
            resolveDetach();
          }

          // Refresh the detached snapshot's heartbeat periodically. The mutator
          // only touches a still-`pending` snapshot (returns null otherwise) so
          // it never resurrects a terminal snapshot or clobbers a concurrent
          // abort. If a read sees this heartbeat go stale, the snapshot is
          // reported as `expired` (the worker is presumed dead). `unref` so the
          // timer never keeps the process alive on its own.
          const ctx = getContext();
          heartbeatTimer = setInterval(() => {
            void store
              .saveSnapshot(
                snapshotId,
                (current) =>
                  current?.status === 'pending'
                    ? { ...current, heartbeatAt: new Date().toISOString() }
                    : null,
                { context: ctx }
              )
              .catch(() => {
                // Best-effort heartbeat; ignore transient store errors.
              });
          }, DEFAULT_HEARTBEAT_INTERVAL_MS);
          heartbeatTimer.unref?.();

          if (store.onSnapshotStateChange) {
            unsubscribe = store.onSnapshotStateChange(
              snapshotId,
              (snap) => {
                if (snap.status === 'aborted') {
                  stopHeartbeat();
                  abortController.abort();
                  if (unsubscribe) unsubscribe();
                }
              },
              { context: getContext() }
            );
          }
        },

        onEndTurn: (snapshotId, finishReason) => {
          if (!runner.isDetached) {
            emitChunk({
              turnEnd: {
                ...(config.store && { snapshotId }),
                ...(finishReason && { finishReason }),
              },
            });
          }
        },
      });

      const sendArtifactChunk = (a: Artifact) => {
        if (!runner.isDetached) {
          emitChunk({ artifact: a });
        }
      };

      session.on('artifactAdded', sendArtifactChunk);
      session.on('artifactUpdated', sendArtifactChunk);

      // Auto-emit a `customPatch` chunk whenever custom state is mutated.
      // The diff is computed AFTER the clientStateTransform so streamed deltas
      // honor redaction and stay consistent with the transformed full state in
      // snapshots / final output. The first patch of every turn is a
      // whole-document replace (re-basing clients that may lack the baseline);
      // subsequent patches are incremental diffs against the last sent value.
      let lastSentCustom: unknown;
      const sendCustomPatch = () => {
        if (runner.isDetached) return;
        const transformed = toClientState(session.getState())?.custom;
        let patch: JsonPatch;
        if (runner.firstCustomPatchInTurn) {
          patch = [
            { op: 'replace', path: '', value: structuredClone(transformed) },
          ];
          runner.firstCustomPatchInTurn = false;
        } else {
          patch = diff(lastSentCustom, transformed);
        }
        lastSentCustom = structuredClone(transformed);
        if (patch.length) {
          emitChunk({ customPatch: patch });
        }
      };
      session.on('customChanged', sendCustomPatch);

      const sendChunk = (chunk: AgentStreamChunk) => {
        if (!runner.isDetached) {
          emitChunk(chunk);
        }
      };

      const flowPromise = (async () => {
        try {
          const result = await runWithSession(registry, session, () =>
            fn(runner, {
              sendChunk,
              abortSignal: abortController.signal,
              context: getContext(),
            })
          );
          // After the handler resolves, persist any state it mutated after the
          // last turn. Omitting a status defaults to a resumable `completed`
          // write, which the version guard skips when nothing changed.
          const finalSnapshotId = await runner.maybeSnapshot();
          return { result, finalSnapshotId };
        } finally {
          // The turn has settled (the snapshot reached a terminal status), so
          // stop refreshing its heartbeat.
          stopHeartbeat();
          if (unsubscribe) unsubscribe();
          session.off('artifactAdded', sendArtifactChunk);
          session.off('artifactUpdated', sendArtifactChunk);
          session.off('customChanged', sendCustomPatch);
        }
      })();

      // We race the background flow execution against the detach signal.
      // If detachment is requested, we yield output metadata early, but allow
      // the flow handler promise to continue its asynchronous completion.
      const outcome = await Promise.race([
        flowPromise,
        detachPromise.then(() => 'detached' as const),
      ]);

      if (outcome === 'detached') {
        return {
          sessionId: session.sessionId,
          snapshotId: detachedSnapshotId!,
          finishReason: 'detached' as AgentFinishReason,
          ...(!config.store && { state: toClientState(session.getState()) }),
        };
      }

      const { result, finalSnapshotId } = outcome;

      // A turn failed: resolve gracefully with `finishReason: 'failed'` and the
      // last-good state (what the failed turn started with), rather than the
      // live state which may hold the failed turn's partial mutations.
      if (runner.lastTurnFinishReason === 'failed' && runner.lastTurnError) {
        const lastGood = (runner.lastGoodState ??
          session.getState()) as SessionState<State>;
        const lastGoodMessages = lastGood.messages;
        return {
          sessionId: session.sessionId,
          finishReason: 'failed' as AgentFinishReason,
          error: runner.lastTurnError,

          ...(result.artifacts?.length && { artifacts: result.artifacts }),
          ...(lastGoodMessages?.length && {
            message: lastGoodMessages[lastGoodMessages.length - 1],
          }),
          // Server-managed: the last successful turn is already persisted (every
          // turn is snapshotted), so point at its snapshot. The failed turn's
          // own snapshot is persisted too but is not resumable - `sessionId`
          // resume skips it back to this last-good `done` snapshot. Undefined on
          // a first-turn failure (no successful turn yet; client holds the seed).
          ...(config.store && { snapshotId: runner.lastGoodSnapshotId }),
          // Client-managed: return the last-good state directly.
          ...(!config.store && { state: toClientState(lastGood) }),
        };
      }

      const finishReason = result.finishReason ?? runner.lastTurnFinishReason;

      return {
        sessionId: session.sessionId,
        ...(result.artifacts?.length && { artifacts: result.artifacts }),
        ...(result.message && { message: result.message }),
        ...(finishReason && { finishReason }),
        ...(config.store && { snapshotId: finalSnapshotId }),
        ...(!config.store && { state: toClientState(session.getState()) }),
      };
    }
  );

  // Helper that applies the clientTransform.state projection to a snapshot's
  // state, returning a new snapshot object with the transformed state.
  const toClientSnapshot = (
    snapshot: SessionSnapshot<State>
  ): SessionSnapshot => {
    if (!config.clientTransform?.state || !snapshot.state) {
      return snapshot as SessionSnapshot;
    }
    return {
      ...snapshot,
      state: config.clientTransform.state(snapshot.state),
    };
  };

  // Shared snapshot/abort implementations, reused by both the `defineAction`
  // surfaces (which inject the ambient request context) and the ergonomic
  // composite methods (which accept caller-supplied options).
  const resolveSnapshot = async (
    lookup: GetSnapshotDataInput
  ): Promise<SessionSnapshot | undefined> => {
    requireStore(config.store, 'getSnapshotData', config.name);
    const snapshot = await config.store.getSnapshot(lookup);
    if (!snapshot) return undefined;
    // Compute `expired` on read: a `pending` snapshot whose heartbeat has gone
    // stale is presumed orphaned (its background worker died), so surface it as
    // `expired` rather than leaving it `pending` forever. This is read-only -
    // the status is not written back to the store.
    const effective = isHeartbeatExpired(snapshot)
      ? { ...snapshot, status: 'expired' as const }
      : snapshot;
    return toClientSnapshot(effective);
  };

  const runAbort = (
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot['status'] | undefined> => {
    requireStore(config.store, 'abort', config.name);
    return abortSnapshotInStore(config.store, snapshotId, options);
  };

  const getSnapshotDataAction = defineAction(
    registry,
    {
      name: config.name,
      description: `Gets snapshot data for ${config.name} by snapshotId or sessionId`,
      actionType: 'agent-snapshot',
      inputSchema: GetSnapshotRequestSchema,
      outputSchema: SessionSnapshotSchema,
    },
    async (lookup) => {
      const snap = await resolveSnapshot({ ...lookup, context: getContext() });
      if (!snap) {
        const target = lookup.snapshotId || lookup.sessionId || 'unknown';
        throw new GenkitError({
          status: 'NOT_FOUND',
          message: `Snapshot '${target}' not found for agent '${config.name}'.`,
        });
      }
      return snap;
    }
  );

  const abortAgentAction = defineAction(
    registry,
    {
      name: config.name,
      description: `Aborts ${config.name} agent by snapshotId. Returns the snapshot id and its status after the abort attempt.`,
      actionType: 'agent-abort',
      inputSchema: AgentAbortRequestSchema,
      outputSchema: AgentAbortResponseSchema,
    },
    async ({ snapshotId }) => {
      const status = await runAbort(snapshotId, { context: getContext() });
      return { snapshotId, status };
    }
  );

  const composite = Object.assign(primaryAction, {
    getSnapshotData: (opts: GetSnapshotDataInput) => resolveSnapshot(opts),
    abort: (snapshotId: string, options?: SessionStoreOptions) =>
      runAbort(snapshotId, options),
    getSnapshotDataAction:
      getSnapshotDataAction as unknown as GetSnapshotDataAction<State>,
    abortAgentAction: abortAgentAction as unknown as Action<
      typeof AgentAbortRequestSchema,
      typeof AgentAbortResponseSchema
    >,
  });

  // Opens a single-turn bidi stream: send the input, close the send side, and
  // hand back the live `{ stream, output }` handle.
  const startBidi = (
    input: AgentInput,
    init: AgentInit,
    opts: { abortSignal: AbortSignal }
  ) => {
    const bidi = primaryAction.streamBidi(init, {
      abortSignal: opts.abortSignal,
    });
    bidi.send(input);
    bidi.close();
    return bidi;
  };

  // In-process transport: drives the agent action directly (no HTTP). This lets
  // the server-side agent expose the same ergonomic AgentAPI (`chat`,
  // `loadChat`, `getSnapshot`, `abort`) as the HTTP `remoteAgent` client.
  const transport: AgentTransport = {
    stateManagement: config.store ? 'server' : 'client',

    runTurn(input, init, opts) {
      const bidi = startBidi(input, init, opts);
      return { stream: bidi.stream, output: bidi.output };
    },

    async getSnapshot(lookup: SnapshotLookup) {
      return composite.getSnapshotData(lookup);
    },

    abort(snapshotId: string) {
      return composite.abort(snapshotId);
    },
  };

  const agentApi = createAgentAPI<State>(transport);

  // Expose the AgentAPI surface on the composite. `abort`/`getSnapshotData`
  // already exist on the composite (richer signatures); we add `chat`,
  // `loadChat`, and `getSnapshot`.
  Object.assign(composite, {
    chat: agentApi.chat,
    loadChat: agentApi.loadChat,
    getSnapshot: agentApi.getSnapshot,
  });

  return composite as unknown as Agent<State>;
}

/**
 * Registers an agent from an existing PromptAction.
 *
 * The `promptInput` option supplies values for the referenced prompt's input
 * variables, so a single prompt can be reused and customized by multiple
 * agents. Provide the prompt's input schema as the `I` type parameter to get
 * a type-checked `promptInput`.
 */
export function definePromptAgent<
  State = unknown,
  I extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  config: {
    promptName: string;
    /** Human-readable description, surfaced on the agent action's metadata. */
    description?: string;
    /**
     * Input values for the referenced prompt's input variables. Lets a single
     * prompt be reused/customized across multiple agents (e.g. supplying a
     * different `role` or `tone` to a shared dotprompt template).
     */
    promptInput?: z.infer<I>;
    stateSchema?: z.ZodType<State>;
    store?: SessionStore<State>;
    clientTransform?: ClientTransform<State>;
  }
) {
  let cachedPromptAction: PromptAction | undefined;

  const fn: AgentFn<State> = async (sess, { sendChunk, abortSignal }) => {
    await sess.run(async (input) => {
      const promptInput = config.promptInput ?? {};

      if (!cachedPromptAction) {
        cachedPromptAction = (await registry.lookupAction(
          `/prompt/${config.promptName}`
        )) as PromptAction;
        if (!cachedPromptAction) {
          throw new Error(
            `Prompt '${config.promptName}' not found. Ensure it is defined before the agent is invoked.`
          );
        }
      }

      const historyTag = '_genkit_history';
      const promptTag = 'agentPreamble';

      // Tag every history message so we can identify them after render.
      const history = (sess.getMessages() || []).map((m) => ({
        ...m,
        metadata: { ...m.metadata, [historyTag]: true },
      }));

      // Let the prompt control where history is placed (e.g. dotprompt
      // {{history}}).  When the prompt has no explicit `messages` config
      // the render helper simply appends history after system/user.
      const genOpts = await cachedPromptAction.__executablePrompt.render(
        promptInput as unknown as z.ZodTypeAny,
        { messages: history }
      );

      // After render: tag everything that is NOT history as a prompt
      // message so we can strip it after generation.  Also strip the
      // internal history tag - it is an implementation detail that
      // should not leak to the model.
      if (genOpts.messages) {
        genOpts.messages = genOpts.messages.map((m) => {
          if (m.metadata?.[historyTag]) {
            // Strip the history tag before sending to the model.
            const { [historyTag]: _, ...restMeta } = m.metadata!;
            return {
              ...m,
              metadata: Object.keys(restMeta).length ? restMeta : undefined,
            };
          }
          return { ...m, metadata: { ...m.metadata, [promptTag]: true } };
        });
      }

      if (input.resume) {
        // Safety: validate that every restart/respond entry references
        // a tool request that actually exists in the session history.
        // For restarts, also verify that the input has not been tampered with.
        validateResumeAgainstHistory(input.resume, sess.getMessages());

        genOpts.resume = {
          ...(input.resume.restart?.length && {
            restart: input.resume.restart as ToolRequestPart[],
          }),
          ...(input.resume.respond?.length && {
            respond: input.resume.respond as ToolResponsePart[],
          }),
        };
      }

      const result = generateStream(registry, { ...genOpts, abortSignal });

      for await (const chunk of result.stream) {
        sendChunk({ modelChunk: chunk });
      }

      const res = await result.response;

      // Keep everything that is NOT a prompt-template message:
      //   • history messages (clean - history tag was stripped before generate)
      //   • new messages from tool loops (untagged)
      //   • model response
      if (res.request?.messages) {
        const msgs = res.request.messages.filter(
          (m) => !m.metadata?.[promptTag]
        );
        if (res.message) {
          msgs.push(res.message);
        }
        sess.setMessages(msgs);
      } else if (res.message) {
        sess.addMessages([res.message]);
      }

      if (res.finishReason === 'interrupted') {
        const parts =
          res.message?.content?.filter((p) => !!p.toolRequest) || [];
        if (parts.length > 0) {
          sendChunk({
            modelChunk: {
              role: 'tool',
              content: parts,
            },
          });
        }
      }

      // Surface the generate finish reason as the turn's finish reason. The
      // generate `FinishReason` enum is a subset of `AgentFinishReason`, so it
      // maps through directly.
      return { finishReason: res.finishReason as AgentFinishReason };
    });

    const msgs = sess.getMessages();
    return {
      artifacts: sess.getArtifacts(),
      message: msgs.length > 0 ? msgs[msgs.length - 1] : undefined,
      ...(sess.lastTurnFinishReason && {
        finishReason: sess.lastTurnFinishReason,
      }),
    };
  };

  return defineCustomAgent<State>(
    registry,
    {
      name: config.promptName,
      description: config.description,
      stateSchema: config.stateSchema,
      store: config.store,
      clientTransform: config.clientTransform,
    },
    fn
  );
}

// ---------------------------------------------------------------------------
// Resume validation - ensure restart/respond entries match session history
// ---------------------------------------------------------------------------

/**
 * Validates that every `resume.restart` and `resume.respond` entry references
 * a tool request that actually exists in the session history.
 *
 * For **restart** entries, also validates that the `input` has not been modified
 * compared to the original tool request - preventing a malicious client from
 * forging tool inputs.
 *
 * For **respond** entries, validates that a matching tool request (by name + ref)
 * exists in history.
 *
 * Searches the **entire history** (all model messages), not just the last one.
 */
export function validateResumeAgainstHistory(
  resume: {
    restart?: Array<{
      toolRequest: { name: string; ref?: string; input?: unknown };
      metadata?: Record<string, unknown>;
    }>;
    respond?: Array<{
      toolResponse: { name: string; ref?: string; output?: unknown };
    }>;
  },
  history: MessageData[]
): void {
  // Collect all tool requests from all model messages in the stored history.
  const allToolRequests: Array<{
    name: string;
    ref?: string;
    input?: unknown;
  }> = [];
  for (const msg of history) {
    if (msg.role === 'model') {
      for (const part of msg.content) {
        if (part.toolRequest) {
          allToolRequests.push(part.toolRequest);
        }
      }
    }
  }

  // Validate restart entries: name + ref must exist AND input must match exactly
  for (const restart of resume.restart || []) {
    const { name, ref, input } = restart.toolRequest;
    const match = allToolRequests.find(
      (tr) => tr.name === name && tr.ref === ref
    );
    if (!match) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          `resume.restart references tool '${name}'` +
          (ref ? ` (ref: ${ref})` : '') +
          ` which was not found in session history.`,
      });
    }
    if (!deepEqual(input, match.input)) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          `resume.restart for tool '${name}'` +
          (ref ? ` (ref: ${ref})` : '') +
          ` has modified inputs that do not match the original tool request ` +
          `in session history. Restart inputs must exactly match the ` +
          `interrupted tool request.`,
      });
    }
  }

  // Validate respond entries: name + ref must match a tool request in history
  for (const respond of resume.respond || []) {
    const { name, ref } = respond.toolResponse;
    const match = allToolRequests.find(
      (tr) => tr.name === name && tr.ref === ref
    );
    if (!match) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message:
          `resume.respond references tool '${name}'` +
          (ref ? ` (ref: ${ref})` : '') +
          ` which was not found in session history.`,
      });
    }
  }
}

// ---------------------------------------------------------------------------
// defineAgent - shortcut that combines definePrompt + definePromptAgent
// ---------------------------------------------------------------------------

/**
 * Configuration for `defineAgent`, which combines prompt definition and agent
 * registration into a single call.
 */
export interface AgentConfig<
  State = unknown,
  I extends z.ZodTypeAny = z.ZodTypeAny,
> extends PromptConfig<I> {
  /**
   * Optional Zod schema describing the shape of the custom session state.
   *
   * When provided:
   * - The `State` type is inferred from the schema (no explicit generic needed).
   * - The JSON Schema is included in action metadata (`metadata.agent.stateSchema`)
   *   so the Dev UI and other tooling can inspect / validate the state.
   * - Custom state is validated at load time (from a snapshot store or from the
   *   client-supplied `init.state`).
   */
  stateSchema?: z.ZodType<State>;
  store?: SessionStore<State>;
  clientTransform?: ClientTransform<State>;
  /**
   * Input values for the prompt's input variables. Lets the same prompt
   * definition power differently-customized agents (e.g. supplying a different
   * `role` or `tone`). Type-checked against the prompt's `input.schema`.
   */
  promptInput?: z.infer<I>;
}

/**
 * Defines and registers an agent by creating a prompt and wiring it into a
 * multi-turn agent in one step.
 *
 * This is a convenience shortcut for:
 * ```ts
 * definePrompt(registry, promptConfig);
 * definePromptAgent(registry, { promptName: promptConfig.name, ... });
 * ```
 */
export function defineAgent<
  State = unknown,
  I extends z.ZodTypeAny = z.ZodTypeAny,
>(registry: Registry, config: AgentConfig<State, I>): Agent<State> {
  // Extract agent-specific fields from the combined config; the rest is
  // forwarded to definePrompt.
  const { stateSchema, store, clientTransform, promptInput, ...promptConfig } =
    config;

  // Register the prompt.
  definePrompt(registry, promptConfig);

  // Wire it into a prompt agent.
  return definePromptAgent<State, I>(registry, {
    promptName: promptConfig.name,
    description: promptConfig.description,
    promptInput,
    stateSchema,
    store,
    clientTransform,
  });
}
