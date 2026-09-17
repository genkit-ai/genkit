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
  StatusNameSchema,
  deepEqual,
  defineAction,
  defineBidiAction,
  getContext,
  getErrorMessage,
  run,
  z,
  type Action,
  type ActionContext,
  type ActionFnArg,
  type BidiAction,
} from '@genkit-ai/core';
import { Channel } from '@genkit-ai/core/async';
import { logger } from '@genkit-ai/core/logging';
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
import {
  GenerateResponse,
  GenerationAbortedError,
  GenerationResponseError,
  generateStream,
} from './generate.js';
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
 * Reports whether an aborted snapshot carrying no state is still waiting for
 * the write that stamps one on. The abort flips the pending row's status and
 * leaves its heartbeat where the worker left it; only the finalize writes the
 * state, and it clears the heartbeat as it lands. A beat inside `timeoutMs`
 * therefore says a live worker is between the two writes, and a stale or
 * absent one says it died there.
 */
function finalizeInFlight(
  snapshot: SessionSnapshot,
  timeoutMs: number = DEFAULT_HEARTBEAT_TIMEOUT_MS
): boolean {
  if (!snapshot.heartbeatAt) {
    return false;
  }
  const last = Date.parse(snapshot.heartbeatAt);
  if (Number.isNaN(last)) {
    return false;
  }
  return Date.now() - last <= timeoutMs;
}

/**
 * Result returned by a single turn handler passed to {@link SessionRunner.run}.
 *
 * Returning a `finishReason` lets a custom agent explicitly state why the turn
 * ended (e.g. `interrupted`, `length`). When omitted, no per-turn reason is
 * reported.
 *
 * Carried on a {@link CommittedTurnError}, it also says a failed turn left
 * state worth continuing from, which is what makes the turn snapshot and the
 * session resumable; see {@link SessionRunner.run}.
 */
export interface TurnResult {
  finishReason?: AgentFinishReason;
}

/**
 * Thrown by a turn handler to fail the turn while committing its state as a
 * resume point.
 *
 * A turn handler that throws any other error rolls the turn back: nothing is
 * persisted and the previous snapshot stays the resume point. Throwing this
 * instead commits the turn: the session state as the handler left it is
 * persisted as a `failed` snapshot carrying the error, and that snapshot (or,
 * client-managed, that state) is what the invocation's failed output reports
 * as the resume point. The handler is responsible for leaving the session at
 * a state that can be continued from; a prompt-backed agent commits whenever
 * the generate call produced a partial response, since that partial ends at
 * a turn seam.
 *
 * `cause` is the error the turn failed with, and is what the output and the
 * snapshot report. `result` may carry the turn's own finish reason; without
 * one the runner derives it from how the turn ended.
 */
export class CommittedTurnError extends Error {
  readonly result: TurnResult;

  constructor(cause: unknown, result: TurnResult = {}) {
    super(getErrorMessage(cause), { cause });
    this.name = 'CommittedTurnError';
    this.result = result;
    // Restore prototype chain for `instanceof` across transpilation targets.
    Object.setPrototypeOf(this, CommittedTurnError.prototype);
  }
}

/** Detects a {@link CommittedTurnError}, by class or by brand across bundles. */
function isCommittedTurnError(e: unknown): e is CommittedTurnError {
  return (
    e instanceof CommittedTurnError ||
    (e as { name?: unknown } | undefined)?.name === 'CommittedTurnError'
  );
}

/**
 * Reports whether an input carries data of its own: a message, or resume
 * directives. It filters pure detach signals out of the runner's queue, and it
 * marks the inputs a turn can start from: one without a payload runs on the
 * conversation already in the session, which is how a failed turn is
 * re-attempted, so there has to be a conversation there.
 */
function hasInputPayload(input: AgentInput | undefined): boolean {
  return !!(
    input?.message ||
    input?.resume?.restart?.length ||
    input?.resume?.respond?.length
  );
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
  /**
   * ID of the most recent turn-end snapshot for this invocation. Empty when
   * no store is configured, or when nothing has been committed yet: a
   * first-turn failure that rolled back on a fresh session. On a resumed
   * session whose first turn rolls back it is the resumed snapshot's id. When
   * `finishReason` is `detached` it is the pending detach snapshot. When
   * `failed` or `aborted`, it is the resume point: the turn's own snapshot
   * when the turn committed anything, otherwise the last committed turn's
   * snapshot.
   */
  snapshotId?: string;
  /**
   * Final conversation state (only when client-managed). When `finishReason`
   * is `failed` or `aborted`, this is the resume point: what the turn
   * committed, or the last committed turn's state when the turn ended before
   * committing anything.
   */
  state?: SessionState<S>;
  finishReason?: AgentFinishReason;
  /**
   * Present when `finishReason` is `failed` or `aborted`. Carries the original
   * error details (RuntimeError shape): what broke, or what stopped the run;
   * `state`/`snapshotId` hold the resume point.
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
 * The error a prompt-backed agent hands the runner for a generation the loop
 * reported on the response (see `throwOnError`): the classification the loop
 * would have thrown, so the runner reads the failure's status, text and
 * details, and tells a caller's stop from a break by the error's identity.
 */
function generationError(res: GenerateResponse): GenerationResponseError {
  const { status, message } = res.error!;
  const Ctor =
    res.finishReason === 'aborted'
      ? GenerationAbortedError
      : GenerationResponseError;
  // The loop's statuses are canonical; a status a model wrote itself may not
  // be, and an error is still owed for it.
  const known = StatusNameSchema.safeParse(status);
  return new Ctor(res, message, known.success ? known.data : 'INTERNAL');
}

/**
 * Reports whether a run ended because the caller stopped it rather than
 * because something inside it broke. Two roads reach the same place:
 *
 * The caller's abort signal fired. An attached caller aborts the signal it
 * passed to the action, or the transport under it closes; a detached caller
 * calls the `abort` companion action, which aborts the signal on the status
 * flip.
 *
 * Or the run reached a limit the caller set. The generate loop throws a
 * {@link GenerationAbortedError} at `maxTurns`, and reports every stop it
 * classified on a response it returned as one; a cancellation or timeout it
 * observed reaches a turn that let the loop throw as the `AbortError` or
 * `TimeoutError` itself. A turn that lets any of these through, whatever
 * raised it, is stopped, not broken, and reports so without having to say it
 * in a {@link TurnResult}, as a Go turn propagating a context error does.
 *
 * Both roads are read from the signal and the error's identity, never from a
 * classified status, which is a wider set than the caller's own doing: a
 * provider answering with ABORTED or DEADLINE_EXCEEDED did not stop the run on
 * the caller's request, and persisting that as an aborted row would tell a
 * retry client the one thing that is not true of it. Same rule as
 * `generate`'s, so the snapshot status and a partial's finish reason agree on
 * who ended the run.
 */
function callerStopped(
  abortSignal: AbortSignal | undefined,
  cause: unknown
): boolean {
  if (abortSignal?.aborted) return true;
  if (cause instanceof GenerationAbortedError) return true;
  const name = (cause as { name?: unknown } | undefined)?.name;
  return name === 'AbortError' || name === 'TimeoutError';
}

/**
 * How a turn or run that ended with `cause` reports itself: `aborted` when
 * the caller stopped it, `failed` when it broke.
 */
function terminalReason(
  abortSignal: AbortSignal | undefined,
  cause: unknown
): AgentFinishReason {
  return callerStopped(abortSignal, cause) ? 'aborted' : 'failed';
}

/**
 * The snapshot status that goes with the reason a turn or run ended on, for
 * the rows that ended with an error. Deriving it keeps the two from
 * disagreeing: a row is aborted exactly when it says the caller stopped the
 * run.
 */
function terminalStatus(
  reason: AgentFinishReason | undefined
): 'aborted' | 'failed' {
  return reason === 'aborted' ? 'aborted' : 'failed';
}

/**
 * The error a stop observed at the signal reports: the signal's reason when
 * it is one, else a CANCELLED error carrying it.
 */
function stopCause(abortSignal: AbortSignal): unknown {
  const reason = abortSignal.reason;
  if (reason instanceof Error) return reason;
  return new GenkitError({
    status: 'CANCELLED',
    message:
      reason === undefined ? 'The agent run was aborted.' : String(reason),
  });
}

/** Resolves the runner's stop promise once its abort signal fires. */
const STOPPED = Symbol('stopped');

/**
 * Normalizes a thrown value into the structured error shape used across the
 * agent (in `AgentOutput.error` and `SessionRunner.lastTurnError`). An error
 * without a status of its own is classified the way `generate` classifies
 * it: a timeout by its name, a cancellation by its name or by `stopped`
 * (the caller stopped the run with it), anything else as INTERNAL.
 */
function toErrorDetails(e: any, stopped = false): AgentErrorDetails {
  const status =
    e?.status ||
    (e?.name === 'TimeoutError'
      ? 'DEADLINE_EXCEEDED'
      : e?.name === 'AbortError' || stopped
        ? 'CANCELLED'
        : 'INTERNAL');
  return {
    status,
    // A GenkitError's own text: the status it prefixes its message with
    // travels on `status`, and a client matching the recorded message across
    // runtimes reads the same words a Go or Python agent records.
    message: e?.originalMessage || e?.message || 'Internal failure',
    details: toErrorDetailsPayload(e?.detail ?? e?.details),
  };
}

/**
 * The structured details an error carries onto the wire and into a snapshot.
 * Only explicitly-provided structured details are surfaced, never the raw
 * thrown value: it is serialized over the wire (AgentOutput.error) and
 * persisted into snapshots, so leaking it could expose stack traces or
 * internal state, or break JSON.stringify on circular error objects. A
 * generation error's partial response (`detail.response`) is dropped for the
 * same reason: it carries the request's whole conversation, which a failed
 * turn commits to the session on its own terms, and the classified error the
 * response reports stands in for it.
 */
function toErrorDetailsPayload(detail: unknown): unknown {
  if (!detail || typeof detail !== 'object' || Array.isArray(detail)) {
    return detail;
  }
  const {
    response,
    request: _request,
    ...rest
  } = detail as Record<string, any>;
  const inner = response?.error?.details;
  if (inner !== undefined) return inner;
  return Object.keys(rest).length > 0 ? rest : undefined;
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
  /**
   * How the most recent turn ended, or `aborted` when the caller stopped the
   * run before the next turn started.
   */
  public lastTurnFinishReason?: AgentFinishReason;
  /**
   * Error details of the most recent turn that failed or was stopped. Set
   * when a turn throws, or the caller stops the run, and the runner resolves
   * gracefully instead of propagating the exception.
   */
  public lastTurnError?: AgentErrorDetails;
  /**
   * Whether the most recent turn left state worth continuing from. A
   * successful turn always has; a failed one has when its handler threw a
   * {@link CommittedTurnError}. Decides whether the turn snapshots and whether
   * the live state is the resume point. True until a turn fails without
   * committing.
   */
  public lastTurnCommitted: boolean = true;
  /**
   * A deep copy of the session state as of the most recent *committed* turn
   * (or the initial state when no turn has committed yet). On a turn that
   * failed without committing this is the state that turn started with, which
   * is the resume point the failed output hands back (client-managed) and the
   * state a detached run's finalize records.
   */
  public lastGoodState?: SessionState<State>;
  /**
   * The snapshotId of the most recently *committed* turn: the failed turn's
   * own snapshot when it committed, otherwise the last successful turn's.
   * `undefined` when no turn has committed yet (e.g. a first-turn failure
   * that rolled back).
   */
  public lastGoodSnapshotId?: string;
  private lastSnapshot?: SessionSnapshot<State>;

  private lastSnapshotVersion: number = 0;

  private store?: SessionStore<State>;
  /**
   * True once the client detached. Per-turn snapshot writes are suspended
   * from then on: the pending row written by {@link detach} already captures
   * the invocation, and a single {@link finalizePendingSnapshot} rewrite
   * records the cumulative state once the queued inputs drain.
   */
  public isDetached: boolean = false;
  /**
   * The id of the pending row a detach wrote, which the finalize rewrites.
   * Undefined until the client detaches.
   */
  public pendingSnapshotId?: string;
  /**
   * Set as soon as a detach is requested, before its pending-row write
   * lands. From then on the caller's signal no longer stops the run (see the
   * agent action's abort wiring): the client that detached may close its
   * transport while the row is still being written.
   */
  public detachRequested: boolean = false;
  /** The requested detach's pending-row write; see {@link detach}. */
  private detachInFlight?: Promise<string>;
  /**
   * Set once the run has settled and its output is decided. A detach that
   * arrives after that has no run to move to the background: {@link detach}
   * refuses it rather than write a pending row nothing would finalize.
   */
  public runEnded: boolean = false;
  /**
   * The pending row as written, kept so the finalize can rebuild it (its
   * lineage and timestamps) even when the store no longer returns it.
   */
  private pendingRow?: SessionSnapshotInput<State>;
  /**
   * Serializes snapshot writes. The detach write and an in-flight turn-end
   * write run on different tasks; under the lock the detach either waits for
   * that write to land or suspends before it starts, so the two never land as
   * sibling leaves and a turn-end write never follows the pending row.
   */
  private snapLock: Promise<unknown> = Promise.resolve();
  /**
   * Fires when the caller stops the run: an attached caller aborting the
   * signal it passed to the action, or the `abort` companion action flipping
   * a detached run's row. A turn in flight observes it through the signal the
   * agent function receives; the runner reads it to report that turn, or an
   * input still queued, as `aborted` rather than `failed` (see `run`).
   */
  private abortSignal?: AbortSignal;
  /** Resolves once `abortSignal` fires; never, without one. */
  private stopped: Promise<typeof STOPPED> = new Promise(() => {});

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
    const signal = this.abortSignal;
    if (signal) {
      this.stopped = new Promise((resolve) => {
        if (signal.aborted) {
          resolve(STOPPED);
        } else {
          signal.addEventListener('abort', () => resolve(STOPPED), {
            once: true,
          });
        }
      });
    }
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
   * reason is reported.
   *
   * When the handler throws, the runner records the failure, stops looping,
   * and lets the invocation resolve with `finishReason: 'failed'`, or
   * `'aborted'` when the caller stopped the run: its abort signal fired, or
   * the error is a cancellation, a timeout, or a caller-set limit such as
   * `maxTurns` (see `callerStopped`). What it does with the turn's state
   * depends on what was thrown: a
   * {@link CommittedTurnError} commits the turn, which snapshots the session
   * as `failed` with the error on the row and makes it the resume point, and
   * any other error rolls the turn back, leaving the previous snapshot as the
   * resume point. A prompt-backed agent commits whenever the generate call
   * produced a partial response, because that partial ends at a turn seam and
   * is a conversation the caller can continue from; a custom agent commits
   * when it knows the same of its own state.
   *
   * A stop that lands between turns drops the inputs still queued and reports
   * `aborted` with the last committed turn as the resume point.
   */
  async run(
    fn: (input: AgentInput, ctx: TurnContext) => Promise<TurnResult | void>
  ): Promise<void> {
    const inputs = this.inputCh[Symbol.asyncIterator]();
    while (true) {
      const nextInput = inputs.next();
      const next = await Promise.race([nextInput, this.stopped]);
      // The caller stopping the run wins over an input still queued for it
      // and over a run that would otherwise end on its own terms: the
      // invocation reports the stop and the resume point the last committed
      // turn left. An input the race left unread is dropped.
      if (this.abortSignal?.aborted) {
        nextInput.catch(() => {});
        this.recordStop();
        break;
      }
      if (next === STOPPED || next.done) break;
      const input = next.value;
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

      // The turn's own error, once the failure arm has recorded it. It ends
      // the loop gracefully; any other error out of the span (a store failing
      // while the turn was recorded) propagates.
      let turnError: unknown;
      let turnThrew = false;
      try {
        await run(`runTurn-${this.turnIndex + 1}`, input, async () => {
          let turnResult: TurnResult | void;
          try {
            turnResult = await fn(input, turnContext);
          } catch (e) {
            turnThrew = true;
            turnError = e;
            const snapshotId = await this.endFailedTurn(e, turnSnapshotId);
            // Tag the span with the failed turn's snapshot as a success is
            // tagged with its own, so a trace correlates either with its row.
            if (snapshotId) {
              setCustomMetadataAttribute('agent:snapshotId', snapshotId);
            }
            throw e;
          }
          const finishReason = turnResult?.finishReason;
          this.lastTurnFinishReason = finishReason;
          this.lastTurnError = undefined;
          this.lastTurnCommitted = true;

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
      } catch (e) {
        if (!turnThrew || e !== turnError) throw e;
        // Graceful failure: rather than propagating the exception (which would
        // discard the action's final return - and with it the resume point and
        // all prior committed turns), stop processing further inputs and let
        // the invocation resolve with `finishReason: 'failed'`. The caller
        // recovers the resume point from the returned AgentOutput.
        break;
      }
    }
  }

  /**
   * Records a turn whose handler threw `e`, writing the turn's snapshot when
   * it committed, and returns that snapshot's id.
   *
   * What the turn threw decides what happens to its state: a
   * CommittedTurnError commits the turn as a resume point, anything else
   * rolls it back (see `run`). The error the output and the snapshot report
   * is the underlying cause either way. Who ended the turn decides how it
   * reports itself: the caller stopping the run wins over whatever the turn
   * reported, so a stop that lands while a turn is settling keeps the
   * aborted terminal, attached or detached. Otherwise the turn's own finish
   * reason, when the result names one, goes on the turn-end chunk and the
   * row; the invocation reports how it ended.
   */
  private async endFailedTurn(
    e: unknown,
    turnSnapshotId: string | undefined
  ): Promise<string | undefined> {
    const committed = isCommittedTurnError(e);
    const cause = committed ? e.cause : e;
    const finishReason: AgentFinishReason = callerStopped(
      this.abortSignal,
      cause
    )
      ? 'aborted'
      : (committed && e.result?.finishReason) || 'failed';
    this.lastTurnFinishReason = finishReason;
    // A turn that rethrew a signal's reason that is not an error (a string,
    // nothing at all) is recorded through the signal, as a stop between
    // turns is.
    const stopped = finishReason === 'aborted';
    const reported =
      stopped &&
      (cause === null || typeof cause !== 'object') &&
      this.abortSignal
        ? stopCause(this.abortSignal)
        : cause;
    this.lastTurnError = toErrorDetails(reported, stopped);
    this.lastTurnCommitted = committed;

    let snapshotId: string | undefined;
    if (committed) {
      // The turn's own snapshot, with the error on the row and the status
      // its reason says, is the newest snapshot and so the resume point
      // the output reports.
      snapshotId = await this.maybeSnapshot(
        terminalStatus(finishReason),
        this.lastTurnError,
        turnSnapshotId,
        finishReason
      );
      this.lastGoodState = this.session.getState();
      this.lastGoodSnapshotId = snapshotId ?? this.lastGoodSnapshotId;
    }
    this.notifyEndTurn(snapshotId, finishReason);
    return snapshotId;
  }

  /**
   * Records that the caller stopped the run between turns, before an input
   * still queued was started. No turn ran, so the last committed turn keeps
   * its place as the resume point; the invocation reports `aborted` with the
   * signal's reason as the error.
   */
  private recordStop(): void {
    this.lastTurnFinishReason = 'aborted';
    this.lastTurnError = toErrorDetails(stopCause(this.abortSignal!), true);
  }

  /**
   * The session state as of the last turn that committed: the live state when
   * that is the last turn, else the copy taken at the last committed turn,
   * which is the state the invocation began with until a turn commits. It is
   * what a run stopping mid-turn hands back or lands on its row, since the
   * live state holds the unfinished turn's mutations.
   */
  committedState(): SessionState<State> {
    return this.lastTurnCommitted
      ? this.session.getState()
      : (this.lastGoodState ?? this.session.getState());
  }

  /** Runs `fn` under the snapshot lock; see {@link snapLock}. */
  private withSnapLock<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.snapLock.then(fn, fn);
    this.snapLock = run.catch(() => {});
    return run;
  }

  /**
   * Saves a snapshot of the current session state to the persistent store.
   *
   * When a store is configured every turn is persisted (snapshotting is no
   * longer opt-out). Uses the mutator-based `saveSnapshot` to atomically check
   * that the snapshot has not been concurrently aborted before writing -
   * preventing a race where a "done" write could overwrite a concurrent
   * "aborted" status.
   *
   * Once the client has detached no per-turn row is written: the pending row
   * {@link detach} wrote already captures the invocation, so this returns its
   * id and the finalize records the cumulative state when the run settles.
   * Pending rows themselves are written by {@link detach}, not here.
   */
  async maybeSnapshot(
    status?: 'pending' | 'completed' | 'failed' | 'aborted',
    error?: { status?: string; message: string; details?: any },
    snapshotId?: string,
    finishReason?: AgentFinishReason
  ): Promise<string | undefined> {
    if (!this.store) return undefined;
    return this.withSnapLock(async () => {
      // Re-checked under the lock: a detach that landed while this write was
      // waiting suspends it, so the turn's state goes to the finalize instead
      // of landing beside the pending row.
      if (this.isDetached) return this.pendingSnapshotId;

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
        // Default to a `completed` status. The only caller that omits a status
        // is the post-invocation write, which fires when the handler mutates
        // state after the last turn of a run that ended well; the row it
        // writes is a settled resume point.
        status: status ?? 'completed',
        ...(finishReason && { finishReason }),
        error,
      };

      const effectiveId = snapshotId || this.newSnapshotId;

      // Use the mutator-based saveSnapshot to atomically check the current
      // status before writing.  If the snapshot was concurrently aborted,
      // the mutator returns null and the write is skipped.
      const assignedId = await this.store!.saveSnapshot(
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
    });
  }

  /**
   * Detaches the invocation: writes the pending row that stands for the
   * background work and suspends per-turn snapshot writes from here on.
   *
   * The row carries no state. The live state holds an unfinished turn's
   * mutations, and the state the row lands with is what
   * {@link finalizePendingSnapshot} records once the run settles. Its id is
   * the next turn's reserved id when one is waiting, so a handler naming
   * external resources after its turn's `snapshotId` names the row the
   * invocation finalizes.
   *
   * Idempotent: a second detach input returns the row already written.
   * Resolves with the pending row's id.
   */
  async detach(): Promise<string> {
    if (!this.store) {
      throw new GenkitError({
        status: 'FAILED_PRECONDITION',
        message: 'Detach is only supported when a session store is provided.',
      });
    }
    if (this.detachInFlight) return this.detachInFlight;
    if (this.runEnded) {
      throw new GenkitError({
        status: 'FAILED_PRECONDITION',
        message: 'The run has ended; there is nothing left to detach.',
      });
    }
    this.detachRequested = true;
    this.detachInFlight = this.withSnapLock(async () => {
      const snapshotId = this.newSnapshotId || reserveSnapshotId();
      const now = new Date().toISOString();
      const row: SessionSnapshotInput<State> = {
        snapshotId,
        sessionId: this.session.sessionId,
        parentId: this.lastSnapshot?.snapshotId,
        createdAt: now,
        updatedAt: now,
        status: 'pending',
        // A background heartbeat loop refreshes this; if it goes stale the
        // row is reported as `expired` on read (the worker is presumed dead).
        heartbeatAt: now,
      };
      await this.store!.saveSnapshot(snapshotId, () => row, {
        context: getContext(),
      });
      this.pendingRow = row;
      this.pendingSnapshotId = snapshotId;
      this.newSnapshotId = snapshotId;
      this.isDetached = true;
      this.onDetach?.(snapshotId);
      return snapshotId;
    });
    return this.detachInFlight;
  }

  /**
   * Resolves once a requested detach has written its pending row, or failed
   * to: the finalize waits on it, so a row still being written when the run
   * settles is rewritten rather than left pending.
   */
  async detachSettled(): Promise<void> {
    await this.detachInFlight?.catch(() => {});
  }

  /**
   * Rewrites the pending row a detach wrote with how the run ended: the
   * cumulative session state, the terminal status, the last turn's finish
   * reason, and the error when it failed. The row keeps its lineage and
   * creation time; its heartbeat is cleared, since the row is settled.
   *
   * A late abort wins over the terminal the run was about to land: the row
   * keeps `aborted`, and the finish reason and the committed state are
   * stamped, so the row is self-describing and resumable (the abort write
   * flips the status of a pending row that carries no state; the runtime
   * owns the reason and the state). Nothing is written once that stamp is on.
   *
   * `cause` is an error the agent function itself threw, which ends the run
   * whatever its turns did: `failed`, or `aborted` when the caller stopped it
   * (see `callerStopped`). Persistence is best-effort: a store failure is
   * logged and does not surface.
   */
  async finalizePendingSnapshot(cause?: unknown): Promise<void> {
    const store = this.store;
    const snapshotId = this.pendingSnapshotId;
    if (!store || !snapshotId) return;

    // How the run ended. An error the agent function threw is read the way
    // the runner reads a turn's: a background run that reached a caller-set
    // limit was stopped, not broken. Otherwise the last turn's own error and
    // finish reason stand.
    const finishReason =
      cause !== undefined
        ? terminalReason(this.abortSignal, cause)
        : this.lastTurnFinishReason;
    const error =
      cause !== undefined ? toErrorDetails(cause) : this.lastTurnError;
    const status = error ? terminalStatus(finishReason) : 'completed';
    // The state the row lands with: everything through the last turn that
    // committed, so an unfinished turn's mutations do not ride onto a row
    // that is a resume point.
    const state = this.committedState();
    const now = new Date().toISOString();

    try {
      await this.withSnapLock(() =>
        store.saveSnapshot(
          snapshotId,
          (existing) => {
            const { heartbeatAt: _, ...base } = existing ?? this.pendingRow!;
            if (base.status === 'aborted') {
              if (base.finishReason === 'aborted') return null;
              return {
                ...base,
                finishReason: 'aborted',
                state,
                updatedAt: now,
              };
            }
            return {
              ...base,
              status,
              ...(finishReason && { finishReason }),
              error,
              state,
              updatedAt: now,
            };
          },
          { context: getContext() }
        )
      );
    } catch (e) {
      logger.error(
        `agent: failed to finalize detached snapshot ${snapshotId}: ${getErrorMessage(e)}`
      );
    }
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

    assertResumable(snapshot);
    validateCustomState(snapshot.state?.custom);
    return {
      snapshot,
      session: new Session<State>(snapshot.state as SessionState<State>),
    };
  }

  if (init?.sessionId) {
    // Resume the session's latest snapshot. The store returns the literal
    // latest leaf whatever its status, and it is validated the way a snapshot
    // named by id is: a caller wanting to continue past a dead-end tip names
    // an earlier snapshot explicitly via `snapshotId`. When the session has no
    // snapshot yet, seed a fresh session bound to the requested sessionId so
    // subsequent turns can find it.
    const snapshot = await store.getSnapshot({
      sessionId: init.sessionId,
      context: getContext(),
    });
    if (snapshot) {
      assertResumable(snapshot);
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
 * Rejects a snapshot that cannot be continued from. A `pending` row is still
 * being written by its detached invocation. A `failed` or `aborted` row can
 * be continued from: the turn that wrote it committed a conversation ending
 * at a turn seam, and whether to continue from a run that broke or one that
 * was stopped is the caller's judgement, not the framework's. The exception
 * is an aborted row carrying no state, which is one caught between the
 * abort's status flip and the finalize that stamps the state on.
 */
function assertResumable(snapshot: SessionSnapshot<unknown>): void {
  switch (snapshot.status) {
    case 'pending':
      throw new GenkitError({
        status: 'FAILED_PRECONDITION',
        message:
          `Snapshot ${snapshot.snapshotId} is still pending: its detached ` +
          `invocation is still running; wait for it to finalize or abort it ` +
          `before resuming.`,
      });
    case 'aborted':
      // An aborted row is the one terminal shape written twice: the abort
      // flips the pending row, which carries no state, and the finalize that
      // follows stamps the state onto it. A row still between the two holds
      // nothing, and resuming it would silently hand back an empty session in
      // place of the conversation the caller asked to continue.
      //
      // Which half of that window this is decides what the caller should do
      // next, and the heartbeat says: the abort leaves it running and the
      // finalize clears it, so a live beat means the state is one write away
      // and this same id is the thing to wait on. Sending that caller to an
      // earlier snapshot would fork the run away from the work the finalize
      // is about to commit. Only a quiet beat means the write is never coming
      // (the process died, or the write failed), and the earlier snapshot
      // really is the resume point.
      if (!snapshot.state) {
        if (finalizeInFlight(snapshot)) {
          throw new GenkitError({
            status: 'FAILED_PRECONDITION',
            message:
              `Snapshot ${snapshot.snapshotId} is still being finalized: its ` +
              `invocation was aborted and has not recorded the state yet; ` +
              `retry this same snapshot ID.`,
          });
        }
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message:
            `Snapshot ${snapshot.snapshotId} was aborted before its ` +
            `invocation recorded any state; resume from an earlier snapshot.`,
        });
      }
  }
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
        // Once the run has settled nothing reads the queue any more, and a
        // detach has no run to move to the background: later inputs are
        // dropped rather than queued or written up as pending.
        if (getRunner()?.runEnded) continue;
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
            // Writes the pending row, suspends per-turn snapshots, and fires
            // onDetach. Under the runner's snapshot lock, so an in-flight
            // turn-end write either lands first or is suspended.
            await getRunner().detach();
          }
          // Only forward to the runner if the input carries a payload beyond
          // the detach directive; a detach-only message has no turn to process.
          if (hasInputPayload(input)) {
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

      // The caller's signal is the attached lever: aborting it stops the run
      // the way the abort companion action stops a detached one. An in-flight
      // turn observes the stop through the signal the agent function gets,
      // inputs still queued are dropped, and the invocation resolves with
      // `aborted` naming the resume point (see SessionRunner.run). A run that
      // is detaching or detached has no caller left to stop it, so the signal
      // is ignored from the detach request on: the transport closing behind
      // a detach is not an abort, even while the pending row is still being
      // written.
      const onCallerAbort = () => {
        if (runner?.detachRequested) return;
        abortController.abort(arg.abortSignal.reason);
      };
      if (arg.abortSignal.aborted) {
        onCallerAbort();
      } else {
        arg.abortSignal.addEventListener('abort', onCallerAbort, {
          once: true,
        });
      }

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
        let result: AgentResult;
        let finalSnapshotId: string | undefined;
        // An error the agent function threw outside a turn. Attached, it
        // propagates as the action's own failure; detached, it lands on the
        // pending row, since there is no longer a caller to throw to.
        let fnError: unknown;
        let fnThrew = false;
        try {
          result = await runWithSession(registry, session, () =>
            fn(runner, {
              sendChunk,
              abortSignal: abortController.signal,
              context: getContext(),
            })
          );
          // After the handler resolves, persist any state it mutated after the
          // last turn. Omitting a status defaults to a resumable `completed`
          // write, which the version guard skips when nothing changed. A
          // detached run has nothing to write here: its finalize records the
          // cumulative state. Nor does a run that failed or was stopped: the
          // resume point is the last committed snapshot, whether that is the
          // turn's own row or its predecessor's, and a completed row on top
          // of it would displace it as the session's latest.
          finalSnapshotId =
            runner.lastTurnCommitted && !runner.lastTurnError
              ? await runner.maybeSnapshot()
              : runner.lastGoodSnapshotId;
        } catch (e) {
          fnError = e;
          fnThrew = true;
        } finally {
          // The run has settled: a detach from here on is refused, and the
          // pending row's heartbeat stops before the finalize clears it.
          runner.runEnded = true;
          stopHeartbeat();
          arg.abortSignal.removeEventListener('abort', onCallerAbort);
          if (unsubscribe) unsubscribe();
          session.off('artifactAdded', sendArtifactChunk);
          session.off('artifactUpdated', sendArtifactChunk);
          session.off('customChanged', sendCustomPatch);
          // A detach whose pending-row write is still in flight lands before
          // the finalize rewrites that row.
          if (runner.detachRequested) await runner.detachSettled();
          if (runner.isDetached) {
            await runner.finalizePendingSnapshot(fnThrew ? fnError : undefined);
          }
        }
        if (fnThrew && !runner.isDetached) throw fnError;
        return { result: result!, finalSnapshotId };
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

      // A turn failed, or the caller stopped the run: resolve gracefully with
      // `finishReason: 'failed'` or `'aborted'`, the error, and the resume
      // point. That is the state through the last committed turn, which is
      // the turn itself when it committed and its predecessor when it did
      // not, since only a committed turn snapshots and advances the last-good
      // state. No message and no artifacts: they describe the result of a
      // run that finished, and the live artifacts would carry a rolled-back
      // turn's. The turn's own finish reason, when its result named one, is
      // on the turn-end chunk and the row; the invocation reports how it
      // ended.
      if (runner.lastTurnError) {
        return {
          sessionId: session.sessionId,
          finishReason: (runner.lastTurnFinishReason === 'aborted'
            ? 'aborted'
            : 'failed') as AgentFinishReason,
          error: runner.lastTurnError,
          // Server-managed: the newest snapshot is the resume point. Undefined
          // when nothing has been committed yet (a first-turn failure that
          // rolled back on a fresh session).
          ...(config.store && { snapshotId: runner.lastGoodSnapshotId }),
          // Client-managed: return the resume point's state directly.
          ...(!config.store && {
            state: toClientState(runner.committedState()),
          }),
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

      // An input with no payload of its own runs the turn on the conversation
      // as it stands, which is how a failed turn is re-attempted: the failed
      // snapshot holds the messages the turn committed, and the model is
      // called on them again. There has to be something to continue.
      const sessionMessages = sess.getMessages();
      if (!hasInputPayload(input) && sessionMessages.length === 0) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message:
            'agent input message or resume is required to start a conversation',
        });
      }

      // Tag every history message so we can identify them after render.
      const history = sessionMessages.map((m) => ({
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

      // Failures come back on the response: the loop's classification of
      // what broke or what stopped it, over the conversation it completed. A
      // failure before the request resolved (a render or validation failure)
      // still throws, and rolls the turn back.
      const result = generateStream(registry, {
        ...genOpts,
        abortSignal,
        throwOnError: false,
      });

      // Keep everything that is NOT a prompt-template message:
      //   • history messages (clean - history tag was stripped before generate)
      //   • new messages from tool loops (untagged)
      //   • model response
      const turnSessionMessages = (messages: MessageData[]) =>
        messages.filter((m) => !m.metadata?.[promptTag]);

      for await (const chunk of result.stream) {
        sendChunk({ modelChunk: chunk });
      }
      const res = await result.response;
      // A failure the loop reported is one the response cannot pass for a
      // valid one: something broke, the caller stopped the loop, the model
      // blocked the response or returned none, or the output it completed
      // does not match the schema. An `error` a model wrote onto a response
      // that is otherwise valid is not one of those, and the turn keeps the
      // reply. Nor is an interrupt: `generate` reports a restarted tool that
      // interrupted again with a FAILED_PRECONDITION, because its caller
      // asked for a completed generation; the agent's caller did not. The
      // tip that comes back is the same answerable interrupt a first-run
      // interrupt leaves, and only `resume` can answer either, so the turn
      // takes the success path and commits the same way both times.
      if (res.error && res.finishReason !== 'interrupted' && !res.isValid()) {
        // The turn commits the conversation the failing step began from,
        // which ends at a turn seam (see `generate`) and is what the next
        // attempt sends again. A response the model completed and the loop
        // rejected (blocked, without a message, output off the schema) keeps
        // that message for the caller to see, but not as a message to
        // continue from: the seam is the request's messages. The turn is
        // committed as a resume point: a CommittedTurnError is what says so
        // (see SessionRunner.run). The runner reads a failure from the error
        // it is handed, so the response is handed over as one, built before
        // the session changes. The turn records `failed`, not the response's
        // own finish reason: a rejected completion still carries the model's
        // `stop`.
        const failure = generationError(res);
        sess.setMessages(
          turnSessionMessages(res.request?.messages ?? res.messages)
        );
        throw new CommittedTurnError(failure);
      }

      if (res.request?.messages) {
        const msgs = turnSessionMessages(res.request.messages);
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
  // Searches history newest-first so a resume matches the most recent tool
  // request for a given name + ref.
  const findToolRequest = (name: string, ref?: string) => {
    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      if (msg.role === 'model') {
        for (const part of msg.content) {
          const tr = part.toolRequest;
          if (tr && tr.name === name && tr.ref === ref) {
            return tr;
          }
        }
      }
    }
    return undefined;
  };

  // Validate restart entries: name + ref must exist AND input must match exactly
  for (const restart of resume.restart || []) {
    const { name, ref, input } = restart.toolRequest;
    const match = findToolRequest(name, ref);
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
    const match = findToolRequest(name, ref);
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
