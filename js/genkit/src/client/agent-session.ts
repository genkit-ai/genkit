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

/**
 * `AgentSession` — framework-agnostic session core for the Agents API.
 *
 * Owns the entire client-side state machine: streaming chunk dispatch,
 * continuation-token round-trip, mid-stream interrupt detection,
 * foreground → background phase transitions with polling, snapshot
 * rehydration, and lifecycle cleanup.
 *
 * Framework adapters (React, Angular, Vue, Svelte, Solid, …) wrap this
 * with their own reactivity primitives. The adapter is responsible for
 * subscribing to `subscribe()` and forwarding state changes to its UI
 * layer; the session does not depend on any framework.
 *
 * @example
 * ```ts
 * const session = new AgentSession({ url: '/api/agent' });
 * const unsubscribe = session.subscribe(() => {
 *   const state = session.getState();
 *   // ...render
 * });
 * session.submit({ messages: [{ role: 'user', content: [{ text: 'hi' }] }] });
 * // later
 * session.dispose();
 * ```
 */
import { runFlow, streamFlow } from './client.js';
import type { AgentEvent } from './agent-events.js';
import { walkAgentEvent } from './agent-events.js';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type ToolCallState = 'call' | 'result' | 'error';

export interface ToolCall<I = unknown, O = unknown> {
  id: string;
  name: string;
  input: I;
  output?: O;
  state: ToolCallState;
  /** Populated when `state === 'error'`. */
  errorText?: string;
  errorCode?: string;
}

export interface AgentMessage {
  role: 'user' | 'model' | 'tool' | 'system' | string;
  content: Array<{
    text?: string;
    reasoning?: string;
    toolRequest?: { name: string; input?: unknown; ref?: string };
    toolResponse?: { name: string; output?: unknown; ref?: string };
    [k: string]: unknown;
  }>;
}

export interface PendingInterrupt {
  toolCallId: string;
  toolName: string;
  input: unknown;
  kind: 'respond' | 'restart';
  metadata?: unknown;
}

export type AgentPhase =
  | 'idle'
  | 'streaming'
  | 'background'
  | 'awaiting-interrupt'
  | 'done'
  | 'error';

export interface AgentSessionOptions {
  url: string;
  stateUrl?: string;
  abortUrl?: string;
  /**
   * If provided, the session immediately fetches the snapshot and
   * rehydrates state. Accepts either a continuationId (opaque,
   * preferred) or a raw snapshotId (convenient when the value came from
   * a URL route param).
   */
  resumeFromContinuation?: string;
  resumeFromSnapshotId?: string;
  headers?: Record<string, string>;
}

export interface AgentInputBody {
  messages?: AgentMessage[];
  resume?: {
    respond?: Array<{
      toolResponse: { name: string; output?: unknown; ref?: string };
    }>;
    restart?: Array<{
      toolRequest: { name: string; input?: unknown; ref?: string };
      metadata?: unknown;
    }>;
  };
  detach?: boolean;
}

export interface AgentVariant<S = unknown> {
  /** Continuation token for this branch — pass to `continueFrom` to pick it. */
  continuationId: string | undefined;
  /** Raw snapshotId convenience (server-stored agents only). */
  snapshotId: string | undefined;
  /** Final assistant message for this branch. */
  message: AgentMessage | undefined;
  state:
    | { messages?: AgentMessage[]; custom?: S; artifacts?: unknown[] }
    | undefined;
}

export interface AgentSessionState<S = unknown> {
  messages: AgentMessage[];
  artifacts: unknown[];
  customState: S | undefined;
  streamingText: string;
  streamingReasoning: string;
  toolCalls: ToolCall[];
  statusLabel: string | null;
  progress: { current: number; total: number; label?: string } | null;
  phase: AgentPhase;
  error: Error | null;
  /** Opaque token; round-tripping is automatic, exposed for debugging/URL. */
  continuationId: string | undefined;
  /**
   * Derived snapshotId for server-stored agents (extracted from the
   * continuation token). Undefined when the underlying agent is stateless.
   */
  snapshotId: string | undefined;
  pendingInterrupt: PendingInterrupt | null;
}

interface AgentOutputBody<S = unknown> {
  continuationId?: string;
  message?: AgentMessage;
  artifacts?: unknown[];
  state?: { messages?: AgentMessage[]; custom?: S; artifacts?: unknown[] };
}

interface AgentInitBody {
  continuationId?: string;
}

type Listener = () => void;

// ---------------------------------------------------------------------------
// Session class
// ---------------------------------------------------------------------------

export class AgentSession<S = unknown> {
  private options: AgentSessionOptions;
  private state: AgentSessionState<S>;
  private listeners = new Set<Listener>();

  private abortController: AbortController | null = null;
  // Active background-poll timeout. Cleared on reset/abort/unmount so we
  // don't leak timers or call notify() after dispose.
  private pollTimeout: ReturnType<typeof setTimeout> | null = null;
  // Generation token bumped on every reset/abort so an in-flight poll cycle
  // started before the reset stops calling notify().
  private pollGen = 0;
  // Continuations we've already rehydrated this session — avoids re-fetching
  // the snapshot every time the URL pushes a new id mid-conversation.
  private rehydrated = new Set<string>();
  private disposed = false;

  constructor(options: AgentSessionOptions) {
    this.options = options;
    const initial = normalizeResumeToken(options);
    this.state = makeInitialState<S>(initial);
    if (initial) {
      void this.rehydrate(initial);
    }
  }

  // -------------------------------------------------------------------------
  // Adapter API: subscribe, getState, setOptions, dispose
  // -------------------------------------------------------------------------

  /**
   * Subscribe to state changes. Returns an unsubscribe function. The
   * listener is called synchronously after each state mutation; consumers
   * can read the latest state via `getState()`.
   */
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Returns the current state snapshot. The reference is stable across
   * calls until the next mutation — safe to use with React's
   * `useSyncExternalStore`, Solid's `from()`, etc.
   */
  getState(): AgentSessionState<S> {
    return this.state;
  }

  /**
   * Update options at runtime. Used by adapters that re-receive options
   * on every render (e.g. React) so the latest URL / headers / etc.
   * apply to subsequent requests. Resume tokens are NOT re-triggered
   * here — callers that want re-rehydration should construct a new
   * session or use `continueFrom()`.
   */
  setOptions(options: AgentSessionOptions): void {
    this.options = options;
  }

  /**
   * Release all resources (in-flight requests, timers, subscribers).
   * After dispose, no further state updates are emitted. Idempotent.
   *
   * Adapters generally don't need to call this — long-running internals
   * (background poll) self-terminate when `listeners.size === 0`, and
   * GC handles the rest. Use it when you want explicit cleanup (tests,
   * teardown of long-lived sessions).
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.abortController?.abort();
    this.abortController = null;
    this.clearActivePoll();
    this.listeners.clear();
  }

  // -------------------------------------------------------------------------
  // Action API
  // -------------------------------------------------------------------------

  /** Begin a new turn. State updates flow through `subscribe` listeners. */
  submit(input: AgentInputBody): void {
    if (this.disposed) return;
    this.internalSubmit(input);
  }

  /**
   * Cancel the current turn. In foreground phase, aborts the streaming
   * connection. In background phase, also calls the server's abort
   * endpoint so the background work actually stops. Idempotent.
   */
  async abort(): Promise<void> {
    if (this.disposed) return;
    this.abortController?.abort();
    this.abortController = null;
    const wasBackground = this.state.phase === 'background';
    const sidForAbort = wasBackground ? this.state.snapshotId : undefined;
    this.clearActivePoll();
    if (
      this.state.phase === 'streaming' ||
      this.state.phase === 'background'
    ) {
      this.mutate({ phase: 'idle' });
    }
    if (wasBackground && sidForAbort) {
      try {
        await runFlow({
          url: this.options.abortUrl ?? `${this.options.url}/abort`,
          input: sidForAbort,
          headers: this.options.headers,
        });
      } catch (_) {
        // ignore — local state already cleared
      }
    }
  }

  /** Reset to initial state, clearing all conversation history. */
  reset(): void {
    if (this.disposed) return;
    this.abortController?.abort();
    this.abortController = null;
    this.clearActivePoll();
    this.rehydrated = new Set();
    this.state = makeInitialState<S>(undefined);
    this.notify();
  }

  /** Respond to the pending interrupt (the `respond` resume kind). */
  respondToInterrupt(output: unknown): void {
    const pi = this.state.pendingInterrupt;
    if (!pi) return;
    this.submit({
      resume: {
        respond: [
          {
            toolResponse: { name: pi.toolName, ref: pi.toolCallId, output },
          },
        ],
      },
    });
  }

  /** Restart the pending interrupt's tool (the `restart` resume kind). */
  restartInterrupt(metadata?: unknown): void {
    const pi = this.state.pendingInterrupt;
    if (!pi) return;
    this.submit({
      resume: {
        restart: [
          {
            toolRequest: {
              name: pi.toolName,
              ref: pi.toolCallId,
              input: pi.input,
            },
            metadata,
          },
        ],
      },
    });
  }

  /**
   * Run the same turn N times in parallel from the current continuation
   * point. Doesn't mutate session state — caller picks a variant with
   * `continueFrom`.
   */
  async runVariants(
    input: AgentInputBody,
    count = 2
  ): Promise<AgentVariant<S>[]> {
    const init: AgentInitBody = this.state.continuationId
      ? { continuationId: this.state.continuationId }
      : {};
    const calls = Array.from({ length: count }, () =>
      runFlow<AgentOutputBody<S>, AgentInitBody>({
        url: this.options.url,
        input,
        init,
        headers: this.options.headers,
      })
    );
    const outputs = await Promise.all(calls);
    return outputs.map((o) => ({
      continuationId: o?.continuationId,
      snapshotId: o?.continuationId
        ? (extractSnapshotId(o.continuationId) ?? undefined)
        : undefined,
      message: o?.message,
      state: o?.state,
    }));
  }

  /**
   * Advance the session to a specific continuation or snapshot. Fetches
   * the snapshot's state and rehydrates `messages` / `customState` /
   * `artifacts`.
   */
  async continueFrom(continuationOrSnapshotId: string): Promise<void> {
    if (this.disposed) return;
    const token = continuationOrSnapshotId.startsWith(SNAP_PREFIX)
      ? continuationOrSnapshotId
      : toContinuationToken(continuationOrSnapshotId);
    const sid = extractSnapshotId(token);
    this.mutate({
      continuationId: token,
      snapshotId: sid ?? undefined,
    });
    if (!sid) return;
    try {
      const status = await this.applySnapshot(sid);
      if (this.disposed) return;
      // continueFrom is foreground: a 'pending' snapshot still ends here,
      // since we explicitly want to "land at" this point, not subscribe
      // to its background continuation.
      if (!status && this.state.phase !== 'error') {
        this.mutate({ phase: 'done' });
      }
    } catch (e) {
      if (this.disposed) return;
      this.mutate({
        error: e instanceof Error ? e : new Error(String(e)),
      });
    }
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private mutate(patch: Partial<AgentSessionState<S>>): void {
    this.state = { ...this.state, ...patch };
    this.notify();
  }

  private notify(): void {
    if (this.disposed) return;
    for (const l of this.listeners) l();
  }

  private clearActivePoll(): void {
    if (this.pollTimeout) {
      clearTimeout(this.pollTimeout);
      this.pollTimeout = null;
    }
    this.pollGen += 1;
  }

  /**
   * Fetch `/state` for the given snapshotId and apply the result to
   * session state. Returns the snapshot's terminal status so the caller
   * can decide whether to keep polling. Sets `phase` to `'done'` on
   * `'done'`, to `'error'` on `'failed'` / `'aborted'`, and otherwise
   * leaves the phase untouched (the caller controls foreground vs.
   * background semantics).
   */
  private async applySnapshot(
    snapshotId: string
  ): Promise<'pending' | 'done' | 'failed' | 'aborted' | undefined> {
    const snap = await runFlow<any, AgentInitBody>({
      url: this.options.stateUrl ?? `${this.options.url}/state`,
      input: snapshotId,
      headers: this.options.headers,
    });
    if (this.disposed) return undefined;
    const state = snap?.state ?? {};
    const patch: Partial<AgentSessionState<S>> = {};
    if (Array.isArray(state.messages)) patch.messages = state.messages;
    if (Array.isArray(state.artifacts)) patch.artifacts = state.artifacts;
    if (state.custom !== undefined) patch.customState = state.custom as S;
    if (snap?.status === 'done') {
      patch.phase = 'done';
    } else if (snap?.status === 'failed' || snap?.status === 'aborted') {
      patch.phase = 'error';
      patch.error = new Error(snap.error?.message ?? 'background run failed');
    }
    this.mutate(patch);
    return snap?.status;
  }

  private async rehydrate(token: string): Promise<void> {
    if (this.rehydrated.has(token)) return;
    try {
      const sid = extractSnapshotId(token);
      if (!sid) return;
      const status = await this.applySnapshot(sid);
      if (this.disposed) return;
      if (status === 'pending') {
        // Resumed mid-background run — switch to polling phase.
        this.mutate({ phase: 'background' });
        this.startBackgroundPoll(sid);
      } else if (!status || status === 'done') {
        // No status (foreground turn) or already done — sit at 'done'
        // unless applySnapshot already set 'error'.
        if (this.state.phase !== 'error')
          this.mutate({ phase: 'done' });
      }
      this.rehydrated.add(token);
    } catch (e) {
      if (this.disposed) return;
      this.mutate({
        error: e instanceof Error ? e : new Error(String(e)),
      });
      this.rehydrated.add(token);
    }
  }

  private startBackgroundPoll(snapshotId: string): void {
    const gen = this.pollGen;
    const tick = async () => {
      this.pollTimeout = null;
      if (gen !== this.pollGen || this.disposed) return;
      // Self-terminate when no one's listening — adapters that unmounted
      // their component don't need us to keep hitting the network. The
      // poll resumes on a subsequent submit() / continueFrom().
      if (this.listeners.size === 0) return;
      try {
        const status = await this.applySnapshot(snapshotId);
        if (gen !== this.pollGen || this.disposed) return;
        if (status === 'done' || status === 'failed' || status === 'aborted') {
          return;
        }
        this.pollTimeout = setTimeout(tick, 2000);
      } catch (e) {
        if (gen !== this.pollGen || this.disposed) return;
        this.mutate({
          error: e instanceof Error ? e : new Error(String(e)),
          phase: 'error',
        });
      }
    };
    this.pollTimeout = setTimeout(tick, 1000);
  }

  private internalSubmit(input: AgentInputBody): void {
    this.abortController?.abort();
    const controller = new AbortController();
    this.abortController = controller;

    const newMessages = input.messages?.length
      ? [...this.state.messages, ...(input.messages as AgentMessage[])]
      : this.state.messages;

    this.mutate({
      messages: newMessages,
      streamingText: '',
      streamingReasoning: '',
      toolCalls: [],
      statusLabel: null,
      progress: null,
      error: null,
      pendingInterrupt: null,
      phase: 'streaming',
    });

    const init: AgentInitBody = this.state.continuationId
      ? { continuationId: this.state.continuationId }
      : {};

    const { output: outputPromise, stream } = streamFlow<
      AgentOutputBody<S>,
      AgentEvent,
      AgentInitBody
    >({
      url: this.options.url,
      input,
      init,
      headers: this.options.headers,
      abortSignal: controller.signal,
    });

    void (async () => {
      let interruptDetected: PendingInterrupt | null = null;
      let detachedDuringStream = false;
      try {
        for await (const event of stream) {
          walkAgentEvent(event, {
            onText: (delta) =>
              this.mutate({
                streamingText: this.state.streamingText + delta,
              }),
            onReasoning: (delta) =>
              this.mutate({
                streamingReasoning: this.state.streamingReasoning + delta,
              }),
            onToolRequest: ({ toolCallId, toolName, input }) => {
              this.mutate({
                toolCalls: upsertToolCall(this.state.toolCalls, {
                  id: toolCallId,
                  name: toolName,
                  input,
                  state: 'call',
                }),
              });
            },
            onToolResponse: ({ toolCallId, toolName, output }) => {
              this.mutate({
                toolCalls: upsertToolResponse(
                  this.state.toolCalls,
                  toolCallId,
                  toolName,
                  output
                ),
              });
            },
            onToolError: ({ toolCallId, toolName, errorText, errorCode }) => {
              this.mutate({
                toolCalls: upsertToolError(
                  this.state.toolCalls,
                  toolCallId,
                  toolName,
                  errorText,
                  errorCode
                ),
              });
            },
            onStatus: (s) => this.mutate({ statusLabel: s.label }),
            onProgress: (p) =>
              this.mutate({
                progress: {
                  current: p.current,
                  total: p.total,
                  label: p.label,
                },
              }),
            onPhase: (p) => this.mutate({ statusLabel: p }),
            onArtifact: (artifact) => {
              this.mutate({
                artifacts: [...this.state.artifacts, artifact],
              });
            },
            onInterrupt: (irpt) => {
              interruptDetected = irpt;
            },
            onDetached: ({ continuationId: cid }) => {
              detachedDuringStream = true;
              if (cid) {
                this.mutate({
                  continuationId: cid,
                  snapshotId: extractSnapshotId(cid) ?? undefined,
                });
              }
            },
            onTurnEnd: ({ continuationId: cid }) => {
              if (cid) {
                this.mutate({
                  continuationId: cid,
                  snapshotId: extractSnapshotId(cid) ?? undefined,
                });
              }
            },
          });
          if (detachedDuringStream) break;
        }

        const result = await outputPromise;

        const patch: Partial<AgentSessionState<S>> = {};
        if (result?.continuationId) {
          patch.continuationId = result.continuationId;
          patch.snapshotId =
            extractSnapshotId(result.continuationId) ?? undefined;
        }
        let nextMessages = this.state.messages;
        let nextArtifacts = this.state.artifacts;
        if (result?.message) {
          nextMessages = [...nextMessages, result.message as AgentMessage];
        }
        if (Array.isArray(result?.state?.messages)) {
          nextMessages = result.state.messages as AgentMessage[];
        }
        if (Array.isArray(result?.state?.artifacts)) {
          nextArtifacts = result.state.artifacts as unknown[];
        }
        if (Array.isArray(result?.artifacts)) {
          nextArtifacts = mergeArtifacts(
            nextArtifacts,
            result.artifacts as unknown[]
          );
        }
        if (result?.state?.custom !== undefined) {
          patch.customState = result.state.custom as S;
        }
        patch.messages = nextMessages;
        patch.artifacts = nextArtifacts;

        // Clear the in-flight buffers — the canonical, post-commit form
        // of all three lives in `messages` now. See useGenkitAgent docs
        // for the rationale.
        patch.streamingText = '';
        patch.streamingReasoning = '';
        patch.toolCalls = [];

        if (detachedDuringStream || input.detach) {
          patch.phase = 'background';
          this.mutate(patch);
          const sid = result?.continuationId
            ? extractSnapshotId(result.continuationId)
            : null;
          if (sid) this.startBackgroundPoll(sid);
          return;
        }

        if (interruptDetected) {
          patch.pendingInterrupt = interruptDetected;
          patch.phase = 'awaiting-interrupt';
          this.mutate(patch);
          return;
        }

        patch.phase = 'done';
        this.mutate(patch);
      } catch (err) {
        if (controller.signal.aborted || this.disposed) return;
        this.mutate({
          error: err instanceof Error ? err : new Error(String(err)),
          phase: 'error',
        });
      }
    })();
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeInitialState<S>(
  resumeFromContinuation: string | undefined
): AgentSessionState<S> {
  return {
    messages: [],
    artifacts: [],
    customState: undefined,
    streamingText: '',
    streamingReasoning: '',
    toolCalls: [],
    statusLabel: null,
    progress: null,
    phase: 'idle',
    error: null,
    continuationId: resumeFromContinuation,
    snapshotId: resumeFromContinuation
      ? (extractSnapshotId(resumeFromContinuation) ?? undefined)
      : undefined,
    pendingInterrupt: null,
  };
}

function normalizeResumeToken(opts: AgentSessionOptions): string | undefined {
  if (opts.resumeFromContinuation) return opts.resumeFromContinuation;
  if (opts.resumeFromSnapshotId)
    return toContinuationToken(opts.resumeFromSnapshotId);
  return undefined;
}

function upsertToolCall(
  prev: ToolCall[],
  call: {
    id: string;
    name: string;
    input: unknown;
    state: ToolCallState;
  }
): ToolCall[] {
  const next = prev.slice();
  const idx = next.findIndex((tc) => tc.id === call.id);
  if (idx === -1) {
    next.push(call);
  } else {
    next[idx] = { ...next[idx], input: call.input };
  }
  return next;
}

function upsertToolResponse(
  prev: ToolCall[],
  toolCallId: string,
  toolName: string,
  output: unknown
): ToolCall[] {
  const next = prev.slice();
  const idx = next.findIndex((tc) => tc.id === toolCallId);
  if (idx === -1) {
    next.push({
      id: toolCallId,
      name: toolName,
      input: undefined,
      output,
      state: 'result',
    });
  } else if (next[idx].state !== 'error') {
    // `tool-error` events win over a concurrent `tool-response` for the
    // same id — they carry the structured error metadata.
    next[idx] = { ...next[idx], output, state: 'result' };
  } else {
    next[idx] = { ...next[idx], output };
  }
  return next;
}

function upsertToolError(
  prev: ToolCall[],
  toolCallId: string,
  toolName: string,
  errorText: string,
  errorCode: string | undefined
): ToolCall[] {
  const next = prev.slice();
  const idx = next.findIndex((tc) => tc.id === toolCallId);
  if (idx === -1) {
    next.push({
      id: toolCallId,
      name: toolName,
      input: undefined,
      state: 'error',
      errorText,
      errorCode,
    });
  } else {
    next[idx] = {
      ...next[idx],
      state: 'error',
      errorText,
      errorCode,
    };
  }
  return next;
}

function mergeArtifacts(a: unknown[], b: unknown[]): unknown[] {
  const seen = new Set(
    a.map((x) => (x as { name?: string })?.name).filter(Boolean)
  );
  return [
    ...a,
    ...b.filter((x) => {
      const name = (x as { name?: string })?.name;
      return !name || !seen.has(name);
    }),
  ];
}

/** Continuation token format — mirrors `genkit/beta`'s prefixes. */
const SNAP_PREFIX = 'snap:';

function extractSnapshotId(continuationId?: string): string | null {
  if (!continuationId) return null;
  if (continuationId.startsWith(SNAP_PREFIX))
    return continuationId.slice(SNAP_PREFIX.length);
  return null;
}

function toContinuationToken(snapshotId: string): string {
  return SNAP_PREFIX + snapshotId;
}
