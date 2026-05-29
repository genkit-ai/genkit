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
import type { AgentContinuation, AgentEvent } from './agent-events.js';
import { walkAgentEvent } from './agent-events.js';

export type { AgentContinuation } from './agent-events.js';

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
   * If provided, the session immediately fetches the snapshot's state
   * and rehydrates from it. Accepts the structured continuation handed
   * back by a prior turn.
   */
  resumeFromContinuation?: AgentContinuation;
  /**
   * Convenience for URL-route restoration where only a raw snapshotId
   * is in the route (the common case). Equivalent to passing
   * `resumeFromContinuation: { kind: 'snapshot', snapshotId }`.
   */
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
  /** Structured continuation for this branch — pass to `continueFrom` to pick it. */
  continuation: AgentContinuation | undefined;
  /** Raw snapshotId convenience (server-stored agents only). */
  snapshotId: string | undefined;
  /** Final assistant message for this branch. */
  message: AgentMessage | undefined;
  state:
    | { messages?: AgentMessage[]; custom?: S; artifacts?: unknown[] }
    | undefined;
}

export interface AgentSessionState<S = unknown, TStatus = unknown> {
  messages: AgentMessage[];
  artifacts: unknown[];
  customState: S | undefined;
  streamingText: string;
  streamingReasoning: string;
  toolCalls: ToolCall[];
  /**
   * Latest application-defined status payload, as emitted by the agent
   * via `sendChunk({ type: 'status', status })`. Shape is per-agent;
   * consumers narrow it through the session's `<TStatus>` generic.
   */
  status: TStatus | null;
  phase: AgentPhase;
  error: Error | null;
  /** Structured continuation; round-tripped automatically, exposed for debugging. */
  continuation: AgentContinuation | undefined;
  /**
   * Derived snapshotId for server-stored agents (extracted from
   * `continuation` when `kind === 'snapshot'`). Undefined when the
   * underlying agent is stateless. Useful for URL bookmarks.
   */
  snapshotId: string | undefined;
  pendingInterrupt: PendingInterrupt | null;
}

interface AgentOutputBody<S = unknown> {
  continuation?: AgentContinuation;
  snapshotId?: string;
  message?: AgentMessage;
  artifacts?: unknown[];
  state?: { messages?: AgentMessage[]; custom?: S; artifacts?: unknown[] };
}

interface AgentInitBody {
  continuation?: AgentContinuation;
}

type Listener = () => void;

// ---------------------------------------------------------------------------
// Session class
// ---------------------------------------------------------------------------

export class AgentSession<S = unknown, TStatus = unknown> {
  private options: AgentSessionOptions;
  private state: AgentSessionState<S, TStatus>;
  private listeners = new Set<Listener>();

  private abortController: AbortController | null = null;
  // Active background-poll timeout. Cleared on reset/abort/unmount so we
  // don't leak timers or call notify() after dispose.
  private pollTimeout: ReturnType<typeof setTimeout> | null = null;
  // Generation token bumped on every reset/abort so an in-flight poll cycle
  // started before the reset stops calling notify().
  private pollGen = 0;
  // SnapshotIds we've already rehydrated this session — avoids re-fetching
  // the snapshot every time the URL pushes a new id mid-conversation.
  private rehydrated = new Set<string>();
  private disposed = false;

  constructor(options: AgentSessionOptions) {
    this.options = options;
    const initial = normalizeResumeContinuation(options);
    this.state = makeInitialState<S, TStatus>(initial);
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
  getState(): AgentSessionState<S, TStatus> {
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
    this.state = makeInitialState<S, TStatus>(undefined);
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
    const init: AgentInitBody = this.state.continuation
      ? { continuation: this.state.continuation }
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
      continuation: o?.continuation,
      snapshotId: o?.snapshotId ?? snapshotIdFromContinuation(o?.continuation),
      message: o?.message,
      state: o?.state,
    }));
  }

  /**
   * Advance the session to a specific continuation or snapshot. Accepts
   * the structured continuation (from a `runVariants` result) or a raw
   * snapshotId for URL-bookmark convenience. Fetches the snapshot's
   * state and rehydrates `messages` / `customState` / `artifacts`.
   */
  async continueFrom(
    continuationOrSnapshotId: AgentContinuation | string
  ): Promise<void> {
    if (this.disposed) return;
    const continuation: AgentContinuation =
      typeof continuationOrSnapshotId === 'string'
        ? { kind: 'snapshot', snapshotId: continuationOrSnapshotId }
        : continuationOrSnapshotId;
    const sid = snapshotIdFromContinuation(continuation);
    this.mutate({
      continuation,
      snapshotId: sid,
    });
    // For 'state' continuations there's nothing to fetch — the state is
    // already inline. Land at 'done' and let the next submit() use it.
    if (continuation.kind === 'state') {
      const state = continuation.state as {
        messages?: AgentMessage[];
        artifacts?: unknown[];
        custom?: S;
      } | undefined;
      const patch: Partial<AgentSessionState<S, TStatus>> = { phase: 'done' };
      if (Array.isArray(state?.messages)) patch.messages = state!.messages;
      if (Array.isArray(state?.artifacts)) patch.artifacts = state!.artifacts;
      if (state?.custom !== undefined) patch.customState = state.custom as S;
      this.mutate(patch);
      return;
    }
    try {
      const status = await this.applySnapshot(sid!);
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

  private mutate(patch: Partial<AgentSessionState<S, TStatus>>): void {
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
    const patch: Partial<AgentSessionState<S, TStatus>> = {};
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

  private async rehydrate(continuation: AgentContinuation): Promise<void> {
    // `state`-kind rehydration is inline — no fetch needed. The state is
    // already in the continuation; copy it into session state and land
    // at 'done'.
    if (continuation.kind === 'state') {
      const state = continuation.state as {
        messages?: AgentMessage[];
        artifacts?: unknown[];
        custom?: S;
      } | undefined;
      const patch: Partial<AgentSessionState<S, TStatus>> = { phase: 'done' };
      if (Array.isArray(state?.messages)) patch.messages = state!.messages;
      if (Array.isArray(state?.artifacts)) patch.artifacts = state!.artifacts;
      if (state?.custom !== undefined) patch.customState = state.custom as S;
      this.mutate(patch);
      return;
    }
    const sid = continuation.snapshotId;
    if (this.rehydrated.has(sid)) return;
    try {
      const status = await this.applySnapshot(sid);
      if (this.disposed) return;
      if (status === 'pending') {
        // Resumed mid-background run — switch to polling phase.
        this.mutate({ phase: 'background' });
        this.startBackgroundPoll(sid);
      } else if (!status || status === 'done') {
        // No status (foreground turn) or already done — sit at 'done'
        // unless applySnapshot already set 'error'.
        if (this.state.phase !== 'error') this.mutate({ phase: 'done' });
      }
      this.rehydrated.add(sid);
    } catch (e) {
      if (this.disposed) return;
      this.mutate({
        error: e instanceof Error ? e : new Error(String(e)),
      });
      this.rehydrated.add(sid);
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
      status: null,
      error: null,
      pendingInterrupt: null,
      phase: 'streaming',
    });

    const init: AgentInitBody = this.state.continuation
      ? { continuation: this.state.continuation }
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
            onStatus: (s) => this.mutate({ status: s as TStatus }),
            onArtifact: (artifact) => {
              this.mutate({
                artifacts: [...this.state.artifacts, artifact],
              });
            },
            onInterrupt: (irpt) => {
              interruptDetected = irpt;
            },
            onDetached: ({ continuation, snapshotId: sid }) => {
              detachedDuringStream = true;
              if (continuation) {
                this.mutate({
                  continuation,
                  snapshotId:
                    sid ?? snapshotIdFromContinuation(continuation),
                });
              }
            },
            onTurnEnd: ({ continuation, snapshotId: sid }) => {
              if (continuation) {
                this.mutate({
                  continuation,
                  snapshotId:
                    sid ?? snapshotIdFromContinuation(continuation),
                });
              }
            },
          });
          if (detachedDuringStream) break;
        }

        const result = await outputPromise;

        const patch: Partial<AgentSessionState<S, TStatus>> = {};
        if (result?.continuation) {
          patch.continuation = result.continuation;
          patch.snapshotId =
            result.snapshotId ?? snapshotIdFromContinuation(result.continuation);
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
          const sid = snapshotIdFromContinuation(result?.continuation);
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

function makeInitialState<S, TStatus>(
  resumeFromContinuation: AgentContinuation | undefined
): AgentSessionState<S, TStatus> {
  return {
    messages: [],
    artifacts: [],
    customState: undefined,
    streamingText: '',
    streamingReasoning: '',
    toolCalls: [],
    status: null,
    phase: 'idle',
    error: null,
    continuation: resumeFromContinuation,
    snapshotId: snapshotIdFromContinuation(resumeFromContinuation),
    pendingInterrupt: null,
  };
}

function normalizeResumeContinuation(
  opts: AgentSessionOptions
): AgentContinuation | undefined {
  if (opts.resumeFromContinuation) return opts.resumeFromContinuation;
  if (opts.resumeFromSnapshotId)
    return { kind: 'snapshot', snapshotId: opts.resumeFromSnapshotId };
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

/**
 * Convenience: pull the `snapshotId` out of a structured continuation,
 * or undefined if the continuation is `state`-kind / absent.
 */
function snapshotIdFromContinuation(
  continuation: AgentContinuation | undefined
): string | undefined {
  if (!continuation) return undefined;
  return continuation.kind === 'snapshot' ? continuation.snapshotId : undefined;
}
