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
 * Transport-agnostic agent client core.
 *
 * This module is browser-safe: it has **no** runtime dependency on the rest of
 * `@genkit-ai/ai` (it only imports types, which are erased at compile time) and
 * no Node-specific APIs. Both the in-process server agent (`ai.defineAgent`)
 * and the HTTP `remoteAgent` client compose the same {@link AgentChatImpl} /
 * {@link createAgentAPI} core over a transport that implements
 * {@link AgentTransport}.
 *
 * @module agent-core
 */

import type {
  AgentInit,
  AgentInput,
  AgentOutput,
  AgentStreamChunk,
} from './agent.js';
import { applyPatch, type JsonPatch } from './json-patch.js';
import type { MessageData } from './model-types.js';
import type {
  Media,
  Part,
  ToolRequestPart,
  ToolResponsePart,
} from './parts.js';
import type {
  AgentFinishReason,
  Artifact,
  SessionSnapshot,
  SessionState,
} from './session.js';

// ---------------------------------------------------------------------------
// Public interfaces
// ---------------------------------------------------------------------------

/**
 * Identifies a snapshot to load: an exact `snapshotId`, or a `sessionId` (the
 * session's latest leaf snapshot). Provide exactly one.
 */
export type SnapshotLookup = { snapshotId: string } | { sessionId: string };

/**
 * The transport-agnostic surface for talking to an agent. The same shape is
 * returned by `ai.defineAgent(...)` on the server and by `remoteAgent(...)` on
 * the client.
 */

export interface AgentAPI<State = unknown> {
  /** Starts a new chat, or attaches to one via init. */
  chat(init?: AgentInit<State>): AgentChat<State>;

  /**
   * Loads a server snapshot and returns a chat with history restored. Accepts
   * either a `snapshotId` (an exact snapshot) or a `sessionId` (the session's
   * latest snapshot).
   */
  loadChat(opts: SnapshotLookup): Promise<AgentChat<State>>;

  /**
   * Reads a snapshot without starting a chat. Requires a server store. Accepts
   * a `snapshotId` string, or a lookup object (`{ snapshotId }` /
   * `{ sessionId }`).
   */
  getSnapshot(
    lookup: string | SnapshotLookup
  ): Promise<SessionSnapshot<State> | undefined>;

  /** Aborts a running snapshot. Requires a server store. */
  abort(snapshotId: string): Promise<SessionSnapshot['status'] | undefined>;
}

/**
 * A stateful conversation with an agent. Tracks state across turns so callers
 * do not have to thread `snapshotId`/`state` by hand.
 */
export interface AgentChat<State = unknown> {
  /**
   * Runs a single turn and resolves with the completed {@link AgentResponse}.
   * The non-streaming analog of {@link generate}; for incremental chunks use
   * {@link sendStream}.
   */
  send(
    input: string | AgentInput,
    opts?: { abortSignal?: AbortSignal }
  ): Promise<AgentResponse<State>>;

  /**
   * Runs a single turn and returns an {@link AgentTurn} exposing `.stream` and
   * `.response`. The streaming analog of {@link generateStream}.
   */
  sendStream(
    input: string | AgentInput,
    opts?: { abortSignal?: AbortSignal }
  ): AgentTurn<State>;

  /** Resumes after an interrupt. Sugar for `send({ resume })`. */
  resume(
    resume: AgentInput['resume'],
    opts?: { abortSignal?: AbortSignal }
  ): Promise<AgentResponse<State>>;

  /** Streaming resume. Sugar for `sendStream({ resume })`. */
  resumeStream(
    resume: AgentInput['resume'],
    opts?: { abortSignal?: AbortSignal }
  ): AgentTurn<State>;

  /** Submits a detached (background) turn. */
  detach(input: string | AgentInput): Promise<DetachedTask<State>>;

  /** Aborts the current snapshot. */
  abort(): Promise<SessionSnapshot['status'] | undefined>;

  readonly snapshotId?: string;
  readonly state?: State;
  readonly messages: MessageData[];
  readonly artifacts: Artifact[];
}

/**
 * A single in-flight turn - the analog of `ai.generateStream`'s
 * `{ stream, response }`.
 */
export interface AgentTurn<State = unknown, O = unknown> {
  /** Chunks as the turn progresses. */
  readonly stream: AsyncIterable<AgentChunk<State>>;

  /** The completed turn, with generate-style accessors. */
  readonly response: Promise<AgentResponse<State, O>>;

  /** Aborts this in-flight turn. */
  abort(): void;
}

/**
 * The completed result of a turn. Mirrors `GenerateResponse` and adds the
 * agent fields (`snapshotId`, `state`, `artifacts`).
 */
export interface AgentResponse<State = unknown, O = unknown> {
  readonly message?: MessageData;
  readonly text: string;
  readonly reasoning: string;
  readonly media: Media | null;
  readonly data: O | null;
  readonly toolRequests: ToolRequestPart[];
  readonly interrupts: AgentInterrupt[];
  readonly messages: MessageData[];
  readonly finishReason: AgentFinishReason;
  readonly finishMessage?: string;
  readonly raw: AgentOutput<State>;
  assertValid(): void;

  readonly snapshotId?: string;
  readonly state?: State;
  readonly artifacts: Artifact[];
}

/**
 * A streamed chunk. Mirrors `GenerateResponseChunk` and adds the agent fields
 * (`artifact`, `custom`).
 */
export interface AgentChunk<State = unknown> {
  readonly text: string;
  readonly reasoning: string;
  readonly accumulatedText: string;
  readonly toolRequests: ToolRequestPart[];
  readonly data: unknown;
  readonly media: Media | null;

  readonly artifact?: Artifact;
  /**
   * The full, post-patch custom state. Present only on chunks that carry a
   * custom-state update; `undefined` on text / model chunks. Each value is a
   * fresh object reference (the client applies the streamed RFC 6902 JSON Patch
   * onto a clone), so it is safe to use for `===` change detection in UI
   * frameworks. Equivalent to the value {@link AgentChat.state} returns at the
   * moment this chunk is yielded.
   */
  readonly custom?: State;
  readonly raw: AgentStreamChunk;
}

/**
 * A single tool request a turn paused on. `respond`/`restart` are builders:
 * they return the part to put into a `resume` payload, they do not send.
 */
export interface AgentInterrupt<Input = unknown, Output = unknown> {
  name: string;
  ref?: string;
  input: Input;

  /** Builds a `respond` entry for this interrupt. Does not send. */
  respond(output: Output): ToolResponsePart;

  /** Builds a `restart` entry re-issuing the original tool request. */
  restart(): ToolRequestPart;
}

/**
 * A handle to a background (detached) task.
 */
export interface DetachedTask<State = unknown> {
  readonly snapshotId: string;

  /** Yields status until a terminal state. */
  poll(opts?: { intervalMs?: number }): AsyncIterable<SessionSnapshot<State>>;

  /** Resolves when the task reaches a terminal state. */
  wait(opts?: { intervalMs?: number }): Promise<SessionSnapshot<State>>;

  /** Aborts the task. */
  abort(): Promise<SessionSnapshot['status'] | undefined>;
}

/**
 * Thrown when a turn fails. Carries the last-good state so the session is
 * recoverable.
 */
export class AgentError<State = unknown> extends Error {
  readonly status: string;
  readonly details?: unknown;
  readonly state?: State;
  readonly snapshotId?: string;
  readonly response: AgentResponse<State>;

  constructor(opts: {
    message: string;
    status: string;
    details?: unknown;
    state?: State;
    snapshotId?: string;
    response: AgentResponse<State>;
  }) {
    super(opts.message);
    this.name = 'AgentError';
    this.status = opts.status;
    this.details = opts.details;
    this.state = opts.state;
    this.snapshotId = opts.snapshotId;
    this.response = opts.response;
    // Restore prototype chain for `instanceof` across transpilation targets.
    Object.setPrototypeOf(this, AgentError.prototype);
  }
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/**
 * The pluggable backend the agent-client core runs over. Implementations exist
 * for the in-process server agent (driving the agent action directly) and for
 * the HTTP `remoteAgent` (driving `streamFlow`/`runFlow`).
 */
export interface AgentTransport {
  /** Declares server- vs client-managed state; auto-detected when omitted. */
  stateManagement?: 'server' | 'client';

  /**
   * Runs a single turn. Returns the streamed chunks plus a promise for the
   * final, non-throwing {@link AgentOutput} (failures resolve with
   * `finishReason: 'failed'`).
   */
  runTurn(
    input: AgentInput,
    init: AgentInit,
    opts: { abortSignal: AbortSignal }
  ): {
    stream: AsyncIterable<AgentStreamChunk>;
    output: Promise<AgentOutput>;
  };

  /**
   * Runs a single turn without streaming and resolves with the final,
   * non-throwing {@link AgentOutput} (failures resolve with
   * `finishReason: 'failed'`). Optional: when omitted, the core falls back to
   * consuming {@link runTurn}'s `output`.
   */
  run?(
    input: AgentInput,
    init: AgentInit,
    opts: { abortSignal: AbortSignal }
  ): Promise<AgentOutput>;

  /** Reads a snapshot. Requires a server store. */
  getSnapshot(
    lookup: SnapshotLookup
  ): Promise<SessionSnapshot<any> | undefined>;

  /** Aborts a running snapshot. Requires a server store. */
  abort(snapshotId: string): Promise<SessionSnapshot['status'] | undefined>;
}

const TERMINAL_STATUSES = new Set(['done', 'failed', 'aborted']);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function toAgentInput(input: string | AgentInput): AgentInput {
  if (typeof input === 'string') {
    return { messages: [{ role: 'user', content: [{ text: input }] }] };
  }
  return input;
}

// ── Message-derived accessor helpers (mirroring generate) ──────────────────

function partsText(parts?: Part[]): string {
  return (parts ?? []).map((p) => p.text ?? '').join('');
}

function partsReasoning(parts?: Part[]): string {
  return (parts ?? [])
    .filter((p) => p.reasoning !== undefined)
    .map((p) => p.reasoning ?? '')
    .join('');
}

function firstMedia(parts?: Part[]): Media | null {
  const p = (parts ?? []).find((p) => !!p.media);
  return p?.media
    ? { url: p.media.url, contentType: p.media.contentType }
    : null;
}

function firstData(parts?: Part[]): any {
  const p = (parts ?? []).find((p) => p.data !== undefined);
  return p?.data ?? null;
}

function toolRequestParts(parts?: Part[]): ToolRequestPart[] {
  return (parts ?? []).filter((p) => !!p.toolRequest) as ToolRequestPart[];
}

// ---------------------------------------------------------------------------
// AgentInterrupt
// ---------------------------------------------------------------------------

class AgentInterruptImpl<Input = unknown, Output = unknown>
  implements AgentInterrupt<Input, Output>
{
  readonly name: string;
  readonly ref?: string;
  readonly input: Input;

  constructor(part: ToolRequestPart) {
    this.name = part.toolRequest.name;
    this.ref = part.toolRequest.ref;
    this.input = part.toolRequest.input as Input;
  }

  respond(output: Output): ToolResponsePart {
    return {
      toolResponse: {
        name: this.name,
        ...(this.ref !== undefined && { ref: this.ref }),
        output,
      },
    } as ToolResponsePart;
  }

  restart(): ToolRequestPart {
    return {
      toolRequest: {
        name: this.name,
        ...(this.ref !== undefined && { ref: this.ref }),
        input: this.input,
      },
    } as ToolRequestPart;
  }
}

// ---------------------------------------------------------------------------
// AgentResponse
// ---------------------------------------------------------------------------

class AgentResponseImpl<State = unknown, O = unknown>
  implements AgentResponse<State, O>
{
  constructor(
    private readonly _raw: AgentOutput<State>,
    private readonly _messages: MessageData[]
  ) {}

  get message(): MessageData | undefined {
    return this._raw.message;
  }

  get text(): string {
    return partsText(this._raw.message?.content);
  }

  get reasoning(): string {
    return partsReasoning(this._raw.message?.content);
  }

  get media(): Media | null {
    return firstMedia(this._raw.message?.content);
  }

  get data(): O | null {
    return firstData(this._raw.message?.content) as O | null;
  }

  get toolRequests(): ToolRequestPart[] {
    return toolRequestParts(this._raw.message?.content);
  }

  get interrupts(): AgentInterrupt[] {
    return this.toolRequests
      .filter((p) => !!(p as any).metadata?.interrupt)
      .map((p) => new AgentInterruptImpl(p));
  }

  get messages(): MessageData[] {
    return this._messages;
  }

  get finishReason(): AgentFinishReason {
    return (this._raw.finishReason ?? 'unknown') as AgentFinishReason;
  }

  get finishMessage(): string | undefined {
    return this._raw.error?.message;
  }

  get raw(): AgentOutput<State> {
    return this._raw;
  }

  get snapshotId(): string | undefined {
    return this._raw.snapshotId;
  }

  get state(): State | undefined {
    return this._raw.state?.custom as State | undefined;
  }

  get artifacts(): Artifact[] {
    return this._raw.artifacts ?? this._raw.state?.artifacts ?? [];
  }

  assertValid(): void {
    if (this.finishReason === 'blocked') {
      throw new Error(
        `Generation blocked${this.finishMessage ? `: ${this.finishMessage}` : ''}.`
      );
    }
    if (!this._raw.message) {
      throw new Error('Agent response has no message.');
    }
  }
}

// ---------------------------------------------------------------------------
// AgentChunk
// ---------------------------------------------------------------------------

class AgentChunkImpl<State = unknown> implements AgentChunk<State> {
  /**
   * The post-patch custom state, set by the {@link AgentChatImpl.sendStream}
   * generator after it applies this chunk's `customPatch`. Left `undefined` on
   * chunks that do not carry a custom-state update.
   */
  private _custom?: State;

  constructor(
    private readonly _raw: AgentStreamChunk,
    private readonly _previousText: string
  ) {}

  /** Internal: records the post-patch custom state this chunk reports. */
  _setCustom(custom: State | undefined): void {
    this._custom = custom;
  }

  private get _content(): Part[] | undefined {
    return this._raw.modelChunk?.content as Part[] | undefined;
  }

  get text(): string {
    return partsText(this._content);
  }

  get reasoning(): string {
    return partsReasoning(this._content);
  }

  get accumulatedText(): string {
    return this._previousText + this.text;
  }

  get toolRequests(): ToolRequestPart[] {
    return toolRequestParts(this._content);
  }

  get data(): unknown {
    return firstData(this._content);
  }

  get media(): Media | null {
    return firstMedia(this._content);
  }

  get artifact(): Artifact | undefined {
    return this._raw.artifact;
  }

  get custom(): State | undefined {
    return this._custom;
  }

  get raw(): AgentStreamChunk {
    return this._raw;
  }
}

// ---------------------------------------------------------------------------
// DetachedTask
// ---------------------------------------------------------------------------

class DetachedTaskImpl<State = unknown> implements DetachedTask<State> {
  constructor(
    readonly snapshotId: string,
    private readonly transport: AgentTransport
  ) {}

  async *poll(opts?: {
    intervalMs?: number;
  }): AsyncIterable<SessionSnapshot<State>> {
    const intervalMs = opts?.intervalMs ?? 1000;
    while (true) {
      const snap = (await this.transport.getSnapshot({
        snapshotId: this.snapshotId,
      })) as SessionSnapshot<State> | undefined;

      if (snap) {
        yield snap;
        if (snap.status && TERMINAL_STATUSES.has(snap.status)) {
          return;
        }
      }
      await sleep(intervalMs);
    }
  }

  async wait(opts?: { intervalMs?: number }): Promise<SessionSnapshot<State>> {
    let last: SessionSnapshot<State> | undefined;
    for await (const snap of this.poll(opts)) {
      last = snap;
    }
    if (!last) {
      throw new Error(
        `Detached task ${this.snapshotId} did not produce a snapshot.`
      );
    }
    return last;
  }

  abort(): Promise<SessionSnapshot['status'] | undefined> {
    return this.transport.abort(this.snapshotId);
  }
}

// ---------------------------------------------------------------------------
// AgentChat
// ---------------------------------------------------------------------------

export class AgentChatImpl<State = unknown> implements AgentChat<State> {
  snapshotId?: string;
  messages: MessageData[] = [];
  artifacts: Artifact[] = [];

  private clientState?: SessionState<State>;

  constructor(
    private readonly transport: AgentTransport,
    private readonly connectInit?: AgentInit<State>
  ) {
    if (connectInit?.snapshotId) {
      this.snapshotId = connectInit.snapshotId;
    }
    if (connectInit?.state) {
      this.hydrateFromState(connectInit.state);
    }
  }

  get state(): State | undefined {
    return this.clientState?.custom as State | undefined;
  }

  /**
   * Replaces the tracked `clientState`/`messages`/`artifacts` aggregates with
   * (copies of) those carried by a session state.
   */
  private hydrateFromState(state: SessionState<State>): void {
    this.clientState = state;
    this.messages = state?.messages ? [...state.messages] : [];
    this.artifacts = state?.artifacts ? [...state.artifacts] : [];
  }

  /** Loads aggregates from a server snapshot (used by `loadChat`). */
  _loadFromSnapshot(snapshot: SessionSnapshot<State>): void {
    this.snapshotId = snapshot.snapshotId;
    this.hydrateFromState(snapshot.state);
  }

  /**
   * Builds the init for the next turn from tracked aggregates. Always returns
   * an object (never `undefined`) because the agent validates `init` against
   * `AgentInitSchema` - an empty object is the valid "fresh session" init.
   */
  private buildInit(): AgentInit<State> {
    if (this.snapshotId) {
      return { snapshotId: this.snapshotId };
    }
    if (this.clientState) {
      return { state: this.clientState };
    }
    return this.connectInit ?? {};
  }

  /** Applies a completed turn's output to the running aggregates. */
  private applyOutput(raw: AgentOutput<State>): void {
    if (raw.snapshotId !== undefined) {
      this.snapshotId = raw.snapshotId;
    }
    if (raw.state !== undefined) {
      this.clientState = raw.state;
    }
    if (this.transport.stateManagement === undefined) {
      if (raw.snapshotId !== undefined) {
        this.transport.stateManagement = 'server';
      } else if (raw.state !== undefined) {
        this.transport.stateManagement = 'client';
      }
    }
    if (raw.state?.messages !== undefined) {
      this.messages = [...raw.state.messages];
    } else if (raw.message) {
      this.messages.push(raw.message);
    }
    if (raw.state?.artifacts !== undefined) {
      this.artifacts = [...raw.state.artifacts];
    } else if (raw.artifacts?.length) {
      for (const a of raw.artifacts) {
        const idx = a.name
          ? this.artifacts.findIndex((x) => x.name === a.name)
          : -1;
        if (idx >= 0) {
          this.artifacts[idx] = a;
        } else {
          this.artifacts.push(a);
        }
      }
    }
  }

  /**
   * Wires an optional external abort signal to a fresh {@link AbortController}
   * and returns it alongside a getter for whether the turn was aborted.
   */
  private setupAbort(opts?: { abortSignal?: AbortSignal }): {
    controller: AbortController;
    isAborted: () => boolean;
  } {
    const controller = new AbortController();
    let aborted = false;
    if (opts?.abortSignal) {
      if (opts.abortSignal.aborted) {
        controller.abort();
        aborted = true;
      } else {
        opts.abortSignal.addEventListener('abort', () => {
          aborted = true;
          controller.abort();
        });
      }
    }
    return {
      controller,
      isAborted: () => aborted || controller.signal.aborted,
    };
  }

  /**
   * Resolves a turn's raw `output` into an {@link AgentResponse}, applying it to
   * the running aggregates and throwing an {@link AgentError} on a failed turn.
   * Aborted turns resolve to a synthetic `aborted` response. Shared by both
   * {@link send} and {@link sendStream}.
   */
  private buildResponse(
    output: Promise<AgentOutput>,
    isAborted: () => boolean
  ): Promise<AgentResponse<State>> {
    return (async (): Promise<AgentResponse<State>> => {
      let raw: AgentOutput<State>;
      try {
        raw = (await output) as AgentOutput<State>;
      } catch (e) {
        if (isAborted()) {
          raw = { finishReason: 'aborted' as AgentFinishReason };
        } else {
          throw this.toAgentError(e);
        }
      }
      this.applyOutput(raw);
      const response = new AgentResponseImpl<State>(raw, [...this.messages]);
      if (raw.finishReason === 'failed') {
        throw new AgentError<State>({
          message: raw.error?.message ?? 'Agent turn failed.',
          status: raw.error?.status ?? 'UNKNOWN',
          details: raw.error?.details,
          state: raw.state?.custom as State | undefined,
          snapshotId: raw.snapshotId,
          response,
        });
      }
      return response;
    })();
  }

  async send(
    input: string | AgentInput,
    opts?: { abortSignal?: AbortSignal }
  ): Promise<AgentResponse<State>> {
    // Prefer a non-streaming transport call when available; otherwise fall
    // back to running the streaming path and awaiting its response.
    if (this.transport.run) {
      const agentInput = toAgentInput(input);
      if (agentInput.messages?.length) {
        this.messages.push(...agentInput.messages);
      }
      const init = this.buildInit();
      const { controller, isAborted } = this.setupAbort(opts);
      const output = this.transport.run(agentInput, init, {
        abortSignal: controller.signal,
      });
      return this.buildResponse(output, isAborted);
    }
    return this.sendStream(input, opts).response;
  }

  sendStream(
    input: string | AgentInput,
    opts?: { abortSignal?: AbortSignal }
  ): AgentTurn<State> {
    const agentInput = toAgentInput(input);
    if (agentInput.messages?.length) {
      this.messages.push(...agentInput.messages);
    }
    const init = this.buildInit();

    const { controller, isAborted } = this.setupAbort(opts);

    const { stream: rawStream, output } = this.transport.runTurn(
      agentInput,
      init,
      { abortSignal: controller.signal }
    );

    const responsePromise = this.buildResponse(output, isAborted);
    // Avoid unhandled-rejection warnings when only the stream is consumed.
    responsePromise.catch(() => {});

    const self = this;
    const stream = (async function* (): AsyncIterable<AgentChunk<State>> {
      let previousText = '';
      try {
        for await (const raw of rawStream) {
          const chunk = new AgentChunkImpl<State>(raw, previousText);
          previousText = chunk.accumulatedText;
          // Keep the locally tracked custom state live mid-stream by applying
          // each streamed JSON Patch to it, then surface the resulting
          // post-patch custom state on the chunk as `chunk.custom` so consumers
          // get an explicit change notification (with a fresh object reference
          // for `===`-based change detection). The first patch of a turn is a
          // whole-document replace that re-bases us onto the server baseline.
          if (raw.customPatch) {
            self.applyCustomPatch(raw.customPatch);
            chunk._setCustom(self.state);
          }
          yield chunk;
        }
      } catch (e) {
        if (!isAborted()) {
          // Surface a failed turn / transport error as an AgentError.
          await responsePromise;
          throw self.toAgentError(e);
        }
      }
      // Re-surface a failed turn (which resolves the wire, but rejects here).
      await responsePromise;
    })();

    return {
      stream,
      response: responsePromise as Promise<AgentResponse<State, unknown>>,
      abort() {
        controller.abort();
      },
    };
  }

  resume(
    resume: AgentInput['resume'],
    opts?: { abortSignal?: AbortSignal }
  ): Promise<AgentResponse<State>> {
    return this.send({ resume }, opts);
  }

  resumeStream(
    resume: AgentInput['resume'],
    opts?: { abortSignal?: AbortSignal }
  ): AgentTurn<State> {
    return this.sendStream({ resume }, opts);
  }

  /**
   * Applies a streamed RFC 6902 JSON Patch to the locally tracked custom
   * state, keeping {@link state} live as the turn streams. The first patch of
   * a turn is a whole-document replace (rooted at `""`) that re-bases the
   * client onto the server's current baseline.
   */
  private applyCustomPatch(patch: JsonPatch): void {
    const current = this.clientState?.custom;
    const next = applyPatch(current, patch) as State;
    if (this.clientState) {
      this.clientState = { ...this.clientState, custom: next };
    } else {
      this.clientState = { custom: next } as SessionState<State>;
    }
  }

  async detach(input: string | AgentInput): Promise<DetachedTask<State>> {
    const agentInput: AgentInput = { ...toAgentInput(input), detach: true };
    if (agentInput.messages?.length) {
      this.messages.push(...agentInput.messages);
    }
    const init = this.buildInit();
    const controller = new AbortController();
    const { output } = this.transport.runTurn(agentInput, init, {
      abortSignal: controller.signal,
    });
    const raw = (await output) as AgentOutput<State>;
    this.applyOutput(raw);
    if (!raw.snapshotId) {
      throw new Error('detach did not return a snapshotId.');
    }
    return new DetachedTaskImpl<State>(raw.snapshotId, this.transport);
  }

  async abort(): Promise<SessionSnapshot['status'] | undefined> {
    if (!this.snapshotId) {
      return undefined;
    }
    return this.transport.abort(this.snapshotId);
  }

  private toAgentError(e: unknown): AgentError<State> {
    if (e instanceof AgentError) {
      return e;
    }
    const message = e instanceof Error ? e.message : String(e);
    const match = /^([A-Z_]+):/.exec(message);
    const status = match ? match[1] : 'UNKNOWN';
    const raw: AgentOutput<State> = {
      finishReason: 'failed' as AgentFinishReason,
      error: { status, message },
    };
    const response = new AgentResponseImpl<State>(raw, [...this.messages]);
    return new AgentError<State>({
      message,
      status,
      details: e,
      state: this.clientState?.custom as State | undefined,
      snapshotId: this.snapshotId,
      response,
    });
  }
}

// ---------------------------------------------------------------------------
// createAgentAPI - builds the AgentAPI surface over any transport.
// ---------------------------------------------------------------------------

/**
 * Composes the {@link AgentAPI} surface (`chat`/`loadChat`/`getSnapshot`/
 * `abort`) over a {@link AgentTransport}. Shared by the in-process server agent
 * and the HTTP `remoteAgent`.
 */
export function createAgentAPI<State = unknown>(
  transport: AgentTransport
): AgentAPI<State> {
  return {
    chat(init?: AgentInit<State>): AgentChat<State> {
      return new AgentChatImpl<State>(transport, init);
    },

    async loadChat(opts: SnapshotLookup): Promise<AgentChat<State>> {
      const snapshot = (await transport.getSnapshot(opts)) as
        | SessionSnapshot<State>
        | undefined;
      if (!snapshot) {
        const id =
          'snapshotId' in opts ? opts.snapshotId : `session ${opts.sessionId}`;
        throw new Error(`Snapshot ${id} not found.`);
      }
      const chat = new AgentChatImpl<State>(transport);
      chat._loadFromSnapshot(snapshot);
      return chat;
    },

    getSnapshot(
      lookup: string | SnapshotLookup
    ): Promise<SessionSnapshot<State> | undefined> {
      const normalized: SnapshotLookup =
        typeof lookup === 'string' ? { snapshotId: lookup } : lookup;
      return transport.getSnapshot(normalized) as Promise<
        SessionSnapshot<State> | undefined
      >;
    },

    abort(snapshotId: string): Promise<SessionSnapshot['status'] | undefined> {
      return transport.abort(snapshotId);
    },
  };
}
