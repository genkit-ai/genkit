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
import { parseSchema, toJsonSchema } from '@genkit-ai/core/schema';
import { setCustomMetadataAttributes } from '@genkit-ai/core/tracing';
import { generateStream } from './generate.js';
import {
  MessageData,
  MessageSchema,
  ModelResponseChunkSchema,
} from './model-types.js';
import {
  ToolRequestPartSchema,
  ToolResponsePartSchema,
  type ToolRequestPart,
  type ToolResponsePart,
} from './parts.js';
import {
  definePrompt,
  type PromptAction,
  type PromptConfig,
} from './prompt.js';
import {
  Artifact,
  ArtifactSchema,
  InMemorySessionStore,
  Session,
  SessionSnapshot,
  SessionState,
  SessionStateSchema,
  SessionStore,
  SnapshotCallback,
  runWithSession,
  type SessionSnapshotInput,
  type SessionStoreOptions,
} from './session.js';

/**
 * Schema for initializing an agent turn.
 *
 * **Clients** pass `continuationId` — an opaque token from a prior turn's
 * `output.continuationId`. **Server-side direct callers** (tests,
 * sub-agent middleware, `agent.run()` invocations) may pass `snapshotId`
 * directly as a convenience for stored agents. The two are equivalent;
 * if both are supplied, `continuationId` wins.
 */
export const AgentInitSchema = z.object({
  continuationId: z.string().optional(),
  /**
   * Server-side convenience: raw snapshotId for stored agents. Clients
   * should round-trip `continuationId` instead, which works regardless of
   * whether the underlying agent is stored or stateless.
   */
  snapshotId: z.string().optional(),
});

/**
 * Initialization options for an agent turn.
 */
export interface AgentInit<S = unknown> {
  /** Opaque continuation token from a prior turn's `output.continuationId`. */
  continuationId?: string;
  /** Server-side convenience: raw snapshotId for stored agents. */
  snapshotId?: string;
  /** @internal — seeded by the runtime. */
  newSnapshotId?: string;
  /** @internal — populated server-side after continuationId decode. */
  _decodedState?: SessionState<S>;
  /** @internal — populated server-side after continuationId decode. */
  _decodedSnapshotId?: string;
}

/**
 * Continuation token format. The token is opaque to clients; the server
 * uses the prefix to pick the decode path:
 *
 *   `snap:<snapshotId>`             — server-stored agents
 *   `state:<base64(JSON(state))>`   — client-stored agents
 *
 * The prefix is a discriminator, not a version. If the format evolves, the
 * server detects the new shape at decode time.
 */
const CONT_SNAP_PREFIX = 'snap:';
const CONT_STATE_PREFIX = 'state:';

/** Encode a snapshotId as a continuation token. */
export function encodeSnapshotContinuation(snapshotId: string): string {
  return CONT_SNAP_PREFIX + snapshotId;
}

/** Encode a client-side state blob as a continuation token. */
export function encodeStateContinuation(state: unknown): string {
  const json = JSON.stringify(state);
  if (typeof Buffer !== 'undefined') {
    return CONT_STATE_PREFIX + Buffer.from(json, 'utf8').toString('base64');
  }
  return CONT_STATE_PREFIX + btoa(unescape(encodeURIComponent(json)));
}

/**
 * Decode a continuation token into either a snapshotId or a state blob.
 * Returns null if the token is malformed.
 */
export function decodeContinuation(
  token: string
):
  | { kind: 'snapshot'; snapshotId: string }
  | { kind: 'state'; state: unknown }
  | null {
  if (token.startsWith(CONT_SNAP_PREFIX)) {
    return {
      kind: 'snapshot',
      snapshotId: token.slice(CONT_SNAP_PREFIX.length),
    };
  }
  if (token.startsWith(CONT_STATE_PREFIX)) {
    try {
      const b64 = token.slice(CONT_STATE_PREFIX.length);
      const json =
        typeof Buffer !== 'undefined'
          ? Buffer.from(b64, 'base64').toString('utf8')
          : decodeURIComponent(escape(atob(b64)));
      return { kind: 'state', state: JSON.parse(json) };
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Server-internal convenience: extract the snapshotId from a snap-token,
 * or return undefined if the token is state-shaped or malformed. Useful
 * for code that needs to call `agent.getSnapshotData(...)` or compose
 * filesystem paths from a continuationId received from a client.
 */
export function continuationToSnapshotId(token?: string): string | undefined {
  if (!token) return undefined;
  const decoded = decodeContinuation(token);
  return decoded?.kind === 'snapshot' ? decoded.snapshotId : undefined;
}

/**
 * Detect whether a toolResponse part represents a tool execution failure.
 * The runtime uses this to surface a typed `tool-error` event alongside
 * the model-chunk that already carries the response.
 *
 * Recognized signals (in priority order):
 *   1. `metadata.toolError` — explicit marker set by middleware that
 *      catches thrown tool errors (e.g. the filesystem middleware).
 *   2. Output is an object with `{ error: ... }` or `{ status: 'error', ... }`.
 *   3. Output is a string starting with `Tool '<name>' failed:` — the
 *      format produced by stock genkit middleware.
 *
 * Returns the extracted error info, or null if the response is a success.
 */
function detectToolError(tr: {
  name: string;
  output?: unknown;
  metadata?: Record<string, unknown>;
}): { errorText: string; errorCode?: string; details?: unknown } | null {
  if (tr.metadata?.toolError) {
    const m = tr.metadata.toolError as
      | { message?: string; code?: string; details?: unknown }
      | true;
    if (m === true) {
      return { errorText: tr.name + ' failed' };
    }
    return {
      errorText: m.message ?? tr.name + ' failed',
      ...(m.code && { errorCode: m.code }),
      ...(m.details !== undefined && { details: m.details }),
    };
  }
  const out = tr.output;
  if (out && typeof out === 'object') {
    const o = out as Record<string, unknown>;
    if (typeof o.error === 'string') {
      return { errorText: o.error };
    }
    if (o.status === 'error' && typeof o.message === 'string') {
      return { errorText: o.message };
    }
  }
  if (typeof out === 'string' && out.startsWith(`Tool '${tr.name}' failed:`)) {
    return {
      errorText: out.slice(`Tool '${tr.name}' failed:`.length).trim(),
    };
  }
  return null;
}

/**
 * Schema for agent input messages and commands.
 */
export const AgentInputSchema = z.object({
  messages: z.array(MessageSchema).optional(),
  /** Options for resuming an interrupted generation. */
  resume: z
    .object({
      respond: z.array(ToolResponsePartSchema).optional(),
      restart: z.array(ToolRequestPartSchema).optional(),
    })
    .optional(),
  detach: z.boolean().optional(),
});

/**
 * Input received by an agent turn.
 */
export type AgentInput = z.infer<typeof AgentInputSchema>;

/**
 * Schema identifying a turn termination event.
 */
export const TurnEndSchema = z.object({
  snapshotId: z.string().optional(),
});

/**
 * Identifies a turn termination event.
 */
export type TurnEnd = z.infer<typeof TurnEndSchema>;

// ---------------------------------------------------------------------------
// AgentEvent — discriminated event stream
// ---------------------------------------------------------------------------
//
// The agent stream is a tagged discriminated union. Every chunk carries a
// `type` discriminator, so consumers can `switch (chunk.type) { ... }`
// exhaustively. New event types (interrupts, detach, sub-agent delegation,
// streaming artifacts) extend the union without competing for field
// namespace.

/** Typed status events with a consistent shape across all agents. */
export const StatusEventSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('status'),
    label: z.string(),
    key: z.string().optional(),
  }),
  z.object({
    type: z.literal('progress'),
    label: z.string().optional(),
    current: z.number(),
    total: z.number(),
  }),
  z.object({
    type: z.literal('phase'),
    phase: z.string(),
  }),
]);
export type StatusEvent = z.infer<typeof StatusEventSchema>;

/**
 * Discriminated union of every event an agent may emit during a turn.
 *
 * Generic agent UIs can render every event type. Specialized UIs can
 * exhaustively switch on `chunk.type`.
 */
export const AgentStreamChunkSchema = z.discriminatedUnion('type', [
  // Model-generated content (text, reasoning, tool requests).
  z.object({
    type: z.literal('model-chunk'),
    chunk: ModelResponseChunkSchema,
  }),
  // Agent-emitted status, progress, or phase updates.
  z.object({
    type: z.literal('status'),
    label: z.string(),
    key: z.string().optional(),
  }),
  z.object({
    type: z.literal('progress'),
    label: z.string().optional(),
    current: z.number(),
    total: z.number(),
  }),
  z.object({
    type: z.literal('phase'),
    phase: z.string(),
  }),
  // Artifact lifecycle. `artifact-emitted` covers the one-shot case;
  // `artifact-start`/`artifact-delta`/`artifact-complete` cover streaming
  // (schema reserved; runtime emits `artifact-emitted` for now).
  z.object({
    type: z.literal('artifact-emitted'),
    artifact: ArtifactSchema,
  }),
  z.object({
    type: z.literal('artifact-start'),
    id: z.string(),
    name: z.string(),
    mediaType: z.string().optional(),
  }),
  z.object({
    type: z.literal('artifact-delta'),
    id: z.string(),
    delta: z.any(),
  }),
  z.object({
    type: z.literal('artifact-complete'),
    id: z.string(),
  }),
  // Snapshot/continuation lifecycle.
  z.object({
    type: z.literal('snapshot'),
    snapshotId: z.string(),
    continuationId: z.string(),
  }),
  // Interrupts — explicit in-stream event with addressable refs.
  z.object({
    type: z.literal('interrupt'),
    toolCallId: z.string(),
    toolName: z.string(),
    input: z.unknown(),
    kind: z.enum(['respond', 'restart']),
    metadata: z.unknown().optional(),
  }),
  // Tool execution failure — addressable counterpart to `interrupt`. Fires
  // when a tool's handler threw (or returned an error-shaped output) and
  // the runtime caught it. The error is still surfaced to the model via
  // a toolResponse so the conversation can continue; this event lets the
  // client mark the corresponding in-flight tool call as failed without
  // string-matching the toolResponse output.
  z.object({
    type: z.literal('tool-error'),
    toolCallId: z.string(),
    toolName: z.string(),
    errorText: z.string(),
    errorCode: z.string().optional(),
    details: z.unknown().optional(),
  }),
  // Background-execution transition.
  z.object({
    type: z.literal('detached'),
    snapshotId: z.string(),
    continuationId: z.string(),
  }),
  // Turn boundary — last event emitted for a foreground turn.
  z.object({
    type: z.literal('turn-end'),
    snapshotId: z.string().optional(),
    continuationId: z.string().optional(),
  }),
  // Terminal error.
  z.object({
    type: z.literal('error'),
    errorText: z.string(),
  }),
]);

/** Backwards-compatible alias — was the wrapper schema, now is the union. */
export const AgentEventSchema = AgentStreamChunkSchema;
export type AgentEvent = z.infer<typeof AgentEventSchema>;

/**
 * Streamed chunk emitted during agent execution. The `Stream` type
 * parameter is reserved for future use (typing custom data-* events);
 * the public discriminated union is fixed across agents.
 */
export type AgentStreamChunk<_Stream = unknown> = AgentEvent;

/**
 * Schema for final results of an agent execution.
 */
export const AgentResultSchema = z.object({
  message: MessageSchema.optional(),
  artifacts: z.array(ArtifactSchema).optional(),
});

/**
 * Result returned upon completing an agent execution.
 */
export type AgentResult = z.infer<typeof AgentResultSchema>;

/**
 * Schema for output returned at turn completion.
 */
export const AgentOutputSchema = z.object({
  /**
   * Opaque continuation token to round-trip on the next turn. **Clients**
   * pass back as `init.continuationId`. Always populated.
   */
  continuationId: z.string().optional(),
  /**
   * Server-side convenience: raw snapshotId for stored agents. Populated
   * alongside `continuationId`. Useful for direct `.getSnapshotData(...)`
   * calls, filesystem-snapshot inspection, sub-agent middleware, and URL
   * bookmarks. **Clients** should round-trip `continuationId`, not this.
   */
  snapshotId: z.string().optional(),
  /** The model's final message for this turn. */
  message: MessageSchema.optional(),
  /**
   * Artifacts produced during this turn. Also surfaced via `artifact-*`
   * events on the stream; this is the cumulative snapshot.
   */
  artifacts: z.array(ArtifactSchema).optional(),
  /**
   * Read-only snapshot of session state. Includes `messages` (full
   * conversation history) and `custom` (typed custom state). Clients
   * should not send this back — round-trip the `continuationId` instead.
   */
  state: SessionStateSchema.optional(),
});

/**
 * Output returned at turn completion.
 */
export interface AgentOutput<S = unknown> {
  /** Opaque continuation token; clients pass back as `init.continuationId`. */
  continuationId?: string;
  /** Server-side convenience: raw snapshotId for stored agents. */
  snapshotId?: string;
  message?: MessageData;
  artifacts?: Artifact[];
  /** Read-only session state snapshot (messages, custom, artifacts). */
  state?: SessionState<S>;
}

/**
 * Executor responsible for running turns over input streams and persisting state.
 */
export class SessionRunner<State = unknown> {
  readonly session: Session<State>;
  readonly inputCh: AsyncIterable<AgentInput>;
  turnIndex: number = 0;
  public onEndTurn?: (snapshotId?: string) => void;
  public onDetach?: (snapshotId: string) => void;
  public newSnapshotId?: string;
  private snapshotCallback?: SnapshotCallback<State>;
  private lastSnapshot?: SessionSnapshot<State>;

  private lastSnapshotVersion: number = 0;
  private store?: SessionStore<State>;
  public isDetached: boolean = false;

  constructor(
    session: Session<State>,
    inputCh: AsyncIterable<AgentInput>,
    options?: {
      snapshotCallback?: SnapshotCallback<State>;
      lastSnapshot?: SessionSnapshot<State>;
      store?: SessionStore<State>;
      onEndTurn?: (snapshotId?: string) => void;
      onDetach?: (snapshotId: string) => void;
      newSnapshotId?: string;
    }
  ) {
    this.session = session;
    this.inputCh = inputCh;

    this.snapshotCallback = options?.snapshotCallback;
    this.lastSnapshot = options?.lastSnapshot;
    this.store = options?.store;
    this.onEndTurn = options?.onEndTurn;
    this.onDetach = options?.onDetach;
    this.newSnapshotId = options?.newSnapshotId;
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

  /**
   * Executes the flow handler against incoming input messages sequentially.
   */
  async run(fn: (input: AgentInput) => Promise<void>): Promise<void> {
    for await (const input of this.inputCh) {
      if (input.messages) {
        this.session.addMessages(input.messages);
      }

      const turnSnapshotId = this.newSnapshotId;
      this.newSnapshotId = undefined;

      try {
        await run(`runTurn-${this.turnIndex + 1}`, input, async () => {
          await fn(input);

          const snapshotId = await this.maybeSnapshot(
            'turnEnd',
            'done',
            undefined,
            turnSnapshotId
          );
          try {
            if (this.onEndTurn) {
              this.onEndTurn(snapshotId);
            }
          } catch (e) {
            // Stream was closed, absorb exception
          }
          return {
            lastSnapshot: this.lastSnapshot,
          };
        });
        this.turnIndex++;
      } catch (e: any) {
        const errStatus = e.status || 'INTERNAL';
        const errMessage = e.message || 'Internal failure';
        const errDetails = e.detail || e.details || e;
        const snapshotId = await this.maybeSnapshot(
          'turnEnd',
          'failed',
          {
            status: errStatus,
            message: errMessage,
            details: errDetails,
          },
          turnSnapshotId
        );
        try {
          if (this.onEndTurn) {
            this.onEndTurn(snapshotId);
          }
        } catch (_) {
          // Stream was closed, absorb exception
        }
        throw e;
      }
    }
  }

  /**
   * Evaluates whether to save a snapshot to the persistent store.
   *
   * Uses the mutator-based `saveSnapshot` to atomically check that the
   * snapshot has not been concurrently aborted before writing — preventing
   * a race where a "done" write could overwrite a concurrent "aborted"
   * status.
   */
  async maybeSnapshot(
    event: 'turnEnd' | 'invocationEnd',
    status?: 'pending' | 'done' | 'failed',
    error?: { status: string; message: string; details?: any },
    snapshotId?: string
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
    const prevState = this.lastSnapshot ? this.lastSnapshot.state : undefined;

    if (this.snapshotCallback && !this.isDetached) {
      if (
        !this.snapshotCallback({
          state: currentState as SessionState<State>,
          prevState: prevState as SessionState<State> | undefined,
          turnIndex: this.turnIndex,
          event: event,
        })
      ) {
        return undefined;
      }
    }

    const snapshotInput: SessionSnapshotInput<State> = {
      ...(snapshotId || this.newSnapshotId
        ? { snapshotId: (snapshotId || this.newSnapshotId)! }
        : {}),
      createdAt: new Date().toISOString(),
      event: event,
      state: currentState as SessionState<State>,
      parentId: this.lastSnapshot?.snapshotId,
      status,
      error,
    };

    const effectiveId = snapshotId || this.newSnapshotId;

    // Use the mutator-based saveSnapshot to atomically check the current
    // status before writing.  If the snapshot was concurrently aborted,
    // the mutator returns null and the write is skipped.
    const assignedId = await this.store.saveSnapshot(
      effectiveId,
      (current) => {
        if (current?.status === 'aborted') {
          return null; // Respect the abort — skip the write.
        }
        return snapshotInput;
      },
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
 * Optional transform applied to session state before it is exposed to the
 * client (e.g. in `AgentOutput.state` or via `getSnapshotData`).  This lets
 * agents redact sensitive fields or reshape the state for the client.
 */
export type ClientStateTransform<S = unknown> = (
  state: SessionState<S>
) => SessionState;

/**
 * Function handler definition for custom agent actions.
 */
export type AgentFn<Stream, State> = (
  sess: SessionRunner<State>,
  options: {
    sendChunk: (chunk: AgentStreamChunk<Stream>) => void;
    abortSignal?: AbortSignal;
    context?: ActionContext;
  }
) => Promise<AgentResult>;

export type GetSnapshotDataAction<S = unknown> = Action<
  z.ZodString,
  z.ZodType<SessionSnapshot<S>>
>;

/**
 * Represents a configured, registered Agent.
 */
export interface Agent<State = unknown>
  extends BidiAction<
    typeof AgentInputSchema,
    typeof AgentOutputSchema,
    typeof AgentStreamChunkSchema,
    typeof AgentInitSchema
  > {
  getSnapshotData(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<State> | undefined>;

  abort(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot['status'] | undefined>;

  readonly getSnapshotDataAction: GetSnapshotDataAction<State>;
  readonly abortAgentAction: Action<z.ZodString, z.ZodType<string | undefined>>;
}

/**
 * Registers a multi-turn custom agent action capable of maintaining persistent state.
 *
 * When `stateSchema` is provided the custom state is validated at load time
 * (from a snapshot store or from the client-supplied `init.state`) and the
 * JSON Schema representation is included in the action metadata so that
 * tooling (e.g. the Dev UI) can inspect / validate the state shape.
 */
export function defineCustomAgent<Stream = unknown, State = unknown>(
  registry: Registry,
  config: {
    name: string;
    description?: string;
    stateSchema?: z.ZodType<State>;
    store?: SessionStore<State>;
    snapshotCallback?: SnapshotCallback<State>;
    clientStateTransform?: ClientStateTransform<State>;
  },
  fn: AgentFn<Stream, State>
): Agent<State> {
  // Helper that applies the optional transform before exposing state to the
  // client.  When no transform is configured it returns the raw state.
  const toClientState = (
    state: SessionState<State>
  ): SessionState | undefined => {
    if (config.clientStateTransform) {
      return config.clientStateTransform(state);
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
  const validateCustomState = (custom: unknown, label: string): void => {
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
      let init = arg.init;
      const store = config.store || new InMemorySessionStore<State>();

      // Resolve init into the right internal handle. Three valid inputs:
      //   1. `continuationId` (client-facing) — opaque token; decode it
      //   2. `snapshotId` (server-internal convenience) — used directly
      //   3. neither — fresh session
      // If both v2 and snapshotId are supplied, continuationId wins.
      if (init?.continuationId) {
        const decoded = decodeContinuation(init.continuationId);
        if (!decoded) {
          // Reject malformed tokens loudly rather than silently dropping
          // state. A typo'd / corrupted continuationId starting a fresh
          // session would mask real bugs in the caller.
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message: `init.continuationId is malformed: ${init.continuationId.slice(0, 40)}...`,
          });
        }
        if (decoded.kind === 'snapshot') {
          init = { ...init, _decodedSnapshotId: decoded.snapshotId };
        } else {
          init = {
            ...init,
            _decodedState: decoded.state as SessionState<State>,
          };
        }
      } else if (init?.snapshotId) {
        init = { ...init, _decodedSnapshotId: init.snapshotId };
      }

      let session: Session<State>;
      let snapshot: SessionSnapshot<State> | undefined;

      if (init?._decodedSnapshotId && config.store) {
        snapshot = await store.getSnapshot(init._decodedSnapshotId, {
          context: getContext(),
        });
        if (!snapshot) {
          throw new Error(`Snapshot ${init._decodedSnapshotId} not found`);
        }
        validateCustomState(
          snapshot.state?.custom,
          `snapshot ${init._decodedSnapshotId}`
        );
        session = new Session<State>(snapshot.state as SessionState<State>);
      } else if (init?._decodedState && !config.store) {
        validateCustomState(
          init._decodedState.custom,
          'client-supplied continuation state'
        );
        session = new Session<State>(init._decodedState as SessionState<State>);
      } else {
        session = new Session<State>({
          custom: {} as State,
          artifacts: [],
          messages: [],
        });
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

      let runner!: SessionRunner<State>;

      // We construct an asynchronous proxy channel over the inputStream.
      // This enables immediate interception of `detach: true` directives. Without this proxy,
      // a backlog of pre-queued inputs would have to be resolved sequentially by the runner first.
      const runnerInputChannel = new Channel<AgentInput>();

      (async () => {
        try {
          for await (const input of arg.inputStream) {
            if (input.detach) {
              if (!config.store) {
                if (rejectDetach) {
                  rejectDetach(
                    new GenkitError({
                      status: 'FAILED_PRECONDITION',
                      message:
                        'Detach is only supported when a session store is provided.',
                    })
                  );
                }
              } else {
                const turnSnapshotId =
                  runner.newSnapshotId || globalThis.crypto.randomUUID();
                runner.newSnapshotId = turnSnapshotId;
                await runner.maybeSnapshot(
                  'turnEnd',
                  'pending',
                  undefined,
                  turnSnapshotId
                );
                runner.isDetached = true;

                if (runner.onDetach) {
                  runner.onDetach(turnSnapshotId);
                }
              }
              // Only forward to runner if the input carries a payload beyond the
              // detach directive; a detach-only message has no turn to process.
              const hasPayload = !!(
                input.messages?.length ||
                input.resume?.restart?.length ||
                input.resume?.respond?.length
              );
              if (hasPayload) {
                runnerInputChannel.send(input);
              }
            } else {
              runnerInputChannel.send(input);
            }
          }
          runnerInputChannel.close();
        } catch (e) {
          runnerInputChannel.error(e);
        }
      })();

      runner = new SessionRunner<State>(session, runnerInputChannel, {
        store,
        snapshotCallback: config.snapshotCallback,
        lastSnapshot: snapshot,
        newSnapshotId: init?.newSnapshotId,
        onDetach: (snapshotId) => {
          detachedSnapshotId = snapshotId;
          try {
            arg.sendChunk({
              type: 'detached',
              snapshotId,
              continuationId: encodeSnapshotContinuation(snapshotId),
            });
          } catch (_) {
            // Stream may already be closed; that's fine.
          }
          if (resolveDetach) {
            resolveDetach();
          }

          if (store.onSnapshotStateChange) {
            unsubscribe = store.onSnapshotStateChange(
              snapshotId,
              (snap) => {
                if (snap.status === 'aborted') {
                  abortController.abort();
                  if (unsubscribe) unsubscribe();
                }
              },
              { context: getContext() }
            );
          }
        },

        onEndTurn: (snapshotId) => {
          if (!runner.isDetached) {
            arg.sendChunk({
              type: 'turn-end',
              ...(config.store &&
                snapshotId && {
                  snapshotId,
                  continuationId: encodeSnapshotContinuation(snapshotId),
                }),
            });
          }
        },
      });

      const sendArtifactChunk = (a: Artifact) => {
        if (!runner.isDetached) {
          arg.sendChunk({ type: 'artifact-emitted', artifact: a });
        }
      };
      session.on('artifactAdded', sendArtifactChunk);
      session.on('artifactUpdated', sendArtifactChunk);

      // The chunk emitter passed to user code (e.g. defineCustomAgent's
      // body). Accepts any AgentEvent. Suppresses emission once detached.
      const sendChunk = (chunk: AgentStreamChunk<Stream>) => {
        if (runner.isDetached) return;
        arg.sendChunk(chunk as AgentEvent);
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
          const finalSnapshotId = await runner.maybeSnapshot('invocationEnd');
          return { result, finalSnapshotId };
        } finally {
          if (unsubscribe) unsubscribe();
          session.off('artifactAdded', sendArtifactChunk);
          session.off('artifactUpdated', sendArtifactChunk);
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
          continuationId: encodeSnapshotContinuation(detachedSnapshotId!),
          // Server-side convenience field populated alongside continuationId.
          snapshotId: detachedSnapshotId!,
          state: toClientState(session.getState()),
        };
      }

      const { result, finalSnapshotId } = outcome;
      const clientState = toClientState(session.getState());

      // Continuation token: snapshotId for stored agents, encoded state
      // blob for stateless. Clients round-trip the opaque continuationId;
      // server-internal callers may use snapshotId directly.
      const continuationId = config.store
        ? finalSnapshotId
          ? encodeSnapshotContinuation(finalSnapshotId)
          : undefined
        : clientState
          ? encodeStateContinuation(clientState)
          : undefined;

      return {
        ...(continuationId && { continuationId }),
        ...(config.store && finalSnapshotId && { snapshotId: finalSnapshotId }),
        ...(result.artifacts?.length && { artifacts: result.artifacts }),
        ...(result.message && { message: result.message }),
        ...(clientState && { state: clientState }),
      };
    }
  );

  // Helper that applies the clientStateTransform to a snapshot's state,
  // returning a new snapshot object with the transformed state.
  const toClientSnapshot = (
    snapshot: SessionSnapshot<State>
  ): SessionSnapshot => {
    if (!config.clientStateTransform) {
      return snapshot as SessionSnapshot;
    }
    return {
      ...snapshot,
      state: config.clientStateTransform(snapshot.state),
    };
  };

  const getSnapshotDataAction = defineAction(
    registry,
    {
      name: config.name,
      description: `Gets snapshot data for ${config.name}. Accepts a raw snapshotId (server-internal callers) or a continuationId (clients).`,
      actionType: 'agent-snapshot',
      inputSchema: z.string(),
      outputSchema: z.any(),
    },
    async (token) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `getSnapshotData requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      // Accept either a raw snapshotId (server-internal) or a continuationId
      // (clients). Decode the continuation token if it has the snap: prefix;
      // otherwise treat as a raw snapshotId for backwards compatibility with
      // any direct API caller that already has one.
      const snapshotId = token.startsWith(CONT_SNAP_PREFIX)
        ? token.slice(CONT_SNAP_PREFIX.length)
        : token;
      const snapshot = await config.store.getSnapshot(snapshotId, {
        context: getContext(),
      });
      return snapshot ? toClientSnapshot(snapshot) : undefined;
    }
  );

  const abortAgentAction = defineAction(
    registry,
    {
      name: config.name,
      description: `Aborts ${config.name} agent by snapshotId. Returns the previous status of the snapshot before it was set to 'aborted', or undefined if the snapshot was not found.`,
      actionType: 'agent-abort',
      inputSchema: z.string(),
      outputSchema: z.string().optional(),
    },
    async (snapshotId) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `abort requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      let previousStatus: SessionSnapshot['status'] | undefined;
      await config.store.saveSnapshot(
        snapshotId,
        (current) => {
          if (!current) return null;
          previousStatus = current.status;
          if (
            current.status === 'done' ||
            current.status === 'failed' ||
            current.status === 'aborted'
          ) {
            return null; // Already terminal — don't override.
          }
          return { ...current, status: 'aborted' };
        },
        { context: getContext() }
      );
      return previousStatus;
    }
  );

  const composite = Object.assign(primaryAction, {
    getSnapshotData: async (
      snapshotId: string,
      options?: SessionStoreOptions
    ) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `getSnapshotData requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      const snapshot = await config.store.getSnapshot(snapshotId, options);
      return snapshot ? toClientSnapshot(snapshot) : undefined;
    },
    abort: async (snapshotId: string, options?: SessionStoreOptions) => {
      if (!config.store) {
        throw new GenkitError({
          status: 'FAILED_PRECONDITION',
          message: `abort requires a persistent store. Provide a 'store' when defining '${config.name}'.`,
        });
      }
      let previousStatus: SessionSnapshot['status'] | undefined;
      await config.store.saveSnapshot(
        snapshotId,
        (current) => {
          if (!current) return null;
          previousStatus = current.status;
          if (
            current.status === 'done' ||
            current.status === 'failed' ||
            current.status === 'aborted'
          ) {
            return null; // Already terminal — don't override.
          }
          return { ...current, status: 'aborted' };
        },
        options
      );
      return previousStatus;
    },
    getSnapshotDataAction:
      getSnapshotDataAction as unknown as GetSnapshotDataAction<State>,
    abortAgentAction: abortAgentAction as unknown as Action<
      z.ZodString,
      z.ZodType<string | undefined>
    >,
  });

  return composite as unknown as Agent<State>;
}

/**
 * Registers an agent from an existing PromptAction.
 */
export function definePromptAgent<State = unknown>(
  registry: Registry,
  config: {
    promptName: string;
    stateSchema?: z.ZodType<State>;
    store?: SessionStore<State>;
    snapshotCallback?: SnapshotCallback<State>;
    clientStateTransform?: ClientStateTransform<State>;
  }
) {
  let cachedPromptAction: PromptAction | undefined;

  const fn: AgentFn<unknown, State> = async (
    sess,
    { sendChunk, abortSignal }
  ) => {
    await sess.run(async (input) => {
      const promptInput = {};

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
      // internal history tag — it is an implementation detail that
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
        sendChunk({ type: 'model-chunk', chunk });
        // Tool-error detection: when a tool's handler throws (or returns
        // an error-shaped output via middleware), the runtime packages
        // it as a toolResponse so the model can react and recover. The
        // chunk still flows through, but the client also gets a typed,
        // addressable `tool-error` event so it can mark the matching
        // in-flight tool call as failed without string-matching outputs.
        for (const part of chunk?.content ?? []) {
          const tr = part?.toolResponse;
          if (!tr) continue;
          const detected = detectToolError(tr);
          if (!detected) continue;
          sendChunk({
            type: 'tool-error',
            toolCallId: tr.ref ?? tr.name,
            toolName: tr.name,
            errorText: detected.errorText,
            ...(detected.errorCode && { errorCode: detected.errorCode }),
            ...(detected.details !== undefined && {
              details: detected.details,
            }),
          });
        }
      }

      const res = await result.response;

      // Keep everything that is NOT a prompt-template message:
      //   • history messages (clean — history tag was stripped before generate)
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
        // Emit one `interrupt` event per pending tool request. The client
        // sees these in-stream with addressable refs and can resume each
        // individually via `resume.respond` / `resume.restart`.
        for (const p of parts) {
          const tr = p.toolRequest!;
          sendChunk({
            type: 'interrupt',
            toolCallId: tr.ref ?? tr.name,
            toolName: tr.name,
            input: tr.input,
            kind: 'respond',
            ...(p.metadata?.interrupt !== undefined && {
              metadata: p.metadata.interrupt,
            }),
          });
        }
      }
    });

    const msgs = sess.getMessages();
    return {
      artifacts: sess.getArtifacts(),
      message: msgs.length > 0 ? msgs[msgs.length - 1] : undefined,
    };
  };

  return defineCustomAgent<unknown, State>(
    registry,
    {
      name: config.promptName,
      stateSchema: config.stateSchema,
      store: config.store,
      snapshotCallback: config.snapshotCallback,
      clientStateTransform: config.clientStateTransform,
    },
    fn
  );
}

// ---------------------------------------------------------------------------
// Resume validation — ensure restart/respond entries match session history
// ---------------------------------------------------------------------------

/**
 * Validates that every `resume.restart` and `resume.respond` entry references
 * a tool request that actually exists in the session history.
 *
 * For **restart** entries, also validates that the `input` has not been modified
 * compared to the original tool request — preventing a malicious client from
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
// defineAgent — shortcut that combines definePrompt + definePromptAgent
// ---------------------------------------------------------------------------

/**
 * Configuration for `defineAgent`, which combines prompt definition and agent
 * registration into a single call.
 */
export interface AgentConfig<State = unknown> extends PromptConfig {
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
  snapshotCallback?: SnapshotCallback<State>;
  clientStateTransform?: ClientStateTransform<State>;
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
export function defineAgent<State = unknown>(
  registry: Registry,
  config: AgentConfig<State>
): Agent<State> {
  // Extract agent-specific fields from the combined config; the rest is
  // forwarded to definePrompt.
  const {
    stateSchema,
    store,
    snapshotCallback,
    clientStateTransform,
    ...promptConfig
  } = config;

  // Register the prompt.
  definePrompt(registry, promptConfig);

  // Wire it into a prompt agent.
  return definePromptAgent<State>(registry, {
    promptName: promptConfig.name,
    stateSchema,
    store,
    snapshotCallback,
    clientStateTransform,
  });
}
