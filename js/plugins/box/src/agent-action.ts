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

import { z, type ActionContext, type Genkit } from 'genkit';
import {
  AgentInitSchema,
  AgentInputSchema,
  AgentOutputSchema,
  AgentStreamChunkSchema,
  createAgentAPI,
  type Agent,
  type AgentInit,
  type AgentInput,
  type AgentOutput,
  type AgentStreamChunk,
  type AgentTransport,
  type SessionSnapshot,
} from 'genkit/beta';
import { toJsonSchema } from 'genkit/schema';
import { Channel } from './channel.js';
import { tracedDispatch, type ProxyDispatcher } from './proxy.js';

/** What the Dev UI needs to know about a boxed agent it cannot inspect. */
export interface BoxedAgentSpec<State = unknown> {
  name: string;
  description?: string;
  /**
   * Must match the boxed agent: `'server'` when it has a session store,
   * `'client'` when callers carry the state. Required because the Dev UI and
   * `chat()` send different inits for each, and the boxed agent rejects a
   * mismatch.
   */
  stateManagement: 'server' | 'client';
  /** Whether the boxed agent's store supports aborting. Defaults to false. */
  abortable?: boolean;
  /** Schema of the agent's custom state, shown in the Dev UI. */
  stateSchema?: z.ZodType<State>;
}

/** The `metadata.agent` block core agents carry, read by the Dev UI. */
export interface AgentCapabilities {
  stateManagement: 'server' | 'client';
  abortable: boolean;
  stateSchema?: unknown;
}

function field(value: unknown, key: string): unknown {
  return typeof value === 'object' && value !== null
    ? Reflect.get(value, key)
    : undefined;
}

function stringField(value: unknown, key: string): string | undefined {
  const v = field(value, key);
  return typeof v === 'string' && v !== '' ? v : undefined;
}

/** Reads `metadata.agent` off a real agent, so the proxy advertises the same. */
export function capabilitiesOf(agent: {
  __action: { metadata?: Record<string, unknown> };
}): AgentCapabilities {
  const raw = agent.__action.metadata?.agent;
  const stateSchema = field(raw, 'stateSchema');
  return {
    stateManagement:
      field(raw, 'stateManagement') === 'client' ? 'client' : 'server',
    abortable: field(raw, 'abortable') === true,
    ...(stateSchema !== undefined && { stateSchema }),
  };
}

export function capabilitiesFromSpec<State>(
  spec: BoxedAgentSpec<State>
): AgentCapabilities {
  return {
    stateManagement: spec.stateManagement,
    abortable: spec.abortable ?? false,
    ...(spec.stateSchema && {
      stateSchema: toJsonSchema({ schema: spec.stateSchema }),
    }),
  };
}

/** Options for {@link defineBoxedAgent}. */
export interface DefineBoxedAgentOptions {
  /** Registered name (`/agent/<name>`). */
  name: string;
  /** Name of the agent inside the box. */
  target: string;
  description?: string;
  capabilities: AgentCapabilities;
}

/** Whether a turn's output ends the invocation before its remaining inputs. */
function endsInvocation(output: AgentOutput): boolean {
  return output.finishReason === 'failed' || output.finishReason === 'detached';
}

/**
 * The init for the turn after `output`, mirroring how `AgentChat` threads
 * state: server-managed agents resume from the new snapshot (keeping the
 * session id, which routing may depend on), client-managed ones carry the
 * returned state.
 */
function nextInit(
  prev: AgentInit,
  output: AgentOutput,
  stateManagement: 'server' | 'client'
): AgentInit {
  if (stateManagement === 'client') {
    return output.state ? { state: output.state } : prev;
  }
  const sessionId = output.sessionId ?? prev.sessionId;
  return output.snapshotId
    ? { snapshotId: output.snapshotId, ...(sessionId && { sessionId }) }
    : prev;
}

/** Companion request schemas, mirroring the agent wire contract. */
const SnapshotLookupSchema = z.object({
  snapshotId: z.string().optional(),
  sessionId: z.string().optional(),
});
const AbortRequestSchema = z.object({ snapshotId: z.string() });

/** The options an agent action's `run` receives (from reflection or chat). */
interface AgentRunOptions {
  init?: AgentInit;
  context?: ActionContext;
  onChunk?: (chunk: AgentStreamChunk) => void;
  onTraceStart?: (trace: { traceId: string; spanId: string }) => void;
  abortSignal?: AbortSignal;
  inputStream?: AsyncIterable<AgentInput>;
}

/**
 * Remembers which session each snapshot belongs to, from the outputs it sees,
 * and stamps that onto calls that only carry a `snapshotId` (resumes, snapshot
 * reads, aborts). Routes keyed on `sessionIdOf(req)` then keep those calls on
 * the session's box. The hint is routing-only; runners never send it.
 */
class SessionTracker {
  private readonly bySnapshot = new Map<string, string>();

  remember(output: AgentOutput): void {
    if (!output.snapshotId || !output.sessionId) return;
    this.bySnapshot.set(output.snapshotId, output.sessionId);
    // Only recent snapshots get resumed; keep the map bounded.
    if (this.bySnapshot.size > 1000) {
      const oldest = this.bySnapshot.keys().next().value;
      if (oldest !== undefined) this.bySnapshot.delete(oldest);
    }
  }

  wrap(dispatcher: ProxyDispatcher): ProxyDispatcher {
    return {
      boxId: dispatcher.boxId,
      acquire: (req, signal) => {
        const snapshotId =
          stringField(req.init, 'snapshotId') ??
          stringField(req.input, 'snapshotId');
        const sessionId =
          req.sessionId ??
          (snapshotId ? this.bySnapshot.get(snapshotId) : undefined);
        return dispatcher.acquire({ ...req, sessionId }, signal);
      },
    };
  }
}

/**
 * Runs one agent invocation against the box. Reflection v1 has no input
 * streaming, so an invocation is run turn by turn: each input becomes one
 * boxed call, with the init threaded from the previous output. A single input
 * (`chat().send()`, the Dev UI) is exactly one call with identical semantics.
 * With several inputs the box records one trace per turn, and the final output
 * carries only the last turn's `artifacts`.
 */
async function runInvocation(
  dispatcher: ProxyDispatcher,
  tracker: SessionTracker,
  opts: DefineBoxedAgentOptions,
  inputs: AsyncIterable<AgentInput>,
  run: AgentRunOptions
): Promise<{ output: AgentOutput; traceId: string; spanId: string }> {
  let init: AgentInit = run.init ?? {};
  let output: AgentOutput = {};
  let trace = { traceId: '', spanId: '' };
  for await (const input of inputs) {
    const { res, traceId, spanId } = await tracedDispatch<AgentOutput>(
      dispatcher,
      { name: opts.name, subtype: 'agent' },
      { key: `/agent/${opts.target}`, input, init, context: run.context },
      {
        // The wire carries JSON; the boxed agent's schema is the contract.
        onChunk: (c) => run.onChunk?.(c as AgentStreamChunk),
        // Report the first turn's span: it is the invocation's entry point.
        onTraceStart: trace.traceId ? undefined : run.onTraceStart,
        abortSignal: run.abortSignal,
      }
    );
    if (!trace.traceId) trace = { traceId, spanId };
    output = res.result ?? {};
    tracker.remember(output);
    if (endsInvocation(output)) break;
    init = nextInit(init, output, opts.capabilities.stateManagement);
  }
  return { output, ...trace };
}

async function* once<T>(value: T): AsyncIterable<T> {
  yield value;
}

/**
 * Builds the registered `/agent/<name>` action. Hand-rolled like the tool and
 * flow proxies (see `createProxyAction`) rather than defined through core, so
 * box depends on `genkit` only; the shape is what the registry, reflection
 * and `streamBidi` callers use.
 */
function agentProxyAction(
  dispatcher: ProxyDispatcher,
  tracker: SessionTracker,
  opts: DefineBoxedAgentOptions
) {
  const run = async (input: AgentInput | undefined, o?: AgentRunOptions) => {
    const inputs = o?.inputStream ?? once(input ?? {});
    const { output, traceId, spanId } = await runInvocation(
      dispatcher,
      tracker,
      opts,
      inputs,
      o ?? {}
    );
    return { result: output, telemetry: { traceId, spanId } };
  };

  const stream = (input: AgentInput | undefined, o?: AgentRunOptions) => {
    const chunks = new Channel<AgentStreamChunk>();
    const output = run(input, { ...o, onChunk: (c) => chunks.send(c) }).then(
      (r) => {
        chunks.close();
        return r.result;
      },
      (e) => {
        chunks.error(e);
        throw e;
      }
    );
    output.catch(() => {});
    return { stream: chunks, output };
  };

  const callable = async (input?: AgentInput, o?: AgentRunOptions) =>
    (await run(input, o)).result;

  return Object.assign(callable, {
    __action: {
      name: opts.name,
      key: `/agent/${opts.name}`,
      actionType: 'agent',
      description: opts.description,
      inputSchema: AgentInputSchema,
      outputSchema: AgentOutputSchema,
      streamSchema: AgentStreamChunkSchema,
      initSchema: AgentInitSchema,
      metadata: {
        agent: opts.capabilities,
        bidi: true,
        box: dispatcher.boxId ?? true,
      },
    },
    run,
    stream,
    streamBidi(init?: AgentInit, o?: AgentRunOptions) {
      const inputs = new Channel<AgentInput>();
      const res = stream(undefined, {
        ...o,
        init,
        inputStream: o?.inputStream ?? inputs,
      });
      return {
        ...res,
        send: (chunk: AgentInput) => inputs.send(chunk),
        close: () => inputs.close(),
      };
    },
  });
}

/** A plain (non-streaming) companion proxy, e.g. `/agent-snapshot/<name>`. */
function companionAction<I extends z.ZodTypeAny, O>(
  dispatcher: ProxyDispatcher,
  actionType: 'agent-snapshot' | 'agent-abort',
  opts: DefineBoxedAgentOptions,
  description: string,
  inputSchema: I
) {
  const target = `/${actionType}/${opts.target}`;
  const run = async (
    input: z.infer<I>,
    o?: Pick<AgentRunOptions, 'context' | 'onTraceStart' | 'abortSignal'>
  ) => {
    const { res, traceId, spanId } = await tracedDispatch<O>(
      dispatcher,
      { name: opts.name, subtype: actionType },
      { key: target, input, context: o?.context },
      { onTraceStart: o?.onTraceStart, abortSignal: o?.abortSignal }
    );
    return { result: res.result, telemetry: { traceId, spanId } };
  };
  const callable = async (input: z.infer<I>, o?: AgentRunOptions) =>
    (await run(input, o)).result;
  return Object.assign(callable, {
    __action: {
      name: opts.name,
      key: `/${actionType}/${opts.name}`,
      actionType,
      description,
      inputSchema,
      metadata: { box: dispatcher.boxId ?? true },
    },
    run,
    stream: (input: z.infer<I>, o?: AgentRunOptions) => {
      const output = callable(input, o);
      return { stream: new Channel<never>(), output };
    },
  });
}

/**
 * Registers a boxed agent: an `agent` action plus its `agent-snapshot` and
 * `agent-abort` companions, all forwarding to the agent `target` in the box.
 * Returns it as an {@link Agent}, chattable in-process and from the Dev UI
 * like a local one.
 */
export function defineBoxedAgent<State>(
  ai: Genkit,
  dispatcher: ProxyDispatcher,
  opts: DefineBoxedAgentOptions
): Agent<State> {
  const tracker = new SessionTracker();
  const routed = tracker.wrap(dispatcher);

  const agent = agentProxyAction(routed, tracker, opts);
  const snapshotAction = companionAction<
    typeof SnapshotLookupSchema,
    SessionSnapshot<State>
  >(
    routed,
    'agent-snapshot',
    opts,
    `Gets snapshot data for ${opts.name} by snapshotId or sessionId`,
    SnapshotLookupSchema
  );
  const abortAction = companionAction<
    typeof AbortRequestSchema,
    { snapshotId: string; status?: SessionSnapshot['status'] }
  >(
    routed,
    'agent-abort',
    opts,
    `Aborts ${opts.name} agent by snapshotId.`,
    AbortRequestSchema
  );

  // The proxies implement the callable Action surface the registry and
  // reflection use (`__action`, `run`, `stream`); the registry's parameter
  // type is the full core Action, which box builds structurally instead.
  type RegistryAction = Parameters<typeof ai.registry.registerAction>[1];
  const register = (
    type: 'agent' | 'agent-snapshot' | 'agent-abort',
    action: { __action: { actionType: string } }
  ) => ai.registry.registerAction(type, action as unknown as RegistryAction);
  register('agent', agent);
  register('agent-snapshot', snapshotAction);
  register('agent-abort', abortAction);

  // In-process AgentAPI: drive the registered agent action, like core agents
  // do, so in-process and Dev UI calls share one code path and trace shape.
  const transport: AgentTransport = {
    stateManagement: opts.capabilities.stateManagement,
    runTurn(input, init, turn) {
      const bidi = agent.streamBidi(init, { abortSignal: turn.abortSignal });
      bidi.send(input);
      bidi.close();
      return { stream: bidi.stream, output: bidi.output };
    },
    getSnapshot: (lookup) => snapshotAction(lookup),
    abort: async (snapshotId) => (await abortAction({ snapshotId }))?.status,
  };
  const api = createAgentAPI<State>(transport);

  // Structurally the Agent surface: the bidi action above plus the AgentAPI
  // and companion handles core agents expose.
  return Object.assign(agent, {
    chat: api.chat,
    loadChat: api.loadChat,
    getSnapshot: api.getSnapshot,
    getSnapshotData: (lookup: { snapshotId?: string; sessionId?: string }) =>
      snapshotAction(lookup),
    abort: (snapshotId: string) => transport.abort(snapshotId),
    getSnapshotDataAction: snapshotAction,
    abortAgentAction: abortAction,
  }) as unknown as Agent<State>;
}
