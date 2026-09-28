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

import type {
  Action,
  ActionContext,
  ActionMetadata,
  Flow,
  Genkit,
  ToolAction,
  z,
} from 'genkit';
import type { AgentAPI } from 'genkit/beta';
import { createAgentProxy } from './agent-proxy.js';
import { BOX_SELF_ID_ENV } from './env.js';
import {
  actionKeyFor,
  createProxyAction,
  type ProxyDispatcher,
} from './proxy.js';
import { resolveRetention, singleton } from './route.js';
import type {
  BoxConnection,
  BoxOptions,
  BoxRunner,
  Retention,
  RouteFn,
} from './types.js';

/** Ordinal counter for auto-naming boxes when no `name` is given. */
let boxOrdinal = 0;

/** Calls in flight for one routing key, and its pending idle reclaim. */
interface Lease {
  inflight: number;
  idleTimer?: NodeJS.Timeout;
}

/** Options for creating a box, with the self-nesting `name` override. */
export interface BoxCreateOptions extends BoxOptions {
  /**
   * Stable id for this box. Required for reliable self-mode nesting; defaults
   * to a construction-order ordinal (`box:0`, `box:1`, ...), which is stable as
   * long as boxes are constructed deterministically.
   */
  name?: string;
}

/** A tool/flow spec used when there is no local action to hand over. */
export interface ProxySpec<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> {
  name: string;
  description?: string;
  inputSchema?: I;
  outputSchema?: O;
  /** Stream chunk schema, for flows that stream. */
  streamSchema?: S;
}

function metaFromAction<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(action: Action<I, O, S>): Partial<ActionMetadata<I, O, S>> {
  return {
    actionType: action.__action.actionType,
    description: action.__action.description,
    inputSchema: action.__action.inputSchema,
    outputSchema: action.__action.outputSchema,
    streamSchema: action.__action.streamSchema,
    metadata: action.__action.metadata,
    key: action.__action.key,
    name: action.__action.name,
  };
}

/**
 * A box: runs Genkit actions somewhere else (a subprocess, sandbox, ...) and
 * hands you proxies that call them as if they were local.
 */
export class Box {
  readonly id: string;
  private readonly route: RouteFn;
  private readonly retention: Retention;
  private readonly leases = new Map<string, Lease>();
  /** True when this process is itself the runtime for this box (self-mode). */
  readonly isSelfRuntime: boolean;

  constructor(
    private readonly ai: Genkit,
    private readonly options: BoxCreateOptions
  ) {
    this.id = options.name ?? `box:${boxOrdinal++}`;
    this.route = options.route ?? singleton;
    this.retention = resolveRetention(this.route, options.retention);
    this.isSelfRuntime = process.env[BOX_SELF_ID_ENV] === this.id;
    options.runner.attach?.(this);
  }

  /** The underlying runner. */
  get runner(): BoxRunner {
    return this.options.runner;
  }

  /**
   * Builds the dispatcher a proxy uses: runs the route fn to get a key, then
   * acquires the box for it under a lease that drives idle reclaim.
   */
  private dispatcher(): ProxyDispatcher {
    return {
      boxId: this.id,
      acquire: async (req, signal) => {
        if (this.isSelfRuntime) {
          throw new Error(
            `Box '${this.id}': called a boxed proxy from within the box's own ` +
              `runtime. Use the real (local) action here, or run the box from ` +
              `a separate entry point.`
          );
        }
        const key = this.route(req, req.context);
        const lease = this.openLease(key);
        try {
          const conn = await this.runner.acquire(key, signal);
          return this.leasedConnection(conn, key, lease);
        } catch (e) {
          this.closeLease(key, lease);
          throw e;
        }
      },
    };
  }

  private openLease(key: string): Lease {
    let lease = this.leases.get(key);
    if (!lease) {
      lease = { inflight: 0 };
      this.leases.set(key, lease);
    }
    clearTimeout(lease.idleTimer);
    lease.idleTimer = undefined;
    lease.inflight++;
    return lease;
  }

  /**
   * Ends one call on `key`. When it was the last one in flight, the box is
   * released right away (`idle: 0`) or after the idle window; a call arriving
   * in the meantime cancels the pending release.
   */
  private closeLease(key: string, lease: Lease): void {
    lease.inflight--;
    const idle = this.retention.idle;
    if (lease.inflight > 0 || idle === undefined) return;
    const release = () => {
      this.leases.delete(key);
      this.runner.release(key).catch(() => {});
    };
    if (idle === 0) {
      release();
    } else {
      lease.idleTimer = setTimeout(release, idle);
      // An idle box must not keep the process alive.
      lease.idleTimer.unref();
    }
  }

  /**
   * Wraps a connection so its single dispatched call closes the lease when it
   * settles. Each dispatcher `acquire` backs exactly one `runAction`.
   */
  private leasedConnection(
    conn: BoxConnection,
    key: string,
    lease: Lease
  ): BoxConnection {
    let closed = false;
    return {
      listActions: () => conn.listActions(),
      runAction: async (req, opts) => {
        try {
          return await conn.runAction(req, opts);
        } finally {
          if (!closed) {
            closed = true;
            this.closeLease(key, lease);
          }
        }
      },
    };
  }

  /** Unregistered tool proxy from a spec (cross-language / remote box). */
  tool<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
  >(spec: ProxySpec<I, O>): ToolAction<I, O> {
    return createProxyAction<I, O>(this.dispatcher(), 'tool', spec.name, {
      description: spec.description,
      inputSchema: spec.inputSchema,
      outputSchema: spec.outputSchema,
    }) as ToolAction<I, O>;
  }

  /** Unregistered tool proxy from a real action (same-language sugar). */
  fromTool<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
  >(action: ToolAction<I, O>): ToolAction<I, O> {
    const meta = metaFromAction(action);
    return createProxyAction<I, O>(
      this.dispatcher(),
      'tool',
      action.__action.name,
      { ...meta, metadata: { ...meta.metadata, dynamic: true } }
    ) as ToolAction<I, O>;
  }

  /** Registered tool proxy from a spec. Visible/runnable in the Dev UI. */
  defineTool<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
  >(spec: ProxySpec<I, O>): ToolAction<I, O> {
    const proxy = this.tool<I, O>(spec);
    this.register(proxy);
    return proxy;
  }

  /**
   * Registered tool proxy from a real action. MUST be renamed, because the
   * original already occupies its name in the registry.
   */
  defineFromTool<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
  >(action: ToolAction<I, O>, opts: { name: string }): ToolAction<I, O> {
    if (opts.name === action.__action.name) {
      throw new Error(
        `defineFromTool: the proxy must be renamed (got '${opts.name}', same ` +
          `as the original). The original already occupies that name in the ` +
          `registry.`
      );
    }
    // Registered under the new name, but still calls the original in the box.
    const proxy = createProxyAction<I, O>(
      this.dispatcher(),
      'tool',
      opts.name,
      metaFromAction(action),
      actionKeyFor('tool', action.__action.name)
    ) as ToolAction<I, O>;
    this.register(proxy);
    return proxy;
  }

  /** Unregistered flow proxy from a spec. */
  flow<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    S extends z.ZodTypeAny = z.ZodTypeAny,
    Init extends z.ZodTypeAny = z.ZodTypeAny,
  >(spec: ProxySpec<I, O, S>): Flow<I, O, S, Init> {
    return createProxyAction<I, O, S, Init>(
      this.dispatcher(),
      'flow',
      spec.name,
      {
        description: spec.description,
        inputSchema: spec.inputSchema,
        outputSchema: spec.outputSchema,
        streamSchema: spec.streamSchema,
      }
    ) as Flow<I, O, S, Init>;
  }

  /** Unregistered flow proxy from a real action (same-language sugar). */
  fromFlow<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    S extends z.ZodTypeAny = z.ZodTypeAny,
    Init extends z.ZodTypeAny = z.ZodTypeAny,
  >(action: Flow<I, O, S, Init>): Flow<I, O, S, Init> {
    return createProxyAction<I, O, S, Init>(
      this.dispatcher(),
      'flow',
      action.__action.name,
      metaFromAction(action)
    ) as Flow<I, O, S, Init>;
  }

  /** Registered flow proxy from a spec. Visible/runnable in the Dev UI. */
  defineFlow<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    S extends z.ZodTypeAny = z.ZodTypeAny,
    Init extends z.ZodTypeAny = z.ZodTypeAny,
  >(spec: ProxySpec<I, O, S>): Flow<I, O, S, Init> {
    const proxy = this.flow<I, O, S, Init>(spec);
    this.ai.registry.registerAction('flow', proxy);
    return proxy;
  }

  /**
   * Registered flow proxy from a real action. MUST be renamed (see
   * {@link defineFromTool}).
   */
  defineFromFlow<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
    S extends z.ZodTypeAny = z.ZodTypeAny,
    Init extends z.ZodTypeAny = z.ZodTypeAny,
  >(action: Flow<I, O, S, Init>, opts: { name: string }): Flow<I, O, S, Init> {
    if (opts.name === action.__action.name) {
      throw new Error(
        `defineFromFlow: the proxy must be renamed (got '${opts.name}', same ` +
          `as the original).`
      );
    }
    const proxy = createProxyAction<I, O, S, Init>(
      this.dispatcher(),
      'flow',
      opts.name,
      metaFromAction(action),
      actionKeyFor('flow', action.__action.name)
    ) as Flow<I, O, S, Init>;
    this.ai.registry.registerAction('flow', proxy);
    return proxy;
  }

  /**
   * A boxed {@link AgentAPI} for the agent named `spec.name`. Calls surface the
   * self-runtime guard lazily on first turn (via the dispatcher), matching the
   * tool/flow behavior.
   *
   * `spec.context` is passed to `route` (and on to the boxed agent) for every
   * call through this proxy. Proxies are cheap, so to route per session build
   * one per call: `box.agent({ name, context: { sessionId } })`.
   */
  agent<State = unknown>(spec: {
    name: string;
    context?: ActionContext;
  }): AgentAPI<State> {
    return createAgentProxy<State>(this.dispatcher(), spec.name, spec.context);
  }

  /** A boxed {@link AgentAPI} from a real agent (same-language sugar). */
  fromAgent<State = unknown>(
    agent: { __action?: { name: string } },
    opts?: { context?: ActionContext }
  ): AgentAPI<State> {
    const name = agent.__action?.name;
    if (!name) {
      throw new Error('fromAgent: could not determine the agent name.');
    }
    return createAgentProxy<State>(this.dispatcher(), name, opts?.context);
  }

  /** Registers a tool proxy in the Genkit registry so the Dev UI can see it. */
  private register<
    I extends z.ZodTypeAny = z.ZodTypeAny,
    O extends z.ZodTypeAny = z.ZodTypeAny,
  >(proxy: Action<I, O>): void {
    this.ai.registry.registerAction('tool', proxy);
  }

  /** Tears down the box and its runner. */
  async close(): Promise<void> {
    for (const lease of this.leases.values()) clearTimeout(lease.idleTimer);
    this.leases.clear();
    await this.runner.close();
  }
}

/** Creates a {@link Box}. */
export function box(ai: Genkit, options: BoxCreateOptions): Box {
  return new Box(ai, options);
}
