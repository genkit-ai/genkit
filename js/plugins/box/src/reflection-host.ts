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

import type { ActionMetadata } from 'genkit';
import { logger } from 'genkit/logging';
import { randomBytes } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { WebSocket, WebSocketServer } from 'ws';
import { REFLECTION_AUTH_ERROR_CODE, secretsEqual } from './reflection-auth.js';
import type { RunActionRequest, RunActionResult, RunOptions } from './types.js';

/** Information about a connected box runtime. */
export interface ConnectedRuntimeInfo {
  id: string;
  pid?: number;
  name?: string;
  genkitVersion?: string;
}

/** Emitted host lifecycle events. */
export enum HostEvent {
  RUNTIME_CONNECT = 'runtimeConnect',
  RUNTIME_DISCONNECT = 'runtimeDisconnect',
}

interface JsonRpcRequest {
  jsonrpc: '2.0';
  method: string;
  params?: unknown;
  id?: string;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
  id: string;
}

type JsonRpcMessage = JsonRpcRequest | JsonRpcResponse;

interface ConnectedRuntime {
  ws: WebSocket;
  info: ConnectedRuntimeInfo;
}

/** An error carrying the structured `data` payload from a reflection error. */
export class BoxRuntimeError extends Error {
  constructor(
    message: string,
    readonly data?: unknown
  ) {
    super(message);
    this.name = 'BoxRuntimeError';
  }
}

/** Options for {@link ReflectionHost}. */
export interface ReflectionHostOptions {
  /**
   * Secret runtimes must present in `register`. Defaults to a fresh random
   * secret per host; pass it to spawned runtimes as
   * `GENKIT_REFLECTION_SECRET_TOKEN` (see {@link ReflectionHost.secret}).
   * `false` accepts any runtime (for runtimes that predate reflection auth).
   */
  secret?: string | false;
}

/** WebSocket close code for a runtime that failed auth. */
const WS_POLICY_VIOLATION = 1008;

/**
 * A self-contained implementation of the Genkit Reflection V2 "manager" role.
 * Hosts a WebSocket server and coordinates JSON-RPC requests with connected box
 * runtimes. Deliberately standalone so the box package has no dependency on
 * developer tooling (`genkit-tools/common` is CLI-only).
 *
 * The server binds loopback only, but anything local could still dial it, so
 * runtimes must present {@link secret} in `register` (same handshake as the
 * Genkit CLI).
 */
export class ReflectionHost {
  private wss?: WebSocketServer;
  private _port?: number;
  private runtimes = new Map<string, ConnectedRuntime>();
  private emitter = new EventEmitter();
  private requestIdCounter = 0;
  private pendingRequests = new Map<
    string,
    { resolve: (value: unknown) => void; reject: (reason?: unknown) => void }
  >();
  private streamCallbacks = new Map<string, (chunk: unknown) => void>();
  private traceIdCallbacks = new Map<string, (traceId: string) => void>();
  /** The secret runtimes must present, or undefined when auth is off. */
  readonly secret: string | undefined;

  constructor(options?: ReflectionHostOptions) {
    this.secret =
      options?.secret === false
        ? undefined
        : (options?.secret ?? randomBytes(32).toString('base64url'));
  }

  get port(): number | undefined {
    return this._port;
  }

  /**
   * The `ws://` URL box runtimes should dial. Uses the loopback IP literal
   * (`127.0.0.1`) rather than `localhost` to avoid IPv4/IPv6 resolution
   * mismatches: the server binds to `127.0.0.1`, but `localhost` may resolve to
   * `::1` first on some systems (notably macOS).
   */
  get url(): string {
    return `ws://127.0.0.1:${this._port}`;
  }

  /** Starts the WebSocket server on the given (or an OS-assigned) port. */
  async start(port?: number): Promise<number> {
    this.wss = new WebSocketServer({ port: port ?? 0, host: '127.0.0.1' });
    await new Promise<void>((resolve, reject) => {
      this.wss!.once('listening', () => resolve());
      this.wss!.once('error', reject);
    });
    const address = this.wss.address();
    this._port =
      typeof address === 'object' && address ? address.port : (port ?? 0);
    logger.debug(`Box reflection host listening on ${this.url}`);

    this.wss.on('connection', (ws) => {
      ws.on('error', (err) => logger.error(`Box WebSocket error: ${err}`));
      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString()) as JsonRpcMessage;
          this.handleMessage(ws, message);
        } catch (err) {
          logger.error(`Failed to parse box WebSocket message: ${err}`);
        }
      });
      ws.on('close', () => this.handleDisconnect(ws));
    });
    return this._port;
  }

  /** Subscribe to a host lifecycle event. Returns an unsubscribe function. */
  on(
    event: HostEvent,
    listener: (info: ConnectedRuntimeInfo) => void
  ): () => void {
    this.emitter.on(event, listener);
    return () => this.emitter.off(event, listener);
  }

  listRuntimeIds(): string[] {
    return Array.from(this.runtimes.keys());
  }

  hasRuntime(id: string): boolean {
    return this.runtimes.has(id);
  }

  /**
   * Waits until a runtime with the given id connects (or any runtime, when no
   * id is provided). Rejects on timeout.
   */
  waitForRuntime(
    id?: string,
    timeoutMs = 30_000
  ): Promise<ConnectedRuntimeInfo> {
    if (id && this.runtimes.has(id)) {
      return Promise.resolve(this.runtimes.get(id)!.info);
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        unsub();
        reject(new Error('Timed out waiting for box runtime to connect'));
      }, timeoutMs);
      const unsub = this.on(HostEvent.RUNTIME_CONNECT, (info) => {
        if (!id || info.id === id) {
          clearTimeout(timer);
          unsub();
          resolve(info);
        }
      });
    });
  }

  private handleMessage(ws: WebSocket, message: JsonRpcMessage) {
    if ('method' in message) {
      this.handleRequest(ws, message);
    } else {
      this.handleResponse(message);
    }
  }

  private handleRequest(ws: WebSocket, request: JsonRpcRequest) {
    switch (request.method) {
      case 'register':
        this.handleRegister(ws, request);
        break;
      case 'streamChunk': {
        const { requestId, chunk } =
          (request.params as { requestId?: string; chunk?: unknown }) ?? {};
        if (requestId) this.streamCallbacks.get(requestId)?.(chunk);
        break;
      }
      case 'runActionState': {
        const { requestId, state } =
          (request.params as {
            requestId?: string;
            state?: { traceId?: string };
          }) ?? {};
        if (requestId && state?.traceId) {
          this.traceIdCallbacks.get(requestId)?.(state.traceId);
        }
        break;
      }
      default:
        logger.debug(`Box host ignoring unknown method: ${request.method}`);
    }
  }

  private handleRegister(ws: WebSocket, request: JsonRpcRequest) {
    const params =
      (request.params as
        | (Partial<ConnectedRuntimeInfo> & { secret?: string })
        | undefined) ?? {};
    if (!params.id) {
      logger.warn('Box runtime register missing id; ignoring.');
      return;
    }
    if (
      this.secret !== undefined &&
      (!params.secret || !secretsEqual(params.secret, this.secret))
    ) {
      logger.warn(
        `Box runtime ${params.id} (pid ${params.pid}) rejected: ` +
          `${params.secret ? 'invalid' : 'missing'} reflection secret.`
      );
      if (request.id) {
        // The auth error code tells the runtime not to reconnect.
        ws.send(
          JSON.stringify({
            jsonrpc: '2.0',
            error: {
              code: REFLECTION_AUTH_ERROR_CODE,
              message: 'Invalid reflection secret.',
            },
            id: request.id,
          })
        );
      }
      ws.close(WS_POLICY_VIOLATION, 'unauthorized');
      return;
    }
    const info: ConnectedRuntimeInfo = {
      id: params.id,
      pid: params.pid,
      name: params.name,
      genkitVersion: params.genkitVersion,
    };
    this.runtimes.set(info.id, { ws, info });
    this.emitter.emit(HostEvent.RUNTIME_CONNECT, info);
    logger.debug(`Box runtime registered: ${info.id} (pid ${info.pid})`);
    if (request.id) {
      ws.send(JSON.stringify({ jsonrpc: '2.0', result: {}, id: request.id }));
    }
  }

  private handleResponse(response: JsonRpcResponse) {
    const pending = this.pendingRequests.get(response.id);
    if (!pending) {
      logger.debug(`Box host got response for unknown id ${response.id}`);
      return;
    }
    this.pendingRequests.delete(response.id);
    if (response.error) {
      pending.reject(
        new BoxRuntimeError(response.error.message, response.error.data)
      );
    } else {
      pending.resolve(response.result);
    }
  }

  private handleDisconnect(ws: WebSocket) {
    for (const [id, runtime] of this.runtimes.entries()) {
      if (runtime.ws === ws) {
        this.runtimes.delete(id);
        this.emitter.emit(HostEvent.RUNTIME_DISCONNECT, runtime.info);
        logger.debug(`Box runtime disconnected: ${id}`);
        break;
      }
    }
  }

  private nextId(): string {
    return (++this.requestIdCounter).toString();
  }

  private sendRequest(
    runtimeId: string,
    method: string,
    params?: unknown,
    timeoutMs = 0
  ): Promise<unknown> {
    const runtime = this.runtimes.get(runtimeId);
    if (!runtime) {
      return Promise.reject(new Error(`Box runtime ${runtimeId} not found`));
    }
    const id = this.nextId();
    return new Promise((resolve, reject) => {
      let timer: NodeJS.Timeout | undefined;
      if (timeoutMs > 0) {
        timer = setTimeout(() => {
          if (this.pendingRequests.has(id)) {
            this.pendingRequests.delete(id);
            reject(new Error(`Box request '${method}' timed out`));
          }
        }, timeoutMs);
      }
      this.pendingRequests.set(id, {
        resolve: (v) => {
          if (timer) clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          if (timer) clearTimeout(timer);
          reject(e);
        },
      });
      runtime.ws.send(JSON.stringify({ jsonrpc: '2.0', method, params, id }));
    });
  }

  private sendNotification(
    runtimeId: string,
    method: string,
    params?: unknown
  ) {
    const runtime = this.runtimes.get(runtimeId);
    if (!runtime) {
      logger.warn(`Cannot notify unknown box runtime ${runtimeId}`);
      return;
    }
    runtime.ws.send(JSON.stringify({ jsonrpc: '2.0', method, params }));
  }

  /** Lists the actions registered by a runtime. */
  async listActions(
    runtimeId: string
  ): Promise<Record<string, ActionMetadata>> {
    const result = (await this.sendRequest(
      runtimeId,
      'listActions',
      undefined,
      30_000
    )) as { actions?: Record<string, ActionMetadata> } | undefined;
    return result?.actions ?? {};
  }

  /**
   * Runs an action in the given runtime. Supports server streaming (via
   * `onChunk`), early trace ids (via `onTraceId`), and cancellation (via
   * `abortSignal`).
   */
  async runAction<O = unknown>(
    runtimeId: string,
    req: RunActionRequest,
    opts?: RunOptions
  ): Promise<RunActionResult<O>> {
    const runtime = this.runtimes.get(runtimeId);
    if (!runtime) {
      throw new Error(`Box runtime ${runtimeId} not found`);
    }
    const id = this.nextId();
    let traceId: string | undefined;
    if (opts?.onChunk) this.streamCallbacks.set(id, opts.onChunk);
    this.traceIdCallbacks.set(id, (tid) => {
      traceId = tid;
      opts?.onTraceId?.(tid);
    });

    let onAbort: (() => void) | undefined;
    const cleanup = () => {
      this.streamCallbacks.delete(id);
      this.traceIdCallbacks.delete(id);
      if (opts?.abortSignal && onAbort) {
        opts.abortSignal.removeEventListener('abort', onAbort);
      }
    };

    if (opts?.abortSignal) {
      onAbort = () => {
        if (traceId) {
          this.cancelAction(runtimeId, traceId).catch((e) =>
            logger.debug(`Box cancel failed: ${e}`)
          );
        }
      };
      if (opts.abortSignal.aborted) onAbort();
      else opts.abortSignal.addEventListener('abort', onAbort);
    }

    const message: JsonRpcRequest = {
      jsonrpc: '2.0',
      method: 'runAction',
      params: {
        key: req.key,
        input: req.input,
        init: req.init,
        context: req.context,
        telemetryLabels: req.telemetryLabels,
        stream: !!opts?.onChunk,
        streamInput: false,
      },
      id,
    };

    return new Promise<RunActionResult<O>>((resolve, reject) => {
      this.pendingRequests.set(id, {
        resolve: (v) => resolve(v as RunActionResult<O>),
        reject,
      });
      runtime.ws.send(JSON.stringify(message));
    }).finally(cleanup);
  }

  /** Cancels an in-flight action by trace id. */
  async cancelAction(
    runtimeId: string,
    traceId: string
  ): Promise<{ message: string }> {
    return (await this.sendRequest(
      runtimeId,
      'cancelAction',
      { traceId },
      10_000
    )) as { message: string };
  }

  /** Sends a `configure` notification (e.g. telemetry URL) to a runtime. */
  configure(runtimeId: string, telemetryServerUrl?: string): void {
    this.sendNotification(runtimeId, 'configure', { telemetryServerUrl });
  }

  /** Stops the WebSocket server and drops all connections. */
  async stop(): Promise<void> {
    for (const { ws } of this.runtimes.values()) {
      try {
        ws.close();
      } catch {
        // best effort
      }
    }
    this.runtimes.clear();
    if (this.wss) {
      await new Promise<void>((resolve) => this.wss!.close(() => resolve()));
      this.wss = undefined;
    }
  }
}
