/**
 * Copyright 2024 Google LLC
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

import EventEmitter from 'events';
import getPort, { makeRange } from 'get-port';
import path from 'path';
import { WebSocket, WebSocketServer } from 'ws';
import {
  Action,
  RunActionResponse,
  RunActionResponseSchema,
} from '../types/action';
import * as apis from '../types/apis';
import {
  ReflectionCancelActionParamsSchema,
  ReflectionCancelActionResponseSchema,
  ReflectionConfigureParamsSchema,
  ReflectionEndInputStreamParamsSchema,
  ReflectionListActionsResponseSchema,
  ReflectionListValuesParamsSchema,
  ReflectionListValuesResponseSchema,
  ReflectionRegisterParamsSchema,
  ReflectionRunActionParamsSchema,
  ReflectionRunActionStateParamsSchema,
  ReflectionSendInputStreamChunkParamsSchema,
  ReflectionStreamChunkParamsSchema,
} from '../types/reflection';
import { logger } from '../utils/logger';
import { DevToolsInfo } from '../utils/utils';
import { BaseRuntimeManager, RuntimeManagerOptions } from './manager';
import { ProcessManager } from './process-manager';
import {
  GenkitToolsError,
  RuntimeEvent,
  RuntimeInfo,
  StreamingCallback,
} from './types';

interface JsonRpcRequest {
  jsonrpc: '2.0';
  method: string;
  params?: any;
  id?: string;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
  id: string;
}

type JsonRpcMessage = JsonRpcRequest | JsonRpcResponse;

interface ConnectedRuntime {
  ws: WebSocket;
  info: RuntimeInfo;
}

export class RuntimeManagerV2 extends BaseRuntimeManager {
  private _port?: number;
  private wss?: WebSocketServer;
  private runtimes: Map<string, ConnectedRuntime> = new Map();

  get port(): number | undefined {
    return this._port;
  }
  private pendingRequests: Map<
    number | string,
    {
      method: string;
      resolve: (value: any) => void;
      reject: (reason?: any) => void;
    }
  > = new Map();
  private streamCallbacks: Map<number | string, StreamingCallback<any>> =
    new Map();
  private traceIdCallbacks: Map<number | string, (traceId: string) => void> =
    new Map();
  private eventEmitter = new EventEmitter();
  private requestIdCounter = 0;

  constructor(
    telemetryServerUrl: string | undefined,
    readonly manageHealth: boolean,
    projectRoot: string,
    processManager?: ProcessManager,
    disableRealtimeTelemetry: boolean = false
  ) {
    super(
      telemetryServerUrl,
      projectRoot,
      processManager,
      disableRealtimeTelemetry
    );
  }

  static async create(
    options: RuntimeManagerOptions
  ): Promise<RuntimeManagerV2> {
    const manager = new RuntimeManagerV2(
      options.telemetryServerUrl,
      options.manageHealth ?? true,
      options.projectRoot,
      options.processManager,
      options.disableRealtimeTelemetry
    );
    await manager.startWebSocketServer(options.reflectionV2Port);
    return manager;
  }

  /**
   * Starts a WebSocket server.
   */
  private async startWebSocketServer(port?: number): Promise<{ port: number }> {
    if (!port) {
      port = await getPort({ port: makeRange(3200, 3400) });
    }
    this.wss = new WebSocketServer({ port });

    this._port = port;
    logger.info(`Starting reflection server: ws://localhost:${port}`);

    this.wss.on('connection', (ws) => {
      ws.on('error', (err) => logger.error(`WebSocket error: ${err}`));

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString()) as JsonRpcMessage;
          this.handleMessage(ws, message);
        } catch (error) {
          logger.error('Failed to parse WebSocket message:', error);
        }
      });

      ws.on('close', () => {
        this.handleDisconnect(ws);
      });
    });
    return { port };
  }

  private handleMessage(ws: WebSocket, message: JsonRpcMessage) {
    if ('method' in message) {
      this.handleRequest(ws, message as JsonRpcRequest);
    } else {
      this.handleResponse(message as JsonRpcResponse);
    }
  }

  private handleRequest(ws: WebSocket, request: JsonRpcRequest) {
    switch (request.method) {
      case 'register':
        this.handleRegister(ws, request);
        break;
      case 'streamChunk':
        this.handleStreamChunk(request);
        break;
      case 'runActionState':
        this.handleRunActionState(request);
        break;
      default:
        logger.warn(`Unknown method: ${request.method}`);
    }
  }

  private handleRegister(ws: WebSocket, request: JsonRpcRequest) {
    const params = ReflectionRegisterParamsSchema.parse(request.params);
    const runtimeInfo: RuntimeInfo = {
      id: params.id,
      pid: params.pid,
      name: params.name,
      genkitVersion: params.genkitVersion,
      reflectionApiSpecVersion: params.reflectionApiSpecVersion,
      reflectionServerUrl: `ws://localhost:${this.port}`, // Virtual URL for compatibility
      timestamp: new Date().toISOString(),
      projectName: path.basename(this.projectRoot), // Or derive from other means if needed
    };

    this.runtimes.set(runtimeInfo.id, { ws, info: runtimeInfo });
    this.eventEmitter.emit(RuntimeEvent.ADD, runtimeInfo);

    // Send success response
    if (request.id) {
      ws.send(
        JSON.stringify({
          jsonrpc: '2.0',
          result: {
            telemetryServerUrl: this.telemetryServerUrl,
          },
          id: request.id,
        })
      );
    }
  }

  private handleStreamChunk(notification: JsonRpcRequest) {
    const { requestId, chunk } = ReflectionStreamChunkParamsSchema.parse(
      notification.params
    );
    const callback = this.streamCallbacks.get(requestId);
    if (callback) {
      callback(chunk);
    }
  }

  private handleRunActionState(notification: JsonRpcRequest) {
    const { requestId, state } = ReflectionRunActionStateParamsSchema.parse(
      notification.params
    );
    const callback = this.traceIdCallbacks.get(requestId);
    if (callback && state?.traceId) {
      callback(state.traceId);
    }
  }

  private handleResponse(response: JsonRpcResponse) {
    const pending = this.pendingRequests.get(response.id);
    if (pending) {
      if (response.error) {
        const errorData = response.error.data || {};
        const massagedData = {
          ...errorData,
          stack: errorData.details?.stack,
          data: {
            genkitErrorMessage: errorData.message,
            genkitErrorDetails: errorData.details,
          },
        };
        const error = new GenkitToolsError(response.error.message);
        error.data = massagedData;
        pending.reject(error);
      } else {
        let result = response.result;
        if (pending.method === 'listActions') {
          result = ReflectionListActionsResponseSchema.parse(result);
        } else if (pending.method === 'listValues') {
          result = ReflectionListValuesResponseSchema.parse(result);
        } else if (pending.method === 'cancelAction') {
          result = ReflectionCancelActionResponseSchema.parse(result);
        }
        pending.resolve(result);
      }
      this.pendingRequests.delete(response.id);
    } else {
      logger.warn(`Received response for unknown request ID ${response.id}`);
    }
  }

  private handleDisconnect(ws: WebSocket) {
    for (const [id, runtime] of this.runtimes.entries()) {
      if (runtime.ws === ws) {
        this.runtimes.delete(id);
        this.eventEmitter.emit(RuntimeEvent.REMOVE, runtime.info);
        break;
      }
    }
  }

  private async sendRequest(
    runtimeId: string,
    method: string,
    params?: any
  ): Promise<any> {
    const runtime = this.runtimes.get(runtimeId);
    if (!runtime) {
      throw new Error(`Runtime ${runtimeId} not found`);
    }

    switch (method) {
      case 'listValues':
        ReflectionListValuesParamsSchema.parse(params);
        break;
      case 'runAction':
        ReflectionRunActionParamsSchema.parse(params);
        break;
      case 'cancelAction':
        ReflectionCancelActionParamsSchema.parse(params);
        break;
    }

    const id = (++this.requestIdCounter).toString();
    const message: JsonRpcRequest = {
      jsonrpc: '2.0',
      method,
      params,
      id,
    };

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${id} timed out`));
        }
      }, 30000);

      this.pendingRequests.set(id, {
        method,
        resolve: (value) => {
          clearTimeout(timeoutId);
          resolve(value);
        },
        reject: (reason) => {
          clearTimeout(timeoutId);
          reject(reason);
        },
      });

      runtime.ws.send(JSON.stringify(message));
    });
  }

  private sendNotification(runtimeId: string, method: string, params?: any) {
    const runtime = this.runtimes.get(runtimeId);
    if (!runtime) {
      logger.warn(`Runtime ${runtimeId} not found, cannot send notification`);
      return;
    }

    switch (method) {
      case 'configure':
        ReflectionConfigureParamsSchema.parse(params);
        break;
      case 'sendInputStreamChunk':
        ReflectionSendInputStreamChunkParamsSchema.parse(params);
        break;
      case 'endInputStream':
        ReflectionEndInputStreamParamsSchema.parse(params);
        break;
    }

    const message: JsonRpcRequest = {
      jsonrpc: '2.0',
      method,
      params,
    };
    runtime.ws.send(JSON.stringify(message));
  }

  listRuntimes(): RuntimeInfo[] {
    return Array.from(this.runtimes.values()).map((r) => r.info);
  }

  getRuntimeById(id: string): RuntimeInfo | undefined {
    return this.runtimes.get(id)?.info;
  }

  getMostRecentRuntime(): RuntimeInfo | undefined {
    const runtimes = this.listRuntimes();
    if (runtimes.length === 0) return undefined;
    return runtimes[runtimes.length - 1];
  }

  getMostRecentDevUI(): DevToolsInfo | undefined {
    // Not applicable for V2 yet
    return undefined;
  }

  onRuntimeEvent(
    listener: (eventType: RuntimeEvent, runtime: RuntimeInfo) => void
  ) {
    const listeners: Array<{ event: string; fn: (rt: RuntimeInfo) => void }> =
      [];
    Object.values(RuntimeEvent).forEach((event) => {
      const fn = (rt: RuntimeInfo) => listener(event, rt);
      this.eventEmitter.on(event, fn);
      listeners.push({ event, fn });
    });
    return () => {
      listeners.forEach(({ event, fn }) => {
        this.eventEmitter.off(event, fn);
      });
    };
  }

  async listActions(
    input?: apis.ListActionsRequest
  ): Promise<Record<string, Action>> {
    const runtimeId = input?.runtimeId || this.getMostRecentRuntime()?.id;
    if (!runtimeId) {
      throw new Error(
        input?.runtimeId
          ? `No runtime found with ID ${input.runtimeId}.`
          : 'No runtimes found. Make sure your app is running using the `start_runtime` MCP tool or the CLI: `genkit start -- ...`. See getting started documentation.'
      );
    }
    const result: Record<string, Action> = {};
    const response = ReflectionListActionsResponseSchema.parse(
      await this.sendRequest(runtimeId, 'listActions')
    );
    // make sure key is set on the action metadata
    for (const key of Object.keys(response.actions)) {
      const action = response.actions[key];
      if (!action.key) {
        action.key = key;
      }
      result[key] = action as Action;
    }
    return result;
  }

  async listValues(
    input: apis.ListValuesRequest
  ): Promise<Record<string, unknown>> {
    const runtimeId = input?.runtimeId || this.getMostRecentRuntime()?.id;
    if (!runtimeId) {
      throw new Error(
        input?.runtimeId
          ? `No runtime found with ID ${input.runtimeId}.`
          : 'No runtimes found. Make sure your app is running using `genkit start -- ...`. See getting started documentation.'
      );
    }
    const response = await this.sendRequest(runtimeId, 'listValues', {
      type: input.type,
    });
    return response.values;
  }

  async stop() {
    if (this.wss) {
      this.wss.close();
    }
    if (this.processManager) {
      await this.processManager.kill();
    }
  }

  async runAction(
    input: apis.RunActionRequest,
    streamingCallback?: StreamingCallback<any>,
    onTraceId?: (traceId: string) => void,
    inputStream?: AsyncIterable<any>
  ): Promise<RunActionResponse> {
    const runtimeId = input.runtimeId || this.getMostRecentRuntime()?.id;
    if (!runtimeId) {
      throw new Error(
        'No runtimes found. Make sure your app is running using the `start_runtime` MCP tool or the CLI: `genkit start -- ...`. See getting started documentation.'
      );
    }

    const runtime = this.runtimes.get(runtimeId);
    if (!runtime) {
      throw new Error(`Runtime ${runtimeId} not found`);
    }

    const id = (++this.requestIdCounter).toString();

    if (streamingCallback) {
      this.streamCallbacks.set(id, streamingCallback);
    }
    if (onTraceId) {
      this.traceIdCallbacks.set(id, onTraceId!);
    }

    const message: JsonRpcRequest = {
      jsonrpc: '2.0',
      method: 'runAction',
      params: {
        ...input,
        stream: !!streamingCallback,
        streamInput: !!inputStream,
      },
      id,
    };

    const promise = new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { method: 'runAction', resolve, reject });
      runtime!.ws.send(JSON.stringify(message));
    })
      .then((result) => {
        return RunActionResponseSchema.parse(result);
      })
      .finally(() => {
        if (streamingCallback) {
          this.streamCallbacks.delete(id);
        }
        if (onTraceId) {
          this.traceIdCallbacks.delete(id);
        }
      });

    if (inputStream) {
      (async () => {
        try {
          for await (const chunk of inputStream!) {
            this.sendNotification(runtimeId!, 'sendInputStreamChunk', {
              requestId: id,
              chunk,
            });
          }
          this.sendNotification(runtimeId!, 'endInputStream', {
            requestId: id,
          });
        } catch (e) {
          logger.error(`Error streaming input: ${e}`);
        }
      })();
    }

    return promise as Promise<RunActionResponse>;
  }

  async cancelAction(input: {
    traceId: string;
    runtimeId?: string;
  }): Promise<{ message: string }> {
    const runtimeId = input.runtimeId || this.getMostRecentRuntime()?.id;
    if (!runtimeId) {
      throw new Error('No runtime found');
    }
    // Assuming cancelAction is a request that returns a message
    return this.sendRequest(runtimeId, 'cancelAction', {
      traceId: input.traceId,
    });
  }
}
