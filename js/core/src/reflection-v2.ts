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

import WebSocket from 'ws';
import { StatusCodes, type Status } from './action.js';
import { GENKIT_REFLECTION_API_SPEC_VERSION, GENKIT_VERSION } from './index.js';
import { logger } from './logging.js';
import {
  ReflectionCancelActionParamsSchema,
  ReflectionConfigureParamsSchema,
  ReflectionListActionsResponse,
  ReflectionListValuesParamsSchema,
  ReflectionListValuesResponseSchema,
  ReflectionRegisterParams,
  ReflectionRunActionParamsSchema,
  ReflectionRunActionStateParamsSchema,
  ReflectionStreamChunkParamsSchema,
} from './reflection-types.js';
import type { Registry } from './registry.js';
import { toJsonSchema } from './schema.js';
import { flushTracing, setTelemetryServerUrl } from './tracing.js';

let apiIndex = 0;

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

export interface ReflectionServerV2Options {
  configuredEnvs?: string[];
  name?: string;
  url: string;
}

export class ReflectionServerV2 {
  private registry: Registry;
  private options: ReflectionServerV2Options;
  private ws: WebSocket | null = null;
  private url: string;
  private index = apiIndex++;
  private activeActions = new Map<
    string,
    {
      abortController: AbortController;
      startTime: Date;
    }
  >();
  private reconnectCount = 0;
  private isStopped = false;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private baseDelayMs = 500;
  private maxDelayMs = 5000;
  private pendingRequests = new Map<
    string,
    {
      resolve: (value: any) => void;
      reject: (reason?: any) => void;
    }
  >();
  private requestIdCounter = 0;

  constructor(registry: Registry, options: ReflectionServerV2Options) {
    this.registry = registry;
    this.options = {
      configuredEnvs: ['dev'],
      ...options,
    };
    // The URL should be provided via environment variable by the CLI manager
    this.url = this.options.url;
  }

  async start() {
    this.isStopped = false;
    this.reconnectCount = 0;
    await this.connect();
  }

  private async connect() {
    if (this.isStopped) return;

    logger.debug(`Connecting to Reflection V2 server at ${this.url}`);
    const ws = new WebSocket(this.url);
    this.ws = ws;

    this.ws.on('open', async () => {
      logger.debug('Connected to Reflection V2 server.');
      this.reconnectCount = 0;
      await this.register();
    });

    this.ws.on('message', async (data) => {
      try {
        const message = JSON.parse(data.toString()) as any;
        if ('method' in message) {
          await this.handleRequest(message);
        } else if ('id' in message) {
          this.handleResponse(message);
        }
      } catch (error) {
        logger.error(`Failed to parse message: ${error}`);
      }
    });

    this.ws.on('error', (error) => {
      logger.error(`Reflection V2 WebSocket error: ${error}`);
    });

    this.ws.on('close', (code, reason) => {
      logger.debug(
        `Reflection V2 WebSocket closed. Code: ${code}, Reason: ${reason}`
      );
      for (const [id, resolver] of this.pendingRequests.entries()) {
        resolver.reject(
          new Error(
            `Connection closed before response was received (id: ${id})`
          )
        );
      }
      this.pendingRequests.clear();

      if (!this.isStopped) {
        this.scheduleReconnect();
      }
    });
  }

  private scheduleReconnect() {
    if (this.reconnectTimeout) return;

    const delay = Math.min(
      this.baseDelayMs * Math.pow(2, this.reconnectCount),
      this.maxDelayMs
    );
    this.reconnectCount++;

    logger.debug(
      `Scheduling reconnection in ${delay}ms (attempt ${this.reconnectCount})`
    );

    this.reconnectTimeout = setTimeout(async () => {
      this.reconnectTimeout = null;
      await this.connect();
    }, delay);
  }

  async stop() {
    this.isStopped = true;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private send(message: JsonRpcMessage) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private sendResponse(id: string, result: any) {
    this.send({
      jsonrpc: '2.0',
      result,
      id,
    });
  }

  private sendError(id: string, code: number, message: string, data?: any) {
    this.send({
      jsonrpc: '2.0',
      error: { code, message, data },
      id,
    });
  }

  private sendNotification(method: string, params: any) {
    this.send({
      jsonrpc: '2.0',
      method,
      params,
    });
  }

  private sendRequest(method: string, params: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = (++this.requestIdCounter).toString();
      this.pendingRequests.set(id, { resolve, reject });
      this.send({
        jsonrpc: '2.0',
        id,
        method,
        params,
      });
    });
  }

  private async register() {
    const params: ReflectionRegisterParams = {
      id: process.env.GENKIT_RUNTIME_ID || this.runtimeId,
      pid: process.pid,
      name: this.options.name || this.runtimeId,
      genkitVersion: GENKIT_VERSION,
      reflectionApiSpecVersion: GENKIT_REFLECTION_API_SPEC_VERSION,
      envs: this.options.configuredEnvs,
    };
    try {
      const response = await this.sendRequest('register', params);
      if (response && response.telemetryServerUrl) {
        if (!process.env.GENKIT_TELEMETRY_SERVER) {
          setTelemetryServerUrl(response.telemetryServerUrl);
          logger.debug(
            `Connected to telemetry server on ${response.telemetryServerUrl} via handshake`
          );
        }
      }
    } catch (err) {
      logger.error(`Failed to register with CLI: ${err}`);
    }
  }

  get runtimeId() {
    return `${process.pid}${this.index ? `-${this.index}` : ''}`;
  }

  private handleResponse(response: any) {
    const resolver = this.pendingRequests.get(response.id);
    if (!resolver) {
      logger.error(`Unknown response ID: ${response.id}`);
      return;
    }
    this.pendingRequests.delete(response.id);
    if ('error' in response) {
      resolver.reject(response.error);
    } else {
      resolver.resolve(response.result);
    }
  }

  private async handleRequest(request: JsonRpcRequest) {
    try {
      switch (request.method) {
        case 'listActions':
          await this.handleListActions(request);
          break;
        case 'listValues':
          await this.handleListValues(request);
          break;
        case 'runAction':
          await this.handleRunAction(request);
          break;
        case 'configure':
          this.handleConfigure(request);
          break;
        case 'cancelAction':
          await this.handleCancelAction(request);
          break;
        case 'sendInputStreamChunk':
          this.handleSendInputStreamChunk(request);
          break;
        case 'endInputStream':
          this.handleEndInputStream(request);
          break;
        default:
          if (request.id) {
            this.sendError(
              request.id,
              -32601,
              `Method not found: ${request.method}`
            );
          }
      }
    } catch (error: any) {
      if (request.id) {
        this.sendError(request.id, -32000, error.message, {
          stack: error.stack,
        });
      }
    }
  }

  private async handleListActions(request: JsonRpcRequest) {
    if (!request.id) return; // Should be a request
    const actions = await this.registry.listResolvableActions();
    const convertedActions: Record<string, any> = {};

    Object.keys(actions).forEach((key) => {
      const action = actions[key];
      convertedActions[key] = {
        key,
        name: action.name,
        description: action.description,
        metadata: action.metadata,
      };
      if (action.inputSchema || action.inputJsonSchema) {
        convertedActions[key].inputSchema = toJsonSchema({
          schema: action.inputSchema,
          jsonSchema: action.inputJsonSchema,
        });
      }
      if (action.outputSchema || action.outputJsonSchema) {
        convertedActions[key].outputSchema = toJsonSchema({
          schema: action.outputSchema,
          jsonSchema: action.outputJsonSchema,
        });
      }
    });

    this.sendResponse(request.id, <ReflectionListActionsResponse>{
      actions: convertedActions,
    });
  }

  private async handleListValues(request: JsonRpcRequest) {
    if (!request.id) return;
    const { type } = ReflectionListValuesParamsSchema.parse(request.params);
    if (type !== 'defaultModel' && type !== 'middleware') {
      this.sendError(
        request.id,
        -32602,
        `'type' ${type} is not supported. Only 'defaultModel' and 'middleware' are supported`
      );
      return;
    }
    const values = await this.registry.listValues(type);
    const mappedValues: Record<string, any> = {};
    for (const [key, value] of Object.entries(values)) {
      mappedValues[key] =
        value &&
        typeof value === 'object' &&
        'toJson' in value &&
        typeof (value as any).toJson === 'function'
          ? (value as any).toJson()
          : value;
    }
    this.sendResponse(
      request.id,
      ReflectionListValuesResponseSchema.parse({ values: mappedValues })
    );
  }

  private async handleRunAction(request: JsonRpcRequest) {
    if (!request.id) return;

    const { key, input, context, telemetryLabels, stream } =
      ReflectionRunActionParamsSchema.parse(request.params);
    const action = await this.registry.lookupAction(key);

    if (!action) {
      this.sendError(request.id, -32602, `action ${key} not found`);
      return;
    }

    const abortController = new AbortController();
    let traceId: string | undefined;

    try {
      const onTraceStartCallback = ({ traceId: tid }: { traceId: string }) => {
        traceId = tid;
        this.activeActions.set(tid, {
          abortController,
          startTime: new Date(),
        });
        // Send early trace ID notification
        this.sendNotification(
          'runActionState',
          ReflectionRunActionStateParamsSchema.parse({
            requestId: request.id,
            state: { traceId: tid },
          })
        );
      };

      if (stream) {
        const callback = (chunk: any) => {
          this.sendNotification(
            'streamChunk',
            ReflectionStreamChunkParamsSchema.parse({
              requestId: request.id,
              chunk,
            })
          );
        };

        const result = await action.run(input, {
          context,
          onChunk: callback,
          telemetryLabels,
          onTraceStart: onTraceStartCallback,
          abortSignal: abortController.signal,
        });

        await flushTracing();

        // Send final result
        this.sendResponse(request.id, {
          result: result.result,
          telemetry: {
            traceId: result.telemetry.traceId,
          },
        });
      } else {
        const result = await action.run(input, {
          context,
          telemetryLabels,
          onTraceStart: onTraceStartCallback,
          abortSignal: abortController.signal,
        });
        await flushTracing();

        this.sendResponse(request.id, {
          result: result.result,
          telemetry: {
            traceId: result.telemetry.traceId,
          },
        });
      }
    } catch (err: any) {
      const isAbort =
        err?.name === 'AbortError' ||
        (typeof DOMException !== 'undefined' &&
          err instanceof DOMException &&
          err.name === 'AbortError');

      const errorResponse: Status = {
        code: isAbort ? StatusCodes.CANCELLED : StatusCodes.INTERNAL,
        message: isAbort ? 'Action was cancelled' : err.message,
        details: {
          stack: err.stack,
        },
      };
      if (err.traceId || traceId) {
        errorResponse.details.traceId = err.traceId || traceId;
      }

      this.sendError(request.id, -32000, errorResponse.message, errorResponse);
    } finally {
      if (traceId) {
        this.activeActions.delete(traceId);
      }
    }
  }

  private handleConfigure(request: JsonRpcRequest) {
    const { telemetryServerUrl } = ReflectionConfigureParamsSchema.parse(
      request.params
    );
    if (telemetryServerUrl && !process.env.GENKIT_TELEMETRY_SERVER) {
      setTelemetryServerUrl(telemetryServerUrl);
      logger.debug(`Connected to telemetry server on ${telemetryServerUrl}`);
    }
  }

  private async handleCancelAction(request: JsonRpcRequest) {
    if (!request.id) return;
    const { traceId } = ReflectionCancelActionParamsSchema.parse(
      request.params
    );
    const activeAction = this.activeActions.get(traceId);
    if (activeAction) {
      activeAction.abortController.abort();
      this.activeActions.delete(traceId);
      this.sendResponse(request.id, { message: 'Action cancelled' });
    } else {
      this.sendError(
        request.id,
        -32602,
        'Action not found or already completed'
      );
    }
  }

  private handleSendInputStreamChunk(request: JsonRpcRequest) {
    // ReflectionSendInputStreamChunkParamsSchema.parse(request.params);
    throw new Error('Not implemented');
  }

  private handleEndInputStream(request: JsonRpcRequest) {
    // ReflectionEndInputStreamParamsSchema.parse(request.params);
    throw new Error('Not implemented');
  }
}
