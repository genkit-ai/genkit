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

import axios, { type AxiosError } from 'axios';
import chokidar from 'chokidar';
import EventEmitter from 'events';
import * as fsSync from 'fs';
import fs from 'fs/promises';
import path from 'path';
import { GenkitError } from '../types';
import {
  RunActionResponseSchema,
  type Action,
  type RunActionResponse,
} from '../types/action';
import * as apis from '../types/apis';
import type { TraceData } from '../types/trace';
import { logger } from '../utils/logger';
import {
  checkServerHealth,
  findRuntimesDir,
  findServersDir,
  isValidDevToolsInfo,
  projectNameFromGenkitFilePath,
  removeToolsInfoFile,
  retriable,
  type DevToolsInfo,
} from '../utils/utils';
import { ProcessManager } from './process-manager';
import {
  GenkitToolsError,
  RuntimeEvent,
  type RuntimeInfo,
  type StreamingCallback,
} from './types';

const STREAM_DELIMITER = '\n';
const HEALTH_CHECK_INTERVAL = 5000;
export const GENKIT_REFLECTION_API_SPEC_VERSION = 1;

export interface RuntimeManagerOptions {
  /** URL of the telemetry server. */
  telemetryServerUrl?: string;
  /** Whether to clean up unhealthy runtimes. */
  manageHealth?: boolean;
  /** Project root dir. If not provided will be inferred from CWD. */
  projectRoot: string;
  /** An optional process manager for the main application process. */
  processManager?: ProcessManager;
  /** Whether to disable realtime telemetry streaming. Defaults to false. */
  disableRealtimeTelemetry?: boolean;
  /** Experimental Reflection V2 flag */
  experimentalReflectionV2?: boolean;
  /** Reflection V2 Port */
  reflectionV2Port?: number;
}

export abstract class BaseRuntimeManager {
  constructor(
    readonly telemetryServerUrl: string | undefined,
    readonly projectRoot: string,
    readonly processManager?: ProcessManager,
    readonly disableRealtimeTelemetry: boolean = false
  ) {}

  abstract listRuntimes(): RuntimeInfo[];
  abstract getRuntimeById(id: string): RuntimeInfo | undefined;
  abstract getMostRecentRuntime(): RuntimeInfo | undefined;
  abstract getMostRecentDevUI(): DevToolsInfo | undefined;
  abstract onRuntimeEvent(
    listener: (eventType: RuntimeEvent, runtime: RuntimeInfo) => void
  ): () => void;
  abstract listActions(
    input?: apis.ListActionsRequest
  ): Promise<Record<string, Action>>;
  abstract runAction(
    input: apis.RunActionRequest,
    streamingCallback?: StreamingCallback<any>,
    onTraceId?: (traceId: string) => void,
    inputStream?: AsyncIterable<any>
  ): Promise<RunActionResponse>;
  abstract cancelAction(input: {
    traceId: string;
    runtimeId?: string;
  }): Promise<{ message: string }>;
  abstract listValues(
    input: apis.ListValuesRequest
  ): Promise<Record<string, unknown>>;

  abstract stop(): Promise<void>;

  /**
   * Retrieves all traces
   */
  async listTraces(
    input: apis.ListTracesRequest
  ): Promise<apis.ListTracesResponse> {
    const { limit, continuationToken, filter } = input;
    let query = '';
    if (limit) {
      query += `limit=${limit}`;
    }
    if (continuationToken) {
      if (query !== '') {
        query += '&';
      }
      query += `continuationToken=${continuationToken}`;
    }
    if (filter) {
      if (query !== '') {
        query += '&';
      }
      query += `filter=${encodeURI(JSON.stringify(filter))}`;
    }

    const response = await axios
      .get(`${this.telemetryServerUrl}/api/traces?${query}`)
      .catch((err) =>
        this.httpErrorHandler(err, `Error listing traces for query='${query}'.`)
      );

    return apis.ListTracesResponseSchema.parse(response.data);
  }

  /**
   * Retrieves a trace for a given ID.
   */
  async getTrace(input: apis.GetTraceRequest): Promise<TraceData> {
    const { traceId } = input;
    const response = await axios
      .get(`${this.telemetryServerUrl}/api/traces/${traceId}`)
      .catch((err) =>
        this.httpErrorHandler(
          err,
          `Error getting trace for traceId='${traceId}'`
        )
      );

    return response.data as TraceData;
  }

  /**
   * Streams trace updates in real-time from the telemetry server.
   * Connects to the telemetry server's SSE endpoint and forwards updates via callback.
   */
  async streamTrace(
    input: apis.StreamTraceRequest,
    streamingCallback: StreamingCallback<any>
  ): Promise<void> {
    const { traceId } = input;

    if (!this.telemetryServerUrl) {
      throw new Error(
        'Telemetry server URL not configured. Cannot stream trace updates.'
      );
    }

    const response = await axios
      .get(`${this.telemetryServerUrl}/api/traces/${traceId}/stream`, {
        headers: {
          Accept: 'text/event-stream',
        },
        responseType: 'stream',
      })
      .catch((err) =>
        this.httpErrorHandler(
          err,
          `Error streaming trace for traceId='${traceId}'`
        )
      );

    const stream = response.data;
    let buffer = '';

    // Return a promise that resolves when the stream ends
    return new Promise<void>((resolve, reject) => {
      stream.on('data', (chunk: Buffer) => {
        buffer += chunk.toString();

        // Process complete messages (ending with \n\n)
        while (buffer.includes('\n\n')) {
          const messageEnd = buffer.indexOf('\n\n');
          const message = buffer.substring(0, messageEnd).trim();
          buffer = buffer.substring(messageEnd + 2);

          // Skip empty messages
          if (!message) {
            continue;
          }
          // Parse SSE data line - strip "data: " prefix
          try {
            const jsonData = message.startsWith('data: ')
              ? message.slice(6)
              : message;
            const parsed = JSON.parse(jsonData);
            streamingCallback(parsed);
          } catch (err) {
            logger.error(`Error parsing stream data: ${err}`);
          }
        }
      });

      stream.on('end', () => {
        resolve();
      });

      stream.on('error', (err: Error) => {
        logger.error(`Stream error for traceId='${traceId}': ${err}`);
        reject(err);
      });
    });
  }

  /**
   * Adds a trace to the trace store
   */
  async addTrace(input: TraceData): Promise<void> {
    await axios
      .post(`${this.telemetryServerUrl}/api/traces/`, input)
      .catch((err) =>
        this.httpErrorHandler(err, 'Error writing trace to store.')
      );
  }

  /**
   * Handles an HTTP error.
   */
  protected httpErrorHandler(error: AxiosError, message?: string): never {
    const newError = new GenkitToolsError(message || 'Internal Error');

    if (error.response) {
      // we got a non-200 response; copy the payload and rethrow
      newError.data = error.response.data as GenkitError;
      newError.stack = (error.response?.data as any).message;
      if ((error.response?.data as any).message) {
        newError.data.data = {
          ...newError.data.data,
          genkitErrorMessage: message,
          genkitErrorDetails: {
            stack: (error.response?.data as any).message,
            traceId: (error.response?.data as any).traceId,
          },
        };
      }
      throw newError;
    }

    // We actually have an exception; wrap it and re-throw.
    throw new GenkitToolsError(message || 'Internal Error', {
      cause: error.cause,
    });
  }
}

export class RuntimeManager extends BaseRuntimeManager {
  private filenameToRuntimeMap: Record<string, RuntimeInfo> = {};
  private filenameToDevUiMap: Record<string, DevToolsInfo> = {};
  private idToFileMap: Record<string, string> = {};
  private eventEmitter = new EventEmitter();
  private watchers: chokidar.FSWatcher[] = [];
  private healthCheckInterval?: NodeJS.Timeout;

  private constructor(
    telemetryServerUrl: string | undefined,
    private manageHealth: boolean,
    projectRoot: string,
    processManager?: ProcessManager,
    disableRealtimeTelemetry?: boolean
  ) {
    super(
      telemetryServerUrl,
      projectRoot,
      processManager,
      disableRealtimeTelemetry
    );
  }

  /**
   * Creates a new runtime manager.
   */
  static async create(
    options: RuntimeManagerOptions
  ): Promise<BaseRuntimeManager> {
    if (options.experimentalReflectionV2) {
      const { RuntimeManagerV2 } = await import('./manager-v2');
      return RuntimeManagerV2.create(options);
    }

    const manager = new RuntimeManager(
      options.telemetryServerUrl,
      options.manageHealth ?? true,
      options.projectRoot,
      options.processManager,
      options.disableRealtimeTelemetry
    );
    await manager.setupRuntimesWatcher();
    await manager.setupDevUiWatcher();
    if (manager.manageHealth) {
      manager.healthCheckInterval = setInterval(
        async () => await manager.performHealthChecks(),
        HEALTH_CHECK_INTERVAL
      );
    }
    return manager;
  }

  /**
   * Stops the runtime manager and cleans up resources.
   */
  async stop() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    await Promise.all(this.watchers.map((watcher) => watcher.close()));
    if (this.processManager) {
      await this.processManager.kill();
    }
  }

  /**
   * Lists all active runtimes
   */
  listRuntimes(): RuntimeInfo[] {
    return Object.values(this.filenameToRuntimeMap);
  }

  /**
   * Gets the runtime info for a given ID.
   */
  getRuntimeById(id: string): RuntimeInfo | undefined {
    const fileName = this.idToFileMap[id];
    return fileName ? this.filenameToRuntimeMap[fileName] : undefined;
  }

  /**
   * Gets the runtime that was started most recently.
   */
  getMostRecentRuntime(): RuntimeInfo | undefined {
    const runtimes = Object.values(this.filenameToRuntimeMap);
    return runtimes.length === 0
      ? undefined
      : runtimes.reduce((a, b) =>
          new Date(a.timestamp) > new Date(b.timestamp) ? a : b
        );
  }

  /**
   * Gets the Dev UI that was started most recently.
   */
  getMostRecentDevUI(): DevToolsInfo | undefined {
    const toolsInfo = Object.values(this.filenameToDevUiMap);
    return toolsInfo.length === 0
      ? undefined
      : toolsInfo.reduce((a, b) =>
          new Date(a.timestamp) > new Date(b.timestamp) ? a : b
        );
  }

  /**
   * Subscribe to changes to the available runtimes. e.g.) whenever a new
   * runtime is added or removed.
   *
   * The `listener` will be called with the `eventType` that occured and the
   * `runtime` to which it applies.
   *
   * @param listener the callback function.
   * @returns an unsubscriber function.
   */
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

  /**
   * Retrieves all runnable actions.
   */
  async listActions(
    input?: apis.ListActionsRequest
  ): Promise<Record<string, Action>> {
    const runtime = input?.runtimeId
      ? this.getRuntimeById(input.runtimeId)
      : this.getMostRecentRuntime();
    if (!runtime) {
      throw new Error(
        input?.runtimeId
          ? `No runtime found with ID ${input.runtimeId}.`
          : 'No runtimes found. Make sure your app is running using the `start_runtime` MCP tool or the CLI: `genkit start -- ...`. See getting started documentation.'
      );
    }
    const response = await axios
      .get(`${runtime.reflectionServerUrl}/api/actions`)
      .catch((err) => this.httpErrorHandler(err, 'Error listing actions.'));
    return response.data as Record<string, Action>;
  }

  /**
   * Retrieves all values.
   */
  async listValues(
    input: apis.ListValuesRequest
  ): Promise<Record<string, unknown>> {
    const runtime = input.runtimeId
      ? this.getRuntimeById(input.runtimeId)
      : this.getMostRecentRuntime();
    if (!runtime) {
      throw new Error(
        input?.runtimeId
          ? `No runtime found with ID ${input.runtimeId}.`
          : 'No runtimes found. Make sure your app is running using `genkit start -- ...`. See getting started documentation.'
      );
    }
    try {
      const response = await axios.get(
        `${runtime.reflectionServerUrl}/api/values`,
        {
          params: {
            type: input.type,
          },
        }
      );
      return response.data as Record<string, unknown>;
    } catch (err) {
      if ((err as AxiosError).response?.status === 404) {
        return {};
      } else if ((err as AxiosError).response?.status === 400) {
        throw new GenkitToolsError(
          `Bad request: ${(err as AxiosError).response?.data}`
        );
      }
      this.httpErrorHandler(err as AxiosError, 'Error listing values.');
    }
  }

  /**
   * Runs an action.
   */
  async runAction(
    input: apis.RunActionRequest,
    streamingCallback?: StreamingCallback<any>,
    onTraceId?: (traceId: string) => void,
    inputStream?: AsyncIterable<any>
  ): Promise<RunActionResponse> {
    const runtime = input.runtimeId
      ? this.getRuntimeById(input.runtimeId)
      : this.getMostRecentRuntime();
    if (!runtime) {
      throw new Error(
        input.runtimeId
          ? `No runtime found with ID ${input.runtimeId}.`
          : 'No runtimes found. Make sure your app is running using the `start_runtime` MCP tool or the CLI: `genkit start -- ...`. See getting started documentation.'
      );
    }
    if (streamingCallback) {
      const response = await axios
        .post(
          `${runtime.reflectionServerUrl}/api/runAction?stream=true`,
          input,
          {
            headers: {
              'Content-Type': 'application/json',
            },
            responseType: 'stream',
          }
        )
        .catch((err) =>
          this.handleStreamError(
            err,
            `Error running action key='${input.key}'.`
          )
        );
      let genkitVersion: string;
      if (response.headers['x-genkit-version']) {
        genkitVersion = response.headers['x-genkit-version'];
      }

      const traceId = response.headers['x-genkit-trace-id'];
      if (traceId && onTraceId) {
        onTraceId(traceId);
      }

      const stream = response.data;

      let buffer = '';
      stream.on('data', (data: string) => {
        buffer += data;
        while (buffer.includes(STREAM_DELIMITER)) {
          try {
            streamingCallback(
              JSON.parse(buffer.substring(0, buffer.indexOf(STREAM_DELIMITER)))
            );
            buffer = buffer.substring(
              buffer.indexOf(STREAM_DELIMITER) + STREAM_DELIMITER.length
            );
          } catch (err) {
            logger.error(`Bad stream: ${err}`);
            break;
          }
        }
      });
      let resolver: (op: RunActionResponse) => void;
      let rejecter: (err: Error) => void;
      const promise = new Promise<RunActionResponse>((resolve, reject) => {
        resolver = resolve;
        rejecter = reject;
      });
      stream.on('end', () => {
        const parsedBuffer = JSON.parse(buffer);
        if (parsedBuffer.error) {
          const err = new GenkitToolsError(
            `Error running action key='${input.key}'.`
          );
          // massage the error into a shape dev ui expects
          err.data = {
            ...parsedBuffer.error,
            stack: (parsedBuffer.error?.details as any).stack,
            data: {
              genkitErrorMessage: parsedBuffer.error?.message,
              genkitErrorDetails: parsedBuffer.error?.details,
            },
          };
          rejecter(err);
          return;
        }
        const actionResponse = RunActionResponseSchema.parse(parsedBuffer);
        if (genkitVersion) {
          actionResponse.genkitVersion = genkitVersion;
        }
        resolver(actionResponse);
      });
      stream.on('error', (err: Error) => {
        rejecter(err);
      });
      return promise;
    } else {
      // runAction should use chunked JSON streaming to send early headers
      const response = await axios
        .post(`${runtime.reflectionServerUrl}/api/runAction`, input, {
          headers: {
            'Content-Type': 'application/json',
          },
          responseType: 'stream', // Use stream to get early headers
        })
        .catch((err) =>
          this.handleStreamError(
            err,
            `Error running action key='${input.key}'.`
          )
        );

      const traceId = response.headers['x-genkit-trace-id'];
      if (traceId && onTraceId) {
        onTraceId(traceId);
      }

      return new Promise<RunActionResponse>((resolve, reject) => {
        let buffer = '';

        response.data.on('data', (chunk: Buffer) => {
          buffer += chunk.toString();
        });

        response.data.on('end', () => {
          try {
            const responseData = JSON.parse(buffer);

            if (responseData.error) {
              const err = new GenkitToolsError(
                `Error running action key='${input.key}'.`
              );
              // massage the error into a shape dev ui expects
              err.data = {
                ...responseData.error,
                stack: (responseData.error?.details as any).stack,
                data: {
                  genkitErrorMessage: responseData.error?.message,
                  genkitErrorDetails: responseData.error?.details,
                },
              };
              reject(err);
              return;
            }

            // Handle backward compatibility - add trace ID from header if not in body
            if (!responseData.telemetry && traceId) {
              responseData.telemetry = { traceId: traceId };
            }

            const parsed = RunActionResponseSchema.parse(responseData);
            if (response.headers['x-genkit-version']) {
              parsed.genkitVersion = response.headers['x-genkit-version'];
            }
            resolve(parsed);
          } catch (err) {
            reject(new GenkitToolsError(`Failed to parse response: ${err}`));
          }
        });

        response.data.on('error', (err: Error) => {
          reject(err);
        });
      });
    }
  }

  /**
   * Cancels an in-flight action by trace ID
   */
  async cancelAction(input: {
    traceId: string;
    runtimeId?: string;
  }): Promise<{ message: string }> {
    const runtime = input.runtimeId
      ? this.getRuntimeById(input.runtimeId)
      : this.getMostRecentRuntime();
    if (!runtime) {
      throw new Error(
        input.runtimeId
          ? `No runtime found with ID ${input.runtimeId}.`
          : 'No runtimes found. Make sure your app is running.'
      );
    }

    try {
      const response = await axios.post(
        `${runtime.reflectionServerUrl}/api/cancelAction`,
        { traceId: input.traceId },
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      return response.data;
    } catch (err) {
      const axiosError = err as AxiosError;
      if (axiosError.response?.status === 404) {
        const error = new GenkitToolsError(
          'Action not found or already completed'
        );
        error.data = {
          message: 'Action not found or already completed',
        } as any;
        (error.data as any).statusCode = 404;
        throw error;
      }
      throw this.httpErrorHandler(axiosError);
    }
  }

  /**
   * Notifies the runtime of dependencies it may need (e.g. telemetry server URL).
   */
  private async notifyRuntime(runtime: RuntimeInfo) {
    try {
      await axios.post(`${runtime.reflectionServerUrl}/api/notify`, {
        telemetryServerUrl: this.telemetryServerUrl,
        reflectionApiSpecVersion: GENKIT_REFLECTION_API_SPEC_VERSION,
      });
    } catch (error) {
      logger.error(`Failed to notify runtime ${runtime.id}: ${error}`);
    }
  }

  /**
   * Sets up a watcher for the runtimes directory.
   */
  private async setupRuntimesWatcher() {
    try {
      const runtimesDir = await findRuntimesDir(this.projectRoot);
      await fs.mkdir(runtimesDir, { recursive: true });
      logger.debug(`Watching runtimes in ${runtimesDir}`);
      const watcher = chokidar.watch(runtimesDir, {
        persistent: true,
        ignoreInitial: false,
      });
      this.watchers.push(watcher);
      watcher.on('add', (filePath) => this.handleNewRuntime(filePath));
      if (this.manageHealth) {
        watcher.on('unlink', (filePath) => this.handleRemovedRuntime(filePath));
      }
      // eagerly check existing runtimes on first load.
      for (const runtime of await fs.readdir(runtimesDir)) {
        await this.handleNewRuntime(path.resolve(runtimesDir, runtime));
      }
    } catch (error) {
      logger.error('Failed to set up runtimes watcher:', error);
    }
  }

  /**
   * Sets up a watcher for the servers directory.
   */
  private async setupDevUiWatcher() {
    try {
      const serversDir = await findServersDir(this.projectRoot);
      await fs.mkdir(serversDir, { recursive: true });
      const watcher = chokidar.watch(serversDir, {
        persistent: true,
        ignoreInitial: false,
      });
      this.watchers.push(watcher);
      watcher.on('add', (filePath) => this.handleNewDevUi(filePath));
      if (this.manageHealth) {
        watcher.on('unlink', (filePath) => this.handleRemovedDevUi(filePath));
      }
      // eagerly check existing Dev UI on first load.
      for (const toolsInfo of await fs.readdir(serversDir)) {
        await this.handleNewDevUi(path.resolve(serversDir, toolsInfo));
      }
    } catch (error) {
      logger.error('Failed to set up tools server watcher:', error);
    }
  }

  /**
   * Handles a new Dev UI file.
   */
  private async handleNewDevUi(filePath: string) {
    try {
      if (!fsSync.existsSync(filePath)) {
        // file already got deleted, ignore...
        return;
      }
      const { content, toolsInfo } = await retriable(
        async () => {
          const content = await fs.readFile(filePath, 'utf-8');
          const toolsInfo = JSON.parse(content) as DevToolsInfo;
          return { content, toolsInfo };
        },
        { maxRetries: 10, delayMs: 500 }
      );

      if (isValidDevToolsInfo(toolsInfo)) {
        const fileName = path.basename(filePath);
        if (await checkServerHealth(toolsInfo.url)) {
          this.filenameToDevUiMap[fileName] = toolsInfo;
        } else {
          logger.debug('Found an unhealthy tools config file', fileName);
          await removeToolsInfoFile(fileName, this.projectRoot);
        }
      } else {
        logger.error(`Unexpected file in the servers directory: ${content}`);
      }
    } catch (error) {
      logger.error('Error reading tools config', error);
      return undefined;
    }
  }

  /**
   * Handles a removed Dev UI file.
   */
  private handleRemovedDevUi(filePath: string) {
    const fileName = path.basename(filePath);
    if (fileName in this.filenameToDevUiMap) {
      const toolsInfo = this.filenameToDevUiMap[fileName];
      delete this.filenameToDevUiMap[fileName];
      logger.debug(`Removed Dev UI with url ${toolsInfo.url}.`);
    }
  }

  /**
   * Handles a new runtime file.
   */
  private async handleNewRuntime(filePath: string) {
    try {
      if (!fsSync.existsSync(filePath)) {
        // file already got deleted, ignore...
        return;
      }
      const { content, runtimeInfo } = await retriable(
        async () => {
          const content = await fs.readFile(filePath, 'utf-8');
          const runtimeInfo = JSON.parse(content) as RuntimeInfo;
          runtimeInfo.projectName = projectNameFromGenkitFilePath(filePath);
          return { content, runtimeInfo };
        },
        { maxRetries: 10, delayMs: 500 }
      );

      if (isValidRuntimeInfo(runtimeInfo)) {
        if (!runtimeInfo.name) {
          runtimeInfo.name = runtimeInfo.id;
        }
        const fileName = path.basename(filePath);
        if (
          await checkServerHealth(
            runtimeInfo.reflectionServerUrl,
            runtimeInfo.id
          )
        ) {
          if (
            runtimeInfo.reflectionApiSpecVersion !=
            GENKIT_REFLECTION_API_SPEC_VERSION
          ) {
            if (
              !runtimeInfo.reflectionApiSpecVersion ||
              runtimeInfo.reflectionApiSpecVersion <
                GENKIT_REFLECTION_API_SPEC_VERSION
            ) {
              logger.warn(
                'Genkit CLI is newer than runtime library. Some feature may not be supported. ' +
                  'Consider upgrading your runtime library version (debug info: expected ' +
                  `${GENKIT_REFLECTION_API_SPEC_VERSION}, got ${runtimeInfo.reflectionApiSpecVersion}).`
              );
            } else {
              logger.error(
                'Genkit CLI version is outdated. Please update `genkit-cli` to the latest version.'
              );
              process.exit(1);
            }
          }
          this.filenameToRuntimeMap[fileName] = runtimeInfo;
          this.idToFileMap[runtimeInfo.id] = fileName;
          this.eventEmitter.emit(RuntimeEvent.ADD, runtimeInfo);
          await this.notifyRuntime(runtimeInfo);
          logger.debug(
            `Added runtime with ID ${runtimeInfo.id} at URL: ${runtimeInfo.reflectionServerUrl}`
          );
        } else {
          await this.removeRuntime(fileName);
        }
      } else {
        logger.error(`Unexpected file in the runtimes directory: ${content}`);
      }
    } catch (error) {
      logger.error(`Error processing file ${filePath}:`, error);
    }
  }

  /**
   * Handles a removed runtime file.
   */
  private handleRemovedRuntime(filePath: string) {
    const fileName = path.basename(filePath);
    if (fileName in this.filenameToRuntimeMap) {
      const runtime = this.filenameToRuntimeMap[fileName];
      delete this.filenameToRuntimeMap[fileName];
      delete this.idToFileMap[runtime.id];
      this.eventEmitter.emit(RuntimeEvent.REMOVE, runtime);
      logger.debug(`Removed runtime with id ${runtime.id}.`);
    }
  }

  /**
   * Handles a stream error by reading the stream and then calling httpErrorHandler.
   */
  private async handleStreamError(
    error: AxiosError,
    message: string
  ): Promise<never> {
    if (
      error.response &&
      error.config?.responseType === 'stream' &&
      (error.response.data as any).on
    ) {
      try {
        const body = await this.streamToString(error.response.data);
        try {
          error.response.data = JSON.parse(body);
        } catch (e) {
          error.response.data = {
            message: body || 'Unknown error',
          };
        }
      } catch (e) {
        // If stream reading fails, we must replace the stream object with a safe error object
        // to prevent circular structure errors during JSON serialization.
        error.response.data = {
          message: 'Failed to read error response stream',
          details: String(e),
        };
      }
    }
    this.httpErrorHandler(error, message);
  }

  /**
   * Helper to convert a stream to string.
   */
  private streamToString(stream: any): Promise<string> {
    return new Promise((resolve, reject) => {
      let buffer = '';
      stream.on('data', (chunk: Buffer) => {
        buffer += chunk.toString();
      });
      stream.on('end', () => {
        resolve(buffer);
      });
      stream.on('error', (err: Error) => {
        reject(err);
      });
    });
  }

  /**
   * Performs health checks on all runtimes.
   */
  private async performHealthChecks() {
    const healthCheckPromises = Object.entries(this.filenameToRuntimeMap).map(
      async ([fileName, runtime]) => {
        if (
          !(await checkServerHealth(runtime.reflectionServerUrl, runtime.id))
        ) {
          await this.removeRuntime(fileName);
        }
      }
    );
    return Promise.all(healthCheckPromises);
  }

  /**
   * Removes the runtime file which will trigger the removal watcher.
   */
  private async removeRuntime(fileName: string) {
    try {
      const runtimesDir = await findRuntimesDir(this.projectRoot);
      const runtimeFilePath = path.join(runtimesDir, fileName);
      await fs.unlink(runtimeFilePath);
    } catch (error) {
      logger.debug(`Failed to delete runtime file: ${error}`);
    }
    logger.debug(`Removed unhealthy runtime ${fileName} from manager.`);
  }
}

/**
 * Checks if the runtime file is valid.
 */
function isValidRuntimeInfo(data: any): data is RuntimeInfo {
  let timestamp = '';
  // runtime filename might come with underscores due OS filename restrictions
  // revert the underscores so the timestamp gets parsed correctly
  if (typeof data.timestamp === 'string') {
    timestamp = data.timestamp.replaceAll('_', ':');
  }

  return (
    typeof data === 'object' &&
    data !== null &&
    typeof data.id === 'string' &&
    typeof data.pid === 'number' &&
    typeof data.reflectionServerUrl === 'string' &&
    typeof data.timestamp === 'string' &&
    !isNaN(Date.parse(timestamp)) &&
    (data.name === undefined || typeof data.name === 'string')
  );
}
