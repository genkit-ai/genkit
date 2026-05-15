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

import bodyParser from 'body-parser';
import cors, { type CorsOptions } from 'cors';
import { randomUUID } from 'crypto';
import express from 'express';
import {
  Action,
  ActionStreamInput,
  AsyncTaskQueue,
  Flow,
  StreamNotFoundError,
  type ActionContext,
  type StreamManager,
  type z,
} from 'genkit/beta';
import {
  getCallableJSON,
  getHttpStatus,
  type ContextProvider,
  type RequestData,
} from 'genkit/context';
import { logger } from 'genkit/logging';
import type { Server } from 'http';

const streamDelimiter = '\n\n';

/**
 * Exposes provided flow or an action as express handler.
 */
export function expressHandler<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  action: Action<I, O, S>,
  opts?: {
    contextProvider?: ContextProvider<C, I>;
    streamManager?: StreamManager;
  }
): express.RequestHandler {
  return async (
    request: express.Request,
    response: express.Response
  ): Promise<void> => {
    const { stream } = request.query;
    const streamIdHeader = request.headers['x-genkit-stream-id'];
    const streamId = Array.isArray(streamIdHeader)
      ? streamIdHeader[0]
      : streamIdHeader;

    if (!request.body) {
      const errMsg =
        `Error: request.body is undefined. ` +
        `Possible reasons: missing 'content-type: application/json' in request ` +
        `headers or misconfigured JSON middleware ('app.use(express.json()')? `;
      logger.error(errMsg);
      response
        .status(400)
        .json({ message: errMsg, status: 'INVALID ARGUMENT' })
        .end();
      return;
    }

    const input = request.body.data as z.infer<I>;
    let context: Record<string, any>;

    try {
      context =
        (await opts?.contextProvider?.({
          method: request.method as RequestData['method'],
          headers: Object.fromEntries(
            Object.entries(request.headers).map(([key, value]) => [
              key.toLowerCase(),
              Array.isArray(value) ? value.join(' ') : String(value),
            ])
          ),
          input,
        })) || {};
    } catch (e: any) {
      logger.error(
        `Auth policy failed with error: ${(e as Error).message}\n${(e as Error).stack}`
      );
      response.status(getHttpStatus(e)).json(getCallableJSON(e)).end();
      return;
    }

    const abortController = new AbortController();
    request.on('close', () => {
      abortController.abort();
    });
    // when/if using timeout middleware, it will emit 'timeout' event.
    request.on('timeout', () => {
      abortController.abort();
    });

    if (request.get('Accept') === 'text/event-stream' || stream === 'true') {
      const streamManager = opts?.streamManager;
      if (streamManager && streamId) {
        await subscribeToStream(streamManager, streamId, response);
        return;
      }
      const streamIdToUse = randomUUID();
      const headers = {
        'Content-Type': 'text/plain',
        'Transfer-Encoding': 'chunked',
      };
      if (streamManager) {
        headers['x-genkit-stream-id'] = streamIdToUse;
      }
      response.writeHead(200, headers);
      runActionWithDurableStreaming(
        action,
        streamManager,
        streamIdToUse,
        input,
        context,
        response,
        abortController.signal
      );
    } else {
      try {
        const result = await action.run(input, {
          context,
          abortSignal: abortController.signal,
        });
        response.setHeader('x-genkit-trace-id', result.telemetry.traceId);
        response.setHeader('x-genkit-span-id', result.telemetry.spanId);
        // Responses for non-streaming flows are passed back with the flow result stored in a field called "result."
        response
          .status(200)
          .json({
            result: result.result,
          })
          .end();
      } catch (e) {
        // Errors for non-streaming flows are passed back as standard API errors.
        logger.error(
          `Non-streaming request failed with error: ${(e as Error).message}\n${(e as Error).stack}`
        );
        response.status(getHttpStatus(e)).json(getCallableJSON(e)).end();
      }
    }
  };
}

async function runActionWithDurableStreaming<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny,
>(
  action: Action<I, O, S>,
  streamManager: StreamManager | undefined,
  streamId: string,
  input: z.infer<I>,
  context: ActionContext,
  response: express.Response,
  abortSignal: AbortSignal
) {
  let taskQueue: AsyncTaskQueue | undefined;
  let durableStream: ActionStreamInput<any, any> | undefined;
  if (streamManager) {
    taskQueue = new AsyncTaskQueue();
    durableStream = await streamManager.open(streamId);
  }
  try {
    let onChunk = (chunk: z.infer<S>) => {
      response.write(
        'data: ' + JSON.stringify({ message: chunk }) + streamDelimiter
      );
    };
    if (streamManager) {
      const originalOnChunk = onChunk;
      onChunk = (chunk: z.infer<S>) => {
        originalOnChunk(chunk);
        taskQueue!.enqueue(() => durableStream!.write(chunk));
      };
    }
    const result = await action.run(input, {
      onChunk,
      context,
      abortSignal,
    });
    if (streamManager) {
      taskQueue!.enqueue(() => durableStream!.done(result.result));
      await taskQueue!.merge();
    }
    response.write(
      'data: ' + JSON.stringify({ result: result.result }) + streamDelimiter
    );
    response.end();
  } catch (e) {
    if (durableStream) {
      taskQueue!.enqueue(() => durableStream!.error(e));
      await taskQueue!.merge();
    }
    logger.error(
      `Streaming request failed with error: ${(e as Error).message}\n${
        (e as Error).stack
      }`
    );
    response.write(
      `error: ${JSON.stringify({
        error: getCallableJSON(e),
      })}${streamDelimiter}`
    );
    response.end();
  }
}

async function subscribeToStream(
  streamManager: StreamManager,
  streamId: string,
  response: express.Response
): Promise<void> {
  try {
    await streamManager.subscribe(streamId, {
      onChunk: (chunk) => {
        response.write(
          'data: ' + JSON.stringify({ message: chunk }) + streamDelimiter
        );
      },
      onDone: (output) => {
        response.write(
          'data: ' + JSON.stringify({ result: output }) + streamDelimiter
        );
        response.end();
      },
      onError: (err) => {
        logger.error(
          `Streaming request failed with error: ${(err as Error).message}\n${
            (err as Error).stack
          }`
        );
        response.write(
          `error: ${JSON.stringify({
            error: getCallableJSON(err),
          })}${streamDelimiter}`
        );
        response.end();
      },
    });
  } catch (e: any) {
    if (e instanceof StreamNotFoundError) {
      response.status(204).end();
      return;
    }
    if (e.status === 'DEADLINE_EXCEEDED') {
      response.write(
        `error: ${JSON.stringify({
          error: getCallableJSON(e),
        })}${streamDelimiter}`
      );
      response.end();
      return;
    }
    throw e;
  }
}

/**
 * A wrapper object containing a flow with its associated auth policy.
 * @deprecated Use `withFlowOptions` instead.
 */
export type FlowWithContextProvider<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  flow: Flow<I, O, S>;
  context: ContextProvider<C, I>;
};

/**
 * A wrapper object containing a flow with its associated options.
 */
export type FlowWithOptions<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  flow: Flow<I, O, S>;
  options: {
    contextProvider?: ContextProvider<any, I>;
    streamManager?: StreamManager;
    path?: string;
  };
};

/**
 * Adds an auth policy to the flow.
 * @deprecated Use `withFlowOptions` instead.
 */
export function withContextProvider<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  flow: Flow<I, O, S>,
  context: ContextProvider<C, I>
): FlowWithContextProvider<C, I, O, S> {
  return {
    flow,
    context,
  };
}

/**
 * Adds an auth policy to the flow.
 */
export function withFlowOptions<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny,
>(
  flow: Flow<I, O, S>,
  options: {
    contextProvider?: ContextProvider<any, I>;
    streamManager?: StreamManager;
    path?: string;
  }
): FlowWithOptions<I, O, S> {
  return {
    flow,
    options,
  };
}

/**
 * Options to configure the flow server.
 */
export interface FlowServerOptions {
  /** List of flows to expose via the flow server. */
  flows: (
    | Flow<any, any, any>
    | FlowWithContextProvider<any, any, any>
    | FlowWithOptions<any, any, any>
  )[];
  /** Port to run the server on. Defaults to env.PORT or 3400. */
  port?: number;
  /** CORS options for the server. */
  cors?: CorsOptions;
  /** HTTP method path prefix for the exposed flows. */
  pathPrefix?: string;
  /** JSON body parser options. */
  jsonParserOptions?: bodyParser.OptionsJson;
}

/**
 * Starts an express server with the provided flows and options.
 */
export function startFlowServer(options: FlowServerOptions): FlowServer {
  const server = new FlowServer(options);
  server.start();
  return server;
}

/**
 * Flow server exposes registered flows as HTTP endpoints.
 *
 * This is for use in production environments.
 *
 * @hidden
 */
export class FlowServer {
  /** List of all running servers needed to be cleaned up on process exit. */
  private static RUNNING_SERVERS: FlowServer[] = [];

  /** Options for the flow server configured by the developer. */
  private options: FlowServerOptions;
  /** Port the server is actually running on. This may differ from `options.port` if the original was occupied. Null is server is not running. */
  private port: number | null = null;
  /** Express server instance. Null if server is not running. */
  private server: Server | null = null;

  constructor(options: FlowServerOptions) {
    this.options = {
      ...options,
    };
  }

  /**
   * Starts the server and adds it to the list of running servers to clean up on exit.
   */
  async start() {
    const server = express();

    server.use(bodyParser.json(this.options.jsonParserOptions));
    server.use(cors(this.options.cors));

    logger.debug('Running flow server with flow paths:');
    const pathPrefix = this.options.pathPrefix ?? '';
    this.options.flows?.forEach((flow) => {
      if ('flow' in flow) {
        const flowPath = `/${pathPrefix}${
          ('options' in flow && flow.options.path) || flow.flow.__action.name
        }`;
        logger.debug(` - ${flowPath}`);
        const options =
          'options' in flow ? flow.options : { contextProvider: flow.context };
        server.post(flowPath, expressHandler(flow.flow, options));
      } else {
        const flowPath = `/${pathPrefix}${flow.__action.name}`;
        logger.debug(` - ${flowPath}`);
        server.post(flowPath, expressHandler(flow));
      }
    });
    this.port =
      this.options?.port ||
      (process.env.PORT ? Number.parseInt(process.env.PORT) : 0) ||
      3400;
    this.server = server.listen(this.port, () => {
      logger.debug(`Flow server running on http://localhost:${this.port}`);
      FlowServer.RUNNING_SERVERS.push(this);
    });
  }

  /**
   * Stops the server and removes it from the list of running servers to clean up on exit.
   */
  async stop(): Promise<void> {
    if (!this.server) {
      return;
    }
    return new Promise<void>((resolve, reject) => {
      this.server!.close((err) => {
        if (err) {
          logger.error(
            `Error shutting down flow server on port ${this.port}: ${err}`
          );
          reject(err);
        }
        const index = FlowServer.RUNNING_SERVERS.indexOf(this);
        if (index > -1) {
          FlowServer.RUNNING_SERVERS.splice(index, 1);
        }
        logger.debug(
          `Flow server on port ${this.port} has successfully shut down.`
        );
        this.port = null;
        this.server = null;
        resolve();
      });
    });
  }

  /**
   * Stops all running servers.
   */
  static async stopAll() {
    return Promise.all(
      FlowServer.RUNNING_SERVERS.map((server) => server.stop())
    );
  }
}
