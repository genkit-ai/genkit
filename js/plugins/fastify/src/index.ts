/**
 * Copyright 2025 Google LLC
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

import { randomUUID } from 'crypto';
import type {
  FastifyInstance,
  FastifyPluginAsync,
  FastifyReply,
  FastifyRequest,
  RouteHandlerMethod,
} from 'fastify';
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
import type { ServerResponse } from 'http';
import { getErrorMessage, getErrorStack } from './utils.js';

const streamDelimiter = '\n\n';

/**
 * Options for a {@link fastifyHandler} (context provider, stream manager).
 */
export interface FastifyHandlerOptions<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
> {
  contextProvider?: ContextProvider<C, I>;
  streamManager?: StreamManager;
}

/**
 * Exposes the provided flow (or any action) as a Fastify route handler.
 *
 * Fastify is not Web Fetch native, so this handler bridges Fastify's
 * `request`/`reply` to the Genkit action protocol that `runFlow`/`streamFlow`
 * clients expect: it reads `{ data }` from the parsed JSON body, runs the
 * action, and streams chunks back as Server-Sent Events when the caller asks
 * for them (`Accept: text/event-stream` or `?stream=true`).
 *
 * @example
 * ```typescript
 * import Fastify from 'fastify';
 * import { fastifyHandler } from '@genkit-ai/fastify';
 *
 * const app = Fastify();
 * app.post('/simpleFlow', fastifyHandler(simpleFlow));
 * await app.listen({ port: 8080 });
 * ```
 */
export function fastifyHandler<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  action: Action<I, O, S>,
  opts?: FastifyHandlerOptions<C, I>
): RouteHandlerMethod {
  return async (
    request: FastifyRequest,
    reply: FastifyReply
  ): Promise<void> => {
    const stream = (request.query as { stream?: string } | undefined)?.stream;
    const streamIdHeader = request.headers['x-genkit-stream-id'];
    const streamId = Array.isArray(streamIdHeader)
      ? streamIdHeader[0]
      : streamIdHeader;

    if (request.body === undefined || request.body === null) {
      const errMsg =
        `Error: request.body is undefined. ` +
        `Possible reasons: missing 'content-type: application/json' in request ` +
        `headers, or a content type parser that did not populate request.body. `;
      logger.error(errMsg);
      reply.code(400).send({ message: errMsg, status: 'INVALID_ARGUMENT' });
      return;
    }

    const input = (request.body as { data?: z.infer<I> }).data as z.infer<I>;
    let context: Record<string, any>;

    try {
      context =
        (await opts?.contextProvider?.({
          method: request.method as RequestData['method'],
          headers: Object.fromEntries(
            Object.entries(request.headers)
              // Skip headers explicitly set to undefined so they don't become
              // the literal string "undefined" via String(value).
              .filter(([, value]) => value !== undefined)
              .map(([key, value]) => [
                key.toLowerCase(),
                // RFC 9110 5.3: combine repeated field lines with a comma.
                Array.isArray(value) ? value.join(', ') : String(value),
              ])
          ),
          input,
        })) || {};
    } catch (e: any) {
      logger.error(
        `Auth policy failed with error: ${getErrorMessage(e)}\n${getErrorStack(e)}`
      );
      reply.code(getHttpStatus(e)).send(getCallableJSON(e));
      return;
    }

    const abortController = new AbortController();
    request.raw.on('close', () => {
      abortController.abort();
    });
    // When/if a request timeout is configured, the socket emits 'timeout'.
    request.raw.on('timeout', () => {
      abortController.abort();
    });

    if (
      request.headers['accept']?.toLowerCase().includes('text/event-stream') ||
      stream === 'true'
    ) {
      // Headers set by earlier hooks/plugins (e.g. @fastify/cors) live on the
      // Fastify reply. hijack() stops Fastify from flushing them to the socket,
      // so copy them onto the raw response ourselves before streaming, or the
      // browser would reject the streamed response for missing CORS headers.
      const pendingHeaders = reply.getHeaders();
      // Take ownership of the underlying response so Fastify does not also try
      // to serialize and send a reply while we stream raw SSE bytes.
      reply.hijack();
      const raw = reply.raw;
      // hijack() also bypasses Fastify's error handling, so an error thrown
      // while setting up the stream (e.g. streamManager.open, or a rethrow from
      // subscribeToStream) would leave the socket open and leak the connection.
      // Catch it here and close the response cleanly.
      try {
        for (const [key, value] of Object.entries(pendingHeaders)) {
          if (value !== undefined) {
            raw.setHeader(key, value);
          }
        }

        const streamManager = opts?.streamManager;
        if (streamManager && streamId) {
          await subscribeToStream(streamManager, streamId, raw);
          return;
        }

        const streamIdToUse = randomUUID();
        const headers: Record<string, string> = {
          'Content-Type': 'text/plain',
          'Transfer-Encoding': 'chunked',
        };
        if (streamManager) {
          headers['x-genkit-stream-id'] = streamIdToUse;
        }
        raw.writeHead(200, headers);
        await runActionWithDurableStreaming(
          action,
          streamManager,
          streamIdToUse,
          input,
          context,
          raw,
          abortController.signal
        );
      } catch (e) {
        logger.error(
          `Streaming request failed with error: ${getErrorMessage(e)}\n${getErrorStack(e)}`
        );
        if (raw.destroyed) {
          // Client already gone; nothing to send.
        } else if (raw.headersSent) {
          // Streaming already started: emit a trailing SSE error frame.
          raw.write(
            `error: ${JSON.stringify({ error: getCallableJSON(e) })}${streamDelimiter}`
          );
          raw.end();
        } else {
          // Nothing sent yet: respond with a normal API error.
          raw.writeHead(getHttpStatus(e), {
            'Content-Type': 'application/json',
          });
          raw.end(JSON.stringify(getCallableJSON(e)));
        }
      }
    } else {
      try {
        const result = await action.run(input, {
          context,
          abortSignal: abortController.signal,
        });
        // Responses for non-streaming flows are passed back with the flow result stored in a field called "result."
        reply
          .header('x-genkit-trace-id', result.telemetry.traceId)
          .header('x-genkit-span-id', result.telemetry.spanId)
          .code(200)
          .send({ result: result.result });
      } catch (e) {
        // Errors for non-streaming flows are passed back as standard API errors.
        logger.error(
          `Non-streaming request failed with error: ${getErrorMessage(e)}\n${getErrorStack(e)}`
        );
        reply.code(getHttpStatus(e)).send(getCallableJSON(e));
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
  response: ServerResponse,
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
      // The client may have disconnected mid-stream; writing to a destroyed
      // response would throw.
      if (response.destroyed) return;
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
    if (!response.destroyed) {
      response.write(
        'data: ' + JSON.stringify({ result: result.result }) + streamDelimiter
      );
      response.end();
    }
  } catch (e) {
    if (durableStream) {
      taskQueue!.enqueue(() => durableStream!.error(e));
      await taskQueue!.merge();
    }
    logger.error(
      `Streaming request failed with error: ${getErrorMessage(e)}\n${getErrorStack(e)}`
    );
    if (!response.destroyed) {
      response.write(
        `error: ${JSON.stringify({
          error: getCallableJSON(e),
        })}${streamDelimiter}`
      );
      response.end();
    }
  }
}

async function subscribeToStream(
  streamManager: StreamManager,
  streamId: string,
  response: ServerResponse
): Promise<void> {
  // Send the streaming headers lazily on the first event so that a
  // StreamNotFoundError can still respond with a clean 204. Without this the
  // subscribe path would emit body bytes with no Content-Type, which some
  // clients and proxies refuse to treat as a stream.
  const ensureHeaders = () => {
    if (!response.headersSent) {
      response.writeHead(200, {
        'Content-Type': 'text/plain',
        'Transfer-Encoding': 'chunked',
      });
    }
  };
  try {
    await streamManager.subscribe(streamId, {
      onChunk: (chunk) => {
        // The subscribing client may have disconnected; skip writes to a
        // destroyed response to avoid throwing.
        if (response.destroyed) return;
        ensureHeaders();
        response.write(
          'data: ' + JSON.stringify({ message: chunk }) + streamDelimiter
        );
      },
      onDone: (output) => {
        if (response.destroyed) return;
        ensureHeaders();
        response.write(
          'data: ' + JSON.stringify({ result: output }) + streamDelimiter
        );
        response.end();
      },
      onError: (err) => {
        logger.error(
          `Streaming request failed with error: ${getErrorMessage(err)}\n${getErrorStack(err)}`
        );
        if (response.destroyed) return;
        ensureHeaders();
        response.write(
          `error: ${JSON.stringify({
            error: getCallableJSON(err),
          })}${streamDelimiter}`
        );
        response.end();
      },
    });
  } catch (e: any) {
    if (response.destroyed) return;
    if (e instanceof StreamNotFoundError) {
      response.writeHead(204);
      response.end();
      return;
    }
    if (e.status === 'DEADLINE_EXCEEDED') {
      ensureHeaders();
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
 * A wrapper object containing a flow with its associated context provider.
 * @deprecated Use {@link withFlowOptions} instead.
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
 * Attaches a context provider to a flow.
 * @deprecated Use {@link withFlowOptions} instead.
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
 * Attaches options (context provider, stream manager, custom path) to a flow
 * for use with the {@link genkitFastify} plugin.
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
 * Options for the {@link genkitFastify} plugin.
 */
export interface GenkitFastifyOptions {
  /** Flows (or flows wrapped with {@link withFlowOptions}) to expose as routes. */
  flows: (
    | Flow<any, any, any>
    | FlowWithContextProvider<any, any, any>
    | FlowWithOptions<any, any, any>
  )[];
  /** HTTP path prefix prepended to each exposed flow (e.g. `/api`). */
  pathPrefix?: string;
}

function registerFlow(
  instance: FastifyInstance,
  path: string,
  handler: RouteHandlerMethod
) {
  logger.debug(` - POST ${path}`);
  instance.post(path, handler);
}

/**
 * A Fastify plugin that exposes a set of Genkit flows as POST routes. Register
 * it with `app.register` for an ergonomic, multi-flow setup:
 *
 * @example
 * ```typescript
 * import Fastify from 'fastify';
 * import { genkitFastify, withFlowOptions } from '@genkit-ai/fastify';
 *
 * const app = Fastify();
 * await app.register(genkitFastify, {
 *   flows: [
 *     menuSuggestionFlow,
 *     withFlowOptions(secureFlow, { contextProvider }),
 *   ],
 * });
 * await app.listen({ port: 8080 });
 * ```
 */
export const genkitFastify: FastifyPluginAsync<GenkitFastifyOptions> = async (
  instance,
  opts
) => {
  const pathPrefix = opts.pathPrefix ?? '';
  // Fastify requires route paths to start with a single '/'. Normalize so a
  // prefix without a leading slash ('api') or with stray slashes ('/api/')
  // still produces a valid route.
  const cleanPath = (p: string) => ('/' + p).replace(/\/+/g, '/');
  logger.debug('Registering Genkit flow routes:');
  for (const flow of opts.flows ?? []) {
    if ('flow' in flow) {
      const flowPath = cleanPath(
        `${pathPrefix}/${
          ('options' in flow && flow.options.path) || flow.flow.__action.name
        }`
      );
      const options =
        'options' in flow ? flow.options : { contextProvider: flow.context };
      registerFlow(instance, flowPath, fastifyHandler(flow.flow, options));
    } else {
      const flowPath = cleanPath(`${pathPrefix}/${flow.__action.name}`);
      registerFlow(instance, flowPath, fastifyHandler(flow));
    }
  }
};

export default genkitFastify;
