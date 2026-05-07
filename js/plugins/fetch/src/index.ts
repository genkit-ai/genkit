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

import {
  Action,
  ActionStreamInput,
  AsyncTaskQueue,
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
import { getErrorMessage, getErrorStack } from './utils.js';

const streamDelimiter = '\n\n';

// Let the runtime provide the randomUUID function.
const randomUUID = () => globalThis.crypto.randomUUID();

/**
 * A wrapper object containing an action with its associated options.
 */
export type ActionWithOptions<
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  action: Action<I, O, S>;
  options: {
    contextProvider?: ContextProvider<any, I>;
    streamManager?: StreamManager;
    path?: string;
  };
};

/**
 * Options for a fetch handler (context provider, stream manager).
 */
export interface FetchHandlerOptions<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
> {
  contextProvider?: ContextProvider<C, I>;
  streamManager?: StreamManager;
}

/**
 * Wraps an action (flow, model, etc.) with options for use with {@link fetchHandlers}.
 */
export function withActionOptions<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny,
>(
  action: Action<I, O, S>,
  options: {
    contextProvider?: ContextProvider<any, I>;
    streamManager?: StreamManager;
    path?: string;
  }
): ActionWithOptions<I, O, S> {
  return {
    action,
    options,
  };
}

/**
 * Converts Headers object to a plain object with lowercase keys.
 */
function headersToObject(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {};
  headers.forEach((value, key) => {
    result[key.toLowerCase()] = value;
  });
  return result;
}

/**
 * Gets context from the request using the context provider if available.
 */
async function getContext<C extends ActionContext, I extends z.ZodTypeAny>(
  request: Request,
  input: z.infer<I>,
  provider?: ContextProvider<C, I>
): Promise<C> {
  const context = {} as C;
  if (!provider) {
    return context;
  }

  const requestData: RequestData = {
    method: request.method as RequestData['method'],
    headers: headersToObject(request.headers),
    input,
  };

  return await provider(requestData);
}

/**
 * Subscribes to an existing stream using StreamManager.
 */
async function subscribeToStream(
  streamManager: StreamManager,
  streamId: string
): Promise<Response | null> {
  try {
    const encoder = new TextEncoder();
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();

    await streamManager.subscribe(streamId, {
      onChunk: (chunk) => {
        writer.write(
          encoder.encode(
            'data: ' + JSON.stringify({ message: chunk }) + streamDelimiter
          )
        );
      },
      onDone: (output) => {
        writer.write(
          encoder.encode(
            'data: ' + JSON.stringify({ result: output }) + streamDelimiter
          )
        );
        writer.close();
      },
      onError: (err) => {
        logger.error(
          `Streaming request failed with error: ${getErrorMessage(err)}\n${getErrorStack(err)}`
        );
        writer.write(
          encoder.encode(
            `error: ${JSON.stringify({
              error: getCallableJSON(err),
            })}${streamDelimiter}`
          )
        );
        writer.close();
      },
    });

    return new Response(readable, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'Transfer-Encoding': 'chunked',
        'x-genkit-stream-id': streamId,
      },
    });
  } catch (e: any) {
    if (e instanceof StreamNotFoundError) {
      return new Response(null, { status: 204 });
    }
    if (e.status === 'DEADLINE_EXCEEDED') {
      const encoder = new TextEncoder();
      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();
      writer.write(
        encoder.encode(
          `error: ${JSON.stringify({
            error: getCallableJSON(e),
          })}${streamDelimiter}`
        )
      );
      writer.close();
      return new Response(readable, {
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          'Transfer-Encoding': 'chunked',
        },
      });
    }
    throw e;
  }
}

/**
 * Runs an action with durable streaming support.
 */
async function runActionWithDurableStreaming<
  I extends z.ZodTypeAny,
  O extends z.ZodTypeAny,
  S extends z.ZodTypeAny,
>(
  action: Action<I, O, S>,
  streamManager: StreamManager | undefined,
  streamId: string,
  input: z.infer<I>,
  init: any,
  context: ActionContext,
  writer: WritableStreamDefaultWriter<Uint8Array>,
  abortSignal: AbortSignal
): Promise<void> {
  const encoder = new TextEncoder();
  let taskQueue: AsyncTaskQueue | undefined;
  let durableStream: ActionStreamInput<any, any> | undefined;

  if (streamManager) {
    taskQueue = new AsyncTaskQueue();
    durableStream = await streamManager.open(streamId);
  }

  try {
    let onChunk = (chunk: z.infer<S>) => {
      writer.write(
        encoder.encode(
          'data: ' + JSON.stringify({ message: chunk }) + streamDelimiter
        )
      );
    };

    if (streamManager && durableStream) {
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
      init,
    });

    if (streamManager && durableStream) {
      taskQueue!.enqueue(() => durableStream!.done(result.result));
      await taskQueue!.merge();
    }

    writer.write(
      encoder.encode(
        'data: ' + JSON.stringify({ result: result.result }) + streamDelimiter
      )
    );
    writer.close();
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
    writer.write(
      encoder.encode(
        `error: ${JSON.stringify({
          error: getCallableJSON(e),
        })}${streamDelimiter}`
      )
    );
    writer.close();
  }
}

/**
 * Returns a Fetch handler for a single action (flow, model, or any Genkit action).
 * Express-like API: pass the action first, then call the returned handler with the request.
 *
 * @param action - The Genkit action to execute (flow, model, etc.)
 * @param options - Optional configuration including contextProvider and streamManager
 * @returns A handler function that takes a Request and returns a Promise<Response>
 *
 * @example
 * ```typescript
 * import { fetchHandler } from '@genkit-ai/fetch';
 *
 * // Flow or model
 * app.post('/myFlow', (c) => fetchHandler(myFlow)(c.req.raw));
 * app.post('/models/gpt5', (c) => fetchHandler(gpt5)(c.req.raw));
 *
 * // With options
 * app.post('/secure', (c) => fetchHandler(secureFlow, { contextProvider })(c.req.raw));
 * ```
 */
async function handleActionRequest<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  request: Request,
  action: Action<I, O, S>,
  options?: FetchHandlerOptions<C, I>
): Promise<Response> {
  const url = new URL(request.url);
  const streamParam = url.searchParams.get('stream');
  const shouldStream = streamParam === 'true';

  const streamIdHeader = request.headers.get('x-genkit-stream-id');
  const streamId = streamIdHeader || undefined;

  let body: any;
  try {
    body = await request.json();
  } catch (e) {
    const errMsg =
      `Error: Failed to parse request body as JSON. ` +
      `Make sure the request has 'Content-Type: application/json' header.`;
    logger.error(errMsg);
    return new Response(
      JSON.stringify({ message: errMsg, status: 'INVALID_ARGUMENT' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }

  if (!body || typeof body !== 'object' || !('data' in body)) {
    const errMsg =
      `Error: Request body must be a JSON object with a 'data' field. ` +
      `Expected format: {"data": ...}`;
    logger.error(errMsg);
    return new Response(
      JSON.stringify({ message: errMsg, status: 'INVALID_ARGUMENT' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }

  const input = body.data as z.infer<I>;
  const init = body.init;

  let context: C;
  try {
    context = await getContext(request, input, options?.contextProvider);
  } catch (e: any) {
    logger.error(
      `Context provider failed with error: ${(e as Error).message}\n${
        (e as Error).stack
      }`
    );
    return new Response(JSON.stringify(getCallableJSON(e)), {
      status: getHttpStatus(e),
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const acceptHeader = request.headers.get('Accept') || '';
  const isStreaming = acceptHeader === 'text/event-stream' || shouldStream;

  if (isStreaming) {
    const streamManager = options?.streamManager;
    if (streamManager && streamId) {
      const response = await subscribeToStream(streamManager, streamId);
      if (response) {
        return response;
      }
    }

    const streamIdToUse = randomUUID();
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();

    runActionWithDurableStreaming(
      action,
      options?.streamManager,
      streamIdToUse,
      input,
      init,
      context,
      writer,
      request.signal
    ).catch((err) => {
      logger.error(`Error in streaming handler: ${err}`);
    });

    const headers: Record<string, string> = {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Transfer-Encoding': 'chunked',
    };

    if (options?.streamManager) {
      headers['x-genkit-stream-id'] = streamIdToUse;
    }

    return new Response(readable, {
      status: 200,
      headers,
    });
  }

  try {
    const result = await action.run(input, {
      context,
      abortSignal: request.signal,
      init,
    });

    const headers: Record<string, string> = {
      'x-genkit-trace-id': result.telemetry.traceId,
      'x-genkit-span-id': result.telemetry.spanId,
    };

    return new Response(JSON.stringify({ result: result.result }), {
      status: 200,
      headers: { 'Content-Type': 'application/json', ...headers },
    });
  } catch (e) {
    logger.error(
      `Non-streaming request failed with error: ${(e as Error).message}\n${
        (e as Error).stack
      }`
    );
    return new Response(JSON.stringify(getCallableJSON(e)), {
      status: getHttpStatus(e),
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

export function fetchHandler<
  C extends ActionContext = ActionContext,
  I extends z.ZodTypeAny = z.ZodTypeAny,
  O extends z.ZodTypeAny = z.ZodTypeAny,
  S extends z.ZodTypeAny = z.ZodTypeAny,
>(
  action: Action<I, O, S>,
  options?: FetchHandlerOptions<C, I>
): (request: Request) => Promise<Response> {
  return (request: Request) => handleActionRequest(request, action, options);
}

/**
 * Returns a Fetch handler for multiple actions with path-based routing.
 * Express-like API: pass actions and optional path prefix, then call the returned handler with the request.
 *
 * @param actions - Array of actions (or ActionWithOptions) to expose; each is routed by path (action name or options.path)
 * @param pathPrefix - Optional path prefix to strip from the URL (e.g., '/api')
 * @returns A handler function that takes a Request and returns a Promise<Response>
 *
 * @example
 * ```typescript
 * import { fetchHandlers } from '@genkit-ai/fetch';
 *
 * // Mount all actions under /api - routes by path (e.g. POST /api/hello, POST /api/models/gpt5)
 * app.all('/api/*', (c) => fetchHandlers(actions, '/api')(c.req.raw));
 * ```
 */
export function fetchHandlers(
  actions: (Action<any, any, any> | ActionWithOptions<any, any, any>)[],
  pathPrefix?: string
): (request: Request) => Promise<Response> {
  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    let pathname = url.pathname;

    if (pathPrefix) {
      const prefix = pathPrefix.endsWith('/') ? pathPrefix : pathPrefix + '/';
      if (pathname === pathPrefix) {
        pathname = '';
      } else if (pathname.startsWith(prefix)) {
        pathname = pathname.slice(prefix.length);
      } else {
        return new Response(
          JSON.stringify({
            status: 'NOT_FOUND',
            message: 'No action matched the request path.',
          }),
          { status: 404, headers: { 'Content-Type': 'application/json' } }
        );
      }
    }

    pathname = pathname.replace(/^\//, '');

    let matchedAction: Action<any, any, any> | null = null;
    let handlerOptions: FetchHandlerOptions<any, any> | undefined = undefined;

    for (const item of actions) {
      if ('action' in item) {
        const options = item.options;
        const actionPath = options.path ?? item.action.__action.name;
        if (pathname === actionPath) {
          matchedAction = item.action;
          handlerOptions = {
            contextProvider: options.contextProvider,
            streamManager: options.streamManager,
          };
          break;
        }
      } else {
        if (pathname === item.__action.name) {
          matchedAction = item;
          break;
        }
      }
    }

    if (!matchedAction) {
      return new Response(
        JSON.stringify({
          status: 'NOT_FOUND',
          message: 'No action matched the request path.',
        }),
        { status: 404, headers: { 'Content-Type': 'application/json' } }
      );
    }

    return handleActionRequest(request, matchedAction, handlerOptions);
  };
}
