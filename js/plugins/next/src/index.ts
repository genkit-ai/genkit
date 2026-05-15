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
import {
  Action,
  AsyncTaskQueue,
  StreamNotFoundError,
  type ActionContext,
  type ActionStreamInput,
  type StreamManager,
  type z,
} from 'genkit/beta';
import {
  getCallableJSON,
  getHttpStatus,
  type ContextProvider,
  type RequestData,
} from 'genkit/context';
import { NextRequest, NextResponse } from 'next/server.js';
export { NextRequest, NextResponse, z, type Action, type ActionContext };

const delimiter = '\n\n';

async function subscribeToStream<S, O>(
  streamManager: StreamManager,
  streamId: string
): Promise<NextResponse | null> {
  try {
    const encoder = new TextEncoder();
    const { readable, writable } = new TransformStream();
    const writer = writable.getWriter();
    await streamManager.subscribe(streamId, {
      onChunk: (chunk) => {
        writer.write(
          encoder.encode(
            'data: ' + JSON.stringify({ message: chunk }) + delimiter
          )
        );
      },
      onDone: (output) => {
        writer.write(
          encoder.encode(
            'data: ' + JSON.stringify({ result: output }) + delimiter
          )
        );
        writer.write(encoder.encode('END'));
        writer.close();
      },
      onError: (err) => {
        console.error(
          `Streaming request failed with error: ${(err as Error).message}\n${
            (err as Error).stack
          }`
        );
        writer.write(
          encoder.encode(
            `error: ${JSON.stringify({
              error: getCallableJSON(err),
            })}${delimiter}`
          )
        );
        writer.write(encoder.encode('END'));
        writer.close();
      },
    });
    return new NextResponse(readable, {
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
      return new NextResponse(null, { status: 204 });
    }
    if (e.status === 'DEADLINE_EXCEEDED') {
      const encoder = new TextEncoder();
      const { readable, writable } = new TransformStream();
      const writer = writable.getWriter();
      writer.write(
        encoder.encode(
          `error: ${JSON.stringify({
            error: getCallableJSON(e),
          })}${delimiter}`
        )
      );
      writer.write(encoder.encode('END'));
      writer.close();
      return new NextResponse(readable, {
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

async function getContext<C extends ActionContext, T>(
  request: NextRequest,
  input: T,
  provider: ContextProvider<C, T> | undefined
): Promise<C> {
  // Type cast is necessary because there is no runtime way to generate a context if C is provided to appRoute
  // but contextProvider is missing. When I'm less sleepy/busy I'll see if I can make this a type error.
  const context = {} as C;
  if (!provider) {
    return context;
  }

  const r: RequestData = {
    method: request.method as RequestData['method'],
    headers: {},
    input,
  };
  request.headers.forEach((val, key) => {
    r.headers[key.toLowerCase()] = val;
  });
  return await provider(r);
}

function appRoute<
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
) {
  return async (req: NextRequest): Promise<NextResponse> => {
    let context: C = {} as C;
    const { data: input } = await req.json();
    const streamId = req.headers.get('x-genkit-stream-id');
    if (req.headers.get('accept') !== 'text/event-stream') {
      try {
        context = await getContext(req, input, opts?.contextProvider);
      } catch (e) {
        console.error('Error gathering context for running action:', e);
        return NextResponse.json(
          { error: getCallableJSON(e) },
          { status: getHttpStatus(e) }
        );
      }
      try {
        const resp = await action.run(input, {
          context,
          abortSignal: req.signal,
        });
        const response = NextResponse.json({ result: resp.result });
        if (opts?.streamManager && streamId) {
          response.headers.set('x-genkit-stream-id', streamId);
        }
        return response;
      } catch (e) {
        // For security reasons, log the error rather than responding with it.
        console.error('Error calling action:', e);
        return NextResponse.json(
          { error: getCallableJSON(e) },
          { status: getHttpStatus(e) }
        );
      }
    }

    try {
      context = await getContext(req, input, opts?.contextProvider);
    } catch (e) {
      console.error('Error gathering context for streaming action:', e);
      return new NextResponse(
        `error: ${JSON.stringify(getCallableJSON(e))}${delimiter}END`,
        { status: getHttpStatus(e) }
      );
    }
    const streamManager = opts?.streamManager;
    if (streamManager && streamId) {
      const response = await subscribeToStream(streamManager, streamId);
      if (response) {
        return response;
      }
    }

    const streamIdToUse = randomUUID();
    const encoder = new TextEncoder();
    const { readable, writable } = new TransformStream();

    // Not using a dangling promise causes this closure to block on the stream being drained,
    // which doesn't happen until the NextResponse is consumed later in the cosure.
    // TODO: Add ping comments at regular intervals between streaming responses to mitigate
    // timeouts.
    (async (): Promise<void> => {
      const writer = writable.getWriter();
      const taskQueue = new AsyncTaskQueue();
      let durableStream: ActionStreamInput<S, O> | undefined = undefined;
      if (streamManager) {
        durableStream = await streamManager.open(streamIdToUse);
      }
      try {
        const output = action.run(input, {
          context,
          abortSignal: req.signal,
          onChunk: (chunk) => {
            if (durableStream) {
              taskQueue.enqueue(() => durableStream!.write(chunk));
            }
            taskQueue.enqueue(() =>
              writer.write(
                encoder.encode(
                  `data: ${JSON.stringify({ message: chunk })}${delimiter}`
                )
              )
            );
          },
        });
        const finalOutput = await output;
        if (durableStream) {
          taskQueue.enqueue(() => durableStream!.done(finalOutput.result));
        }
        taskQueue.enqueue(() =>
          writer.write(
            encoder.encode(
              `data: ${JSON.stringify({ result: finalOutput.result })}${delimiter}`
            )
          )
        );
        taskQueue.enqueue(() => writer.write(encoder.encode('END')));
      } catch (err) {
        if (durableStream) {
          taskQueue.enqueue(() => durableStream!.error(err));
        }
        console.error('Error streaming action:', err);
        taskQueue.enqueue(() =>
          writer.write(
            encoder.encode(
              `error: ${JSON.stringify(getCallableJSON(err))}` + '\n\n'
            )
          )
        );
        taskQueue.enqueue(() => writer.write(encoder.encode('END')));
      } finally {
        await taskQueue.merge();
        await writer.close();
      }
    })();

    const headers = {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Transfer-Encoding': 'chunked',
    };
    if (streamManager) {
      headers['x-genkit-stream-id'] = streamIdToUse;
    }

    return new NextResponse(readable, {
      status: 200,
      headers,
    });
  };
}

export default appRoute;
export { appRoute };
