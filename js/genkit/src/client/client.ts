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

import { Channel, createTask } from '@genkit-ai/core/async';

const __flowStreamDelimiter = '\n\n';

/**
 * Invoke and stream response from a deployed flow.
 *
 * For example:
 *
 * ```js
 * import { streamFlow } from 'genkit/beta/client';
 *
 * const response = streamFlow({
 *   url: 'https://my-flow-deployed-url',
 *   input: 'foo',
 * });
 * for await (const chunk of response.stream) {
 *   console.log(chunk);
 * }
 * console.log(await response.output);
 * ```
 */
export function streamFlow<O = any, S = any>({
  url,
  input,
  streamId,
  headers,
  abortSignal,
}: {
  /** URL of the deployed flow. */
  url: string;
  /** Flow input. */
  input?: any;
  /** Stream ID to connect to. */
  streamId?: string;
  /** A map of HTTP headers to be added to the HTTP call. */
  headers?: Record<string, string>;
  /** Abort signal to abort the request. */
  abortSignal?: AbortSignal;
}): {
  readonly output: Promise<O>;
  readonly stream: AsyncIterable<S>;
  readonly streamId: Promise<string | null>;
} {
  const channel = new Channel<S>();
  const streamIdTask = createTask<string | null>();

  const operationPromise = __flowRunEnvelope({
    url,
    input,
    sendChunk: (c) => channel.send(c),
    headers: streamId
      ? { ...headers, 'x-genkit-stream-id': streamId }
      : headers,
    abortSignal,
    responseCallback: (response) => {
      streamIdTask.resolve(response.headers.get('x-genkit-stream-id'));
    },
  });
  operationPromise.then(
    () => channel.close(),
    (err) => channel.error(err)
  );

  return {
    output: operationPromise,
    stream: channel,
    streamId: streamIdTask.promise,
  };
}

async function __flowRunEnvelope({
  url,
  input,
  sendChunk,
  headers,
  abortSignal,
  responseCallback,
}: {
  url: string;
  input: any;
  sendChunk: (chunk: any) => void;
  headers?: Record<string, string>;
  abortSignal?: AbortSignal;
  responseCallback?: (response: Response) => void;
}) {
  const response = await fetch(url, {
    method: 'POST',
    body: JSON.stringify({
      data: input,
    }),
    headers: {
      Accept: 'text/event-stream',
      'Content-Type': 'application/json',
      ...headers,
    },
    signal: abortSignal,
  });
  if (responseCallback) {
    responseCallback(response);
  }
  if (response.status === 204) {
    throw new Error('NOT_FOUND: Stream not found.');
  }
  if (response.status !== 200) {
    throw new Error(
      `Server returned: ${response.status}: ${await response.text()}`
    );
  }
  if (!response.body) {
    throw new Error('Response body is empty');
  }
  var reader = response.body.getReader();
  var decoder = new TextDecoder();

  let buffer = '';
  while (true) {
    const result = await reader.read();
    const decodedValue = decoder.decode(result.value);
    if (decodedValue) {
      buffer += decodedValue;
    }
    // If buffer includes the delimiter that means we are still receiving chunks.
    while (buffer.includes(__flowStreamDelimiter)) {
      const chunk = JSON.parse(
        buffer
          .substring(0, buffer.indexOf(__flowStreamDelimiter))
          .substring('data: '.length)
      );
      if (chunk.hasOwnProperty('message')) {
        sendChunk(chunk.message);
      } else if (chunk.hasOwnProperty('result')) {
        return chunk.result;
      } else if (chunk.hasOwnProperty('error')) {
        throw new Error(
          `${chunk.error.status}: ${chunk.error.message}\n${chunk.error.details}`
        );
      } else {
        throw new Error('unknown chunk format: ' + JSON.stringify(chunk));
      }
      buffer = buffer.substring(
        buffer.indexOf(__flowStreamDelimiter) + __flowStreamDelimiter.length
      );
    }
  }
  throw new Error('stream did not terminate correctly');
}

/**
 * Invoke a deployed flow over HTTP(s).
 *
 * For example:
 *
 * ```js
 * import { runFlow } from 'genkit/beta/client';
 *
 * const response = await runFlow({
 *   url: 'https://my-flow-deployed-url',
 *   input: 'foo',
 * });
 * console.log(await response);
 * ```
 */
export async function runFlow<O = any>({
  url,
  input,
  headers,
  abortSignal,
}: {
  /** URL of the deployed flow. */
  url: string;
  /** Flow input. */
  input?: any;
  /** A map of HTTP headers to be added to the HTTP call. */
  headers?: Record<string, string>;
  /** Abort signal to abort the request. */
  abortSignal?: AbortSignal;
}): Promise<O> {
  const response = await fetch(url, {
    method: 'POST',
    body: JSON.stringify({
      data: input,
    }),
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    signal: abortSignal,
  });
  if (response.status !== 200) {
    throw new Error(
      `Server returned: ${response.status}: ${await response.text()}`
    );
  }
  const wrappedResult = (await response.json()) as
    | { result: O }
    | { error: unknown };
  if ('error' in wrappedResult) {
    if (typeof wrappedResult.error === 'string') {
      throw new Error(wrappedResult.error);
    }
    // TODO: The callable protocol defines an HttpError that has a JSON format of
    // details?: string
    // httpErrorCode: { canonicalName: string }
    // message: string
    // Should we create a new error class that parses this and exposes it as fields?
    throw new Error(JSON.stringify(wrappedResult.error));
  }
  return wrappedResult.result;
}
