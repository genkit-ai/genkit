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

import { describe, expect, it } from '@jest/globals';
import { UserFacingError, genkit, z } from 'genkit';
import { InMemoryStreamManager } from 'genkit/beta';
import { apiKey, type ApiKeyContext } from 'genkit/context';
import { NextRequest } from 'next/server.js';
import { appRoute } from '../src/index.js';

function chunks(
  text: string
): Array<
  | { type: 'data' | 'error'; content: Record<string, any> }
  | { type: 'comment' | 'invalid'; content: string }
  | { type: 'end' }
> {
  return text
    .split('\n\n')
    .filter((part) => part)
    .map((part) => {
      if (part === 'END') {
        return { type: 'end' };
      }
      const [command, ...rest] = part.split(':');
      // Restore any additional :
      const data = rest.join(':');
      if (command === 'data' || command === 'error') {
        // The split : also took out : from JSON
        return { type: command, content: JSON.parse(data.trim()) };
      }
      return { type: command === '' ? 'comment' : 'invalid', content: data };
    });
}

describe('appRoute', () => {
  const ai = genkit({});
  const echoContext = ai.defineFlow(
    {
      name: 'echoContext',
      outputSchema: z.object({
        auth: z
          .object({
            apiKey: z.string().optional(),
          })
          .optional(),
      }),
    },
    (_: unknown, { context }) => {
      return context as ApiKeyContext;
    }
  );

  const echoScalar = ai.defineFlow(
    {
      name: 'echoScalar',
      inputSchema: z.string(),
      outputSchema: z.string(),
    },
    (input: string) => input
  );

  const echoObject = ai.defineFlow(
    {
      name: 'echoObject',
      inputSchema: z.object({
        user: z.string(),
        email: z.string(),
      }),
      outputSchema: z.object({
        user: z.string(),
        email: z.string(),
      }),
    },
    (input) => input
  );

  const userErrorThrower = ai.defineFlow('userErrorThrower', () => {
    throw new UserFacingError('INVALID_ARGUMENT', 'a user facing error');
  });

  const internalErrorThrower = ai.defineFlow('internalErrorThrower', () => {
    throw new Error('This may have sensitive data');
  });

  it('supports context providers', async () => {
    const request = () =>
      new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'myApiKey',
        },
        body: JSON.stringify({ data: null }),
      });

    let route = appRoute(echoContext);
    let response = await route(request());
    expect(response.status).toEqual(200);
    expect(await response.json()).toEqual({ result: {} });

    let r = request();
    r.headers.set('authorization', 'myApiKey');
    route = appRoute(echoContext, { contextProvider: apiKey() });
    response = await route(r);
    expect(response.status).toEqual(200);
    expect(await response.json()).toEqual({
      result: { auth: { apiKey: 'myApiKey' } },
    });

    route = appRoute(echoContext, { contextProvider: apiKey('myApiKey') });
    r = request();
    r.headers.set('authorization', 'some other API key');
    response = await route(r);
    expect(response.status).toEqual(403);
    expect(await response.json()).toEqual({
      error: { status: 'PERMISSION_DENIED', message: 'Permission Denied' },
    });

    route = appRoute(echoContext, {
      contextProvider: apiKey('Some other key'),
    });
    r = request();
    r.headers.set('authorization', 'some other API key');
    r.headers.set('accept', 'text/event-stream');
    response = await route(r);
    expect(response.status).toEqual(403);
    expect(chunks(await response.text())).toEqual([
      {
        type: 'error',
        content: {
          status: 'PERMISSION_DENIED',
          message: 'Permission Denied',
        },
      },
      {
        type: 'end',
      },
    ]);
  });

  it('supports scalars', async () => {
    const request = () =>
      new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: 'Hello, world!' }),
      });

    const route = appRoute(echoScalar);
    let response = await route(request());
    expect(response.status).toEqual(200);
    expect(await response.json()).toEqual({ result: 'Hello, world!' });

    const r = request();
    r.headers.append('accept', 'text/event-stream');
    response = await route(r);
    expect(chunks(await response.text())).toEqual([
      {
        type: 'data',
        content: {
          result: 'Hello, world!',
        },
      },
      { type: 'end' },
    ]);
  });

  it('supports objects', async () => {
    const request = () =>
      new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: {
            user: 'person',
            email: 'person@google.com',
          },
        }),
      });

    const route = appRoute(echoObject);
    let response = await route(request());
    expect(response.status).toEqual(200);
    expect(await response.json()).toEqual({
      result: { user: 'person', email: 'person@google.com' },
    });

    const r = request();
    r.headers.append('accept', 'text/event-stream');
    response = await route(r);
    expect(response.status).toEqual(200);
    expect(chunks(await response.text())).toEqual([
      {
        type: 'data',
        content: {
          result: {
            user: 'person',
            email: 'person@google.com',
          },
        },
      },
      { type: 'end' },
    ]);
  });

  it('exposes user visible errors', async () => {
    const request = () =>
      new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          authorization: 'myApiKey',
        },
        body: JSON.stringify({ data: null }),
      });

    const route = appRoute(userErrorThrower);
    let response = await route(request());
    expect(response.status).toEqual(400);
    const json = await response.json();
    expect(json).toEqual({
      error: { status: 'INVALID_ARGUMENT', message: 'a user facing error' },
    });

    const r = request();
    r.headers.append('accept', 'text/event-stream');
    response = await route(r);
    expect(response.status).toEqual(200);
    expect(chunks(await response.text())).toEqual([
      {
        type: 'error',
        content: {
          message: 'a user facing error',
          status: 'INVALID_ARGUMENT',
        },
      },
      {
        type: 'end',
      },
    ]);
  });

  it('hides internal errors', async () => {
    const request = () =>
      new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          Authorization: 'myApiKey',
        },
        body: JSON.stringify({ data: null }),
      });

    const route = appRoute(internalErrorThrower);
    let response = await route(request());
    expect(response.status).toEqual(500);
    expect(await response.json()).toEqual({
      error: { status: 'INTERNAL', message: 'Internal Error' },
    });

    const r = request();
    r.headers.append('accept', 'text/event-stream');
    response = await route(r);
    expect(response.status).toEqual(200);
    expect(chunks(await response.text())).toEqual([
      {
        type: 'error',
        content: {
          message: 'Internal Error',
          status: 'INTERNAL',
        },
      },
      {
        type: 'end',
      },
    ]);
  });

  describe('durable streaming', () => {
    const streamingFlow = ai.defineFlow(
      {
        name: 'streamingFlow',
        inputSchema: z.string(),
        outputSchema: z.string(),
      },
      async (input, { sendChunk }) => {
        for (let i = 0; i < 10; i++) {
          await new Promise((r) => {
            sendChunk(i);
            r(i);
          });
        }
        return input;
      }
    );

    it('supports durable streaming', async () => {
      const route = appRoute(streamingFlow, {
        streamManager: new InMemoryStreamManager(),
      });
      const request = new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          accept: 'text/event-stream',
        },
        body: JSON.stringify({ data: 'Hello, world!' }),
      });
      const response = await route(request);
      const streamId = response.headers.get('x-genkit-stream-id');
      expect(streamId).not.toBeNull();
      await response.text();
      const subRequest = new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          accept: 'text/event-stream',
          'x-genkit-stream-id': streamId!,
        },
        body: JSON.stringify({ data: 'Hello, world!' }),
      });
      const subResponse = await route(subRequest);
      const subChunks = chunks(await subResponse.text());
      expect(subChunks.length).toEqual(12);
      expect(subChunks[10]).toEqual({
        type: 'data',
        content: { result: 'Hello, world!' },
      });
      expect(subChunks[11]).toEqual({ type: 'end' });
    });

    it('returns 204 for unknown streamId', async () => {
      const route = appRoute(streamingFlow, {
        streamManager: new InMemoryStreamManager(),
      });
      const request = new NextRequest('http://localhost/api/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          accept: 'text/event-stream',
          'x-genkit-stream-id': 'banana',
        },
        body: JSON.stringify({ data: 'Hello, world!' }),
      });
      const response = await route(request);
      expect(response.status).toEqual(204);
    });
  });
});
