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

import * as assert from 'assert';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { streamFlow } from '../src/client/client.js';

function mockSseFetch(body: string) {
  return async (_url: string, _init?: RequestInit): Promise<Response> => {
    return new Response(body, {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    });
  };
}

describe('streamFlow SSE parser', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('parses spec-compliant events without a space after "data:"', async () => {
    const body =
      `data:${JSON.stringify({ message: 'a' })}\n\n` +
      `data:${JSON.stringify({ message: 'b' })}\n\n` +
      `data:${JSON.stringify({ result: 'done' })}\n\n`;
    globalThis.fetch = mockSseFetch(body) as typeof fetch;

    const response = streamFlow({ url: 'http://example.com/flow' });
    const chunks: string[] = [];
    for await (const chunk of response.stream) {
      chunks.push(chunk);
    }
    assert.deepStrictEqual(chunks, ['a', 'b']);
    assert.strictEqual(await response.output, 'done');
  });

  it('parses events with the conventional "data: " prefix', async () => {
    const body =
      `data: ${JSON.stringify({ message: 'a' })}\n\n` +
      `data: ${JSON.stringify({ result: 'done' })}\n\n`;
    globalThis.fetch = mockSseFetch(body) as typeof fetch;

    const response = streamFlow({ url: 'http://example.com/flow' });
    const chunks: string[] = [];
    for await (const chunk of response.stream) {
      chunks.push(chunk);
    }
    assert.deepStrictEqual(chunks, ['a']);
    assert.strictEqual(await response.output, 'done');
  });
});
