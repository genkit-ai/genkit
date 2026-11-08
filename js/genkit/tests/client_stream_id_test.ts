/**
 * Copyright 2026 Google LLC
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
import { describe, it } from 'node:test';
import { streamFlow } from '../src/client/index.js';

describe('streamFlow stream ID', () => {
  for (const error of [
    new Error('connection failed'),
    new DOMException('request aborted', 'AbortError'),
  ]) {
    it(
      `settles without an ID after ${error.name}`,
      { timeout: 2000 },
      async (t) => {
        t.mock.method(globalThis, 'fetch', async () => {
          throw error;
        });

        const call = streamFlow({ url: 'http://stub', input: 'hello' });
        const streamFailure = assert.rejects(
          call.stream[Symbol.asyncIterator]().next(),
          (actual) => actual === error
        );
        await assert.rejects(call.output, (actual) => actual === error);
        await streamFailure;
        assert.strictEqual(await call.streamId, null);
      }
    );
  }

  for (const streamId of [null, 'stream-123']) {
    it(`returns ${streamId} from response headers`, async (t) => {
      t.mock.method(
        globalThis,
        'fetch',
        async () =>
          new Response('data: {"result":"done"}\n\n', {
            headers: streamId ? { 'x-genkit-stream-id': streamId } : {},
          })
      );

      const call = streamFlow({ url: 'http://stub', input: 'hello' });
      assert.strictEqual(await call.streamId, streamId);
      assert.strictEqual(await call.output, 'done');
      const chunks: unknown[] = [];
      for await (const chunk of call.stream) chunks.push(chunk);
      assert.deepStrictEqual(chunks, []);
    });
  }

  it('retains a received ID when reading the body fails', async (t) => {
    const error = new Error('connection lost');
    t.mock.method(
      globalThis,
      'fetch',
      async () =>
        new Response(
          new ReadableStream({
            start(controller) {
              controller.error(error);
            },
          }),
          { headers: { 'x-genkit-stream-id': 'stream-123' } }
        )
    );

    const call = streamFlow({ url: 'http://stub', input: 'hello' });
    const streamFailure = assert.rejects(
      call.stream[Symbol.asyncIterator]().next(),
      (actual) => actual === error
    );
    await assert.rejects(call.output, (actual) => actual === error);
    await streamFailure;
    assert.strictEqual(await call.streamId, 'stream-123');
  });
});
