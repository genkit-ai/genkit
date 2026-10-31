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
import { afterEach, beforeEach, describe, it } from 'node:test';
import { streamFlow } from '../src/client/client.js';

const encoder = new TextEncoder();

function sse(event: unknown): Uint8Array {
  return encoder.encode(`data: ${JSON.stringify(event)}\n\n`);
}

/**
 * A `fetch` replacement that serves the given byte chunks as the body of a
 * 200 response and then reports the end of the stream.
 *
 * The body reader fails if it is read again many times after it reported
 * `done`. A client that ignores `done` would otherwise spin forever in
 * microtasks, which starves the event loop (timers never fire), so neither a
 * test timeout nor `Promise.race` could rescue it. Failing the read turns that
 * into an ordinary test failure.
 */
function mockStreamingFetch(chunks: Uint8Array[]) {
  const maxReadsAfterDone = 100;
  return async (_url: string | URL | Request, _init?: RequestInit) => {
    const queue = [...chunks];
    let readsAfterDone = 0;
    return {
      status: 200,
      headers: new Headers(),
      body: {
        getReader: () => ({
          read: async (): Promise<ReadableStreamReadResult<Uint8Array>> => {
            const value = queue.shift();
            if (value) {
              return { done: false, value };
            }
            if (++readsAfterDone > maxReadsAfterDone) {
              throw new Error(
                `read() called ${readsAfterDone} times after the stream was done`
              );
            }
            return { done: true, value: undefined };
          },
        }),
      },
    } as unknown as Response;
  };
}

describe('streamFlow', () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  /** Runs streamFlow to completion, collecting chunks and the outcome. */
  async function run() {
    const { stream, output } = streamFlow<string, string>({
      url: 'http://localhost/flow',
      input: 'hi',
    });
    const chunks: string[] = [];
    let streamError: Error | undefined;
    const consumed = (async () => {
      try {
        for await (const chunk of stream) {
          chunks.push(chunk);
        }
      } catch (e) {
        streamError = e as Error;
      }
    })();
    const settled = await output.then(
      (value) => ({ value, error: undefined as Error | undefined }),
      (error: Error) => ({ value: undefined, error })
    );
    await consumed;
    return { chunks, streamError, ...settled };
  }

  describe('multi-byte characters', () => {
    // 2-byte (é), 3-byte (世界) and 4-byte (emoji) UTF-8 sequences.
    const message = 'café 世界 😀';

    it('decodes characters split across network chunks', async () => {
      const full = sse({ message });
      // Split the event at every possible byte offset, including offsets in
      // the middle of a character.
      for (let cut = 1; cut < full.length; cut++) {
        globalThis.fetch = mockStreamingFetch([
          full.slice(0, cut),
          full.slice(cut),
          sse({ result: 'done' }),
        ]);

        const { chunks, value, error } = await run();

        assert.strictEqual(error, undefined, `cut at byte ${cut}`);
        assert.deepStrictEqual(chunks, [message], `cut at byte ${cut}`);
        assert.strictEqual(value, 'done', `cut at byte ${cut}`);
      }
    });

    it('decodes characters split across every byte', async () => {
      const full = sse({ message });
      const chunks = Array.from(full, (byte) => Uint8Array.of(byte));
      globalThis.fetch = mockStreamingFetch([
        ...chunks,
        sse({ result: message }),
      ]);

      const { chunks: received, value, error } = await run();

      assert.strictEqual(error, undefined);
      assert.deepStrictEqual(received, [message]);
      assert.strictEqual(value, message);
    });
  });

  describe('stream that ends early', () => {
    it('fails when the stream ends before the result event', async () => {
      globalThis.fetch = mockStreamingFetch([sse({ message: 'partial' })]);

      const { chunks, streamError, value, error } = await run();

      assert.deepStrictEqual(chunks, ['partial']);
      assert.strictEqual(value, undefined);
      assert.ok(error, 'output should reject');
      assert.match(error.message, /stream did not terminate correctly/);
      assert.ok(streamError, 'stream iteration should fail too');
      assert.match(streamError.message, /stream did not terminate correctly/);
    });

    it('fails when the stream is empty', async () => {
      globalThis.fetch = mockStreamingFetch([]);

      const { chunks, error } = await run();

      assert.deepStrictEqual(chunks, []);
      assert.ok(error, 'output should reject');
      assert.match(error.message, /stream did not terminate correctly/);
    });

    it('fails when the stream ends in the middle of an event', async () => {
      const full = sse({ result: 'done' });
      globalThis.fetch = mockStreamingFetch([full.slice(0, full.length - 4)]);

      const { value, error } = await run();

      assert.strictEqual(value, undefined);
      assert.ok(error, 'output should reject');
      assert.match(error.message, /stream did not terminate correctly/);
    });

    it('still resolves when the result event arrives', async () => {
      globalThis.fetch = mockStreamingFetch([
        sse({ message: 'a' }),
        sse({ message: 'b' }),
        sse({ result: 'done' }),
      ]);

      const { chunks, value, error, streamError } = await run();

      assert.strictEqual(error, undefined);
      assert.strictEqual(streamError, undefined);
      assert.deepStrictEqual(chunks, ['a', 'b']);
      assert.strictEqual(value, 'done');
    });
  });
});
