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
import { afterEach, describe, it } from 'node:test';
import { streamFlow } from '../src/client/client.js';

describe('streamFlow', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('settles streamId to null when fetch rejects before response headers', async () => {
    globalThis.fetch = (async () => {
      throw new Error('connection failed');
    }) as any;

    const call = streamFlow({
      url: 'http://localhost:9999/flow',
      input: 'hello',
    });

    const streamDrain = (async () => {
      for await (const _ of call.stream) {
        // drain
      }
    })().catch(() => {});

    await assert.rejects(call.output, /connection failed/);

    // streamId should resolve cleanly to null rather than staying pending indefinitely
    const streamIdResult = await Promise.race([
      call.streamId,
      new Promise((_, reject) =>
        setTimeout(
          () => reject(new Error('timed out waiting for streamId')),
          500
        )
      ),
    ]);

    assert.strictEqual(streamIdResult, null);
    await streamDrain;
  });

  it('preserves received streamId when headers arrive even if body read fails', async () => {
    globalThis.fetch = (async () => {
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.error(new Error('stream terminated abruptly'));
          },
        }),
        {
          status: 200,
          headers: {
            'Content-Type': 'text/event-stream',
            'x-genkit-stream-id': 'stream-xyz-123',
          },
        }
      );
    }) as any;

    const call = streamFlow({
      url: 'http://localhost:9999/flow',
      input: 'hello',
    });

    const streamDrain = (async () => {
      for await (const _ of call.stream) {
        // drain
      }
    })().catch(() => {});

    const streamId = await call.streamId;
    assert.strictEqual(streamId, 'stream-xyz-123');

    await assert.rejects(call.output, /stream terminated abruptly/);
    await streamDrain;
  });
});
