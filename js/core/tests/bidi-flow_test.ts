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
import { describe, it } from 'node:test';
import { z } from 'zod';
import { bidiFlow } from '../src/flow.js';
import { initNodeFeatures } from '../src/node.js';

initNodeFeatures();

describe('bidi flow', () => {
  it('streamBidi ergonomic (push)', async () => {
    const flow = bidiFlow(
      {
        name: 'chatFlow',
        inputSchema: z.string(),
        outputSchema: z.string(),
      },
      async function* ({ inputStream }) {
        for await (const chunk of inputStream) {
          yield `echo ${chunk}`;
        }
        return 'done';
      }
    );

    const session = flow.streamBidi();
    session.send('1');
    session.send('2');
    session.close();

    const chunks: string[] = [];
    for await (const chunk of session.stream) {
      chunks.push(chunk);
    }

    assert.deepStrictEqual(chunks, ['echo 1', 'echo 2']);
    assert.strictEqual(await session.output, 'done');
    assert.strictEqual(flow.__action.actionType, 'flow');
    assert.ok(flow.__action.metadata?.bidi);
  });

  it('bidi flow receives init data', async () => {
    const flow = bidiFlow(
      {
        name: 'chatFlowWithInit',
        inputSchema: z.string(),
        outputSchema: z.string(),
        initSchema: z.object({ prefix: z.string() }),
      },
      async function* ({ inputStream, init }) {
        const prefix = init?.prefix || '';
        for await (const chunk of inputStream) {
          yield `${prefix}${chunk}`;
        }
        return 'done';
      }
    );

    const session = flow.streamBidi(undefined, { init: { prefix: '>> ' } });
    session.send('1');
    session.send('2');
    session.close();

    const chunks: string[] = [];
    for await (const chunk of session.stream) {
      chunks.push(chunk);
    }

    assert.deepStrictEqual(chunks, ['>> 1', '>> 2']);
    assert.strictEqual(await session.output, 'done');
  });

  it('validates init data in bidi flow', async () => {
    const flow = bidiFlow(
      {
        name: 'chatFlowWithInitValidation',
        inputSchema: z.string(),
        initSchema: z.object({ count: z.number() }),
      },
      async function* ({ init }) {
        yield `count: ${init?.count}`;
      }
    );

    try {
      const session = flow.streamBidi(undefined, {
        init: { count: '123' } as any,
      });
      for await (const _ of session.stream) {
        // consume
      }
      assert.fail('Should have thrown validation error');
    } catch (e: any) {
      assert.ok(e.message.includes('count: must be number'));
    }
  });
});
