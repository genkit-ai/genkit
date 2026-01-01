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
import { beforeEach, describe, it } from 'node:test';
import { z } from 'zod';
import { defineBidiAction } from '../src/action.js';
import { initNodeFeatures } from '../src/node.js';
import { Registry } from '../src/registry.js';

initNodeFeatures();

describe('bidi action', () => {
  var registry: Registry;
  beforeEach(() => {
    registry = new Registry();
  });

  it('streamBidi ergonomic (push)', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chat',
        actionType: 'custom',
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

    const session = act.streamBidi();
    session.send('1');
    session.send('2');
    session.close();

    const chunks: string[] = [];
    for await (const chunk of session.stream) {
      chunks.push(chunk);
    }

    assert.deepStrictEqual(chunks, ['echo 1', 'echo 2']);
    assert.strictEqual(await session.output, 'done');
  });

  it('streamBidi pull (generator)', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chat',
        actionType: 'custom',
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

    async function* inputGen() {
      yield '1';
      yield '2';
    }

    const session = act.streamBidi(inputGen());

    const chunks: string[] = [];
    for await (const chunk of session.stream) {
      chunks.push(chunk);
    }

    assert.deepStrictEqual(chunks, ['echo 1', 'echo 2']);
    assert.strictEqual(await session.output, 'done');
  });

  it('classic run works on bidi action', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chat',
        actionType: 'custom',
        inputSchema: z.string(),
        outputSchema: z.string(),
      },
      async function* ({ inputStream }) {
        for await (const chunk of inputStream) {
          // Should receive exactly one chunk
          yield `echo ${chunk}`;
        }
        return 'done';
      }
    );

    const result = await act.run('1');
    assert.strictEqual(result.result, 'done');
  });

  it('classic run works on bidi action with streaming', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chat',
        actionType: 'custom',
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

    const chunks: string[] = [];
    const result = await act.run('1', {
      onChunk: (c) => chunks.push(c),
    });

    assert.deepStrictEqual(chunks, ['echo 1']);
    assert.strictEqual(result.result, 'done');
  });

  it('validates input stream items', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chat',
        actionType: 'custom',
        inputSchema: z.string(), // Input is string
      },
      async function* ({ inputStream }) {
        for await (const chunk of inputStream) {
          yield `echo ${chunk}`;
        }
      }
    );

    const session = act.streamBidi();
    // Bypass TS check to send invalid data
    (session as any).send(123);
    session.close();

    try {
      for await (const _ of session.stream) {
        // Consume
      }
      assert.fail('Should have thrown validation error');
    } catch (e: any) {
      // Zod validation error or Genkit validation error
      assert.ok(
        e.message.includes('Expected string, received number') ||
          e.name === 'ZodError' ||
          e.code === 'invalid_type' ||
          e.message.includes('Validation failed') ||
          e.message.includes('must be string')
      );
    }
  });

  it('bidi action receives init data', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chatWithInit',
        actionType: 'custom',
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

    const session = act.streamBidi(undefined, { init: { prefix: '>> ' } });
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

  it('validates init data', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chatWithInitValidation',
        actionType: 'custom',
        inputSchema: z.string(),
        initSchema: z.object({ count: z.number() }),
      },
      async function* ({ inputStream, init }) {
        yield `count: ${init?.count}`;
      }
    );

    // Invalid init (string instead of number)
    try {
      const session = act.streamBidi(undefined, {
        init: { count: '123' } as any,
      });
      for await (const _ of session.stream) {
        // Consume
      }
      assert.fail('Should have thrown validation error');
    } catch (e: any) {
      assert.ok(e.message.includes('count: must be number'));
    }
  });
});
