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
        outputSchema: z.string(),
        inputSchema: z.string(), // Option 1: messages are strings
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
        outputSchema: z.string(),
        inputSchema: z.string(),
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

    const session = act.streamBidi(undefined, { inputStream: inputGen() }); // Pass stream in options!

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
        outputSchema: z.string(),
        inputSchema: z.string(),
      },
      async function* ({ inputStream }) {
        const inputs: string[] = [];
        for await (const chunk of inputStream) {
          inputs.push(chunk);
        }
        return `done: ${inputs.join(', ')}`;
      }
    );

    const result = await act.run('1'); // Pass single message as input!
    assert.strictEqual(result.result, 'done: 1');
  });

  it('classic run works on bidi action with streaming', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chat',
        actionType: 'custom',
        outputSchema: z.string(),
        inputSchema: z.string(),
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

  it('bidi action receives init data', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chatWithInit',
        actionType: 'custom',
        outputSchema: z.string(),
        inputSchema: z.string(),
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

    const session = act.streamBidi({ prefix: '>> ' }); // Pass init as first argument!
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
  it('classic run works on bidi action with init', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chatWithInitRun',
        actionType: 'custom',
        outputSchema: z.string(),
        inputSchema: z.string(),
        initSchema: z.object({ prefix: z.string() }),
      },
      async function* ({ inputStream, init }) {
        const prefix = init?.prefix || '';
        const inputs: string[] = [];
        for await (const chunk of inputStream) {
          inputs.push(chunk);
        }
        return `${prefix}${inputs.join(', ')}`;
      }
    );

    const result = await act.run('1', { init: { prefix: '>> ' } });
    assert.strictEqual(result.result, '>> 1');
  });

  it('classic stream works on bidi action with init', async () => {
    const act = defineBidiAction(
      registry,
      {
        name: 'chatWithInitStream',
        actionType: 'custom',
        outputSchema: z.string(),
        inputSchema: z.string(),
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

    const session = act.stream('1', { init: { prefix: '>> ' } });

    const chunks: string[] = [];
    for await (const chunk of session.stream) {
      chunks.push(chunk);
    }

    assert.deepStrictEqual(chunks, ['>> 1']);
    assert.strictEqual(await session.output, 'done');
  });
});
