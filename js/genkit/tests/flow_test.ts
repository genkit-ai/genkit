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

import { z } from '@genkit-ai/core';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, type Genkit } from '../src/index.js';

describe('flow', () => {
  let ai: Genkit;

  beforeEach(() => {
    ai = genkit({
      context: { something: 'extra' },
    });
  });

  it('calls simple flow', async () => {
    const bananaFlow = ai.defineFlow('banana', () => 'banana');

    assert.strictEqual(await bananaFlow(), 'banana');
  });

  it('streams simple chunks (no schema defined)', async () => {
    const streamingBananaFlow = ai.defineFlow(
      'banana',
      (input: string, sendChunk) => {
        for (let i = 0; i < input.length; i++) {
          sendChunk(input.charAt(i));
        }
        return input;
      }
    );

    const { stream, output } = streamingBananaFlow.stream('banana');
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk as string);
    }
    assert.strictEqual(await output, 'banana');
    assert.deepStrictEqual(chunks, ['b', 'a', 'n', 'a', 'n', 'a']);
  });

  it('streams simple chunks with schema defined', async () => {
    const streamingBananaFlow = ai.defineFlow(
      {
        name: 'banana',
        inputSchema: z.string(),
        streamSchema: z.string(),
      },
      (input, sendChunk) => {
        for (let i = 0; i < input.length; i++) {
          sendChunk(input.charAt(i));
        }
        return input;
      }
    );

    const { stream, output } = streamingBananaFlow.stream('banana');
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    assert.deepStrictEqual(chunks, ['b', 'a', 'n', 'a', 'n', 'a']);
    assert.strictEqual(await output, 'banana');

    // a "streaming" flow can be invoked in non-streaming mode.
    assert.strictEqual(await streamingBananaFlow('banana2'), 'banana2');
  });

  it('passes through the context', async () => {
    const bananaFlow = ai.defineFlow('banana', (_, { context }) =>
      JSON.stringify(context)
    );

    assert.strictEqual(
      await bananaFlow(undefined, { context: { foo: 'bar' } }),
      '{"something":"extra","foo":"bar"}'
    );
  });
});
