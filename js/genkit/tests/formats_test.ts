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

import { stripUndefinedProps } from '@genkit-ai/core';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { genkit, type GenkitBeta } from '../src/beta.js';
import { defineEchoModel } from './helpers.js';

describe('formats', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkit({});
    ai.defineFormat(
      {
        name: 'banana',
        format: 'banana',
        constrained: true,
      },
      (schema) => {
        let instructions: string | undefined;

        if (schema) {
          instructions = `Output should be in banana format`;
        }

        return {
          parseChunk: (chunk) => {
            return `banana: ${chunk.content[0].text}`;
          },

          parseMessage: (message) => {
            return `banana: ${message.content[0].text}`;
          },

          instructions,
        };
      }
    );
  });

  it('lets you define and use a custom output format with native constrained generation', async () => {
    defineEchoModel(ai, { supports: { constrained: 'all' } });

    const { output } = await ai.generate({
      model: 'echoModel',
      prompt: 'hi',
      output: { format: 'banana' },
    });

    assert.strictEqual(output, 'banana: Echo: hi');

    const { response, stream } = await ai.generateStream({
      model: 'echoModel',
      prompt: 'hi',
      output: { format: 'banana' },
    });
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(`${chunk.output}`);
    }
    assert.deepStrictEqual(chunks, ['banana: 3', 'banana: 2', 'banana: 1']);
    assert.strictEqual((await response).output, 'banana: Echo: hi');
    assert.deepStrictEqual(stripUndefinedProps((await response).request), {
      config: {},
      messages: [
        {
          content: [{ text: 'hi' }],
          role: 'user',
        },
      ],
      output: {
        constrained: true,
        format: 'banana',
      },
      tools: [],
    });
  });

  it('lets you define and use a custom output format with simulated constrained generation', async () => {
    defineEchoModel(ai, { supports: { constrained: 'none' } });

    const { output } = await ai.generate({
      model: 'echoModel',
      prompt: 'hi',
      output: { format: 'banana' },
    });

    assert.strictEqual(output, 'banana: Echo: hi');

    const { response, stream } = await ai.generateStream({
      model: 'echoModel',
      prompt: 'hi',
      output: { format: 'banana' },
    });
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(`${chunk.output}`);
    }
    assert.deepStrictEqual(chunks, ['banana: 3', 'banana: 2', 'banana: 1']);
    assert.strictEqual((await response).output, 'banana: Echo: hi');
    assert.deepStrictEqual(stripUndefinedProps((await response).request), {
      config: {},
      messages: [
        {
          content: [{ text: 'hi' }],
          role: 'user',
        },
      ],
      output: {
        constrained: true,
        format: 'banana',
      },
      tools: [],
    });
  });

  it('used explicitly specified output options overriding format options', async () => {
    defineEchoModel(ai, { supports: { constrained: 'all' } });
    const response = await ai.generate({
      model: 'echoModel',
      prompt: 'hi',
      output: {
        format: 'banana',
        // Explicitly specified, should ignore whatever format sets
        constrained: false,
        jsonSchema: { type: 'string' },
      },
    });
    assert.deepStrictEqual(stripUndefinedProps(response.request), {
      config: {},
      messages: [
        {
          content: [
            { text: 'hi' },
            {
              text: 'Output should be in banana format',
              metadata: { purpose: 'output' },
            },
          ],
          role: 'user',
        },
      ],
      output: {
        constrained: false,
        format: 'banana',
        schema: {
          type: 'string',
        },
      },
      tools: [],
    });
  });
});
