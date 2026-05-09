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
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import assert from 'node:assert';
import { beforeEach, describe, it } from 'node:test';
import { configureFormats } from '../../src/formats/index.js';
import { jsonFormatter } from '../../src/formats/json.js';
import { GenerateResponseChunk, generateStream } from '../../src/generate.js';
import { Message } from '../../src/message.js';
import type {
  GenerateResponseChunkData,
  MessageData,
} from '../../src/model.js';
import { defineProgrammableModel, runAsync } from '../helpers.js';

initNodeFeatures();

describe('jsonFormat', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
  });

  const streamingTests = [
    {
      desc: 'parses complete JSON object',
      chunks: [
        {
          text: '{"id": 1, "name": "test"}',
          want: { id: 1, name: 'test' },
        },
      ],
    },
    {
      desc: 'handles partial JSON',
      chunks: [
        {
          text: '{"id": 1',
          want: { id: 1 },
        },
        {
          text: ', "name": "test"}',
          want: { id: 1, name: 'test' },
        },
      ],
    },
    {
      desc: 'handles preamble with code fence',
      chunks: [
        {
          text: 'Here is the JSON:\n\n```json\n',
          want: null,
        },
        {
          text: '{"id": 1}\n```',
          want: { id: 1 },
        },
      ],
    },
  ];

  for (const st of streamingTests) {
    it(st.desc, () => {
      const parser = jsonFormatter.handler();
      const chunks: GenerateResponseChunkData[] = [];

      for (const chunk of st.chunks) {
        const newChunk: GenerateResponseChunkData = {
          index: 0,
          role: 'model',
          content: [{ text: chunk.text }],
        };

        const result = parser.parseChunk!(
          new GenerateResponseChunk(newChunk, {
            index: 0,
            role: 'model',
            previousChunks: [...chunks],
          })
        );
        chunks.push(newChunk);

        assert.deepStrictEqual(result, chunk.want);
      }
    });
  }

  const messageTests = [
    {
      desc: 'parses complete JSON response',
      message: {
        role: 'model',
        content: [{ text: '{"id": 1, "name": "test"}' }],
      },
      want: { id: 1, name: 'test' },
    },
    {
      desc: 'handles empty response',
      message: {
        role: 'model',
        content: [{ text: '' }],
      },
      want: null,
    },
    {
      desc: 'parses JSON with preamble and code fence',
      message: {
        role: 'model',
        content: [{ text: 'Here is the JSON:\n\n```json\n{"id": 1}\n```' }],
      },
      want: { id: 1 },
    },
  ];

  for (const rt of messageTests) {
    it(rt.desc, () => {
      const parser = jsonFormatter.handler();
      assert.deepStrictEqual(
        parser.parseMessage(new Message(rt.message as MessageData)),
        rt.want
      );
    });
  }
});

describe('jsonFormat e2e', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
    configureFormats(registry);
  });

  it('injects the instructions into the request', async () => {
    const pm = defineProgrammableModel(registry);
    pm.handleResponse = async (req, sc) => {
      await runAsync(() => sc?.({ content: [{ text: '```\n{' }] }));
      await runAsync(() => sc?.({ content: [{ text: '"foo": "b' }] }));
      await runAsync(() => sc?.({ content: [{ text: 'ar"' }] }));
      await runAsync(() => sc?.({ content: [{ text: '}\n```"' }] }));
      return await runAsync(() => ({
        message: {
          role: 'model',
          content: [{ text: '```\n{"foo": "bar"}\n```' }],
        },
      }));
    };

    const { response, stream } = await generateStream(registry, {
      model: 'programmableModel',
      prompt: 'generate json',
      output: {
        format: 'json',
        schema: z.object({
          foo: z.string(),
        }),
      },
    });
    const chunks: any = [];
    for await (const chunk of stream) {
      chunks.push(chunk.output);
    }
    assert.deepEqual((await response).output, { foo: 'bar' });
    assert.deepStrictEqual(chunks, [
      {},
      { foo: 'b' },
      { foo: 'bar' },
      { foo: 'bar' },
    ]);
  });
});
