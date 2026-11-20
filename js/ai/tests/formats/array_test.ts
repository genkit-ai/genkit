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
import { arrayFormatter } from '../../src/formats/array.js';
import { GenerateResponseChunk } from '../../src/generate.js';
import { Message } from '../../src/message.js';
import type {
  GenerateResponseChunkData,
  MessageData,
} from '../../src/model.js';

describe('arrayFormat', () => {
  const streamingTests = [
    {
      desc: 'emits complete array items as they arrive',
      chunks: [
        {
          text: '[{"id": 1,',
          want: [],
        },
        {
          text: '"name": "first"}',
          want: [{ id: 1, name: 'first' }],
        },
        {
          text: ', {"id": 2, "name": "second"}]',
          want: [{ id: 2, name: 'second' }],
        },
      ],
    },
    {
      desc: 'handles single item arrays',
      chunks: [
        {
          text: '[{"id": 1, "name": "single"}]',
          want: [{ id: 1, name: 'single' }],
        },
      ],
    },
    {
      desc: 'handles preamble with code fence',
      chunks: [
        {
          text: 'Here is the array you requested:\n\n```json\n[',
          want: [],
        },
        {
          text: '{"id": 1, "name": "item"}]\n```',
          want: [{ id: 1, name: 'item' }],
        },
      ],
    },
  ];

  for (const st of streamingTests) {
    it(st.desc, () => {
      const parser = arrayFormatter.handler();
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
            previousChunks: chunks,
          })
        );
        chunks.push(newChunk);

        assert.deepStrictEqual(result, chunk.want);
      }
    });
  }

  const messageTests = [
    {
      desc: 'parses complete array response',
      message: {
        role: 'model',
        content: [{ text: '[{"id": 1, "name": "test"}]' }],
      },
      want: [{ id: 1, name: 'test' }],
    },
    {
      desc: 'parses empty array',
      message: {
        role: 'model',
        content: [{ text: '[]' }],
      },
      want: [],
    },
    {
      desc: 'parses array of strings',
      message: {
        role: 'model',
        content: [{ text: '["red", "green", "blue"]' }],
      },
      want: ['red', 'green', 'blue'],
    },
    {
      desc: 'parses array of numbers',
      message: {
        role: 'model',
        content: [{ text: '[1, 2.5, -3]' }],
      },
      want: [1, 2.5, -3],
    },
    {
      desc: 'parses array with preamble and code fence',
      message: {
        role: 'model',
        content: [{ text: 'Here is the array:\n\n```json\n[{"id": 1}]\n```' }],
      },
      want: [{ id: 1 }],
    },
  ];

  for (const rt of messageTests) {
    it(rt.desc, () => {
      const parser = arrayFormatter.handler();
      assert.deepStrictEqual(
        parser.parseMessage(new Message(rt.message as MessageData)),
        rt.want
      );
    });
  }

  const errorTests = [
    {
      desc: 'throws error for non-array schema type',
      request: {
        messages: [],
        output: {
          schema: { type: 'string' },
        },
      },
      wantError: /Must supply an 'array' schema type/,
    },
    {
      desc: 'throws error for object schema type',
      request: {
        messages: [],
        output: {
          schema: { type: 'object' },
        },
      },
      wantError: /Must supply an 'array' schema type/,
    },
  ];

  for (const et of errorTests) {
    it(et.desc, () => {
      assert.throws(() => {
        arrayFormatter.handler(et.request);
      }, et.wantError);
    });
  }
});
