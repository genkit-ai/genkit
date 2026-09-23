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
import * as assert from 'assert';
import JSON5 from 'json5';
import { describe, it } from 'node:test';
import { defineFormat } from '../../src/formats/index.js';
import { jsonlFormatter } from '../../src/formats/jsonl.js';
import { GenerateResponseChunk, generateStream } from '../../src/generate.js';
import { Message } from '../../src/message.js';
import type {
  GenerateResponseChunkData,
  MessageData,
} from '../../src/model.js';
import { defineProgrammableModel, runAsync } from '../helpers.js';

initNodeFeatures();

describe('jsonlFormat', () => {
  const streamingTests = [
    {
      desc: 'emits complete JSON objects as they arrive',
      chunks: [
        {
          text: '{"id": 1, "name": "first"}\n',
          want: [{ id: 1, name: 'first' }],
        },
        {
          text: '{"id": 2, "name": "second"}\n{"id": 3',
          want: [{ id: 2, name: 'second' }],
        },
        {
          text: ', "name": "third"}\n',
          want: [{ id: 3, name: 'third' }],
        },
      ],
    },
    {
      desc: 'handles single object',
      chunks: [
        {
          text: '{"id": 1, "name": "single"}\n',
          want: [{ id: 1, name: 'single' }],
        },
      ],
    },
    {
      desc: 'emits a completed object only once when its newline arrives later',
      chunks: [
        {
          text: '{"id": 1, "name": "single"',
          want: [],
        },
        {
          text: '}',
          want: [{ id: 1, name: 'single' }],
        },
        {
          text: '\n',
          want: [],
        },
      ],
    },
    {
      desc: 'does not re-emit an object when a block comment is split across chunks',
      chunks: [
        {
          text: '{id: 1}',
          want: [{ id: 1 }],
        },
        {
          text: '/',
          want: [],
        },
        {
          text: '* trailing */',
          want: [],
        },
        {
          text: '\n',
          want: [],
        },
      ],
    },
    {
      desc: 'does not re-emit an object when a line comment is split across chunks',
      chunks: [
        {
          text: '{id: 1}',
          want: [{ id: 1 }],
        },
        {
          text: '/',
          want: [],
        },
        {
          text: '/ trailing',
          want: [],
        },
        {
          text: '\n',
          want: [],
        },
      ],
    },
    {
      desc: 'resumes with the next object after a split trailing comment',
      chunks: [
        {
          text: '{id: 1}',
          want: [{ id: 1 }],
        },
        {
          text: '/',
          want: [],
        },
        {
          text: '* trailing */',
          want: [],
        },
        {
          text: '\n{id: 2}\n',
          want: [{ id: 2 }],
        },
      ],
    },
    {
      desc: 'handles braces in JSON5 strings and comments',
      chunks: [
        {
          text: "{value: '}', nested: [/* } */",
          want: [],
        },
        {
          text: '{id: 1}]}\n',
          want: [{ value: '}', nested: [{ id: 1 }] }],
        },
      ],
    },
    {
      desc: 'handles preamble with code fence',
      chunks: [
        {
          text: 'Here are the objects:\n\n```\n',
          want: [],
        },
        {
          text: '{"id": 1, "name": "item"}\n```',
          want: [{ id: 1, name: 'item' }],
        },
      ],
    },
    {
      desc: 'ignores non-object lines',
      chunks: [
        {
          text: 'First object:\n{"id": 1}\nSecond object:\n{"id": 2}\n',
          want: [{ id: 1 }, { id: 2 }],
        },
      ],
    },
  ];

  for (const st of streamingTests) {
    it(st.desc, () => {
      const parser = jsonlFormatter.handler().parseChunk!;
      const chunks: GenerateResponseChunkData[] = [];

      for (const chunk of st.chunks) {
        const newChunk: GenerateResponseChunkData = {
          index: 0,
          role: 'model',
          content: [{ text: chunk.text }],
        };

        const responseChunk = new GenerateResponseChunk(newChunk, {
          index: 0,
          role: 'model',
          previousChunks: chunks,
          parser,
        });

        const result = responseChunk.output;
        chunks.push(newChunk);

        assert.deepStrictEqual(result, chunk.want);
      }
    });
  }

  it('rebuilds state when an earlier previous chunk changes', () => {
    const parser = jsonlFormatter.handler().parseChunk!;
    const prefixA: GenerateResponseChunkData['content'] = [{ text: '{a:' }];
    const prefixB: GenerateResponseChunkData['content'] = [{ text: '{b:' }];
    const middle: GenerateResponseChunkData['content'] = [{ text: '1' }];
    const close: GenerateResponseChunkData['content'] = [{ text: '}' }];

    const responseChunk = (
      content: GenerateResponseChunkData['content'],
      previousContents: GenerateResponseChunkData['content'][]
    ) =>
      new GenerateResponseChunk(
        { index: 0, role: 'model', content },
        {
          index: 0,
          role: 'model',
          previousChunks: previousContents.map((previousContent) => ({
            index: 0,
            role: 'model',
            content: previousContent,
          })),
          parser,
        }
      );

    assert.deepStrictEqual(responseChunk(prefixA, []).output, []);
    assert.deepStrictEqual(responseChunk(middle, [prefixA]).output, []);
    assert.deepStrictEqual(responseChunk(close, [prefixB, middle]).output, [
      { b: 1 },
    ]);
    assert.deepStrictEqual(responseChunk(close, [prefixA, middle]).output, [
      { a: 1 },
    ]);
  });

  it('does not reparse prior incomplete chunks', () => {
    const originalParse = JSON5.parse;
    let parseCalls = 0;
    let textReads = 0;
    JSON5.parse = ((...args: Parameters<typeof originalParse>) => {
      parseCalls++;
      return originalParse(...args);
    }) as typeof JSON5.parse;

    try {
      const parser = jsonlFormatter.handler().parseChunk!;
      const chunks: GenerateResponseChunkData[] = [];
      const texts = ['{', ...Array(199).fill(' ')];

      for (const text of texts) {
        const newChunk: GenerateResponseChunkData = {
          index: 0,
          role: 'model',
          content: [
            {
              get text() {
                textReads++;
                return text;
              },
            },
          ],
        };
        const responseChunk = new GenerateResponseChunk(newChunk, {
          index: 0,
          role: 'model',
          previousChunks: chunks,
          parser,
        });

        const output = responseChunk.output;
        assert.deepStrictEqual(output, []);
        assert.strictEqual(responseChunk.output, output);
        chunks.push(newChunk);
      }

      assert.strictEqual(
        parseCalls,
        0,
        `expected no incomplete-prefix parse calls, got ${parseCalls}`
      );
      assert.strictEqual(
        textReads,
        texts.length,
        `expected one text read per chunk, got ${textReads}`
      );
    } finally {
      JSON5.parse = originalParse;
    }
  });

  const messageTests = [
    {
      desc: 'parses complete JSONL response',
      message: {
        role: 'model',
        content: [{ text: '{"id": 1, "name": "test"}\n{"id": 2}\n' }],
      },
      want: [{ id: 1, name: 'test' }, { id: 2 }],
    },
    {
      desc: 'handles empty response',
      message: {
        role: 'model',
        content: [{ text: '' }],
      },
      want: [],
    },
    {
      desc: 'parses JSONL with preamble and code fence',
      message: {
        role: 'model',
        content: [
          {
            text: 'Here are the objects:\n\n```\n{"id": 1}\n{"id": 2}\n```',
          },
        ],
      },
      want: [{ id: 1 }, { id: 2 }],
    },
  ];

  for (const rt of messageTests) {
    it(rt.desc, () => {
      const parser = jsonlFormatter.handler();
      assert.deepStrictEqual(
        parser.parseMessage(new Message(rt.message as MessageData)),
        rt.want
      );
    });
  }

  const errorTests = [
    {
      desc: 'throws error for non-array schema type',
      schema: { type: 'string' },
      wantError: /Must supply an 'array' schema type/,
    },
    {
      desc: 'throws error for array schema with non-object items',
      schema: { type: 'array', items: { type: 'string' } },
      wantError: /Must supply an 'array' schema type containing 'object' items/,
    },
  ];

  for (const et of errorTests) {
    it(et.desc, () => {
      assert.throws(() => {
        jsonlFormatter.handler(et.schema);
      }, et.wantError);
    });
  }
});

describe('jsonlFormat e2e', () => {
  it('keeps one parser per concurrent generation', async () => {
    const registry = new Registry();
    let nextParserId = 0;
    const parserCalls = new Map<number, number>();

    defineFormat(
      registry,
      { name: 'test-jsonl', ...jsonlFormatter.config },
      (schema) => {
        const parserId = nextParserId++;
        const handler = jsonlFormatter.handler(schema);
        return {
          ...handler,
          parseChunk: (chunk) => {
            parserCalls.set(parserId, (parserCalls.get(parserId) ?? 0) + 1);
            return handler.parseChunk!(chunk);
          },
        };
      }
    );

    const pm = defineProgrammableModel(registry);
    pm.handleResponse = async (request, streamingCallback) => {
      const prompt = request.messages
        .flatMap((message) => message.content)
        .map((part) => part.text ?? '')
        .join('');
      const id = prompt.includes('first') ? 1 : 2;

      await runAsync(() =>
        streamingCallback?.({ content: [{ text: '{id: ' }] })
      );
      await runAsync(() =>
        streamingCallback?.({ content: [{ text: `${id}}\n` }] })
      );
      return await runAsync(() => ({
        message: {
          role: 'model',
          content: [{ text: `{id: ${id}}\n` }],
        },
      }));
    };

    const collect = async (prompt: string) => {
      const { response, stream } = await generateStream(registry, {
        model: 'programmableModel',
        prompt,
        output: {
          format: 'test-jsonl',
          schema: z.array(z.object({ id: z.number() })),
        },
      });
      const outputs: unknown[] = [];
      for await (const chunk of stream) outputs.push(chunk.output);
      assert.deepStrictEqual((await response).output, [
        { id: prompt === 'first' ? 1 : 2 },
      ]);
      return outputs;
    };

    const [first, second] = await Promise.all([
      collect('first'),
      collect('second'),
    ]);

    assert.deepStrictEqual(first, [[], [{ id: 1 }]]);
    assert.deepStrictEqual(second, [[], [{ id: 2 }]]);
    assert.deepStrictEqual(
      [...parserCalls.values()].sort((a, b) => a - b),
      [2, 2]
    );
  });
});
