/**
 * Copyright 2025 Google LLC
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

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';
import { genkit, z } from 'genkit';
import type { GenerateRequest } from 'genkit/model';
import { toJsonSchema, validateSchema } from 'genkit/schema';
import OpenAI from 'openai';
import { openAICompatible } from '../src';
import {
  fromOpenAIChoice,
  needsRootWrapper,
  openAIModelRunner,
  toOpenAIRequestBody,
} from '../src/model';
import { openAI, type OpenAIPluginOptions } from '../src/openai';
import {
  WrapperStreamUnwrapper,
  hasObjectRoot,
  unwrapValue,
  wrapSchemaRoot,
} from '../src/schema-root';
import { FakeOpenAIServer } from './fake_openai_server';

const arraySchema = {
  type: 'array',
  items: {
    type: 'object',
    properties: { name: { type: 'string' } },
    required: ['name'],
    additionalProperties: false,
  },
};

function request(schema: unknown, format = 'json'): GenerateRequest {
  return {
    messages: [{ role: 'user', content: [{ text: 'hi' }] }],
    output: { format, schema },
  } as unknown as GenerateRequest;
}

describe('hasObjectRoot', () => {
  it.each([
    [{ type: 'object', properties: {} }, true],
    [{ properties: { a: { type: 'string' } } }, false],
    [{ type: 'array', items: {} }, false],
    [{ type: 'string', enum: ['a', 'b'] }, false],
    [{ type: 'integer' }, false],
    [{ anyOf: [{ type: 'object' }, { type: 'string' }] }, false],
  ])('%j -> %s', (schema, expected) => {
    expect(hasObjectRoot(schema)).toBe(expected);
  });
});

describe('wrapSchemaRoot', () => {
  it('wraps the schema as a single required property', () => {
    expect(wrapSchemaRoot(arraySchema)).toStrictEqual({
      type: 'object',
      properties: { value: arraySchema },
      required: ['value'],
      additionalProperties: false,
    });
  });

  it('hoists $schema, $defs and definitions to the wrapper root', () => {
    const schema = {
      $schema: 'http://json-schema.org/draft-07/schema#',
      type: 'array',
      items: { $ref: '#/$defs/Item' },
      $defs: { Item: { type: 'string' } },
      definitions: { Old: { type: 'number' } },
    };
    const wrapped = wrapSchemaRoot(schema) as any;
    expect(wrapped.$schema).toBe('http://json-schema.org/draft-07/schema#');
    expect(wrapped.$defs).toStrictEqual({ Item: { type: 'string' } });
    expect(wrapped.definitions).toStrictEqual({ Old: { type: 'number' } });
    expect(wrapped.properties.value).toStrictEqual({
      type: 'array',
      items: { $ref: '#/$defs/Item' },
    });
  });
});

describe('wrapSchemaRoot local references', () => {
  /** Validates `value` against the wrapped schema, as Ajv resolves refs. */
  function validWrapped(schema: Record<string, unknown>, value: unknown) {
    const result = validateSchema(
      { value },
      { jsonSchema: wrapSchemaRoot(schema) }
    );
    return result.valid;
  }

  it('re-points refs into a reused Zod schema', () => {
    const Inner = z.object({ n: z.number() });
    const schema = toJsonSchema({
      schema: z.array(z.object({ first: Inner, second: Inner })),
    }) as Record<string, unknown>;
    expect(JSON.stringify(schema)).toContain('"$ref":"#/items/');

    const wrapped = wrapSchemaRoot(schema) as any;
    expect(JSON.stringify(wrapped)).toContain(
      '"$ref":"#/properties/value/items/'
    );
    expect(validWrapped(schema, [{ first: { n: 1 }, second: { n: 2 } }])).toBe(
      true
    );
    expect(
      validWrapped(schema, [{ first: { n: 1 }, second: { n: 'x' } }])
    ).toBe(false);
  });

  it('re-points recursive refs to the root', () => {
    type Tree = { name: string; kids: Tree[] };
    const Tree: z.ZodType<Tree> = z.lazy(() =>
      z.object({ name: z.string(), kids: z.array(Tree) })
    );
    const schema = toJsonSchema({ schema: z.array(Tree) }) as Record<
      string,
      unknown
    >;
    const tree = [{ name: 'a', kids: [{ name: 'b', kids: [] }] }];
    expect(validWrapped(schema, tree)).toBe(true);
    expect(validWrapped(schema, [{ name: 'a', kids: [{ name: 1 }] }])).toBe(
      false
    );
  });

  it('rewrites "#" and leaves $defs, definitions and data alone', () => {
    const wrapped = wrapSchemaRoot({
      type: 'array',
      items: {
        anyOf: [
          { $ref: '#' },
          { $ref: '#/$defs/A' },
          { $ref: '#/definitions/B' },
          { $ref: '#/items/anyOf/1' },
          { $ref: 'https://example.com/schema' },
        ],
      },
      default: [{ $ref: '#/items' }],
      $defs: { A: { type: 'string', items: { $ref: '#/items' } } },
    }) as any;
    expect(wrapped.properties.value.items.anyOf).toStrictEqual([
      { $ref: '#/properties/value' },
      { $ref: '#/$defs/A' },
      { $ref: '#/definitions/B' },
      { $ref: '#/properties/value/items/anyOf/1' },
      { $ref: 'https://example.com/schema' },
    ]);
    expect(wrapped.properties.value.default).toStrictEqual([
      { $ref: '#/items' },
    ]);
    expect(wrapped.$defs.A.items).toStrictEqual({
      $ref: '#/properties/value/items',
    });
  });
});

describe('unwrapValue', () => {
  it('returns the wrapped value', () => {
    expect(unwrapValue({ value: [1, 2] })).toStrictEqual([1, 2]);
    expect(unwrapValue({ value: 'a' })).toBe('a');
    expect(unwrapValue({ value: null })).toBeNull();
  });
  it('passes through anything that is not a wrapper', () => {
    expect(unwrapValue([1, 2])).toStrictEqual([1, 2]);
    expect(unwrapValue({ other: 1 })).toStrictEqual({ other: 1 });
    expect(unwrapValue('x')).toBe('x');
  });
});

describe('needsRootWrapper', () => {
  const object = { objectSchemaRoot: true };
  it('is true only for an object-only host, json format, non-object root', () => {
    expect(needsRootWrapper(request(arraySchema), object)).toBe(true);
    expect(
      needsRootWrapper(request(arraySchema), { objectSchemaRoot: false })
    ).toBe(false);
    expect(needsRootWrapper(request(arraySchema), undefined)).toBe(false);
    expect(needsRootWrapper(request({ type: 'object' }), object)).toBe(false);
    expect(needsRootWrapper(request(arraySchema, 'array'), object)).toBe(false);
    expect(needsRootWrapper(request(undefined), object)).toBe(false);
  });
});

describe('toOpenAIRequestBody with non-object roots', () => {
  it('wraps the schema for an object-only host', () => {
    const body = toOpenAIRequestBody(
      'gpt-4o',
      request(arraySchema),
      undefined,
      {
        objectSchemaRoot: true,
      }
    ) as any;
    expect(body.response_format).toStrictEqual({
      type: 'json_schema',
      json_schema: { name: 'output', schema: wrapSchemaRoot(arraySchema) },
    });
  });

  it('keeps strict when configured', () => {
    const req = {
      ...request(arraySchema),
      config: { strict: true },
    } as GenerateRequest;
    const body = toOpenAIRequestBody('gpt-5', req, undefined, {
      objectSchemaRoot: true,
    }) as any;
    expect(body.response_format.json_schema.strict).toBe(true);
    expect(body.response_format.json_schema.schema.required).toStrictEqual([
      'value',
    ]);
  });

  it('sends the schema unchanged by default and for object roots', () => {
    const unchanged = toOpenAIRequestBody(
      'gpt-4o',
      request(arraySchema)
    ) as any;
    expect(unchanged.response_format.json_schema.schema).toStrictEqual(
      arraySchema
    );
    const obj = { type: 'object', properties: { a: { type: 'string' } } };
    const body = toOpenAIRequestBody('gpt-4o', request(obj), undefined, {
      objectSchemaRoot: true,
    }) as any;
    expect(body.response_format.json_schema.schema).toStrictEqual(obj);
  });
});

describe('fromOpenAIChoice with unwrapRoot', () => {
  const choice = (content: string) =>
    ({
      index: 0,
      finish_reason: 'stop',
      message: { role: 'assistant', content },
    }) as any;

  it('unwraps the value into data', () => {
    const out = fromOpenAIChoice(
      choice('{"value":[{"name":"a"}]}'),
      true,
      true
    );
    expect(out.message?.content).toStrictEqual([{ data: [{ name: 'a' }] }]);
  });

  it('leaves data untouched when unwrapRoot is off', () => {
    const out = fromOpenAIChoice(choice('{"value":1}'), true, false);
    expect(out.message?.content).toStrictEqual([{ data: { value: 1 } }]);
  });
});

describe('WrapperStreamUnwrapper', () => {
  function run(deltas: string[]): string {
    const u = new WrapperStreamUnwrapper();
    return deltas.map((d) => u.push(d)).join('') + u.flush();
  }

  const full = '{"value":[{"name":"a"},{"name":"b"}]}';

  it('strips the wrapper however the stream is split', () => {
    expect(run([full])).toBe('[{"name":"a"},{"name":"b"}]');
    expect(run(full.split(''))).toBe('[{"name":"a"},{"name":"b"}]');
    expect(
      run(['{"', 'value', '":[', '{"name":"a"}', ',{"name":"b"}', ']}'])
    ).toBe('[{"name":"a"},{"name":"b"}]');
  });

  it('handles whitespace and pretty-printed output', () => {
    expect(run(['{\n  "value": ', '[1, 2]\n', '}\n'])).toBe('[1, 2]');
  });

  it('keeps braces that belong to the value', () => {
    expect(run(['{"value":{"a":{"b":1}}}'])).toBe('{"a":{"b":1}}');
    expect(run(['{"value":[{"a":1}', ']}'])).toBe('[{"a":1}]');
  });

  it('unwraps scalars', () => {
    expect(run(['{"value":"he', 'llo"}'])).toBe('"hello"');
  });

  it('never emits the closing brace early', () => {
    const u = new WrapperStreamUnwrapper();
    const emitted = ['{"value":[1,2', ']}'].map((d) => u.push(d)).join('');
    expect(emitted).toBe('[1,2]');
    expect(u.flush()).toBe('');
  });

  it('gives the same result however the opening is split', () => {
    const spaced = '{' + ' '.repeat(42) + '"value":1}';
    expect(run([spaced])).toBe('1');
    expect(run(spaced.split(''))).toBe('1');
  });

  it('passes through output that is not a wrapper', () => {
    expect(run(['[1,', '2]'])).toBe('[1,2]');
    expect(run('{"valuable":1}'.split(''))).toBe('{"valuable":1}');
    expect(run('{"other":1}'.split(''))).toBe('{"other":1}');
    expect(
      run(['{"other":1,"more":"text that is long enough to decide"}'])
    ).toBe('{"other":1,"more":"text that is long enough to decide"}');
  });
});

describe('openAIModelRunner with a wrapped root', () => {
  let server: FakeOpenAIServer;
  beforeEach(async () => {
    server = new FakeOpenAIServer();
    await server.start();
  });
  afterEach(() => server.stop());

  const arrayRequest = request(arraySchema);
  const runner = () =>
    openAIModelRunner(
      'gpt-4o',
      new OpenAI({ apiKey: 'k', baseURL: server.baseUrl }),
      undefined,
      {
        name: 'openai',
        objectSchemaRoot: true,
      }
    );

  it('sends a wrapped schema and returns the unwrapped array', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            message: { role: 'assistant', content: '{"value":[{"name":"a"}]}' },
            finish_reason: 'stop',
          },
        ],
      },
    });
    const res = await runner()(arrayRequest);
    expect(
      server.requests[0].body.response_format.json_schema.schema
    ).toStrictEqual(wrapSchemaRoot(arraySchema));
    expect(res.message?.content).toStrictEqual([{ data: [{ name: 'a' }] }]);
  });

  it('streams unwrapped text chunks and an unwrapped final message', async () => {
    const pieces = [
      '{"',
      'value',
      '":[',
      '{"name":"a"}',
      ',{"name":"b"}',
      ']}',
    ];
    server.setNextResponse({
      stream: true,
      chunks: pieces
        .map((content) => ({
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content },
              finish_reason: null,
            },
          ],
        }))
        .concat([
          { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] } as any,
        ]),
    });
    const chunks: string[] = [];
    const res = await runner()(arrayRequest, {
      streamingRequested: true,
      sendChunk: (c) => {
        chunks.push(c.content.map((p) => p.text ?? '').join(''));
      },
    });
    expect(chunks.join('')).toBe('[{"name":"a"},{"name":"b"}]');
    expect(chunks.length).toBeGreaterThan(1);
    expect(res.message?.content).toStrictEqual([
      { data: [{ name: 'a' }, { name: 'b' }] },
    ]);
  });

  describe('with tools', () => {
    const toolRequest = {
      ...arrayRequest,
      tools: [
        {
          name: 'lookup',
          description: 'Look up people',
          inputSchema: { type: 'object', properties: {} },
        },
      ],
    } as GenerateRequest;

    it('returns tool requests and keeps a preamble as text', async () => {
      server.setNextResponse({
        body: {
          choices: [
            {
              message: {
                role: 'assistant',
                content: 'Let me check.',
                tool_calls: [
                  {
                    id: 'call_1',
                    type: 'function',
                    function: { name: 'lookup', arguments: '{}' },
                  },
                ],
              },
              finish_reason: 'tool_calls',
            },
          ],
        },
      });
      const res = await runner()(toolRequest);
      const body = server.requests[0].body;
      expect(body.tools).toHaveLength(1);
      expect(body.response_format.json_schema.schema).toStrictEqual(
        wrapSchemaRoot(arraySchema)
      );
      expect(res.finishReason).toBe('stop');
      expect(res.message?.content).toStrictEqual([
        { text: 'Let me check.' },
        {
          toolRequest: { ref: 'call_1', name: 'lookup', input: {} },
        },
      ]);
    });

    it('unwraps the final answer after the tool turn', async () => {
      server.setNextResponse({
        body: {
          choices: [
            {
              message: {
                role: 'assistant',
                content: '{"value":[{"name":"a"}]}',
              },
              finish_reason: 'stop',
            },
          ],
        },
      });
      const res = await runner()({
        ...toolRequest,
        messages: [
          ...toolRequest.messages,
          {
            role: 'model',
            content: [
              { toolRequest: { ref: 'call_1', name: 'lookup', input: {} } },
            ],
          },
          {
            role: 'tool',
            content: [
              {
                toolResponse: { ref: 'call_1', name: 'lookup', output: ['a'] },
              },
            ],
          },
        ],
      } as GenerateRequest);
      expect(res.message?.content).toStrictEqual([{ data: [{ name: 'a' }] }]);
    });

    it('streams a preamble and tool call through unchanged', async () => {
      server.setNextResponse({
        stream: true,
        chunks: [
          {
            choices: [
              { index: 0, delta: { role: 'assistant', content: 'Let me ' } },
            ],
          },
          { choices: [{ index: 0, delta: { content: 'check. ' } }] },
          {
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [
                    {
                      index: 0,
                      id: 'call_1',
                      type: 'function',
                      function: { name: 'lookup', arguments: '' },
                    },
                  ],
                },
              },
            ],
          },
          {
            choices: [
              {
                index: 0,
                delta: {
                  tool_calls: [{ index: 0, function: { arguments: '{}' } }],
                },
              },
            ],
          },
          { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
        ],
      });
      let text = '';
      let sawToolRequest = false;
      const res = await runner()(toolRequest, {
        streamingRequested: true,
        sendChunk: (c) => {
          for (const p of c.content) {
            if (p.text) text += p.text;
            if (p.toolRequest) sawToolRequest = true;
          }
        },
      });
      expect(text).toBe('Let me check. ');
      expect(sawToolRequest).toBe(true);
      expect(res.message?.content).toContainEqual({
        toolRequest: { ref: 'call_1', name: 'lookup', input: {} },
      });
    });
  });
});

describe('plugins and the object-root setting', () => {
  let server: FakeOpenAIServer;
  beforeEach(async () => {
    server = new FakeOpenAIServer();
    await server.start();
  });
  afterEach(() => server.stop());

  function reply(content: string) {
    server.setNextResponse({
      body: {
        choices: [
          { message: { role: 'assistant', content }, finish_reason: 'stop' },
        ],
      },
    });
  }

  it('openAI() wraps a non-object root and unwraps the reply', async () => {
    reply('{"value":["a","b"]}');
    const ai = genkit({
      // baseURL is not a typed openAI() option, but it reaches the client.
      plugins: [
        openAI({ apiKey: 'k', baseURL: server.baseUrl } as OpenAIPluginOptions),
      ],
    });
    const { output } = await ai.generate({
      model: 'openai/gpt-4o',
      prompt: 'hi',
      output: { schema: z.array(z.string()) },
    });
    const sent = server.requests[0].body.response_format.json_schema.schema;
    expect(sent.type).toBe('object');
    expect(sent.properties.value.type).toBe('array');
    expect(output).toStrictEqual(['a', 'b']);
  });

  it('openAICompatible() sends the schema unchanged', async () => {
    reply('["a","b"]');
    const ai = genkit({
      plugins: [
        openAICompatible({
          name: 'host',
          apiKey: 'k',
          baseURL: server.baseUrl,
        }),
      ],
    });
    const { output } = await ai.generate({
      model: 'host/some-model',
      prompt: 'hi',
      output: { schema: z.array(z.string()), constrained: true },
    });
    const sent = server.requests[0].body.response_format?.json_schema?.schema;
    expect(sent?.properties?.value).toBeUndefined();
    expect(output).toStrictEqual(['a', 'b']);
  });
});
