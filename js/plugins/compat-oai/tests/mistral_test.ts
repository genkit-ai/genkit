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

import { afterEach, beforeEach, describe, expect, it } from '@jest/globals';
import { genkit, z, type Genkit, type MessageData } from 'genkit';
import { mistral } from '../src/mistral';
import { MistralChatCompletionConfigSchema } from '../src/mistral/mistral';
import { FakeOpenAIServer } from './fake_openai_server';

const API_KEY = 'mistral-test-key';

/** Capabilities given to models outside the known list. */
const UNKNOWN_MODEL_SUPPORTS = {
  multiturn: true,
  tools: false,
  media: false,
  systemRole: true,
  output: ['text'],
};

/**
 * Wire keys of Mistral's `ChatCompletionRequest`, and of the message types it
 * takes, copied from the `$Outbound` types in `@mistralai/mistralai` 3.5.0
 * (`models/components/chatcompletionrequest.ts` and friends). They match the
 * schema's properties in https://docs.mistral.ai/openapi.yaml, which sets
 * `additionalProperties: false`, so keys that are not listed are rejected.
 */
const MISTRAL_CHAT_REQUEST_KEYS = [
  'model',
  'temperature',
  'top_p',
  'max_tokens',
  'stream',
  'stop',
  'random_seed',
  'metadata',
  'messages',
  'response_format',
  'tools',
  'tool_choice',
  'presence_penalty',
  'frequency_penalty',
  'min_tokens',
  'repetition_penalty',
  'top_k',
  'logprobs',
  'top_logprobs',
  'prompt_logprobs',
  'top_prompt_logprobs',
  'n',
  'prediction',
  'parallel_tool_calls',
  'reasoning_effort',
  'prompt_mode',
  'guardrails',
  'prompt_cache_key',
  'service_tier',
  'safe_prompt',
];
const MISTRAL_MESSAGE_KEYS: Record<string, string[]> = {
  system: ['role', 'content'],
  user: ['role', 'content'],
  assistant: ['role', 'content', 'tool_calls', 'prefix'],
  tool: ['role', 'content', 'tool_call_id', 'name'],
};
const MISTRAL_THINK_CHUNK_KEYS = ['type', 'thinking', 'signature', 'closed'];
/**
 * Keys of the content chunks the plugin sends in `user` messages, by chunk
 * type. Mistral's `file` chunk only takes an uploaded `file_id`, so inline
 * files are refused before the request goes out.
 */
const MISTRAL_USER_CHUNK_KEYS: Record<string, string[]> = {
  text: ['type', 'text'],
  image_url: ['type', 'image_url'],
};
const MISTRAL_IMAGE_URL_KEYS = ['url', 'detail'];

/** Wire keys of Mistral's `EmbeddingRequest`, from the same SDK. */
const MISTRAL_EMBEDDING_REQUEST_KEYS = [
  'model',
  'metadata',
  'input',
  'output_dimension',
  'output_dtype',
  'encoding_format',
];

function expectKeysIn(value: object, allowed: string[]) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  expect(unknown).toEqual([]);
}

/** Asserts a chat request body only uses keys Mistral accepts. */
function expectMistralChatRequest(body: Record<string, any>) {
  expectKeysIn(body, MISTRAL_CHAT_REQUEST_KEYS);
  for (const message of body.messages) {
    expectKeysIn(message, MISTRAL_MESSAGE_KEYS[message.role] ?? []);
    if (Array.isArray(message.content)) {
      for (const chunk of message.content) {
        if (chunk.type === 'thinking') {
          expectKeysIn(chunk, MISTRAL_THINK_CHUNK_KEYS);
        }
        if (message.role === 'user') {
          expect(Object.keys(MISTRAL_USER_CHUNK_KEYS)).toContain(chunk.type);
          expectKeysIn(chunk, MISTRAL_USER_CHUNK_KEYS[chunk.type] ?? []);
          if (
            chunk.type === 'image_url' &&
            typeof chunk.image_url === 'object'
          ) {
            expectKeysIn(chunk.image_url, MISTRAL_IMAGE_URL_KEYS);
          }
        }
      }
    }
  }
}

/** Asserts an embeddings request body only uses keys Mistral accepts. */
function expectMistralEmbeddingRequest(body: Record<string, any>) {
  expectKeysIn(body, MISTRAL_EMBEDDING_REQUEST_KEYS);
}

/** Checks every captured request body against Mistral's request keys. */
function expectMistralRequests(server: FakeOpenAIServer) {
  for (const { body } of server.requests) {
    if (body.messages) {
      expectMistralChatRequest(body);
    } else if (body.input) {
      expectMistralEmbeddingRequest(body);
    }
  }
}

/** Points the plugin at the fake server through `baseURL`. */
function fakeServerOptions(server: FakeOpenAIServer) {
  return { apiKey: API_KEY, baseURL: server.baseUrl };
}

describe('mistral plugin', () => {
  let server: FakeOpenAIServer;
  let ai: Genkit;

  beforeEach(async () => {
    server = new FakeOpenAIServer(API_KEY);
    await server.start();
    ai = genkit({ plugins: [mistral(fakeServerOptions(server))] });
  });

  afterEach(() => {
    try {
      // Every chat and embeddings request a test made must fit Mistral's
      // schema.
      expectMistralRequests(server);
    } finally {
      server.stop();
    }
  });

  it('requires an API key', () => {
    const saved = process.env.MISTRAL_API_KEY;
    delete process.env.MISTRAL_API_KEY;
    try {
      expect(() => mistral()).toThrow(/MISTRAL_API_KEY/);
    } finally {
      if (saved !== undefined) process.env.MISTRAL_API_KEY = saved;
    }
  });

  it('accepts apiKey: false without MISTRAL_API_KEY', () => {
    const saved = process.env.MISTRAL_API_KEY;
    delete process.env.MISTRAL_API_KEY;
    try {
      expect(mistral({ apiKey: false }).name).toBe('mistral');
    } finally {
      if (saved !== undefined) process.env.MISTRAL_API_KEY = saved;
    }
  });

  it('reads the API key from MISTRAL_API_KEY', () => {
    const saved = process.env.MISTRAL_API_KEY;
    process.env.MISTRAL_API_KEY = 'from-env';
    try {
      expect(mistral().name).toBe('mistral');
    } finally {
      if (saved === undefined) delete process.env.MISTRAL_API_KEY;
      else process.env.MISTRAL_API_KEY = saved;
    }
  });

  it('sends Mistral field names for generation config', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'hello' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      },
    });

    const response = await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'hi',
      config: {
        temperature: 0.3,
        topP: 0.9,
        maxOutputTokens: 128,
        stopSequences: ['END'],
        seed: 42,
        reasoningEffort: 'high',
        presencePenalty: 0.5,
        frequencyPenalty: 0.25,
      },
    });

    expect(response.text).toBe('hello');
    expect(response.usage.inputTokens).toBe(3);
    expect(server.requests).toHaveLength(1);
    expect(server.requests[0].headers['authorization']).toBe(
      `Bearer ${API_KEY}`
    );
    const body = server.requests[0].body;
    expect(body).toEqual({
      model: 'mistral-small-latest',
      messages: [{ role: 'user', content: 'hi' }],
      temperature: 0.3,
      top_p: 0.9,
      max_tokens: 128,
      stop: ['END'],
      random_seed: 42,
      reasoning_effort: 'high',
      presence_penalty: 0.5,
      frequency_penalty: 0.25,
    });
    expectMistralChatRequest(body);
  });

  it('refuses topK before the request goes out', async () => {
    // Mistral's models answer top_k with "top_k sampling is not enabled for
    // this model".
    await expect(
      ai.generate({
        model: mistral.model('mistral-small-latest'),
        prompt: 'hi',
        // `topK` isn't in the config schema, but untyped config can carry it.
        config: { topK: 40 } as any,
      })
    ).rejects.toThrow(/topK/);
    expect(server.requests).toHaveLength(0);
  });

  it('sends seed and temperature when they are 0', async () => {
    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'hi',
      config: { seed: 0, temperature: 0 },
    });

    const body = server.requests[0].body;
    expect(body.random_seed).toBe(0);
    expect(body.temperature).toBe(0);
  });

  it('accepts only Mistral reasoningEffort values in the config schema', () => {
    const schema = MistralChatCompletionConfigSchema;
    for (const value of [
      'none',
      'minimal',
      'low',
      'medium',
      'high',
      'xhigh',
      'max',
    ]) {
      expect(schema.safeParse({ reasoningEffort: value }).success).toBe(true);
    }
    expect(schema.safeParse({ reasoningEffort: 'hgih' }).success).toBe(false);
  });

  it('leaves topK out of the config schema', () => {
    expect(MistralChatCompletionConfigSchema.shape).not.toHaveProperty('topK');
  });

  it('sends images as image_url chunks', async () => {
    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: [
        { text: 'Describe this.' },
        {
          media: {
            url: 'data:image/png;base64,iVBORw0KGgo=',
            contentType: 'image/png',
          },
        },
      ],
    });

    const body = server.requests[0].body;
    expect(body.messages[0].content).toEqual([
      { type: 'text', text: 'Describe this.' },
      {
        type: 'image_url',
        image_url: {
          url: 'data:image/png;base64,iVBORw0KGgo=',
          detail: 'auto',
        },
      },
    ]);
  });

  it('refuses non-image media before the request goes out', async () => {
    await expect(
      ai.generate({
        model: mistral.model('mistral-small-latest'),
        prompt: [
          { text: 'Summarize this.' },
          {
            media: {
              url: 'data:application/pdf;base64,JVBERi0xLjQK',
              contentType: 'application/pdf',
            },
          },
        ],
      })
    ).rejects.toMatchObject({
      status: 'INVALID_ARGUMENT',
      message: expect.stringContaining('application/pdf'),
    });
    expect(server.requests).toHaveLength(0);
  });

  describe('unknown models', () => {
    /** Mistral's answer to an unknown or retired model, as sent live. */
    const INVALID_MODEL = {
      statusCode: 400,
      body: {
        object: 'error',
        message: 'Invalid model: no-such-model-xyz',
        type: 'invalid_model',
        param: null,
        code: '1500',
        raw_status_code: 400,
      },
    };

    it('reports an unknown model as NOT_FOUND', async () => {
      server.setNextResponse(INVALID_MODEL);
      await expect(
        ai.generate({ model: 'mistral/no-such-model-xyz', prompt: 'hi' })
      ).rejects.toMatchObject({
        status: 'NOT_FOUND',
        message: expect.stringContaining('Invalid model: no-such-model-xyz'),
      });
    });

    it('reports an unknown model as NOT_FOUND when streaming', async () => {
      server.setNextResponse(INVALID_MODEL);
      const { stream, response } = ai.generateStream({
        model: 'mistral/no-such-model-xyz',
        prompt: 'hi',
      });
      await expect(
        (async () => {
          for await (const _ of stream) {
          }
          return response;
        })()
      ).rejects.toMatchObject({ status: 'NOT_FOUND' });
      await expect(response).rejects.toMatchObject({ status: 'NOT_FOUND' });
    });

    it('reports an unknown embedder as NOT_FOUND', async () => {
      server.setNextResponse(INVALID_MODEL);
      await expect(
        ai.embed({ embedder: 'mistral/no-such-model-xyz', content: 'hi' })
      ).rejects.toMatchObject({ status: 'NOT_FOUND' });
    });

    it('keeps other 400 errors as INVALID_ARGUMENT', async () => {
      const badRequest = {
        statusCode: 400,
        body: {
          object: 'error',
          message: 'Something else is wrong.',
          type: 'invalid_request_error',
          param: null,
          code: '3000',
        },
      };
      server.setNextResponse(badRequest);
      await expect(
        ai.generate({ model: 'mistral/mistral-small-latest', prompt: 'hi' })
      ).rejects.toMatchObject({
        status: 'INVALID_ARGUMENT',
        message: expect.stringContaining('Something else is wrong.'),
      });
      server.setNextResponse(badRequest);
      await expect(
        ai.embed({ embedder: 'mistral/mistral-embed', content: 'hi' })
      ).rejects.toMatchObject({ status: 'INVALID_ARGUMENT' });
    });
  });

  it('does not pass unknown config through to the request', async () => {
    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'hi',
      config: { user: 'someone', max_completion_tokens: 5 } as any,
    });

    const body = server.requests[0].body;
    expect(body).not.toHaveProperty('user');
    expect(body).not.toHaveProperty('max_completion_tokens');
    expectMistralChatRequest(body);
  });

  it('does not send stream_options when streaming', async () => {
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          id: '1',
          model: 'mistral-small-latest',
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: 'Hel' },
              finish_reason: null,
            },
          ],
        },
        {
          id: '1',
          model: 'mistral-small-latest',
          choices: [
            { index: 0, delta: { content: 'lo' }, finish_reason: 'stop' },
          ],
          // Assumes Mistral sends usage on the last chunk without
          // stream_options. Not confirmed against the live API.
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        },
      ],
    });

    const { stream, response } = ai.generateStream({
      model: mistral.model('mistral-small-latest'),
      prompt: 'hi',
    });
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }
    const final = await response;

    const body = server.requests[0].body;
    expect(body.stream).toBe(true);
    expectMistralChatRequest(body);
    expect(chunks.join('')).toBe('Hello');
    expect(final.text).toBe('Hello');
    expect(final.usage.outputTokens).toBe(2);
  });

  it('reads thinking chunks as reasoning', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: [
                {
                  type: 'thinking',
                  thinking: [{ type: 'text', text: 'Two plus two is four.' }],
                },
                { type: 'text', text: '4' },
              ],
            },
            finish_reason: 'stop',
          },
        ],
      },
    });

    const response = await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
    });

    expect(response.message?.content).toEqual([
      {
        reasoning: 'Two plus two is four.',
        metadata: { reasoningSource: 'mistral' },
      },
      { text: '4' },
    ]);
    expect(response.reasoning).toBe('Two plus two is four.');
    expect(response.text).toBe('4');
  });

  it('reads streamed thinking chunks as reasoning', async () => {
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          id: '1',
          choices: [
            {
              index: 0,
              delta: {
                role: 'assistant',
                content: [
                  {
                    type: 'thinking',
                    thinking: [{ type: 'text', text: 'Two plus ' }],
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: '1',
          choices: [
            {
              index: 0,
              delta: {
                content: [
                  {
                    type: 'thinking',
                    thinking: [{ type: 'text', text: 'two is four.' }],
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: '1',
          choices: [
            { index: 0, delta: { content: '4' }, finish_reason: 'stop' },
          ],
        },
      ],
    });

    const { stream, response } = ai.generateStream({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
    });
    const reasoning: string[] = [];
    const text: string[] = [];
    for await (const chunk of stream) {
      reasoning.push(chunk.reasoning);
      text.push(chunk.text);
    }
    const final = await response;

    expect(reasoning.join('')).toBe('Two plus two is four.');
    expect(text.join('')).toBe('4');
    expect(final.message?.content).toEqual([
      {
        reasoning: 'Two plus two is four.',
        metadata: { reasoningSource: 'mistral' },
      },
      { text: '4' },
    ]);
  });

  it('parses JSON output from text chunks', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: [
                {
                  type: 'thinking',
                  thinking: [{ type: 'text', text: 'Answer is 4.' }],
                },
                { type: 'text', text: '{"answer":' },
                { type: 'text', text: '4}' },
              ],
            },
            finish_reason: 'stop',
          },
        ],
      },
    });

    const response = await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
      output: {
        format: 'json',
        schema: z.object({ answer: z.number() }),
      },
    });

    expect(response.output).toEqual({ answer: 4 });
    expect(response.reasoning).toBe('Answer is 4.');
    expect(server.requests[0].body.response_format).toMatchObject({
      type: 'json_schema',
      json_schema: {
        name: 'output',
        schema: {
          type: 'object',
          properties: { answer: { type: 'number' } },
        },
      },
    });
  });

  it('sends earlier reasoning back as a thinking chunk', async () => {
    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      messages: [
        { role: 'user', content: [{ text: 'What is 2 + 2?' }] },
        {
          role: 'model',
          content: [
            {
              reasoning: 'Two plus two is four.',
              metadata: { reasoningSource: 'mistral' },
            },
            { text: '4' },
          ],
        },
        {
          role: 'model',
          content: [
            {
              reasoning: 'I should call a tool.',
              metadata: { reasoningSource: 'mistral' },
            },
            { toolRequest: { name: 'add', ref: 'call_1', input: { a: 2 } } },
          ],
        },
      ],
      prompt: 'And 3 + 3?',
    });

    const body = server.requests[0].body;
    expectMistralChatRequest(body);
    expect(body.messages[1]).toEqual({
      role: 'assistant',
      content: [
        {
          type: 'thinking',
          thinking: [{ type: 'text', text: 'Two plus two is four.' }],
        },
        { type: 'text', text: '4' },
      ],
    });
    expect(body.messages[2]).toEqual({
      role: 'assistant',
      content: [
        {
          type: 'thinking',
          thinking: [{ type: 'text', text: 'I should call a tool.' }],
        },
      ],
      tool_calls: [
        {
          id: 'call_1',
          type: 'function',
          function: { name: 'add', arguments: '{"a":2}' },
        },
      ],
    });
    for (const message of body.messages) {
      expect(message).not.toHaveProperty('reasoning_content');
    }
  });

  it('sends back only reasoning marked as Mistral', async () => {
    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      messages: [
        { role: 'user', content: [{ text: 'Bread for two?' }] },
        // A Gemini or Claude turn: its reasoning carries a signature under the
        // same key, but isn't Mistral's.
        {
          role: 'model',
          content: [
            {
              reasoning: 'Gemini thinks.',
              metadata: { thoughtSignature: 'gemini-sig' },
            },
            { text: 'Two loaves.' },
          ],
        },
        { role: 'user', content: [{ text: 'And three?' }] },
        {
          role: 'model',
          content: [{ reasoning: 'Unsigned, unmarked.' }, { text: 'Three.' }],
        },
        { role: 'user', content: [{ text: 'And four?' }] },
        {
          role: 'model',
          content: [
            {
              reasoning: 'Mistral thinks.',
              metadata: { reasoningSource: 'mistral', thoughtSignature: 'sig' },
            },
            { text: 'Four.' },
          ],
        },
      ],
      prompt: 'And five?',
    });

    const body = server.requests[0].body;
    const assistant = body.messages.filter(
      (m: { role: string }) => m.role === 'assistant'
    );
    expect(assistant).toEqual([
      { role: 'assistant', content: 'Two loaves.' },
      { role: 'assistant', content: 'Three.' },
      {
        role: 'assistant',
        content: [
          {
            type: 'thinking',
            thinking: [{ type: 'text', text: 'Mistral thinks.' }],
            signature: 'sig',
          },
          { type: 'text', text: 'Four.' },
        ],
      },
    ]);
  });

  it('sends back Mistral reasoning after a JSON round trip of the history', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: [
                {
                  type: 'thinking',
                  thinking: [{ type: 'text', text: 'Two plus two.' }],
                  signature: 'sig-123',
                },
                { type: 'text', text: '4' },
              ],
            },
            finish_reason: 'stop',
          },
        ],
      },
    });
    const first = await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
    });
    const saved = JSON.parse(JSON.stringify(first.messages));
    expect(saved[1].content[0].metadata).toEqual({
      reasoningSource: 'mistral',
      thoughtSignature: 'sig-123',
    });

    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      messages: saved,
      prompt: 'And 3 + 3?',
    });

    expect(server.requests[1].body.messages[1].content[0]).toEqual({
      type: 'thinking',
      thinking: [{ type: 'text', text: 'Two plus two.' }],
      signature: 'sig-123',
    });
  });

  it('streams JSON output alongside thinking chunks', async () => {
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          id: '1',
          choices: [
            {
              index: 0,
              delta: {
                role: 'assistant',
                content: [
                  {
                    type: 'thinking',
                    thinking: [{ type: 'text', text: 'Answer is 4.' }],
                  },
                ],
              },
              finish_reason: null,
            },
          ],
        },
        {
          id: '1',
          choices: [
            {
              index: 0,
              delta: { content: [{ type: 'text', text: '{"answer":' }] },
              finish_reason: null,
            },
          ],
        },
        {
          id: '1',
          choices: [
            { index: 0, delta: { content: '4}' }, finish_reason: 'stop' },
          ],
        },
      ],
    });

    const { stream, response } = ai.generateStream({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
      output: { format: 'json', schema: z.object({ answer: z.number() }) },
    });
    const reasoning: string[] = [];
    const text: string[] = [];
    for await (const chunk of stream) {
      reasoning.push(chunk.reasoning);
      text.push(chunk.text);
    }
    const final = await response;

    expect(reasoning.join('')).toBe('Answer is 4.');
    expect(text.join('')).toBe('{"answer":4}');
    expect(final.output).toEqual({ answer: 4 });
    expect(final.reasoning).toBe('Answer is 4.');
    const body = server.requests[0].body;
    expect(body.stream).toBe(true);
    expect(body.response_format.type).toBe('json_schema');
    expectMistralChatRequest(body);
  });

  it('round-trips a tool call', async () => {
    const add = ai.defineTool(
      {
        name: 'add',
        description: 'Adds two numbers',
        inputSchema: z.object({ a: z.number(), b: z.number() }),
        outputSchema: z.number(),
      },
      async ({ a, b }) => a + b
    );
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: '',
              tool_calls: [
                {
                  id: 'call_1',
                  type: 'function',
                  function: { name: 'add', arguments: '{"a":2,"b":2}' },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
      },
    });
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: '2 + 2 is 4.' },
            finish_reason: 'stop',
          },
        ],
      },
    });

    const response = await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
      tools: [add],
    });

    expect(response.text).toBe('2 + 2 is 4.');
    expect(server.requests).toHaveLength(2);
    const [first, second] = server.requests.map((r) => r.body);
    expect(first.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'add',
          parameters: expect.objectContaining({ type: 'object' }),
        },
      },
    ]);
    expect(second.messages.slice(1)).toEqual([
      {
        role: 'assistant',
        tool_calls: [
          {
            id: 'call_1',
            type: 'function',
            function: { name: 'add', arguments: '{"a":2,"b":2}' },
          },
        ],
      },
      { role: 'tool', tool_call_id: 'call_1', content: '4' },
    ]);
    expectMistralChatRequest(first);
    expectMistralChatRequest(second);
  });

  it('keeps the thinking signature in reasoning metadata', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: [
                {
                  type: 'thinking',
                  thinking: [{ type: 'text', text: 'Two plus two.' }],
                  signature: 'sig-123',
                },
                { type: 'text', text: '4' },
              ],
            },
            finish_reason: 'stop',
          },
        ],
      },
    });

    const response = await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
    });

    expect(response.message?.content).toEqual([
      {
        reasoning: 'Two plus two.',
        metadata: { reasoningSource: 'mistral', thoughtSignature: 'sig-123' },
      },
      { text: '4' },
    ]);
  });

  it('sends the thinking signature back on later turns', async () => {
    server.setNextResponse({
      body: {
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: [
                {
                  type: 'thinking',
                  thinking: [{ type: 'text', text: 'Two plus two.' }],
                  signature: 'sig-123',
                },
                { type: 'text', text: '4' },
              ],
            },
            finish_reason: 'stop',
          },
        ],
      },
    });
    const first = await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'What is 2 + 2?',
    });

    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      messages: first.messages,
      prompt: 'And 3 + 3?',
    });

    const body = server.requests[1].body;
    expect(body.messages[1]).toEqual({
      role: 'assistant',
      content: [
        {
          type: 'thinking',
          thinking: [{ type: 'text', text: 'Two plus two.' }],
          signature: 'sig-123',
        },
        { type: 'text', text: '4' },
      ],
    });
  });

  describe('signed thinking chunks', () => {
    const TWO_SIGNED_CHUNKS = [
      {
        type: 'thinking',
        thinking: [{ type: 'text', text: 'First thought.' }],
        signature: 'sig-1',
      },
      {
        type: 'thinking',
        thinking: [{ type: 'text', text: 'Second thought.' }],
        signature: 'sig-2',
      },
      { type: 'text', text: '4' },
    ];
    const TWO_REASONING_PARTS = [
      {
        reasoning: 'First thought.',
        metadata: { reasoningSource: 'mistral', thoughtSignature: 'sig-1' },
      },
      {
        reasoning: 'Second thought.',
        metadata: { reasoningSource: 'mistral', thoughtSignature: 'sig-2' },
      },
      { text: '4' },
    ];

    /** Checks that each signature goes back with the text it signed. */
    async function expectReplayedChunks(messages: MessageData[]) {
      await ai.generate({
        model: mistral.model('mistral-small-latest'),
        messages,
        prompt: 'And 3 + 3?',
      });
      const body = server.requests[server.requests.length - 1].body;
      expect(body.messages[1]).toEqual({
        role: 'assistant',
        content: [
          ...TWO_SIGNED_CHUNKS.slice(0, 2),
          { type: 'text', text: '4' },
        ],
      });
    }

    it('keeps one reasoning part per thinking chunk', async () => {
      server.setNextResponse({
        body: {
          choices: [
            {
              index: 0,
              message: { role: 'assistant', content: TWO_SIGNED_CHUNKS },
              finish_reason: 'stop',
            },
          ],
        },
      });

      const response = await ai.generate({
        model: mistral.model('mistral-small-latest'),
        prompt: 'What is 2 + 2?',
      });

      expect(response.message?.content).toEqual(TWO_REASONING_PARTS);
      await expectReplayedChunks(response.messages);
    });

    it('joins streamed fragments until a signature closes the chunk', async () => {
      const delta = (content: unknown) => ({
        id: '1',
        choices: [{ index: 0, delta: { content }, finish_reason: null }],
      });
      const fragment = (text: string, signature?: string) => [
        {
          type: 'thinking',
          thinking: [{ type: 'text', text }],
          ...(signature ? { signature } : {}),
        },
      ];
      server.setNextResponse({
        stream: true,
        chunks: [
          {
            id: '1',
            choices: [
              {
                index: 0,
                delta: { role: 'assistant', content: '' },
                finish_reason: null,
              },
            ],
          },
          delta(fragment('First ')),
          delta(fragment('thought.', 'sig-1')),
          delta(fragment('Second ')),
          delta(fragment('thought.')),
          delta(fragment('', 'sig-2')),
          {
            id: '1',
            choices: [
              { index: 0, delta: { content: '4' }, finish_reason: 'stop' },
            ],
          },
        ],
      });

      const { stream, response } = ai.generateStream({
        model: mistral.model('mistral-small-latest'),
        prompt: 'What is 2 + 2?',
      });
      for await (const _ of stream) {
      }
      const final = await response;

      expect(final.message?.content).toEqual(TWO_REASONING_PARTS);
      await expectReplayedChunks(final.messages);
    });
  });

  it('sends output_dimension for embeddings', async () => {
    server.setNextResponse({
      body: {
        object: 'list',
        model: 'codestral-embed',
        data: [{ object: 'embedding', index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 1, total_tokens: 1 },
      },
    });

    const embeddings = await ai.embed({
      embedder: mistral.embedder('codestral-embed', { outputDimension: 256 }),
      content: 'hello',
    });

    expect(embeddings).toEqual([{ embedding: [0.1, 0.2] }]);
    const body = server.requests[0].body;
    expect(body).toEqual({
      model: 'codestral-embed',
      input: ['hello'],
      encoding_format: 'float',
      output_dimension: 256,
    });
    expect(body).not.toHaveProperty('dimensions');
    expectMistralEmbeddingRequest(body);
  });

  it('does not pass unknown embedder options through to the request', async () => {
    server.setNextResponse({
      body: {
        object: 'list',
        data: [{ object: 'embedding', index: 0, embedding: [0.1] }],
      },
    });

    await ai.embed({
      embedder: mistral.embedder('mistral-embed'),
      content: 'hello',
      options: { dimensions: 256 } as any,
    });

    expect(server.requests[0].body).toEqual({
      model: 'mistral-embed',
      input: ['hello'],
      encoding_format: 'float',
    });
  });

  it('defines the known models and embedders', async () => {
    const codestral = await ai.registry.lookupAction(
      '/model/mistral/codestral-latest'
    );
    expect(codestral?.__action.metadata?.model.supports).toEqual({
      multiturn: true,
      tools: true,
      media: false,
      systemRole: true,
      output: ['text', 'json'],
      constrained: 'all',
    });
    const large = await ai.registry.lookupAction(
      '/model/mistral/mistral-large-latest'
    );
    expect(large?.__action.metadata?.model.supports.media).toBe(true);
    const large4 = await ai.registry.lookupAction(
      '/model/mistral/mistral-large-4'
    );
    expect(large4?.__action.metadata?.model.supports).toMatchObject({
      tools: true,
      media: true,
      constrained: 'all',
    });
    const embed = await ai.registry.lookupAction(
      '/embedder/mistral/mistral-embed'
    );
    expect(embed).toBeDefined();
  });

  it('resolves models that are not in the known list', async () => {
    await ai.generate({
      model: 'mistral/ministral-14b-2512',
      prompt: 'hi',
      config: { maxOutputTokens: 10 },
    });

    const body = server.requests[0].body;
    expect(body.model).toBe('ministral-14b-2512');
    expect(body.max_tokens).toBe(10);
    const action = await ai.registry.lookupAction(
      '/model/mistral/ministral-14b-2512'
    );
    expect(action?.__action.metadata?.model.supports).toEqual(
      UNKNOWN_MODEL_SUPPORTS
    );
  });

  it('lists models without capabilities as text only', async () => {
    const plugin = mistral(fakeServerOptions(server));
    server.setNextResponse({
      body: {
        object: 'list',
        data: [
          { id: 'mistral-large-latest', object: 'model' },
          { id: 'ministral-14b-2512', object: 'model' },
          { id: 'mistral-embed', object: 'model' },
        ],
      },
    });

    const actions = await plugin.list!();

    expect(actions.map((a) => ({ name: a.name, type: a.actionType }))).toEqual([
      { name: 'mistral/mistral-large-latest', type: 'model' },
      { name: 'mistral/ministral-14b-2512', type: 'model' },
      { name: 'mistral/mistral-embed', type: 'embedder' },
    ]);
    expect(actions[0].metadata?.model?.supports?.tools).toBe(true);
    expect(actions[1].metadata?.model?.supports).toEqual(
      UNKNOWN_MODEL_SUPPORTS
    );
  });

  describe('listing with /v1/models capabilities', () => {
    // Entries shaped like Mistral's `BaseModelCard`, trimmed to the fields
    // the plugin reads.
    const MODELS_RESPONSE = {
      object: 'list',
      data: [
        {
          id: 'mistral-large-latest',
          object: 'model',
          capabilities: {
            completion_chat: true,
            function_calling: true,
            vision: true,
          },
        },
        {
          id: 'ministral-14b-2512',
          object: 'model',
          capabilities: {
            completion_chat: true,
            function_calling: true,
            vision: true,
          },
        },
        {
          id: 'some-text-model',
          object: 'model',
          capabilities: {
            completion_chat: true,
            function_calling: false,
            vision: false,
          },
        },
        {
          id: 'mistral-ocr-latest',
          object: 'model',
          capabilities: { completion_chat: false, ocr: true },
        },
        {
          id: 'mistral-moderation-latest',
          object: 'model',
          capabilities: { completion_chat: false, moderation: true },
        },
        {
          id: 'mistral-embed',
          object: 'model',
          capabilities: { completion_chat: false },
        },
      ],
    };

    it('lists only chat models, with tools and media from capabilities', async () => {
      const plugin = mistral(fakeServerOptions(server));
      server.setNextResponse({ body: MODELS_RESPONSE });

      const actions = await plugin.list!();

      expect(
        actions.map((a) => ({ name: a.name, type: a.actionType }))
      ).toEqual([
        { name: 'mistral/mistral-large-latest', type: 'model' },
        { name: 'mistral/ministral-14b-2512', type: 'model' },
        { name: 'mistral/some-text-model', type: 'model' },
        { name: 'mistral/mistral-embed', type: 'embedder' },
      ]);
      // Known models keep their curated info.
      expect(actions[0].metadata?.model?.supports?.constrained).toBe('all');
      expect(actions[1].metadata?.model?.supports).toEqual({
        ...UNKNOWN_MODEL_SUPPORTS,
        tools: true,
        media: true,
      });
      expect(actions[2].metadata?.model?.supports).toEqual(
        UNKNOWN_MODEL_SUPPORTS
      );
    });

    it('resolves a listed model with its listed capabilities', async () => {
      server.setNextResponse({ body: MODELS_RESPONSE });
      await ai.registry.listResolvableActions();

      const action = await ai.registry.lookupAction(
        '/model/mistral/ministral-14b-2512'
      );

      expect(action?.__action.metadata?.model.supports).toEqual({
        ...UNKNOWN_MODEL_SUPPORTS,
        tools: true,
        media: true,
      });
    });
  });

  describe('request-scoped API keys', () => {
    const CHAT_RESPONSE = {
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: 'scoped' },
          finish_reason: 'stop',
        },
      ],
    };
    const EMBED_RESPONSE = {
      object: 'list',
      data: [{ object: 'embedding', index: 0, embedding: [0.5] }],
    };

    it('sends a request-scoped key to the configured base URL', async () => {
      const scopedServer = new FakeOpenAIServer('scoped-key');
      await scopedServer.start();
      try {
        const scopedAi = genkit({
          plugins: [
            mistral({
              ...fakeServerOptions(scopedServer),
              apiKey: 'default-key',
            }),
          ],
        });
        scopedServer.setNextResponse({ body: CHAT_RESPONSE });
        scopedServer.setNextResponse({ body: EMBED_RESPONSE });

        const response = await scopedAi.generate({
          model: mistral.model('mistral-small-latest'),
          prompt: 'hi',
          config: { apiKey: 'scoped-key' },
        });
        const embeddings = await scopedAi.embed({
          embedder: mistral.embedder('mistral-embed'),
          content: 'hi',
          options: { apiKey: 'scoped-key' },
        });

        expect(response.text).toBe('scoped');
        expect(embeddings).toEqual([{ embedding: [0.5] }]);
        expect(scopedServer.requests).toHaveLength(2);
        for (const request of scopedServer.requests) {
          expect(request.headers['authorization']).toBe('Bearer scoped-key');
        }
        expectMistralRequests(scopedServer);
      } finally {
        scopedServer.stop();
      }
    });

    it('sends a request-scoped key when the plugin has apiKey: false', async () => {
      const saved = process.env.MISTRAL_API_KEY;
      delete process.env.MISTRAL_API_KEY;
      const scopedServer = new FakeOpenAIServer('scoped-key');
      await scopedServer.start();
      try {
        const scopedAi = genkit({
          plugins: [
            mistral({ ...fakeServerOptions(scopedServer), apiKey: false }),
          ],
        });
        scopedServer.setNextResponse({ body: CHAT_RESPONSE });

        const response = await scopedAi.generate({
          model: mistral.model('mistral-small-latest'),
          prompt: 'hi',
          config: { apiKey: 'scoped-key' },
        });

        expect(response.text).toBe('scoped');
        expect(scopedServer.requests).toHaveLength(1);
        expect(scopedServer.requests[0].headers['authorization']).toBe(
          'Bearer scoped-key'
        );
        expectMistralRequests(scopedServer);
      } finally {
        scopedServer.stop();
        if (saved !== undefined) process.env.MISTRAL_API_KEY = saved;
      }
    });

    it('sends a request-scoped key to api.mistral.ai by default', async () => {
      const calls: { url: string; authorization: string | null }[] = [];
      const bodies: Record<string, any>[] = [];
      const recordingFetch = async (url: any, init?: any) => {
        bodies.push(JSON.parse(init.body));
        calls.push({
          url: String(url),
          authorization: new Headers(init?.headers).get('authorization'),
        });
        const body = String(url).endsWith('/embeddings')
          ? EMBED_RESPONSE
          : CHAT_RESPONSE;
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      };
      const defaultAi = genkit({
        plugins: [
          mistral({ apiKey: 'default-key', fetch: recordingFetch as any }),
        ],
      });

      await defaultAi.generate({
        model: mistral.model('mistral-small-latest'),
        prompt: 'hi',
        config: { apiKey: 'scoped-key' },
      });
      await defaultAi.embed({
        embedder: mistral.embedder('mistral-embed'),
        content: 'hi',
        options: { apiKey: 'scoped-key' },
      });

      expect(calls).toEqual([
        {
          url: 'https://api.mistral.ai/v1/chat/completions',
          authorization: 'Bearer scoped-key',
        },
        {
          url: 'https://api.mistral.ai/v1/embeddings',
          authorization: 'Bearer scoped-key',
        },
      ]);
      expectMistralChatRequest(bodies[0]);
      expectMistralEmbeddingRequest(bodies[1]);
    });
  });
});
