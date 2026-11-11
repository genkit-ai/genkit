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
import { genkit, z, type Genkit } from 'genkit';
import { mistral } from '../src/mistral';
import { FakeOpenAIServer } from './fake_openai_server';

const API_KEY = 'mistral-test-key';

/** Fields the OpenAI plugin sends that Mistral's request schema rejects. */
const NON_MISTRAL_FIELDS = [
  'stream_options',
  'max_completion_tokens',
  'seed',
  'user',
  'reasoning_content',
];

function expectNoNonMistralFields(body: Record<string, unknown>) {
  for (const field of NON_MISTRAL_FIELDS) {
    expect(body).not.toHaveProperty(field);
  }
}

/**
 * `baseURL` is not part of MistralPluginOptions, but the plugin passes it
 * through to the OpenAI client, which lets the tests use the fake server.
 */
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
    server.stop();
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
        topK: 40,
        maxOutputTokens: 128,
        stopSequences: ['END'],
        randomSeed: 42,
        safePrompt: true,
        reasoningEffort: 'high',
        promptMode: 'reasoning',
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
      top_k: 40,
      max_tokens: 128,
      stop: ['END'],
      random_seed: 42,
      safe_prompt: true,
      reasoning_effort: 'high',
      prompt_mode: 'reasoning',
      presence_penalty: 0.5,
      frequency_penalty: 0.25,
    });
    expectNoNonMistralFields(body);
  });

  it('does not pass unknown config through to the request', async () => {
    await ai.generate({
      model: mistral.model('mistral-small-latest'),
      prompt: 'hi',
      config: { seed: 1, user: 'someone' } as any,
    });

    const body = server.requests[0].body;
    expectNoNonMistralFields(body);
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
          // Mistral sends usage on the last chunk without being asked.
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
    expectNoNonMistralFields(body);
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
      model: mistral.model('magistral-small-latest'),
      prompt: 'What is 2 + 2?',
    });

    expect(response.message?.content).toEqual([
      { reasoning: 'Two plus two is four.' },
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
      model: mistral.model('magistral-small-latest'),
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
      { reasoning: 'Two plus two is four.' },
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
      model: mistral.model('magistral-small-latest'),
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
      model: mistral.model('magistral-small-latest'),
      messages: [
        { role: 'user', content: [{ text: 'What is 2 + 2?' }] },
        {
          role: 'model',
          content: [{ reasoning: 'Two plus two is four.' }, { text: '4' }],
        },
        {
          role: 'model',
          content: [
            { reasoning: 'I should call a tool.' },
            { toolRequest: { name: 'add', ref: 'call_1', input: { a: 2 } } },
          ],
        },
      ],
      prompt: 'And 3 + 3?',
    });

    const body = server.requests[0].body;
    expectNoNonMistralFields(body);
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
    const embed = await ai.registry.lookupAction(
      '/embedder/mistral/mistral-embed'
    );
    expect(embed).toBeDefined();
  });

  it('resolves models that are not in the known list', async () => {
    await ai.generate({
      model: 'mistral/open-mistral-nemo',
      prompt: 'hi',
      config: { maxOutputTokens: 10 },
    });

    const body = server.requests[0].body;
    expect(body.model).toBe('open-mistral-nemo');
    expect(body.max_tokens).toBe(10);
  });

  it('lists models and embedders from /v1/models', async () => {
    const plugin = mistral(fakeServerOptions(server));
    server.setNextResponse({
      body: {
        object: 'list',
        data: [
          { id: 'mistral-large-latest', object: 'model' },
          { id: 'open-mistral-nemo', object: 'model' },
          { id: 'mistral-embed', object: 'model' },
        ],
      },
    });

    const actions = await plugin.list!();

    expect(actions.map((a) => ({ name: a.name, type: a.actionType }))).toEqual([
      { name: 'mistral/mistral-large-latest', type: 'model' },
      { name: 'mistral/open-mistral-nemo', type: 'model' },
      { name: 'mistral/mistral-embed', type: 'embedder' },
    ]);
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
      } finally {
        scopedServer.stop();
      }
    });

    it('sends a request-scoped key to api.mistral.ai by default', async () => {
      const calls: { url: string; authorization: string | null }[] = [];
      const recordingFetch = async (url: any, init?: any) => {
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
    });
  });
});
