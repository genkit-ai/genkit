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

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import type { GenerateRequest } from 'genkit';
import type OpenAI from 'openai';
import {
  defineCompatOpenAIModel,
  openAIModelRunner,
  toOpenAIRequestBody,
} from '../src/model';
import {
  miniMaxModelRef,
  miniMaxRequestBuilder,
  MiniMaxChatCompletionConfigSchema,
  SUPPORTED_MINIMAX_MODELS,
} from '../src/minimax/minimax';
import { FakeOpenAIServer } from './fake_openai_server';

jest.mock('genkit/model', () => {
  const originalModule =
    jest.requireActual<typeof import('genkit/model')>('genkit/model');
  return {
    ...originalModule,
    defineModel: jest.fn((_, runner) => {
      return runner;
    }),
  };
});

describe('MiniMax Model Definitions', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should correctly define supported MiniMax models', () => {
    const modelNames = Object.keys(SUPPORTED_MINIMAX_MODELS);
    expect(modelNames).toContain('MiniMax-M2.7');
    expect(modelNames).toContain('MiniMax-M2.7-highspeed');
    expect(modelNames).toContain('MiniMax-M2.5');
    expect(modelNames).toContain('MiniMax-M2.5-highspeed');
    expect(modelNames).toHaveLength(4);
  });

  it('should define MiniMax-M2.7 with correct capabilities', () => {
    const model = defineCompatOpenAIModel({
      name: 'minimax/MiniMax-M2.7',
      client: {} as OpenAI,
      modelRef: miniMaxModelRef({ name: 'MiniMax-M2.7' }),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'minimax/MiniMax-M2.7',
      supports: {
        multiturn: true,
        tools: true,
        media: false,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });

  it('should define MiniMax-M2.7-highspeed with correct capabilities', () => {
    const model = defineCompatOpenAIModel({
      name: 'minimax/MiniMax-M2.7-highspeed',
      client: {} as OpenAI,
      modelRef: miniMaxModelRef({ name: 'MiniMax-M2.7-highspeed' }),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'minimax/MiniMax-M2.7-highspeed',
      supports: {
        multiturn: true,
        tools: true,
        media: false,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });

  it('should define MiniMax-M2.5 with correct capabilities', () => {
    const model = defineCompatOpenAIModel({
      name: 'minimax/MiniMax-M2.5',
      client: {} as OpenAI,
      modelRef: miniMaxModelRef({ name: 'MiniMax-M2.5' }),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'minimax/MiniMax-M2.5',
      supports: {
        multiturn: true,
        tools: true,
        media: false,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });

  it('should define MiniMax-M2.5-highspeed with correct capabilities', () => {
    const model = defineCompatOpenAIModel({
      name: 'minimax/MiniMax-M2.5-highspeed',
      client: {} as OpenAI,
      modelRef: miniMaxModelRef({ name: 'MiniMax-M2.5-highspeed' }),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'minimax/MiniMax-M2.5-highspeed',
      supports: {
        multiturn: true,
        tools: true,
        media: false,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });
});

describe('MiniMax ModelRef', () => {
  it('should create a model ref with minimax namespace', () => {
    const ref = miniMaxModelRef({ name: 'MiniMax-M2.7' });
    expect(ref.name).toBe('minimax/MiniMax-M2.7');
    expect(ref.configSchema).toBe(MiniMaxChatCompletionConfigSchema);
  });

  it('should allow custom model info', () => {
    const customInfo = {
      supports: {
        multiturn: true,
        tools: false,
        media: true,
        systemRole: true,
        output: ['text'],
      },
    };
    const ref = miniMaxModelRef({
      name: 'custom-model',
      info: customInfo,
    });
    expect(ref.info?.supports?.tools).toBe(false);
    expect(ref.info?.supports?.media).toBe(true);
  });
});

describe('MiniMax Request Builder', () => {
  it('should clamp temperature above 1 to 1', () => {
    const req = {
      messages: [],
      config: { temperature: 1.5 },
    } as GenerateRequest;
    const params = { temperature: 1.5 } as any;
    miniMaxRequestBuilder(req, params);
    expect(params.temperature).toBe(1);
  });

  it('should not modify temperature within valid range', () => {
    const req = {
      messages: [],
      config: { temperature: 0.7 },
    } as GenerateRequest;
    const params = { temperature: 0.7 } as any;
    miniMaxRequestBuilder(req, params);
    expect(params.temperature).toBe(0.7);
  });

  it('should allow temperature of 0', () => {
    const req = {
      messages: [],
      config: { temperature: 0 },
    } as GenerateRequest;
    const params = { temperature: 0 } as any;
    miniMaxRequestBuilder(req, params);
    expect(params.temperature).toBe(0);
  });

  it('should handle undefined temperature', () => {
    const req = {
      messages: [],
      config: {},
    } as GenerateRequest;
    const params = {} as any;
    miniMaxRequestBuilder(req, params);
    expect(params.temperature).toBeUndefined();
  });
});

describe('toOpenAIRequestBody for MiniMax models', () => {
  const baseRequest = { messages: [] } as GenerateRequest;

  it('should not throw for MiniMax-M2.7', () => {
    expect(() =>
      toOpenAIRequestBody('MiniMax-M2.7', baseRequest)
    ).not.toThrow();
  });

  it('should not throw for MiniMax-M2.5', () => {
    expect(() =>
      toOpenAIRequestBody('MiniMax-M2.5', baseRequest)
    ).not.toThrow();
  });

  it('should not throw for MiniMax-M2.5-highspeed', () => {
    expect(() =>
      toOpenAIRequestBody('MiniMax-M2.5-highspeed', baseRequest)
    ).not.toThrow();
  });

  it('should generate correct request body with temperature clamped', () => {
    const request = {
      messages: [
        { role: 'user', content: [{ text: 'Hello' }] },
      ],
      config: { temperature: 1.5 },
    } as GenerateRequest;

    const body = toOpenAIRequestBody(
      'MiniMax-M2.7',
      request,
      miniMaxRequestBuilder
    );
    expect(body.model).toBe('MiniMax-M2.7');
    expect(body.temperature).toBe(1);
  });

  it('should generate correct request body with json output format', () => {
    const request = {
      messages: [
        { role: 'user', content: [{ text: 'Hello' }] },
      ],
      output: { format: 'json' },
    } as GenerateRequest;

    const body = toOpenAIRequestBody(
      'MiniMax-M2.7',
      request,
      miniMaxRequestBuilder
    );
    expect(body.response_format).toEqual({ type: 'json_object' });
  });
});

describe('MiniMax Plugin Integration', () => {
  let server: FakeOpenAIServer;

  beforeEach(async () => {
    server = new FakeOpenAIServer();
    await server.start();
  });

  afterEach(() => {
    server.stop();
    jest.clearAllMocks();
  });

  it('should make a successful non-streaming request', async () => {
    server.setNextResponse({
      body: {
        id: 'chatcmpl-test',
        object: 'chat.completion',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Hello from MiniMax!',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 5,
          total_tokens: 15,
        },
      },
    });

    const client = new (await import('openai')).default({
      apiKey: 'test-key',
      baseURL: server.baseUrl,
    });

    const runner = openAIModelRunner(
      'MiniMax-M2.7',
      client,
      miniMaxRequestBuilder
    );

    const request: GenerateRequest = {
      messages: [
        { role: 'user', content: [{ text: 'Hello' }] },
      ],
    };

    const response = await runner(request);
    expect(response.message?.content[0]).toEqual({
      text: 'Hello from MiniMax!',
    });
    expect(response.finishReason).toBe('stop');
    expect(response.usage?.inputTokens).toBe(10);
    expect(response.usage?.outputTokens).toBe(5);
  });

  it('should make a successful streaming request', async () => {
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk',
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', content: 'Hello' },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk',
          choices: [
            {
              index: 0,
              delta: { content: ' from MiniMax!' },
              finish_reason: null,
            },
          ],
        },
        {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk',
          choices: [
            {
              index: 0,
              delta: {},
              finish_reason: 'stop',
            },
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
          },
        },
      ],
    });

    const client = new (await import('openai')).default({
      apiKey: 'test-key',
      baseURL: server.baseUrl,
    });

    const runner = openAIModelRunner(
      'MiniMax-M2.7',
      client,
      miniMaxRequestBuilder
    );

    const chunks: any[] = [];
    const request: GenerateRequest = {
      messages: [
        { role: 'user', content: [{ text: 'Hello' }] },
      ],
    };

    const response = await runner(request, {
      streamingRequested: true,
      sendChunk: (chunk) => {
        chunks.push(chunk);
      },
    });

    expect(chunks.length).toBeGreaterThan(0);
    expect(response.usage?.inputTokens).toBe(10);
    expect(response.usage?.outputTokens).toBe(5);
  });

  it('should handle tool calling', async () => {
    server.setNextResponse({
      body: {
        id: 'chatcmpl-test',
        object: 'chat.completion',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: null,
              tool_calls: [
                {
                  id: 'call_123',
                  type: 'function',
                  function: {
                    name: 'get_weather',
                    arguments: '{"location":"Beijing"}',
                  },
                },
              ],
            },
            finish_reason: 'tool_calls',
          },
        ],
        usage: {
          prompt_tokens: 20,
          completion_tokens: 15,
          total_tokens: 35,
        },
      },
    });

    const client = new (await import('openai')).default({
      apiKey: 'test-key',
      baseURL: server.baseUrl,
    });

    const runner = openAIModelRunner(
      'MiniMax-M2.7',
      client,
      miniMaxRequestBuilder
    );

    const request: GenerateRequest = {
      messages: [
        { role: 'user', content: [{ text: 'What is the weather in Beijing?' }] },
      ],
      tools: [
        {
          name: 'get_weather',
          description: 'Get the weather for a location',
          inputSchema: {
            type: 'object',
            properties: { location: { type: 'string' } },
          },
        },
      ],
    };

    const response = await runner(request);
    expect(response.message?.content[0]).toEqual({
      toolRequest: {
        name: 'get_weather',
        ref: 'call_123',
        input: { location: 'Beijing' },
      },
    });
  });

  it('should send temperature clamped in the request', async () => {
    server.setNextResponse({
      body: {
        id: 'chatcmpl-test',
        object: 'chat.completion',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: 'ok' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
      },
    });

    const client = new (await import('openai')).default({
      apiKey: 'test-key',
      baseURL: server.baseUrl,
    });

    const runner = openAIModelRunner(
      'MiniMax-M2.7',
      client,
      miniMaxRequestBuilder
    );

    const request: GenerateRequest = {
      messages: [
        { role: 'user', content: [{ text: 'test' }] },
      ],
      config: { temperature: 1.8 },
    };

    await runner(request);

    // Verify the request sent to the server has temperature clamped
    expect(server.requests).toHaveLength(1);
    expect(server.requests[0].body.temperature).toBe(1);
    expect(server.requests[0].body.model).toBe('MiniMax-M2.7');
  });
});

describe('MiniMax Config Schema Validation', () => {
  it('should accept temperature within valid range [0, 1]', () => {
    const result = MiniMaxChatCompletionConfigSchema.safeParse({
      temperature: 0.5,
    });
    expect(result.success).toBe(true);
  });

  it('should accept temperature of 0', () => {
    const result = MiniMaxChatCompletionConfigSchema.safeParse({
      temperature: 0,
    });
    expect(result.success).toBe(true);
  });

  it('should accept temperature of 1', () => {
    const result = MiniMaxChatCompletionConfigSchema.safeParse({
      temperature: 1,
    });
    expect(result.success).toBe(true);
  });

  it('should reject temperature greater than 1', () => {
    const result = MiniMaxChatCompletionConfigSchema.safeParse({
      temperature: 1.5,
    });
    expect(result.success).toBe(false);
  });

  it('should reject negative temperature', () => {
    const result = MiniMaxChatCompletionConfigSchema.safeParse({
      temperature: -0.1,
    });
    expect(result.success).toBe(false);
  });

  it('should accept empty config', () => {
    const result = MiniMaxChatCompletionConfigSchema.safeParse({});
    expect(result.success).toBe(true);
  });
});
