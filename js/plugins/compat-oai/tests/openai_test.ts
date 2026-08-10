/**
 * Copyright 2024 The Fire Company
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
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import { modelRef, type GenerateRequest } from 'genkit/model';
import type OpenAI from 'openai';
import {
  ChatCompletionCommonConfigSchema,
  defineCompatOpenAIModel,
  toOpenAIRequestBody,
} from '../src/model';
import { openAI } from '../src/openai/index';
import { FakeOpenAIServer } from './fake_openai_server';

describe('gptModel', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should correctly define supported GPT models', () => {
    const model = defineCompatOpenAIModel({
      name: 'openai/gpt-4o',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4o'),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4o',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });

  it('should correctly define gpt-4.1, gpt-4.1-mini, and gpt-4.1-nano', () => {
    const gpt41 = defineCompatOpenAIModel({
      name: 'openai/gpt-4.1',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4.1'),
    });
    expect({
      name: gpt41.__action.name,
      supports: gpt41.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4.1',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });

    const gpt41mini = defineCompatOpenAIModel({
      name: 'openai/gpt-4.1-mini',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4.1-mini'),
    });
    expect({
      name: gpt41mini.__action.name,
      supports: gpt41mini.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4.1-mini',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });

    const gpt41nano = defineCompatOpenAIModel({
      name: 'openai/gpt-4.1-nano',
      client: {} as OpenAI,
      modelRef: testModelRef('openai/gpt-4.1-nano'),
    });
    expect({
      name: gpt41nano.__action.name,
      supports: gpt41nano.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-4.1-nano',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    });
  });
});

// Additional test to ensure toOpenAiRequestBody works for new models

describe('toOpenAiRequestBody for new GPT-4.1 variants', () => {
  const baseRequest = { messages: [] } as GenerateRequest;
  it('should not throw for gpt-4.1', () => {
    expect(() => toOpenAIRequestBody('gpt-4.1', baseRequest)).not.toThrow();
  });
  it('should not throw for gpt-4.1-mini', () => {
    expect(() =>
      toOpenAIRequestBody('gpt-4.1-mini', baseRequest)
    ).not.toThrow();
  });
  it('should not throw for gpt-4.1-nano', () => {
    expect(() =>
      toOpenAIRequestBody('gpt-4.1-nano', baseRequest)
    ).not.toThrow();
  });
});

describe('listActions model filtering', () => {
  let server: FakeOpenAIServer;
  let previousBaseUrl: string | undefined;

  beforeAll(async () => {
    server = new FakeOpenAIServer();
    await server.start();
    // The openAI plugin does not accept a baseURL, so the fake server is
    // injected the way a user would point the SDK at a proxy.
    previousBaseUrl = process.env.OPENAI_BASE_URL;
    process.env.OPENAI_BASE_URL = server.baseUrl;
  });

  afterAll(() => {
    if (previousBaseUrl === undefined) {
      delete process.env.OPENAI_BASE_URL;
    } else {
      process.env.OPENAI_BASE_URL = previousBaseUrl;
    }
    server.stop();
  });

  it('keeps current codex models and drops the legacy completion families', async () => {
    const plugin = openAI({ apiKey: 'key' });
    server.setNextResponse({
      body: {
        object: 'list',
        data: [
          'gpt-5-codex',
          'gpt-5.1-codex-max',
          'codex-mini-latest',
          'gpt-4o',
          'code-davinci-002',
          'code-cushman-001',
          'davinci-002',
          'babbage-002',
        ].map((id) => ({
          id,
          object: 'model',
          created: 0,
          owned_by: 'openai',
        })),
      },
    });

    const names = (await plugin.list!()).map((a) => a.name);

    expect(names).toStrictEqual([
      'openai/gpt-5-codex',
      'openai/gpt-5.1-codex-max',
      'openai/codex-mini-latest',
      'openai/gpt-4o',
    ]);
  });
});

function testModelRef(name: string) {
  return modelRef({
    name,
    info: {
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    },
    configSchema: ChatCompletionCommonConfigSchema,
  });
}
