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

import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { modelRef, type GenerateRequest } from 'genkit/model';
import type OpenAI from 'openai';
import {
  ChatCompletionCommonConfigSchema,
  defineCompatOpenAIModel,
  toOpenAIRequestBody,
} from '../src/model';
import { SUPPORTED_GPT_MODELS } from '../src/openai/gpt';

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

  it('should correctly define gpt-5', () => {
    const gpt5 = defineCompatOpenAIModel({
      name: 'openai/gpt-5',
      client: {} as OpenAI,
      modelRef: SUPPORTED_GPT_MODELS['gpt-5'],
    });
    expect({
      name: gpt5.__action.name,
      supports: gpt5.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/gpt-5',
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
        constrained: 'all',
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
