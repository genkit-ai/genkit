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

import { describe, expect, it } from '@jest/globals';
import type { GenerateRequest } from 'genkit';
import type { ChatCompletionCreateParamsNonStreaming } from 'openai/resources/index.mjs';
import { toOpenAIRequestBody } from '../src/model';
import {
  GroqChatCompletionConfigSchema,
  groqModelRef,
  groqRequestBuilder,
  SUPPORTED_GROQ_MODELS,
} from '../src/groq/groq';

describe('Groq request builder', () => {
  it('maps Groq-specific config fields onto the OpenAI request body', () => {
    const request = {
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config: {
        temperature: 0.2,
        reasoningEffort: 'high',
        reasoningFormat: 'parsed',
        includeReasoning: true,
        serviceTier: 'on_demand',
      },
    } as GenerateRequest;

    const body = toOpenAIRequestBody(
      'openai/gpt-oss-120b',
      request,
      groqRequestBuilder
    ) as ChatCompletionCreateParamsNonStreaming & {
      reasoning_format?: string;
      include_reasoning?: boolean;
      service_tier?: string;
    };

    expect(body.model).toBe('openai/gpt-oss-120b');
    expect(body.temperature).toBe(0.2);
    expect(body.reasoning_effort).toBe('high');
    expect(body.reasoning_format).toBe('parsed');
    expect(body.include_reasoning).toBe(true);
    expect(body.service_tier).toBe('on_demand');
  });

  it('omits unset Groq-specific fields', () => {
    const request = {
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config: { temperature: 0.5 },
    } as GenerateRequest;

    const body = toOpenAIRequestBody(
      'llama-3.3-70b-versatile',
      request,
      groqRequestBuilder
    ) as unknown as Record<string, unknown>;

    expect(body.temperature).toBe(0.5);
    expect(body).not.toHaveProperty('reasoning_effort');
    expect(body).not.toHaveProperty('reasoning_format');
    expect(body).not.toHaveProperty('include_reasoning');
    expect(body).not.toHaveProperty('service_tier');
  });
});

describe('Groq model refs', () => {
  it('namespaces known models under groq/', () => {
    expect(SUPPORTED_GROQ_MODELS['llama-3.3-70b-versatile'].name).toBe(
      'groq/llama-3.3-70b-versatile'
    );
    expect(SUPPORTED_GROQ_MODELS['openai/gpt-oss-120b'].name).toBe(
      'groq/openai/gpt-oss-120b'
    );
  });

  it('uses Groq config schema and media support for Qwen', () => {
    const qwen = SUPPORTED_GROQ_MODELS['qwen/qwen3.6-27b'];
    expect(qwen.configSchema).toBe(GroqChatCompletionConfigSchema);
    expect(qwen.info?.supports?.media).toBe(true);

    const custom = groqModelRef({ name: 'custom-model' });
    expect(custom.name).toBe('groq/custom-model');
    expect(custom.info?.supports?.media).toBe(false);
  });
});
