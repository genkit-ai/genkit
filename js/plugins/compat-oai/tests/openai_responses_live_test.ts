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

import { describe, expect, test } from '@jest/globals';
import OpenAI from 'openai';
import { openAIResponsesModelRunner } from '../src/responses';

/**
 * Smoke tests against the real Responses API. They only run when
 * `OPENAI_API_KEY` is set, so the suite stays green without credentials.
 *
 * `gpt-5-nano` stands in for the registered Responses-only models: it is served
 * over the same endpoint with the same request and response shapes, at a
 * fraction of the latency and cost of `gpt-5-pro`.
 */
const LIVE_MODEL = 'gpt-5-nano';
const maybeDescribe = process.env.OPENAI_API_KEY ? describe : describe.skip;

maybeDescribe('openAI responses live', () => {
  const runner = () =>
    openAIResponsesModelRunner(
      LIVE_MODEL,
      new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
    );

  test('generates text', async () => {
    const response = await runner()({
      messages: [
        { role: 'system', content: [{ text: 'Answer with one word.' }] },
        {
          role: 'user',
          content: [{ text: 'What is the capital of France?' }],
        },
      ],
      config: { maxOutputTokens: 2048 },
    });

    expect(response.finishReason).toBe('stop');
    expect(response.message?.content.map((p) => p.text ?? '').join('')).toMatch(
      /paris/i
    );
    expect(response.usage?.totalTokens).toBeGreaterThan(0);
  }, 120_000);

  test('honours a json output schema', async () => {
    const response = await runner()({
      messages: [{ role: 'user', content: [{ text: 'Name one colour.' }] }],
      config: { maxOutputTokens: 2048 },
      output: {
        format: 'json',
        schema: {
          type: 'object',
          properties: { colour: { type: 'string' } },
          required: ['colour'],
          additionalProperties: false,
        },
      },
    });

    const data = response.message?.content.find((p) => p.data)?.data as {
      colour?: string;
    };
    expect(typeof data?.colour).toBe('string');
  }, 120_000);
});
