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
import type { GenerateResponseChunkData, MessageData } from 'genkit';
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

  test('streams text deltas', async () => {
    const chunks: GenerateResponseChunkData[] = [];
    const response = await runner()(
      {
        messages: [{ role: 'user', content: [{ text: 'Count from 1 to 5.' }] }],
        config: { maxOutputTokens: 2048 },
      },
      { streamingRequested: true, sendChunk: (chunk) => chunks.push(chunk) }
    );

    expect(response.finishReason).toBe('stop');
    const streamedText = chunks
      .flatMap((chunk) => chunk.content)
      .map((part) => part.text ?? '')
      .join('');
    expect(streamedText).toContain('3');
  }, 120_000);

  test('round-trips a tool call with encrypted reasoning across turns', async () => {
    const tools = [
      {
        name: 'getWeather',
        description: 'Returns the current weather for a city.',
        inputSchema: {
          type: 'object',
          properties: { city: { type: 'string' } },
          required: ['city'],
          additionalProperties: false,
        } as Record<string, unknown>,
      },
    ];
    const messages: MessageData[] = [
      {
        role: 'user',
        content: [
          { text: 'Use the getWeather tool to find the weather in Paris.' },
        ],
      },
    ];

    const first = await runner()({
      messages,
      tools,
      config: { maxOutputTokens: 2048 },
    });

    const toolRequest = first.message?.content.find((p) => p.toolRequest);
    expect(toolRequest?.toolRequest?.name).toBe('getWeather');
    // The stateless default must have carried the reasoning payload back.
    expect(
      first.message?.content.some((p) => p.metadata?.encryptedContent)
    ).toBe(true);

    messages.push(first.message!, {
      role: 'tool',
      content: [
        {
          toolResponse: {
            name: 'getWeather',
            ref: toolRequest!.toolRequest!.ref,
            output: 'Sunny, 24C',
          },
        },
      ],
    });

    const second = await runner()({
      messages,
      tools,
      config: { maxOutputTokens: 2048 },
    });

    expect(second.finishReason).toBe('stop');
    expect(second.message?.content.map((p) => p.text ?? '').join('')).toMatch(
      /sunny|24/i
    );
  }, 240_000);
});
