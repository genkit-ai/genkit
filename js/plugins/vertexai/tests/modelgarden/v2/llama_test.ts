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

import * as assert from 'assert';
import { GenerateRequest, modelRef } from 'genkit/model';
import { describe, it } from 'node:test';
import type {
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/index.mjs';
import { LlamaConfigSchema } from '../../../src/modelgarden/v2/llama';
import {
  fromOpenAiChoice,
  fromOpenAiChunkChoice,
  toRequestBody,
} from '../../../src/modelgarden/v2/openai_compatibility';

const fakeModel = modelRef({
  name: 'vertex-model-garden/meta/llama-4-maverick-17b-128e-instruct-maas',
  info: {
    supports: {
      multiturn: true,
      tools: true,
      media: false,
      systemRole: true,
      output: ['text', 'json'],
    },
  },
  configSchema: LlamaConfigSchema,
});

describe('Llama request conversion', () => {
  it('should convert a simple request', () => {
    const genkitRequest: GenerateRequest<typeof LlamaConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
    };
    const openAIRequest = toRequestBody(fakeModel, genkitRequest);
    assert.deepStrictEqual(
      openAIRequest.model,
      'meta/llama-4-maverick-17b-128e-instruct-maas'
    );
    assert.deepStrictEqual(openAIRequest.messages, [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
    ]);
  });

  it('should convert a request with history', () => {
    const genkitRequest: GenerateRequest<typeof LlamaConfigSchema> = {
      messages: [
        { role: 'user', content: [{ text: 'Hello' }] },
        { role: 'model', content: [{ text: 'Hi there!' }] },
        { role: 'user', content: [{ text: 'How are you?' }] },
      ],
    };
    const openAIRequest = toRequestBody(fakeModel, genkitRequest);
    assert.deepStrictEqual(openAIRequest.messages, [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      { role: 'assistant', content: 'Hi there!' },
      { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
    ]);
  });

  it('should convert a request with tools', () => {
    const genkitRequest: GenerateRequest<typeof LlamaConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
      tools: [
        {
          name: 'my_tool',
          description: 'a tool',
          inputSchema: {
            type: 'object',
            properties: { foo: { type: 'string' } },
          },
        },
      ],
    };
    const openAIRequest = toRequestBody(fakeModel, genkitRequest);
    assert.deepStrictEqual(openAIRequest.tools, [
      {
        type: 'function',
        function: {
          name: 'my_tool',
          parameters: {
            type: 'object',
            properties: { foo: { type: 'string' } },
          },
        },
      },
    ]);
  });

  it('should convert a tool call response', () => {
    const genkitRequest: GenerateRequest<typeof LlamaConfigSchema> = {
      messages: [
        { role: 'user', content: [{ text: 'Search for cats' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'search',
                ref: 'tool1',
                input: { query: 'cats' },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                ref: 'tool1',
                name: 'search',
                output: {
                  results: ['cat1.jpg', 'cat2.jpg'],
                },
              },
            },
          ],
        },
      ],
    };
    const openAIRequest = toRequestBody(fakeModel, genkitRequest);
    assert.deepStrictEqual(openAIRequest.messages, [
      { role: 'user', content: [{ type: 'text', text: 'Search for cats' }] },
      {
        role: 'assistant',
        tool_calls: [
          {
            id: 'tool1',
            type: 'function',
            function: { name: 'search', arguments: '{"query":"cats"}' },
          },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'tool1',
        content: '{"results":["cat1.jpg","cat2.jpg"]}',
      },
    ]);
  });

  it('should send text alongside tool calls on a replayed model turn', () => {
    const genkitRequest: GenerateRequest<typeof LlamaConfigSchema> = {
      messages: [
        { role: 'user', content: [{ text: 'Weather in Paris?' }] },
        {
          role: 'model',
          content: [
            { text: 'Let me check the weather.' },
            {
              toolRequest: {
                name: 'getWeather',
                ref: 'call_1',
                input: { city: 'Paris' },
              },
            },
          ],
        },
      ],
    };
    const openAIRequest = toRequestBody(fakeModel, genkitRequest);
    assert.deepStrictEqual(openAIRequest.messages[1], {
      role: 'assistant',
      content: 'Let me check the weather.',
      tool_calls: [
        {
          id: 'call_1',
          type: 'function',
          function: { name: 'getWeather', arguments: '{"city":"Paris"}' },
        },
      ],
    });
  });
});

describe('Llama response conversion', () => {
  const toolCall = {
    index: 0,
    id: 'call_1',
    type: 'function' as const,
    function: { name: 'getWeather', arguments: '{"city":"Paris"}' },
  };
  const toolRequest = {
    toolRequest: {
      name: 'getWeather',
      ref: 'call_1',
      input: { city: 'Paris' },
    },
  };
  const choice = (
    message: Partial<ChatCompletion.Choice['message']>
  ): ChatCompletion.Choice => ({
    index: 0,
    finish_reason: 'tool_calls',
    logprobs: null,
    message: { role: 'assistant', content: null, refusal: null, ...message },
  });
  const chunk = (
    delta: ChatCompletionChunk.Choice['delta']
  ): ChatCompletionChunk.Choice => ({ index: 0, finish_reason: null, delta });

  it('should keep text that comes with tool calls', () => {
    assert.deepStrictEqual(
      fromOpenAiChoice(
        choice({ content: 'Let me check the weather.', tool_calls: [toolCall] })
      ).message.content,
      [{ text: 'Let me check the weather.' }, toolRequest]
    );
  });

  it('should return only tool requests when there is no text', () => {
    for (const content of [null, '']) {
      assert.deepStrictEqual(
        fromOpenAiChoice(choice({ content, tool_calls: [toolCall] })).message
          .content,
        [toolRequest]
      );
    }
  });

  it('should return no parts for an empty reply', () => {
    assert.deepStrictEqual(fromOpenAiChoice(choice({})).message.content, []);
  });

  it('should parse json only when there are no tool calls', () => {
    assert.deepStrictEqual(
      fromOpenAiChoice(choice({ content: '{"a":1}' }), true).message.content,
      [{ data: { a: 1 } }]
    );
    assert.deepStrictEqual(
      fromOpenAiChoice(
        choice({ content: 'Let me check.', tool_calls: [toolCall] }),
        true
      ).message.content,
      [{ text: 'Let me check.' }, toolRequest]
    );
  });

  it('should keep text in a chunk that also has tool calls', () => {
    assert.deepStrictEqual(
      fromOpenAiChunkChoice(
        chunk({ content: 'Let me check.', tool_calls: [toolCall] })
      ).message.content,
      [{ text: 'Let me check.' }, toolRequest]
    );
  });

  it('should return no parts for an empty chunk', () => {
    for (const delta of [{}, { content: null }, { content: '' }]) {
      assert.deepStrictEqual(
        fromOpenAiChunkChoice(chunk(delta)).message.content,
        []
      );
    }
  });
});
