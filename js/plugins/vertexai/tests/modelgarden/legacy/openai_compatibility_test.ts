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

import * as assert from 'assert';
import { describe, it } from 'node:test';
import type {
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/index.mjs';
import {
  fromOpenAiChoice,
  fromOpenAiChunkChoice,
  toOpenAiMessages,
} from '../../../src/modelgarden/legacy/openai_compatibility';

const toolCall = {
  index: 0,
  id: 'call_1',
  type: 'function' as const,
  function: { name: 'getWeather', arguments: '{"city":"Paris"}' },
};
const toolRequest = {
  toolRequest: { name: 'getWeather', ref: 'call_1', input: { city: 'Paris' } },
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

describe('legacy openai compatibility', () => {
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

  it('should send text alongside tool calls on a replayed model turn', () => {
    const [, assistant, replay] = toOpenAiMessages([
      { role: 'user', content: [{ text: 'Weather in Paris?' }] },
      {
        role: 'model',
        content: [{ text: 'Let me check the weather.' }, toolRequest],
      },
      { role: 'model', content: [toolRequest] },
    ]);
    const toolCalls = [
      {
        id: 'call_1',
        type: 'function',
        function: { name: 'getWeather', arguments: '{"city":"Paris"}' },
      },
    ];
    assert.deepStrictEqual(assistant, {
      role: 'assistant',
      content: 'Let me check the weather.',
      tool_calls: toolCalls,
    });
    assert.deepStrictEqual(replay, {
      role: 'assistant',
      tool_calls: toolCalls,
    });
  });
});
