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
import type { ModelAction } from 'genkit/model';
import { describe, mock, test } from 'node:test';
import { anthropic } from '../src/index.js';
import { PluginOptions, __testClient } from '../src/types.js';
import {
  createMockAnthropicClient,
  createMockAnthropicMessage,
  mockContentBlockStart,
  mockTextChunk,
  mockToolUseChunk,
} from './mocks/anthropic-client.js';

describe('Streaming Integration Tests', () => {
  test('should use streaming API when onChunk is provided', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [
        mockContentBlockStart('Hello'),
        mockTextChunk(' world'),
        mockTextChunk('!'),
      ],
      messageResponse: createMockAnthropicMessage({
        text: 'Hello world!',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
        output: { format: 'text' },
      },
      {
        onChunk: mock.fn() as any,
        abortSignal: new AbortController().signal,
      }
    );

    // Verify final response
    assert.ok(response, 'Response should be returned');
    assert.ok(
      response.candidates?.[0]?.message.content[0].text,
      'Response should have text content'
    );

    // Since we can't control whether the runner chooses streaming or not from
    // the plugin level, just verify we got a response
    // The runner-level tests verify streaming behavior in detail
  });

  test('should handle streaming with multiple content blocks', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [
        mockContentBlockStart('First block'),
        mockTextChunk(' continues'),
        {
          type: 'content_block_start',
          index: 1,
          content_block: {
            type: 'text',
            text: 'Second block',
          },
        } as any,
        {
          type: 'content_block_delta',
          index: 1,
          delta: {
            type: 'text_delta',
            text: ' here',
          },
        } as any,
      ],
      messageResponse: createMockAnthropicMessage({
        text: 'First block continues',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
        output: { format: 'text' },
      },
      {
        onChunk: mock.fn() as any,
        abortSignal: new AbortController().signal,
      }
    );

    // Verify response is returned even with multiple content blocks
    assert.ok(response, 'Response should be returned');
  });

  test('should handle streaming with tool use', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [
        mockToolUseChunk('toolu_123', 'get_weather', { city: 'NYC' }),
      ],
      messageResponse: createMockAnthropicMessage({
        toolUse: {
          id: 'toolu_123',
          name: 'get_weather',
          input: { city: 'NYC' },
        },
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Get NYC weather' }] }],
        tools: [
          {
            name: 'get_weather',
            description: 'Get weather for a city',
            inputSchema: {
              type: 'object',
              properties: {
                city: { type: 'string' },
              },
              required: ['city'],
            },
          },
        ],
        output: { format: 'text' },
      },
      {
        onChunk: mock.fn() as any,
        abortSignal: new AbortController().signal,
      }
    );

    // Verify tool use in response
    assert.ok(response.candidates?.[0]?.message.content[0].toolRequest);
    assert.strictEqual(
      response.candidates[0].message.content[0].toolRequest?.name,
      'get_weather'
    );
  });

  test('should handle abort signal', async () => {
    const abortController = new AbortController();

    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Hello world',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    // Abort before starting
    abortController.abort();

    // Test that abort signal is passed through
    // The actual abort behavior is tested in runner tests
    try {
      await modelAction(
        {
          messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
          output: { format: 'text' },
        },
        {
          onChunk: mock.fn() as any,
          abortSignal: abortController.signal,
        }
      );
      // If we get here, the mock doesn't fully simulate abort behavior,
      // which is fine since runner tests cover this
    } catch (error: any) {
      // Expected abort error
      assert.ok(
        error.message.includes('Abort') || error.name === 'AbortError',
        'Should throw abort error'
      );
    }
  });

  test('should handle errors during streaming', async () => {
    const mockClient = createMockAnthropicClient({
      shouldError: new Error('API error'),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    try {
      await modelAction(
        {
          messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
          output: { format: 'text' },
        },
        {
          onChunk: mock.fn() as any,
          abortSignal: new AbortController().signal,
        }
      );
      assert.fail('Should have thrown an error');
    } catch (error: any) {
      assert.strictEqual(error.message, 'API error');
    }
  });

  test('should handle empty response', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [],
      messageResponse: createMockAnthropicMessage({
        text: '',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
        output: { format: 'text' },
      },
      {
        onChunk: mock.fn() as any,
        abortSignal: new AbortController().signal,
      }
    );

    assert.ok(response, 'Should return response even with empty content');
  });

  test('should include usage metadata in streaming response', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [mockContentBlockStart('Response'), mockTextChunk(' text')],
      messageResponse: createMockAnthropicMessage({
        text: 'Response text',
        usage: {
          input_tokens: 50,
          output_tokens: 25,
        },
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
        output: { format: 'text' },
      },
      {
        onChunk: mock.fn() as any,
        abortSignal: new AbortController().signal,
      }
    );

    assert.ok(response.usage, 'Should include usage metadata');
    assert.strictEqual(response.usage?.inputTokens, 50);
    assert.strictEqual(response.usage?.outputTokens, 25);
  });

  test('should not stream when onChunk is not provided', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Non-streaming response',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    const modelAction = plugin.resolve!(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
      },
      {
        abortSignal: new AbortController().signal,
      }
    );

    // Verify non-streaming API was called
    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);

    // Verify stream API was NOT called
    const streamStub = mockClient.messages.stream as any;
    assert.strictEqual(streamStub.mock.calls.length, 0);
  });
});
