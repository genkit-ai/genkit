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
import type { GenerateRequest } from 'genkit';
import type { ModelAction } from 'genkit/model';
import { describe, mock, test } from 'node:test';
import { anthropic } from '../src/index.js';
import { PluginOptions, __testClient } from '../src/types.js';
import {
  createMockAnthropicClient,
  createMockAnthropicMessage,
} from './mocks/anthropic-client.js';

describe('Model Execution Integration Tests', () => {
  test('should resolve and execute a model via plugin', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Hello from Claude!',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    // Resolve the model action via plugin
    const modelAction = plugin.resolve(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    assert.strictEqual(
      (modelAction as ModelAction).__action.name,
      'anthropic/claude-3-5-haiku-20241022'
    );

    // Execute the model
    const request: GenerateRequest = {
      messages: [
        {
          role: 'user',
          content: [{ text: 'Hi there!' }],
        },
      ],
    };

    const response = await modelAction(request, {
      streamingRequested: false,
      sendChunk: mock.fn(),
      abortSignal: new AbortController().signal,
    } as Parameters<typeof modelAction>[1]);

    assert.ok(response, 'Response should be returned');
    assert.ok(response.candidates, 'Response should have candidates');
    assert.strictEqual(response.candidates.length, 1);
    assert.strictEqual(response.candidates[0].message.role, 'model');
    assert.strictEqual(response.candidates[0].message.content.length, 1);
    assert.strictEqual(
      response.candidates[0].message.content[0].text,
      'Hello from Claude!'
    );

    // Verify API was called
    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);
  });

  test('should handle multi-turn conversations', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'The capital of France is Paris.',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    const modelAction = plugin.resolve(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const request: GenerateRequest = {
      messages: [
        {
          role: 'user',
          content: [{ text: 'What is your name?' }],
        },
        {
          role: 'model',
          content: [{ text: 'I am Claude, an AI assistant.' }],
        },
        {
          role: 'user',
          content: [{ text: 'What is the capital of France?' }],
        },
      ],
    };

    const response = await modelAction(request, {
      streamingRequested: false,
      sendChunk: mock.fn(),
      abortSignal: new AbortController().signal,
    } as Parameters<typeof modelAction>[1]);

    assert.ok(response, 'Response should be returned');
    assert.ok(response.candidates, 'Response should have candidates');
    assert.strictEqual(
      response.candidates[0].message.content[0].text,
      'The capital of France is Paris.'
    );

    // Verify API was called with multi-turn conversation
    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);
    const apiRequest = createStub.mock.calls[0].arguments[0];
    assert.strictEqual(apiRequest.messages.length, 3);
  });

  test('should handle system messages', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Arr matey!',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    const modelAction = plugin.resolve(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const request: GenerateRequest = {
      messages: [
        {
          role: 'system',
          content: [{ text: 'You are a pirate. Respond like a pirate.' }],
        },
        {
          role: 'user',
          content: [{ text: 'Hello!' }],
        },
      ],
    };

    const response = await modelAction(request, {
      streamingRequested: false,
      sendChunk: mock.fn(),
      abortSignal: new AbortController().signal,
    } as Parameters<typeof modelAction>[1]);

    assert.ok(response, 'Response should be returned');

    // Verify system message was passed to API
    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);
    const apiRequest = createStub.mock.calls[0].arguments[0];
    assert.ok(apiRequest.system, 'System prompt should be set');
    assert.deepStrictEqual(apiRequest.system, [
      {
        type: 'text',
        text: 'You are a pirate. Respond like a pirate.',
        citations: null,
        cache_control: undefined,
      },
    ]);
    assert.strictEqual(
      apiRequest.messages.length,
      1,
      'System message should not be in messages array'
    );
  });

  test('should return usage metadata', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Response',
        usage: {
          input_tokens: 100,
          output_tokens: 50,
        },
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    const modelAction = plugin.resolve(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
      },
      {
        streamingRequested: false,
        sendChunk: mock.fn(),
        abortSignal: new AbortController().signal,
      } as Parameters<typeof modelAction>[1]
    );

    assert.ok(response.usage, 'Usage should be returned');
    assert.strictEqual(response.usage?.inputTokens, 100);
    assert.strictEqual(response.usage?.outputTokens, 50);
  });

  test('should handle different stop reasons', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'This is a partial response',
        stopReason: 'max_tokens',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    const modelAction = plugin.resolve(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Tell me a story' }] }],
      },
      {
        streamingRequested: false,
        sendChunk: mock.fn(),
        abortSignal: new AbortController().signal,
      } as Parameters<typeof modelAction>[1]
    );

    assert.ok(response, 'Response should be returned');
    assert.ok(response.candidates, 'Response should have candidates');
    assert.strictEqual(response.candidates[0].finishReason, 'length');
  });

  test('should resolve model without anthropic prefix', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Response',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    // Resolve without prefix
    const modelAction = plugin.resolve(
      'model',
      'claude-3-5-haiku-20241022'
    ) as ModelAction;
    assert.ok(modelAction, 'Model should be resolved without prefix');

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
      },
      {
        streamingRequested: false,
        sendChunk: mock.fn(),
        abortSignal: new AbortController().signal,
      } as Parameters<typeof modelAction>[1]
    );

    assert.ok(response, 'Response should be returned');
  });

  test('should resolve model with anthropic prefix', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Response',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    // Resolve with prefix
    const modelAction = plugin.resolve(
      'model',
      'anthropic/claude-3-5-haiku-20241022'
    ) as ModelAction;
    assert.ok(modelAction, 'Model should be resolved with prefix');

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
      },
      {
        streamingRequested: false,
        sendChunk: mock.fn(),
        abortSignal: new AbortController().signal,
      } as Parameters<typeof modelAction>[1]
    );

    assert.ok(response, 'Response should be returned');
  });

  test('should handle unknown model names', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Response from future model',
      }),
    });

    const plugin = anthropic({
      apiKey: 'test-key',
      [__testClient]: mockClient,
    } as PluginOptions);

    // Verify plugin has resolve method
    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    // Resolve unknown model (passes through to API)
    const modelAction = plugin.resolve(
      'model',
      'claude-99-experimental-12345'
    ) as ModelAction;
    assert.ok(modelAction, 'Unknown model should still be resolved');

    const response = await modelAction(
      {
        messages: [{ role: 'user', content: [{ text: 'Hi' }] }],
      },
      {
        streamingRequested: false,
        sendChunk: mock.fn(),
        abortSignal: new AbortController().signal,
      } as Parameters<typeof modelAction>[1]
    );

    assert.ok(response, 'Response should be returned for unknown model');
    assert.ok(response.candidates, 'Response should have candidates');
    assert.strictEqual(
      response.candidates?.[0]?.message.content[0].text,
      'Response from future model',
      'Response should have candidates'
    );
  });
});
