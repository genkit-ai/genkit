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
import { genkit, z } from 'genkit';
import { describe, it } from 'node:test';
import { anthropic } from '../src/index.js';
import { __testClient } from '../src/types.js';
import {
  createMockAnthropicClient,
  createMockAnthropicMessage,
  mockContentBlockStart,
  mockMessageWithContent,
  mockMessageWithToolUse,
  mockTextChunk,
} from './mocks/anthropic-client.js';

import { PluginOptions } from '../src/types.js';

describe('Anthropic Integration', () => {
  it('should successfully generate a response', async () => {
    const mockClient = createMockAnthropicClient();
    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'Hello',
    });

    assert.strictEqual(result.text, 'Hello! How can I help you today?');
  });

  it('should handle tool calling workflow (call tool, receive result, generate final response)', async () => {
    const mockClient = createMockAnthropicClient({
      sequentialResponses: [
        // First response: tool use request
        mockMessageWithToolUse('get_weather', { city: 'NYC' }),
        // Second response: final text after tool result
        createMockAnthropicMessage({
          text: 'The weather in NYC is sunny, 72°F',
        }),
      ],
    });

    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    // Define the tool
    ai.defineTool(
      {
        name: 'get_weather',
        description: 'Get the weather for a city',
        inputSchema: z.object({
          city: z.string(),
        }),
      },
      async (input: { city: string }) => {
        return `The weather in ${input.city} is sunny, 72°F`;
      }
    );

    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'What is the weather in NYC?',
      tools: ['get_weather'],
    });

    assert.ok(
      result.text.includes('NYC') ||
        result.text.includes('sunny') ||
        result.text.includes('72')
    );
  });

  it('should handle multi-turn conversations', async () => {
    const mockClient = createMockAnthropicClient();
    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    // First turn
    const response1 = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'My name is Alice',
    });

    // Second turn with conversation history
    const response2 = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: "What's my name?",
      messages: response1.messages,
    });

    // Verify conversation history is maintained
    assert.ok(
      response2.messages.length >= 2,
      'Should have conversation history'
    );
    assert.strictEqual(response2.messages[0].role, 'user');
    assert.ok(
      response2.messages[0].content[0].text?.includes('Alice') ||
        response2.messages[0].content[0].text?.includes('name')
    );
  });

  it('should stream responses with streaming callback', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [
        mockContentBlockStart('Hello'),
        mockTextChunk(' world'),
        mockTextChunk('!'),
      ],
      messageResponse: {
        content: [{ type: 'text', text: 'Hello world!', citations: null }],
        usage: {
          input_tokens: 5,
          output_tokens: 15,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
          cache_creation: null,
          server_tool_use: null,
          service_tier: null,
        },
      },
    });

    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const chunks: any[] = [];
    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'Say hello world',
      streamingCallback: (chunk) => {
        chunks.push(chunk);
      },
    });

    assert.ok(chunks.length > 0, 'Should have received streaming chunks');
    assert.ok(result.text, 'Should have final response text');
  });

  it('should handle media/image inputs', async () => {
    const mockClient = createMockAnthropicClient();
    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      messages: [
        {
          role: 'user',
          content: [
            { text: 'Describe this image:' },
            {
              media: {
                url: 'data:image/png;base64,R0lGODlhAQABAAAAACw=',
                contentType: 'image/png',
              },
            },
          ],
        },
      ],
    });

    assert.ok(result.text, 'Should generate response for image input');
  });

  it('should handle WEBP image inputs', async () => {
    const mockClient = createMockAnthropicClient();
    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      messages: [
        {
          role: 'user',
          content: [
            { text: 'Describe this image:' },
            {
              media: {
                url: 'data:image/webp;base64,AAA',
                contentType: 'image/webp',
              },
            },
          ],
        },
      ],
    });

    assert.ok(result.text, 'Should generate response for WEBP image input');
    // Verify the request was made with correct media_type
    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);
    const requestBody = createStub.mock.calls[0].arguments[0];
    const imageContent = requestBody.messages[0].content.find(
      (c: any) => c.type === 'image'
    );
    assert.ok(imageContent, 'Should have image content in request');
    assert.strictEqual(
      imageContent.source.media_type,
      'image/webp',
      'Should use WEBP media type from data URL'
    );
  });

  it('should handle WEBP image with mismatched contentType (prefers data URL)', async () => {
    const mockClient = createMockAnthropicClient();
    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                // Data URL says WEBP, but contentType says PNG - should use WEBP
                url: 'data:image/webp;base64,AAA',
                contentType: 'image/png',
              },
            },
          ],
        },
      ],
    });

    assert.ok(result.text, 'Should generate response for WEBP image input');
    // Verify the request was made with WEBP (from data URL), not PNG (from contentType)
    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);
    const requestBody = createStub.mock.calls[0].arguments[0];
    const imageContent = requestBody.messages[0].content.find(
      (c: any) => c.type === 'image'
    );
    assert.ok(imageContent, 'Should have image content in request');
    assert.strictEqual(
      imageContent.source.media_type,
      'image/webp',
      'Should prefer data URL content type (webp) over contentType (png)'
    );
  });

  it('should throw helpful error for text/plain media', async () => {
    const mockClient = createMockAnthropicClient();
    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    await assert.rejects(
      async () => {
        await ai.generate({
          model: 'anthropic/claude-3-5-haiku',
          messages: [
            {
              role: 'user',
              content: [
                {
                  media: {
                    url: 'data:text/plain;base64,AAA',
                    contentType: 'text/plain',
                  },
                },
              ],
            },
          ],
        });
      },
      (error: Error) => {
        return (
          error.message.includes('Text files should be sent as text content') &&
          error.message.includes('text:')
        );
      },
      'Should throw helpful error for text/plain media'
    );
  });

  it('should forward thinking config and surface reasoning in responses', async () => {
    const thinkingContent = [
      {
        type: 'thinking' as const,
        thinking: 'Let me analyze the problem carefully.',
        signature: 'sig_reasoning_123',
      },
      {
        type: 'text' as const,
        text: 'The answer is 42.',
        citations: null,
      },
    ];
    const mockClient = createMockAnthropicClient({
      messageResponse: mockMessageWithContent(thinkingContent),
    });

    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const thinkingConfig = { enabled: true, budgetTokens: 2048 };
    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'What is the meaning of life?',
      config: { thinking: thinkingConfig },
    });

    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);
    const requestBody = createStub.mock.calls[0].arguments[0];
    assert.deepStrictEqual(requestBody.thinking, {
      type: 'enabled',
      budget_tokens: 2048,
    });

    assert.strictEqual(
      result.reasoning,
      'Let me analyze the problem carefully.'
    );
    const assistantMessage = result.messages[result.messages.length - 1];
    const reasoningPart = assistantMessage.content.find(
      (part) => part.reasoning
    );
    assert.ok(reasoningPart, 'Expected reasoning part in assistant message');
    assert.strictEqual(
      (reasoningPart?.metadata as Record<string, unknown>)?.thoughtSignature,
      'sig_reasoning_123'
    );
  });

  it('should propagate API errors correctly', async () => {
    const apiError = new Error('API Error: 401 Unauthorized');
    const mockClient = createMockAnthropicClient({
      shouldError: apiError,
    });

    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    await assert.rejects(
      async () => {
        await ai.generate({
          model: 'anthropic/claude-3-5-haiku',
          prompt: 'Hello',
        });
      },
      (error: Error) => {
        assert.strictEqual(error.message, 'API Error: 401 Unauthorized');
        return true;
      }
    );
  });

  it('should respect abort signals for cancellation', async () => {
    // Note: Detailed abort signal handling is tested in converters_test.ts
    // This test verifies that errors (including abort errors) are properly propagated at the integration layer
    const mockClient = createMockAnthropicClient({
      shouldError: new Error('AbortError'),
    });
    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    await assert.rejects(
      async () => {
        await ai.generate({
          model: 'anthropic/claude-3-5-haiku',
          prompt: 'Hello',
        });
      },
      (error: Error) => {
        // Should propagate the error
        assert.ok(
          error.message.includes('AbortError'),
          'Should propagate errors'
        );
        return true;
      }
    );
  });

  it('should track token usage in responses', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: {
        usage: {
          input_tokens: 25,
          output_tokens: 50,
          cache_creation_input_tokens: 5,
          cache_read_input_tokens: 10,
          cache_creation: null,
          server_tool_use: null,
          service_tier: null,
        },
      },
    });

    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'Hello',
    });

    assert.ok(result.usage, 'Should have usage information');
    assert.strictEqual(result.usage.inputTokens, 25);
    assert.strictEqual(result.usage.outputTokens, 50);
  });

  it('should route requests through beta surface when plugin default is beta', async () => {
    const mockClient = createMockAnthropicClient();
    const ai = genkit({
      plugins: [
        anthropic({
          apiVersion: 'beta',
          [__testClient]: mockClient,
        } as PluginOptions),
      ],
    });

    await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'Hello',
    });

    const betaCreateStub = mockClient.beta.messages.create as any;
    assert.strictEqual(
      betaCreateStub.mock.calls.length,
      1,
      'Beta API should be used'
    );
    const regularCreateStub = mockClient.messages.create as any;
    assert.strictEqual(
      regularCreateStub.mock.calls.length,
      0,
      'Stable API should not be used'
    );
  });

  it('should stream thinking deltas as reasoning chunks', async () => {
    const thinkingConfig = { enabled: true, budgetTokens: 3072 };
    const streamChunks = [
      {
        type: 'content_block_start',
        index: 0,
        content_block: {
          type: 'thinking',
          thinking: '',
          signature: 'sig_stream_123',
        },
      } as any,
      {
        type: 'content_block_delta',
        index: 0,
        delta: {
          type: 'thinking_delta',
          thinking: 'Analyzing intermediate steps.',
        },
      } as any,
      {
        type: 'content_block_start',
        index: 1,
        content_block: {
          type: 'text',
          text: '',
        },
      } as any,
      mockTextChunk('Final streamed response.'),
    ];
    const finalMessage = mockMessageWithContent([
      {
        type: 'thinking',
        thinking: 'Analyzing intermediate steps.',
        signature: 'sig_stream_123',
      },
      {
        type: 'text',
        text: 'Final streamed response.',
        citations: null,
      },
    ]);
    const mockClient = createMockAnthropicClient({
      streamChunks,
      messageResponse: finalMessage,
    });

    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    const chunks: any[] = [];
    const result = await ai.generate({
      model: 'anthropic/claude-3-5-haiku',
      prompt: 'Explain how you reason.',
      streamingCallback: (chunk) => chunks.push(chunk),
      config: { thinking: thinkingConfig },
    });

    const streamStub = mockClient.messages.stream as any;
    assert.strictEqual(streamStub.mock.calls.length, 1);
    const streamRequest = streamStub.mock.calls[0].arguments[0];
    assert.deepStrictEqual(streamRequest.thinking, {
      type: 'enabled',
      budget_tokens: 3072,
    });

    const hasReasoningChunk = chunks.some((chunk) =>
      (chunk.content || []).some(
        (part: any) => part.reasoning === 'Analyzing intermediate steps.'
      )
    );
    assert.ok(
      hasReasoningChunk,
      'Expected reasoning chunk in streaming callback'
    );
    assert.strictEqual(result.reasoning, 'Analyzing intermediate steps.');
  });
});
