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
import type { Part } from 'genkit';
import { describe, it } from 'node:test';

import { BetaRunner } from '../src/runner/beta.js';
import { createMockAnthropicClient } from './mocks/anthropic-client.js';

describe('BetaRunner.toAnthropicMessageContent', () => {
  function createRunner() {
    return new BetaRunner({
      name: 'anthropic/claude-3-5-haiku',
      client: createMockAnthropicClient(),
    });
  }

  it('converts PDF media parts into document blocks', () => {
    const runner = createRunner();
    const part: Part = {
      media: {
        contentType: 'application/pdf',
        url: 'data:application/pdf;base64,UEsDBAoAAAAAAD',
      },
    };

    const result = (runner as any).toAnthropicMessageContent(part);

    assert.strictEqual(result.type, 'document');
    assert.ok(result.source);
    assert.strictEqual(result.source.type, 'base64');
    assert.strictEqual(result.source.media_type, 'application/pdf');
    assert.ok(result.source.data);
  });

  it('throws when tool request ref is missing', () => {
    const runner = createRunner();
    const part: Part = {
      toolRequest: {
        name: 'do_something',
        input: { foo: 'bar' },
      },
    };

    assert.throws(() => {
      (runner as any).toAnthropicMessageContent(part);
    }, /Tool request ref is required/);
  });

  it('maps tool request with ref into tool_use block', () => {
    const runner = createRunner();
    const part: Part = {
      toolRequest: {
        ref: 'tool-123',
        name: 'do_something',
        input: { foo: 'bar' },
      },
    };

    const result = (runner as any).toAnthropicMessageContent(part);

    assert.strictEqual(result.type, 'tool_use');
    assert.strictEqual(result.id, 'tool-123');
    assert.strictEqual(result.name, 'do_something');
    assert.deepStrictEqual(result.input, { foo: 'bar' });
  });

  it('throws when tool response ref is missing', () => {
    const runner = createRunner();
    const part: Part = {
      toolResponse: {
        name: 'do_something',
        output: 'done',
      },
    };

    assert.throws(() => {
      (runner as any).toAnthropicMessageContent(part);
    }, /Tool response ref is required/);
  });

  it('maps tool response into tool_result block containing text response', () => {
    const runner = createRunner();
    const part: Part = {
      toolResponse: {
        name: 'do_something',
        ref: 'tool-abc',
        output: 'done',
      },
    };

    const result = (runner as any).toAnthropicMessageContent(part);

    assert.strictEqual(result.type, 'tool_result');
    assert.strictEqual(result.tool_use_id, 'tool-abc');
    assert.deepStrictEqual(result.content, [{ type: 'text', text: 'done' }]);
  });

  it('should handle WEBP image data URLs', () => {
    const runner = createRunner();
    const part: Part = {
      media: {
        contentType: 'image/webp',
        url: 'data:image/webp;base64,AAA',
      },
    };

    const result = (runner as any).toAnthropicMessageContent(part);

    assert.strictEqual(result.type, 'image');
    assert.strictEqual(result.source.type, 'base64');
    assert.strictEqual(result.source.media_type, 'image/webp');
    assert.strictEqual(result.source.data, 'AAA');
  });

  it('should prefer data URL content type over media.contentType for WEBP', () => {
    const runner = createRunner();
    const part: Part = {
      media: {
        // Even if contentType says PNG, data URL says WEBP - should use WEBP
        contentType: 'image/png',
        url: 'data:image/webp;base64,AAA',
      },
    };

    const result = (runner as any).toAnthropicMessageContent(part);

    assert.strictEqual(result.type, 'image');
    assert.strictEqual(result.source.type, 'base64');
    // Key fix: should use data URL type (webp), not contentType (png)
    assert.strictEqual(result.source.media_type, 'image/webp');
    assert.strictEqual(result.source.data, 'AAA');
  });

  it('should throw helpful error for text/plain in toAnthropicMessageContent', () => {
    const runner = createRunner();
    const part: Part = {
      media: {
        contentType: 'text/plain',
        url: 'data:text/plain;base64,AAA',
      },
    };

    assert.throws(
      () => {
        (runner as any).toAnthropicMessageContent(part);
      },
      (error: Error) => {
        return (
          error.message.includes('Text files should be sent as text content') &&
          error.message.includes('text:')
        );
      }
    );
  });

  it('should throw helpful error for text/plain with remote URL', () => {
    const runner = createRunner();
    const part: Part = {
      media: {
        contentType: 'text/plain',
        url: 'https://example.com/file.txt',
      },
    };

    assert.throws(
      () => {
        (runner as any).toAnthropicMessageContent(part);
      },
      (error: Error) => {
        return (
          error.message.includes('Text files should be sent as text content') &&
          error.message.includes('text:')
        );
      }
    );
  });

  it('should throw helpful error for text/plain in tool response', () => {
    const runner = createRunner();
    const part: Part = {
      toolResponse: {
        ref: 'call_123',
        name: 'get_file',
        output: {
          url: 'data:text/plain;base64,AAA',
          contentType: 'text/plain',
        },
      },
    };

    assert.throws(
      () => {
        (runner as any).toAnthropicToolResponseContent(part);
      },
      (error: Error) => {
        return error.message.includes(
          'Text files should be sent as text content'
        );
      }
    );
  });
});
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

import { Anthropic } from '@anthropic-ai/sdk';
import { mock } from 'node:test';

describe('BetaRunner', () => {
  it('should map all supported Part shapes to beta content blocks', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const exposed = runner as any;

    const textPart = exposed.toAnthropicMessageContent({
      text: 'Hello',
    } as any);
    assert.deepStrictEqual(textPart, { type: 'text', text: 'Hello' });

    const pdfPart = exposed.toAnthropicMessageContent({
      media: {
        url: 'data:application/pdf;base64,JVBERi0xLjQKJ',
        contentType: 'application/pdf',
      },
    } as any);
    assert.strictEqual(pdfPart.type, 'document');

    const imagePart = exposed.toAnthropicMessageContent({
      media: {
        url: 'data:image/png;base64,AAA',
        contentType: 'image/png',
      },
    } as any);
    assert.strictEqual(imagePart.type, 'image');

    const toolUsePart = exposed.toAnthropicMessageContent({
      toolRequest: {
        ref: 'tool1',
        name: 'get_weather',
        input: { city: 'NYC' },
      },
    } as any);
    assert.deepStrictEqual(toolUsePart, {
      type: 'tool_use',
      id: 'tool1',
      name: 'get_weather',
      input: { city: 'NYC' },
    });

    const toolResultPart = exposed.toAnthropicMessageContent({
      toolResponse: {
        ref: 'tool1',
        name: 'get_weather',
        output: 'Sunny',
      },
    } as any);
    assert.strictEqual(toolResultPart.type, 'tool_result');

    assert.throws(() => exposed.toAnthropicMessageContent({} as any));
  });

  it('should convert beta stream events to Genkit Parts', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const exposed = runner as any;
    const textPart = exposed.toGenkitPart({
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: 'hi' },
    } as any);
    assert.deepStrictEqual(textPart, { text: 'hi' });

    const serverToolEvent = {
      type: 'content_block_start',
      index: 0,
      content_block: {
        type: 'server_tool_use',
        id: 'toolu_test',
        name: 'myTool',
        input: { foo: 'bar' },
        server_name: 'srv',
      },
    } as any;
    const toolPart = exposed.toGenkitPart(serverToolEvent);
    assert.deepStrictEqual(toolPart, {
      text: '[Anthropic server tool srv/myTool] input: {"foo":"bar"}',
      metadata: {
        anthropicServerToolUse: {
          id: 'toolu_test',
          name: 'srv/myTool',
          input: { foo: 'bar' },
        },
      },
    });

    const deltaPart = exposed.toGenkitPart({
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'thinking_delta', thinking: 'hmm' },
    } as any);
    assert.deepStrictEqual(deltaPart, { reasoning: 'hmm' });

    const ignored = exposed.toGenkitPart({ type: 'message_stop' } as any);
    assert.strictEqual(ignored, undefined);
  });

  it('should throw on unsupported mcp tool stream events', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const exposed = runner as any;
    assert.throws(
      () =>
        exposed.toGenkitPart({
          type: 'content_block_start',
          index: 0,
          content_block: {
            type: 'mcp_tool_use',
            id: 'toolu_unsupported',
            input: {},
          },
        }),
      /server-managed tool block 'mcp_tool_use'/
    );
  });

  it('should map beta stop reasons correctly', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const finishReason = runner['fromBetaStopReason'](
      'model_context_window_exceeded'
    );
    assert.strictEqual(finishReason, 'length');

    const pauseReason = runner['fromBetaStopReason']('pause_turn');
    assert.strictEqual(pauseReason, 'stop');
  });

  it('should execute streaming calls and surface errors', async () => {
    const streamError = new Error('stream failed');
    const mockClient = createMockAnthropicClient({
      streamChunks: [
        {
          type: 'content_block_start',
          index: 0,
          content_block: { type: 'text', text: 'hi' },
        } as any,
      ],
      streamErrorAfterChunk: 1,
      streamError,
    });

    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });
    const sendChunk = mock.fn();
    await assert.rejects(async () =>
      runner.run({ messages: [] } as any, {
        streamingRequested: true,
        sendChunk,
        abortSignal: new AbortController().signal,
      })
    );
    assert.strictEqual(sendChunk.mock.calls.length, 1);

    const abortController = new AbortController();
    abortController.abort();
    await assert.rejects(async () =>
      runner.run({ messages: [] } as any, {
        streamingRequested: true,
        sendChunk: () => {},
        abortSignal: abortController.signal,
      })
    );
  });

  it('should throw when tool refs are missing in message content', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });
    const exposed = runner as any;

    assert.throws(() =>
      exposed.toAnthropicMessageContent({
        toolRequest: {
          name: 'get_weather',
          input: {},
        },
      } as any)
    );

    assert.throws(() =>
      exposed.toAnthropicMessageContent({
        toolResponse: {
          name: 'get_weather',
          output: 'ok',
        },
      } as any)
    );

    assert.throws(() =>
      exposed.toAnthropicMessageContent({
        media: {
          url: 'data:image/png;base64,',
          contentType: undefined,
        },
      } as any)
    );
  });

  it('should build request bodies with optional config fields', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-3-5-haiku',
      client: mockClient as Anthropic,
    }) as any;

    const request = {
      messages: [
        {
          role: 'system',
          content: [{ text: 'You are helpful.' }],
        },
        {
          role: 'user',
          content: [{ text: 'Tell me a joke' }],
        },
      ],
      config: {
        maxOutputTokens: 128,
        topK: 4,
        topP: 0.65,
        temperature: 0.55,
        stopSequences: ['DONE'],
        metadata: { user_id: 'beta-user' },
        tool_choice: { type: 'tool', name: 'get_weather' },
        thinking: { enabled: true, budgetTokens: 2048 },
      },
      tools: [
        {
          name: 'get_weather',
          description: 'Returns the weather',
          inputSchema: { type: 'object' },
        },
      ],
    } satisfies any;

    const body = runner.toAnthropicRequestBody(
      'claude-3-5-haiku',
      request,
      true
    );

    assert.strictEqual(body.model, 'claude-3-5-haiku-20241022');
    assert.ok(Array.isArray(body.system));
    assert.strictEqual(body.max_tokens, 128);
    assert.strictEqual(body.top_k, 4);
    assert.strictEqual(body.top_p, 0.65);
    assert.strictEqual(body.temperature, 0.55);
    assert.deepStrictEqual(body.stop_sequences, ['DONE']);
    assert.deepStrictEqual(body.metadata, { user_id: 'beta-user' });
    assert.deepStrictEqual(body.tool_choice, {
      type: 'tool',
      name: 'get_weather',
    });
    assert.strictEqual(body.tools?.length, 1);
    assert.deepStrictEqual(body.thinking, {
      type: 'enabled',
      budget_tokens: 2048,
    });

    const streamingBody = runner.toAnthropicStreamingRequestBody(
      'claude-3-5-haiku',
      request,
      true
    );
    assert.strictEqual(streamingBody.stream, true);
    assert.ok(Array.isArray(streamingBody.system));
    assert.deepStrictEqual(streamingBody.thinking, {
      type: 'enabled',
      budget_tokens: 2048,
    });

    const disabledBody = runner.toAnthropicRequestBody(
      'claude-3-5-haiku',
      {
        messages: [],
        config: {
          thinking: { enabled: false },
        },
      } satisfies any,
      false
    );
    assert.deepStrictEqual(disabledBody.thinking, { type: 'disabled' });
  });

  it('should throw error if system message contains media', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-3-5-haiku',
      client: mockClient as Anthropic,
    }) as any;

    const request = {
      messages: [
        {
          role: 'system',
          content: [
            { text: 'You are a helpful assistant.' },
            {
              media: {
                url: 'data:image/png;base64,iVBORw0KGgoAAAANS',
                contentType: 'image/png',
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'Hi' }] },
      ],
      output: { format: 'text' },
    } satisfies any;

    assert.throws(
      () => runner.toAnthropicRequestBody('claude-3-5-haiku', request, false),
      /System messages can only contain text content/
    );
  });

  it('should throw error if system message contains tool requests', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-3-5-haiku',
      client: mockClient as Anthropic,
    }) as any;

    const request = {
      messages: [
        {
          role: 'system',
          content: [
            { text: 'You are a helpful assistant.' },
            { toolRequest: { name: 'getTool', input: {}, ref: '123' } },
          ],
        },
        { role: 'user', content: [{ text: 'Hi' }] },
      ],
      output: { format: 'text' },
    } satisfies any;

    assert.throws(
      () => runner.toAnthropicRequestBody('claude-3-5-haiku', request, false),
      /System messages can only contain text content/
    );
  });

  it('should throw error if system message contains tool responses', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-3-5-haiku',
      client: mockClient as Anthropic,
    }) as any;

    const request = {
      messages: [
        {
          role: 'system',
          content: [
            { text: 'You are a helpful assistant.' },
            { toolResponse: { name: 'getTool', output: {}, ref: '123' } },
          ],
        },
        { role: 'user', content: [{ text: 'Hi' }] },
      ],
      output: { format: 'text' },
    } satisfies any;

    assert.throws(
      () => runner.toAnthropicRequestBody('claude-3-5-haiku', request, false),
      /System messages can only contain text content/
    );
  });

  it('should not leak maxOutputTokens, stopSequences, version, or apiKey into the request body', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-3-5-haiku',
      client: mockClient as Anthropic,
    }) as any;

    const body = runner.toAnthropicRequestBody(
      'claude-3-5-haiku',
      {
        messages: [],
        config: {
          maxOutputTokens: 100,
          stopSequences: ['END'],
          version: 'claude-3-5-haiku-custom',
          apiKey: 'fake-api-key',
          temperature: 0.5,
        },
      } satisfies any,
      false
    );

    assert.deepStrictEqual(body, {
      model: 'claude-3-5-haiku-custom',
      max_tokens: 100,
      messages: [],
      stop_sequences: ['END'],
      temperature: 0.5,
      betas: [
        'files-api-2025-04-14',
        'effort-2025-11-24',
        'structured-outputs-2025-11-13',
      ],
    });
  });

  it('should throw for unsupported mcp tool use blocks', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });
    const exposed = runner as any;

    assert.throws(
      () =>
        exposed.fromBetaContentBlock({
          type: 'mcp_tool_use',
          id: 'toolu_unknown',
          input: {},
        }),
      /server-managed tool block 'mcp_tool_use'/
    );
  });

  it('should convert additional beta content block types', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const thinkingPart = (runner as any).fromBetaContentBlock({
      type: 'thinking',
      thinking: 'pondering',
      signature: 'sig_456',
    });
    assert.deepStrictEqual(thinkingPart, {
      reasoning: 'pondering',
      metadata: { thoughtSignature: 'sig_456' },
    });

    const redactedPart = (runner as any).fromBetaContentBlock({
      type: 'redacted_thinking',
      data: '[redacted]',
    });
    assert.deepStrictEqual(redactedPart, {
      custom: { redactedThinking: '[redacted]' },
    });

    const toolPart = (runner as any).fromBetaContentBlock({
      type: 'tool_use',
      id: 'toolu_x',
      name: 'plainTool',
      input: { value: 1 },
    });
    assert.deepStrictEqual(toolPart, {
      toolRequest: {
        ref: 'toolu_x',
        name: 'plainTool',
        input: { value: 1 },
      },
    });

    const serverToolPart = (runner as any).fromBetaContentBlock({
      type: 'server_tool_use',
      id: 'srv_tool_1',
      name: 'serverTool',
      input: { arg: 'value' },
      server_name: 'srv',
    });
    assert.deepStrictEqual(serverToolPart, {
      text: '[Anthropic server tool srv/serverTool] input: {"arg":"value"}',
      metadata: {
        anthropicServerToolUse: {
          id: 'srv_tool_1',
          name: 'srv/serverTool',
          input: { arg: 'value' },
        },
      },
    });

    const warnMock = mock.method(console, 'warn', () => {});
    const fallbackPart = (runner as any).fromBetaContentBlock({
      type: 'mystery',
    });
    assert.deepStrictEqual(fallbackPart, { text: '' });
    assert.strictEqual(warnMock.mock.calls.length, 1);
    warnMock.mock.restore();
  });

  it('should map additional stop reasons', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });
    const exposed = runner as any;

    const refusal = exposed.fromBetaStopReason('refusal');
    assert.strictEqual(refusal, 'other');

    const unknown = exposed.fromBetaStopReason('something-new');
    assert.strictEqual(unknown, 'other');

    const nullReason = exposed.fromBetaStopReason(null);
    assert.strictEqual(nullReason, 'unknown');
  });

  it('should convert citations_delta to citation part', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const citationPart = (runner as any).toGenkitPart({
      type: 'content_block_delta',
      index: 0,
      delta: {
        type: 'citations_delta',
        citation: {
          type: 'char_location',
          cited_text: 'The sky is blue.',
          document_index: 0,
          document_title: 'Facts',
          start_char_index: 0,
          end_char_index: 16,
        },
      },
    });

    assert.deepStrictEqual(citationPart, {
      text: '',
      metadata: {
        citations: [
          {
            type: 'char_location',
            citedText: 'The sky is blue.',
            documentIndex: 0,
            documentTitle: 'Facts',
            fileId: undefined,
            startCharIndex: 0,
            endCharIndex: 16,
          },
        ],
      },
    });
  });

  it('should convert text block with citations', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const textPart = (runner as any).fromBetaContentBlock({
      type: 'text',
      text: 'The answer is green.',
      citations: [
        {
          type: 'char_location',
          cited_text: 'grass is green',
          document_index: 0,
          document_title: 'Nature',
          start_char_index: 5,
          end_char_index: 19,
        },
      ],
    });

    assert.deepStrictEqual(textPart, {
      text: 'The answer is green.',
      metadata: {
        citations: [
          {
            type: 'char_location',
            citedText: 'grass is green',
            documentIndex: 0,
            documentTitle: 'Nature',
            fileId: undefined,
            startCharIndex: 5,
            endCharIndex: 19,
          },
        ],
      },
    });
  });

  it('should convert page_location citations from PDF documents', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const textPart = (runner as any).fromBetaContentBlock({
      type: 'text',
      text: 'According to the document...',
      citations: [
        {
          type: 'page_location',
          cited_text: 'Important finding',
          document_index: 0,
          document_title: 'Report.pdf',
          start_page_number: 1,
          end_page_number: 2,
        },
      ],
    });

    assert.deepStrictEqual(textPart, {
      text: 'According to the document...',
      metadata: {
        citations: [
          {
            type: 'page_location',
            citedText: 'Important finding',
            documentIndex: 0,
            documentTitle: 'Report.pdf',
            fileId: undefined,
            startPageNumber: 1,
            endPageNumber: 2,
          },
        ],
      },
    });
  });

  it('should convert content_block_location citations', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const textPart = (runner as any).fromBetaContentBlock({
      type: 'text',
      text: 'Based on the content...',
      citations: [
        {
          type: 'content_block_location',
          cited_text: 'Block content',
          document_index: 1,
          document_title: 'Structured Doc',
          start_block_index: 0,
          end_block_index: 2,
        },
      ],
    });

    assert.deepStrictEqual(textPart, {
      text: 'Based on the content...',
      metadata: {
        citations: [
          {
            type: 'content_block_location',
            citedText: 'Block content',
            documentIndex: 1,
            documentTitle: 'Structured Doc',
            fileId: undefined,
            startBlockIndex: 0,
            endBlockIndex: 2,
          },
        ],
      },
    });
  });

  it('should skip web search citations (no document_index)', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const textPart = (runner as any).fromBetaContentBlock({
      type: 'text',
      text: 'Search results say...',
      citations: [
        {
          type: 'web_search_result_location',
          cited_text: 'From the web',
          url: 'https://example.com',
          title: 'Example Page',
          // Note: no document_index - this is a web search citation
        },
      ],
    });

    // Should return text without citations metadata since web search citations are filtered
    assert.deepStrictEqual(textPart, {
      text: 'Search results say...',
    });
  });

  it('should convert anthropicDocument custom part to beta document block', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const docBlock = (runner as any).toAnthropicMessageContent({
      custom: {
        anthropicDocument: {
          source: {
            type: 'text',
            data: 'Document content here.',
          },
          title: 'My Document',
          citations: { enabled: true },
        },
      },
    });

    assert.strictEqual(docBlock.type, 'document');
    assert.strictEqual(docBlock.title, 'My Document');
    assert.deepStrictEqual(docBlock.citations, { enabled: true });
    assert.strictEqual(docBlock.source.type, 'text');
    assert.strictEqual(docBlock.source.data, 'Document content here.');
  });

  it('should convert file source in beta document block', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const docBlock = (runner as any).toAnthropicMessageContent({
      custom: {
        anthropicDocument: {
          source: {
            type: 'file',
            fileId: 'file-abc123',
          },
          title: 'Uploaded File',
        },
      },
    });

    assert.strictEqual(docBlock.type, 'document');
    assert.strictEqual(docBlock.source.type, 'file');
    assert.strictEqual(docBlock.source.file_id, 'file-abc123');
  });

  it('should convert URL source in beta document block', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const docBlock = (runner as any).toAnthropicMessageContent({
      custom: {
        anthropicDocument: {
          source: {
            type: 'url',
            url: 'https://example.com/doc.pdf',
          },
        },
      },
    });

    assert.strictEqual(docBlock.type, 'document');
    assert.strictEqual(docBlock.source.type, 'url');
    assert.strictEqual(docBlock.source.url, 'https://example.com/doc.pdf');
  });

  it('should convert base64 source in beta document block', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const docBlock = (runner as any).toAnthropicMessageContent({
      custom: {
        anthropicDocument: {
          source: {
            type: 'base64',
            data: 'JVBERi0xLjQ=',
            mediaType: 'application/pdf',
          },
        },
      },
    });

    assert.strictEqual(docBlock.type, 'document');
    assert.strictEqual(docBlock.source.type, 'base64');
    assert.strictEqual(docBlock.source.data, 'JVBERi0xLjQ=');
    assert.strictEqual(docBlock.source.media_type, 'application/pdf');
  });

  it('should convert content source in beta document block', () => {
    const mockClient = createMockAnthropicClient();
    const runner = new BetaRunner({
      name: 'claude-test',
      client: mockClient as Anthropic,
    });

    const docBlock = (runner as any).toAnthropicMessageContent({
      custom: {
        anthropicDocument: {
          source: {
            type: 'content',
            content: [
              { type: 'text', text: 'First block' },
              { type: 'text', text: 'Second block' },
            ],
          },
          citations: { enabled: true },
        },
      },
    });

    assert.strictEqual(docBlock.type, 'document');
    assert.strictEqual(docBlock.source.type, 'content');
    assert.strictEqual(docBlock.source.content.length, 2);
    assert.strictEqual(docBlock.source.content[0].text, 'First block');
  });
});
