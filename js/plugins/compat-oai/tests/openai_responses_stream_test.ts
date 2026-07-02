/**
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

import { describe, expect, it } from '@jest/globals';
import type { GenerateResponseChunkData, StreamingCallback } from 'genkit';
import { streamResponsesEvents } from '../src/openai/responses/stream';

/**
 * Build an async iterable from a static array — mirrors what the OpenAI
 * SDK's `client.responses.stream(...)` returns for testing.
 */
async function* iter<T>(events: T[]): AsyncIterable<T> {
  for (const event of events) {
    yield event;
  }
}

/** Capture all chunks emitted during a stream run. */
function collector() {
  const chunks: GenerateResponseChunkData[] = [];
  const sendChunk: StreamingCallback<GenerateResponseChunkData> = (chunk) => {
    chunks.push(chunk);
  };
  return { chunks, sendChunk };
}

describe('streamResponsesEvents', () => {
  it('aggregates output_text deltas in arrival order', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        { type: 'response.created', response: {}, sequence_number: 0 },
        {
          type: 'response.output_item.added',
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          },
          output_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.output_text.delta',
          delta: 'Hello',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          sequence_number: 2,
        },
        {
          type: 'response.output_text.delta',
          delta: ' world',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          sequence_number: 3,
        },
        {
          type: 'response.output_text.done',
          text: 'Hello world',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          sequence_number: 4,
        },
        {
          type: 'response.output_item.done',
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [
              {
                type: 'output_text',
                text: 'Hello world',
                annotations: [],
              },
            ],
          },
          output_index: 0,
          sequence_number: 5,
        },
        {
          type: 'response.completed',
          response: {},
          sequence_number: 6,
        },
      ] as never),
      sendChunk
    );

    const textChunks = chunks
      .map((c) => (c.content?.[0] as { text?: string })?.text)
      .filter((t): t is string => typeof t === 'string');
    expect(textChunks).toEqual(['Hello', ' world']);
  });

  it('attaches url citations as a final metadata-only chunk', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        {
          type: 'response.output_item.added',
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          },
          output_index: 0,
          sequence_number: 0,
        },
        {
          type: 'response.output_text.delta',
          delta: 'See ACME news.',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.output_text_annotation.added',
          annotation: {
            type: 'url_citation',
            url: 'https://acme.example.com',
            title: 'ACME',
            start_index: 4,
            end_index: 8,
          },
          annotation_index: 0,
          content_index: 0,
          item_id: 'msg_1',
          output_index: 0,
          sequence_number: 2,
        },
        {
          type: 'response.output_item.done',
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [],
          },
          output_index: 0,
          sequence_number: 3,
        },
      ] as never),
      sendChunk
    );

    const finalChunk = chunks[chunks.length - 1];
    const part = finalChunk.content?.[0] as
      | {
          text?: string;
          metadata?: { citations?: unknown[] };
        }
      | undefined;
    expect(part?.metadata?.citations).toEqual([
      {
        type: 'url_citation',
        url: 'https://acme.example.com',
        title: 'ACME',
        startIndex: 4,
        endIndex: 8,
      },
    ]);
  });

  it('aggregates function_call_arguments deltas into a single toolRequest chunk on item.done', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        {
          type: 'response.output_item.added',
          item: {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_42',
            name: 'lookup_user',
            arguments: '',
            status: 'in_progress',
          },
          output_index: 0,
          sequence_number: 0,
        },
        {
          type: 'response.function_call_arguments.delta',
          delta: '{"id":',
          item_id: 'fc_1',
          output_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.function_call_arguments.delta',
          delta: '"u_1"}',
          item_id: 'fc_1',
          output_index: 0,
          sequence_number: 2,
        },
        {
          type: 'response.function_call_arguments.done',
          arguments: '{"id":"u_1"}',
          item_id: 'fc_1',
          output_index: 0,
          sequence_number: 3,
        },
        {
          type: 'response.output_item.done',
          item: {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_42',
            name: 'lookup_user',
            arguments: '{"id":"u_1"}',
            status: 'completed',
          },
          output_index: 0,
          sequence_number: 4,
        },
      ] as never),
      sendChunk
    );

    // No partial chunks for arguments — only the final aggregated
    // toolRequest chunk on item.done.
    const toolChunks = chunks.filter(
      (c) => (c.content?.[0] as { toolRequest?: unknown })?.toolRequest != null
    );
    expect(toolChunks).toHaveLength(1);
    expect(
      (
        toolChunks[0].content?.[0] as {
          toolRequest: {
            name: string;
            ref: string;
            input: unknown;
          };
        }
      ).toolRequest
    ).toEqual({ name: 'lookup_user', ref: 'call_42', input: { id: 'u_1' } });
  });

  it('emits progress chunks for built-in tool call lifecycle events', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        {
          type: 'response.web_search_call.in_progress',
          item_id: 'wsc_1',
          output_index: 0,
          sequence_number: 0,
        },
        {
          type: 'response.web_search_call.searching',
          item_id: 'wsc_1',
          output_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.web_search_call.completed',
          item_id: 'wsc_1',
          output_index: 0,
          sequence_number: 2,
        },
      ] as never),
      sendChunk
    );

    const progress = chunks
      .map(
        (c) =>
          (
            c.content?.[0] as {
              custom?: { kind?: string; status?: string };
            }
          )?.custom
      )
      .filter((c): c is { kind: string; status: string } => c != null);
    expect(progress).toEqual([
      { kind: 'web_search_call', status: 'in_progress', itemId: 'wsc_1' },
      { kind: 'web_search_call', status: 'searching', itemId: 'wsc_1' },
      { kind: 'web_search_call', status: 'completed', itemId: 'wsc_1' },
    ]);
  });

  it('aggregates function_call args even when output_item.added is missed', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        // No response.output_item.added for output_index 0.
        {
          type: 'response.function_call_arguments.delta',
          delta: '{"id":',
          item_id: 'fc_1',
          output_index: 0,
          sequence_number: 0,
        },
        {
          type: 'response.function_call_arguments.delta',
          delta: '"u_1"}',
          item_id: 'fc_1',
          output_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.output_item.done',
          item: {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_42',
            name: 'lookup_user',
            arguments: '{"id":"u_1"}',
            status: 'completed',
          },
          output_index: 0,
          sequence_number: 2,
        },
      ] as never),
      sendChunk
    );
    const toolChunks = chunks.filter(
      (c) => (c.content?.[0] as { toolRequest?: unknown })?.toolRequest != null
    );
    expect(toolChunks).toHaveLength(1);
    expect(
      (
        toolChunks[0].content?.[0] as {
          toolRequest: {
            name: string;
            ref: string;
            input: unknown;
          };
        }
      ).toolRequest
    ).toEqual({ name: 'lookup_user', ref: 'call_42', input: { id: 'u_1' } });
  });

  it('marks malformed JSON args with metadata.malformedArguments=true', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        {
          type: 'response.output_item.added',
          item: {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_42',
            name: 'lookup_user',
            arguments: '',
            status: 'in_progress',
          },
          output_index: 0,
          sequence_number: 0,
        },
        {
          type: 'response.function_call_arguments.delta',
          delta: 'not-json-at-all',
          item_id: 'fc_1',
          output_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.output_item.done',
          item: {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_42',
            name: 'lookup_user',
            arguments: 'not-json-at-all',
            status: 'completed',
          },
          output_index: 0,
          sequence_number: 2,
        },
      ] as never),
      sendChunk
    );
    const last = chunks[chunks.length - 1];
    const part = last.content?.[0] as {
      toolRequest?: { input?: unknown };
      metadata?: { malformedArguments?: boolean };
    };
    expect(part.toolRequest?.input).toBe('not-json-at-all');
    expect(part.metadata?.malformedArguments).toBe(true);
  });

  it('throws GenkitError when stream emits an error event', async () => {
    const { sendChunk } = collector();
    await expect(
      streamResponsesEvents(
        iter([
          {
            type: 'response.output_item.added',
            item: {
              id: 'msg_1',
              type: 'message',
              role: 'assistant',
              status: 'in_progress',
              content: [],
            },
            output_index: 0,
            sequence_number: 0,
          },
          {
            type: 'error',
            message: 'upstream blew up',
            code: 'server_error',
            sequence_number: 1,
          },
        ] as never),
        sendChunk
      )
    ).rejects.toMatchObject({
      status: 'INTERNAL',
      message: expect.stringContaining('upstream blew up'),
    });
  });

  it('refusal.delta emits custom-only chunk (no empty text Part)', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        {
          type: 'response.refusal.delta',
          delta: 'I cannot.',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          sequence_number: 0,
        },
      ] as never),
      sendChunk
    );
    expect(chunks).toHaveLength(1);
    const part = chunks[0].content?.[0] as {
      text?: string;
      custom?: { refusalDelta?: string };
    };
    expect(part.text).toBeUndefined();
    expect(part.custom?.refusalDelta).toBe('I cannot.');
  });

  it('attaches file_citation annotations as discriminated metadata', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        {
          type: 'response.output_item.added',
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          },
          output_index: 0,
          sequence_number: 0,
        },
        {
          type: 'response.output_text.delta',
          delta: 'See file.',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.output_text_annotation.added',
          annotation: {
            type: 'file_citation',
            file_id: 'file_abc',
            index: 1,
          },
          annotation_index: 0,
          content_index: 0,
          item_id: 'msg_1',
          output_index: 0,
          sequence_number: 2,
        },
        {
          type: 'response.output_item.done',
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'completed',
            content: [],
          },
          output_index: 0,
          sequence_number: 3,
        },
      ] as never),
      sendChunk
    );
    const finalChunk = chunks[chunks.length - 1];
    const part = finalChunk.content?.[0] as {
      metadata?: { citations?: unknown[] };
    };
    expect(part?.metadata?.citations).toEqual([
      { type: 'file_citation', fileId: 'file_abc', fileIndex: 1 },
    ]);
  });

  it('forwards reasoning summary deltas as reasoning Parts', async () => {
    const { chunks, sendChunk } = collector();
    await streamResponsesEvents(
      iter([
        {
          type: 'response.output_item.added',
          item: {
            id: 'rsn_1',
            type: 'reasoning',
            summary: [],
          },
          output_index: 0,
          sequence_number: 0,
        },
        {
          type: 'response.reasoning_summary_text.delta',
          delta: 'thinking step 1',
          item_id: 'rsn_1',
          output_index: 0,
          summary_index: 0,
          sequence_number: 1,
        },
        {
          type: 'response.reasoning_summary_text.delta',
          delta: ' / step 2',
          item_id: 'rsn_1',
          output_index: 0,
          summary_index: 0,
          sequence_number: 2,
        },
      ] as never),
      sendChunk
    );

    const reasoning = chunks
      .map((c) => (c.content?.[0] as { reasoning?: string })?.reasoning)
      .filter((r): r is string => typeof r === 'string');
    expect(reasoning).toEqual(['thinking step 1', ' / step 2']);
  });
});
