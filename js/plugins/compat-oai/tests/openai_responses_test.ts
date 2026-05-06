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

import { afterEach, describe, expect, it, jest } from '@jest/globals';
import type { GenerateRequest } from 'genkit';
import OpenAI, { APIError } from 'openai';
import type { Response } from 'openai/resources/responses/responses';
import {
  chatMessagesToResponsesInput,
  toResponsesRequestBody,
} from '../src/openai/responses/request';
import { fromResponsesResponse } from '../src/openai/responses/response';
import { openAIResponsesModelRunner } from '../src/openai/responses/runner';
import {
  OpenAIResponsesConfigSchema,
  SUPPORTED_RESPONSES_MODELS,
} from '../src/openai/responses/types';

jest.mock('genkit/model', () => {
  const originalModule =
    jest.requireActual<typeof import('genkit/model')>('genkit/model');
  return {
    ...originalModule,
    defineModel: jest.fn((_, runner) => runner),
  };
});

afterEach(() => {
  jest.clearAllMocks();
});

/** Build a minimal `Response` object honoring required SDK fields. */
function buildResponse(overrides: Partial<Response>): Response {
  return {
    id: overrides.id ?? 'resp_test_1',
    created_at: 1700000000,
    output_text: '',
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    model: overrides.model ?? 'gpt-5-mini',
    object: 'response',
    output: overrides.output ?? [],
    parallel_tool_calls: true,
    temperature: null,
    tool_choice: 'auto',
    tools: [],
    top_p: null,
    ...(overrides.status
      ? { status: overrides.status }
      : { status: 'completed' }),
    ...overrides,
  } as Response;
}

describe('toResponsesRequestBody', () => {
  it('case 1 — plain text generation defaults to store=false and no tools', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config: {},
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.model).toBe('gpt-5-mini');
    expect(body.store).toBe(false);
    expect(body.tools).toBeUndefined();
    expect(body.input).toEqual([
      { type: 'message', role: 'user', content: 'hi' },
    ]);
  });

  it('case 2 — JSON mode via output.schema produces text.format json_schema', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'list' }] }],
      output: {
        format: 'json',
        schema: { type: 'object', properties: { x: { type: 'number' } } },
      },
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.text?.format).toEqual({
      type: 'json_schema',
      name: 'output',
      schema: { type: 'object', properties: { x: { type: 'number' } } },
      strict: true,
    });
  });

  it('case 3 — function tool is mapped to tools[] with type=function', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'use tool' }] }],
      tools: [
        {
          name: 'lookup_user',
          description: 'Lookup a user by id',
          inputSchema: {
            type: 'object',
            properties: { id: { type: 'string' } },
          },
          outputSchema: { type: 'object' },
        },
      ],
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.tools).toEqual([
      {
        type: 'function',
        name: 'lookup_user',
        description: 'Lookup a user by id',
        parameters: {
          type: 'object',
          properties: { id: { type: 'string' } },
        },
        strict: false,
      },
    ]);
  });

  it('case 4 — built-in web_search_preview tool is appended after function tools', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'news today?' }] }],
      config: {
        builtInTools: [
          { type: 'web_search_preview', searchContextSize: 'high' },
        ],
      },
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.tools).toEqual([
      { type: 'web_search_preview', search_context_size: 'high' },
    ]);
  });

  it('case 5 — file_search built-in tool maps vectorStoreIds + maxNumResults', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'find' }] }],
      config: {
        builtInTools: [
          {
            type: 'file_search',
            vectorStoreIds: ['vs_1', 'vs_2'],
            maxNumResults: 5,
          },
        ],
      },
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.tools).toEqual([
      {
        type: 'file_search',
        vector_store_ids: ['vs_1', 'vs_2'],
        max_num_results: 5,
      },
    ]);
  });

  it('case 6 — reasoning effort + summary plumbed into reasoning field', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'think' }] }],
      config: { reasoning: { effort: 'medium', summary: 'auto' } },
    };
    const body = toResponsesRequestBody('o3', request);
    expect(body.reasoning).toEqual({ effort: 'medium', summary: 'auto' });
  });

  it('case 7 — previousResponseId is forwarded as previous_response_id', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'continue' }] }],
      config: { previousResponseId: 'resp_prev_1' },
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.previous_response_id).toBe('resp_prev_1');
  });

  it('case 8 — leading system messages are lifted into instructions', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [
        { role: 'system', content: [{ text: 'You are concise.' }] },
        { role: 'user', content: [{ text: 'Hi' }] },
      ],
    };
    const body = toResponsesRequestBody('o3', request);
    expect(body.instructions).toBe('You are concise.');
    // System message dropped from input.
    expect(body.input).toEqual([
      { type: 'message', role: 'user', content: 'Hi' },
    ]);
  });

  it('case 9 — tool-result message becomes function_call_output input item', () => {
    const items = chatMessagesToResponsesInput([
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              ref: 'call_1',
              name: 'lookup_user',
              output: { name: 'Ada' },
            },
          },
        ],
      },
    ]);
    expect(items).toEqual([
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: JSON.stringify({ name: 'Ada' }),
      },
    ]);
  });

  it('case 9b — undefined toolResponse.output falls back to "{}"', () => {
    const items = chatMessagesToResponsesInput([
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              ref: 'call_1',
              name: 'noop',
              output: undefined,
            },
          },
        ],
      },
    ]);
    // The Responses API requires `output` to be a string; we must NOT
    // emit `output: undefined` (which JSON serialization drops, leaving
    // the field absent and the body invalid).
    expect(items).toEqual([
      {
        type: 'function_call_output',
        call_id: 'call_1',
        output: '{}',
      },
    ]);
  });

  it('case 10 — explicit instructions override system-message lift', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [
        { role: 'system', content: [{ text: 'ignored' }] },
        { role: 'user', content: [{ text: 'go' }] },
      ],
      config: { instructions: 'use this' },
    };
    const body = toResponsesRequestBody('o3', request);
    expect(body.instructions).toBe('use this');
    // System message stays in input because we did not lift it.
    expect(body.input.length).toBe(2);
  });
});

describe('fromResponsesResponse', () => {
  it('plain text response with usage → text Part + finishReason stop', () => {
    const response = buildResponse({
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [
            { type: 'output_text', text: 'hello world', annotations: [] },
          ],
        },
      ],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        total_tokens: 15,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 0 },
      },
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.finishReason).toBe('stop');
    expect(data.message?.content).toEqual([{ text: 'hello world' }]);
    expect(data.usage).toMatchObject({
      inputTokens: 10,
      outputTokens: 5,
      totalTokens: 15,
    });
    expect(data.custom).toMatchObject({ responseId: 'resp_test_1' });
  });

  it('url citations land on text Part metadata.citations', () => {
    const response = buildResponse({
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [
            {
              type: 'output_text',
              text: 'Per ACME news today...',
              annotations: [
                {
                  type: 'url_citation',
                  url: 'https://acme.example.com/news/1',
                  title: 'ACME News 1',
                  start_index: 4,
                  end_index: 8,
                },
              ],
            },
          ],
        },
      ],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    const textPart = data.message!.content[0];
    expect(textPart.text).toBe('Per ACME news today...');
    expect(textPart.metadata?.citations).toEqual([
      {
        type: 'url_citation',
        url: 'https://acme.example.com/news/1',
        title: 'ACME News 1',
        startIndex: 4,
        endIndex: 8,
      },
    ]);
  });

  it('refusal output → finishReason blocked + finishMessage', () => {
    const response = buildResponse({
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'refusal', refusal: 'I cannot help with that.' }],
        },
      ],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.finishReason).toBe('blocked');
    expect(data.finishMessage).toBe('I cannot help with that.');
  });

  it('function_call output → toolRequest Part with parsed JSON args', () => {
    const response = buildResponse({
      output: [
        {
          id: 'fc_1',
          type: 'function_call',
          call_id: 'call_42',
          name: 'lookup_user',
          arguments: '{"id":"u_1"}',
          status: 'completed',
        },
      ],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.message?.content).toEqual([
      {
        toolRequest: {
          name: 'lookup_user',
          ref: 'call_42',
          input: { id: 'u_1' },
        },
      },
    ]);
  });

  it('reasoning item → reasoning Part with summary text', () => {
    const response = buildResponse({
      output: [
        {
          id: 'rsn_1',
          type: 'reasoning',
          summary: [{ type: 'summary_text', text: 'thought a + thought b' }],
        },
      ],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.message?.content).toEqual([
      { reasoning: 'thought a + thought b' },
    ]);
  });

  it('incomplete with max_output_tokens → finishReason length', () => {
    const response = buildResponse({
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'incomplete',
          content: [{ type: 'output_text', text: 'partial', annotations: [] }],
        },
      ],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.finishReason).toBe('length');
  });
});

describe('openAIResponsesModelRunner', () => {
  function fakeClient() {
    return {
      responses: {
        create: jest.fn(async (_body: unknown) =>
          buildResponse({
            output: [
              {
                id: 'msg_1',
                type: 'message',
                role: 'assistant',
                status: 'completed',
                content: [
                  {
                    type: 'output_text',
                    text: 'mocked',
                    annotations: [],
                  },
                ],
              },
            ],
          })
        ),
      },
    } as unknown as OpenAI;
  }

  it('passes the built request body to client.responses.create', async () => {
    const client = fakeClient();
    const runner = openAIResponsesModelRunner('gpt-5-mini', client);
    const data = await runner({
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config: {},
    });
    expect(client.responses.create).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gpt-5-mini',
        input: [{ type: 'message', role: 'user', content: 'hi' }],
        store: false,
      }),
      { signal: undefined }
    );
    expect(data.message?.content[0].text).toBe('mocked');
  });

  it('maps APIError 429 → RESOURCE_EXHAUSTED GenkitError', async () => {
    const client = {
      responses: {
        create: jest.fn(async () => {
          const err = new APIError(
            429,
            { error: { message: 'Rate limited' } },
            'Rate limited',
            {} as never
          );
          throw err;
        }),
      },
    } as unknown as OpenAI;
    const runner = openAIResponsesModelRunner('gpt-5-mini', client);
    await expect(
      runner({
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        config: {},
      })
    ).rejects.toMatchObject({ status: 'RESOURCE_EXHAUSTED' });
  });

  it('wraps non-APIError into GenkitError INTERNAL', async () => {
    const client = {
      responses: {
        create: jest.fn(async () => {
          throw new TypeError('network blew up');
        }),
      },
    } as unknown as OpenAI;
    const runner = openAIResponsesModelRunner('gpt-5-mini', client);
    await expect(
      runner({
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        config: {},
      })
    ).rejects.toMatchObject({
      status: 'INTERNAL',
      message: expect.stringContaining('network blew up'),
    });
  });

  it('aborted request → GenkitError CANCELLED', async () => {
    const ac = new AbortController();
    const client = {
      responses: {
        create: jest.fn(async () => {
          ac.abort();
          const e = new Error('aborted');
          e.name = 'AbortError';
          throw e;
        }),
      },
    } as unknown as OpenAI;
    const runner = openAIResponsesModelRunner('gpt-5-mini', client);
    await expect(
      runner(
        {
          messages: [{ role: 'user', content: [{ text: 'hi' }] }],
          config: {},
        },
        { abortSignal: ac.signal }
      )
    ).rejects.toMatchObject({ status: 'CANCELLED' });
  });
});

describe('toResponsesRequestBody — invariants & edge cases', () => {
  it('throws when toolResponse is missing ref', () => {
    expect(() =>
      chatMessagesToResponsesInput([
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'lookup_user',
                output: { name: 'Ada' },
              } as never,
            },
          ],
        },
      ])
    ).toThrow(/missing 'ref'/);
  });

  it('throws when toolRequest is missing ref', () => {
    expect(() =>
      chatMessagesToResponsesInput([
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'lookup_user',
                input: { id: 'u_1' },
              } as never,
            },
          ],
        },
      ])
    ).toThrow(/missing 'ref'/);
  });

  it('keeps system message in input array when it carries non-text media', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [
        {
          role: 'system',
          content: [
            { text: 'see also' },
            {
              media: {
                url: 'data:image/png;base64,AAAA',
                contentType: 'image/png',
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'go' }] },
      ],
    };
    const body = toResponsesRequestBody('o3', request);
    // System message NOT lifted because it has non-text content.
    expect(body.instructions).toBeUndefined();
    expect(body.input.length).toBe(2);
    const first = body.input[0] as { type: string; role: string };
    expect(first.type).toBe('message');
    expect(first.role).toBe('system');
  });

  it('code_interpreter container as explicit string id is passed through', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'run' }] }],
      config: {
        builtInTools: [
          { type: 'code_interpreter', container: 'cnt_explicit_123' },
        ],
      },
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.tools).toEqual([
      { type: 'code_interpreter', container: 'cnt_explicit_123' },
    ]);
  });

  it('code_interpreter with auto+fileIds maps to {type:auto, file_ids}', () => {
    const request: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'run' }] }],
      config: {
        builtInTools: [
          {
            type: 'code_interpreter',
            container: { type: 'auto', fileIds: ['file_1', 'file_2'] },
          },
        ],
      },
    };
    const body = toResponsesRequestBody('gpt-5-mini', request);
    expect(body.tools).toEqual([
      {
        type: 'code_interpreter',
        container: { type: 'auto', file_ids: ['file_1', 'file_2'] },
      },
    ]);
  });

  it('file_search ranking_options requires scoreThreshold; ranker alone is omitted', () => {
    const onlyRanker: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'q' }] }],
      config: {
        builtInTools: [
          {
            type: 'file_search',
            vectorStoreIds: ['vs_1'],
            ranker: { ranker: 'auto' },
          },
        ],
      },
    };
    const body = toResponsesRequestBody('gpt-5-mini', onlyRanker);
    const tool = body.tools![0] as unknown as Record<string, unknown>;
    expect(tool.ranking_options).toBeUndefined();
  });

  it('file_search ranking_options included when scoreThreshold is set', () => {
    const withThreshold: GenerateRequest<typeof OpenAIResponsesConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: 'q' }] }],
      config: {
        builtInTools: [
          {
            type: 'file_search',
            vectorStoreIds: ['vs_1'],
            ranker: { ranker: 'auto', scoreThreshold: 0.5 },
          },
        ],
      },
    };
    const body = toResponsesRequestBody('gpt-5-mini', withThreshold);
    const tool = body.tools![0] as unknown as Record<string, unknown>;
    expect(tool.ranking_options).toEqual({
      score_threshold: 0.5,
      ranker: 'auto',
    });
  });
});

describe('fromResponsesResponse — invariants & edge cases', () => {
  it('file_citation annotation maps to {type:file_citation, fileId, fileIndex}', () => {
    const response = buildResponse({
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [
            {
              type: 'output_text',
              text: 'See doc.',
              annotations: [
                {
                  type: 'file_citation',
                  file_id: 'file_abc',
                  index: 2,
                } as never,
              ],
            },
          ],
        },
      ],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.message!.content[0].metadata?.citations).toEqual([
      { type: 'file_citation', fileId: 'file_abc', fileIndex: 2 },
    ]);
  });

  it('function_call with malformed JSON args sets metadata.malformedArguments', () => {
    const response = buildResponse({
      output: [
        {
          id: 'fc_1',
          type: 'function_call',
          call_id: 'call_42',
          name: 'lookup_user',
          arguments: 'not-json',
          status: 'completed',
        },
      ],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.message?.content).toEqual([
      {
        toolRequest: {
          name: 'lookup_user',
          ref: 'call_42',
          input: 'not-json',
        },
        metadata: { malformedArguments: true },
      },
    ]);
  });

  it('status:failed surfaces error.code on custom + warns', () => {
    const response = buildResponse({
      status: 'failed',
      error: { code: 'rate_limit_exceeded', message: 'Slow down' } as never,
      output: [],
    });
    const data = fromResponsesResponse(response, { messages: [] });
    expect(data.finishReason).toBe('other');
    expect(data.finishMessage).toBe('Slow down');
    expect(data.custom).toMatchObject({ errorCode: 'rate_limit_exceeded' });
  });
});

describe('SUPPORTED_RESPONSES_MODELS — model info', () => {
  it('reasoning models advertise systemRole: true so core does not pre-transform system messages', () => {
    // The plugin handles system-message lifting itself
    // (toResponsesRequestBody → instructions). If we advertised
    // systemRole: false, Genkit core would convert system → user
    // before our resolver runs, defeating the lift.
    const o3 = SUPPORTED_RESPONSES_MODELS['o3'];
    expect(o3.info?.supports?.systemRole).toBe(true);
    const o4mini = SUPPORTED_RESPONSES_MODELS['o4-mini'];
    expect(o4mini.info?.supports?.systemRole).toBe(true);
  });
});
