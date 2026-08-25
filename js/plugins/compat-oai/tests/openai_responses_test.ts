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

import {
  afterAll,
  beforeAll,
  describe,
  expect,
  jest,
  test,
} from '@jest/globals';
import { genkit, z, type GenerateRequest } from 'genkit';
import type { ModelAction } from 'genkit/model';
import OpenAI, { APIError } from 'openai';
import type { Response as OpenAIResponse } from 'openai/resources/responses/responses.mjs';
import { openAIModelRunner } from '../src/model';
import { openAI } from '../src/openai/index';
import {
  NON_STREAMING_RESPONSES_MODELS,
  RESPONSES_ONLY_MODELS,
  isNonStreamingResponsesModelName,
  isResponsesOnlyModelName,
  openAIResponsesModelRef,
} from '../src/openai/responses';
import {
  defineCompatOpenAIResponsesModel,
  fromOpenAIResponse,
  isReasoningModelName,
  openAIResponsesModelRunner,
  toOpenAIResponsesRequestBody,
} from '../src/responses';
import { FakeOpenAIServer } from './fake_openai_server';

/** Builds a minimal Response object with the given output items. */
function fakeResponse(overrides: Partial<OpenAIResponse> = {}): OpenAIResponse {
  return {
    id: 'resp_1',
    created_at: 0,
    output_text: '',
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    model: 'gpt-5-pro',
    object: 'response',
    output: [],
    parallel_tool_calls: false,
    temperature: null,
    tool_choice: 'auto',
    tools: [],
    top_p: null,
    status: 'completed',
    ...overrides,
  };
}

/** Builds a Response whose only output item is an assistant text message. */
function textResponse(text: string): OpenAIResponse {
  return fakeResponse({
    output: [
      {
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [{ type: 'output_text', text, annotations: [] }],
      },
    ],
  });
}

describe('isResponsesOnlyModelName', () => {
  test('matches every curated base name exactly', () => {
    for (const name of RESPONSES_ONLY_MODELS) {
      expect(isResponsesOnlyModelName(name)).toBe(true);
    }
  });

  test('matches any suffixed form of a base name', () => {
    expect(isResponsesOnlyModelName('o3-pro-2025-06-10')).toBe(true);
    expect(isResponsesOnlyModelName('gpt-5-pro-2025-10-06')).toBe(true);
    expect(isResponsesOnlyModelName('gpt-5.1-codex-max-2026-01-01')).toBe(true);
    expect(isResponsesOnlyModelName('gpt-5-pro-preview')).toBe(true);
    expect(isResponsesOnlyModelName('o3-pro-anything')).toBe(true);
  });

  test('does not match dual-transport or unrelated models', () => {
    expect(isResponsesOnlyModelName('gpt-5')).toBe(false);
    expect(isResponsesOnlyModelName('o3')).toBe(false);
    expect(isResponsesOnlyModelName('o3-mini')).toBe(false);
    expect(isResponsesOnlyModelName('o1')).toBe(false);
    expect(isResponsesOnlyModelName('gpt-5-mini')).toBe(false);
    expect(isResponsesOnlyModelName('gpt-5.1')).toBe(false);
    expect(isResponsesOnlyModelName('gpt-4o')).toBe(false);
    expect(isResponsesOnlyModelName('')).toBe(false);
    expect(isResponsesOnlyModelName(undefined)).toBe(false);
  });

  test('covers the codex models the model filter now lets through', () => {
    expect(isResponsesOnlyModelName('codex-mini-latest')).toBe(true);
    expect(isResponsesOnlyModelName('gpt-5-codex')).toBe(true);
    expect(isResponsesOnlyModelName('gpt-5.1-codex')).toBe(true);
    expect(isResponsesOnlyModelName('gpt-5.1-codex-mini')).toBe(true);
    expect(isResponsesOnlyModelName('gpt-5.1-codex-max')).toBe(true);
  });
});

describe('isNonStreamingResponsesModelName', () => {
  test('matches the curated non-streaming models and their suffixed forms', () => {
    for (const name of NON_STREAMING_RESPONSES_MODELS) {
      expect(isNonStreamingResponsesModelName(name)).toBe(true);
    }
    expect(isNonStreamingResponsesModelName('o3-pro-2025-06-10')).toBe(true);
  });

  test('leaves the streaming-capable models alone', () => {
    expect(isNonStreamingResponsesModelName('gpt-5-pro')).toBe(false);
    expect(isNonStreamingResponsesModelName('gpt-5-codex')).toBe(false);
    expect(isNonStreamingResponsesModelName('codex-mini-latest')).toBe(false);
    expect(isNonStreamingResponsesModelName(undefined)).toBe(false);
  });
});

describe('isReasoningModelName', () => {
  test('matches the reasoning families', () => {
    expect(isReasoningModelName('o1-pro')).toBe(true);
    expect(isReasoningModelName('o3')).toBe(true);
    expect(isReasoningModelName('o4-mini')).toBe(true);
    expect(isReasoningModelName('gpt-5-nano')).toBe(true);
    expect(isReasoningModelName('gpt-5.3-codex')).toBe(true);
    expect(isReasoningModelName('codex-mini-latest')).toBe(true);
  });

  test('leaves non-reasoning models out', () => {
    expect(isReasoningModelName('gpt-4o')).toBe(false);
    expect(isReasoningModelName('gpt-4o-mini')).toBe(false);
    expect(isReasoningModelName('gpt-5-chat-latest')).toBe(false);
    expect(isReasoningModelName('')).toBe(false);
    expect(isReasoningModelName(undefined)).toBe(false);
  });
});

describe('toOpenAIResponsesRequestBody', () => {
  test('hoists system messages into instructions and keeps the rest as input', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        { role: 'system', content: [{ text: 'be terse' }] },
        { role: 'user', content: [{ text: 'hi' }] },
        { role: 'model', content: [{ text: 'hello' }] },
        { role: 'user', content: [{ text: 'bye' }] },
      ],
    });

    expect(body.instructions).toBe('be terse');
    expect(body.input).toStrictEqual([
      { role: 'user', content: [{ type: 'input_text', text: 'hi' }] },
      { role: 'assistant', content: 'hello' },
      { role: 'user', content: [{ type: 'input_text', text: 'bye' }] },
    ]);
  });

  test('replays a structured-output model turn instead of an empty message', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        { role: 'user', content: [{ text: 'name a colour' }] },
        { role: 'model', content: [{ data: { colour: 'blue' } }] },
        { role: 'user', content: [{ text: 'another one' }] },
      ],
      output: { format: 'json' },
    });

    expect(body.input).toStrictEqual([
      {
        role: 'user',
        content: [{ type: 'input_text', text: 'name a colour' }],
      },
      { role: 'assistant', content: '{"colour":"blue"}' },
      { role: 'user', content: [{ type: 'input_text', text: 'another one' }] },
    ]);
  });

  test('preserves order across a mixed text and data model turn', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'model',
          content: [
            { text: 'here you go: ' },
            { data: { colour: 'blue' } },
            { text: ' (done)' },
          ],
        },
      ],
    });

    expect(body.input).toStrictEqual([
      {
        role: 'assistant',
        content: 'here you go: {"colour":"blue"} (done)',
      },
    ]);
  });

  test('skips reasoning parts in history and drops turns left empty', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'model',
          content: [{ reasoning: 'thinking' }, { text: 'answer' }],
        },
        { role: 'model', content: [{ reasoning: 'thinking harder' }] },
      ],
    });

    expect(body.input).toStrictEqual([
      { role: 'assistant', content: 'answer' },
    ]);
  });

  test('rejects model-turn parts it cannot replay', () => {
    expect(() =>
      toOpenAIResponsesRequestBody('gpt-5-pro', {
        messages: [
          {
            role: 'model',
            content: [{ media: { url: 'https://example.com/cat.png' } }],
          },
        ],
      })
    ).toThrow(/Unsupported genkit part fields/);
  });

  test('joins multiple system messages and omits instructions when there are none', () => {
    const withSystem = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        { role: 'system', content: [{ text: 'one' }] },
        { role: 'system', content: [{ text: 'two' }] },
      ],
    });
    expect(withSystem.instructions).toBe('one\n\ntwo');

    const withoutSystem = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
    });
    expect(withoutSystem).not.toHaveProperty('instructions');
  });

  test('composes the output format over a raw text config passthrough', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config: { text: { verbosity: 'low' } },
      output: { format: 'json' },
    });

    expect(body.text).toStrictEqual({
      verbosity: 'low',
      format: { type: 'json_object' },
    });
  });

  test('rejects stream in config instead of passing it to the wire', () => {
    expect(() =>
      toOpenAIResponsesRequestBody('gpt-5-pro', {
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        config: { stream: true },
      })
    ).toThrow(expect.objectContaining({ status: 'INVALID_ARGUMENT' }));
  });

  test('rejects background in config instead of passing it to the wire', () => {
    expect(() =>
      toOpenAIResponsesRequestBody('gpt-5-pro', {
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        config: { background: true },
      })
    ).toThrow(expect.objectContaining({ status: 'INVALID_ARGUMENT' }));
  });

  test('joins config instructions with system messages instead of clobbering them', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [{ role: 'system', content: [{ text: 'You are X' }] }],
      config: { instructions: 'formatting hint' },
    });

    expect(body.instructions).toBe('You are X\n\nformatting hint');
  });

  test('maps generation config onto Responses API field names', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: {
        version: 'gpt-5-pro-2025-10-06',
        temperature: 0.5,
        topP: 0.9,
        maxOutputTokens: 128,
      },
    });

    expect(body).toStrictEqual({
      model: 'gpt-5-pro-2025-10-06',
      input: [],
      max_output_tokens: 128,
      temperature: 0.5,
      top_p: 0.9,
      include: ['reasoning.encrypted_content'],
      store: false,
    });
  });

  test('drops config keys the Responses API has no equivalent for', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { topK: 3, stopSequences: ['stop'], visualDetailLevel: 'low' },
    });

    expect(body).not.toHaveProperty('topK');
    expect(body).not.toHaveProperty('stopSequences');
    expect(body).not.toHaveProperty('stop');
    expect(body).not.toHaveProperty('visualDetailLevel');
  });

  test('never serializes the transport routing key', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { transport: 'responses', apiKey: 'secret' },
    });

    expect(body).not.toHaveProperty('transport');
    expect(body).not.toHaveProperty('apiKey');
    expect(JSON.stringify(body)).not.toContain('transport');
  });

  test('rejects a transport this model cannot speak', () => {
    expect(() =>
      toOpenAIResponsesRequestBody('gpt-5-pro', {
        messages: [],
        config: { transport: 'chat_completions' },
      })
    ).toThrow(
      expect.objectContaining({
        status: 'INVALID_ARGUMENT',
        message: expect.stringContaining('chat_completions'),
      })
    );
  });

  test('pins store to false unless the caller sets it', () => {
    expect(
      toOpenAIResponsesRequestBody('gpt-5-pro', { messages: [] }).store
    ).toBe(false);
    expect(
      toOpenAIResponsesRequestBody('gpt-5-pro', {
        messages: [],
        config: { store: true },
      }).store
    ).toBe(true);
  });

  test('passes unrecognized config keys through to the body', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: {
        reasoning: { effort: 'high' },
        tools: [{ type: 'web_search_preview' }],
      },
    });

    expect(body.reasoning).toStrictEqual({ effort: 'high' });
    expect(body.tools).toStrictEqual([{ type: 'web_search_preview' }]);
  });

  test('maps json output with a schema onto text.format', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      output: {
        format: 'json',
        schema: { type: 'object', properties: { a: { type: 'string' } } },
      },
    });

    expect(body.text).toStrictEqual({
      format: {
        type: 'json_schema',
        name: 'output',
        // The Responses API validates schemas under strict mode by default,
        // which genkit schemas do not satisfy.
        strict: false,
        schema: { type: 'object', properties: { a: { type: 'string' } } },
      },
    });
  });

  test('maps schemaless json output onto json_object', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      output: { format: 'json' },
    });

    expect(body.text).toStrictEqual({ format: { type: 'json_object' } });
  });

  test('maps text output onto text.format', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      output: { format: 'text' },
    });

    expect(body.text).toStrictEqual({ format: { type: 'text' } });
  });

  test('converts media parts into Responses input content', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'user',
          content: [
            { media: { url: 'https://example.com/cat.png' } },
            {
              media: {
                url: 'data:application/pdf;base64,QUJD',
                contentType: 'application/pdf',
              },
            },
          ],
        },
      ],
      config: { visualDetailLevel: 'high' },
    });

    expect(body.input).toStrictEqual([
      {
        role: 'user',
        content: [
          {
            type: 'input_image',
            detail: 'high',
            image_url: 'https://example.com/cat.png',
          },
          {
            type: 'input_file',
            filename: 'file.pdf',
            file_data: 'data:application/pdf;base64,QUJD',
          },
        ],
      },
    ]);
  });

  test('converts genkit tools into flat function tools', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      tools: [
        {
          name: 'lookup',
          description: 'looks things up',
          inputSchema: {
            type: 'object',
            properties: { q: { type: 'string' } },
          },
        },
      ],
      toolChoice: 'required',
    });

    expect(body.tools).toStrictEqual([
      {
        type: 'function',
        name: 'lookup',
        description: 'looks things up',
        parameters: { type: 'object', properties: { q: { type: 'string' } } },
        strict: null,
      },
    ]);
    expect(body.tool_choice).toBe('required');
  });

  test('merges config.tools with converted genkit tools', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      tools: [{ name: 'lookup', description: 'looks things up' }],
      config: { tools: [{ type: 'web_search_preview' }] },
    });

    expect(body.tools).toHaveLength(2);
    expect(body.tools![0]).toMatchObject({ type: 'function', name: 'lookup' });
    expect(body.tools![1]).toStrictEqual({ type: 'web_search_preview' });
  });

  test('maps tool responses onto function_call_output items', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'tool',
          content: [
            { toolResponse: { name: 'f', ref: 'call_1', output: 'done' } },
            {
              toolResponse: { name: 'g', ref: 'call_2', output: { ok: true } },
            },
          ],
        },
      ],
    });

    expect(body.input).toStrictEqual([
      { type: 'function_call_output', call_id: 'call_1', output: 'done' },
      {
        type: 'function_call_output',
        call_id: 'call_2',
        output: '{"ok":true}',
      },
    ]);
  });

  test('serializes a void tool response instead of dropping the output key', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'f', ref: 'call_1' } }],
        },
      ],
    });

    expect(body.input).toStrictEqual([
      { type: 'function_call_output', call_id: 'call_1', output: 'null' },
    ]);
    expect(JSON.parse(JSON.stringify(body.input))[0]).toHaveProperty('output');
  });

  test('replays tool calls and encrypted reasoning in order', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'model',
          content: [
            {
              reasoning: 'thinking',
              metadata: { itemId: 'rs_1', encryptedContent: 'ENC' },
            },
            { reasoning: 'more', metadata: { itemId: 'rs_1' } },
            {
              toolRequest: { name: 'lookup', ref: 'call_1', input: { q: 'x' } },
              metadata: { itemId: 'fc_1' },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            { toolResponse: { name: 'lookup', ref: 'call_1', output: 'done' } },
          ],
        },
      ],
    });

    expect(body.input).toStrictEqual([
      {
        type: 'reasoning',
        id: 'rs_1',
        summary: [
          { type: 'summary_text', text: 'thinking' },
          { type: 'summary_text', text: 'more' },
        ],
        encrypted_content: 'ENC',
      },
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'lookup',
        arguments: '{"q":"x"}',
        id: 'fc_1',
      },
      { type: 'function_call_output', call_id: 'call_1', output: 'done' },
    ]);
  });

  test('replays an encrypted reasoning item that has no summary', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'model',
          content: [
            {
              reasoning: '',
              metadata: { itemId: 'rs_1', encryptedContent: 'ENC' },
            },
            { text: 'answer' },
          ],
        },
      ],
    });

    expect(body.input).toStrictEqual([
      { type: 'reasoning', id: 'rs_1', summary: [], encrypted_content: 'ENC' },
      { role: 'assistant', content: 'answer' },
    ]);
  });

  test('preserves text around a tool call instead of dropping it', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [
        {
          role: 'model',
          content: [
            { text: 'let me check' },
            { toolRequest: { name: 'lookup', ref: 'call_1', input: {} } },
          ],
        },
      ],
    });

    expect(body.input).toStrictEqual([
      { role: 'assistant', content: 'let me check' },
      {
        type: 'function_call',
        call_id: 'call_1',
        name: 'lookup',
        arguments: '{}',
      },
    ]);
  });

  test('requests encrypted reasoning on stateless requests', () => {
    expect(
      toOpenAIResponsesRequestBody('gpt-5-pro', { messages: [] }).include
    ).toStrictEqual(['reasoning.encrypted_content']);
  });

  test('never requests encrypted reasoning from a non-reasoning model', () => {
    // Non-reasoning models reject the include with a 400, so a dual-transport
    // opt-in (gpt-4o over Responses) must not carry it.
    const body = toOpenAIResponsesRequestBody('gpt-4o', { messages: [] });

    expect(body).not.toHaveProperty('include');
  });

  test('gates encrypted reasoning on the version when one is set', () => {
    const body = toOpenAIResponsesRequestBody('gpt-4o', {
      messages: [],
      config: { version: 'o3-pro-2025-06-10' },
    });

    expect(body.include).toStrictEqual(['reasoning.encrypted_content']);
  });

  test('merges caller include values without duplicating', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { include: ['reasoning.encrypted_content'] },
    });

    expect(body.include).toStrictEqual(['reasoning.encrypted_content']);
  });

  test('normalizes a bare-string include instead of spreading its characters', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { include: 'message.output_text.logprobs' },
    });

    expect(body.include).toStrictEqual([
      'message.output_text.logprobs',
      'reasoning.encrypted_content',
    ]);
  });

  test('leaves include alone when the caller opts into server-side storage', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { store: true, include: ['message.output_text.logprobs'] },
    });

    expect(body.include).toStrictEqual(['message.output_text.logprobs']);
  });

  test('drops declared chat-schema keys that have no Responses equivalent', () => {
    const body = toOpenAIResponsesRequestBody('gpt-4o', {
      messages: [],
      config: {
        transport: 'responses',
        frequencyPenalty: 0.5,
        presencePenalty: 0.5,
        logProbs: true,
        topLogProbs: 3,
      },
    });

    expect(body).not.toHaveProperty('frequencyPenalty');
    expect(body).not.toHaveProperty('presencePenalty');
    expect(body).not.toHaveProperty('logProbs');
    expect(body).not.toHaveProperty('topLogProbs');
  });

  test('lets the declared previousResponseId win over a raw passthrough key', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: {
        previousResponseId: 'resp_declared',
        previous_response_id: 'resp_raw',
      },
    });

    expect(body.previous_response_id).toBe('resp_declared');
  });

  test('keeps a raw previous_response_id passthrough when no declared key is set', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { previous_response_id: 'resp_raw' },
    });

    expect(body.previous_response_id).toBe('resp_raw');
  });

  test('maps previousResponseId onto the wire field', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { previousResponseId: 'resp_prev' },
    });

    expect(body.previous_response_id).toBe('resp_prev');
    expect(body).not.toHaveProperty('previousResponseId');
  });

  test('maps reasoning effort and summary into the reasoning object', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: { reasoningEffort: 'low', reasoningSummary: 'auto' },
    });

    expect(body.reasoning).toStrictEqual({ effort: 'low', summary: 'auto' });
    expect(body).not.toHaveProperty('reasoningEffort');
    expect(body).not.toHaveProperty('reasoningSummary');
  });

  test('composes declared reasoning fields over a raw passthrough object', () => {
    const body = toOpenAIResponsesRequestBody('gpt-5-pro', {
      messages: [],
      config: {
        reasoningSummary: 'detailed',
        reasoning: { effort: 'high', summary: 'auto' },
      },
    });

    expect(body.reasoning).toStrictEqual({
      effort: 'high',
      summary: 'detailed',
    });
  });
});

describe('fromOpenAIResponse', () => {
  test('converts output text, usage and finish reason', () => {
    const response = textResponse('hello');
    response.usage = {
      input_tokens: 10,
      output_tokens: 4,
      total_tokens: 14,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens_details: { reasoning_tokens: 0 },
    };

    expect(fromOpenAIResponse(response)).toStrictEqual({
      finishReason: 'stop',
      message: { role: 'model', content: [{ text: 'hello' }] },
      usage: {
        inputTokens: 10,
        outputTokens: 4,
        totalTokens: 14,
        thoughtsTokens: 0,
        cachedContentTokens: 0,
      },
      raw: response,
    });
  });

  test('maps reasoning and cached token details into usage', () => {
    const response = textResponse('hello');
    response.usage = {
      input_tokens: 10,
      output_tokens: 20,
      total_tokens: 30,
      input_tokens_details: { cached_tokens: 4 },
      output_tokens_details: { reasoning_tokens: 15 },
    };

    expect(fromOpenAIResponse(response).usage).toStrictEqual({
      inputTokens: 10,
      outputTokens: 20,
      totalTokens: 30,
      thoughtsTokens: 15,
      cachedContentTokens: 4,
    });
  });

  test('parses output text as data in json mode', () => {
    const converted = fromOpenAIResponse(textResponse('{"a":1}'), true);
    expect(converted.message?.content).toStrictEqual([{ data: { a: 1 } }]);
  });

  test('maps reasoning summaries to reasoning parts, preserving order', () => {
    const response = fakeResponse({
      output: [
        {
          id: 'rs_1',
          type: 'reasoning',
          summary: [
            { type: 'summary_text', text: 'thinking' },
            { type: 'summary_text', text: 'more' },
          ],
        },
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'answer', annotations: [] }],
        },
      ],
    });

    expect(fromOpenAIResponse(response).message?.content).toStrictEqual([
      { reasoning: 'thinking', metadata: { itemId: 'rs_1' } },
      { reasoning: 'more', metadata: { itemId: 'rs_1' } },
      { text: 'answer' },
    ]);
  });

  test('carries encrypted reasoning on the item first part only', () => {
    const response = fakeResponse({
      output: [
        {
          id: 'rs_1',
          type: 'reasoning',
          encrypted_content: 'ENC',
          summary: [
            { type: 'summary_text', text: 'thinking' },
            { type: 'summary_text', text: 'more' },
          ],
        },
      ],
    });

    expect(fromOpenAIResponse(response).message?.content).toStrictEqual([
      {
        reasoning: 'thinking',
        metadata: { itemId: 'rs_1', encryptedContent: 'ENC' },
      },
      { reasoning: 'more', metadata: { itemId: 'rs_1' } },
    ]);
  });

  test('keeps a summary-less encrypted reasoning item as an empty part', () => {
    const response = fakeResponse({
      output: [
        {
          id: 'rs_1',
          type: 'reasoning',
          encrypted_content: 'ENC',
          summary: [],
        },
      ],
    });

    expect(fromOpenAIResponse(response).message?.content).toStrictEqual([
      {
        reasoning: '',
        metadata: { itemId: 'rs_1', encryptedContent: 'ENC' },
      },
    ]);
  });

  test('drops a reasoning item with neither summary nor encrypted content', () => {
    const response = fakeResponse({
      output: [{ id: 'rs_1', type: 'reasoning', summary: [] }],
    });

    expect(fromOpenAIResponse(response).message?.content).toStrictEqual([]);
  });

  test('surfaces citation annotations on the text part', () => {
    const annotation = {
      type: 'url_citation' as const,
      url: 'https://example.com',
      title: 'Example',
      start_index: 0,
      end_index: 5,
    };
    const response = fakeResponse({
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [
            { type: 'output_text', text: 'cited', annotations: [annotation] },
          ],
        },
      ],
    });

    expect(fromOpenAIResponse(response).message?.content).toStrictEqual([
      { text: 'cited', metadata: { annotations: [annotation] } },
    ]);
  });

  test('maps a refusal to a blocked finish reason', () => {
    const response = fakeResponse({
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'refusal', refusal: 'I cannot help with that' }],
        },
      ],
    });

    const converted = fromOpenAIResponse(response);
    expect(converted.finishReason).toBe('blocked');
    expect(converted.message?.content).toStrictEqual([
      { text: 'I cannot help with that' },
    ]);
  });

  test('maps incomplete_details.reason to a finish reason', () => {
    expect(
      fromOpenAIResponse(
        fakeResponse({
          status: 'incomplete',
          incomplete_details: { reason: 'max_output_tokens' },
        })
      ).finishReason
    ).toBe('length');

    expect(
      fromOpenAIResponse(
        fakeResponse({
          status: 'incomplete',
          incomplete_details: { reason: 'content_filter' },
        })
      ).finishReason
    ).toBe('blocked');
  });

  test('converts a function call into a tool request part', () => {
    const response = fakeResponse({
      output: [
        {
          id: 'fc_1',
          type: 'function_call',
          call_id: 'call_1',
          name: 'lookup',
          arguments: '{"q":"x"}',
        },
      ],
    });

    const converted = fromOpenAIResponse(response);
    expect(converted.finishReason).toBe('stop');
    expect(converted.message?.content).toStrictEqual([
      {
        toolRequest: { name: 'lookup', ref: 'call_1', input: { q: 'x' } },
        metadata: { itemId: 'fc_1' },
      },
    ]);
  });

  test('survives truncated function-call arguments and reports length', () => {
    const response = fakeResponse({
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [
        {
          id: 'fc_1',
          type: 'function_call',
          call_id: 'call_1',
          name: 'lookup',
          arguments: '{"q":',
          status: 'incomplete',
        },
      ],
    });

    const converted = fromOpenAIResponse(response);
    expect(converted.finishReason).toBe('length');
    const part = converted.message?.content[0];
    expect(part?.toolRequest?.name).toBe('lookup');
    expect(part?.toolRequest?.ref).toBe('call_1');
  });

  test('rejects a custom tool call', () => {
    const item = {
      id: 'ct_1',
      type: 'custom_tool_call',
      call_id: 'call_1',
      name: 'grep',
      input: 'foo',
    } as unknown as OpenAIResponse['output'][number];
    const response = fakeResponse({ output: [item] });

    expect(() => fromOpenAIResponse(response)).toThrow(
      expect.objectContaining({
        status: 'UNIMPLEMENTED',
        message: expect.stringContaining('custom_tool_call'),
      })
    );
  });

  test('skips records of tools OpenAI ran itself', () => {
    const response = fakeResponse({
      output: [
        { id: 'ws_1', type: 'web_search_call', status: 'completed' },
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'searched', annotations: [] }],
        },
      ],
    });

    const converted = fromOpenAIResponse(response);
    expect(converted.finishReason).toBe('stop');
    expect(converted.message?.content).toStrictEqual([{ text: 'searched' }]);
  });

  test('maps non-terminal statuses', () => {
    expect(
      fromOpenAIResponse(fakeResponse({ status: 'in_progress' })).finishReason
    ).toBe('unknown');
  });

  test('rejects a failed response even without an error payload', () => {
    expect(() =>
      fromOpenAIResponse(fakeResponse({ status: 'failed' }))
    ).toThrow(
      expect.objectContaining({
        status: 'INTERNAL',
        message: expect.stringContaining('without an error payload'),
      })
    );
  });

  test('survives truncated json output and reports length', () => {
    const response = fakeResponse({
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [
        {
          id: 'msg_1',
          type: 'message',
          role: 'assistant',
          status: 'incomplete',
          content: [
            { type: 'output_text', text: '{"colour":', annotations: [] },
          ],
        },
      ],
    });

    const converted = fromOpenAIResponse(response, true);
    expect(converted.finishReason).toBe('length');
    expect(converted.message?.content).toHaveLength(1);
  });

  test('surfaces the error of a failed response', () => {
    const response = fakeResponse({
      status: 'failed',
      error: { code: 'server_error', message: 'upstream exploded' },
    });

    expect(() => fromOpenAIResponse(response)).toThrow(
      expect.objectContaining({
        status: 'INTERNAL',
        message: expect.stringContaining('upstream exploded'),
      })
    );
  });

  test('explains a cancelled or unexplained incomplete response', () => {
    expect(
      fromOpenAIResponse(fakeResponse({ status: 'cancelled' })).finishMessage
    ).toBe('Response cancelled.');

    const incomplete = fromOpenAIResponse(
      fakeResponse({ status: 'incomplete', incomplete_details: {} })
    );
    expect(incomplete.finishReason).toBe('other');
    expect(incomplete.finishMessage).toBe(
      'Response incomplete: no reason given.'
    );
  });
});

describe('openAIResponsesModelRunner', () => {
  let server: FakeOpenAIServer;

  beforeAll(async () => {
    server = new FakeOpenAIServer();
    await server.start();
  });

  afterAll(() => {
    server.stop();
  });

  test('posts to the Responses endpoint with a stateless body', async () => {
    server.setNextResponse({ body: textResponse('hi there') });
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIResponsesModelRunner('gpt-5-pro', client);

    const response = await runner({
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
    });

    const request = server.requests[server.requests.length - 1];
    expect(request.url).toBe('/v1/responses');
    expect(request.body).toStrictEqual({
      model: 'gpt-5-pro',
      input: [{ role: 'user', content: [{ type: 'input_text', text: 'hi' }] }],
      include: ['reasoning.encrypted_content'],
      store: false,
    });
    expect(response.message?.content).toStrictEqual([{ text: 'hi there' }]);
  });

  test('answers a streaming caller without streaming when the model cannot stream', async () => {
    server.setNextResponse({ body: textResponse('non-streamed') });
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIResponsesModelRunner('o3-pro', client, undefined, {
      streaming: false,
    });
    const sendChunk = jest.fn();

    const response = await runner(
      { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
      { streamingRequested: true, sendChunk }
    );

    const request = server.requests[server.requests.length - 1];
    expect(request.body).not.toHaveProperty('stream');
    expect(sendChunk).toHaveBeenCalledTimes(1);
    expect(sendChunk).toHaveBeenCalledWith({
      content: [{ text: 'non-streamed' }],
    });
    expect(response.message?.content).toStrictEqual([{ text: 'non-streamed' }]);
  });

  test('streams text and reasoning deltas as chunks', async () => {
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          type: 'response.created',
          response: fakeResponse(),
          sequence_number: 0,
        },
        {
          type: 'response.output_item.added',
          output_index: 0,
          item: { id: 'rs_1', type: 'reasoning', summary: [] },
          sequence_number: 1,
        },
        {
          type: 'response.reasoning_summary_text.delta',
          item_id: 'rs_1',
          output_index: 0,
          summary_index: 0,
          delta: 'thinking',
          sequence_number: 2,
        },
        {
          type: 'response.output_item.added',
          output_index: 1,
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          },
          sequence_number: 3,
        },
        {
          type: 'response.content_part.added',
          item_id: 'msg_1',
          output_index: 1,
          content_index: 0,
          part: { type: 'output_text', text: '', annotations: [] },
          sequence_number: 4,
        },
        {
          type: 'response.output_text.delta',
          item_id: 'msg_1',
          output_index: 1,
          content_index: 0,
          delta: 'hel',
          sequence_number: 5,
        },
        {
          type: 'response.output_text.delta',
          item_id: 'msg_1',
          output_index: 1,
          content_index: 0,
          delta: 'lo',
          sequence_number: 6,
        },
        {
          type: 'response.completed',
          response: textResponse('hello'),
          sequence_number: 7,
        },
      ],
    });
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIResponsesModelRunner('gpt-5-pro', client);
    const sendChunk = jest.fn();

    const response = await runner(
      { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
      { streamingRequested: true, sendChunk }
    );

    expect(sendChunk.mock.calls.map((call) => call[0])).toStrictEqual([
      { content: [{ reasoning: 'thinking' }] },
      { content: [{ text: 'hel' }] },
      { content: [{ text: 'lo' }] },
    ]);
    expect(response.message?.content).toStrictEqual([{ text: 'hello' }]);
    expect(server.requests[server.requests.length - 1].body.stream).toBe(true);
  });

  test('streams a function call as accumulating partial tool requests', async () => {
    const finalResponse = fakeResponse({
      output: [
        {
          id: 'fc_1',
          type: 'function_call',
          call_id: 'call_1',
          name: 'lookup',
          arguments: '{"q":"x"}',
        },
      ],
    });
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          type: 'response.created',
          response: fakeResponse(),
          sequence_number: 0,
        },
        {
          type: 'response.output_item.added',
          output_index: 0,
          item: {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_1',
            name: 'lookup',
            arguments: '',
            status: 'in_progress',
          },
          sequence_number: 1,
        },
        {
          type: 'response.function_call_arguments.delta',
          item_id: 'fc_1',
          output_index: 0,
          delta: '{"q":',
          sequence_number: 2,
        },
        {
          type: 'response.function_call_arguments.delta',
          item_id: 'fc_1',
          output_index: 0,
          delta: '"x"}',
          sequence_number: 3,
        },
        {
          type: 'response.completed',
          response: finalResponse,
          sequence_number: 4,
        },
      ],
    });
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIResponsesModelRunner('gpt-5-pro', client);
    const sendChunk = jest.fn();

    const response = await runner(
      { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
      { streamingRequested: true, sendChunk }
    );

    const chunks = sendChunk.mock.calls.map((call) => call[0]);
    expect(chunks[0]).toStrictEqual({
      content: [
        {
          toolRequest: {
            name: 'lookup',
            ref: 'call_1',
            input: {},
            partial: true,
          },
        },
      ],
    });
    expect(chunks[chunks.length - 1]).toStrictEqual({
      content: [
        {
          toolRequest: {
            name: 'lookup',
            ref: 'call_1',
            input: { q: 'x' },
            partial: true,
          },
        },
      ],
    });
    expect(response.message?.content).toStrictEqual([
      {
        toolRequest: { name: 'lookup', ref: 'call_1', input: { q: 'x' } },
        metadata: { itemId: 'fc_1' },
      },
    ]);
  });

  test('reports truncation from a stream that ends incomplete', async () => {
    const incompleteResponse = fakeResponse({
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      usage: {
        input_tokens: 10,
        output_tokens: 4,
        total_tokens: 14,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 0 },
      },
      output: [
        {
          id: 'fc_1',
          type: 'function_call',
          call_id: 'call_1',
          name: 'lookup',
          arguments: '{"q":',
          status: 'incomplete',
        },
      ],
    });
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          type: 'response.created',
          response: fakeResponse(),
          sequence_number: 0,
        },
        {
          type: 'response.output_item.added',
          output_index: 0,
          item: {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_1',
            name: 'lookup',
            arguments: '',
            status: 'in_progress',
          },
          sequence_number: 1,
        },
        {
          type: 'response.function_call_arguments.delta',
          item_id: 'fc_1',
          output_index: 0,
          delta: '{"q":',
          sequence_number: 2,
        },
        {
          type: 'response.incomplete',
          response: incompleteResponse,
          sequence_number: 3,
        },
      ],
    });
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIResponsesModelRunner('gpt-5-pro', client);

    const response = await runner(
      { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
      { streamingRequested: true, sendChunk: jest.fn() }
    );

    expect(response.finishReason).toBe('length');
    expect(response.usage?.totalTokens).toBe(14);
  });

  test('surfaces a failure from a stream that ends failed', async () => {
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          type: 'response.created',
          response: fakeResponse(),
          sequence_number: 0,
        },
        {
          type: 'response.failed',
          response: fakeResponse({
            status: 'failed',
            error: { code: 'server_error', message: 'upstream exploded' },
          }),
          sequence_number: 1,
        },
      ],
    });
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIResponsesModelRunner('gpt-5-pro', client);

    await expect(
      runner(
        { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
        { streamingRequested: true, sendChunk: jest.fn() }
      )
    ).rejects.toThrow(
      expect.objectContaining({
        status: 'INTERNAL',
        message: expect.stringContaining('upstream exploded'),
      })
    );
  });

  test('surfaces an SSE error event as a GenkitError', async () => {
    // The SDK throws an APIError the moment it sees an `event: error` frame,
    // so the event never reaches the runner's loop; this pins that the
    // rejection still comes out as a GenkitError.
    server.setNextResponse({
      stream: true,
      rawSse: [
        `data: ${JSON.stringify({
          type: 'response.created',
          response: fakeResponse(),
          sequence_number: 0,
        })}\n\n`,
        `event: error\ndata: ${JSON.stringify({
          type: 'error',
          code: 'server_error',
          message: 'stream blew up',
          param: null,
          sequence_number: 1,
        })}\n\n`,
      ],
    });
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIResponsesModelRunner('gpt-5-pro', client);

    await expect(
      runner(
        { messages: [{ role: 'user', content: [{ text: 'hi' }] }] },
        { streamingRequested: true, sendChunk: jest.fn() }
      )
    ).rejects.toThrow(
      expect.objectContaining({
        name: 'GenkitError',
        message: expect.stringContaining('stream blew up'),
      })
    );
  });

  test('converts an APIError into a GenkitError', async () => {
    const client = {
      responses: {
        create: jest.fn(async () => {
          throw new APIError(
            429,
            { error: { message: 'Rate limit exceeded' } },
            '',
            {}
          );
        }),
      },
    };
    const runner = openAIResponsesModelRunner(
      'gpt-5-pro',
      client as unknown as OpenAI
    );

    await expect(runner({ messages: [] })).rejects.toThrow(
      expect.objectContaining({ status: 'RESOURCE_EXHAUSTED' })
    );
  });
});

describe('defineCompatOpenAIResponsesModel', () => {
  test('declares the Responses model info and config schema', () => {
    const action = defineCompatOpenAIResponsesModel({
      name: 'openai/gpt-5-pro',
      client: {} as OpenAI,
      modelRef: openAIResponsesModelRef({ name: 'gpt-5-pro' }),
    });

    expect(action.__action.name).toBe('openai/gpt-5-pro');
    expect(action.__action.metadata?.model.supports).toStrictEqual({
      multiturn: true,
      tools: true,
      toolChoice: true,
      media: true,
      systemRole: true,
      output: ['text', 'json'],
      constrained: 'all',
    });
  });
});

describe('openAI plugin routing', () => {
  let server: FakeOpenAIServer;
  let previousBaseUrl: string | undefined;

  beforeAll(async () => {
    server = new FakeOpenAIServer();
    await server.start();
    // The openAI plugin does not accept a baseURL, so the fake server is
    // injected the way a user would point the SDK at a proxy.
    previousBaseUrl = process.env.OPENAI_BASE_URL;
    process.env.OPENAI_BASE_URL = server.baseUrl;
  });

  afterAll(() => {
    if (previousBaseUrl === undefined) {
      delete process.env.OPENAI_BASE_URL;
    } else {
      process.env.OPENAI_BASE_URL = previousBaseUrl;
    }
    server.stop();
  });

  test('resolves Responses-only names through the Responses runner', async () => {
    const plugin = openAI({ apiKey: 'key' });
    const action = await plugin.model('o3-pro-2025-06-10');

    expect(action.__action.name).toBe('openai/o3-pro-2025-06-10');
    // Only the Responses config schema declares the transport routing key.
    expect(
      Object.keys(action.__action.metadata?.model.customOptions.properties)
    ).toContain('transport');
  });

  test('declares the transport key on dual-transport chat models', async () => {
    const plugin = openAI({ apiKey: 'key' });
    const action = await plugin.model('gpt-5');

    expect(
      Object.keys(action.__action.metadata?.model.customOptions.properties)
    ).toContain('transport');
  });

  test('gives Responses-only refs the transport-aware config schema', () => {
    const ref = openAI.model('gpt-5-pro');
    const keys = Object.keys(ref.configSchema!.shape);

    expect(keys).toContain('transport');
    expect(keys).toContain('store');
    // Neither has a Responses API equivalent, so the schema must not offer them.
    expect(keys).not.toContain('topK');
    expect(keys).not.toContain('stopSequences');
  });

  test('registers every curated Responses-only model', async () => {
    const plugin = openAI({ apiKey: 'key' });
    const registered = (await plugin.init!()).map(
      (action) => (action as ModelAction).__action.name
    );

    for (const name of RESPONSES_ONLY_MODELS) {
      expect(registered).toContain(`openai/${name}`);
    }
  });

  test('streams a non-streaming model through the plugin without stream: true', async () => {
    const ai = genkit({ plugins: [openAI({ apiKey: 'key' })] });
    server.setNextResponse({ body: textResponse('pro answer') });

    const { response, stream } = ai.generateStream({
      model: openAI.model('o3-pro'),
      prompt: 'hi',
    });
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }
    const result = await response;

    expect(result.text).toBe('pro answer');
    expect(chunks.join('')).toBe('pro answer');
    const sent = server.requests[server.requests.length - 1];
    expect(sent.url).toBe('/v1/responses');
    expect(sent.body).not.toHaveProperty('stream');
  });

  test('sends Responses-only models to the Responses endpoint', async () => {
    const plugin = openAI({ apiKey: 'key' });
    const action = await plugin.model('gpt-5-pro');
    server.setNextResponse({ body: textResponse('ok') });

    await action({ messages: [{ role: 'user', content: [{ text: 'hi' }] }] });

    expect(server.requests[server.requests.length - 1].url).toBe(
      '/v1/responses'
    );
  });

  test('sends an output schema to the wire rather than into the prompt', async () => {
    const ai = genkit({ plugins: [openAI({ apiKey: 'key' })] });
    server.setNextResponse({ body: textResponse('{"colour":"blue"}') });

    await ai.generate({
      model: openAI.model('gpt-5-pro'),
      prompt: 'pick one',
      output: { schema: z.object({ colour: z.string() }) },
    });

    const sent = server.requests[server.requests.length - 1];
    expect(sent.body.text.format.type).toBe('json_schema');
    expect(sent.body.text.format.schema.properties).toHaveProperty('colour');
    // Constrained generation is native here, so the schema must not have been
    // simulated by appending it to the prompt.
    expect(JSON.stringify(sent.body.input)).not.toContain('colour');
  });

  test('runs a full tool-calling loop over the Responses transport', async () => {
    const ai = genkit({ plugins: [openAI({ apiKey: 'key' })] });
    const getWeather = ai.defineTool(
      {
        name: 'getWeather',
        description: 'gets the weather',
        inputSchema: z.object({ city: z.string() }),
        outputSchema: z.string(),
      },
      async ({ city }) => `sunny in ${city}`
    );
    server.setNextResponse({
      body: fakeResponse({
        output: [
          {
            id: 'fc_1',
            type: 'function_call',
            call_id: 'call_1',
            name: 'getWeather',
            arguments: '{"city":"Paris"}',
          },
        ],
      }),
    });
    server.setNextResponse({ body: textResponse('It is sunny.') });

    const result = await ai.generate({
      model: openAI.model('gpt-5-pro'),
      prompt: 'weather in Paris?',
      tools: [getWeather],
    });

    expect(result.text).toBe('It is sunny.');
    const secondRequest = server.requests[server.requests.length - 1];
    expect(secondRequest.body.input).toContainEqual({
      type: 'function_call',
      call_id: 'call_1',
      name: 'getWeather',
      arguments: '{"city":"Paris"}',
      id: 'fc_1',
    });
    expect(secondRequest.body.input).toContainEqual({
      type: 'function_call_output',
      call_id: 'call_1',
      output: 'sunny in Paris',
    });
  });

  test('sends dual-transport models to the Chat Completions endpoint', async () => {
    const plugin = openAI({ apiKey: 'key' });
    const action = await plugin.model('gpt-4o');
    server.setNextResponse({
      body: {
        choices: [
          {
            message: { role: 'assistant', content: 'ok' },
            finish_reason: 'stop',
          },
        ],
      },
    });

    await action({ messages: [{ role: 'user', content: [{ text: 'hi' }] }] });

    expect(server.requests[server.requests.length - 1].url).toBe(
      '/v1/chat/completions'
    );
  });

  test('dispatches a dual-transport model to Responses on config opt-in', async () => {
    const plugin = openAI({ apiKey: 'key' });
    const action = await plugin.model('gpt-4o');
    server.setNextResponse({ body: textResponse('via responses') });

    const result = await action({
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config: { transport: 'responses' },
    });

    expect(result.message?.content[0].text).toBe('via responses');
    const sent = server.requests[server.requests.length - 1];
    expect(sent.url).toBe('/v1/responses');
    expect(sent.body).not.toHaveProperty('transport');
  });

  test('streams a dual-transport model over Responses on config opt-in', async () => {
    const ai = genkit({ plugins: [openAI({ apiKey: 'key' })] });
    server.setNextResponse({
      stream: true,
      chunks: [
        {
          type: 'response.created',
          response: fakeResponse(),
          sequence_number: 0,
        },
        {
          type: 'response.output_item.added',
          output_index: 0,
          item: {
            id: 'msg_1',
            type: 'message',
            role: 'assistant',
            status: 'in_progress',
            content: [],
          },
          sequence_number: 1,
        },
        {
          type: 'response.content_part.added',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          part: { type: 'output_text', text: '', annotations: [] },
          sequence_number: 2,
        },
        {
          type: 'response.output_text.delta',
          item_id: 'msg_1',
          output_index: 0,
          content_index: 0,
          delta: 'streamed',
          sequence_number: 3,
        },
        {
          type: 'response.completed',
          response: textResponse('streamed'),
          sequence_number: 4,
        },
      ],
    });

    const { response, stream } = ai.generateStream({
      model: openAI.model('gpt-4o'),
      prompt: 'hi',
      config: { transport: 'responses' },
    });
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }
    const result = await response;

    expect(result.text).toBe('streamed');
    expect(chunks.join('')).toBe('streamed');
    const sent = server.requests[server.requests.length - 1];
    expect(sent.url).toBe('/v1/responses');
    expect(sent.body.stream).toBe(true);
  });

  test('lists Responses-only models with the transport-aware config schema', async () => {
    const plugin = openAI({ apiKey: 'key' });
    server.setNextResponse({
      body: {
        object: 'list',
        data: [
          { id: 'gpt-5-pro', object: 'model', created: 0, owned_by: 'openai' },
        ],
      },
    });

    const [metadata] = await plugin.list!();

    expect(metadata.name).toBe('openai/gpt-5-pro');
    expect(metadata.metadata?.model.supports.tools).toBe(true);
    expect(
      Object.keys(metadata.metadata?.model.customOptions.properties)
    ).toContain('transport');
  });
});

describe('openAI.responsesModel', () => {
  test('returns a namespaced ref pinned to the responses transport', () => {
    const ref = openAI.responsesModel('gpt-4o');

    expect(ref.name).toBe('openai/gpt-4o');
    expect(ref.config).toStrictEqual({ transport: 'responses' });
    expect(Object.keys(ref.configSchema!.shape)).toContain(
      'previousResponseId'
    );
  });

  test('merges call-site config under the pin', () => {
    const ref = openAI.responsesModel('gpt-4o', { reasoningEffort: 'low' });

    expect(ref.config).toStrictEqual({
      reasoningEffort: 'low',
      transport: 'responses',
    });
  });

  test('withConfig replaces config wholesale but re-injects the pin', () => {
    const ref = openAI
      .responsesModel('gpt-4o', { reasoningEffort: 'low' })
      .withConfig({ temperature: 0.5 });

    expect(ref.config).toStrictEqual({
      temperature: 0.5,
      transport: 'responses',
    });
  });

  test('keeps the pin through a withConfig().withVersion() chain', () => {
    const ref = openAI
      .responsesModel('gpt-4o')
      .withConfig({ temperature: 0.5 })
      .withVersion('gpt-4o-2024-08-06');

    expect(ref.version).toBe('gpt-4o-2024-08-06');
    expect(ref.config).toStrictEqual({
      temperature: 0.5,
      transport: 'responses',
    });
    // The chained ref must stay pinned for the next withConfig too.
    expect(ref.withConfig({ temperature: 1 }).config).toStrictEqual({
      temperature: 1,
      transport: 'responses',
    });
  });

  test('routes a dual-transport model over the Responses API end to end', async () => {
    const server = new FakeOpenAIServer();
    await server.start();
    const previousBaseUrl = process.env.OPENAI_BASE_URL;
    process.env.OPENAI_BASE_URL = server.baseUrl;
    try {
      const ai = genkit({ plugins: [openAI({ apiKey: 'key' })] });
      server.setNextResponse({ body: textResponse('opted in') });

      const result = await ai.generate({
        model: openAI.responsesModel('gpt-4o'),
        prompt: 'hi',
      });

      expect(result.text).toBe('opted in');
      const sent = server.requests[server.requests.length - 1];
      expect(sent.url).toBe('/v1/responses');
      expect(sent.body.model).toBe('gpt-4o');
      expect(sent.body).not.toHaveProperty('transport');
    } finally {
      if (previousBaseUrl === undefined) {
        delete process.env.OPENAI_BASE_URL;
      } else {
        process.env.OPENAI_BASE_URL = previousBaseUrl;
      }
      server.stop();
    }
  });

  test('sends an output schema natively on the dispatched path', async () => {
    const server = new FakeOpenAIServer();
    await server.start();
    const previousBaseUrl = process.env.OPENAI_BASE_URL;
    process.env.OPENAI_BASE_URL = server.baseUrl;
    try {
      const ai = genkit({ plugins: [openAI({ apiKey: 'key' })] });
      server.setNextResponse({ body: textResponse('{"colour":"blue"}') });

      // gpt-5's chat info declares no `constrained`, so without the dispatch
      // default this would be simulated into the prompt instead.
      await ai.generate({
        model: openAI.responsesModel('gpt-5'),
        prompt: 'pick one',
        output: { schema: z.object({ colour: z.string() }) },
      });

      const sent = server.requests[server.requests.length - 1];
      expect(sent.url).toBe('/v1/responses');
      expect(sent.body.text.format.type).toBe('json_schema');
      expect(JSON.stringify(sent.body.input)).not.toContain('colour');
    } finally {
      if (previousBaseUrl === undefined) {
        delete process.env.OPENAI_BASE_URL;
      } else {
        process.env.OPENAI_BASE_URL = previousBaseUrl;
      }
      server.stop();
    }
  });
});

describe('chat completions transport handling', () => {
  let server: FakeOpenAIServer;

  beforeAll(async () => {
    server = new FakeOpenAIServer();
    await server.start();
  });

  afterAll(() => {
    server.stop();
  });

  test('transport never reaches the Chat Completions wire', async () => {
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIModelRunner('gpt-4o', client);
    const request: GenerateRequest = {
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      config: { transport: 'chat_completions' },
    };

    await runner(request);

    const sent = server.requests[server.requests.length - 1];
    expect(sent.url).toBe('/v1/chat/completions');
    expect(sent.body).not.toHaveProperty('transport');
  });

  test('rejects an opt-in to the responses transport', async () => {
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIModelRunner('gpt-4o', client);

    await expect(
      runner({
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        config: { transport: 'responses' },
      })
    ).rejects.toThrow(expect.objectContaining({ status: 'INVALID_ARGUMENT' }));
  });

  test('rejects a near-miss transport value instead of serving it silently', async () => {
    const client = new OpenAI({ apiKey: 'key', baseURL: server.baseUrl });
    const runner = openAIModelRunner('gpt-4o', client);

    await expect(
      runner({
        messages: [{ role: 'user', content: [{ text: 'hi' }] }],
        config: { transport: 'Responses' },
      })
    ).rejects.toThrow(expect.objectContaining({ status: 'INVALID_ARGUMENT' }));
  });
});
