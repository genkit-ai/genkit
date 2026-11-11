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

import { z, type MessageData } from 'genkit';
import { ModelInfo, ModelReference } from 'genkit/model';
import type { ChatCompletionMessageParam } from 'openai/resources/index.mjs';
import {
  ChatCompletionCommonConfigSchema,
  ModelRequestBuilder,
  ModelRunnerOptions,
  compatOaiModelRef,
} from '../model.js';

/**
 * Models with text and image input, tool calling and structured output.
 * Mistral accepts `response_format` with `json_schema`, so output is
 * constrained natively.
 */
const MISTRAL_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: true,
    systemRole: true,
    output: ['text', 'json'],
    constrained: 'all',
  },
};

/** Models with text input only, tool calling and structured output. */
const MISTRAL_TEXT_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: false,
    systemRole: true,
    output: ['text', 'json'],
    constrained: 'all',
  },
};

/** Mistral custom configuration schema. */
export const MistralChatCompletionConfigSchema =
  ChatCompletionCommonConfigSchema.extend({
    randomSeed: z
      .number()
      .int()
      .optional()
      .describe('Seed for random sampling. Sent as `random_seed`.'),
    safePrompt: z
      .boolean()
      .optional()
      .describe('Inject a safety prompt before the conversation.'),
    reasoningEffort: z
      .enum(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
      .optional()
      .describe('Reasoning effort, for models that support it.'),
    promptMode: z
      .enum(['reasoning'])
      .optional()
      .describe('Use the reasoning system prompt, for reasoning models.'),
  });

/**
 * Builds the parts of the request that differ from OpenAI. Mistral's request
 * schema rejects unknown fields, so only fields it accepts are set.
 */
export const mistralRequestBuilder: ModelRequestBuilder = (req, params) => {
  const {
    maxOutputTokens,
    topK,
    randomSeed,
    safePrompt,
    reasoningEffort,
    promptMode,
  } = req.config ?? {};
  // Fields that are not part of the OpenAI request type.
  const body = params as unknown as Record<string, unknown>;

  params.max_tokens = maxOutputTokens;
  body.top_k = topK;
  body.random_seed = randomSeed;
  body.safe_prompt = safePrompt;
  body.reasoning_effort = reasoningEffort;
  body.prompt_mode = promptMode;
  params.messages = toMistralMessages(req.messages, params.messages);
};

/**
 * Mistral has no `reasoning_content` field. Reasoning from earlier assistant
 * turns is sent back as `thinking` chunks in the `content` array instead,
 * which is the shape Mistral returns it in, with the chunk's `signature` when
 * the reasoning part kept one in `metadata.signature`.
 *
 * Each Genkit `model` message becomes exactly one `assistant` message, so the
 * two are matched up in order.
 */
function toMistralMessages(
  source: MessageData[],
  messages: ChatCompletionMessageParam[]
): ChatCompletionMessageParam[] {
  const modelMessages = source.filter((m) => m.role === 'model');
  let next = 0;
  return messages.map((message) => {
    if (message.role !== 'assistant') {
      return message;
    }
    const { reasoning_content: _, ...rest } = message as typeof message & {
      reasoning_content?: string;
    };
    const thinking = (modelMessages[next++]?.content ?? [])
      .filter(
        (part) =>
          part.reasoning !== undefined &&
          (part.reasoning || part.metadata?.signature)
      )
      .map((part) => ({
        type: 'thinking',
        thinking: [{ type: 'text', text: part.reasoning }],
        ...(typeof part.metadata?.signature === 'string'
          ? { signature: part.metadata.signature }
          : {}),
      }));
    if (thinking.length === 0) {
      return rest;
    }
    const content: unknown[] = [...thinking];
    if (typeof rest.content === 'string' && rest.content) {
      content.push({ type: 'text', text: rest.content });
    }
    return { ...rest, content } as ChatCompletionMessageParam;
  });
}

/**
 * Mistral does not accept `stream_options` on streaming requests, and streams
 * reasoning as `thinking` chunks in a `content` array.
 */
export const MISTRAL_RUNNER_OPTIONS: ModelRunnerOptions = {
  streamOptions: false,
  contentChunks: true,
};

/** Mistral ModelRef helper, with Mistral specific config. */
export function mistralModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: any;
}): ModelReference<typeof MistralChatCompletionConfigSchema> {
  return compatOaiModelRef({
    ...params,
    info: params.info ?? MISTRAL_MODEL_INFO,
    configSchema: MistralChatCompletionConfigSchema,
    namespace: 'mistral',
  });
}

export const SUPPORTED_MISTRAL_MODELS = {
  'mistral-large-latest': mistralModelRef({ name: 'mistral-large-latest' }),
  'mistral-medium-latest': mistralModelRef({ name: 'mistral-medium-latest' }),
  'mistral-small-latest': mistralModelRef({ name: 'mistral-small-latest' }),
  'magistral-medium-latest': mistralModelRef({
    name: 'magistral-medium-latest',
  }),
  'magistral-small-latest': mistralModelRef({
    name: 'magistral-small-latest',
  }),
  'ministral-14b-latest': mistralModelRef({ name: 'ministral-14b-latest' }),
  'ministral-8b-latest': mistralModelRef({ name: 'ministral-8b-latest' }),
  'ministral-3b-latest': mistralModelRef({ name: 'ministral-3b-latest' }),
  'codestral-latest': mistralModelRef({
    name: 'codestral-latest',
    info: MISTRAL_TEXT_MODEL_INFO,
  }),
};
