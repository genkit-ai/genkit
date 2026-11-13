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

import { GenkitError, z, type MessageData } from 'genkit';
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

/**
 * Default for models that are not in the known list, such as ones found
 * through `/v1/models`: text only, with no tools or constrained output, since
 * their capabilities are unknown.
 */
const MISTRAL_UNKNOWN_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: false,
    media: false,
    systemRole: true,
    output: ['text'],
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
    seed: z
      .number()
      .int()
      .optional()
      .describe('Seed for random sampling. Sent as `random_seed`.'),
    reasoningEffort: z
      .string()
      .optional()
      .describe(
        'How much the model reasons before answering, such as `none`, `low`, ' +
          '`medium` or `high`. Allowed values vary by model: Mistral Small and ' +
          'Medium take `none` and `high`, Mistral Large 4 takes `low`, ' +
          '`medium` and `high`. Sent as `reasoning_effort`.'
      ),
  });

/**
 * Builds the parts of the request that differ from OpenAI. Mistral's request
 * schema rejects unknown fields, so only fields it accepts are set.
 */
export const mistralRequestBuilder: ModelRequestBuilder = (req, params) => {
  const { maxOutputTokens, topK, seed, reasoningEffort } = req.config ?? {};
  // Mistral's schema declares `top_k`, but its models reject it ("top_k
  // sampling is not enabled for this model"), so say so before sending.
  if (topK !== undefined) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        "Mistral doesn't support topK: its models reject top_k sampling. Remove topK from the config.",
    });
  }
  // `random_seed` and Mistral's `reasoning_effort` values are not part of the
  // OpenAI request type.
  const body = params as unknown as Record<string, unknown>;

  params.max_tokens = maxOutputTokens;
  body.random_seed = seed;
  body.reasoning_effort = reasoningEffort;
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
 * Mistral reports usage on the last streamed chunk without being asked, so
 * `stream_options` isn't sent, and it streams reasoning as `thinking` chunks
 * in a `content` array.
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
    info: params.info ?? MISTRAL_UNKNOWN_MODEL_INFO,
    configSchema: MistralChatCompletionConfigSchema,
    namespace: 'mistral',
  });
}

/**
 * Known models, by the aliases Mistral moves to each new snapshot. Magistral is
 * left out: Mistral retired it, and reasoning now comes from
 * `reasoningEffort` on the models below.
 */
export const SUPPORTED_MISTRAL_MODELS = {
  // Mistral Large 4. It has no `-latest` alias; `mistral-large-latest` still
  // names Mistral Large 3.
  'mistral-large-4': mistralModelRef({
    name: 'mistral-large-4',
    info: MISTRAL_MODEL_INFO,
  }),
  'mistral-large-latest': mistralModelRef({
    name: 'mistral-large-latest',
    info: MISTRAL_MODEL_INFO,
  }),
  'mistral-medium-latest': mistralModelRef({
    name: 'mistral-medium-latest',
    info: MISTRAL_MODEL_INFO,
  }),
  'mistral-small-latest': mistralModelRef({
    name: 'mistral-small-latest',
    info: MISTRAL_MODEL_INFO,
  }),
  'ministral-14b-latest': mistralModelRef({
    name: 'ministral-14b-latest',
    info: MISTRAL_MODEL_INFO,
  }),
  'ministral-8b-latest': mistralModelRef({
    name: 'ministral-8b-latest',
    info: MISTRAL_MODEL_INFO,
  }),
  'ministral-3b-latest': mistralModelRef({
    name: 'ministral-3b-latest',
    info: MISTRAL_MODEL_INFO,
  }),
  'codestral-latest': mistralModelRef({
    name: 'codestral-latest',
    info: MISTRAL_TEXT_MODEL_INFO,
  }),
};
