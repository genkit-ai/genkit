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

import { z } from 'genkit';
import { ModelInfo, ModelReference } from 'genkit/model';
import {
  ChatCompletionCommonConfigSchema,
  ModelRequestBuilder,
  compatOaiModelRef,
} from '../model.js';

const DeepSeekReasoningEffortSchema = z
  .enum(['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
  .describe(
    'How much effort the model spends thinking before answering. ' +
      'Sent as the top-level `reasoning_effort` request field.'
  );
const DeepSeekThinkingSchema = z
  .object({
    type: z.enum(['adaptive', 'enabled', 'disabled']),
  })
  .describe(
    'Whether the model thinks before answering: always (enabled), ' +
      'never (disabled), or as needed (adaptive).'
  );

/** DeepSeek Custom configuration schema. */
export const DeepSeekChatCompletionConfigSchema =
  ChatCompletionCommonConfigSchema.extend({
    maxTokens: z.number().int().min(1).max(8192).optional(),
    reasoningEffort: DeepSeekReasoningEffortSchema.optional(),
    thinking: DeepSeekThinkingSchema.optional(),
  });

interface DeepSeekRequestFieldsBeyondOpenAISdk {
  reasoning_effort?: z.infer<typeof DeepSeekReasoningEffortSchema>;
  thinking?: z.infer<typeof DeepSeekThinkingSchema>;
}

export const deepSeekRequestBuilder: ModelRequestBuilder = (req, params) => {
  const { maxTokens, reasoningEffort, thinking } = req.config as z.infer<
    typeof DeepSeekChatCompletionConfigSchema
  >;
  // DeepSeek still uses max_tokens
  params.max_tokens = maxTokens;
  Object.assign(params, {
    reasoning_effort: reasoningEffort,
    thinking,
  } satisfies DeepSeekRequestFieldsBeyondOpenAISdk);
};

/** DeepSeek ModelRef helper, with DeepSeek specific config. */
export function deepSeekModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: any;
}): ModelReference<typeof DeepSeekChatCompletionConfigSchema> {
  return compatOaiModelRef({
    ...params,
    configSchema: DeepSeekChatCompletionConfigSchema,
    info: params.info ?? {
      supports: {
        multiturn: true,
        tools: true,
        media: false,
        systemRole: true,
        output: ['text', 'json'],
      },
    },
    namespace: 'deepseek',
  });
}

export const SUPPORTED_DEEPSEEK_MODELS = {
  'deepseek-reasoner': deepSeekModelRef({ name: 'deepseek-reasoner' }),
  'deepseek-chat': deepSeekModelRef({ name: 'deepseek-chat' }),
};
