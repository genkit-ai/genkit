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

import { z } from 'genkit';
import { ModelInfo, ModelReference } from 'genkit/model';
import {
  ChatCompletionCommonConfigSchema,
  ModelRequestBuilder,
  compatOaiModelRef,
} from '../model.js';

/**
 * Language models that support text -> text, tool calling, structured output.
 */
const GROQ_LANGUAGE_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: false,
    systemRole: true,
    output: ['text', 'json'],
  },
};

/** Groq Custom configuration schema. */
export const GroqChatCompletionConfigSchema =
  ChatCompletionCommonConfigSchema.extend({
    /**
     * Controls reasoning effort for supported models.
     * - Qwen: `none` | `default`
     * - GPT-OSS: `low` | `medium` | `high`
     */
    reasoningEffort: z
      .enum(['none', 'default', 'low', 'medium', 'high'])
      .optional(),
    /**
     * How to surface reasoning tokens. Mutually exclusive with
     * `includeReasoning`.
     */
    reasoningFormat: z.enum(['hidden', 'raw', 'parsed']).optional(),
    /**
     * Whether to include reasoning in the response. Mutually exclusive with
     * `reasoningFormat`. Used by GPT-OSS models.
     */
    includeReasoning: z.boolean().optional(),
    /**
     * Service tier for the request. Defaults to `on_demand` on Groq.
     */
    serviceTier: z
      .enum(['auto', 'on_demand', 'flex', 'performance'])
      .optional(),
  });

/** Groq ModelRef helper, with Groq specific config. */
export function groqModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: any;
}): ModelReference<typeof GroqChatCompletionConfigSchema> {
  return compatOaiModelRef({
    ...params,
    info: params.info ?? GROQ_LANGUAGE_MODEL_INFO,
    configSchema: GroqChatCompletionConfigSchema,
    namespace: 'groq',
  });
}

export const groqRequestBuilder: ModelRequestBuilder = (req, params) => {
  const {
    reasoningEffort,
    reasoningFormat,
    includeReasoning,
    serviceTier,
  } = req.config ?? {};

  params.reasoning_effort = reasoningEffort;
  // These fields are Groq-specific extensions beyond the OpenAI SDK types.
  (params as any).reasoning_format = reasoningFormat;
  (params as any).include_reasoning = includeReasoning;
  (params as any).service_tier = serviceTier;
};

export const SUPPORTED_GROQ_MODELS = {
  'llama-3.1-8b-instant': groqModelRef({
    name: 'llama-3.1-8b-instant',
  }),
  'llama-3.3-70b-versatile': groqModelRef({
    name: 'llama-3.3-70b-versatile',
  }),
  'openai/gpt-oss-120b': groqModelRef({
    name: 'openai/gpt-oss-120b',
  }),
  'openai/gpt-oss-20b': groqModelRef({
    name: 'openai/gpt-oss-20b',
  }),
  'groq/compound': groqModelRef({
    name: 'groq/compound',
  }),
  'groq/compound-mini': groqModelRef({
    name: 'groq/compound-mini',
  }),
  'qwen/qwen3.6-27b': groqModelRef({
    name: 'qwen/qwen3.6-27b',
    info: {
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    },
  }),
};
