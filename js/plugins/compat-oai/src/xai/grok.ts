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
} from '../model';

/**
 * Language models that support text -> text, tool calling, structured output
 */
const XAI_LANGUGAGE_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: false,
    systemRole: true,
    output: ['text', 'json'],
  },
};

/** XAI Custom configuration schema. */
export const XaiChatCompletionConfigSchema =
  ChatCompletionCommonConfigSchema.extend({
    deferred: z.boolean().optional(),
    reasoningEffort: z.enum(['low', 'medium', 'high']).optional(),
    webSearchOptions: z.object({}).passthrough().optional(),
  });

/** XAI ModelRef helper, with XAI specific config. */
export function xaiModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: any;
}): ModelReference<typeof XaiChatCompletionConfigSchema> {
  return compatOaiModelRef({
    ...params,
    info: params.info ?? XAI_LANGUGAGE_MODEL_INFO,
    configSchema: XaiChatCompletionConfigSchema,
    namespace: 'xai',
  });
}

export const grokRequestBuilder: ModelRequestBuilder = (req, params) => {
  const { deferred, webSearchOptions, reasoningEffort } = req.config ?? {};

  params.web_search_options = webSearchOptions;
  params.reasoning_effort = reasoningEffort;
  // Deferred is not a standard field on the request type
  (params as any).deferred = deferred;
};

export const SUPPORTED_LANGUAGE_MODELS = {
  'grok-3': xaiModelRef({
    name: 'grok-3',
  }),
  'grok-3-fast': xaiModelRef({
    name: 'grok-3-fast',
  }),
  'grok-3-mini': xaiModelRef({
    name: 'grok-3-mini',
  }),
  'grok-3-mini-fast': xaiModelRef({
    name: 'grok-3-mini-fast',
  }),
  'grok-2-vision-1212': xaiModelRef({
    name: 'grok-2-vision-1212',
    info: {
      supports: {
        multiturn: false,
        tools: true,
        media: true,
        systemRole: false,
        output: ['text', 'json'],
      },
    },
  }),
};
