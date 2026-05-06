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
const MINIMAX_LANGUAGE_MODEL_INFO: ModelInfo = {
  supports: {
    multiturn: true,
    tools: true,
    media: false,
    systemRole: true,
    output: ['text', 'json'],
  },
};

/** MiniMax Custom configuration schema. */
export const MiniMaxChatCompletionConfigSchema =
  ChatCompletionCommonConfigSchema.extend({
    temperature: z.number().min(0).max(1).optional(),
  });

/** MiniMax ModelRef helper, with MiniMax specific config. */
export function miniMaxModelRef(params: {
  name: string;
  info?: ModelInfo;
  config?: any;
}): ModelReference<typeof MiniMaxChatCompletionConfigSchema> {
  return compatOaiModelRef({
    ...params,
    info: params.info ?? MINIMAX_LANGUAGE_MODEL_INFO,
    configSchema: MiniMaxChatCompletionConfigSchema,
    namespace: 'minimax',
  });
}

export const miniMaxRequestBuilder: ModelRequestBuilder = (req, params) => {
  // Clamp temperature to MiniMax's [0, 1] range
  if (
    params.temperature !== undefined &&
    params.temperature !== null &&
    params.temperature > 1
  ) {
    params.temperature = 1;
  }
};

export const SUPPORTED_MINIMAX_MODELS: Record<
  string,
  ModelReference<typeof MiniMaxChatCompletionConfigSchema>
> = {
  'MiniMax-M2.7': miniMaxModelRef({
    name: 'MiniMax-M2.7',
  }),
  'MiniMax-M2.7-highspeed': miniMaxModelRef({
    name: 'MiniMax-M2.7-highspeed',
  }),
  'MiniMax-M2.5': miniMaxModelRef({
    name: 'MiniMax-M2.5',
  }),
  'MiniMax-M2.5-highspeed': miniMaxModelRef({
    name: 'MiniMax-M2.5-highspeed',
  }),
};
