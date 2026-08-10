/**
 * Copyright 2024 The Fire Company
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

import type { ModelReference } from 'genkit';
import { modelRef, z } from 'genkit';
import type { ModelAction, ModelInfo } from 'genkit/model';
import { model } from 'genkit/plugin';
import OpenAI from 'openai';
import {
  TranscriptionConfigSchema,
  toSttRequest,
  transcriptionToGenerateResponse,
} from '../audio.js';
import type { PluginOptions } from '../index.js';
import {
  toTranslationRequest,
  translationToGenerateResponse,
} from '../translate.js';
import { maybeCreateRequestScopedOpenAIClient, toModelName } from '../utils.js';

export const WHISPER_MODEL_INFO: ModelInfo = {
  supports: {
    media: true,
    output: ['text', 'json'],
    multiturn: false,
    systemRole: false,
    tools: false,
  },
};

/**
 * Config schema for Whisper models. Extends the transcription config with
 * a `translate` flag that switches between transcription and translation APIs.
 */
export const WhisperConfigSchema = TranscriptionConfigSchema.extend({
  /** When true, uses Translation API instead of Transcription. Default: false */
  translate: z.boolean().optional().default(false),
});

/**
 * Method to define an OpenAI Whisper model that can perform both transcription and
 * translation based on the `translate` config flag.
 *
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.
 *
 * @returns the created {@link ModelAction}
 */
export function defineOpenAIWhisperModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  client: OpenAI;
  modelRef?: ModelReference<CustomOptions>;
  pluginOptions?: PluginOptions;
}): ModelAction {
  const { name, client: defaultClient, pluginOptions, modelRef } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    modelRef?.name ?? `${pluginOptions?.name ?? 'openai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    async (request, { abortSignal }) => {
      const { translate, ...cleanConfig } = (request.config ?? {}) as Record<
        string,
        unknown
      >;
      const cleanRequest = { ...request, config: cleanConfig };
      const client = maybeCreateRequestScopedOpenAIClient(
        pluginOptions,
        request,
        defaultClient
      );

      if (translate === true) {
        const params = toTranslationRequest(modelName, cleanRequest);
        const result = await client.audio.translations.create(params, {
          signal: abortSignal,
        });
        return translationToGenerateResponse(result);
      } else {
        const params = toSttRequest(modelName, cleanRequest);
        // Explicitly setting stream to false ensures we use the non-streaming overload
        const result = await client.audio.transcriptions.create(
          {
            ...params,
            stream: false,
          },
          { signal: abortSignal }
        );
        return transcriptionToGenerateResponse(result);
      }
    }
  );
}

/** OpenAI whisper ModelRef helper. */
export function openAIWhisperModelRef<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  info?: ModelInfo;
  configSchema?: CustomOptions;
  config?: any;
}) {
  const {
    name,
    info = WHISPER_MODEL_INFO,
    configSchema,
    config = undefined,
  } = params;
  return modelRef({
    name,
    configSchema: configSchema || (WhisperConfigSchema as any),
    info,
    config,
    namespace: 'openai',
  });
}

export const SUPPORTED_WHISPER_MODELS = {
  'whisper-1': openAIWhisperModelRef({
    name: 'whisper-1',
  }),
};
