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

import type {
  GenerateRequest,
  GenerateResponseData,
  ModelReference,
} from 'genkit';
import { GenerationCommonConfigSchema, Message, modelRef, z } from 'genkit';
import type { ModelAction, ModelInfo } from 'genkit/model';
import { model } from 'genkit/plugin';
import OpenAI from 'openai';
import type {
  TranslationCreateParams,
  TranslationCreateResponse,
} from 'openai/resources/audio/index.mjs';
import { PluginOptions } from './index.js';
import { maybeCreateRequestScopedOpenAIClient, toModelName } from './utils.js';

export type TranslationRequestBuilder = (
  req: GenerateRequest,
  params: TranslationCreateParams
) => void;

export const TRANSLATION_MODEL_INFO: ModelInfo = {
  supports: {
    media: true,
    output: ['text', 'json'],
    multiturn: false,
    systemRole: false,
    tools: false,
  },
};

export const TranslationConfigSchema = GenerationCommonConfigSchema.pick({
  temperature: true,
}).extend({
  response_format: z
    .enum(['json', 'text', 'srt', 'verbose_json', 'vtt'])
    .optional(),
});

export function toTranslationRequest(
  modelName: string,
  request: GenerateRequest,
  requestBuilder?: TranslationRequestBuilder
): TranslationCreateParams {
  const message = new Message(request.messages[0]);
  const media = message.media;
  if (!media?.url) {
    throw new Error('No media found in the request');
  }
  const mediaBuffer = Buffer.from(
    media.url.slice(media.url.indexOf(',') + 1),
    'base64'
  );
  const mediaFile = new File([mediaBuffer], 'input', {
    type:
      media.contentType ??
      media.url.slice('data:'.length, media.url.indexOf(';')),
  });
  const {
    temperature,
    version: modelVersion,
    maxOutputTokens,
    stopSequences,
    topK,
    topP,
    ...restOfConfig
  } = request.config ?? {};

  let options: TranslationCreateParams = {
    model: modelVersion ?? modelName,
    file: mediaFile,
    prompt: message.text,
    temperature,
  };
  if (requestBuilder) {
    requestBuilder(request, options);
  } else {
    options = {
      ...options,
      ...restOfConfig, // passthrough rest of the config
    };
  }
  const outputFormat = request.output?.format as 'json' | 'text' | 'media';
  const customFormat = request.config?.response_format;
  if (outputFormat && customFormat) {
    if (
      outputFormat === 'json' &&
      customFormat !== 'json' &&
      customFormat !== 'verbose_json'
    ) {
      throw new Error(
        `Custom response format ${customFormat} is not compatible with output format ${outputFormat}`
      );
    }
  }
  if (outputFormat === 'media') {
    throw new Error(`Output format ${outputFormat} is not supported.`);
  }
  options.response_format = customFormat || outputFormat || 'text';
  for (const k in options) {
    if (options[k] === undefined) {
      delete options[k];
    }
  }
  return options;
}

export function translationToGenerateResponse(
  result: TranslationCreateResponse | string
): GenerateResponseData {
  return {
    message: {
      role: 'model',
      content: [
        {
          text: typeof result === 'string' ? result : result.text,
        },
      ],
    },
    finishReason: 'stop',
    raw: result,
  };
}

/**
 * Method to define a new Genkit Model that is compatible with Open AI
 * Translation API.
 *
 * These models are to be used to translate audio to text.
 *
 * @param params An object containing parameters for defining the OpenAI
 * translation model.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.
 *
 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAITranslationModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  client: OpenAI;
  pluginOptions?: PluginOptions;
  modelRef?: ModelReference<CustomOptions>;
  requestBuilder?: TranslationRequestBuilder;
}) {
  const {
    name,
    client: defaultClient,
    pluginOptions,
    modelRef,
    requestBuilder,
  } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName = `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    async (request, { abortSignal }) => {
      const params = toTranslationRequest(modelName, request, requestBuilder);
      const client = maybeCreateRequestScopedOpenAIClient(
        pluginOptions,
        request,
        defaultClient
      );
      const result = await client.audio.translations.create(params, {
        signal: abortSignal,
      });
      return translationToGenerateResponse(result);
    }
  );
}

/** Translation ModelRef helper, with reasonable defaults for
 * OpenAI-compatible providers */
export function compatOaiTranslationModelRef<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  info?: ModelInfo;
  configSchema?: CustomOptions;
  config?: any;
  namespace?: string;
}) {
  const {
    name,
    info = TRANSLATION_MODEL_INFO,
    configSchema,
    config = undefined,
    namespace,
  } = params;
  return modelRef({
    name,
    configSchema: configSchema || (TranslationConfigSchema as any),
    info,
    config,
    namespace,
  });
}
