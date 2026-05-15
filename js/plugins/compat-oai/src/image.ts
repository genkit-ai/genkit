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
import { Message, modelRef, z } from 'genkit';
import { ModelAction, ModelInfo } from 'genkit/model';
import { model } from 'genkit/plugin';
import OpenAI from 'openai';
import type {
  ImageGenerateParams,
  ImagesResponse,
} from 'openai/resources/images.mjs';
import { PluginOptions } from './index.js';
import { maybeCreateRequestScopedOpenAIClient, toModelName } from './utils.js';

export type ImageRequestBuilder = (
  req: GenerateRequest,
  params: ImageGenerateParams
) => void;

export const IMAGE_GENERATION_MODEL_INFO: ModelInfo = {
  supports: {
    media: false,
    output: ['media'],
    multiturn: false,
    systemRole: false,
    tools: false,
  },
};

export const ImageGenerationCommonConfigSchema = z.object({
  size: z.enum(['1024x1024', '1792x1024', '1024x1792']).optional(),
  style: z.enum(['vivid', 'natural']).optional(),
  user: z.string().optional(),
  n: z.number().int().min(1).max(10).default(1),
  quality: z.enum(['standard', 'hd']).optional(),
  response_format: z.enum(['b64_json', 'url']).default('b64_json').optional(),
});

function toImageGenerateParams(
  modelName: string,
  request: GenerateRequest,
  requestBuilder?: ImageRequestBuilder
): ImageGenerateParams {
  const {
    temperature,
    version: modelVersion,
    maxOutputTokens,
    stopSequences,
    topK,
    topP,
    response_format,
    ...restOfConfig
  } = request.config ?? {};

  let options: ImageGenerateParams = {
    model: modelVersion ?? modelName,
    prompt: new Message(request.messages[0]).text,
    response_format: response_format || 'b64_json',
  };
  if (requestBuilder) {
    requestBuilder(request, options);
  } else {
    options = { ...options, ...restOfConfig };
  }
  for (const k in options) {
    if (options[k] === undefined) {
      delete options[k];
    }
  }
  return options;
}

function toGenerateResponse(result: ImagesResponse): GenerateResponseData {
  const images = result.data;
  if (!images) {
    return { finishReason: 'stop' };
  } else {
    const content = (result.data ?? []).map((image) => ({
      media: {
        contentType: 'image/png',
        url: image.url || `data:image/png;base64,${image.b64_json}`,
      },
    }));
    return { message: { role: 'model', content }, raw: result };
  }
}

/**
 * Method to define a new Genkit Model that is compatible with Open AI
 * Images API. 
 *
 * These models are to be used to create images from a user prompt.
 *
 * @param params An object containing parameters for defining the OpenAI
 * image model.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.

 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAIImageModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  client: OpenAI;
  pluginOptions?: PluginOptions;
  modelRef?: ModelReference<CustomOptions>;
  requestBuilder?: ImageRequestBuilder;
}): ModelAction<CustomOptions> {
  const {
    name,
    client: defaultClient,
    pluginOptions,
    modelRef,
    requestBuilder,
  } = params;

  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    modelRef?.name ?? `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    async (request, { abortSignal }) => {
      const client = maybeCreateRequestScopedOpenAIClient(
        pluginOptions,
        request,
        defaultClient
      );
      const result = await client.images.generate(
        toImageGenerateParams(modelName, request, requestBuilder),
        { signal: abortSignal }
      );
      return toGenerateResponse(result);
    }
  );
}

/** Image generation ModelRef helper, with reasonable defaults for
 * OpenAI-compatible providers */
export function compatOaiImageModelRef<
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
    info = IMAGE_GENERATION_MODEL_INFO,
    configSchema,
    config = undefined,
    namespace,
  } = params;
  return modelRef({
    name,
    configSchema:
      configSchema || (ImageGenerationCommonConfigSchema as z.AnyZodObject),
    info,
    config,
    namespace,
  });
}
