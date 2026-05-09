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

import {
  ActionMetadata,
  GenkitError,
  modelActionMetadata,
  ModelReference,
  z,
} from 'genkit';
import { logger } from 'genkit/logging';
import { ResolvableAction, type GenkitPluginV2 } from 'genkit/plugin';
import { ActionType } from 'genkit/registry';
import OpenAI from 'openai';
import {
  defineCompatOpenAIImageModel,
  ImageGenerationCommonConfigSchema,
} from '../image.js';
import { openAICompatible, PluginOptions } from '../index.js';
import { defineCompatOpenAIModel } from '../model.js';
import { SUPPORTED_IMAGE_MODELS, xaiImageModelRef } from './grok-image.js';
import {
  grokRequestBuilder,
  SUPPORTED_LANGUAGE_MODELS,
  XaiChatCompletionConfigSchema,
  xaiModelRef,
} from './grok.js';

export type XAIPluginOptions = Omit<PluginOptions, 'name' | 'baseURL'>;

function createResolver(pluginOptions: PluginOptions) {
  return async (client: OpenAI, actionType: ActionType, actionName: string) => {
    if (actionType === 'model') {
      const modelRef = xaiModelRef({ name: actionName });
      return defineCompatOpenAIModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
        requestBuilder: grokRequestBuilder,
      });
    } else {
      logger.warn('Only model actions are supported by the XAI plugin');
    }
    return undefined;
  };
}

const listActions = async (client: OpenAI): Promise<ActionMetadata[]> => {
  return await client.models.list().then((response) =>
    response.data
      .filter((model) => model.object === 'model')
      .map((model: OpenAI.Model) => {
        if (model.id.includes('image')) {
          const modelRef =
            SUPPORTED_IMAGE_MODELS[model.id] ??
            xaiImageModelRef({ name: model.id });
          return modelActionMetadata({
            name: modelRef.name,
            info: modelRef.info,
            configSchema: modelRef.configSchema,
          });
        } else {
          const modelRef =
            SUPPORTED_LANGUAGE_MODELS[model.id] ??
            xaiModelRef({ name: model.id });
          return modelActionMetadata({
            name: modelRef.name,
            info: modelRef.info,
            configSchema: modelRef.configSchema,
          });
        }
      })
  );
};

export function xAIPlugin(options?: XAIPluginOptions): GenkitPluginV2 {
  const apiKey = options?.apiKey ?? process.env.XAI_API_KEY;
  if (!apiKey) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        'Please pass in the API key or set the XAI_API_KEY environment variable.',
    });
  }
  const pluginOptions = { name: 'xai', ...options };
  return openAICompatible({
    name: 'xai',
    baseURL: 'https://api.x.ai/v1',
    apiKey,
    ...options,
    initializer: async (client) => {
      const models = [] as ResolvableAction[];
      models.push(
        ...Object.values(SUPPORTED_LANGUAGE_MODELS).map((modelRef) =>
          defineCompatOpenAIModel({
            name: modelRef.name,
            client,
            pluginOptions,
            modelRef,
            requestBuilder: grokRequestBuilder,
          })
        )
      );
      models.push(
        ...Object.values(SUPPORTED_IMAGE_MODELS).map((modelRef) =>
          defineCompatOpenAIImageModel({
            name: modelRef.name,
            client,
            pluginOptions,
            modelRef,
          })
        )
      );
      return models;
    },
    resolver: createResolver(pluginOptions),
    listActions,
  });
}

export type XAIPlugin = {
  (params?: XAIPluginOptions): GenkitPluginV2;
  model(
    name: keyof typeof SUPPORTED_LANGUAGE_MODELS,
    config?: z.infer<typeof XaiChatCompletionConfigSchema>
  ): ModelReference<typeof XaiChatCompletionConfigSchema>;
  model(
    name: keyof typeof SUPPORTED_IMAGE_MODELS,
    config?: z.infer<typeof ImageGenerationCommonConfigSchema>
  ): ModelReference<typeof ImageGenerationCommonConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
};

const model = ((name: string, config?: any): ModelReference<z.ZodTypeAny> => {
  if (name.includes('image')) {
    return xaiImageModelRef({
      name,
      config,
    });
  }
  return xaiModelRef({
    name,
    config,
  });
}) as XAIPlugin['model'];

/**
 * This module provides an interface to the XAI models through the Genkit
 * plugin system. It allows users to interact with various models by providing
 * an API key and optional configuration.
 *
 * The main export is the `xai` plugin, which can be configured with an API
 * key either directly or through environment variables. It initializes the
 * OpenAI client and makes available the models for use.
 *
 * Exports:
 * - xAI: The main plugin function to interact with XAI, via OpenAI
 *   compatible API.
 *
 * Usage: To use the models, initialize the xAI plugin inside
 * `configureGenkit` and pass the configuration options. If no API key is
 * provided in the options, the environment variable `OPENAI_API_KEY` must be
 * set.
 *
 * Example:
 * ```
 * import { xAI } from '@genkit-ai/compat-oai/xai';
 *
 * export default configureGenkit({
 *  plugins: [
 *    xAI()
 *    ... // other plugins
 *  ]
 * });
 * ```
 */
export const xAI: XAIPlugin = Object.assign(xAIPlugin, {
  model,
});

export default xAI;
