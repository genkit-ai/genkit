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
import { type GenkitPluginV2 } from 'genkit/plugin';
import { ActionType } from 'genkit/registry';
import OpenAI from 'openai';
import { openAICompatible, PluginOptions } from '../index.js';
import { defineCompatOpenAIModel } from '../model.js';
import {
  DeepSeekChatCompletionConfigSchema,
  deepSeekModelRef,
  deepSeekRequestBuilder,
  SUPPORTED_DEEPSEEK_MODELS,
} from './deepseek.js';

export type DeepSeekPluginOptions = Omit<PluginOptions, 'name' | 'baseURL'>;

function createResolver(pluginOptions: PluginOptions) {
  return async (client: OpenAI, actionType: ActionType, actionName: string) => {
    if (actionType === 'model') {
      const modelRef = deepSeekModelRef({
        name: actionName,
      });
      return defineCompatOpenAIModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
        requestBuilder: deepSeekRequestBuilder,
      });
    } else {
      logger.warn('Only model actions are supported by the DeepSeek plugin');
      return undefined;
    }
  };
}

const listActions = async (client: OpenAI): Promise<ActionMetadata[]> => {
  return await client.models.list().then((response) =>
    response.data
      .filter((model) => model.object === 'model')
      .map((model: OpenAI.Model) => {
        const modelRef =
          SUPPORTED_DEEPSEEK_MODELS[model.id] ??
          deepSeekModelRef({
            name: model.id,
          });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      })
  );
};

export function deepSeekPlugin(
  options?: DeepSeekPluginOptions
): GenkitPluginV2 {
  const apiKey = options?.apiKey ?? process.env.DEEPSEEK_API_KEY;
  if (!apiKey) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        'Please pass in the API key or set the DEEPSEEK_API_KEY environment variable.',
    });
  }
  const pluginOptions = { name: 'deepseek', ...options };
  return openAICompatible({
    name: 'deepseek',
    baseURL: 'https://api.deepseek.com',
    apiKey,
    ...options,
    initializer: async (client) => {
      return Object.values(SUPPORTED_DEEPSEEK_MODELS).map((modelRef) =>
        defineCompatOpenAIModel({
          name: modelRef.name,
          client,
          pluginOptions,
          modelRef,
          requestBuilder: deepSeekRequestBuilder,
        })
      );
    },
    resolver: createResolver(pluginOptions),
    listActions,
  });
}

export type DeepSeekPlugin = {
  (params?: DeepSeekPluginOptions): GenkitPluginV2;
  model(
    name: keyof typeof SUPPORTED_DEEPSEEK_MODELS,
    config?: z.infer<typeof DeepSeekChatCompletionConfigSchema>
  ): ModelReference<typeof DeepSeekChatCompletionConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
};

const model = ((name: string, config?: any): ModelReference<z.ZodTypeAny> => {
  return deepSeekModelRef({
    name,
    config,
  });
}) as DeepSeekPlugin['model'];

/**
 * This module provides an interface to the DeepSeek models through the Genkit
 * plugin system. It allows users to interact with various models by providing
 * an API key and optional configuration.
 *
 * The main export is the `deepseek` plugin, which can be configured with an API
 * key either directly or through environment variables. It initializes the
 * OpenAI client and makes available the models for use.
 *
 * Exports:
 * - deepSeek: The main plugin function to interact with DeepSeek, via OpenAI
 *   compatible API.
 *
 * Usage: To use the models, initialize the deepseek plugin inside
 * `configureGenkit` and pass the configuration options. If no API key is
 * provided in the options, the environment variable `OPENAI_API_KEY` must be
 * set.
 *
 * Example:
 * ```
 * import { deepSeek } from '@genkit-ai/compat-oai/deepseek';
 *
 * export default configureGenkit({
 *  plugins: [
 *    deepSeek()
 *    ... // other plugins
 *  ]
 * });
 * ```
 */
export const deepSeek: DeepSeekPlugin = Object.assign(deepSeekPlugin, {
  model,
});

export default deepSeek;
