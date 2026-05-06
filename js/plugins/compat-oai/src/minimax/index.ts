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

import { GenkitError, ModelReference, z } from 'genkit';
import { logger } from 'genkit/logging';
import { type GenkitPluginV2 } from 'genkit/plugin';
import { ActionType } from 'genkit/registry';
import OpenAI from 'openai';
import { openAICompatible, PluginOptions } from '../index.js';
import { defineCompatOpenAIModel } from '../model.js';
import {
  MiniMaxChatCompletionConfigSchema,
  miniMaxModelRef,
  miniMaxRequestBuilder,
  SUPPORTED_MINIMAX_MODELS,
} from './minimax.js';

export type MiniMaxPluginOptions = Omit<PluginOptions, 'name' | 'baseURL'>;

function createResolver(pluginOptions: PluginOptions) {
  return async (client: OpenAI, actionType: ActionType, actionName: string) => {
    if (actionType === 'model') {
      const modelRef = miniMaxModelRef({
        name: actionName,
      });
      return defineCompatOpenAIModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
        requestBuilder: miniMaxRequestBuilder,
      });
    } else {
      logger.warn('Only model actions are supported by the MiniMax plugin');
      return undefined;
    }
  };
}

export function miniMaxPlugin(
  options?: MiniMaxPluginOptions
): GenkitPluginV2 {
  const apiKey = options?.apiKey ?? process.env.MINIMAX_API_KEY;
  if (!apiKey) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        'Please pass in the API key or set the MINIMAX_API_KEY environment variable.',
    });
  }
  const pluginOptions = { name: 'minimax', ...options };
  return openAICompatible({
    name: 'minimax',
    baseURL: 'https://api.minimax.io/v1',
    apiKey,
    ...options,
    initializer: async (client) => {
      return Object.values(SUPPORTED_MINIMAX_MODELS).map((modelRef) =>
        defineCompatOpenAIModel({
          name: modelRef.name,
          client,
          pluginOptions,
          modelRef,
          requestBuilder: miniMaxRequestBuilder,
        })
      );
    },
    resolver: createResolver(pluginOptions),
  });
}

export type MiniMaxPlugin = {
  (params?: MiniMaxPluginOptions): GenkitPluginV2;
  model(
    name: keyof typeof SUPPORTED_MINIMAX_MODELS,
    config?: z.infer<typeof MiniMaxChatCompletionConfigSchema>
  ): ModelReference<typeof MiniMaxChatCompletionConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
};

const model = ((name: string, config?: any): ModelReference<z.ZodTypeAny> => {
  return miniMaxModelRef({
    name,
    config,
  });
}) as MiniMaxPlugin['model'];

/**
 * This module provides an interface to the MiniMax models through the Genkit
 * plugin system. It allows users to interact with various models by providing
 * an API key and optional configuration.
 *
 * The main export is the `miniMax` plugin, which can be configured with an API
 * key either directly or through environment variables. It initializes the
 * OpenAI client and makes available the models for use.
 *
 * Exports:
 * - miniMax: The main plugin function to interact with MiniMax, via OpenAI
 *   compatible API.
 *
 * Usage: To use the models, initialize the miniMax plugin inside
 * `configureGenkit` and pass the configuration options. If no API key is
 * provided in the options, the environment variable `MINIMAX_API_KEY` must be
 * set.
 *
 * Example:
 * ```
 * import { miniMax } from '@genkit-ai/compat-oai/minimax';
 *
 * export default configureGenkit({
 *  plugins: [
 *    miniMax(),
 *    // ... other plugins
 *  ]
 * });
 * ```
 */
export const miniMax: MiniMaxPlugin = Object.assign(miniMaxPlugin, {
  model,
});

export default miniMax;
