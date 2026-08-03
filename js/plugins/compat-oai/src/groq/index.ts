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
  GroqChatCompletionConfigSchema,
  groqModelRef,
  groqRequestBuilder,
  SUPPORTED_GROQ_MODELS,
} from './groq.js';

export type GroqPluginOptions = Omit<PluginOptions, 'name' | 'baseURL'>;

function createResolver(pluginOptions: PluginOptions) {
  return async (client: OpenAI, actionType: ActionType, actionName: string) => {
    if (actionType === 'model') {
      const modelRef = groqModelRef({
        name: actionName,
      });
      return defineCompatOpenAIModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
        requestBuilder: groqRequestBuilder,
      });
    } else {
      logger.warn('Only model actions are supported by the Groq plugin');
      return undefined;
    }
  };
}

const listActions = async (client: OpenAI): Promise<ActionMetadata[]> => {
  return await client.models.list().then((response) =>
    response.data
      .filter((model) => model.object === 'model')
      // Whisper / TTS / STT / embedding ids are not chat-completion models.
      .filter((model) => !model.id.startsWith('whisper'))
      .filter((model) => !model.id.includes('orpheus'))
      .filter((model) => !model.id.includes('embed'))
      .map((model: OpenAI.Model) => {
        const modelRef =
          SUPPORTED_GROQ_MODELS[model.id] ??
          groqModelRef({
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

export function groqPlugin(options?: GroqPluginOptions): GenkitPluginV2 {
  const apiKey = options?.apiKey ?? process.env.GROQ_API_KEY;
  // Allow `apiKey: false` (openAICompatible maps it to a placeholder for
  // local proxies / tests). Only reject missing/empty keys.
  if (apiKey === undefined || apiKey === '') {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        'Please pass in the API key or set the GROQ_API_KEY environment variable.',
    });
  }
  const pluginOptions = { name: 'groq', ...options };
  return openAICompatible({
    name: 'groq',
    baseURL: 'https://api.groq.com/openai/v1',
    apiKey,
    ...options,
    initializer: async (client) => {
      return Object.values(SUPPORTED_GROQ_MODELS).map((modelRef) =>
        defineCompatOpenAIModel({
          name: modelRef.name,
          client,
          pluginOptions,
          modelRef,
          requestBuilder: groqRequestBuilder,
        })
      );
    },
    resolver: createResolver(pluginOptions),
    listActions,
  });
}

export type GroqPlugin = {
  (params?: GroqPluginOptions): GenkitPluginV2;
  model(
    name: keyof typeof SUPPORTED_GROQ_MODELS,
    config?: z.infer<typeof GroqChatCompletionConfigSchema>
  ): ModelReference<typeof GroqChatCompletionConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
};

const model = ((name: string, config?: any): ModelReference<z.ZodTypeAny> => {
  return groqModelRef({
    name,
    config,
  });
}) as GroqPlugin['model'];

/**
 * This module provides an interface to the Groq models through the Genkit
 * plugin system. It allows users to interact with various models by providing
 * an API key and optional configuration.
 *
 * The main export is the `groq` plugin, which can be configured with an API
 * key either directly or through environment variables. It initializes the
 * OpenAI client against Groq's OpenAI-compatible endpoint and makes available
 * the models for use.
 *
 * Exports:
 * - groq: The main plugin function to interact with Groq, via OpenAI
 *   compatible API.
 *
 * Usage: To use the models, initialize the groq plugin inside
 * `configureGenkit` and pass the configuration options. If no API key is
 * provided in the options, the environment variable `GROQ_API_KEY` must be
 * set.
 *
 * Example:
 * ```
 * import { groq } from '@genkit-ai/compat-oai/groq';
 *
 * export default configureGenkit({
 *  plugins: [
 *    groq()
 *    ... // other plugins
 *  ]
 * });
 * ```
 */
export const groq: GroqPlugin = Object.assign(groqPlugin, {
  model,
});

export default groq;
