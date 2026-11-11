/**
 * Copyright 2026 Google LLC
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
  embedderActionMetadata,
  EmbedderReference,
  GenkitError,
  modelActionMetadata,
  ModelReference,
  z,
} from 'genkit';
import { type GenkitPluginV2 } from 'genkit/plugin';
import { ActionType } from 'genkit/registry';
import OpenAI from 'openai';
import { defineCompatOpenAIEmbedder } from '../embedder.js';
import { openAICompatible, PluginOptions } from '../index.js';
import { defineCompatOpenAIModel } from '../model.js';
import { lookup } from '../utils.js';
import {
  mistralEmbedderRef,
  mistralEmbedderRequestBuilder,
  MistralEmbeddingConfigSchema,
  SUPPORTED_MISTRAL_EMBEDDERS,
} from './embedder.js';
import {
  MISTRAL_RUNNER_OPTIONS,
  MistralChatCompletionConfigSchema,
  mistralModelRef,
  mistralRequestBuilder,
  SUPPORTED_MISTRAL_MODELS,
} from './mistral.js';

const MISTRAL_BASE_URL = 'https://api.mistral.ai/v1';

export type MistralPluginOptions = Omit<PluginOptions, 'name' | 'baseURL'>;

function defineMistralModel(
  client: OpenAI,
  pluginOptions: PluginOptions,
  modelRef: ModelReference<any>
) {
  return defineCompatOpenAIModel({
    name: modelRef.name,
    client,
    pluginOptions,
    modelRef,
    requestBuilder: mistralRequestBuilder,
    runnerOptions: MISTRAL_RUNNER_OPTIONS,
  });
}

function defineMistralEmbedder(
  client: OpenAI,
  pluginOptions: PluginOptions,
  embedderRef: EmbedderReference<any>
) {
  return defineCompatOpenAIEmbedder({
    name: embedderRef.name,
    client,
    pluginOptions,
    embedderRef,
    requestBuilder: mistralEmbedderRequestBuilder,
  });
}

function createResolver(pluginOptions: PluginOptions) {
  return async (client: OpenAI, actionType: ActionType, actionName: string) => {
    if (actionType === 'embedder') {
      return defineMistralEmbedder(
        client,
        pluginOptions,
        lookup(SUPPORTED_MISTRAL_EMBEDDERS, actionName) ??
          mistralEmbedderRef({ name: actionName })
      );
    }
    if (actionType === 'model') {
      return defineMistralModel(
        client,
        pluginOptions,
        lookup(SUPPORTED_MISTRAL_MODELS, actionName) ??
          mistralModelRef({ name: actionName })
      );
    }
    return undefined;
  };
}

const listActions = async (client: OpenAI): Promise<ActionMetadata[]> => {
  return await client.models.list().then((response) =>
    response.data
      .filter((model) => model.object === 'model')
      .map((model: OpenAI.Model) => {
        if (model.id.includes('embed')) {
          const embedderRef =
            lookup(SUPPORTED_MISTRAL_EMBEDDERS, model.id) ??
            mistralEmbedderRef({ name: model.id });
          return embedderActionMetadata({
            name: embedderRef.name,
            info: embedderRef.info,
            configSchema: embedderRef.configSchema,
          });
        }
        const modelRef =
          lookup(SUPPORTED_MISTRAL_MODELS, model.id) ??
          mistralModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      })
  );
};

export function mistralPlugin(options?: MistralPluginOptions): GenkitPluginV2 {
  const apiKey = options?.apiKey ?? process.env.MISTRAL_API_KEY;
  if (!apiKey) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        'Please pass in the API key or set the MISTRAL_API_KEY environment variable.',
    });
  }
  // Request-scoped clients (a per-request `apiKey`) are built from
  // `pluginOptions`, so it must carry the Mistral base URL too.
  const pluginOptions: PluginOptions = {
    name: 'mistral',
    baseURL: MISTRAL_BASE_URL,
    ...options,
  };
  return openAICompatible({
    ...pluginOptions,
    apiKey,
    initializer: async (client) => {
      return [
        ...Object.values(SUPPORTED_MISTRAL_MODELS).map((modelRef) =>
          defineMistralModel(client, pluginOptions, modelRef)
        ),
        ...Object.values(SUPPORTED_MISTRAL_EMBEDDERS).map((embedderRef) =>
          defineMistralEmbedder(client, pluginOptions, embedderRef)
        ),
      ];
    },
    resolver: createResolver(pluginOptions),
    listActions,
  });
}

export type MistralPlugin = {
  (params?: MistralPluginOptions): GenkitPluginV2;
  model(
    name: keyof typeof SUPPORTED_MISTRAL_MODELS,
    config?: z.infer<typeof MistralChatCompletionConfigSchema>
  ): ModelReference<typeof MistralChatCompletionConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
  embedder(
    name: keyof typeof SUPPORTED_MISTRAL_EMBEDDERS,
    config?: z.infer<typeof MistralEmbeddingConfigSchema>
  ): EmbedderReference<typeof MistralEmbeddingConfigSchema>;
  embedder(name: string, config?: any): EmbedderReference<z.ZodTypeAny>;
};

const model = ((name: string, config?: any): ModelReference<z.ZodTypeAny> => {
  return mistralModelRef({
    name,
    info: lookup(SUPPORTED_MISTRAL_MODELS, name)?.info,
    config,
  });
}) as MistralPlugin['model'];

const embedder = ((
  name: string,
  config?: any
): EmbedderReference<z.ZodTypeAny> => {
  return mistralEmbedderRef({
    name,
    info: lookup(SUPPORTED_MISTRAL_EMBEDDERS, name)?.info,
    config,
  });
}) as MistralPlugin['embedder'];

/**
 * This module provides an interface to the Mistral models through the Genkit
 * plugin system. It allows users to interact with various models by providing
 * an API key and optional configuration.
 *
 * The main export is the `mistral` plugin, which can be configured with an API
 * key either directly or through environment variables. It initializes the
 * OpenAI client and makes available the models and embedders for use.
 *
 * Exports:
 * - mistral: The main plugin function to interact with Mistral, via OpenAI
 *   compatible API.
 *
 * Usage: To use the models, initialize the mistral plugin inside `genkit()`
 * and pass the configuration options. If no API key is provided in the
 * options, the environment variable `MISTRAL_API_KEY` must be set.
 *
 * Example:
 * ```
 * import { mistral } from '@genkit-ai/compat-oai/mistral';
 *
 * const ai = genkit({
 *  plugins: [
 *    mistral()
 *    ... // other plugins
 *  ]
 * });
 * ```
 */
export const mistral: MistralPlugin = Object.assign(mistralPlugin, {
  model,
  embedder,
});

export default mistral;
