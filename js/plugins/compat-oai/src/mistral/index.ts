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
import type { ModelInfo } from 'genkit/model';
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

/**
 * The `capabilities` object of a `/v1/models` entry (Mistral's
 * `ModelCapabilities`). Only the fields the plugin reads are listed.
 */
interface MistralModelCapabilities {
  completion_chat?: boolean;
  function_calling?: boolean;
  vision?: boolean;
}

/**
 * Model info for a model outside the known list, from its `/v1/models`
 * capabilities. Output stays text only, since capabilities don't say whether
 * the model supports constrained output.
 */
function modelInfoFromCapabilities(
  capabilities: MistralModelCapabilities
): ModelInfo {
  return {
    supports: {
      multiturn: true,
      tools: capabilities.function_calling === true,
      media: capabilities.vision === true,
      systemRole: true,
      output: ['text'],
    },
  };
}

/**
 * @param listedModelInfo Model info derived from `/v1/models` capabilities,
 *   keyed by model id, so a listed model resolves with the info it was listed
 *   with.
 */
function createResolver(
  pluginOptions: PluginOptions,
  listedModelInfo: Map<string, ModelInfo>
) {
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
          mistralModelRef({
            name: actionName,
            info: listedModelInfo.get(actionName),
          })
      );
    }
    return undefined;
  };
}

/**
 * Lists Mistral's models and embedders from `/v1/models`. Ids containing
 * `embed` are embedders. Other entries are chat models only when their
 * `capabilities.completion_chat` is set, which leaves out OCR, moderation,
 * audio and similar models. Models outside the known list take tools and
 * media support from `capabilities`, or the text-only default when an entry
 * has no `capabilities`.
 */
function createListActions(listedModelInfo: Map<string, ModelInfo>) {
  return async (client: OpenAI): Promise<ActionMetadata[]> => {
    const response = await client.models.list();
    const actions: ActionMetadata[] = [];
    for (const model of response.data) {
      if (model.object !== 'model') continue;
      if (model.id.includes('embed')) {
        const embedderRef =
          lookup(SUPPORTED_MISTRAL_EMBEDDERS, model.id) ??
          mistralEmbedderRef({ name: model.id });
        actions.push(
          embedderActionMetadata({
            name: embedderRef.name,
            info: embedderRef.info,
            configSchema: embedderRef.configSchema,
          })
        );
        continue;
      }
      // `capabilities` is a Mistral field, not part of the OpenAI model type.
      const capabilities = (
        model as OpenAI.Model & { capabilities?: MistralModelCapabilities }
      ).capabilities;
      if (capabilities && !capabilities.completion_chat) continue;
      let modelRef = lookup(SUPPORTED_MISTRAL_MODELS, model.id);
      if (!modelRef) {
        const info = capabilities
          ? modelInfoFromCapabilities(capabilities)
          : undefined;
        if (info) listedModelInfo.set(model.id, info);
        modelRef = mistralModelRef({ name: model.id, info });
      }
      actions.push(
        modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        })
      );
    }
    return actions;
  };
}

export function mistralPlugin(options?: MistralPluginOptions): GenkitPluginV2 {
  const apiKey = options?.apiKey ?? process.env.MISTRAL_API_KEY;
  if (!apiKey) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        'Please pass in the API key or set the MISTRAL_API_KEY environment variable.',
    });
  }
  const listedModelInfo = new Map<string, ModelInfo>();
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
    resolver: createResolver(pluginOptions, listedModelInfo),
    listActions: createListActions(listedModelInfo),
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
