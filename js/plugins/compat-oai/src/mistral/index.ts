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
import { mistralErrorBodyFetch, mistralErrorStatus } from './errors.js';
import {
  MISTRAL_RUNNER_OPTIONS,
  MistralChatCompletionConfigSchema,
  mistralModelRef,
  mistralRequestBuilder,
  SUPPORTED_MISTRAL_MODELS,
} from './mistral.js';

const MISTRAL_BASE_URL = 'https://api.mistral.ai/v1';

/**
 * Options for the Mistral plugin. `baseURL` defaults to
 * `https://api.mistral.ai/v1`; set it to go through a proxy.
 */
export type MistralPluginOptions = Omit<PluginOptions, 'name'>;

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
    errorStatus: mistralErrorStatus,
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
      toolChoice: capabilities.function_calling === true,
      media: capabilities.vision === true,
      systemRole: true,
      output: ['text'],
    },
  };
}

/**
 * @param lookupModelInfo Finds the info of a model outside the known list, as
 *   `/v1/models` lists it, so a model resolves with the same capabilities
 *   whether or not the models were listed first.
 */
function createResolver(
  pluginOptions: PluginOptions,
  lookupModelInfo: (
    client: OpenAI,
    name: string
  ) => Promise<ModelInfo | undefined>
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
            info: await lookupModelInfo(client, actionName),
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
async function listMistralActions(
  client: OpenAI,
  listedModelInfo: Map<string, ModelInfo>
): Promise<ActionMetadata[]> {
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
}

/**
 * Lists `/v1/models` at most once, for both `listActions` and the resolver,
 * and keeps the info of each listed model outside the known list. A model
 * resolved before the models were listed (the usual case outside the Dev UI)
 * then gets the same capabilities as a listed one. A failed listing is not
 * kept, so the next call tries again.
 *
 * @param canList Whether the plugin has a default API key to list with.
 *   Without one (`apiKey: false`), unknown models get the text-only default.
 */
function createModelCatalog(canList: boolean) {
  const listedModelInfo = new Map<string, ModelInfo>();
  let listing: Promise<ActionMetadata[]> | undefined;
  const list = (client: OpenAI): Promise<ActionMetadata[]> => {
    listing ??= listMistralActions(client, listedModelInfo).catch((e) => {
      listing = undefined;
      throw e;
    });
    return listing;
  };
  const lookupModelInfo = async (
    client: OpenAI,
    name: string
  ): Promise<ModelInfo | undefined> => {
    if (!listedModelInfo.has(name) && canList) {
      try {
        await list(client);
      } catch {
        // Fall back to the text-only default.
      }
    }
    return listedModelInfo.get(name);
  };
  return { list, lookupModelInfo };
}

export function mistralPlugin(options?: MistralPluginOptions): GenkitPluginV2 {
  const apiKey = options?.apiKey ?? process.env.MISTRAL_API_KEY;
  // `apiKey: false` means no default key: every request brings its own.
  if (apiKey == null || apiKey === '') {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message:
        'Please pass in the API key or set the MISTRAL_API_KEY environment variable.',
    });
  }
  const catalog = createModelCatalog(apiKey !== false);
  // Request-scoped clients (a per-request `apiKey`) are built from
  // `pluginOptions`, so it must carry the Mistral base URL too.
  const pluginOptions: PluginOptions = {
    ...options,
    name: 'mistral',
    baseURL: options?.baseURL ?? MISTRAL_BASE_URL,
    // Lets the SDK read Mistral's error bodies, e.g. to report an unknown
    // model as NOT_FOUND.
    fetch: mistralErrorBodyFetch(options),
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
    resolver: createResolver(pluginOptions, catalog.lookupModelInfo),
    listActions: catalog.list,
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
