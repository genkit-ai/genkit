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
import { ActionMetadata } from 'genkit';
import { ResolvableAction, genkitPluginV2 } from 'genkit/plugin';
import { ActionType } from 'genkit/registry';
import OpenAI, { type ClientOptions } from 'openai';
import { compatOaiModelRef, defineCompatOpenAIModel } from './model.js';
import { toModelName } from './utils.js';

export {
  SpeechConfigSchema,
  TranscriptionConfigSchema,
  compatOaiSpeechModelRef,
  compatOaiTranscriptionModelRef,
  defineCompatOpenAISpeechModel,
  defineCompatOpenAITranscriptionModel,
  type SpeechRequestBuilder,
  type TranscriptionRequestBuilder,
} from './audio.js';
export { defineCompatOpenAIEmbedder } from './embedder.js';
export {
  ImageGenerationCommonConfigSchema,
  compatOaiImageModelRef,
  defineCompatOpenAIImageModel,
  type ImageRequestBuilder,
} from './image.js';
export {
  ChatCompletionCommonConfigSchema,
  compatOaiModelRef,
  defineCompatOpenAIModel,
  openAIModelRunner,
  type ModelRequestBuilder,
} from './model.js';
export {
  TranslationConfigSchema,
  compatOaiTranslationModelRef,
  defineCompatOpenAITranslationModel,
  type TranslationRequestBuilder,
} from './translate.js';

export interface PluginOptions extends Partial<Omit<ClientOptions, 'apiKey'>> {
  apiKey?: ClientOptions['apiKey'] | false;
  name: string;
  initializer?: (client: OpenAI) => Promise<ResolvableAction[]>;
  resolver?: (
    client: OpenAI,
    actionType: ActionType,
    actionName: string
  ) => Promise<ResolvableAction | undefined> | ResolvableAction | undefined;
  listActions?: (client: OpenAI) => Promise<ActionMetadata[]>;
}

/**
 * This module provides the `openAICompatible` plugin factory for Genkit. It
 * enables interaction with OpenAI-compatible API endpoints, allowing users to
 * leverage various AI models by configuring API keys and other client options.
 *
 * The core export is `openAICompatible`, a function that accepts
 * `PluginOptions` and returns a Genkit plugin.
 *
 * Key `PluginOptions` include:
 *  - `name`: A string to uniquely identify this plugin instance
 *    (e.g., 'deepSeek', 'customOpenAI').
 *  - `apiKey`: The API key for the service. If not provided directly, the
 *    plugin will attempt to use the `OPENAI_API_KEY` environment variable.
 *  - `initializer`: An optional asynchronous function for custom setup after
 *    the OpenAI client is initialized. It receives the Genkit instance and the
 *    OpenAI client.
 *  - Additional properties from OpenAI's `ClientOptions` (like `baseURL`,
 *    `timeout`, etc.) can be passed to customize the OpenAI client.
 *
 * The returned plugin initializes an OpenAI client tailored to the provided
 * options, making configured models available for use within Genkit flows.
 *
 * @param {PluginOptions} options - Configuration options for the plugin.
 * @returns A Genkit plugin configured for an OpenAI-compatible service.
 *
 * Usage: Import `openAICompatible` (or your chosen import name for the default
 * export) from this package (e.g., `genkitx-openai`). Then, invoke it within
 * the `plugins` array of `configureGenkit`, providing the necessary
 * `PluginOptions`.
 *
 * Example:
 * ```typescript
 * import myOpenAICompatiblePlugin from 'genkitx-openai'; // Default import
 *
 * export default configureGenkit({
 *  plugins: [
 *    myOpenAICompatiblePlugin({
 *      name: 'gpt4o', // Name for this specific plugin configuration
 *      apiKey: process.env.OPENAI_API_KEY,
 *      // For a non-OpenAI compatible endpoint:
 *      // baseURL: 'https://api.custom-llm-provider.com/v1',
 *    }),
 *    myOpenAICompatiblePlugin({
 *      name: 'localLlama',
 *      apiKey: 'ollama', // Or specific key if required by local server
 *      baseURL: 'http://localhost:11434/v1', // Example for Ollama
 *    }),
 *    // ... other plugins
 *  ],
 * });
 * ```
 */
export const openAICompatible = (options: PluginOptions) => {
  let listActionsCache;
  var client: OpenAI;
  function createClient() {
    if (client) return client;
    const { apiKey, ...restofOptions } = options;
    client = new OpenAI({
      ...restofOptions,
      apiKey: apiKey === false ? 'placeholder' : apiKey,
    });
    return client;
  }
  return genkitPluginV2({
    name: options.name,
    async init() {
      if (!options.initializer) {
        return [];
      }
      return await options.initializer(createClient());
    },
    async resolve(actionType: ActionType, actionName: string) {
      if (options.resolver) {
        return await options.resolver(createClient(), actionType, actionName);
      } else {
        if (actionType === 'model') {
          return defineCompatOpenAIModel({
            name: toModelName(actionName, options.name),
            client: createClient(),
            pluginOptions: options,
            modelRef: compatOaiModelRef({
              name: actionName,
              namespace: options.name,
            }),
          });
        }
        return undefined;
      }
    },
    list:
      // Don't attempt to list models if apiKey set to false
      options.listActions && options.apiKey !== false
        ? async () => {
            if (listActionsCache) return listActionsCache;
            listActionsCache = await options.listActions!(createClient());
            return listActionsCache;
          }
        : undefined,
  });
};

export default openAICompatible;
