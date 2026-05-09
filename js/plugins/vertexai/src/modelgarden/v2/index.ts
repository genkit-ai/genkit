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

import { GenkitError, ModelReference, z } from 'genkit';
import { genkitPluginV2, type GenkitPluginV2 } from 'genkit/plugin';
import { ActionType } from 'genkit/registry';
import { getDerivedParams } from '../../common/index.js';
import * as anthropic from './anthropic.js';
import * as llama from './llama.js';
import * as mistral from './mistral.js';
import type { PluginOptions } from './types.js';

export type { PluginOptions };

async function initializer(pluginOptions?: PluginOptions) {
  const clientOptions = await getDerivedParams(pluginOptions);
  return [
    ...anthropic.listKnownModels(clientOptions, pluginOptions),
    ...mistral.listKnownModels(clientOptions, pluginOptions),
    ...llama.listKnownModels(clientOptions, pluginOptions),
  ];
}

async function resolver(
  actionType: ActionType,
  actionName: string,
  pluginOptions?: PluginOptions
) {
  const clientOptions = await getDerivedParams(pluginOptions);
  switch (actionType) {
    case 'model':
      if (anthropic.isAnthropicModelName(actionName)) {
        return anthropic.defineModel(actionName, clientOptions, pluginOptions);
      } else if (mistral.isMistralModelName(actionName)) {
        return mistral.defineModel(actionName, clientOptions, pluginOptions);
      } else if (llama.isLlamaModelName(actionName)) {
        return llama.defineModel(actionName, clientOptions, pluginOptions);
      }
      break;
  }
  return undefined;
}

async function listActions(options?: PluginOptions) {
  try {
    const clientOptions = await getDerivedParams(options);
    return [
      ...anthropic.listActions(clientOptions),
      ...mistral.listActions(clientOptions),
      ...llama.listActions(clientOptions),
    ];
  } catch (e: unknown) {
    return [];
  }
}

/**
 * Add Google Cloud Vertex AI Model Garden to Genkit.
 */
export function vertexModelGardenPlugin(
  options: PluginOptions
): GenkitPluginV2 {
  let listActionsCache;
  return genkitPluginV2({
    name: 'vertex-model-garden',
    init: async () => await initializer(options),
    resolve: async (actionType: ActionType, actionName: string) =>
      await resolver(actionType, actionName, options),
    list: async () => {
      if (listActionsCache) return listActionsCache;
      listActionsCache = await listActions(options);
      return listActionsCache;
    },
  });
}

export type VertexModelGardenPlugin = {
  (pluginOptions?: PluginOptions): GenkitPluginV2;
  model(
    name: anthropic.KnownModels | (anthropic.AnthropicModelName & {}),
    config?: anthropic.AnthropicConfig
  ): ModelReference<anthropic.AnthropicConfigSchemaType>;
  model(
    name: mistral.KnownModels | (mistral.MistralModelName & {}),
    config: mistral.MistralConfig
  ): ModelReference<mistral.MistralConfigSchemaType>;
  model(
    name: llama.KnownModels | (llama.LlamaModelName & {}),
    config: llama.LlamaConfig
  ): ModelReference<llama.LlamaConfigSchemaType>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
};

export const vertexModelGarden =
  vertexModelGardenPlugin as VertexModelGardenPlugin;
(vertexModelGarden as any).model = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  if (anthropic.isAnthropicModelName(name)) {
    return anthropic.model(name, config);
  }
  if (mistral.isMistralModelName(name)) {
    return mistral.model(name, config);
  }
  if (llama.isLlamaModelName(name)) {
    return llama.model(name, config);
  }
  throw new GenkitError({
    status: 'INVALID_ARGUMENT',
    message: `model '${name}' is not a recognized model name`,
  });
};
