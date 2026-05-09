/**
 * @license
 *
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

/**
 * @module /
 */

import { EmbedderReference, ModelReference, z } from 'genkit';
import {
  genkitPluginV2,
  type GenkitPluginV2,
  type ResolvableAction,
} from 'genkit/plugin';
import { type ActionType } from 'genkit/registry';
import { listModels } from './client.js';

import * as embedder from './embedder.js';
import * as gemini from './gemini.js';
import * as imagen from './imagen.js';
import * as lyria from './lyria.js';
import * as veo from './veo.js';

import { VertexPluginOptions } from './types.js';
import { getDerivedOptions } from './utils.js';

export { type EmbeddingConfig } from './embedder.js';
export { type GeminiConfig } from './gemini.js';
export { type ImagenConfig } from './imagen.js';
export { type LyriaConfig } from './lyria.js';
export { type VertexPluginOptions } from './types.js';
export { type VeoConfig } from './veo.js';

async function initializer(pluginOptions?: VertexPluginOptions) {
  const clientOptions = await getDerivedOptions(pluginOptions);
  return [
    ...veo.listKnownModels(clientOptions, pluginOptions),
    ...imagen.listKnownModels(clientOptions, pluginOptions),
    ...lyria.listKnownModels(clientOptions, pluginOptions),
    ...gemini.listKnownModels(clientOptions, pluginOptions),
    ...embedder.listKnownModels(clientOptions, pluginOptions),
  ];
}

async function resolver(
  actionType: ActionType,
  actionName: string,
  pluginOptions?: VertexPluginOptions
): Promise<ResolvableAction | undefined> {
  const clientOptions = await getDerivedOptions(pluginOptions);
  switch (actionType) {
    case 'model':
      if (lyria.isLyriaModelName(actionName)) {
        return lyria.defineModel(actionName, clientOptions, pluginOptions);
      } else if (imagen.isImagenModelName(actionName)) {
        return imagen.defineModel(actionName, clientOptions, pluginOptions);
      } else if (veo.isVeoModelName(actionName)) {
        return undefined;
      } else {
        return gemini.defineModel(actionName, clientOptions, pluginOptions);
      }
      break;
    case 'background-model':
      if (veo.isVeoModelName(actionName)) {
        return veo.defineModel(actionName, clientOptions, pluginOptions);
      }
      break;
    case 'embedder':
      return embedder.defineEmbedder(actionName, clientOptions, pluginOptions);
      break;
  }
  return undefined;
}

async function listActions(options?: VertexPluginOptions) {
  try {
    const clientOptions = await getDerivedOptions(options);
    const models = await listModels(clientOptions);
    return [
      ...gemini.listActions(models),
      ...imagen.listActions(models),
      ...lyria.listActions(models),
      ...veo.listActions(models),
      // We don't list embedders here
    ];
  } catch (e: unknown) {
    // Errors are already logged in the client code.
    return [];
  }
}

/**
 * Add Google Cloud Vertex AI to Genkit. Includes Gemini and Imagen models and text embedder.
 */
function vertexAIPlugin(options?: VertexPluginOptions): GenkitPluginV2 {
  let listActionsCache;
  return genkitPluginV2({
    name: 'vertexai',
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

export type VertexAIPlugin = {
  (pluginOptions?: VertexPluginOptions): GenkitPluginV2;
  model(
    name: gemini.KnownImageModels | (gemini.ImageModelName & {}),
    config?: gemini.GeminiImageConfig
  ): ModelReference<gemini.GeminiImageConfigSchemaType>;
  model(
    name: gemini.KnownGeminiModels | (gemini.GeminiModelName & {}),
    config?: gemini.GeminiConfig
  ): ModelReference<gemini.GeminiConfigSchemaType>;
  model(
    name: imagen.KnownModels | (imagen.ImagenModelName & {}),
    config?: imagen.ImagenConfig
  ): ModelReference<imagen.ConfigSchemaType>;
  model(
    name: lyria.KnownModels | (lyria.LyriaModelName & {}),
    config: lyria.LyriaConfig
  ): ModelReference<lyria.LyriaConfigSchemaType>;
  model(
    name: veo.KnownModels | (veo.VeoModelName & {}),
    config: veo.VeoConfig
  ): ModelReference<veo.VeoConfigSchemaType>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;

  embedder(
    name: embedder.KnownModels | (embedder.EmbedderModelName & {}),
    config?: embedder.EmbeddingConfig
  ): EmbedderReference<embedder.EmbeddingConfigSchemaType>;
};

/**
 * Google Cloud Vertex AI plugin for Genkit.
 * Includes Gemini and Imagen models and text embedder.
 */
export const vertexAI = vertexAIPlugin as VertexAIPlugin;
// provide generic implementation for the model function overloads.
(vertexAI as any).model = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  if (imagen.isImagenModelName(name)) {
    return imagen.model(name, config);
  }
  if (lyria.isLyriaModelName(name)) {
    return lyria.model(name, config);
  }
  if (veo.isVeoModelName(name)) {
    return veo.model(name, config);
  }
  // gemini, image and unknown model families
  return gemini.model(name, config);
};
vertexAI.embedder = (
  name: string,
  config?: embedder.EmbeddingConfig
): EmbedderReference<embedder.EmbeddingConfigSchemaType> => {
  return embedder.model(name, config);
};

export default vertexAI;
