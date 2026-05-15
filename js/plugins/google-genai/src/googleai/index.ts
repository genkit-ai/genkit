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

import { ActionMetadata, EmbedderReference, ModelReference, z } from 'genkit';
import { logger } from 'genkit/logging';
import {
  genkitPluginV2,
  type GenkitPluginV2,
  type ResolvableAction,
} from 'genkit/plugin';
import { ActionType } from 'genkit/registry';
import { extractErrMsg } from '../common/utils.js';
import { listModels } from './client.js';
import { GoogleAIPluginOptions } from './types.js';
import { calculateApiKey } from './utils.js';

// These are namespaced because they all intentionally have
// functions of the same name with the same arguments.
// (All exports from these files are used here)
import * as deepResearch from './deep-research.js';
import * as embedder from './embedder.js';
import * as gemini from './gemini.js';
import * as imagen from './imagen.js';
import * as lyria from './lyria.js';
import * as veo from './veo.js';

export { type DeepResearchConfig } from './deep-research.js';
export { type EmbeddingConfig } from './embedder.js';
export { type GeminiConfig, type GeminiTtsConfig } from './gemini.js';
export { type ImagenConfig } from './imagen.js';
export { type LyriaConfig } from './lyria.js';
export { type GoogleAIPluginOptions };

async function initializer(options?: GoogleAIPluginOptions) {
  return [
    ...imagen.listKnownModels(options),
    ...gemini.listKnownModels(options),
    ...embedder.listKnownModels(options),
    ...veo.listKnownModels(options),
    ...deepResearch.listKnownModels(options),
    ...lyria.listKnownModels(options),
  ];
}

async function resolver(
  actionType: ActionType,
  actionName: string,
  options: GoogleAIPluginOptions
): Promise<ResolvableAction | undefined> {
  switch (actionType) {
    case 'model':
      if (
        veo.isVeoModelName(actionName) ||
        deepResearch.isDeepResearchModelName(actionName)
      ) {
        // These are background models. Do not define them here.
        return undefined;
      } else if (imagen.isImagenModelName(actionName)) {
        return imagen.defineModel(actionName, options);
      } else if (lyria.isLyriaModelName(actionName)) {
        return lyria.defineModel(actionName, options);
      } else {
        // gemini, tts, image, gemma, unknown models
        return gemini.defineModel(actionName, options);
      }
    case 'background-model':
      if (veo.isVeoModelName(actionName)) {
        return veo.defineModel(actionName, options);
      } else if (deepResearch.isDeepResearchModelName(actionName)) {
        return deepResearch.defineModel(actionName, options);
      }
      break; // No default
    case 'embedder':
      return embedder.defineEmbedder(actionName, options);
  }
  return undefined;
}

async function listActions(
  options?: GoogleAIPluginOptions
): Promise<ActionMetadata[]> {
  // Don't attempt to list models if apiKey is set to false.
  if (options?.apiKey === false) {
    return [];
  }
  try {
    const apiKey = calculateApiKey(options?.apiKey, undefined);
    const allModels = await listModels(apiKey, {
      baseUrl: options?.baseUrl,
      apiVersion: options?.apiVersion,
    });

    return [
      ...gemini.listActions(allModels),
      ...imagen.listActions(allModels),
      ...veo.listActions(allModels),
      ...deepResearch.listActions(allModels),
      ...lyria.listActions(allModels),
      ...embedder.listActions(allModels),
    ];
  } catch (e: unknown) {
    logger.error(extractErrMsg(e));
    return [];
  }
}

/**
 * Google Gemini Developer API plugin.
 */
export function googleAIPlugin(
  options?: GoogleAIPluginOptions
): GenkitPluginV2 {
  let listActionsCache;
  return genkitPluginV2({
    name: 'googleai',
    init: async () => await initializer(options),
    resolve: async (actionType: ActionType, actionName: string) =>
      await resolver(actionType, actionName, options || {}),
    list: async () => {
      if (listActionsCache) return listActionsCache;
      listActionsCache = await listActions(options);
      return listActionsCache;
    },
  });
}

export type GoogleAIPlugin = {
  (pluginOptions?: GoogleAIPluginOptions): GenkitPluginV2;
  model(
    name: gemini.KnownGemmaModels | (gemini.GemmaModelName & {}),
    config: gemini.GemmaConfig
  ): ModelReference<gemini.GemmaConfigSchemaType>;
  model(
    name: gemini.KnownTtsModels | (gemini.TTSModelName & {}),
    config: gemini.GeminiTtsConfig
  ): ModelReference<gemini.GeminiTtsConfigSchemaType>;
  model(
    name: gemini.KnownImageModels | (gemini.ImageModelName & {}),
    config: gemini.GeminiImageConfig
  ): ModelReference<gemini.GeminiImageConfigSchemaType>;
  model(
    name: gemini.KnownGeminiModels | (gemini.GeminiModelName & {}),
    config?: gemini.GeminiConfig
  ): ModelReference<gemini.GeminiConfigSchemaType>;
  model(
    name: imagen.KnownModels | (imagen.ImagenModelName & {}),
    config?: imagen.ImagenConfig
  ): ModelReference<imagen.ImagenConfigSchemaType>;
  model(
    name: veo.KnownModels | (veo.VeoModelName & {}),
    config?: veo.VeoConfig
  ): ModelReference<veo.VeoConfigSchemaType>;
  model(
    name: deepResearch.KnownModels | (deepResearch.DeepResearchModelName & {}),
    config?: deepResearch.DeepResearchConfig
  ): ModelReference<deepResearch.DeepResearchConfigSchemaType>;
  model(
    name: lyria.KnownModels | (lyria.LyriaModelName & {}),
    config?: lyria.LyriaConfig
  ): ModelReference<lyria.LyriaConfigSchemaType>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;

  embedder(
    name: embedder.KnownModels | (embedder.EmbedderModelName & {}),
    config?: embedder.EmbeddingConfig
  ): EmbedderReference<embedder.EmbeddingConfigSchemaType>;
};

/**
 * Google Gemini Developer API plugin.
 */
export const googleAI = googleAIPlugin as GoogleAIPlugin;
(googleAI as any).model = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  if (veo.isVeoModelName(name)) {
    return veo.model(name, config);
  }
  if (deepResearch.isDeepResearchModelName(name)) {
    return deepResearch.model(name, config);
  }
  if (lyria.isLyriaModelName(name)) {
    return lyria.model(name, config);
  }
  if (imagen.isImagenModelName(name)) {
    return imagen.model(name, config);
  }
  // gemma, tts, image, gemini and unknown model families.
  return gemini.model(name, config);
};
googleAI.embedder = (
  name: string,
  config?: embedder.EmbeddingConfig
): EmbedderReference<embedder.EmbeddingConfigSchemaType> => {
  return embedder.model(name, config);
};

export default googleAI;
