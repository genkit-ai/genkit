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
  EmbedderReference,
  GenkitError,
  ModelReference,
  z,
} from 'genkit';
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
import * as antigravity from './antigravity.js';
import * as deepResearch from './deep-research.js';
import * as embedder from './embedder.js';
import * as gemini from './gemini.js';
import * as lyria from './lyria.js';

export {
  type GeminiInteraction,
  type TextAnnotation,
  type UrlCitation,
} from '../common/interaction-types.js';
export { type AntigravityConfig } from './antigravity.js';
export { type DeepResearchConfig } from './deep-research.js';
export { type EmbeddingConfig } from './embedder.js';
export {
  type GeminiConfig,
  type GeminiImageConfig,
  type GeminiTtsConfig,
  type GeminiVideoConfig,
} from './gemini.js';
export { type LyriaConfig } from './lyria.js';
export { type GoogleAIPluginOptions };

/**
 * Veo models on the Gemini API have been shut down (Oct 22 2026; the server
 * returns 404). Without this, `veo-*` names would resolve as Gemini models
 * and fail with a confusing error. vertexAI still has Veo.
 */
const VEO_SHUT_DOWN_ERROR = (name: string) =>
  new GenkitError({
    status: 'NOT_FOUND',
    message:
      `Model '${name}' is not available: Veo models on the Gemini API have ` +
      'been shut down. Use gemini-omni-1.1-flash instead.',
  });

async function initializer(options?: GoogleAIPluginOptions) {
  return [
    ...gemini.listKnownModels(options),
    ...embedder.listKnownModels(options),
    ...deepResearch.listKnownModels(options),
    ...lyria.listKnownModels(options),
    ...antigravity.listKnownModels(options),
  ];
}

async function resolver(
  actionType: ActionType,
  actionName: string,
  options: GoogleAIPluginOptions
): Promise<ResolvableAction | undefined> {
  switch (actionType) {
    case 'model':
      if (deepResearch.isDeepResearchModelName(actionName)) {
        // A background model. Do not define it here.
        return undefined;
      } else if (actionName.startsWith('imagen-')) {
        // Otherwise these would resolve as Gemini models and fail with a
        // confusing "model not found".
        throw new GenkitError({
          status: 'NOT_FOUND',
          message:
            `Model '${actionName}' is not available: Imagen models on the ` +
            'Gemini API have been shut down. Use gemini-nano-banana-2.1 ' +
            'instead.',
        });
      } else if (actionName.startsWith('veo-')) {
        throw VEO_SHUT_DOWN_ERROR(actionName);
      } else if (lyria.isLyriaModelName(actionName)) {
        return lyria.defineModel(actionName, options);
      } else if (antigravity.isAntigravityModelName(actionName)) {
        return antigravity.defineModel(actionName, options);
      } else {
        // gemini, tts, image, video, gemma, unknown models
        return gemini.defineModel(actionName, options);
      }
    case 'background-model':
      if (actionName.startsWith('veo-')) {
        throw VEO_SHUT_DOWN_ERROR(actionName);
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
      ...deepResearch.listActions(allModels),
      ...lyria.listActions(allModels),
      ...antigravity.listActions(allModels),
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
  let listActionsCache: ActionMetadata[];
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
    name: gemini.KnownVideoModels | (gemini.VideoModelName & {}),
    config?: gemini.GeminiVideoConfig
  ): ModelReference<gemini.GeminiVideoConfigSchemaType>;
  model(
    name: gemini.KnownGeminiModels | (gemini.GeminiModelName & {}),
    config?: gemini.GeminiConfig
  ): ModelReference<gemini.GeminiConfigSchemaType>;
  model(
    name: deepResearch.KnownModels | (deepResearch.DeepResearchModelName & {}),
    config?: deepResearch.DeepResearchConfig
  ): ModelReference<deepResearch.DeepResearchConfigSchemaType>;
  model(
    name: lyria.KnownModels | (lyria.LyriaModelName & {}),
    config?: lyria.LyriaConfig
  ): ModelReference<lyria.LyriaConfigSchemaType>;
  model(
    name: antigravity.KnownModels | (antigravity.AntigravityModelName & {}),
    config?: antigravity.AntigravityConfig
  ): ModelReference<antigravity.AntigravityConfigSchemaType>;
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
  if (deepResearch.isDeepResearchModelName(name)) {
    return deepResearch.model(name, config);
  }
  if (lyria.isLyriaModelName(name)) {
    return lyria.model(name, config);
  }
  if (antigravity.isAntigravityModelName(name)) {
    return antigravity.model(name, config);
  }
  // gemma, tts, image, video, gemini and unknown model families.
  return gemini.model(name, config);
};
googleAI.embedder = (
  name: string,
  config?: embedder.EmbeddingConfig
): EmbedderReference<embedder.EmbeddingConfigSchemaType> => {
  return embedder.model(name, config);
};

export default googleAI;
