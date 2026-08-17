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

import type { ActionMetadata, ModelReference, z } from 'genkit';
import type { ModelAction } from 'genkit/model';
import { genkitPluginV2, type GenkitPluginV2 } from 'genkit/plugin';
import type { ActionType } from 'genkit/registry';
import * as music from './music.js';
import type { MiniMaxPluginOptions } from './types.js';
import { PLUGIN_NAME, checkApiKey, modelName } from './utils.js';

export {
  MiniMaxMusicConfigSchema,
  type MiniMaxMusicConfig,
  type MiniMaxMusicConfigSchemaType,
  type MiniMaxMusicModelName,
  type KnownMusicModels,
} from './music.js';
export {
  AUDIO_URL_TTL_HOURS,
  MINIMAX_REGIONS,
  MUSIC_AUDIO_FORMATS,
  MUSIC_OUTPUT_FORMATS,
  REGION_BASE_URLS,
  type MiniMaxPluginOptions,
  type MiniMaxRegion,
  type MusicAudioFormat,
  type MusicOutputFormat,
} from './types.js';

/**
 * This module provides an interface to the MiniMax music generation API through
 * the Genkit plugin system.
 *
 * The main export is the `minimax` plugin, which can be configured with an API
 * key either directly or through the `MINIMAX_API_KEY` environment variable.
 * The `region` option selects the endpoint that requests are sent to.
 *
 * Example:
 * ```ts
 * import { minimax } from '@genkit-ai/minimax';
 * import { genkit } from 'genkit';
 *
 * const ai = genkit({
 *   plugins: [minimax({ apiKey: 'your-api-key' })],
 * });
 *
 * const { message } = await ai.generate({
 *   model: minimax.model('music-3.0'),
 *   prompt: 'An upbeat acoustic folk song about the sunrise',
 *   config: { lyrics: '[Verse]\nMorning light\n[Chorus]\nHere comes the sun' },
 * });
 *
 * // The generated audio is returned as a media part.
 * const audio = message?.content.find((part) => part.media)?.media;
 * ```
 */
function minimaxPlugin(options?: MiniMaxPluginOptions): GenkitPluginV2 {
  checkApiKey(options?.apiKey);

  let listActionsCache: ActionMetadata[] | null = null;

  return genkitPluginV2({
    name: PLUGIN_NAME,
    init: async () => music.listKnownModels(options),
    resolve: (actionType: ActionType, name: string) => {
      if (actionType !== 'model') {
        return undefined;
      }
      const version = modelName(name);
      if (!music.isMiniMaxMusicModelName(version)) {
        return undefined;
      }
      return music.defineModel(version, options) as ModelAction;
    },
    list: async () => {
      if (!listActionsCache) {
        listActionsCache = music.listActions();
      }
      return listActionsCache;
    },
  });
}

/**
 * The `minimax` plugin, which also exposes model references through
 * {@link MiniMaxPlugin.model}.
 */
export type MiniMaxPlugin = {
  (pluginOptions?: MiniMaxPluginOptions): GenkitPluginV2;
  model(
    name: music.KnownMusicModels | (music.MiniMaxMusicModelName & {}),
    config?: music.MiniMaxMusicConfig
  ): ModelReference<music.MiniMaxMusicConfigSchemaType>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
};

/**
 * MiniMax plugin for Genkit, providing the music generation models.
 */
export const minimax = minimaxPlugin as MiniMaxPlugin;
(minimax as any).model = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  return music.model(name, config);
};

export default minimax;
