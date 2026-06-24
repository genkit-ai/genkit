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
  embedderRef,
  type EmbedderReference,
  type Genkit,
  type ModelReference,
} from 'genkit';
import { modelRef } from 'genkit/model';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import { defineTwelveLabsEmbedder } from './embedder.js';
import { defineTwelveLabsModel } from './models.js';
import {
  DEFAULT_BASE_URL,
  TwelveLabsConfigSchema,
  type TwelveLabsConfig,
  type TwelveLabsPluginParams,
} from './types.js';

export { TwelveLabsConfigSchema };
export type { TwelveLabsConfig, TwelveLabsPluginParams };

export type TwelveLabsPlugin = {
  (params?: TwelveLabsPluginParams): GenkitPlugin;

  model(
    name: string,
    config?: TwelveLabsConfig
  ): ModelReference<typeof TwelveLabsConfigSchema>;
  embedder(name: string, config?: Record<string, any>): EmbedderReference;
};

function resolveApiKey(params?: TwelveLabsPluginParams): string {
  const apiKey = params?.apiKey ?? process.env.TWELVELABS_API_KEY;
  if (!apiKey) {
    throw new Error(
      'TwelveLabs API key is required. Pass it as `apiKey` or set the ' +
        'TWELVELABS_API_KEY environment variable. Get a free key at https://twelvelabs.io.'
    );
  }
  return apiKey;
}

function twelvelabsPlugin(params?: TwelveLabsPluginParams): GenkitPlugin {
  const baseUrl = params?.baseUrl ?? DEFAULT_BASE_URL;
  return genkitPlugin('twelvelabs', async (ai: Genkit) => {
    const apiKey = resolveApiKey(params);
    params?.models?.forEach((model) =>
      defineTwelveLabsModel(ai, { apiKey, baseUrl, model })
    );
    params?.embedders?.forEach((embedder) =>
      defineTwelveLabsEmbedder(ai, { apiKey, baseUrl, embedder })
    );
  });
}

/**
 * Genkit plugin for {@link https://twelvelabs.io | TwelveLabs} video AI.
 *
 * Exposes Pegasus video-understanding models (as Genkit models) and Marengo
 * multimodal embedding models (as Genkit embedders). The plugin is fully
 * opt-in: nothing is registered unless you list `models` and/or `embedders`.
 *
 * @example
 * ```ts
 * import { genkit } from 'genkit';
 * import { twelvelabs } from 'genkitx-twelvelabs';
 *
 * const ai = genkit({
 *   plugins: [
 *     twelvelabs({
 *       models: [{ name: 'pegasus1.5' }],
 *       embedders: [{ name: 'marengo3.0', dimensions: 512 }],
 *     }),
 *   ],
 * });
 *
 * // Describe a video (Pegasus). The video is a media part with a public URL.
 * const { text } = await ai.generate({
 *   model: 'twelvelabs/pegasus1.5',
 *   messages: [
 *     {
 *       role: 'user',
 *       content: [
 *         { text: 'Describe this video.' },
 *         { media: { url: 'https://example.com/video.mp4', contentType: 'video/mp4' } },
 *       ],
 *     },
 *   ],
 * });
 *
 * // Embed text into the Marengo space (Marengo).
 * const embedding = await ai.embed({
 *   embedder: 'twelvelabs/marengo3.0',
 *   content: 'a cat playing piano',
 * });
 * ```
 */
export const twelvelabs = twelvelabsPlugin as TwelveLabsPlugin;

twelvelabs.model = (
  name: string,
  config?: TwelveLabsConfig
): ModelReference<typeof TwelveLabsConfigSchema> => {
  return modelRef({
    name: `twelvelabs/${name}`,
    config,
    configSchema: TwelveLabsConfigSchema,
  });
};

twelvelabs.embedder = (
  name: string,
  config?: Record<string, any>
): EmbedderReference => {
  return embedderRef({
    name: `twelvelabs/${name}`,
    config,
  });
};

export default twelvelabs;
