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
  genkitPluginV2,
  ResolvableAction,
  type GenkitPluginV2,
} from 'genkit/plugin';
import { ActionType } from 'genkit/registry';

import { RerankerReference, z } from 'genkit';
import { getDerivedOptions } from '../../common/utils.js';
import * as reranker from './reranker.js';
import { type VertexRerankerPluginOptions } from './types.js';
export { type VertexRerankerPluginOptions };

async function initializer(pluginOptions: VertexRerankerPluginOptions) {
  const clientOptions = await getDerivedOptions(
    'vertex-rerankers',
    pluginOptions
  );
  return await reranker.listKnownRerankers(clientOptions);
}

async function resolver(
  actionType: ActionType,
  actionName: string,
  pluginOptions: VertexRerankerPluginOptions
): Promise<ResolvableAction | undefined> {
  const clientOptions = await getDerivedOptions(
    'vertex-rerankers',
    pluginOptions
  );
  if (actionType == 'reranker' && reranker.isRerankerModelName(actionName)) {
    return reranker.defineReranker(actionName, clientOptions);
  }
  return undefined;
}

/**
 * Add Google Cloud Vertex AI Rerankers API to Genkit.
 */
export function vertexRerankersPlugin(
  options: VertexRerankerPluginOptions
): GenkitPluginV2 {
  return genkitPluginV2({
    name: 'vertex-rerankers',
    init: async () => await initializer(options),
    resolve: async (actionType: ActionType, actionName: string) =>
      await resolver(actionType, actionName, options),
  });
}

export type VertexAIRerankers = {
  (pluginOptions?: VertexRerankerPluginOptions): GenkitPluginV2;
  reranker(
    name: reranker.KnownModels | (reranker.RerankerModelName & {}),
    config?: reranker.VertexRerankerConfig
  ): RerankerReference<reranker.VertexRerankerConfigSchemaType>;
};

export const vertexRerankers = vertexRerankersPlugin as VertexAIRerankers;
(vertexRerankers as any).reranker = (
  name: string,
  config?: any
): RerankerReference<z.ZodTypeAny> => {
  return reranker.reranker(name, config);
};
