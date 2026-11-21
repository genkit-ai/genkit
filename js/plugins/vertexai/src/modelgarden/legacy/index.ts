/**
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

import {
  genkitPlugin,
  type GenkitPlugin,
  type ResolvableAction,
} from 'genkit/plugin';
import { getDerivedParams } from '../../common/index.js';
import { SUPPORTED_ANTHROPIC_MODELS, anthropicModel } from './anthropic.js';
import { SUPPORTED_MISTRAL_MODELS, mistralModel } from './mistral.js';
import {
  SUPPORTED_OPENAI_FORMAT_MODELS,
  modelGardenOpenaiCompatibleModel,
} from './model_garden.js';
import type { PluginOptions } from './types.js';

/**
 * Add Google Cloud Vertex AI Model Garden models to Genkit.
 * @deprecated Please use vertexModelGarden
 */
export function vertexAIModelGarden(options: PluginOptions): GenkitPlugin {
  return genkitPlugin({
    name: 'vertexAIModelGarden',
    init: async () => {
      const { projectId, location, authClient } =
        await getDerivedParams(options);

      return options.models.map((m): ResolvableAction => {
        const anthropicEntry = Object.entries(SUPPORTED_ANTHROPIC_MODELS).find(
          ([_, value]) => value.name === m.name
        );
        if (anthropicEntry) {
          return anthropicModel(anthropicEntry[0], projectId, location);
        }
        const mistralEntry = Object.entries(SUPPORTED_MISTRAL_MODELS).find(
          ([_, value]) => value.name === m.name
        );
        if (mistralEntry) {
          return mistralModel(mistralEntry[0], projectId, location);
        }
        const openaiModel = Object.entries(SUPPORTED_OPENAI_FORMAT_MODELS).find(
          ([_, value]) => value.name === m.name
        );
        if (openaiModel) {
          return modelGardenOpenaiCompatibleModel(
            openaiModel[0],
            projectId,
            location,
            authClient,
            options.openAiBaseUrlTemplate
          );
        }
        throw new Error(`Unsupported model garden model: ${m.name}`);
      });
    },
  });
}

export {
  claude35Sonnet,
  claude35SonnetV2,
  claude3Haiku,
  claude3Opus,
  claude3Sonnet,
  claudeOpus4,
  claudeOpus41,
  claudeSonnet4,
} from './anthropic.js';
export { codestral, mistralLarge, mistralNemo } from './mistral.js';
export { llama3, llama31, llama32 } from './model_garden.js';
//export type { PluginOptions };  // Same one will be exported by v2 now
