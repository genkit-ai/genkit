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

import type { Genkit } from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import { getDerivedParams } from '../../common/index.js';
import { SUPPORTED_ANTHROPIC_MODELS, anthropicModel } from './anthropic.js';
import { SUPPORTED_MISTRAL_MODELS, mistralModel } from './mistral.js';
import {
  SUPPORTED_OPENAI_FORMAT_MODELS,
  modelGardenOpenaiCompatibleModel,
} from './model_garden.js';
import type { PluginOptions } from './types.js';

/**
 * Add Google Cloud Vertex AI Rerankers API to Genkit.
 * @deprecated Please use vertexModelGarden
 */
export function vertexAIModelGarden(options: PluginOptions): GenkitPlugin {
  return genkitPlugin('vertexAIModelGarden', async (ai: Genkit) => {
    const { projectId, location, authClient } = await getDerivedParams(options);

    options.models.forEach((m) => {
      const anthropicEntry = Object.entries(SUPPORTED_ANTHROPIC_MODELS).find(
        ([_, value]) => value.name === m.name
      );
      if (anthropicEntry) {
        anthropicModel(ai, anthropicEntry[0], projectId, location);
        return;
      }
      const mistralEntry = Object.entries(SUPPORTED_MISTRAL_MODELS).find(
        ([_, value]) => value.name === m.name
      );
      if (mistralEntry) {
        mistralModel(ai, mistralEntry[0], projectId, location);
        return;
      }
      const openaiModel = Object.entries(SUPPORTED_OPENAI_FORMAT_MODELS).find(
        ([_, value]) => value.name === m.name
      );
      if (openaiModel) {
        modelGardenOpenaiCompatibleModel(
          ai,
          openaiModel[0],
          projectId,
          location,
          authClient,
          options.openAiBaseUrlTemplate
        );
        return;
      }
      throw new Error(`Unsupported model garden model: ${m.name}`);
    });
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
