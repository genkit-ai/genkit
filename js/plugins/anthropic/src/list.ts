/**
 * Copyright 2024 Bloom Labs Inc
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

import Anthropic from '@anthropic-ai/sdk';
import { modelActionMetadata } from 'genkit/plugin';

import { ActionMetadata } from 'genkit';
import { claudeModelReference } from './models.js';

/**
 * Retrieves available Anthropic models from the API and converts them into Genkit action metadata.
 *
 * This function queries the Anthropic API for the list of available models and generates metadata
 * for all discovered models.
 *
 * @param client - The Anthropic API client instance
 * @returns A promise that resolves to an array of action metadata for all discovered models
 */
export async function listActions(
  client: Anthropic
): Promise<ActionMetadata[]> {
  // Use beta.models.list to get the list of ALL (stable and beta) models
  const clientModels = (await client.beta.models.list()).data;
  const seenNames = new Set<string>();

  return clientModels
    .filter((modelInfo) => {
      const modelId = modelInfo.id;
      if (!modelId) {
        return false;
      }

      const ref = claudeModelReference(modelId);
      const name = ref.name;

      // Deduplicate by name
      if (seenNames.has(name)) {
        return false;
      }
      seenNames.add(name);
      return true;
    })
    .map((modelInfo) => {
      const modelId = modelInfo.id!;
      const ref = claudeModelReference(modelId);

      return modelActionMetadata({
        name: ref.name,
        info: ref.info,
        configSchema: ref.configSchema,
      });
    });
}
