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
import { getDerivedParams } from '../common/index.js';
import { vertexEvaluators } from './evaluation.js';
import type { PluginOptions } from './types.js';
export { VertexAIEvaluationMetricType } from './types.js';
export type { PluginOptions };

/**
 * Add Google Cloud Vertex AI evaluation metrics to Genkit.
 */
export function vertexAIEvaluation(options: PluginOptions): GenkitPlugin {
  return genkitPlugin('vertexAIEvaluation', async (ai: Genkit) => {
    const { projectId, location, authClient } = await getDerivedParams(options);

    vertexEvaluators(ai, authClient, options.metrics, projectId, location);
  });
}
