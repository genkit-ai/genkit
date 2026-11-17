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

import type { Genkit, GenkitSchema, ModelReference } from 'genkit';
import type { BaseEvalDataPoint, EvaluatorAction } from 'genkit/evaluator';
import type { InferOutput } from 'genkit/schema';
import type { ByoMetric } from '..';
import { funninessScore } from './funniness';

export const FUNNINESS: ByoMetric = {
  name: 'funniness',
};

/**
 * Create the Funniness evaluator.
 */
export function createFunninessEvaluator<
  ModelCustomOptions extends GenkitSchema,
>(
  ai: Genkit,
  judge: ModelReference<ModelCustomOptions>,
  judgeConfig?: InferOutput<ModelCustomOptions>
): EvaluatorAction {
  return ai.defineEvaluator(
    {
      name: `byo/${FUNNINESS.name}`,
      displayName: 'Funniness',
      definition:
        'Judges whether a statement is a joke and whether that joke is funny.',
    },
    async (datapoint: BaseEvalDataPoint) => {
      const score = await funninessScore(judge, datapoint, judgeConfig);
      return {
        testCaseId: datapoint.testCaseId,
        evaluation: score,
      };
    }
  );
}
