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

import type { Genkit, ModelArgument } from 'genkit';
import type { BaseEvalDataPoint } from 'genkit/evaluator';
import { loadEvaluator, type Criteria } from 'langchain/evaluation';
import { genkitModel } from './model.js';
import { GenkitTracer } from './tracing.js';

export function langchainEvaluator(
  ai: Genkit,
  type: 'labeled_criteria' | 'criteria',
  criteria: Criteria,
  judgeLlm: ModelArgument,
  judgeConfig?: any
) {
  return ai.defineEvaluator(
    {
      name: `langchain/${criteria}`,
      displayName: `${criteria}`,
      definition: `${criteria}: refer to https://js.langchain.com/docs/guides/evaluation`,
    },
    async (datapoint: BaseEvalDataPoint) => {
      try {
        switch (type) {
          case 'labeled_criteria':
          case 'criteria':
            const evaluator = await loadEvaluator(
              type as 'labeled_criteria' | 'criteria',
              {
                criteria,
                // TODO: Figure out why this breaks with TypeScript 5.3
                llm: genkitModel(ai, judgeLlm, judgeConfig),
                chainOptions: {
                  callbacks: [new GenkitTracer()],
                },
              }
            );
            const lcData: {
              input: string;
              prediction: string;
              reference?: string;
            } = {
              input: datapoint.input as string,
              prediction: datapoint.output as string,
            };
            if (datapoint.reference) {
              lcData.reference = datapoint.reference as string;
            }
            const res = await evaluator.evaluateStrings(lcData);
            return {
              testCaseId: datapoint.testCaseId,
              evaluation: {
                score: res.score,
                details: {
                  reasoning: res.reasoning,
                  value: res.value,
                },
              },
            };
          default:
            throw new Error(`unsupported evaluator type ${type}`);
        }
      } catch (e) {
        return {
          testCaseId: datapoint.testCaseId,
          evaluation: {
            error: `${e}`,
          },
        };
      }
    }
  );
}
