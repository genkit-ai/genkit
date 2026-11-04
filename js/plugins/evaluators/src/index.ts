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

import { z, type Genkit, type GenkitSchema } from 'genkit';
import {
  evaluatorRef,
  type BaseEvalDataPoint,
  type EvalResponse,
  type EvalStatusEnum,
  type Score,
} from 'genkit/evaluator';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import { answerAccuracyScore } from './metrics/answer_accuracy.js';
import {
  answerRelevancyScore,
  deepEqual,
  faithfulnessScore,
  jsonata,
  maliciousnessScore,
  regexp,
} from './metrics/index.js';
import {
  GenkitMetric,
  isGenkitMetricConfig,
  type AnswerRelevancyGenkitMetricConfig,
  type GenkitMetricConfig,
  type PluginOptions,
  type ResolvedConfig,
} from './types.js';
export { GenkitMetric, type GenkitMetricConfig, type PluginOptions };

const PLUGIN_NAME = 'genkitEval';

/**
 * Reference to the Genkit evaluator for a specified metric
 */
export const genkitEvalRef = (metric: GenkitMetric) =>
  evaluatorRef({
    name: `${PLUGIN_NAME}/${metric.toLocaleLowerCase()}`,
    configSchema: z.undefined(),
    info: {
      label: `Genkit RAG Evaluator for ${metric}`,
      metrics: [metric],
    },
  });

/**
 * Genkit evaluation plugin that provides the RAG evaluators
 */
export function genkitEval<
  ModelCustomOptions extends GenkitSchema,
  EmbedderCustomOptions extends GenkitSchema,
>(
  params: PluginOptions<ModelCustomOptions, EmbedderCustomOptions>
): GenkitPlugin {
  return genkitPlugin(`${PLUGIN_NAME}`, async (ai: Genkit) => {
    genkitEvaluators(ai, params);
  });
}

export default genkitEval;

function fillScores(
  dataPoint: BaseEvalDataPoint,
  score: Score,
  statusOverrideFn?: (args: { score: Score }) => EvalStatusEnum
): EvalResponse {
  let status = score.status;
  if (statusOverrideFn) {
    status = statusOverrideFn({ score });
  }
  return { testCaseId: dataPoint.testCaseId, evaluation: { ...score, status } };
}

/**
 * Configures a Genkit evaluator
 */
export function genkitEvaluators<
  ModelCustomOptions extends GenkitSchema,
  EmbedderCustomOptions extends GenkitSchema,
>(
  ai: Genkit,
  params: PluginOptions<ModelCustomOptions, EmbedderCustomOptions>
) {
  const { metrics } = params;
  if (metrics.length === 0) {
    throw new Error('No metrics configured in genkitEval plugin');
  }
  return metrics.map((metric) => {
    const {
      type,
      judge,
      judgeConfig,
      embedder,
      embedderOptions,
      statusOverrideFn,
    } = resolveConfig(metric, params);
    const evaluator = `${PLUGIN_NAME}/${type.toLocaleLowerCase()}`;
    switch (type) {
      case GenkitMetric.ANSWER_RELEVANCY: {
        if (!judge) {
          throw new Error(
            'Judge llms must be specified if computing answer relvancy'
          );
        }
        if (!embedder) {
          throw new Error(
            'Embedder must be specified if computing answer relvancy'
          );
        }
        return ai.defineEvaluator(
          {
            name: evaluator,
            displayName: 'Answer Relevancy',
            definition:
              'Assesses how pertinent the generated answer is to the given prompt',
          },
          async (datapoint: BaseEvalDataPoint) => {
            const answerRelevancy = await answerRelevancyScore(
              ai,
              judge!,
              datapoint,
              embedder!,
              judgeConfig,
              embedderOptions
            );
            return fillScores(datapoint, answerRelevancy, statusOverrideFn);
          }
        );
      }
      case GenkitMetric.FAITHFULNESS: {
        if (!judge) {
          throw new Error(
            'Judge llms must be specified if computing faithfulness'
          );
        }
        return ai.defineEvaluator(
          {
            name: evaluator,
            displayName: 'Faithfulness',
            definition:
              'Measures the factual consistency of the generated answer against the given context',
          },
          async (datapoint: BaseEvalDataPoint) => {
            const faithfulness = await faithfulnessScore(
              ai,
              judge!,
              datapoint,
              judgeConfig
            );
            return fillScores(datapoint, faithfulness, statusOverrideFn);
          }
        );
      }
      case GenkitMetric.MALICIOUSNESS: {
        if (!judge) {
          throw new Error(
            'Judge llms must be specified if computing maliciousness'
          );
        }
        return ai.defineEvaluator(
          {
            name: evaluator,
            displayName: 'Maliciousness',
            definition:
              'Measures whether the generated output intends to deceive, harm, or exploit',
          },
          async (datapoint: BaseEvalDataPoint) => {
            const maliciousness = await maliciousnessScore(
              ai,
              judge!,
              datapoint,
              judgeConfig
            );
            return fillScores(datapoint, maliciousness, statusOverrideFn);
          }
        );
      }
      case GenkitMetric.ANSWER_ACCURACY: {
        if (!judge) {
          throw new Error(
            'Judge llms must be specified if computing answer accuracy'
          );
        }
        return ai.defineEvaluator(
          {
            name: evaluator,
            displayName: 'Answer Accuracy',
            definition:
              'Measures how accurately the generated output matches against the reference output',
          },
          async (datapoint: BaseEvalDataPoint) => {
            const answerAccuracy = await answerAccuracyScore(
              ai,
              judge!,
              datapoint,
              judgeConfig
            );
            return fillScores(datapoint, answerAccuracy, statusOverrideFn);
          }
        );
      }
      case GenkitMetric.REGEX: {
        return ai.defineEvaluator(
          {
            name: evaluator,
            displayName: 'RegExp',
            definition: 'Tests output against the regexp provided as reference',
          },
          async (datapoint: BaseEvalDataPoint) => {
            return fillScores(datapoint, await regexp(datapoint));
          }
        );
      }
      case GenkitMetric.DEEP_EQUAL: {
        return ai.defineEvaluator(
          {
            name: evaluator,
            displayName: 'Deep Equals',
            definition:
              'Tests equality of output against the provided reference',
          },
          async (datapoint: BaseEvalDataPoint) => {
            return fillScores(
              datapoint,
              await deepEqual(datapoint),
              statusOverrideFn
            );
          }
        );
      }
      case GenkitMetric.JSONATA: {
        return ai.defineEvaluator(
          {
            name: evaluator,
            displayName: 'JSONata',
            definition:
              'Tests JSONata expression (provided in reference) against output',
          },
          async (datapoint: BaseEvalDataPoint) => {
            return fillScores(
              datapoint,
              await jsonata(datapoint),
              statusOverrideFn
            );
          }
        );
      }
    }
  });
}

function resolveConfig<M extends GenkitSchema, E extends GenkitSchema>(
  metric: GenkitMetricConfig<M, E>,
  params: PluginOptions<M, E>
): ResolvedConfig<M, E> {
  if (isGenkitMetricConfig(metric)) {
    return {
      type: metric.type,
      statusOverrideFn: metric.statusOverrideFn,
      judge: metric.judge ?? params.judge,
      judgeConfig: metric.judgeConfig ?? params.judgeConfig,
      embedder:
        metric.type === GenkitMetric.ANSWER_RELEVANCY
          ? (metric as AnswerRelevancyGenkitMetricConfig<M, E>).embedder
          : undefined,
      embedderOptions:
        metric.type === GenkitMetric.ANSWER_RELEVANCY
          ? (metric as AnswerRelevancyGenkitMetricConfig<M, E>).embedderOptions
          : undefined,
    } as ResolvedConfig<M, E>;
  }
  return { type: metric, ...params };
}
