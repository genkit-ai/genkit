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

import type {
  EmbedderArgument,
  EmbedderReference,
  ModelArgument,
  ModelReference,
  z,
} from 'genkit';
import type { EvalStatusEnum, Score } from 'genkit/evaluator';

export enum GenkitMetric {
  FAITHFULNESS = 'FAITHFULNESS',
  ANSWER_RELEVANCY = 'ANSWER_RELEVANCY',
  ANSWER_ACCURACY = 'ANSWER_ACCURACY',
  MALICIOUSNESS = 'MALICIOUSNESS',
  REGEX = 'REGEX',
  DEEP_EQUAL = 'DEEP_EQUAL',
  JSONATA = 'JSONATA',
}

export interface BaseGenkitMetricConfig {
  type: GenkitMetric;
  statusOverrideFn?: (args: { score: Score }) => EvalStatusEnum;
}

export interface FaithfulnessGenkitMetricConfig<
  ModelCustomOptions extends z.ZodType<any, any>,
> extends BaseGenkitMetricConfig {
  type: GenkitMetric.FAITHFULNESS;
  judge: ModelReference<ModelCustomOptions>;
  judgeConfig?: z.infer<ModelCustomOptions>;
}

export interface MaliciousnessGenkitMetricConfig<
  ModelCustomOptions extends z.ZodType<any, any>,
> extends BaseGenkitMetricConfig {
  type: GenkitMetric.MALICIOUSNESS;
  judge: ModelReference<ModelCustomOptions>;
  judgeConfig?: z.infer<ModelCustomOptions>;
}

export interface AnswerAccuracyGenkitMetricConfig<
  ModelCustomOptions extends z.ZodType<any, any>,
> extends BaseGenkitMetricConfig {
  type: GenkitMetric.ANSWER_ACCURACY;
  judge: ModelReference<ModelCustomOptions>;
  judgeConfig?: z.infer<ModelCustomOptions>;
}

export interface AnswerRelevancyGenkitMetricConfig<
  ModelCustomOptions extends z.ZodType<any, any>,
  EmbedderCustomOptions extends z.ZodType<any, any>,
> extends BaseGenkitMetricConfig {
  type: GenkitMetric.ANSWER_RELEVANCY;
  judge: ModelReference<ModelCustomOptions>;
  judgeConfig?: z.infer<ModelCustomOptions>;
  embedder: EmbedderReference<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}
export type GenkitMetricConfig<
  M extends z.ZodType<any, any>,
  E extends z.ZodType<any, any>,
> =
  | GenkitMetric
  | FaithfulnessGenkitMetricConfig<M>
  | MaliciousnessGenkitMetricConfig<M>
  | AnswerAccuracyGenkitMetricConfig<M>
  | AnswerRelevancyGenkitMetricConfig<M, E>;

export interface PluginOptions<
  ModelCustomOptions extends z.ZodType<any, any>,
  EmbedderCustomOptions extends z.ZodType<any, any>,
> {
  metrics: Array<GenkitMetricConfig<ModelCustomOptions, EmbedderCustomOptions>>;
  judge?: ModelArgument<ModelCustomOptions>;
  judgeConfig?: z.infer<ModelCustomOptions>;
  embedder?: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}

export type ResolvedConfig<
  ModelCustomOptions extends z.ZodType<any, any>,
  EmbedderCustomOptions extends z.ZodType<any, any>,
> = Omit<PluginOptions<ModelCustomOptions, EmbedderCustomOptions>, 'metrics'> &
  BaseGenkitMetricConfig;

export function isGenkitMetricConfig(
  input: any
): input is BaseGenkitMetricConfig {
  return Object.hasOwn(input, 'type');
}
