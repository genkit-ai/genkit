/**
 * @license
 *
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

/**
 * Evaluator types — schemas, reference helpers, and action types for building
 * AI output quality evaluators.
 *
 * ```ts
 * import { evaluatorRef, type EvalResponse } from 'genkit/evaluator';
 * ```
 *
 * @module evaluator
 */

export {
  BaseDataPointSchema,
  BaseEvalDataPointSchema,
  EvalResponseSchema,
  EvalResponsesSchema,
  EvalStatusEnum,
  EvaluatorInfoSchema,
  ScoreSchema,
  evaluatorRef,
  type BaseDataPoint,
  type BaseEvalDataPoint,
  type Dataset,
  type EvalResponse,
  type EvalResponses,
  type EvaluatorAction,
  type EvaluatorArgument,
  type EvaluatorInfo,
  type EvaluatorParams,
  type EvaluatorReference,
  type Score,
} from '@genkit-ai/ai/evaluator';
