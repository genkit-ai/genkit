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

import type { Genkit, GenkitSchema, ModelArgument } from 'genkit';
import {
  EvalStatusEnum,
  type BaseEvalDataPoint,
  type Score,
} from 'genkit/evaluator';
import type { InferOutput } from 'genkit/schema';
import path from 'path';
import { getDirName, loadPromptFile, renderText } from './helper.js';

export async function answerAccuracyScore<
  CustomModelOptions extends GenkitSchema,
>(
  ai: Genkit,
  judgeLlm: ModelArgument<CustomModelOptions>,
  dataPoint: BaseEvalDataPoint,
  judgeConfig?: InferOutput<CustomModelOptions>
): Promise<Score> {
  if (!dataPoint.output) {
    throw new Error('Output was not provided');
  }
  if (!dataPoint.reference) {
    throw new Error('Reference was not provided');
  }
  const input =
    typeof dataPoint.input === 'string'
      ? dataPoint.input
      : JSON.stringify(dataPoint.input);
  const output =
    typeof dataPoint.output === 'string'
      ? dataPoint.output
      : JSON.stringify(dataPoint.output);
  const reference =
    typeof dataPoint.reference === 'string'
      ? dataPoint.reference
      : JSON.stringify(dataPoint.reference);

  const prompt = await loadPromptFile(
    path.resolve(getDirName(), '../../prompts/answer_accuracy.prompt')
  );
  const origResp = await ai.generate({
    model: judgeLlm,
    config: judgeConfig,
    prompt: await renderText(prompt, {
      query: input,
      output,
      reference,
    }),
  });
  const origScore = Number.parseInt(origResp.text);
  if (Number.isNaN(origScore)) {
    throw new Error('Error generating original response for answer accuracy');
  }

  const invResp = await ai.generate({
    model: judgeLlm,
    config: judgeConfig,
    prompt: await renderText(prompt, {
      query: input,
      output: reference,
      reference: output,
    }),
  });
  const invScore = Number.parseInt(invResp.text);
  if (Number.isNaN(invScore)) {
    throw new Error('Error generating inverted response for answer accuracy');
  }
  const score = (origScore + invScore) / 8;

  return {
    score,
    status: score >= 0.5 ? EvalStatusEnum.PASS : EvalStatusEnum.FAIL,
  };
}
