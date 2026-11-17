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

import similarity from 'compute-cosine-similarity';
import { z, type Genkit, type GenkitSchema, type ModelArgument } from 'genkit';
import type { EmbedderArgument } from 'genkit/embedder';
import {
  EvalStatusEnum,
  type BaseEvalDataPoint,
  type Score,
} from 'genkit/evaluator';
import type { InferOutput } from 'genkit/schema';
import path from 'path';
import { getDirName, loadPromptFile, renderText } from './helper.js';

const AnswerRelevancyResponseSchema = z.object({
  question: z.string(),
  answered: z.boolean(),
  noncommittal: z.boolean(),
});

export async function answerRelevancyScore<
  CustomModelOptions extends GenkitSchema,
  CustomEmbedderOptions extends GenkitSchema,
>(
  ai: Genkit,
  judgeLlm: ModelArgument<CustomModelOptions>,
  dataPoint: BaseEvalDataPoint,
  embedder: EmbedderArgument<CustomEmbedderOptions>,
  judgeConfig?: InferOutput<CustomModelOptions>,
  embedderOptions?: InferOutput<CustomEmbedderOptions>
): Promise<Score> {
  try {
    if (!dataPoint.input) {
      throw new Error('Input was not provided');
    }
    if (!dataPoint.output) {
      throw new Error('Output was not provided');
    }
    if (!dataPoint.context?.length) {
      throw new Error('Context was not provided');
    }

    const input =
      typeof dataPoint.input === 'string'
        ? dataPoint.input
        : JSON.stringify(dataPoint.input);
    const output =
      typeof dataPoint.output === 'string'
        ? dataPoint.output
        : JSON.stringify(dataPoint.output);
    const context = dataPoint.context.map((i) => JSON.stringify(i));

    const prompt = await loadPromptFile(
      path.resolve(getDirName(), '../../prompts/answer_relevancy.prompt')
    );
    const response = await ai.generate({
      model: judgeLlm,
      config: judgeConfig,
      prompt: await renderText(prompt, {
        question: input,
        answer: output,
        context: context.join(' '),
      }),
      output: {
        schema: AnswerRelevancyResponseSchema,
      },
    });
    const genQuestion = response.output?.question;
    if (!genQuestion)
      throw new Error('Error generating question for answer relevancy');

    const questionEmbed = (
      await ai.embed({
        embedder,
        content: input,
        options: embedderOptions,
      })
    )[0].embedding; // Single embedding for text
    const genQuestionEmbed = (
      await ai.embed({
        embedder,
        content: genQuestion,
        options: embedderOptions,
      })
    )[0].embedding; // Single embedding for text
    const score = cosineSimilarity(questionEmbed, genQuestionEmbed);
    const answered = response.output?.answered ?? false;
    const isNonCommittal = response.output?.noncommittal ?? false;
    const answeredPenalty = !answered ? 0.5 : 0;
    const adjustedScore =
      score - answeredPenalty < 0 ? 0 : score - answeredPenalty;
    const reasoning = isNonCommittal
      ? 'Noncommittal'
      : answered
        ? 'Cosine similarity'
        : 'Cosine similarity with penalty for insufficient answer';
    const finalScore = adjustedScore * (isNonCommittal ? 0 : 1);
    return {
      score: finalScore,
      details: {
        reasoning,
      },
      status: finalScore > 0.5 ? EvalStatusEnum.PASS : EvalStatusEnum.FAIL,
    };
  } catch (err) {
    console.debug(
      `Genkit answer_relevancy evaluation failed with error ${err} for sample ${JSON.stringify(
        dataPoint
      )}`
    );
    throw err;
  }
}

function cosineSimilarity(v1: Array<number>, v2: Array<number>) {
  const maybeScore = similarity(v1, v2);
  if (!maybeScore) {
    throw new Error('Unable to compute cosine similarity');
  }
  return Math.abs(maybeScore);
}
