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

import { z, type Genkit, type GenkitSchema, type ModelArgument } from 'genkit';
import {
  EvalStatusEnum,
  type BaseEvalDataPoint,
  type Score,
} from 'genkit/evaluator';
import type { InferOutput } from 'genkit/schema';
import path from 'path';
import { getDirName, loadPromptFile, renderText } from './helper.js';

const LongFormResponseSchema = z.object({ statements: z.array(z.string()) });

const NliResponseBaseSchema = z.object({
  statement: z.string(),
  reason: z.string(),
  verdict: z.boolean(),
});

type NliResponseBase = z.infer<typeof NliResponseBaseSchema>;
const NliResponseSchema = z.object({
  responses: z.array(NliResponseBaseSchema),
});

/**
 *
 */
export async function faithfulnessScore<
  CustomModelOptions extends GenkitSchema,
>(
  ai: Genkit,
  judgeLlm: ModelArgument<CustomModelOptions>,
  dataPoint: BaseEvalDataPoint,
  judgeConfig?: InferOutput<CustomModelOptions>
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

    const longFormPrompt = await loadPromptFile(
      path.resolve(getDirName(), '../../prompts/faithfulness_long_form.prompt')
    );
    const longFormResponse = await ai.generate({
      model: judgeLlm,
      config: judgeConfig,
      prompt: await renderText(longFormPrompt, {
        question: input,
        answer: output,
      }),
      output: {
        schema: LongFormResponseSchema,
      },
    });
    const parsedLongFormResponse = longFormResponse.output;
    const statements = parsedLongFormResponse?.statements ?? [];
    if (statements.length === 0) {
      throw new Error('No statements returned');
    }
    const allStatements = statements.map((s) => `statement: ${s}`).join('\n');
    const allContext = context.join('\n');
    const nliPrompt = await loadPromptFile(
      path.resolve(getDirName(), '../../prompts/faithfulness_nli.prompt')
    );
    const response = await ai.generate({
      model: judgeLlm,
      prompt: await renderText(nliPrompt, {
        context: allContext,
        statements: allStatements,
      }),
      output: {
        schema: NliResponseSchema,
      },
    });
    const parsedResponse = response.output;
    return nliResponseToScore(parsedResponse?.responses ?? []);
  } catch (err) {
    console.debug(
      `Genkit faithfulness evaluation failed with error ${err} for sample ${JSON.stringify(
        dataPoint
      )}`
    );
    throw err;
  }
}

function nliResponseToScore(input: NliResponseBase[] | null): Score {
  if (!input) {
    throw new Error(`Evaluator response empty`);
  }
  const faithfulStatements = input.reduce((total, resp) => {
    return total + (resp.verdict ? 1 : 0);
  }, 0);
  const score = faithfulStatements / input.length;
  return {
    score,
    details: { reasoning: input.map((r) => r.reason).join('; ') },
    status: score > 0.5 ? EvalStatusEnum.PASS : EvalStatusEnum.FAIL,
  };
}
