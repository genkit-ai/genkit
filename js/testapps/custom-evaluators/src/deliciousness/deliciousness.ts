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

import { z, type GenkitSchema, type ModelArgument } from 'genkit';
import type { BaseEvalDataPoint, Score } from 'genkit/evaluator';
import type { InferOutput } from 'genkit/schema';
import { ai } from '../index.js';

const DELICIOUSNESS_VALUES = ['yes', 'no', 'maybe'] as const;

// Define the response schema expected from the LLM
const DeliciousnessDetectionResponseSchema = z.object({
  reason: z.string(),
  verdict: z.enum(DELICIOUSNESS_VALUES),
});

const deliciousnessPrompt = ai.prompt('deliciousness');

export async function deliciousnessScore<
  CustomModelOptions extends GenkitSchema,
>(
  judgeLlm: ModelArgument<CustomModelOptions>,
  dataPoint: BaseEvalDataPoint,
  judgeConfig?: InferOutput<CustomModelOptions>
): Promise<Score> {
  const d = dataPoint;
  try {
    if (!d.output) {
      throw new Error('Output is required for Funniness detection');
    }
    const finalPrompt = (
      await deliciousnessPrompt.render({
        output: d.output as string,
      })
    ).messages
      ?.map((m) => m.content.map((c) => c.text).join())
      .join();
    const response = await ai.generate({
      model: judgeLlm,
      prompt: finalPrompt,
      config: judgeConfig,
      output: {
        schema: DeliciousnessDetectionResponseSchema,
      },
    });
    const parsedResponse = response.output;
    if (!parsedResponse) {
      throw new Error(`Unable to parse evaluator response: ${response.text}`);
    }
    return {
      score: parsedResponse.verdict,
      details: { reasoning: parsedResponse.reason },
    };
  } catch (err) {
    console.debug(
      `BYO deliciousness evaluation failed with error ${err} for sample ${JSON.stringify(
        d
      )}`
    );
    throw err;
  }
}
