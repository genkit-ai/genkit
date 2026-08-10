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
import { googleAI } from '@genkit-ai/google-genai';
import { genkit, type Genkit, type ModelReference, type z } from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import {
  PERMISSIVE_SAFETY_SETTINGS,
  URL_REGEX,
  US_PHONE_REGEX,
} from './constants.js';
import {
  DELICIOUSNESS,
  createDeliciousnessEvaluator,
} from './deliciousness/deliciousness_evaluator.js';
import {
  FUNNINESS,
  createFunninessEvaluator,
} from './funniness/funniness_evaluator.js';
import { PII_DETECTION, createPiiEvaluator } from './pii/pii_evaluator.js';
import {
  createRegexEvaluators,
  isRegexMetric,
  regexMatcher,
  type RegexMetric,
} from './regex/regex_evaluator.js';

export const ai = genkit({
  plugins: [
    googleAI(),
    byoEval({
      judge: googleAI.model('gemini-2.5-flash'),
      judgeConfig: PERMISSIVE_SAFETY_SETTINGS,
      metrics: [
        // regexMatcher will register an evaluator with a name in the format
        // byo/regex_match_{suffix}. In this case, byo/regex_match_url
        regexMatcher('url', URL_REGEX),
        // byo/regex_match_us_phone
        regexMatcher('us_phone', US_PHONE_REGEX),
        PII_DETECTION,
        DELICIOUSNESS,
        FUNNINESS,
      ],
    }),
  ],
});

/**
 * Generic metric definition with flexible configuration.
 */
export interface ByoMetric {
  name: string;
}

/**
 * Plugin option definition for the PII evaluator
 */
export interface PluginOptions<ModelCustomOptions extends z.ZodTypeAny> {
  judge: ModelReference<ModelCustomOptions>;
  judgeConfig?: z.infer<ModelCustomOptions>;
  metrics?: Array<ByoMetric>;
}

/**
 * Configurable Eval plugin that provides different kinds of custom evaluators.
 */
export function byoEval<ModelCustomOptions extends z.ZodTypeAny>(
  params: PluginOptions<ModelCustomOptions>
): GenkitPlugin {
  // Define the new plugin
  return genkitPlugin('byo', async (ai: Genkit) => {
    const { judge, judgeConfig, metrics } = params;
    if (!metrics) {
      throw new Error(`Found no configured metrics.`);
    }
    const regexMetrics = metrics?.filter((metric) => isRegexMetric(metric));
    const hasPiiMetric = metrics?.includes(PII_DETECTION);
    const hasFunninessMetric = metrics?.includes(FUNNINESS);
    const hasDelicousnessMetric = metrics?.includes(DELICIOUSNESS);

    if (regexMetrics) {
      createRegexEvaluators(ai, regexMetrics as RegexMetric[]);
    }

    if (hasPiiMetric) {
      createPiiEvaluator(ai, judge, judgeConfig);
    }

    if (hasFunninessMetric) {
      createFunninessEvaluator(ai, judge, judgeConfig);
    }

    if (hasDelicousnessMetric) {
      createDeliciousnessEvaluator(ai, judge, judgeConfig);
    }
  });
}
