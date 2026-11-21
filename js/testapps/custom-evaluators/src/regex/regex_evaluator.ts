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
  BaseEvalDataPoint,
  EvalResponse,
  EvaluatorAction,
  Score,
} from 'genkit/evaluator';
import { evaluator } from 'genkit/plugin';
import type { ByoMetric } from '..';

/** We allow multiple regex matchers to be defined. This is the prefix to use. */
const REGEX_MATCH_NAME_PREFIX = 'REGEX_MATCH';

/**
 * Create an EvalResponse from an individual scored datapoint.
 */
function fillScores(dataPoint: BaseEvalDataPoint, score: Score): EvalResponse {
  return {
    testCaseId: dataPoint.testCaseId,
    evaluation: score,
  };
}

/** Metric definition for a regex metric */
export interface RegexMetric extends ByoMetric {
  regex: RegExp;
}

/**
 * Generate a new regexMatcher instance.
 */
export const regexMatcher = (suffix: string, pattern: RegExp): RegexMetric => ({
  name: `${REGEX_MATCH_NAME_PREFIX}_${suffix.toUpperCase()}`,
  regex: pattern,
});

/** Determine if a ByoMetric is a RegexMetric */
export function isRegexMetric(metric: ByoMetric) {
  return metric.name.startsWith(REGEX_MATCH_NAME_PREFIX) && 'regex' in metric;
}

/**
 * Configures regex evaluators.
 */
export function createRegexEvaluators(
  metrics: RegexMetric[]
): EvaluatorAction[] {
  return metrics.map((metric) => {
    const regexMetric = metric as RegexMetric;
    return evaluator(
      {
        name: metric.name.toLocaleLowerCase(),
        displayName: 'Regex Match',
        definition:
          'Runs the output against a regex and responds with 1 if a match is found and 0 otherwise.',
        isBilled: false,
      },
      async (datapoint: BaseEvalDataPoint) => {
        const score = await regexMatchScore(datapoint, regexMetric.regex);
        return fillScores(datapoint, score);
      }
    );
  });
}

/**
 * Score an individual datapoint.
 */
export async function regexMatchScore(
  dataPoint: BaseEvalDataPoint,
  regex: RegExp
): Promise<Score> {
  const d = dataPoint;
  try {
    if (!d.output || typeof d.output !== 'string') {
      throw new Error('String output is required for regex matching');
    }
    const matches = regex.test(d.output as string);
    const reasoning = matches
      ? `Output matched regex ${regex.source}`
      : `Output did not match regex ${regex.source}`;
    return {
      score: matches,
      details: { reasoning },
    };
  } catch (err) {
    console.debug(
      `BYO regex matcher failed with error ${err} for sample ${JSON.stringify(
        d
      )}`
    );
    throw err;
  }
}
