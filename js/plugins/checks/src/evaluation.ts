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

import { z, type EvaluatorAction, type Genkit } from 'genkit';
import type { BaseEvalDataPoint } from 'genkit/evaluator';
import { runInNewSpan } from 'genkit/tracing';
import type { GoogleAuth } from 'google-auth-library';
import {
  isConfig,
  type ChecksEvaluationMetric,
  type ChecksEvaluationMetricConfig,
} from './metrics.js';

export function checksEvaluators(
  ai: Genkit,
  auth: GoogleAuth,
  metrics: ChecksEvaluationMetric[],
  projectId: string
): EvaluatorAction {
  const policy_configs: ChecksEvaluationMetricConfig[] = metrics.map(
    (metric) => {
      const metricType = isConfig(metric) ? metric.type : metric;
      const threshold = isConfig(metric) ? metric.threshold : undefined;

      return {
        type: metricType,
        threshold,
      };
    }
  );

  return createPolicyEvaluator(projectId, auth, ai, policy_configs);
}

const ResponseSchema = z.object({
  policyResults: z.array(
    z.object({
      policyType: z.string(),
      score: z.number(),
      violationResult: z.string(),
    })
  ),
});

function createPolicyEvaluator(
  projectId: string,
  auth: GoogleAuth,
  ai: Genkit,
  policy_config: ChecksEvaluationMetricConfig[]
): EvaluatorAction {
  return ai.defineEvaluator(
    {
      name: 'checks/guardrails',
      displayName: 'checks/guardrails',
      definition: `Evaluates input text against the Checks ${policy_config.map((policy) => policy.type)} policies.`,
    },
    async (datapoint: BaseEvalDataPoint) => {
      const partialRequest = {
        input: {
          text_input: {
            content: datapoint.output as string,
          },
        },
        policies: policy_config.map((config) => {
          return {
            policy_type: config.type,
            threshold: config.threshold,
          };
        }),
      };

      const response = await checksEvalInstance(
        ai,
        projectId,
        auth,
        partialRequest,
        ResponseSchema
      );

      const evaluationResults = response.policyResults.map((result) => {
        return {
          id: result.policyType,
          score: result.score,
          details: {
            reasoning: `Status ${result.violationResult}`,
          },
        };
      });

      return {
        evaluation: evaluationResults,
        testCaseId: datapoint.testCaseId,
      };
    }
  );
}

async function checksEvalInstance<ResponseType extends z.ZodTypeAny>(
  ai: Genkit,
  projectId: string,
  auth: GoogleAuth,
  partialRequest: any,
  responseSchema: ResponseType
): Promise<z.infer<ResponseType>> {
  return await runInNewSpan(
    ai,
    {
      metadata: {
        name: 'EvaluationService#evaluateInstances',
      },
    },
    async (metadata, _otSpan) => {
      const request = {
        ...partialRequest,
      };

      metadata.input = request;
      const client = await auth.getClient();
      const url =
        'https://checks.googleapis.com/v1alpha/aisafety:classifyContent';

      const response = await client.request({
        url,
        method: 'POST',
        body: JSON.stringify(request),
        headers: {
          'x-goog-user-project': projectId,
          'Content-Type': 'application/json',
        },
      });
      metadata.output = response.data;

      try {
        return responseSchema.parse(response.data);
      } catch (e) {
        throw new Error(`Error parsing ${url} API response: ${e}`);
      }
    }
  );
}
