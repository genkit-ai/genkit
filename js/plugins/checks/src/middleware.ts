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

import type { ModelMiddleware } from 'genkit/model';
import type { GoogleAuth } from 'google-auth-library';
import { Guardrails } from './guardrails.js';
import type { ChecksEvaluationMetric } from './metrics.js';

export function checksMiddleware(options: {
  auth: GoogleAuth;
  metrics: ChecksEvaluationMetric[];
  projectId?: string;
}): ModelMiddleware {
  const guardrails = new Guardrails(options.auth, options?.projectId);

  const classifyContent = async (content: string) => {
    const response = await guardrails.classifyContent(content, options.metrics);

    // Filter for violations
    const violatedPolicies = response.policyResults.filter(
      (policy) => policy.violationResult === 'VIOLATIVE'
    );

    return violatedPolicies;
  };

  return async (req, next) => {
    for (const message of req.messages) {
      for (const content of message.content) {
        if (content.text) {
          const violatedPolicies = await classifyContent(content.text);

          // If any input message violates a checks policy. Stop processing,
          // return a blocked response and list of violated policies.
          if (violatedPolicies.length > 0) {
            return {
              finishReason: 'blocked',
              finishMessage: `Model input violated Checks policies: [${violatedPolicies.map((result) => result.policyType).join(' ')}], further processing blocked.`,
            };
          }
        }
      }
    }

    const generatedContent = await next(req);

    for (const candidate of generatedContent.candidates ?? []) {
      for (const content of candidate.message.content ?? []) {
        if (content.text) {
          const violatedPolicies = await classifyContent(content.text);

          // If the output message violates a checks policy. Stop processing,
          // return a blocked response and list of violated policies.
          if (violatedPolicies.length > 0) {
            return {
              finishReason: 'blocked',
              finishMessage: `Model output violated Checks policies: [${violatedPolicies.map((result) => result.policyType).join(' ')}], output blocked.`,
            };
          }
        }
      }
    }

    return generatedContent;
  };
}
