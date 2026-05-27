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

import { z } from 'genkit';
import type { GoogleAuth } from 'google-auth-library';
import { isConfig, type ChecksEvaluationMetric } from './metrics.js';

const GUARDRAILS_URL =
  'https://checks.googleapis.com/v1alpha/aisafety:classifyContent';

/**
 * Request to the Checks Guardrails ClisifyContent endpoint.
 */
type GuardrailsRequest = {
  input: {
    text_input: {
      content: string;
    };
  };
  policies: {
    policy_type: string;
    threshold?: number;
  }[];
};

/**
 * Response type returned by the classifyContent endpoint.
 */
const ResponseSchema = z.object({
  policyResults: z.array(
    z.object({
      policyType: z.string(),
      score: z.number().optional(),
      violationResult: z.string(),
    })
  ),
});

type ResponseType = z.infer<typeof ResponseSchema>;

/**
 * API implementation for making requests to the guardrails api.
 */
export class Guardrails {
  private auth: GoogleAuth;
  private projectId: string | undefined;

  constructor(auth: GoogleAuth, projectId?: string) {
    this.auth = auth;
    this.projectId = projectId;
  }

  async classifyContent(
    content: string,
    policies: ChecksEvaluationMetric[]
  ): Promise<ResponseType> {
    const body: GuardrailsRequest = {
      input: {
        text_input: {
          content: content,
        },
      },
      policies: policies.map((policy) => {
        const policyType = isConfig(policy) ? policy.type : policy;
        const threshold = isConfig(policy) ? policy.threshold : undefined;

        return {
          policy_type: policyType,
          threshold: threshold,
        };
      }),
    };

    const client = await this.auth.getClient();
    const response = await client.request({
      url: GUARDRAILS_URL,
      method: 'POST',
      body: JSON.stringify(body),
      headers: {
        'x-goog-user-project': this.projectId,
        'Content-Type': 'application/json',
      },
    });

    try {
      return ResponseSchema.parse(response.data);
    } catch (e) {
      throw new Error(`Error parsing ${GUARDRAILS_URL} API response: ${e}`);
    }
  }
}
