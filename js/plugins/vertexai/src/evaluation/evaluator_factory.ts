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

import { type Action, type Genkit, type z } from 'genkit';
import type { BaseEvalDataPoint, Score } from 'genkit/evaluator';
import { runInNewSpan } from 'genkit/tracing';
import type { GoogleAuth } from 'google-auth-library';
import { getGenkitClientHeader } from '../common/index.js';
import type { VertexAIEvaluationMetricType } from './evaluation.js';

export class EvaluatorFactory {
  constructor(
    private readonly auth: GoogleAuth,
    private readonly location: string,
    private readonly projectId: string
  ) {}

  create<ResponseType extends z.ZodTypeAny>(
    ai: Genkit,
    config: {
      metric: VertexAIEvaluationMetricType;
      displayName: string;
      definition: string;
      responseSchema: ResponseType;
    },
    toRequest: (datapoint: BaseEvalDataPoint) => any,
    responseHandler: (response: z.infer<ResponseType>) => Score
  ): Action {
    return ai.defineEvaluator(
      {
        name: `vertexai/${config.metric.toLocaleLowerCase()}`,
        displayName: config.displayName,
        definition: config.definition,
      },
      async (datapoint: BaseEvalDataPoint) => {
        const responseSchema = config.responseSchema;
        const response = await this.evaluateInstances(
          ai,
          toRequest(datapoint),
          responseSchema
        );

        return {
          evaluation: responseHandler(response),
          testCaseId: datapoint.testCaseId,
        };
      }
    );
  }

  async evaluateInstances<ResponseType extends z.ZodTypeAny>(
    ai: Genkit,
    partialRequest: any,
    responseSchema: ResponseType
  ): Promise<z.infer<ResponseType>> {
    const locationName = `projects/${this.projectId}/locations/${this.location}`;
    return await runInNewSpan(
      ai,
      {
        metadata: {
          name: 'EvaluationService#evaluateInstances',
        },
      },
      async (metadata, _otSpan) => {
        const request = {
          location: locationName,
          ...partialRequest,
        };

        metadata.input = request;
        const client = await this.auth.getClient();
        const url = `https://${this.location}-aiplatform.googleapis.com/v1beta1/${locationName}:evaluateInstances`;
        const response = await client.request({
          url,
          method: 'POST',
          body: JSON.stringify(request),
          headers: {
            'X-Goog-Api-Client': getGenkitClientHeader(),
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
}
