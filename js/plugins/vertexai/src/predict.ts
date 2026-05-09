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

import type { GoogleAuth } from 'google-auth-library';
import { getGenkitClientHeader } from './common/index.js';
import type { PluginOptions } from './common/types.js';

function endpoint(options: {
  projectId: string;
  location: string;
  model: string;
}) {
  return (
    `https://${options.location}-aiplatform.googleapis.com/v1/` +
    `projects/${options.projectId}/locations/${options.location}/` +
    `publishers/google/models/${options.model}:predict`
  );
}

interface PredictionResponse<R> {
  predictions: R[];
}

export type PredictClient<I = unknown, R = unknown, P = unknown> = (
  instances: I[],
  parameters: P
) => Promise<PredictionResponse<R>>;

export function predictModel<I = unknown, R = unknown, P = unknown>(
  auth: GoogleAuth,
  { location, projectId }: PluginOptions,
  model: string
): PredictClient<I, R, P> {
  return async (
    instances: I[],
    parameters: P
  ): Promise<PredictionResponse<R>> => {
    const fetch = (await import('node-fetch')).default;

    const accessToken = await auth.getAccessToken();
    const req = {
      instances,
      parameters,
    };

    const response = await fetch(
      endpoint({
        projectId: projectId!,
        location,
        model,
      }),
      {
        method: 'POST',
        body: JSON.stringify(req),
        headers: {
          Authorization: `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
          'X-Goog-Api-Client': getGenkitClientHeader(),
        },
      }
    );

    if (!response.ok) {
      throw new Error(
        `Error from Vertex AI predict: HTTP ${
          response.status
        }: ${await response.text()}`
      );
    }

    return (await response.json()) as PredictionResponse<R>;
  };
}
