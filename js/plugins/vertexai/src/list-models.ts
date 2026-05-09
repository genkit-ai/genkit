/**
 * Copyright 2025 Google LLC
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

// Gemini  model definition
export interface Model {
  name: string;
  launchStage: string;
}

// Gemini list models response
interface ListModelsResponse {
  publisherModels: Model[];
}

/**
 * List Gemini models by making an RPC call to the API.
 */
export async function listModels(
  authClient: GoogleAuth,
  location: string,
  projectId: string
): Promise<Model[]> {
  const fetch = (await import('node-fetch')).default;
  const accessToken = await authClient.getAccessToken();
  const response = await fetch(
    `https://${location}-aiplatform.googleapis.com/v1beta1/publishers/google/models`,
    {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${accessToken}`,
        'x-goog-user-project': projectId,
        'Content-Type': 'application/json',
        'X-Goog-Api-Client': getGenkitClientHeader(),
      },
    }
  );
  if (!response.ok) {
    const ee = await response.text();
    throw new Error(
      `Error from Vertex AI predict: HTTP ${response.status}: ${ee}`
    );
  }

  const modelResponse = (await response.json()) as ListModelsResponse;
  return modelResponse.publisherModels;
}
