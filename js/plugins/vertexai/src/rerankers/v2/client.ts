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

import { GoogleAuth } from 'google-auth-library';
import { extractErrMsg } from '../../common/utils.js';
import {
  RerankRequest,
  RerankResponse,
  VertexRerankerClientOptions,
} from './types.js';

const DEFAULT_LOCATION = 'us-central1';

export async function rerankerRank(
  model: string,
  request: RerankRequest,
  clientOptions: VertexRerankerClientOptions
): Promise<RerankResponse> {
  const url = getVertexRerankUrl(clientOptions);
  const token = await getToken(clientOptions.authClient);
  const headers = {
    Authorization: `Bearer ${token}`,
    'x-goog-user-project': clientOptions.projectId,
    'Content-Type': 'application/json',
  };
  const fetchOptions = {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(request),
  };
  const response = await makeRequest(url, fetchOptions);

  return response.json() as Promise<RerankResponse>;
}

export function getVertexRerankUrl(clientOptions: VertexRerankerClientOptions) {
  const loc = clientOptions.location || DEFAULT_LOCATION;
  return `https://discoveryengine.googleapis.com/v1/projects/${clientOptions.projectId}/locations/${loc}/rankingConfigs/default_ranking_config:rank`;
}

async function getToken(authClient: GoogleAuth): Promise<string> {
  const CREDENTIAL_ERROR_MESSAGE =
    '\nUnable to authenticate your request\
        \nDepending on your run time environment, you can get authentication by\
        \n- if in local instance or cloud shell: `!gcloud auth login`\
        \n- if in Colab:\
        \n    -`from google.colab import auth`\
        \n    -`auth.authenticate_user()`\
        \n- if in service account or other: please follow guidance in https://cloud.google.com/docs/authentication';
  const token = await authClient.getAccessToken().catch((e) => {
    throw new Error(CREDENTIAL_ERROR_MESSAGE, e);
  });
  if (!token) {
    throw new Error(CREDENTIAL_ERROR_MESSAGE);
  }
  return token;
}

async function makeRequest(
  url: string,
  fetchOptions: RequestInit
): Promise<Response> {
  try {
    const response = await fetch(url, fetchOptions);
    if (!response.ok) {
      let errorText = await response.text();
      let errorMessage = errorText;
      try {
        const json = JSON.parse(errorText);
        if (json.error && json.error.message) {
          errorMessage = json.error.message;
        }
      } catch (e) {
        // Not JSON or expected format, use the raw text
      }
      throw new Error(
        `Error fetching from ${url}: [${response.status} ${response.statusText}] ${errorMessage}`
      );
    }
    return response;
  } catch (e: unknown) {
    console.error(e);
    throw new Error(`Failed to fetch from ${url}: ${extractErrMsg(e)}`);
  }
}
