/**
 * Copyright 2026 Google LLC
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

import { GenkitError } from 'genkit';
import {
  MUSIC_GENERATION_PATH,
  REGION_BASE_URLS,
  SUCCESS_STATUS_CODE,
  type ClientOptions,
  type MusicGenerationRequest,
  type MusicGenerationResponse,
} from './types.js';

/**
 * Builds the absolute music generation URL for the resolved client options.
 *
 * An explicit `baseUrl` wins over the region default so that proxies and
 * gateways can be used without changing the region semantics.
 */
export function getMusicGenerationUrl(clientOptions: ClientOptions): string {
  const baseUrl =
    clientOptions.baseUrl ?? REGION_BASE_URLS[clientOptions.region];
  return `${baseUrl.replace(/\/+$/, '')}${MUSIC_GENERATION_PATH}`;
}

/**
 * Builds the request headers, keeping caller-provided headers from
 * overwriting authentication.
 */
export function getHeaders(
  clientOptions: ClientOptions
): Record<string, string> {
  return {
    ...(clientOptions.customHeaders ?? {}),
    'Content-Type': 'application/json',
    Authorization: `Bearer ${clientOptions.apiKey}`,
  };
}

/**
 * Calls the music generation endpoint and returns the decoded payload.
 *
 * The endpoint answers with HTTP 200 even for some failures, so the
 * `base_resp.status_code` envelope is checked as well.
 */
export async function generateMusic(
  request: MusicGenerationRequest,
  clientOptions: ClientOptions
): Promise<MusicGenerationResponse> {
  const url = getMusicGenerationUrl(clientOptions);

  const response = await fetch(url, {
    method: 'POST',
    headers: getHeaders(clientOptions),
    body: JSON.stringify(request),
    signal: clientOptions.signal,
  });

  if (!response.ok) {
    const details = await response.text().catch(() => '');
    throw new GenkitError({
      status: 'UNKNOWN',
      message: `Error fetching from ${url}: [${response.status} ${response.statusText}] ${details}`,
    });
  }

  let payload: MusicGenerationResponse;
  try {
    payload = (await response.json()) as MusicGenerationResponse;
  } catch (e) {
    throw new GenkitError({
      status: 'UNKNOWN',
      message: `Error parsing the response from ${url}: ${e}`,
    });
  }

  const statusCode = payload.base_resp?.status_code;
  if (statusCode !== undefined && statusCode !== SUCCESS_STATUS_CODE) {
    throw new GenkitError({
      status: 'UNKNOWN',
      message: `Error fetching from ${url}: [${statusCode}] ${
        payload.base_resp?.status_msg ?? 'unknown error'
      }`,
    });
  }

  return payload;
}
