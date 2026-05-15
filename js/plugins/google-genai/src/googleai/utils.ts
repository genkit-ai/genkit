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

import { GenerateRequest, GenkitError, z } from 'genkit';
import process from 'process';
import { extractMedia } from '../common/utils.js';
import { ClientOptions, ImagenInstance, VeoImage, VeoVideo } from './types.js';

export {
  checkModelName,
  cleanSchema,
  extractText,
  extractVersion,
  modelName,
} from '../common/utils.js';

/**
 * Retrieves an API key from environment variables.
 *
 * @returns The API key as a string, or `undefined` if none of the specified
 *          environment variables are set.
 */
export function getApiKeyFromEnvVar(): string | undefined {
  return (
    process.env.GEMINI_API_KEY ||
    process.env.GOOGLE_API_KEY ||
    process.env.GOOGLE_GENAI_API_KEY
  );
}

export const MISSING_API_KEY_ERROR = new GenkitError({
  status: 'FAILED_PRECONDITION',
  message:
    'Please pass in the API key or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n' +
    'For more details see https://genkit.dev/docs/plugins/google-genai/',
});

export const API_KEY_FALSE_ERROR = new GenkitError({
  status: 'INVALID_ARGUMENT',
  message:
    'GoogleAI plugin was initialized with {apiKey: false} but no apiKey configuration was passed at call time.',
});

/**
 * Checks and retrieves an API key based on the provided argument and environment variables.
 *
 * - If `pluginApiKey` is a non-empty string, it's used as the API key.
 * - If `pluginApiKey` is `undefined` or an empty string, it attempts to fetch the API key from environment
 * - If `pluginApiKey` is `false`, key retrieval from the environment is skipped, and the function
 *   will return `undefined`. This mode indicates that the API key is expected to be provided
 *   at a later stage or in a different context.
 *
 * @param pluginApiKey - An optional API key string, `undefined` to check the environment, or `false` to bypass all checks in this function.
 * @returns The resolved API key as a string, or `undefined` if `pluginApiKey` is `false`.
 * @throws {Error} MISSING_API_KEY_ERROR - Thrown if `pluginApiKey` is not `false` and no API key
 *   can be found either in the `pluginApiKey` argument or from the environment.
 */
export function checkApiKey(
  pluginApiKey: string | false | undefined
): string | undefined {
  let apiKey: string | undefined;

  // Don't get the key from the environment if pluginApiKey is false
  if (pluginApiKey !== false) {
    apiKey = pluginApiKey || getApiKeyFromEnvVar();
  }

  // If pluginApiKey is false, then we don't throw because we are waiting for
  // the apiKey passed into the individual call
  if (pluginApiKey !== false && !apiKey) {
    throw MISSING_API_KEY_ERROR;
  }
  return apiKey;
}

/**
 * Calculates and returns the effective API key based on multiple potential sources.
 * The order of precedence for determining the API key is:
 * 1. `requestApiKey` (if provided)
 * 2. `pluginApiKey` (if provided and not `false`)
 * 3. Environment variable (if `pluginApiKey` is not `false` and `pluginApiKey` is not provided)
 *
 * @param pluginApiKey - The apiKey value provided during plugin initialization.
 * @param requestApiKey - The apiKey provided to an individual generate call.
 * @returns The resolved API key as a string.
 * @throws {Error} API_KEY_FALSE_ERROR - Thrown if `pluginApiKey` is `false` and `requestApiKey` is not provided
 * @throws {Error} MISSING_API_KEY_ERROR - Thrown if no API key can be resolved from any source
 */
export function calculateApiKey(
  pluginApiKey: string | false | undefined,
  requestApiKey: string | undefined
): string | undefined {
  let apiKey: string | undefined;

  // Don't get the key from the environment if pluginApiKey is false
  if (pluginApiKey !== false) {
    apiKey = pluginApiKey || getApiKeyFromEnvVar();
  }

  apiKey = requestApiKey || apiKey;

  if (pluginApiKey === false && !requestApiKey) {
    return undefined;
  }

  if (!apiKey) {
    throw MISSING_API_KEY_ERROR;
  }
  return apiKey;
}

export function extractVeoImage(
  request: GenerateRequest
): VeoImage | undefined {
  const media = request.messages.at(-1)?.content.find((p) => !!p.media)?.media;
  if (!media?.contentType?.startsWith('image/')) {
    return undefined;
  }
  const bytes = media?.url.split(',')[1];
  if (bytes) {
    return {
      bytesBase64Encoded: bytes,
      mimeType: media.contentType,
    };
  }
  return undefined;
}

export function extractVeoVideo(
  request: GenerateRequest
): VeoVideo | undefined {
  const media = request.messages.at(-1)?.content.find((p) => !!p.media)?.media;
  if (!media?.contentType?.startsWith('video/')) {
    return undefined;
  }
  return {
    uri: media.url,
  };
}

export function extractImagenImage(
  request: GenerateRequest
): ImagenInstance['image'] | undefined {
  const image = extractMedia(request, {
    metadataType: 'base',
    isDefault: true,
  })?.url.split(',')[1];

  if (image) {
    return { bytesBase64Encoded: image };
  }
  return undefined;
}

/**
 * For each field in ClientOptions, if the request config object has
 * a matching non-empty/non-null field, it overrides the original.
 */
export function calculateRequestOptions<T extends z.ZodObject<any, any, any>>(
  clientOptions: ClientOptions,
  reqConfig?: z.infer<T>
): ClientOptions {
  if (!reqConfig) {
    return clientOptions;
  }

  let newOptions = { ...clientOptions };

  if (typeof reqConfig.timeout == 'number') {
    newOptions.timeout = reqConfig.timeout;
  }

  if (typeof reqConfig.apiKey == 'string') {
    newOptions.apiKey = reqConfig.apiKey;
  }

  if (typeof reqConfig.apiVersion == 'string') {
    newOptions.apiVersion = reqConfig.apiVersion;
  }

  if (typeof reqConfig.apiClient == 'string') {
    newOptions.apiClient = reqConfig.apiClient;
  }

  if (typeof reqConfig.baseUrl == 'string') {
    newOptions.baseUrl = reqConfig.baseUrl;
  }

  if (reqConfig.customHeaders && typeof reqConfig.customHeaders === 'object') {
    newOptions.customHeaders = reqConfig.customHeaders;
  }

  return newOptions;
}

/**
 * The config can have client option overrides, but they should not
 * be sent with the request like normal config.
 * @param requestConfig
 */
export function removeClientOptionOverrides<
  T extends z.ZodObject<any, any, any>,
>(requestConfig?: z.infer<T>): z.infer<T> {
  let newConfig = { ...requestConfig };

  delete newConfig?.timeout;
  delete newConfig?.apiKey;
  delete newConfig?.apiVersion;
  delete newConfig?.apiClient;
  delete newConfig?.baseUrl;
  delete newConfig?.customHeaders;

  return newConfig;
}
