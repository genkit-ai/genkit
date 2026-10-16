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

import { GenerateRequest, GenkitError, z, type ActionContext } from 'genkit';
import process from 'process';
import { ClientOptions, VeoImage, VeoVideo } from './types.js';

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

/**
 * Applies connection-related overrides carried in the action run context
 * (e.g. `options.context` on `start`/`check`/`cancel` for background models)
 * on top of the given client options. This is the mechanism by which callers
 * can supply per-call overrides for calls (like
 * `checkOperation`/`cancelOperation`) that don't have access to the original
 * request's `config`.
 *
 * The API key is read from `context.secrets.apiKey` (scrubbed from traces).
 * All other overrides (e.g. `baseUrl`, `apiVersion`) are read from
 * `context.config`, which is where `checkOperation`/`cancelOperation` fold
 * their top-level `config` option. This lets callers write
 * `ai.checkOperation(op, { config: { baseUrl } })`.
 *
 * All context-derived values take precedence over whatever is already in
 * `clientOptions`.
 *
 * @param clientOptions The base client options to apply overrides on top of.
 * @param context The action run context (`options.context`), if any.
 */
export function applyContextOverrides(
  clientOptions: ClientOptions,
  context?: ActionContext
): ClientOptions {
  if (!context) {
    return clientOptions;
  }

  let newOptions = { ...clientOptions };

  if (typeof context.secrets?.apiKey === 'string') {
    newOptions.apiKey = context.secrets.apiKey;
  }

  // All non-secret overrides (e.g. baseUrl, apiVersion) are carried in
  // `context.config` (from checkOperation/cancelOperation's top-level
  // `config`).
  return calculateRequestOptions(newOptions, context.config);
}

/**
 * Converts a string in camelCase to lowercase snake_case.
 */
export function camelToSnakeCase(str: string): string {
  return str.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

export function convertObjectKeysToSnakeCase(obj: Array<unknown>): unknown[];
export function convertObjectKeysToSnakeCase(
  obj: Record<string, unknown>
): Record<string, unknown>;
export function convertObjectKeysToSnakeCase(obj: unknown): unknown;

export function convertObjectKeysToSnakeCase(obj: unknown): unknown {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => convertObjectKeysToSnakeCase(item));
  }

  const newObj: Record<string, unknown> = {};
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const snakeCaseKey = camelToSnakeCase(key);
      newObj[snakeCaseKey] = convertObjectKeysToSnakeCase(obj[key]);
    }
  }
  return newObj;
}

export function isObject(val: unknown): val is Record<string, unknown> {
  return typeof val === 'object' && val !== null && !Array.isArray(val);
}

/** Reports whether every element of an array is a string. */
export function isStringArray(values: unknown[]): values is string[] {
  return values.every((v) => typeof v === 'string');
}

/**
 * Inserts raw audio bytes at the end of a WAV file's `data` chunk and updates
 * the RIFF and `data` sizes. Chunks after `data` (the server sends one) stay
 * after the audio. Returns `undefined` if `wav` isn't a WAV file with a
 * `data` chunk.
 */
export function appendToWavData(
  wav: Buffer,
  audio: Buffer
): Buffer | undefined {
  if (
    wav.length < 12 ||
    wav.toString('ascii', 0, 4) !== 'RIFF' ||
    wav.toString('ascii', 8, 12) !== 'WAVE'
  ) {
    return undefined;
  }
  // Walk the chunks: each is a 4-byte id, a 4-byte little-endian size, then
  // the data, padded to an even length.
  let offset = 12;
  while (offset + 8 <= wav.length) {
    const id = wav.toString('ascii', offset, offset + 4);
    const size = wav.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    if (id === 'data') {
      const dataEnd = Math.min(dataStart + size, wav.length);
      const result = Buffer.concat([
        wav.subarray(0, dataEnd),
        audio,
        wav.subarray(dataEnd),
      ]);
      result.writeUInt32LE(size + audio.length, offset + 4);
      result.writeUInt32LE(result.length - 8, 4);
      return result;
    }
    offset = dataStart + size + (size % 2);
  }
  return undefined;
}

/**
 * Returns `bytes` without any ID3v2 tags at the start. A tag is only removed
 * if its header is valid ("ID3", version 2-4, size bytes below 0x80) and the
 * whole tag fits in `bytes`; otherwise `bytes` is returned unchanged.
 */
export function stripLeadingId3Tags(bytes: Buffer): Buffer {
  let start = 0;
  while (start + 10 <= bytes.length) {
    const b = bytes.subarray(start);
    const isTag =
      b[0] === 0x49 && // I
      b[1] === 0x44 && // D
      b[2] === 0x33 && // 3
      b[3] >= 2 &&
      b[3] <= 4 &&
      b[4] !== 0xff &&
      b[6] < 0x80 &&
      b[7] < 0x80 &&
      b[8] < 0x80 &&
      b[9] < 0x80;
    if (!isTag) break;
    // The size is "syncsafe" (7 bits per byte) and excludes the 10-byte
    // header and the optional 10-byte footer (flag 0x10).
    const size = (b[6] << 21) | (b[7] << 14) | (b[8] << 7) | b[9];
    const tagLength = 10 + size + (b[5] & 0x10 ? 10 : 0);
    if (tagLength > b.length) break;
    start += tagLength;
  }
  return start === 0 ? bytes : bytes.subarray(start);
}

export function toSnakeCaseObj(val: unknown): Record<string, unknown> {
  const snakeVal = convertObjectKeysToSnakeCase(val);
  return isObject(snakeVal) ? snakeVal : {};
}
