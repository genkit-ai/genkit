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

import {
  GenerateRequest,
  GenkitError,
  StatusName,
  type ErrorResponseMetadata,
} from 'genkit';
import { EmbedRequest } from 'genkit/embedder';
import OpenAI, { APIError } from 'openai';
import { PluginOptions } from '.';

/**
 * Inspects the request and if apiKey is provided in config, creates a new client.
 * Otherwise falls back on the `defaultClient`.
 */
export function maybeCreateRequestScopedOpenAIClient(
  pluginOptions: PluginOptions | undefined,
  request: GenerateRequest | EmbedRequest,
  defaultClient: OpenAI
): OpenAI {
  const requestApiKey =
    (request as GenerateRequest)?.config?.apiKey ??
    (request as EmbedRequest)?.options?.apiKey;
  if (!requestApiKey) {
    return defaultClient;
  }
  return new OpenAI({
    // if pluginOptions are not passed in we attempt to get options from the default client.
    ...(pluginOptions ?? defaultClient['_options']),
    apiKey: requestApiKey,
  });
}

/**
 * Parses a `Retry-After` header value into milliseconds.
 * Supports delay-seconds and HTTP-date formats (RFC 7231 §7.1.3).
 */
function parseRetryAfterMs(value: string): number | undefined {
  if (!value || !value.trim()) return undefined;
  const seconds = Number(value);
  if (!isNaN(seconds) && seconds >= 0) return seconds * 1000;
  const date = new Date(value);
  if (!isNaN(date.getTime())) return Math.max(0, date.getTime() - Date.now());
  return undefined;
}

/**
 * Rethrows an error raised by the OpenAI SDK. An `APIError` is converted into a
 * {@link GenkitError} carrying the equivalent Genkit status and, when the
 * response supplied one, a `retryAfterMs` hint. Anything else is rethrown
 * untouched.
 */
export function rethrowOpenAIError(e: unknown): never {
  if (e instanceof APIError) {
    let status: StatusName = 'UNKNOWN';
    switch (e.status) {
      case 429:
        status = 'RESOURCE_EXHAUSTED';
        break;
      case 401:
        status = 'PERMISSION_DENIED';
        break;
      case 403:
        status = 'UNAUTHENTICATED';
        break;
      case 400:
        status = 'INVALID_ARGUMENT';
        break;
      case 500:
        status = 'INTERNAL';
        break;
      case 503:
        status = 'UNAVAILABLE';
        break;
    }
    const retryAfterHeader =
      e.headers?.get?.('retry-after') ?? (e.headers as any)?.['retry-after'];
    const retryAfterMs = retryAfterHeader
      ? parseRetryAfterMs(retryAfterHeader)
      : undefined;
    const responseMetadata: ErrorResponseMetadata | undefined =
      retryAfterMs !== undefined ? { retryAfterMs } : undefined;
    throw new GenkitError({
      status,
      message: e.message,
      responseMetadata,
    });
  }
  throw e;
}

/**
 * Checks if a content type is an image type.
 * @param contentType The content type to check.
 * @returns True if the content type is an image type.
 */
export function isImageContentType(contentType?: string): boolean {
  if (!contentType) return false;
  return contentType.startsWith('image/');
}

/**
 * Extracts the base64 data and content type from a data URL.
 * @param url The data URL to parse.
 * @returns The base64 data and content type, or null if invalid.
 */
export function extractDataFromBase64Url(url: string): {
  data: string;
  contentType: string;
} | null {
  const match = url.match(/^data:([^;]+);base64,(.+)$/);
  return (
    match && {
      contentType: match[1],
      data: match[2],
    }
  );
}

/**
 * Map of content types to file extensions.
 */
const FILE_EXTENSIONS: Record<string, string> = {
  'application/pdf': 'pdf',
  'application/msword': 'doc',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
    'docx',
  'text/plain': 'txt',
  'text/csv': 'csv',
};

/**
 * Generates a filename from a content type.
 * @param contentType The content type.
 * @returns A filename with appropriate extension.
 */
export function generateFilenameFromContentType(contentType: string): string {
  const ext = FILE_EXTENSIONS[contentType] || '';
  return ext ? `file.${ext}` : 'file';
}

/**
 * Gets the model name without certain prefixes.
 */
export function toModelName(name: string, prefix?: string): string {
  const pattern = '^/(background-model|model|models|embedder|embedders)/';
  const refPrefixes = new RegExp(pattern);
  const maybePluginRef = name.replace(refPrefixes, '');
  if (prefix) {
    const pluginPrefix = new RegExp(`^${prefix}/`, 'g');
    return maybePluginRef.replace(pluginPrefix, '');
  } else {
    return maybePluginRef;
  }
}
