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
import { GenkitError, z, type Document } from 'genkit';
import { VoyageAIClient, VoyageAIError } from 'voyageai';
import type { VoyageAIPluginOptions } from './types.js';

/** Environment variable the plugin reads the API key from. */
export const API_KEY_ENV_VAR = 'VOYAGE_API_KEY';

/**
 * Creates the Voyage client. The key comes from the plugin options or from
 * `VOYAGE_API_KEY`; without one the plugin refuses to start.
 */
export function createClient(options: VoyageAIPluginOptions): VoyageAIClient {
  const apiKey = options.apiKey || process.env[API_KEY_ENV_VAR];
  if (!apiKey) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: `Voyage AI API key is missing. Pass \`apiKey\` to the voyageai plugin or set the ${API_KEY_ENV_VAR} environment variable.`,
    });
  }
  return new VoyageAIClient({
    apiKey,
    baseUrl: options.baseUrl,
    maxRetries: options.maxRetries,
    timeoutInSeconds: options.timeoutInSeconds,
    fetch: options.fetch,
  });
}

/**
 * Extracts the message from a Voyage error body, which has the shape
 * `{"detail": "<text>"}`. Returns undefined for any other body.
 */
export function extractVoyageErrorDetail(body: unknown): string | undefined {
  if (typeof body !== 'object' || body === null) return undefined;
  const detail = (body as { detail?: unknown }).detail;
  return typeof detail === 'string' && detail.length > 0 ? detail : undefined;
}

function statusName(statusCode: number | undefined): GenkitError['status'] {
  switch (statusCode) {
    case 400:
    case 422:
      return 'INVALID_ARGUMENT';
    case 401:
      return 'UNAUTHENTICATED';
    case 403:
      return 'PERMISSION_DENIED';
    case 404:
      return 'NOT_FOUND';
    case 429:
      return 'RESOURCE_EXHAUSTED';
    case 502:
    case 503:
      return 'UNAVAILABLE';
    case 504:
      return 'DEADLINE_EXCEEDED';
    default:
      return 'INTERNAL';
  }
}

/** Runs a Voyage SDK call and turns SDK failures into `GenkitError`s carrying Voyage's `detail` text. */
export async function callVoyage<T>(
  operation: string,
  call: () => PromiseLike<T>
): Promise<T> {
  try {
    return await call();
  } catch (e) {
    if (e instanceof VoyageAIError) {
      const detail = extractVoyageErrorDetail(e.body) ?? e.message;
      throw new GenkitError({
        status: statusName(e.statusCode),
        message: `Voyage AI ${operation} failed${e.statusCode ? ` (HTTP ${e.statusCode})` : ''}: ${detail}`,
        cause: e,
      });
    }
    throw e;
  }
}

/** Shorthand for an `INVALID_ARGUMENT` error. */
export function invalidArgument(message: string): GenkitError {
  return new GenkitError({ status: 'INVALID_ARGUMENT', message });
}

/**
 * Validates embedder/reranker options against a strict schema. Genkit does not
 * validate options for plugin actions, so unknown or mistyped fields would
 * otherwise be dropped silently.
 */
export function parseConfig<S extends z.ZodTypeAny>(
  schema: S,
  options: unknown,
  actionName: string
): z.infer<S> {
  // Genkit adds keys such as `version: undefined` to the options it passes on.
  const defined = Object.fromEntries(
    Object.entries((options as Record<string, unknown>) || {}).filter(
      ([, v]) => v !== undefined
    )
  );
  const parsed = schema.safeParse(defined);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .map((i) => `${i.path.join('.') || 'options'}: ${i.message}`)
      .join('; ');
    throw invalidArgument(`Invalid options for ${actionName}: ${issues}`);
  }
  return parsed.data;
}

/**
 * Validates a requested output dimension against the model's supported ones
 * and returns it, or undefined when none was requested.
 */
export function checkOutputDimension(
  model: string,
  supported: readonly number[],
  requested: number | undefined
): number | undefined {
  if (requested === undefined) return undefined;
  if (!supported.includes(requested)) {
    throw invalidArgument(
      `${model} does not support outputDimension ${requested}; supported: ${supported.join(', ')}.`
    );
  }
  return requested;
}

/** Returns the text of a document that must contain text only. */
export function textOnly(doc: Document, actionName: string, index: number) {
  if (doc.media.length > 0) {
    throw invalidArgument(
      `${actionName} embeds text only, but document ${index} contains media. Use a voyage-multimodal-* embedder for media.`
    );
  }
  return doc.text;
}
