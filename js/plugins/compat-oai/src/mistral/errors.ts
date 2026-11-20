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

import type { ClientOptions } from 'openai';
import type { ErrorStatusClassifier } from '../model.js';

type Fetch = NonNullable<ClientOptions['fetch']>;

/**
 * Whether `body` is an error body the OpenAI SDK can't read: a JSON object
 * with no `error` key. Mistral sends top-level errors such as
 * `{"object": "error", "message": "...", "type": "invalid_model", ...}`, and
 * its OpenAPI spec documents HTTP 422 as `{"detail": [...]}`
 * (`HTTPValidationError`), neither wrapped in an `error` key as OpenAI's are.
 */
function isUnwrappedError(body: unknown): body is Record<string, unknown> {
  return (
    typeof body === 'object' &&
    body !== null &&
    !Array.isArray(body) &&
    !('error' in body)
  );
}

/**
 * Wraps `fetch` so that Mistral's error bodies reach the OpenAI SDK. The SDK
 * only reads errors under an `error` key, so for any other JSON object body
 * it reports, for example, "400 status code (no body)" and drops the message
 * and `type`. This moves the body under `error`, so the API error has
 * Mistral's message, `type` and `code`, or the `detail` of a validation
 * error.
 *
 * Returns `undefined`, keeping the SDK's own fetch, when `httpAgent` is set
 * without a `fetch`: Node's built-in fetch would ignore the agent.
 */
export function mistralErrorBodyFetch(
  options: Pick<ClientOptions, 'fetch' | 'httpAgent'> | undefined
): Fetch | undefined {
  if (!options?.fetch && options?.httpAgent) return undefined;
  // `globalThis.fetch` is looked up on each call, not when the plugin is
  // created, so a fetch patched in later (tracing, test stubs) is used.
  const inner: typeof fetch = (input, init) =>
    ((options?.fetch as unknown as typeof fetch) ?? globalThis.fetch)(
      input,
      init
    );
  const wrapped: typeof fetch = async (input, init) => {
    const response = await inner(input, init);
    if (response.ok) return response;
    let body: unknown;
    try {
      body = await response.clone().json();
    } catch {
      return response;
    }
    if (!isUnwrappedError(body)) return response;
    const headers = new Headers(response.headers);
    // The body is re-encoded, so the original length and encoding no longer
    // apply.
    headers.delete('content-length');
    headers.delete('content-encoding');
    return new Response(JSON.stringify({ error: body }), {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  };
  return wrapped as unknown as Fetch;
}

/**
 * Mistral answers an unknown or retired model with HTTP 400 and
 * `"type": "invalid_model"`. Report that as NOT_FOUND; other errors keep the
 * status from the HTTP status code.
 */
export const mistralErrorStatus: ErrorStatusClassifier = (e) =>
  e.type === 'invalid_model' ? 'NOT_FOUND' : undefined;
