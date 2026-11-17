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

import { logger } from '../logging.js';

export let telemetryServerUrl: string | undefined;

/**
 * @hidden
 */
export function setTelemetryServerUrl(url: string) {
  telemetryServerUrl = url;
}

// Global-keyed (like the instrumentation registry) so flushTracing sees posts
// started by a duplicated copy of this module.
const pendingPostsKey = '__GENKIT_PENDING_TELEMETRY_POSTS';

function pendingPosts(): Set<Promise<void>> {
  if (!global[pendingPostsKey]) {
    global[pendingPostsKey] = new Set<Promise<void>>();
  }
  return global[pendingPostsKey];
}

/**
 * POST to a telemetry server endpoint (the Developer UI's trace/log sink).
 * Used by DirectTelemetryInstrumentation.
 *
 * Callers may fire-and-forget; in-flight posts are tracked so
 * {@link flushTelemetryServerPosts} (via `flushTracing`) can await them.
 *
 * @hidden
 */
export function postToTelemetryServer(
  endpoint: string,
  payload: unknown
): Promise<void> {
  const post = doPost(endpoint, payload);
  const pending = pendingPosts();
  pending.add(post);
  // Errors are the caller's to handle; this branch only untracks.
  post.then(
    () => pending.delete(post),
    () => pending.delete(post)
  );
  return post;
}

/**
 * Resolves once every in-flight telemetry server post has settled. Never
 * rejects.
 *
 * @hidden
 */
export async function flushTelemetryServerPosts(): Promise<void> {
  await Promise.allSettled([...pendingPosts()]);
}

async function doPost(endpoint: string, payload: unknown): Promise<void> {
  // In dev the CLI passes the server URL via env; the handshake only calls
  // setTelemetryServerUrl when the env var is absent, so fall back to it here.
  const url = telemetryServerUrl ?? process.env.GENKIT_TELEMETRY_SERVER;
  if (!url) {
    return;
  }
  // Core has no OpenTelemetry dependency, so this fetch is not wrapped in
  // `suppressTracing`. Apps using OTel HTTP auto-instrumentation will see a
  // client span per post. Dev only: the telemetry server is never set in prod.
  const response = await fetch(`${url}${endpoint}`, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  // fetch resolves on 4xx/5xx; surface it so a misconfigured server isn't silent.
  if (!response.ok) {
    logger.debug(
      `Telemetry server POST ${endpoint} failed: ${response.status}`
    );
  }
}
