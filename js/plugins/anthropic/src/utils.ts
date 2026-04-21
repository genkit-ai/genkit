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

import type { CacheControlEphemeral } from '@anthropic-ai/sdk/resources/messages';

/**
 * Creates a cache control metadata object for prompt caching.
 * Returns `{ cache_control: ... }` so it can be spread into metadata.
 *
 * @param options - Cache control options. Type defaults to 'ephemeral'.
 * @returns Object with cache_control property to spread into part metadata.
 *
 * @example
 * ```ts
 * import { anthropic, cacheControl } from '@genkit-ai/anthropic';
 *
 * const response = await ai.generate({
 *   model: anthropic.model('claude-sonnet-4-5'),
 *   system: {
 *     text: longSystemPrompt,
 *     metadata: { ...cacheControl() }  // default ephemeral
 *   },
 *   messages: [{ role: 'user', content: [{ text: 'Hello' }] }]
 * });
 *
 * // Or with explicit TTL:
 * metadata: { ...cacheControl({ ttl: '1h' }) }
 * ```
 */
export function cacheControl(options?: Partial<CacheControlEphemeral>): {
  cache_control: CacheControlEphemeral;
} {
  return {
    cache_control: {
      type: options?.type ?? 'ephemeral',
      ...(options?.ttl && { ttl: options.ttl }),
    },
  };
}

/**
 * Strips the 'anthropic/' namespace prefix if present.
 * Throws if the model name is missing.
 */
export function checkModelName(name?: string): string {
  const cleanName = name?.replace(/^anthropic\//, '');
  if (!cleanName) {
    throw new Error('Model name is required.');
  }
  return cleanName;
}

export function removeUndefinedProperties<T>(obj: T): T {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  return Object.fromEntries(
    Object.entries(obj).filter(([_, value]) => value !== undefined)
  ) as T;
}

export function isKnownKey<T extends object>(
  key: string | number | symbol,
  obj: T
): key is keyof T {
  return key in obj;
}
