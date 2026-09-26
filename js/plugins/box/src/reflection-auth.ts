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

// Reflection auth wire contract. These mirror `@genkit-ai/core`'s
// reflection-config, which `genkit` does not re-export; box depends on
// `genkit` only, and the values are protocol constants, not implementation.

import { createHash, timingSafeEqual } from 'node:crypto';

/** Env var a runtime reads its reflection secret from. */
export const REFLECTION_SECRET_ENV = 'GENKIT_REFLECTION_SECRET_TOKEN';

/** Header carrying the reflection secret on v1 requests. */
export const REFLECTION_SECRET_HEADER = 'x-genkit-reflection-secret';

/** JSON-RPC error code for a failed v2 `register`; tells runtimes not to retry. */
export const REFLECTION_AUTH_ERROR_CODE = -32001;

/** Constant-time comparison over sha256 digests (hides the expected length). */
export function secretsEqual(a: string, b: string): boolean {
  const ha = createHash('sha256').update(a).digest();
  const hb = createHash('sha256').update(b).digest();
  return timingSafeEqual(ha, hb);
}
