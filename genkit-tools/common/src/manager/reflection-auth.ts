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

import { createHash, randomBytes, timingSafeEqual } from 'crypto';

/** Header carrying the reflection secret on v1 requests. */
export const REFLECTION_SECRET_HEADER = 'x-genkit-reflection-secret';

/** Environment variable holding the reflection secret, for CLI and runtimes. */
export const REFLECTION_SECRET_ENV = 'GENKIT_REFLECTION_SECRET_TOKEN';

/**
 * JSON-RPC error code the v2 server returns when `register` fails auth.
 * Runtimes treat it as terminal and stop reconnecting.
 */
export const REFLECTION_AUTH_ERROR_CODE = -32001;

/**
 * Interface the v2 WebSocket server binds. Loopback only: a registered socket
 * can be sent runAction, so it must not be reachable from other hosts.
 */
export const REFLECTION_V2_HOST = '127.0.0.1';

/** Generates a fresh, per-run reflection secret. */
export function generateReflectionSecret(): string {
  return randomBytes(32).toString('base64url');
}

/**
 * Constant-time secret comparison. Hashing first gives equal-length inputs,
 * which timingSafeEqual requires, without leaking the expected length.
 */
export function secretsEqual(a: string, b: string): boolean {
  const ha = createHash('sha256').update(a).digest();
  const hb = createHash('sha256').update(b).digest();
  return timingSafeEqual(ha, hb);
}
