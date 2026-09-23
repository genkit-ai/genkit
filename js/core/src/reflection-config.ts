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

import { createHash, timingSafeEqual } from 'crypto';

/** Header carrying the reflection secret on v1 requests. */
export const REFLECTION_SECRET_HEADER = 'x-genkit-reflection-secret';

/** Default interface for the v1 server. */
export const DEFAULT_REFLECTION_HOST = '127.0.0.1';

/** First port tried when no exact port is configured. */
export const DEFAULT_REFLECTION_PORT = 3100;

/**
 * JSON-RPC error code the CLI returns when a v2 `register` fails auth.
 * Terminal: the runtime must not reconnect, the secret will not change.
 */
export const REFLECTION_AUTH_ERROR_CODE = -32001;

/** Port either pinned by the environment or used as the start of a probe. */
export type ReflectionPort =
  | { kind: 'pinned'; port: number }
  | { kind: 'probeFrom'; port: number };

/**
 * How the reflection API runs, if at all.
 *
 * `disabled` and `off` differ by who decided. `disabled` is an explicit kill
 * switch that even a direct `ReflectionServer.start()` honours. `off` only
 * means nothing in the environment asked for a server, so an explicit start
 * still runs one with defaults.
 */
export type ReflectionConfig =
  | { kind: 'disabled' }
  | { kind: 'off' }
  | { kind: 'v2'; url: string; secret?: string }
  | {
      kind: 'v1';
      host: string;
      port: ReflectionPort;
      secret?: string;
    };

/** Minimal view of the environment, so this stays testable and platform-free. */
export type ReflectionEnv = Record<string, string | undefined>;

/**
 * Parses `GENKIT_REFLECTION_PORT`.
 *
 * Anything that is not an integer in 0..65535 throws rather than falling back
 * to probing: a typo in a deployment config should fail loudly, not quietly
 * bind a port nobody published.
 */
function parsePort(raw: string | undefined): number | undefined {
  if (raw === undefined || raw === '') {
    return undefined;
  }
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    throw new Error(
      `GENKIT_REFLECTION_PORT must be an integer between 0 and 65535, got "${raw}".`
    );
  }
  return port;
}

/**
 * Resolves how the reflection API should run.
 *
 * First match wins:
 * 1. `GENKIT_REFLECTION_DISABLED === 'true'` turns everything off.
 * 2. `GENKIT_REFLECTION_V2_SERVER` dials out instead of listening.
 * 3. `GENKIT_REFLECTION_PORT` or `GENKIT_REFLECTION_HOST` starts the v1 server.
 * 4. `GENKIT_ENV === 'dev'` starts the v1 server with defaults.
 * 5. Otherwise off.
 *
 * Setting host or port is itself the on-switch, so there is no way to configure
 * the server and then wonder why it did not start. The environment beats
 * `options.port` on purpose: whoever set the variable is typically the
 * supervisor that already published that port and cannot be overruled by a
 * library call they do not control.
 */
export function resolveReflectionConfig(
  env: ReflectionEnv,
  options: { port?: number } = {}
): ReflectionConfig {
  if (env.GENKIT_REFLECTION_DISABLED === 'true') {
    return { kind: 'disabled' };
  }
  const secret = env.GENKIT_REFLECTION_SECRET_TOKEN || undefined;
  if (env.GENKIT_REFLECTION_V2_SERVER) {
    return { kind: 'v2', url: env.GENKIT_REFLECTION_V2_SERVER, secret };
  }
  const envPort = parsePort(env.GENKIT_REFLECTION_PORT);
  const host = env.GENKIT_REFLECTION_HOST;
  if (envPort === undefined && !host && env.GENKIT_ENV !== 'dev') {
    return { kind: 'off' };
  }
  return {
    kind: 'v1',
    host: host || DEFAULT_REFLECTION_HOST,
    port:
      envPort !== undefined
        ? { kind: 'pinned', port: envPort }
        : {
            kind: 'probeFrom',
            port: options.port ?? DEFAULT_REFLECTION_PORT,
          },
    secret,
  };
}

/** Whether a host is loopback, and so unreachable from other machines. */
export function isLoopbackHost(host: string): boolean {
  return (
    host === 'localhost' ||
    host === '::1' ||
    host === '[::1]' ||
    /^127\./.test(host)
  );
}

/**
 * Constant-time secret comparison over sha256 digests, which gives equal-length
 * inputs without leaking the expected length.
 */
export function secretsEqual(a: string, b: string): boolean {
  const ha = createHash('sha256').update(a).digest();
  const hb = createHash('sha256').update(b).digest();
  return timingSafeEqual(ha, hb);
}
