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

/**
 * Programmatic port value meaning "let the OS pick". Code cannot use 0 for
 * this (it is Go's zero value and falsy in Python, so it reads as "unset"
 * there); the environment spells the same thing `GENKIT_REFLECTION_PORT=0`.
 */
export const REFLECTION_PORT_AUTO = -1;

/**
 * Either an exact port (0 lets the OS pick) or a probe upward from
 * {@link DEFAULT_REFLECTION_PORT}, used only when nobody chose a port.
 */
export type ReflectionPort =
  | { kind: 'pinned'; port: number }
  | { kind: 'probe' };

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
  // Plain decimal digits only: Number() alone would accept "0x10", "1e3",
  // "+7" and surrounding whitespace, silently binding a different port.
  const port = /^[0-9]+$/.test(raw) ? Number(raw) : NaN;
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    throw new Error(
      `GENKIT_REFLECTION_PORT must be an integer between 0 and 65535, got "${raw}".`
    );
  }
  return port;
}

/**
 * Resolves the port for the v1 server. Whoever chose a port, the environment
 * or the code, gets exactly that port; only an unchosen port is probed.
 *
 * Code values: `undefined` or `0` is unset, {@link REFLECTION_PORT_AUTO} (-1)
 * lets the OS pick, 1..65535 is exact. Anything else throws.
 */
export function resolveReflectionPort(
  envPort: number | undefined,
  optionPort: number | undefined
): ReflectionPort {
  if (envPort !== undefined) {
    return { kind: 'pinned', port: envPort };
  }
  if (optionPort === undefined || optionPort === 0) {
    return { kind: 'probe' };
  }
  if (optionPort === REFLECTION_PORT_AUTO) {
    return { kind: 'pinned', port: 0 };
  }
  if (!Number.isInteger(optionPort) || optionPort < 1 || optionPort > 65535) {
    throw new Error(
      `reflectionPort must be -1 (OS-assigned) or an integer between 1 and 65535, got ${optionPort}.`
    );
  }
  return { kind: 'pinned', port: optionPort };
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
 * library call they do not control. `options.port` does not turn the server
 * on by itself, and is validated even when unused so a bad value fails early.
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
  const port = resolveReflectionPort(envPort, options.port);
  const host = env.GENKIT_REFLECTION_HOST;
  if (envPort === undefined && !host && env.GENKIT_ENV !== 'dev') {
    return { kind: 'off' };
  }
  return {
    kind: 'v1',
    host: host || DEFAULT_REFLECTION_HOST,
    port,
    secret,
  };
}

/**
 * Host to advertise in the runtime discovery file for a server bound to
 * `host`. A wildcard bind is reachable on loopback, and `0.0.0.0` is not a
 * valid destination everywhere, so it is advertised as `127.0.0.1`. IPv6
 * literals are bracketed for use in a URL.
 */
export function advertisedReflectionHost(host: string): string {
  if (host === '0.0.0.0' || host === '::' || host === '[::]') {
    return '127.0.0.1';
  }
  return host.includes(':') && !host.startsWith('[') ? `[${host}]` : host;
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
