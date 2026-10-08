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
 * switch (`GENKIT_REFLECTION_ENABLED=false`) that even a direct
 * `ReflectionServer.start()` honours. `off` only means nothing in the
 * environment asked for a server, so an explicit start still runs one with
 * defaults.
 */
export type ReflectionConfig =
  | { kind: 'disabled' }
  | { kind: 'off' }
  | ReflectionServerConfig;

/** How a running reflection API connects: dial out (v2) or listen (v1). */
export type ReflectionServerConfig =
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
 * Parses `GENKIT_REFLECTION_ENABLED`: `true`, `false`, or unset (empty counts
 * as unset). Anything else throws, for the same reason as {@link parsePort}.
 */
function parseEnabled(raw: string | undefined): boolean | undefined {
  if (raw === undefined || raw === '') {
    return undefined;
  }
  if (raw === 'true' || raw === 'false') {
    return raw === 'true';
  }
  throw new Error(
    `GENKIT_REFLECTION_ENABLED must be "true" or "false", got "${raw}".`
  );
}

/**
 * Resolves the port for the v1 server. Whoever chose a port, the environment
 * or the code, gets exactly that port; only an unchosen port is probed.
 *
 * Code values: `undefined` is unset, `0` lets the OS pick (same as
 * `GENKIT_REFLECTION_PORT=0` and `net.Server.listen`), 1..65535 is exact.
 * Anything else throws, even when the environment port wins, so a bad value
 * fails regardless of deployment.
 */
export function resolveReflectionPort(
  envPort: number | undefined,
  optionPort: number | undefined
): ReflectionPort {
  const fromOption = resolveOptionPort(optionPort);
  return envPort !== undefined ? { kind: 'pinned', port: envPort } : fromOption;
}

function resolveOptionPort(optionPort: number | undefined): ReflectionPort {
  if (optionPort === undefined) {
    return { kind: 'probe' };
  }
  if (!Number.isInteger(optionPort) || optionPort < 0 || optionPort > 65535) {
    throw new Error(
      `reflectionPort must be an integer between 0 and 65535, got ${optionPort}.`
    );
  }
  return { kind: 'pinned', port: optionPort };
}

/**
 * Resolves how the reflection API should run.
 *
 * Whether it runs:
 * - `GENKIT_REFLECTION_ENABLED=false` turns it off, even under dev.
 * - `GENKIT_REFLECTION_ENABLED=true` turns it on in any environment.
 * - Unset, it runs only under `GENKIT_ENV=dev`, as it always has.
 *
 * How it runs, once on: `GENKIT_REFLECTION_V2_SERVER` dials out; otherwise the
 * v1 server listens on `GENKIT_REFLECTION_HOST`/`GENKIT_REFLECTION_PORT`.
 * Those are settings, not on-switches: a stray value in a production env does
 * not expose the API, and is not even parsed while reflection is off.
 *
 * The environment port beats `options.port` on purpose: whoever set the
 * variable is typically the supervisor that already published that port.
 * `options.port` is validated even when unused so a bad value fails early.
 *
 * @hidden
 */
export function resolveReflectionConfig(
  env: ReflectionEnv,
  options: { port?: number } = {}
): ReflectionConfig {
  // Validated before the on/off check so a bad value fails early.
  resolveOptionPort(options.port);
  const enabled = parseEnabled(env.GENKIT_REFLECTION_ENABLED);
  if (enabled === false) {
    return { kind: 'disabled' };
  }
  if (enabled === undefined && env.GENKIT_ENV !== 'dev') {
    return { kind: 'off' };
  }
  return resolveReflectionServerConfig(env, options);
}

/**
 * The "how it runs" half of {@link resolveReflectionConfig}, without the
 * on/off decision. For callers that already decided to run a server, such as
 * a direct `ReflectionServer.start()`.
 */
export function resolveReflectionServerConfig(
  env: ReflectionEnv,
  options: { port?: number } = {}
): ReflectionServerConfig {
  // Validated even for v2, which has no port, so a bad value always fails.
  resolveOptionPort(options.port);
  const secret = env.GENKIT_REFLECTION_SECRET_TOKEN || undefined;
  if (env.GENKIT_REFLECTION_V2_SERVER) {
    return { kind: 'v2', url: env.GENKIT_REFLECTION_V2_SERVER, secret };
  }
  return {
    kind: 'v1',
    host: env.GENKIT_REFLECTION_HOST || DEFAULT_REFLECTION_HOST,
    port: resolveReflectionPort(
      parsePort(env.GENKIT_REFLECTION_PORT),
      options.port
    ),
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

/**
 * Whether a host is loopback, and so unreachable from other machines. Only
 * `localhost` and literal loopback IPs count (matching the CLI); a hostname
 * like `127.internal.example` may resolve anywhere.
 */
export function isLoopbackHost(host: string): boolean {
  if (host === 'localhost') {
    return true;
  }
  if (/^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(host)) {
    return true;
  }
  const unbracketed = host.replace(/^\[(.*)\]$/, '$1');
  if (!unbracketed.includes(':')) {
    return false;
  }
  // URL normalizes IPv6 literals, which covers every spelling of ::1
  // (e.g. 0:0:0:0:0:0:0:1). Invalid literals throw.
  try {
    return new URL(`http://[${unbracketed}]`).hostname === '[::1]';
  } catch {
    return false;
  }
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
