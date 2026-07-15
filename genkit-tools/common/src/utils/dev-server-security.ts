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

import type { NextFunction, Request, Response } from 'express';

/**
 * Environment variable that lets a developer expose the local dev servers on a
 * non-loopback interface (e.g. when running inside a remote dev container).
 * When set, the servers bind to its value and the loopback `Host`-header check
 * is skipped, since the developer has explicitly opted into remote access.
 */
export const DEV_SERVER_HOST_ENV_VAR = 'GENKIT_DEV_SERVER_HOST';

/**
 * Returns the network interface the local dev servers should bind to. Defaults
 * to the loopback interface (`localhost`) so the servers are not reachable from
 * other hosts on the network. Override with `GENKIT_DEV_SERVER_HOST`.
 */
export function getDevServerHost(): string {
  return process.env[DEV_SERVER_HOST_ENV_VAR] || 'localhost';
}

/**
 * Returns true if the given `Host` header refers to the local loopback
 * interface (`localhost`, `127.0.0.1` or `[::1]`, with an optional port). Also
 * accepts the RFC 6761 `*.localhost` names, which browsers resolve to loopback.
 */
export function isLoopbackHost(hostHeader: string | undefined): boolean {
  if (!hostHeader) {
    return false;
  }
  // IPv6 literals are wrapped in brackets, e.g. "[::1]:4000".
  const hostname = hostHeader.startsWith('[')
    ? hostHeader.slice(1, hostHeader.indexOf(']'))
    : hostHeader.split(':')[0];
  return (
    hostname === 'localhost' ||
    hostname === '127.0.0.1' ||
    hostname === '::1' ||
    hostname.endsWith('.localhost')
  );
}

/**
 * Express middleware that rejects requests whose `Host` header is not a
 * loopback address.
 *
 * The local dev servers are meant to be reached only from the developer's
 * machine. The CORS origin check alone does not enforce this: it only controls
 * which origins may read responses, the handler still runs, and it is bypassed
 * by DNS rebinding (where a malicious page the developer visits points its own
 * hostname at 127.0.0.1, making the request same-origin to the attacker). A
 * browser cannot forge the `Host` header, so validating it defeats that attack.
 *
 * Skipped when the developer has opted into exposing the servers via
 * `GENKIT_DEV_SERVER_HOST`.
 */
export function rejectNonLoopbackHost(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  if (
    !process.env[DEV_SERVER_HOST_ENV_VAR] &&
    !isLoopbackHost(req.headers.host)
  ) {
    res.status(403).send('Forbidden: requests must originate from localhost.');
    return;
  }
  next();
}
