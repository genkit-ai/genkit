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

import type { ChildProcess } from 'node:child_process';

/**
 * Inherited vars a box keeps even with `inheritEnv: false`: what a runtime
 * needs to start (PATH to find `node`/`tsx`, locale, temp dirs, CA certs),
 * plus the caller's run mode (`GENKIT_ENV`) and, under `genkit start`, where
 * to export traces (`GENKIT_TELEMETRY_SERVER`). The box's own reflection vars
 * are set by the runner, not inherited.
 */
export const BASE_ENV: readonly RegExp[] = [
  /^(PATH|HOME|USER|LOGNAME|SHELL|TMPDIR|TMP|TEMP|LANG|TZ|TERM)$/,
  /^LC_/,
  /^NODE_/,
  /^(SSL_CERT_FILE|SSL_CERT_DIR)$/,
  /^GENKIT_(ENV|TELEMETRY_SERVER)$/,
  // Windows: process creation and temp/home resolution. Names there are
  // case-insensitive (`Path`, `SystemRoot`), hence `i`.
  /^(PATH|PATHEXT|SYSTEMROOT|COMSPEC|USERPROFILE|APPDATA|LOCALAPPDATA)$/i,
];

/**
 * Which of the parent's env vars a box inherits: all of them (`true`), only
 * {@link BASE_ENV} (`false`), or {@link BASE_ENV} plus the named vars.
 */
export type InheritEnv = boolean | readonly string[];

/**
 * The env a box child is spawned with, lowest precedence first: the inherited
 * (filtered) parent env, then `env`, then `overrides` (the runner's own vars,
 * which must win so a user value can't break the link to the host).
 */
export function childEnv(
  parent: NodeJS.ProcessEnv,
  inherit: InheritEnv,
  env: Record<string, string> | undefined,
  overrides: Record<string, string>
): Record<string, string> {
  const allow =
    typeof inherit === 'boolean' ? undefined : new Set<string>(inherit);
  const inherited: Record<string, string> = {};
  for (const [name, value] of Object.entries(parent)) {
    if (value === undefined) continue;
    if (
      inherit === true ||
      allow?.has(name) ||
      BASE_ENV.some((re) => re.test(name))
    ) {
      inherited[name] = value;
    }
  }
  return { ...inherited, ...env, ...overrides };
}

/**
 * Tokenizes a runner `cmd`. A string is split on whitespace (convenient, but
 * no quoting); pass an array when an argument contains spaces.
 */
export function commandArgv(cmd: string | string[]): string[] {
  const argv = Array.isArray(cmd) ? cmd : cmd.split(/\s+/).filter(Boolean);
  if (argv.length === 0) throw new Error('Box runner: `cmd` is empty.');
  return argv;
}

/**
 * Waits for `ready`, failing fast instead when the child can't be spawned,
 * exits first, or `signal` aborts. Without this a bad command or a crashing
 * entry point only surfaces as the readiness timeout, and a missing binary
 * crashes the host with an unhandled `'error'` event.
 */
export async function untilReady<T>(
  ready: Promise<T>,
  child: ChildProcess,
  what: string,
  signal?: AbortSignal
): Promise<T> {
  const cleanup: Array<() => void> = [];
  const failure = new Promise<never>((_, reject) => {
    const onError = (err: Error) =>
      reject(
        new Error(`${what} failed to start: ${err.message}`, { cause: err })
      );
    const onExit = (code: number | null, sig: NodeJS.Signals | null) =>
      reject(
        new Error(
          `${what} exited before it was ready (${sig ?? `code ${code}`}).`
        )
      );
    child.once('error', onError);
    child.once('exit', onExit);
    cleanup.push(() => {
      child.off('error', onError);
      child.off('exit', onExit);
    });
    if (signal) {
      const onAbort = () =>
        reject(new Error('Aborted before box became ready.'));
      if (signal.aborted) onAbort();
      signal.addEventListener('abort', onAbort, { once: true });
      cleanup.push(() => signal.removeEventListener('abort', onAbort));
    }
  });
  // Whichever loses the race must not surface as an unhandled rejection.
  ready.catch(() => {});
  failure.catch(() => {});
  try {
    return await Promise.race([ready, failure]);
  } finally {
    for (const fn of cleanup) fn();
  }
}
