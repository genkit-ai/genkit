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
