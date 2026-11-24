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

/**
 * Writes a line of primary command output to `stdout` (appending a trailing newline).
 *
 * Use this for intentional CLI payload output (such as JSON, JSONL, tables, or
 * formatted command results) rather than diagnostic messages, which belong on
 * `logger.*`.
 */
export function writeStdout(text = ''): void {
  process.stdout.write(`${text}\n`);
}

/**
 * Writes a line of human-facing CLI output to `stderr` (appending a trailing newline).
 *
 * Use this for unprefixed CLI chrome (such as section headers, hints, empty-state
 * notices, and banners) that should remain visible regardless of log level while
 * keeping `stdout` clean for primary command payloads.
 */
export function writeStderr(text = ''): void {
  process.stderr.write(`${text}\n`);
}

/**
 * Returns `true` when `stderr` is attached to an interactive terminal (TTY).
 */
export function isStderrTTY(): boolean {
  return Boolean(process.stderr.isTTY);
}

/**
 * Returns `true` when both `stdin` and `stderr` are attached to a TTY and
 * `--non-interactive` was not passed on the command line.
 */
export function isInteractive(): boolean {
  return (
    Boolean(process.stdin.isTTY && process.stderr.isTTY) &&
    !process.argv.includes('--non-interactive')
  );
}
