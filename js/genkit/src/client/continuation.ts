/**
 * @license
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * Client-safe continuation token helpers. Mirror the server-side codec in
 * `@genkit-ai/ai/agent.ts` — duplicated here so client bundles don't need
 * to pull in the AI package (which transitively imports Node built-ins).
 *
 * Token format (opaque to clients, server discriminates on prefix):
 *
 *   `snap:<snapshotId>`           — server-stored agents
 *   `state:<base64(JSON(state))>` — client-stored agents
 *
 * The prefix is a storage-kind discriminator, not a version.
 */

const SNAP_PREFIX = 'snap:';
const STATE_PREFIX = 'state:';

/** Encode a snapshotId as a continuation token. */
export function encodeSnapshotContinuation(snapshotId: string): string {
  return SNAP_PREFIX + snapshotId;
}

/** Encode a state blob as a continuation token (client-stored agents). */
export function encodeStateContinuation(state: unknown): string {
  const json = JSON.stringify(state);
  if (typeof btoa !== 'undefined') {
    return STATE_PREFIX + btoa(unescape(encodeURIComponent(json)));
  }
  // Node fallback for SSR/test contexts.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const B: any = (globalThis as any).Buffer;
  return STATE_PREFIX + B.from(json, 'utf8').toString('base64');
}

/**
 * Extract the snapshotId from a snap-shaped continuation token, or
 * undefined if the token is state-shaped or malformed. Useful when
 * constructing snapshot-bound URLs or calling sibling endpoints.
 */
export function continuationToSnapshotId(token?: string): string | undefined {
  if (!token) return undefined;
  return token.startsWith(SNAP_PREFIX)
    ? token.slice(SNAP_PREFIX.length)
    : undefined;
}
