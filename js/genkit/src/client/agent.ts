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

import {
  createAgentAPI,
  type AgentAPI,
  type AgentTransport,
  type SnapshotLookup,
  type WaitForSnapshotOptions,
} from '@genkit-ai/ai/agent-core';

import type {
  AgentInit,
  AgentInput,
  AgentOutput,
  AgentStreamChunk,
} from '@genkit-ai/ai';
import type { SessionSnapshot } from '@genkit-ai/ai/session';
import { HttpStatusError, runFlow, streamFlow } from './client.js';

// Re-export the transport-agnostic agent-client surface so existing imports
// from `genkit/beta/client` keep working.
export {
  AgentError,
  type AgentAPI,
  type AgentChat,
  type AgentChunk,
  type AgentInterrupt,
  type AgentResponse,
  type AgentTurn,
  type DetachedTask,
  type WaitForSnapshotOptions,
} from '@genkit-ai/ai/agent-core';

// Re-export the JSON Patch helper so apps can apply a chunk's `customPatch` to
// their own locally tracked copy of the agent's custom state.
export { applyPatch, type JsonPatch } from '@genkit-ai/ai/json-patch';

/**
 * Options for {@link remoteAgent}.
 */
export interface RemoteAgentOptions {
  /** Required. The agent endpoint. */
  url: string;
  /** Optional. Defaults to `${url}/getSnapshot`. */
  getSnapshotUrl?: string;
  /**
   * Optional. Defaults to `${url}/waitForSnapshot`. Where the agent's
   * `waitForSnapshotAction` is mounted, `waitForSnapshot` and
   * `DetachedTask.wait` follow a background task server-side; where nothing
   * is mounted (the route answers 404), they poll `getSnapshotUrl` instead.
   */
  waitForSnapshotUrl?: string;
  /** Optional. Defaults to `${url}/abort`. */
  abortUrl?: string;
  /** Optional. Static headers, or a function called per request. */
  headers?:
    | Record<string, string>
    | (() => Record<string, string> | Promise<Record<string, string>>);
  /** Optional. Declares server- vs client-managed state; inferred otherwise. */
  stateManagement?: 'server' | 'client';
}

// ---------------------------------------------------------------------------
// remoteAgent factory
// ---------------------------------------------------------------------------

/**
 * Whether a request failed because nothing is mounted at its URL, as opposed
 * to the action there reporting an error: a 404 whose body is not a Genkit
 * error (an action's NOT_FOUND carries its `status` in a JSON body).
 */
function isMissingRoute(e: unknown): boolean {
  if (!(e instanceof HttpStatusError) || e.httpStatus !== 404) return false;
  try {
    return typeof JSON.parse(e.body)?.status !== 'string';
  } catch {
    return true;
  }
}

/**
 * Creates a typed client for talking to a Genkit agent over HTTP.
 *
 * ```ts
 * import { remoteAgent } from 'genkit/beta/client';
 *
 * const agent = remoteAgent<WeatherState>({
 *   url: '/api/weatherAgent',
 * });
 * const chat = agent.chat();
 * const res = await chat.send('Weather in Tokyo?').response;
 * console.log(res.text);
 * ```
 */
export function remoteAgent<State = unknown>(
  options: RemoteAgentOptions
): AgentAPI<State> {
  const { url } = options;
  const getSnapshotUrl = options.getSnapshotUrl ?? `${url}/getSnapshot`;
  const waitForSnapshotUrl =
    options.waitForSnapshotUrl ?? `${url}/waitForSnapshot`;
  const abortUrl = options.abortUrl ?? `${url}/abort`;
  let waitRouteMissing = false;

  const resolveHeaders = async (): Promise<
    Record<string, string> | undefined
  > => {
    if (!options.headers) return undefined;
    if (typeof options.headers === 'function') {
      return options.headers();
    }
    return options.headers;
  };

  const transport: AgentTransport = {
    stateManagement: options.stateManagement,

    runTurn(
      input: AgentInput,
      init: AgentInit,
      opts: { abortSignal: AbortSignal }
    ) {
      // Kick off the request lazily so headers can be resolved asynchronously.
      const started = (async () => {
        const headers = await resolveHeaders();
        return streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({
          url,
          input,
          init,
          headers,
          abortSignal: opts.abortSignal,
        });
      })();

      const output = (async () => {
        const { output } = await started;
        return output;
      })();

      const stream = (async function* (): AsyncIterable<AgentStreamChunk> {
        const { stream: rawStream } = await started;
        yield* rawStream;
      })();

      return { stream, output };
    },

    async getSnapshot(lookup: SnapshotLookup) {
      const headers = await resolveHeaders();
      return runFlow<SessionSnapshot<State> | undefined>({
        url: getSnapshotUrl,
        input: lookup,
        headers,
      });
    },

    // The server blocks next to its store and answers once the snapshot
    // settles, or once its wait limit passes (the API then asks again), so a
    // client neither picks a cadence nor pays a round trip per tick. Aborting
    // the signal drops the request. A server without the wait route (one
    // deployed before it existed) is remembered, and the API polls
    // getSnapshot for it instead.
    async waitForSnapshot(snapshotId: string, opts?: WaitForSnapshotOptions) {
      if (!waitRouteMissing) {
        const headers = await resolveHeaders();
        try {
          return await runFlow<SessionSnapshot<State> | undefined>({
            url: waitForSnapshotUrl,
            input: { snapshotId },
            headers,
            abortSignal: opts?.abortSignal,
          });
        } catch (e) {
          if (!isMissingRoute(e)) throw e;
          waitRouteMissing = true;
        }
      }
      throw Object.assign(
        new Error(`No waitForSnapshot route at ${waitForSnapshotUrl}.`),
        { status: 'UNIMPLEMENTED' }
      );
    },

    async abort(snapshotId: string) {
      const headers = await resolveHeaders();
      const result = await runFlow<{
        snapshotId: string;
        status?: SessionSnapshot['status'];
      }>({
        url: abortUrl,
        input: { snapshotId },
        headers,
      });
      return result?.status;
    },
  };

  return createAgentAPI<State>(transport);
}
