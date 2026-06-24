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
} from '@genkit-ai/ai/agent-core';

import type {
  AgentInit,
  AgentInput,
  AgentOutput,
  AgentStreamChunk,
} from '@genkit-ai/ai';
import type { SessionSnapshot } from '@genkit-ai/ai/session';
import { runFlow, streamFlow } from './client.js';

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
  const abortUrl = options.abortUrl ?? `${url}/abort`;

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
