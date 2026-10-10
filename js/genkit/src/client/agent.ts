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
  type AgentTurnOptions,
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

// Re-export the transport-agnostic agent-client surface, including the pieces
// needed to build an `AgentAPI` over a custom transport (`createAgentAPI`,
// `AgentTransport`, `SnapshotLookup`).
export {
  AgentError,
  createAgentAPI,
  type AgentAPI,
  type AgentChat,
  type AgentChunk,
  type AgentInterrupt,
  type AgentResponse,
  type AgentTransport,
  type AgentTurn,
  type AgentTurnOptions,
  type DetachedTask,
  type SnapshotLookup,
} from '@genkit-ai/ai/agent-core';

// Re-export the JSON Patch helper so apps can apply a chunk's `customPatch` to
// their own locally tracked copy of the agent's custom state.
export { applyPatch, type JsonPatch } from '@genkit-ai/ai/json-patch';

/**
 * Per-chat or per-call options for a {@link remoteAgent}. Bound once with
 * `agent.chat(init, opts)` or passed per call (`chat.send(input, opts)`,
 * `task.abort(opts)`, ...).
 *
 * A per-call `headers` replaces the chat-bound `headers` wholesale (the same
 * shallow semantics as in-process `context`). Either one is layered over the
 * `headers` given to {@link remoteAgent}, key by key.
 */
export interface RemoteAgentCallOptions {
  /** Extra HTTP headers for the request, ex. a user's bearer token. */
  headers?: Record<string, string>;
}

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
  /**
   * Optional. Static headers, or a function called per request. Overridden key
   * by key by {@link RemoteAgentCallOptions.headers}.
   */
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
 * const res = await chat.send('Weather in Tokyo?');
 * console.log(res.text);
 * ```
 *
 * Unlike in-process agents, the returned API takes no `context` option: over
 * HTTP the action context is derived server-side from the request (ex. from
 * the `headers` option). Per-chat or per-call {@link RemoteAgentCallOptions}
 * carry request-scoped headers, ex. `agent.chat({}, { headers })`.
 */
export function remoteAgent<State = unknown>(
  options: RemoteAgentOptions
): AgentAPI<State, RemoteAgentCallOptions> {
  return createAgentAPI<State, RemoteAgentCallOptions>(
    remoteAgentTransport(options)
  );
}

/**
 * The HTTP {@link AgentTransport} behind {@link remoteAgent}. Useful for
 * decorating the HTTP transport (logging, retries, ...) before wrapping it
 * with {@link createAgentAPI}:
 *
 * ```ts
 * const http = remoteAgentTransport({ url: '/api/weatherAgent' });
 * const agent = createAgentAPI<WeatherState>({
 *   ...http,
 *   runTurn(input, init, opts) {
 *     console.log('turn from', init.snapshotId ?? 'new session');
 *     return http.runTurn(input, init, opts);
 *   },
 * });
 * ```
 */
export function remoteAgentTransport(
  options: RemoteAgentOptions
): AgentTransport<RemoteAgentCallOptions> {
  const { url } = options;
  const getSnapshotUrl = options.getSnapshotUrl ?? `${url}/getSnapshot`;
  const abortUrl = options.abortUrl ?? `${url}/abort`;

  const resolveHeaders = async (
    callHeaders?: Record<string, string>
  ): Promise<Record<string, string> | undefined> => {
    const base =
      typeof options.headers === 'function'
        ? await options.headers()
        : options.headers;
    return base || callHeaders ? { ...base, ...callHeaders } : undefined;
  };

  return {
    stateManagement: options.stateManagement,

    runTurn(
      input: AgentInput,
      init: AgentInit,
      opts: AgentTurnOptions<RemoteAgentCallOptions> & {
        abortSignal: AbortSignal;
      }
    ) {
      // Kick off the request lazily so headers can be resolved asynchronously.
      const started = (async () => {
        const headers = await resolveHeaders(opts.headers);
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

    async getSnapshot(lookup: SnapshotLookup, opts?: RemoteAgentCallOptions) {
      const headers = await resolveHeaders(opts?.headers);
      return runFlow<SessionSnapshot<unknown> | undefined>({
        url: getSnapshotUrl,
        input: lookup,
        headers,
      });
    },

    async abort(snapshotId: string, opts?: RemoteAgentCallOptions) {
      const headers = await resolveHeaders(opts?.headers);
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
}
