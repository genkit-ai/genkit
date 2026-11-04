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

import type { ActionContext } from 'genkit';
import {
  createAgentAPI,
  type AgentAPI,
  type AgentOutput,
  type AgentStreamChunk,
  type AgentTransport,
  type SessionSnapshot,
} from 'genkit/beta';
import { Channel } from './channel.js';
import type { ProxyDispatcher } from './proxy.js';
import type { RunActionRequest } from './types.js';

/**
 * Builds a boxed {@link AgentAPI} for the agent named `name`.
 *
 * Mirrors `remoteAgent` (which drives an agent over HTTP): we implement an
 * {@link AgentTransport} whose operations dispatch to the box over a
 * {@link ProxyDispatcher}/{@link BoxConnection}, then hand it to the shared
 * `createAgentAPI` core so callers get the full `chat()` / `loadChat()` /
 * `getSnapshot()` / `abort()` surface, identical to an in-process agent.
 *
 * Wire mapping (matches the actions `defineAgent` registers):
 * - a turn  -> `runAction('/agent/<name>', input, init, stream)`
 * - snapshot -> `runAction('/agent-snapshot/<name>', lookup)`
 * - abort    -> `runAction('/agent-abort/<name>', { snapshotId })`
 */
export function createAgentProxy<State = unknown>(
  dispatcher: ProxyDispatcher,
  name: string,
  /**
   * Routing context for every call this proxy makes (turns, snapshot reads,
   * aborts). Fixed per proxy because `AgentChat` has no per-call context yet;
   * build one proxy per routing key (they are cheap).
   */
  context?: ActionContext
): AgentAPI<State> {
  const agentKey = `/agent/${name}`;
  const snapshotKey = `/agent-snapshot/${name}`;
  const abortKey = `/agent-abort/${name}`;

  const transport: AgentTransport = {
    runTurn(input, init, opts) {
      const chunks = new Channel<AgentStreamChunk>();
      const req: RunActionRequest = { key: agentKey, input, init, context };
      const output = dispatcher
        .acquire(req, opts.abortSignal)
        .then((conn) =>
          conn.runAction<AgentOutput>(req, {
            // The wire carries JSON; the boxed agent's schema is the contract.
            onChunk: (chunk) => chunks.send(chunk as AgentStreamChunk),
            abortSignal: opts.abortSignal,
          })
        )
        .then((res): AgentOutput => {
          chunks.close();
          return res.result ?? {};
        })
        .catch((err) => {
          chunks.error(err);
          throw err;
        });
      // Avoid unhandled rejection when the caller only reads the stream.
      output.catch(() => {});
      return { stream: chunks, output };
    },

    async getSnapshot(lookup) {
      const req: RunActionRequest = {
        key: snapshotKey,
        input: lookup,
        context,
      };
      const conn = await dispatcher.acquire(req);
      const res = await conn.runAction<SessionSnapshot>(req);
      return res.result;
    },

    async abort(snapshotId: string) {
      const req: RunActionRequest = {
        key: abortKey,
        input: { snapshotId },
        context,
      };
      const conn = await dispatcher.acquire(req);
      const res = await conn.runAction<{ status?: SessionSnapshot['status'] }>(
        req
      );
      return res.result?.status;
    },
  };

  return createAgentAPI<State>(transport);
}
