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

import type { A2AClient } from '@a2a-js/sdk/client';
import type { Agent, GenkitBeta } from 'genkit/beta';

import {
  createA2aClient,
  runA2aTurn,
  type A2aConnectionOptions,
} from './a2a-client-turn.js';

/**
 * Options for {@link defineA2aAgent}.
 */
export interface A2aAgentOptions extends A2aConnectionOptions {
  /**
   * Name to register the agent under in the Genkit registry. Defaults to the
   * remote agent card's `name`.
   */
  name?: string;
  /**
   * Description surfaced on the agent's action metadata (e.g. in the Dev UI).
   * Defaults to the remote agent card's `description`.
   */
  description?: string;
  /**
   * Internal seam for tests: a pre-built A2A client (and its card) to use
   * instead of resolving one from `url`/`card`. Not part of the public API.
   *
   * @internal
   */
  _client?: { client: A2AClient; name: string; description?: string };
}

/**
 * Defines and registers a Genkit agent backed by a **remote A2A agent**.
 *
 * This is the inverse of {@link GenkitA2ARequestHandler}: instead of exposing a
 * Genkit agent over A2A, it consumes a remote A2A agent and exposes it as a
 * first-class, registered Genkit agent (via `ai.defineCustomAgent`). Because it
 * is a real registered action, it shows up in the Dev UI, participates in
 * tracing/observability, can be used as a sub-agent/tool, and can even be
 * re-exposed over A2A with `GenkitA2ARequestHandler`.
 *
 * Each Genkit turn drives one turn against the remote A2A agent: the Genkit
 * session is kept in sync locally, and the A2A `contextId` (the Genkit
 * `sessionId`) threads conversation continuity to the remote. Interrupts on the
 * remote (`input-required`) surface as Genkit interrupts and can be resumed.
 *
 * @example
 * ```ts
 * import { genkit } from 'genkit/beta';
 * import { defineA2aAgent } from '@genkit-ai/a2a';
 *
 * const ai = genkit({});
 * const weather = await defineA2aAgent(ai, { url: 'https://some-host' });
 *
 * const chat = weather.chat();
 * const res = await chat.send('Weather in Tokyo?').response;
 * console.log(res.text);
 * ```
 */
export async function defineA2aAgent(
  ai: GenkitBeta,
  options: A2aAgentOptions
): Promise<Agent> {
  // Resolve the client + card up front so the registered agent has a
  // meaningful name/description (defaulting to the remote card's).
  let client: A2AClient;
  let cardName: string;
  let cardDescription: string | undefined;

  if (options._client) {
    client = options._client.client;
    cardName = options._client.name;
    cardDescription = options._client.description;
  } else {
    const resolved = await createA2aClient(options);
    client = resolved.client;
    cardName = resolved.card.name;
    cardDescription = resolved.card.description;
  }

  const name = options.name ?? cardName;
  const description = options.description ?? cardDescription;

  if (!name) {
    throw new Error(
      'Unable to define an A2A agent: no `name` provided and the remote ' +
        'agent card has no name.'
    );
  }

  // Per-context (sessionId) task tracking so a follow-up turn can resume an
  // `input-required` task on the remote.
  const taskByContext = new Map<string, string>();

  return ai.defineCustomAgent(
    { name, description },
    async (sess, { sendChunk, abortSignal }) => {
      await sess.run(async (input) => {
        const contextId = sess.session.sessionId;
        const taskId = taskByContext.get(contextId);

        const turn = runA2aTurn(client, input, {
          contextId,
          taskId,
          abortSignal,
        });

        // Stream artifact parts as model chunks as they arrive.
        let next = await turn.next();
        while (!next.done) {
          sendChunk({ modelChunk: { role: 'model', content: next.value } });
          next = await turn.next();
        }
        const result = next.value;

        // Remember the task for this context so a follow-up can resume it.
        if (result.taskId) {
          taskByContext.set(contextId, result.taskId);
        }

        if (result.finishReason === 'failed') {
          throw new Error(result.error?.message ?? 'Remote A2A agent failed.');
        }

        if (result.message) {
          sess.addMessages([result.message]);

          // Surface interrupt tool requests as a `tool` chunk (mirroring the
          // prompt agent), so streaming clients see the pending interrupts.
          if (result.finishReason === 'interrupted') {
            const parts = (result.message.content ?? []).filter(
              (p) => !!p.toolRequest
            );
            if (parts.length > 0) {
              sendChunk({ modelChunk: { role: 'tool', content: parts } });
            }
          }
        }

        return { finishReason: result.finishReason };
      });

      const msgs = sess.getMessages();
      return {
        artifacts: sess.getArtifacts(),
        message: msgs.length > 0 ? msgs[msgs.length - 1] : undefined,
        ...(sess.lastTurnFinishReason && {
          finishReason: sess.lastTurnFinishReason,
        }),
      };
    }
  );
}
