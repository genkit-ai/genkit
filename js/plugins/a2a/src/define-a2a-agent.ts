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

import { ClientFactory } from '@a2a-js/sdk/client';
import type {
  Message as A2AMessage,
  Task as A2ATask,
  MessageSendParams,
} from '@a2a-js/sdk';
import type {
  GenkitBeta,
  AgentFn,
  SessionStore,
  SnapshotCallback,
} from 'genkit/beta';

import {
  mapGenkitPartToA2A,
  mapA2APartToGenkit,
  mapA2AArtifactToGenkit,
} from './mapping.js';

/**
 * Interface for A2A client creation. Allows injecting custom client factories
 * for testing or custom transport configurations.
 */
export interface A2AClientFactory {
  createFromUrl(url: string): Promise<A2AClientLike>;
}

/**
 * Minimal A2A client interface — the subset we actually use.
 */
export interface A2AClientLike {
  sendMessage(params: MessageSendParams): Promise<A2AMessage | A2ATask>;
}

/**
 * Configuration for `defineA2AAgent`.
 */
export interface A2AAgentConfig<State = unknown> {
  /** Name to register this agent under in the Genkit registry. */
  name: string;
  /** URL of the remote A2A agent (used to discover agent card and send messages). */
  agentUrl: string;
  /** Optional description override. */
  description?: string;
  /** Optional session store for server-managed state. */
  store?: SessionStore<State>;
  /** Optional snapshot callback. */
  snapshotCallback?: SnapshotCallback<State>;
  /**
   * Optional custom A2A client factory.
   * Defaults to `@a2a-js/sdk`'s `ClientFactory`.
   */
  clientFactory?: A2AClientFactory;
}

/**
 * Defines a Genkit Agent that consumes a remote A2A agent.
 *
 * The remote agent is called via the A2A protocol using `message/send`
 * (blocking mode). The resulting Agent can be used exactly like any other
 * Genkit agent — via `streamBidi()`, `.run()`, or as a sub-agent through
 * the `agents()` middleware.
 *
 * @example
 * ```ts
 * import { genkit } from 'genkit/beta';
 * import { defineA2AAgent } from '@genkit-ai/a2a';
 *
 * const ai = genkit({});
 *
 * const remoteAgent = defineA2AAgent(ai, {
 *   name: 'weather',
 *   agentUrl: 'https://weather-agent.example.com',
 * });
 *
 * // Use like any Genkit agent:
 * const result = await remoteAgent.run(
 *   { messages: [{ role: 'user', content: [{ text: 'Weather in London?' }] }] },
 *   { init: {} }
 * );
 * ```
 */
export function defineA2AAgent<State = unknown>(
  ai: GenkitBeta,
  config: A2AAgentConfig<State>
) {
  let cachedClient: A2AClientLike | undefined;

  const fn: AgentFn<unknown, State> = async (sess, { sendChunk }) => {
    // Lazily create the A2A client on first invocation
    if (!cachedClient) {
      const factory = config.clientFactory || new ClientFactory();
      cachedClient = await factory.createFromUrl(config.agentUrl);
    }

    // Get or create a stable A2A contextId for this session.
    // This ensures multi-turn conversations within the same invocation
    // are correctly associated on the remote side.
    let a2aContextId: string | undefined =
      (sess.session.getCustom() as any)?.a2aContextId;
    if (!a2aContextId) {
      a2aContextId = crypto.randomUUID();
      sess.session.updateCustom((custom) => ({
        ...(custom as any),
        a2aContextId,
      }));
    }

    // Track the last A2A taskId for multi-turn continuity
    let a2aTaskId: string | undefined =
      (sess.session.getCustom() as any)?.a2aTaskId;

    await sess.run(async (input) => {
      // Get the latest user message parts
      const latestMessage = input.messages?.[input.messages.length - 1];
      if (!latestMessage) return;

      const a2aParts = latestMessage.content.map(mapGenkitPartToA2A);

      // Determine the role
      const isToolResponse = latestMessage.content.some(
        (p) => 'toolResponse' in p && p.toolResponse !== undefined
      );

      // Build the A2A message
      const a2aMessage: A2AMessage = {
        kind: 'message',
        messageId: crypto.randomUUID(),
        role: isToolResponse ? 'agent' : 'user',
        parts: a2aParts,
        contextId: a2aContextId,
        ...(a2aTaskId && { taskId: a2aTaskId }),
      } as A2AMessage;

      const params: MessageSendParams = {
        message: a2aMessage,
      };

      // Call the remote A2A agent (blocking mode)
      const response = await cachedClient!.sendMessage(params);

      // Process the response
      if (response.kind === 'task') {
        const task = response as A2ATask;

        // Store taskId for multi-turn continuity
        a2aTaskId = task.id;
        sess.session.updateCustom((custom) => ({
          ...(custom as any),
          a2aTaskId: task.id,
        }));

        // Extract artifacts
        if (task.artifacts && task.artifacts.length > 0) {
          const genkitArtifacts = task.artifacts.map(mapA2AArtifactToGenkit);
          sess.session.addArtifacts(genkitArtifacts);
        }

        // Extract the final agent message from history.
        // We look for the last agent message with text parts — this is the
        // final response (ignoring intermediate tool call/response messages).
        const agentMessages = (task.history || []).filter(
          (m) => m.role === 'agent'
        );
        const finalAgentMessage = findFinalTextMessage(agentMessages);

        if (finalAgentMessage) {
          const genkitParts = finalAgentMessage.parts.map(mapA2APartToGenkit);
          sess.session.addMessages([
            { role: 'model', content: genkitParts },
          ]);
        }

        // Stream status if available
        if (task.status?.state) {
          sendChunk({ status: task.status.state });
        }
      } else if (response.kind === 'message') {
        // Direct message response (no task wrapper)
        const msg = response as A2AMessage;
        const genkitParts = msg.parts.map(mapA2APartToGenkit);
        sess.session.addMessages([{ role: 'model', content: genkitParts }]);
      }
    });

    // Return the final result
    const msgs = sess.session.getMessages();
    return {
      message: msgs.length > 0 ? msgs[msgs.length - 1] : undefined,
      artifacts: sess.session.getArtifacts(),
    };
  };

  return ai.defineCustomAgent<unknown, State>(
    {
      name: config.name,
      description: config.description,
      store: config.store,
      snapshotCallback: config.snapshotCallback,
    },
    fn
  );
}

/**
 * Finds the last message in a list that contains at least one text part.
 * This represents the "final answer" from the remote agent, ignoring
 * intermediate tool-call and tool-response messages.
 */
function findFinalTextMessage(
  messages: A2AMessage[]
): A2AMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.parts.some((p) => p.kind === 'text')) {
      return msg;
    }
  }
  return undefined;
}
