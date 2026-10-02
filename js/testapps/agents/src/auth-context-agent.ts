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

import { z } from 'genkit';
import { InMemorySessionStore } from 'genkit/beta';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Auth Context Agent: demonstrates passing action context (ex. auth) to an
// in-process agent.
//
// Over HTTP, context is derived from the request by a context provider. When
// calling an agent in-process (from a script, worker, or custom handler) pass
// it explicitly:
//
//   const chat = agent.chat({}, { context: { auth: { uid } } });
//
// The context is bound to the chat and flows to the agent function, its
// tools (via `context` / `getContext()`), and the session store (turns,
// `loadChat`, `getSnapshot`, `abort`, detached task polling). Without an
// explicit context, the ambient one (if any) is used.
// ---------------------------------------------------------------------------

const ORDERS: Record<string, string[]> = {
  alice: ['#1001 espresso machine', '#1002 coffee beans'],
  bob: ['#2001 hiking boots'],
};

export const listMyOrders = ai.defineTool(
  {
    name: 'listMyOrders',
    description: "Lists the signed-in user's orders.",
    inputSchema: z.object({}),
    outputSchema: z.array(z.string()),
  },
  async (_, { context }) => {
    const uid = context.auth?.uid;
    if (typeof uid !== 'string') {
      throw new Error('Not signed in.');
    }
    return ORDERS[uid] ?? [];
  }
);

export const authContextAgent = ai.defineAgent({
  name: 'authContextAgent',
  system:
    'You are a shopping assistant. Use the listMyOrders tool to answer questions about the user orders.',
  tools: [listMyOrders],
  store: new InMemorySessionStore(),
});

export const testAuthContextAgent = ai.defineFlow(
  {
    name: 'testAuthContextAgent',
    inputSchema: z.object({
      uid: z.string().default('alice'),
      text: z.string().default('What did I order?'),
    }),
    outputSchema: z.any(),
  },
  async ({ uid, text }, { sendChunk }) => {
    const context = { auth: { uid } };

    // Bind the caller's context to the chat; every turn uses it.
    const chat = authContextAgent.chat({}, { context });
    const turn = chat.sendStream(text);
    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    await turn.response;

    // Store reads take context too (ex. for tenant-scoped stores), and the
    // restored chat stays bound to it.
    const restored = await authContextAgent.loadChat(
      { snapshotId: chat.snapshotId! },
      { context }
    );
    const res = await restored.send('How many orders is that?');
    return res.raw;
  }
);
