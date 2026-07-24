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
 * A2A server — exposes the Genkit `conciergeAgent` over the A2A protocol.
 *
 * Run with:  pnpm server
 * Then talk to it with the bundled client:  pnpm client
 * (or any A2A-compatible client / the A2A inspector).
 */

import {
  agentCardHandler,
  jsonRpcHandler,
  UserBuilder,
} from '@a2a-js/sdk/server/express';
import { GenkitA2ARequestHandler, InMemoryA2ATaskStore } from '@genkit-ai/a2a';
import express from 'express';
import { conciergeAgent } from './concierge-agent.js';

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 4000;
const URL = process.env.A2A_URL ?? `http://localhost:${PORT}`;

// Build the A2A request handler around the Genkit agent. The agent card is
// derived automatically from the agent's name/description; `url` tells the card
// where the agent is hosted.
//
// Because `conciergeAgent` is server-managed (it has a `store`), the handler is
// snapshot-native: an A2A `taskId` is the Genkit snapshot id of the turn that
// originated it, and `getTask` reads straight from the agent's SessionStore.
// The `taskStore` below only records an advancement pointer when an interrupted
// task is resumed (a resumed turn writes a new snapshot). It defaults to
// `InMemoryA2ATaskStore`; passing it explicitly here makes that visible — swap
// in a durable implementation to survive restarts or span processes.
const a2aHandler = new GenkitA2ARequestHandler({
  agent: conciergeAgent,
  url: URL,
  version: '1.0.0',
  taskStore: new InMemoryA2ATaskStore(),
});

const app = express();
app.use(express.json());

// A2A JSON-RPC endpoint (the primary A2A transport).
app.use(
  '/',
  jsonRpcHandler({
    // Cast guards against duplicate @a2a-js/sdk copies (express 4 vs 5) in the
    // monorepo resolving to structurally-identical-but-nominally-distinct types.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    requestHandler: a2aHandler as any,
    userBuilder: UserBuilder.noAuthentication,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }) as any
);

// Agent card discovery.
app.use(
  '/.well-known/agent-card.json',
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  agentCardHandler({ agentCardProvider: a2aHandler as any }) as any
);

app.listen(PORT, () => {
  console.log(`\n🤖 A2A concierge agent running at ${URL}`);
  console.log(`   Agent card: ${URL}/.well-known/agent-card.json`);
  console.log(`\n   Try it:  pnpm client\n`);
});
