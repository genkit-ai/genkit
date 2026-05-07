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
 * A2A Server Demo — Expose a Genkit Agent as an A2A endpoint
 *
 * This starts an Express server that serves a Genkit agent via the A2A
 * protocol (JSON-RPC). You can point the A2A Inspector or any A2A client
 * at http://localhost:41245 to interact with it.
 *
 * The agent is a simple weather assistant backed by Gemini that uses a
 * mock `getWeather` tool.
 *
 * Usage:
 *   npx tsx src/a2a-server.ts
 *
 * Then open the A2A Inspector and connect to:
 *   http://localhost:41245
 */

import {
  agentCardHandler,
  jsonRpcHandler,
  UserBuilder,
} from '@a2a-js/sdk/server/express';
import { GenkitA2ARequestHandler } from '@genkit-ai/a2a';
import express from 'express';
import { z } from 'genkit';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Define a simple weather tool
// ---------------------------------------------------------------------------

const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Gets the current weather for a given city.',
    inputSchema: z.object({
      city: z.string().describe('The city to get weather for'),
    }),
    outputSchema: z.object({
      city: z.string(),
      temperature: z.number(),
      condition: z.string(),
      humidity: z.number(),
    }),
  },
  async ({ city }) => {
    // Mock weather data
    const conditions = ['sunny', 'cloudy', 'rainy', 'partly cloudy', 'windy'];
    return {
      city,
      temperature: Math.round(15 + Math.random() * 20),
      condition: conditions[Math.floor(Math.random() * conditions.length)],
      humidity: Math.round(40 + Math.random() * 40),
    };
  }
);

// ---------------------------------------------------------------------------
// Define the Genkit weather agent
// ---------------------------------------------------------------------------

const weatherAgent = ai.defineAgent({
  name: 'weatherAgent',
  description:
    'A weather assistant powered by Genkit + Gemini. Ask about weather in any city!',
  model: 'googleai/gemini-flash-latest',
  system: `You are a helpful weather assistant. When users ask about the weather,
use the getWeather tool to look up current conditions. Present the information
in a friendly, conversational way. If the user doesn't specify a city, ask
them which city they'd like weather for.`,
  tools: [getWeather],
});

// ---------------------------------------------------------------------------
// Configure the A2A handler
// ---------------------------------------------------------------------------

const PORT = parseInt(process.env.PORT || '41245', 10);

// The card is derived automatically from the agent's name and description.
// Only `url` is needed so the card knows where the agent is hosted.
const a2aHandler = new GenkitA2ARequestHandler({
  agent: weatherAgent,
  url: `http://localhost:${PORT}`,
});

// ---------------------------------------------------------------------------
// Wire up Express
// ---------------------------------------------------------------------------

const app = express();
app.use(express.json());

// A2A JSON-RPC endpoint (primary A2A transport)
app.use(
  '/',
  jsonRpcHandler({
    requestHandler: a2aHandler,
    userBuilder: UserBuilder.noAuthentication,
  }) as any // eslint-disable-line @typescript-eslint/no-explicit-any
);

// Agent card discovery
app.use(
  '/.well-known/agent-card.json',
  agentCardHandler({ agentCardProvider: a2aHandler }) as any // eslint-disable-line @typescript-eslint/no-explicit-any
);

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', agent: weatherAgent.__action.name });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`\n🚀 A2A Weather Agent running at http://localhost:${PORT}`);
  console.log(
    `   Agent Card: http://localhost:${PORT}/.well-known/agent-card.json`
  );
  console.log(`   JSON-RPC:   http://localhost:${PORT}/`);
  console.log(
    `\nPoint the A2A Inspector at http://localhost:${PORT} to try it out!\n`
  );
});
