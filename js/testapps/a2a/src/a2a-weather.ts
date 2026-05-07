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
 * A2A Weather Agent Demo
 *
 * Demonstrates two ways of using a remote A2A weather agent:
 *
 * 1. **Direct call** — `defineA2AAgent` wraps the remote agent as a Genkit
 *    Agent, then a test flow calls it directly with a weather question.
 *
 * 2. **Sub-agent delegation** — An orchestrator agent uses the `agents()`
 *    middleware to delegate weather-related questions to the remote A2A agent,
 *    just like it would to any local Genkit sub-agent.
 */

import { defineA2AAgent } from '@genkit-ai/a2a';
import { agents } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Remote A2A Weather Agent — wrapped as a Genkit Agent
// ---------------------------------------------------------------------------

const A2A_WEATHER_URL =
  process.env.A2A_WEATHER_URL || 'http://localhost:8001';

export const weatherA2AAgent = defineA2AAgent(ai, {
  name: 'weatherA2A',
  agentUrl: A2A_WEATHER_URL,
  description:
    'A remote A2A weather agent. Ask it about weather in any city.',
});

// ---------------------------------------------------------------------------
// Flow 1: Direct call to the A2A weather agent
// ---------------------------------------------------------------------------

export const testDirectA2AWeather = ai.defineFlow(
  {
    name: 'testDirectA2AWeather',
    inputSchema: z
      .string()
      .default("What's the weather like in Tokyo?"),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await weatherA2AAgent.run(
      { messages: [{ role: 'user' as const, content: [{ text }] }] },
      { init: {}, onChunk: sendChunk }
    );
    return res.result;
  }
);

// ---------------------------------------------------------------------------
// Flow 2: Sub-agent delegation — orchestrator delegates to the A2A agent
// ---------------------------------------------------------------------------

export const orchestratorWithA2A = ai.defineAgent({
  name: 'orchestratorWithA2A',
  model: 'googleai/gemini-flash-latest',
  system: `You are a helpful assistant. You have access to specialized sub-agents:

- **weatherA2A**: A remote weather agent. Use it to answer any weather-related questions.

When the user asks about weather, delegate to the weatherA2A sub-agent using the call_agent tool.
After receiving the sub-agent's response, present the weather information clearly to the user.
For non-weather questions, answer directly.`,
  use: [agents({ agents: ['weatherA2A'] })],
});

export const testSubAgentA2AWeather = ai.defineFlow(
  {
    name: 'testSubAgentA2AWeather',
    inputSchema: z
      .string()
      .default("What's the weather in Paris right now?"),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await orchestratorWithA2A.run(
      { messages: [{ role: 'user' as const, content: [{ text }] }] },
      { init: {}, onChunk: sendChunk }
    );
    return res.result;
  }
);
