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
 * Client-managed state weather agent — same as the regular weather agent but
 * with NO server-side store. The client owns the session state blob and must
 * echo it back on every subsequent turn via `init: { state }`.
 *
 * This demonstrates that tool-calling and multi-turn work just fine without
 * a server store — the session history lives in the state blob that the
 * client round-trips.
 */

import { z } from 'genkit';
import { ai } from './genkit.js';

// Reuse the same getWeather tool from tool-agent.
import { getWeather } from './tool-agent.js';

// No store — client-managed state!
export const clientStateAgent = ai.defineAgent({
  name: 'clientWeatherPrompt',
  model: 'googleai/gemini-flash-latest',
  input: { schema: z.object({ name: z.string() }) },
  system:
    'You are a helpful weather assistant for {{ name }}. Use the getWeather tool to look up weather. Be concise.',
  tools: [getWeather],
  defaultInput: { name: 'Friend' },
  // No `store` property → stateless. The client must round-trip the `state` blob.
});

export const testClientStateAgent = ai.defineFlow(
  {
    name: 'testClientStateAgent',
    inputSchema: z.object({
      // `state` is the full SessionState returned from a prior turn; omit on
      // first call. The client owns this blob and must echo it back each turn.
      state: z.any().optional(),
      text: z.string().default('What is the weather in Tokyo?'),
    }),
    outputSchema: z.any(),
  },
  async (input, { sendChunk }) => {
    const res = await clientStateAgent.run(
      {
        messages: [{ role: 'user' as const, content: [{ text: input.text }] }],
      },
      {
        init: {
          // Resume from the state returned by the previous turn, or start fresh.
          state: input.state,
        },
        onChunk: sendChunk,
      }
    );
    // Return the updated state so the caller can pass it back on the next turn.
    return {
      state: res.result.state,
      message: res.result.message,
    };
  }
);
