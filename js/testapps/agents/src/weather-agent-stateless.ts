/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * Client-managed-state weather agent. No server-side store: the session
 * state lives inside the opaque `continuationId` token that the client
 * round-trips on every turn. Tool calling and multi-turn work the same as
 * the server-stored variant — clients don't have to know which mode the
 * server uses.
 */

import { z } from 'genkit';
import { ai } from './genkit.js';

// Reuse the same getWeather tool from weather-agent.
import { getWeather } from './weather-agent.js';

export const weatherAgentStateless = ai.defineAgent({
  name: 'weatherAgentStateless',
  system:
    'You are a helpful weather assistant. Use the getWeather tool to look up weather. Be concise.',
  tools: [getWeather],
  // No `store` → stateless. The continuationId encodes the session state.
});

export const testWeatherAgentStateless = ai.defineFlow(
  {
    name: 'testWeatherAgentStateless',
    inputSchema: z.object({
      /** Structured continuation from a previous turn; omit on first call. */
      continuation: z.any().optional(),
      text: z.string().default('What is the weather in Tokyo?'),
    }),
    outputSchema: z.any(),
  },
  async (input, { sendChunk }) => {
    const res = await weatherAgentStateless.run(
      {
        messages: [{ role: 'user' as const, content: [{ text: input.text }] }],
      },
      {
        init: input.continuation ? { continuation: input.continuation } : {},
        onChunk: sendChunk,
      }
    );
    return {
      continuation: res.result.continuation,
      message: res.result.message,
    };
  }
);
