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
 * Travel Concierge Agent
 *
 * A deliberately "interesting" agent that exercises every part of the
 * `@genkit-ai/a2a` mapping when driven over the A2A protocol:
 *
 *   • Tool calls + streamed text   -> A2A artifact-update events
 *   • A human-in-the-loop interrupt -> A2A `input-required` (resume the task)
 *   • Multi-turn session memory    -> shared A2A `contextId` == Genkit sessionId
 */

import { z } from 'genkit';
import { InMemorySessionStore } from 'genkit/beta';
import { ai } from './genkit.js';

// A server-managed session store. Because the A2A handler maps the A2A
// `contextId` to the Genkit `sessionId`, conversations sharing a context
// resume their history automatically.
const store = new InMemorySessionStore();

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Get the current weather forecast for a destination.',
    inputSchema: z.object({
      destination: z.string().describe('City or place to check.'),
    }),
    outputSchema: z.object({
      summary: z.string(),
      highC: z.number(),
      lowC: z.number(),
    }),
  },
  async ({ destination }) => {
    // Deterministic fake forecast so the demo is reproducible.
    const seed = destination.length;
    return {
      summary: ['Sunny', 'Partly cloudy', 'Light rain', 'Clear skies'][
        seed % 4
      ],
      highC: 18 + (seed % 12),
      lowC: 8 + (seed % 6),
    };
  }
);

export const searchFlights = ai.defineTool(
  {
    name: 'searchFlights',
    description: 'Search for available flights to a destination.',
    inputSchema: z.object({
      destination: z.string(),
      departureCity: z.string().default('San Francisco'),
    }),
    outputSchema: z.object({
      flights: z.array(
        z.object({
          airline: z.string(),
          priceUsd: z.number(),
          durationHours: z.number(),
        })
      ),
    }),
  },
  async ({ destination }) => {
    const base = 400 + (destination.length % 5) * 120;
    return {
      flights: [
        { airline: 'Genkit Air', priceUsd: base, durationHours: 11 },
        { airline: 'Vertex Wings', priceUsd: base + 150, durationHours: 9 },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Interrupt — human-in-the-loop booking confirmation
// ---------------------------------------------------------------------------

export const confirmBooking = ai.defineInterrupt({
  name: 'confirmBooking',
  description:
    'Ask the traveler to confirm before booking a flight. ALWAYS call this ' +
    'before finalizing any booking.',
  inputSchema: z.object({
    airline: z.string(),
    priceUsd: z.number(),
    destination: z.string(),
  }),
  outputSchema: z.object({
    confirmed: z.boolean(),
    note: z.string().optional(),
  }),
});

export const bookFlight = ai.defineTool(
  {
    name: 'bookFlight',
    description: 'Book a flight once the traveler has confirmed.',
    inputSchema: z.object({
      airline: z.string(),
      destination: z.string(),
    }),
    outputSchema: z.object({
      confirmationCode: z.string(),
    }),
  },
  async () => {
    return { confirmationCode: `GK-${Math.floor(Math.random() * 1e6)}` };
  }
);

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

export const conciergeAgent = ai.defineAgent({
  name: 'conciergeAgent',
  description:
    'A travel concierge that checks the weather, finds flights, and books ' +
    'trips after confirming with you.',
  system: `You are a friendly travel concierge.

Help the user plan a trip:
  1. Use getWeather to report the destination forecast.
  2. Use searchFlights to list flight options.
  3. When the user wants to book, ALWAYS use the confirmBooking interrupt to
     get explicit approval (pass the chosen airline, price, and destination).
  4. Only after the user confirms, call bookFlight and report the confirmation
     code.

Keep responses concise and friendly.`,
  tools: [getWeather, searchFlights, confirmBooking, bookFlight],
  store,
});
