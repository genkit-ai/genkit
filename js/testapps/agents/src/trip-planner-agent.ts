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
 * Prompt-File Agent — demonstrates definePromptAgent
 *
 * This sample shows how to define an agent whose prompt lives in a
 * `.prompt` file (dotprompt) instead of being specified inline.
 *
 * The prompt template is in `prompts/tripPlanner.prompt` and references
 * tools declared here. `definePromptAgent` wires the prompt file into
 * a multi-turn agent without needing `defineAgent` or inline prompt
 * configuration.
 *
 * This is useful when:
 *   • You want prompt authors (non-developers) to edit `.prompt` files
 *   • You want to reuse a single prompt across multiple agents
 *   • You prefer separating prompt content from agent wiring logic
 */

import { z } from 'genkit';
import { FileSessionStore } from 'genkit/beta';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Tools referenced by the .prompt file
// ---------------------------------------------------------------------------

export const getAttractions = ai.defineTool(
  {
    name: 'getAttractions',
    description: 'Get popular tourist attractions for a given city.',
    inputSchema: z.object({ city: z.string() }),
    outputSchema: z.object({
      attractions: z.array(
        z.object({
          name: z.string(),
          description: z.string(),
        })
      ),
    }),
  },
  async (input) => {
    // Mock data for demonstration
    const data: Record<string, Array<{ name: string; description: string }>> = {
      paris: [
        { name: 'Eiffel Tower', description: 'Iconic iron lattice tower' },
        { name: 'Louvre Museum', description: 'World-renowned art museum' },
        {
          name: 'Notre-Dame Cathedral',
          description: 'Medieval Catholic cathedral',
        },
      ],
      tokyo: [
        { name: 'Senso-ji Temple', description: 'Ancient Buddhist temple' },
        { name: 'Shibuya Crossing', description: 'Famous busy intersection' },
        { name: 'Meiji Shrine', description: 'Shinto shrine in a forest' },
      ],
    };
    const key = input.city.toLowerCase();
    return {
      attractions: data[key] || [
        {
          name: `${input.city} Central Park`,
          description: 'A lovely park in the city center',
        },
        {
          name: `${input.city} History Museum`,
          description: 'Learn about the local history',
        },
      ],
    };
  }
);

export const getFlightInfo = ai.defineTool(
  {
    name: 'getFlightInfo',
    description:
      'Get mock flight information between two cities on a given date.',
    inputSchema: z.object({
      from: z.string(),
      to: z.string(),
      date: z.string().optional(),
    }),
    outputSchema: z.object({
      flights: z.array(
        z.object({
          airline: z.string(),
          departure: z.string(),
          arrival: z.string(),
          price: z.string(),
        })
      ),
    }),
  },
  async (input) => {
    return {
      flights: [
        {
          airline: 'SkyAir',
          departure: '08:00',
          arrival: '11:30',
          price: '$350',
        },
        {
          airline: 'GlobalJet',
          departure: '14:15',
          arrival: '17:45',
          price: '$420',
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// Agent — wired from the .prompt file via definePromptAgent
// ---------------------------------------------------------------------------

export const tripPlannerAgent = ai.definePromptAgent({
  promptName: 'tripPlanner',
  store: new FileSessionStore('./.snapshots'),
});

// ---------------------------------------------------------------------------
// Test flow
// ---------------------------------------------------------------------------

export const testTripPlannerAgent = ai.defineFlow(
  {
    name: 'testTripPlannerAgent',
    inputSchema: z
      .string()
      .default('I want to plan a trip to Paris. What should I see there?'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const chat = tripPlannerAgent.chat();
    const turn = chat.sendStream(text);

    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    const res = await turn.response;
    return res.raw;
  }
);
