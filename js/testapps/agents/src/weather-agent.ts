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
import { FileSessionStore } from 'genkit/beta';
import { ai } from './genkit.js';

export const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Get the current weather for a given location.',
    inputSchema: z.object({ location: z.string() }),
    outputSchema: z.object({ weather: z.string() }),
  },
  async (input) => {
    return { weather: `Sunny in ${input.location}`, temperature: '71F' };
  }
);

export const weatherAgent = ai.defineAgent({
  name: 'weatherAgent',
  system:
    'You are an assistant helping with weather information. Use the getWeather tool.',
  tools: [getWeather],
  store: new FileSessionStore('./.snapshots'),
});

export const testWeatherAgent = ai.defineFlow(
  {
    name: 'testWeatherAgent',
    inputSchema: z
      .string()
      .default('Hello, what is the weather like in London?'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    // The ergonomic agent API: start a chat, send a message, stream the
    // chunks, and await the final response.
    const chat = weatherAgent.chat();
    const turn = chat.sendStream(text);
    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    const res = await turn.response;
    return res.raw;
  }
);

export const testWeatherAgentStream = ai.defineFlow(
  {
    name: 'testWeatherAgentStream',
    inputSchema: z.string().default('What is the weather like in Paris?'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    // Multi-turn: a single `chat` carries state across turns automatically,
    // so we just call `send` again for the follow-up.
    const chat = weatherAgent.chat();

    const turn1 = chat.sendStream(text);
    for await (const chunk of turn1.stream) {
      sendChunk(chunk.raw);
    }
    await turn1.response;

    const turn2 = chat.sendStream('now say that in French');
    for await (const chunk of turn2.stream) {
      sendChunk(chunk.raw);
    }
    const res = await turn2.response;

    return res.raw;
  }
);
