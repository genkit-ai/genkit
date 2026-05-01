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
  name: 'weatherPrompt',
  model: 'googleai/gemini-flash-latest',
  input: { schema: z.object({ name: z.string() }) },
  system:
    'You are an assistant helping {{ name }} with weather information. Use the getWeather tool.',
  tools: [getWeather],
  defaultInput: { name: 'Bratwurst' },
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
    const res = await weatherAgent.run(
      {
        messages: [{ role: 'user', content: [{ text }] }],
      },
      {
        onChunk: sendChunk,
        init: { state: { inputVariables: { name: 'Pavel' } } },
      }
    );
    return res.result;
  }
);

export const testWeatherAgentStream = ai.defineFlow(
  {
    name: 'testWeatherAgentStream',
    inputSchema: z.string().default('What is the weather like in Paris?'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const session = weatherAgent.streamBidi({});
    session.send({
      messages: [{ role: 'user' as const, content: [{ text }] }],
    });
    session.send({
      messages: [
        {
          role: 'user' as const,
          content: [{ text: 'now say that in French' }],
        },
      ],
    });
    session.close();

    for await (const chunk of session.stream) {
      sendChunk(chunk);
    }

    return await session.output;
  }
);
