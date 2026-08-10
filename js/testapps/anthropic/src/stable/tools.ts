/**
 * Copyright 2025 Google LLC
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

import { anthropic } from '@genkit-ai/anthropic';
import { genkit, z } from 'genkit';

const ai = genkit({
  plugins: [
    // Configure the plugin with environment-driven API key
    anthropic(),
  ],
});

const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Gets the current weather in a given location',
    inputSchema: z.object({
      location: z
        .string()
        .describe('The location to get the current weather for'),
    }),
    outputSchema: z.string(),
  },
  async (input: any) => {
    // Here, we would typically make an API call or database query. For this
    // example, we just return a fixed value.
    return `The current weather in ${input.location} is 63°F and sunny.`;
  }
);

ai.defineFlow(
  'anthropic-stable-tools',
  async ({ place }: { place: string }) => {
    const { text } = await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      tools: [getWeather],
      prompt: `What is the weather in ${place}?`,
    });

    return text;
  }
);
