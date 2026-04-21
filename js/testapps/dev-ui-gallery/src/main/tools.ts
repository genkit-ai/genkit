/**
 * Copyright 2024 Google LLC
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

import { googleAI } from '@genkit-ai/google-genai';
import { z } from 'genkit';
import { WeatherSchema } from '../common/types';
import { ai } from '../genkit.js';

ai.defineTool(
  {
    name: 'getWeather',
    description: 'Get the weather for the given location.',
    inputSchema: z.object({ city: z.string() }),
    outputSchema: z.object({
      temperatureF: z.number(),
      conditions: z.string(),
    }),
  },
  async () => {
    const conditions = ['Sunny', 'Cloudy', 'Partially Cloudy', 'Raining'];
    const c = Math.floor(Math.random() * conditions.length);
    const temp = Math.floor(Math.random() * (120 - 32) + 32);

    return {
      temperatureF: temp,
      conditions: conditions[c],
    };
  }
);

ai.defineTool(
  {
    name: 'getTime',
    description: 'Get the current time',
    inputSchema: z.object({
      city: z.string().min(16),
      state: z.string().optional(),
      hours: z.number().multipleOf(12).min(12).max(24),
      timezone: z.string().default('EST'),
    }),
    outputSchema: z.object({ time: z.number() }),
  },
  async () => {
    return { time: Date.now() };
  }
);

ai.defineTool(
  {
    name: 'getSomePreformattedText',
    description: 'Gets some pre-formatted text to display on the UI.',
    inputSchema: z.object({ message: z.string().optional() }),
    outputSchema: z.string(),
  },
  async (input) => {
    return `Here is some pre-formatted text\n\n${input.message}\n\nWow, that's nice!`;
  }
);

ai.defineTool(
  {
    name: 'getSomeMarkdown',
    description: 'Gets some markdown formatted text to display on the UI.',
    inputSchema: z.object({ message: z.string().optional() }),
    outputSchema: z.string(),
  },
  async (input) => {
    return `# Rendering Markdown\nHere is some **markdown** text:\n${input.message}\nWow, that's nice!`;
  }
);

const template = `
  {{role "system"}}
  Always try to be as efficient as possible, and request tool calls in batches.

  {{role "user"}}
  Help me decide which is a better place to visit today based on the weather.
  I want to be outside as much as possible. Here are the cities I am
  considering:\n\n{{#each cities}}{{this}}\n{{/each}}`;

export const weatherPrompt = ai.definePrompt({
  name: 'weatherPrompt',
  model: googleAI.model('gemini-flash-latest'),
  input: {
    schema: WeatherSchema,
  },
  output: {
    format: 'text',
  },
  config: {
    maxOutputTokens: 2048,
    temperature: 0.6,
    topK: 16,
    topP: 0.95,
  },
  tools: ['getWeather'],
  messages: template,
});

ai.defineFlow(
  {
    name: 'flowWeather',
    inputSchema: WeatherSchema,
    outputSchema: z.string(),
  },
  async (input) => (await weatherPrompt(input)).text
);
