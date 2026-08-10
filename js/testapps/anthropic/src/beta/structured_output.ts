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
    // Default all flows in this sample to the beta surface
    anthropic({
      apiVersion: 'beta',
      apiKey: process.env.ANTHROPIC_API_KEY,
    }),
  ],
});

const betaSonnet = anthropic.model('claude-sonnet-4-5', { apiVersion: 'beta' });

ai.defineFlow('anthropic-beta-generate-person-json', async () => {
  const { text } = await ai.generate({
    model: betaSonnet,
    prompt:
      'Generate a fictional person with a random name, random age, and random city.',
    config: { temperature: 0.6 },
    output: {
      schema: z.object({
        name: z.string(),
        age: z.number(),
        city: z.string(),
      }),
      format: 'json',
      constrained: true,
    },
  });

  return text;
});

ai.defineFlow(
  'anthropic-beta-generate-person-json-stream',
  async (_, { sendChunk }) => {
    const { stream } = ai.generateStream({
      model: betaSonnet,
      prompt: [
        {
          text: 'Generate a fictional person with a random name, random age, and random city.',
        },
      ],
      config: { temperature: 0.6 },
      output: {
        schema: z.object({
          name: z.string(),
          age: z.number(),
          city: z.string(),
        }),
        format: 'json',
      },
    });

    const collected: any[] = [];
    for await (const chunk of stream) {
      if (chunk.text) {
        collected.push(chunk.output);
        sendChunk(chunk.output);
      }
    }

    return collected.join('');
  }
);
