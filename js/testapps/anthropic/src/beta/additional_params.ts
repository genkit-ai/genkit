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
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [
    // Default all flows in this sample to the beta surface
    anthropic({
      apiVersion: 'beta',
      apiKey: process.env.ANTHROPIC_API_KEY,
    }),
  ],
});

const betaOpus45 = anthropic.model('claude-opus-4-5', { apiVersion: 'beta' });

ai.defineFlow('anthropic-beta-additional-params', async () => {
  const { text } = await ai.generate({
    model: betaOpus45,
    prompt:
      'You are Claude on the beta API. Provide a concise greeting that mentions that you are using the beta API.',
    config: {
      temperature: 0.6,
      // Additional param (not directly supported by the plugin, but can be passed through to the API)
      betas: ['effort-2025-11-24'],
      // Additional param (not directly supported by the plugin, but can be passed through to the API)
      output_config: {
        effort: 'medium',
      },
    },
  });

  return text;
});

ai.defineFlow(
  'anthropic-beta-additional-params-stream',
  async (_, { sendChunk }) => {
    const { stream } = ai.generateStream({
      model: betaOpus45,
      prompt: [
        {
          text: 'Outline two experimental capabilities unlocked by the Anthropic beta API.',
        },
      ],
      config: {
        temperature: 0.4,
        // Additional param (not directly supported by the plugin, but can be passed through to the API)
        betas: ['effort-2025-11-24'],
        // Additional param (not directly supported by the plugin, but can be passed through to the API)
        output_config: {
          effort: 'medium',
        },
      },
    });

    const collected: string[] = [];
    for await (const chunk of stream) {
      if (chunk.text) {
        collected.push(chunk.text);
        sendChunk(chunk.text);
      }
    }

    return collected.join('');
  }
);
