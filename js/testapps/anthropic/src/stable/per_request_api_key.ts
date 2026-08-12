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

/**
 * Demonstrates per-request API key support.
 *
 * This allows deferring the API key to request time instead of plugin initialization,
 * useful for multi-tenant applications where different users have different API keys.
 */

import { anthropic } from '@genkit-ai/anthropic';
import { genkit } from 'genkit';

// Initialize plugin with apiKey: false to defer API key to request time
const ai = genkit({
  plugins: [anthropic({ apiKey: false })],
});

// Flow that accepts an API key as input and uses it for the request
ai.defineFlow('anthropic-per-request-api-key', async (apiKey: string) => {
  const { text } = await ai.generate({
    model: anthropic.model('claude-haiku-4-5'),
    prompt: 'Say "hello" and nothing else.',
    config: {
      apiKey, // Pass API key per-request
    },
  });

  return text;
});

// Flow that demonstrates API key override (plugin has a key but request overrides it)
const aiWithKey = genkit({
  plugins: [anthropic()], // Uses ANTHROPIC_API_KEY env var
});

aiWithKey.defineFlow(
  'anthropic-api-key-override',
  async (overrideApiKey: string) => {
    const { text } = await aiWithKey.generate({
      model: anthropic.model('claude-haiku-4-5'),
      prompt: 'Say "world" and nothing else.',
      config: {
        apiKey: overrideApiKey, // Override the plugin-level API key
      },
    });

    return text;
  }
);
