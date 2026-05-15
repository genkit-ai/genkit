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
    // Configure the plugin with environment-driven API key
    anthropic(),
  ],
});

ai.defineFlow('anthropic-stable-hello', async () => {
  const { text } = await ai.generate({
    model: anthropic.model('claude-sonnet-4-5'),
    prompt: 'You are a friendly Claude assistant. Greet the user briefly.',
  });

  return text;
});

ai.defineFlow('anthropic-stable-stream', async (_, { sendChunk }) => {
  const { stream } = ai.generateStream({
    model: anthropic.model('claude-sonnet-4-5'),
    prompt: 'Compose a short limerick about using Genkit with Anthropic.',
  });

  let response = '';
  for await (const chunk of stream) {
    response += chunk.text ?? '';
    if (chunk.text) {
      sendChunk(chunk.text);
    }
  }

  return response;
});
