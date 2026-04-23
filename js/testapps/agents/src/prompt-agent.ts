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
import { ai } from './genkit.js';

export const translatorPrompt = ai.definePrompt({
  name: 'translatorPrompt',
  model: 'googleai/gemini-flash-latest',
  input: { schema: z.object({ language: z.string() }) },
  system:
    'You are a translator. Translate everything the user says to {{ language }}.',
});

export const translatorAgent = ai.defineSessionFlowFromPrompt({
  promptName: 'translatorPrompt',
  defaultInput: { language: 'French' },
});

export const testTranslatorAgent = ai.defineFlow(
  {
    name: 'testTranslatorAgent',
    inputSchema: z.string().default('Hello, how are you?'),
    outputSchema: z.any(),
  },
  async (text) => {
    const res = await translatorAgent.run(
      {
        messages: [{ role: 'user', content: [{ text }] }],
      },
      { init: {} }
    );
    return res.result;
  }
);
