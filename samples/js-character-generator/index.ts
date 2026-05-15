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

import { googleAI } from '@genkit-ai/google-genai';
import { genkit, z } from 'genkit';

const ai = genkit({
  plugins: [googleAI()],
});

const prompt = ai.definePrompt({
  name: 'Character Prompt',
  model: googleAI.model('gemini-flash-latest'),
  input: {
    schema: z.object({
      inspiration: z.string(),
    }),
  },
  output: {
    format: 'json',
    schema: z.object({
      name: z.string(),
      strength: z.number(),
      intelligence: z.number(),
      description: z.string(),
    }),
  },
  prompt: `You're a expert DnD designer, create a new character.
    Base the character on {{inspiration}} but don't make it
    an exact match.`,
});

(async () => {
  console.log((await prompt({ inspiration: 'Yogi Berra' })).output);
})();
