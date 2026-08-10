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
import { ai } from '../genkit.js';
import { TextMenuQuestionInputSchema } from '../types';

export const s05_readMenuPrompt = ai.definePrompt(
  {
    name: 's05_readMenu',
    model: googleAI.model('gemini-flash-latest'),
    input: {
      schema: z.object({
        imageUrl: z.string(),
      }),
    },
    output: { format: 'text' },
    config: { temperature: 0.1 },
  },
  `
Extract _all_ of the text, in order,
from the following image of a restaurant menu.

{{media url=imageUrl}}
`
);

export const s05_textMenuPrompt = ai.definePrompt(
  {
    name: 's05_textMenu',
    model: googleAI.model('gemini-flash-latest'),
    input: { schema: TextMenuQuestionInputSchema },
    output: { format: 'text' },
    config: { temperature: 0.3 },
  },
  `
You are acting as Walt, a helpful AI assistant here at the restaurant.
You can answer questions about the food on the menu or any other questions
customers have about food in general.

Here is the text of today's menu to help you answer the customer's question:
{{menuText}}

Answer this customer's question:
{{question}}?
`
);
