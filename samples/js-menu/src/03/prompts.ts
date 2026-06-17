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
import { ai } from '../genkit.js';
import { DataMenuQuestionInputSchema } from '../types';

// This prompt will generate two messages when rendered.
// These two messages will be used to seed the exchange with the model.

export const s03_chatPreamblePrompt = ai.definePrompt(
  {
    name: 's03_chatPreamble',
    model: googleAI.model('gemini-flash-latest'),
    input: { schema: DataMenuQuestionInputSchema },
    output: { format: 'text' },
    config: { temperature: 0.3 },
  },
  `
  {{ role "user" }}
  Hi. What's on the menu today?

  {{ role "model" }}
  I am Walt, a helpful AI assistant here at the restaurant.
  I can answer questions about the food on the menu or any other questions
  you have about food in general. I probably can't help you with anything else.
  Here is today's menu:
  {{#each menuData~}}
  - {{this.title}} \${{this.price}}
    {{this.description}}
  {{~/each}}
  Do you have any questions about the menu?
`
);
