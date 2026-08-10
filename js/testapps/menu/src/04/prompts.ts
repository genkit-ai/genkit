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
import { DataMenuQuestionInputSchema } from '../types.js';

export const s04_ragDataMenuPrompt = ai.definePrompt({
  name: 's04_ragDataMenu',
  model: googleAI.model('gemini-flash-latest'),
  input: { schema: DataMenuQuestionInputSchema },
  output: { format: 'text' },
  config: { temperature: 0.3 },
  messages: `
You are acting as Walt, a helpful AI assistant here at the restaurant.
You can answer questions about the food on the menu or any other questions
customers have about food in general.

Here are some items that are on today's menu that are relevant to
helping you answer the customer's question:
{{#each menuData~}}
- {{this.title}} \${{this.price}}
  {{this.description}}
{{~/each}}

Answer this customer's question:
{{question}}?
`,
});
