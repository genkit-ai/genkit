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
import { MenuQuestionInputSchema, type MenuQuestionInput } from '../types.js';

// Define a prompt to handle a customer question about the menu.
// This prompt uses definePrompt directly.

export const s01_vanillaPrompt = ai.definePrompt({
  name: 's01_vanillaPrompt',
  input: {
    schema: MenuQuestionInputSchema,
  },
  config: { temperature: 0.3 },
  messages: async (input: MenuQuestionInput) => {
    const promptText = `
      You are acting as a helpful AI assistant named "Walt" that can answer
      questions about the food available on the menu at Walt's Burgers.
      Customer says: ${input.question}
      `;

    return [{ role: 'user', content: [{ text: promptText }] }];
  },
});

// Define another prompt which uses the Dotprompt library
// that also gives us a type-safe handlebars template system,
// and well-defined output schemas.

export const s01_staticMenuDotPrompt = ai.definePrompt({
  name: 's01_staticMenuDotPrompt',
  model: googleAI.model('gemini-flash-latest'),
  input: { schema: MenuQuestionInputSchema },
  output: { format: 'text' },
  messages: `
You are acting as a helpful AI assistant named "Walt" that can answer
questions about the food available on the menu at Walt's Burgers.
Here is today's menu:

- The Regular Burger $12
  The classic charbroiled to perfection with your choice of cheese

- The Fancy Burger $13
  Classic burger topped with bacon & Blue Cheese

- The Bacon Burger $13
  Bacon cheeseburger with your choice of cheese.

- Everything Burger $14
  Heinz 57 sauce, American cheese, bacon, fried egg & crispy onion bits

- Chicken Breast Sandwich $12
  Tender juicy chicken breast on a brioche roll.
  Grilled, blackened, or fried

Our fresh 1/2 lb. beef patties are made using choice cut
brisket, short rib & sirloin. Served on a toasted
brioche roll with chips. Served with lettuce, tomato & pickles.
Onions upon request. Substitute veggie patty $2

Answer this customer's question, in a concise and helpful manner,
as long as it is about food.

Question:
{{question}} ?
`,
});
