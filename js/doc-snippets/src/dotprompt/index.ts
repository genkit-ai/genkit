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

import { genkit } from 'genkit';

// [START promptDir]
const ai = genkit({
  promptDir: './llm_prompts',
  // (Other settings...)
});
// [END promptDir]

// [START MenuItemSchema]
import { z } from 'genkit';

const MenuItemSchema = ai.defineSchema(
  'MenuItemSchema',
  z.object({
    dishname: z.string(),
    description: z.string(),
    calories: z.coerce.number(),
    allergens: z.array(z.string()),
  })
);
// [END MenuItemSchema]

async function fn02() {
  // [START loadPrompt]
  const helloPrompt = ai.prompt('hello');
  // [END loadPrompt]

  // [START loadPromptVariant]
  const myPrompt = ai.prompt('my_prompt', { variant: 'gemini15pro' });
  // [END loadPromptVariant]

  // [START callPrompt]
  const response = await helloPrompt();

  // Alternatively, use destructuring assignments to get only the properties
  // you're interested in:
  const { text } = await helloPrompt();
  // [END callPrompt]

  // [START callPromptOpts]
  const response2 = await helloPrompt(
    // Prompt input:
    { name: 'Ted' },

    // Generation options:
    {
      config: {
        temperature: 0.4,
      },
    }
  );
  // [END callPromptOpts]
}

async function fn03() {
  const helloPrompt = ai.prompt('hello');

  // [START callPromptCfg]
  const response3 = await helloPrompt(
    {},
    {
      config: {
        temperature: 1.4,
        topK: 50,
        topP: 0.4,
        maxOutputTokens: 400,
        stopSequences: ['<end>', '<fin>'],
      },
    }
  );
  // [END callPromptCfg]
}

async function fn04() {
  // [START outSchema]
  // [START inSchema]
  const menuPrompt = ai.prompt('menu');
  const { output } = await menuPrompt({ theme: 'medieval' });
  // [END inSchema]

  const dishName = output['dishname'];
  const description = output['description'];
  // [END outSchema]
}

async function fn05() {
  // [START outSchema2]
  const menuPrompt = ai.prompt<
    z.ZodTypeAny, // Input schema
    typeof MenuItemSchema, // Output schema
    z.ZodTypeAny // Custom options schema
  >('menu');
  const { output } = await menuPrompt({ theme: 'medieval' });

  // Now data is strongly typed as MenuItemSchema:
  const dishName = output?.dishname;
  const description = output?.description;
  // [END outSchema2]
}

async function fn06() {
  // [START multiTurnPrompt]
  const multiTurnPrompt = ai.prompt('multiTurnPrompt');
  const result = await multiTurnPrompt({
    messages: [
      { role: 'user', content: [{ text: 'Hello.' }] },
      { role: 'model', content: [{ text: 'Hi there!' }] },
    ],
  });
  // [END multiTurnPrompt]
}

async function fn07() {
  // [START multiModalPrompt]
  const multimodalPrompt = ai.prompt('multimodal');
  const { text } = await multimodalPrompt({
    photoUrl: 'https://example.com/photo.jpg',
  });
  // [END multiModalPrompt]
}

async function fn08() {
  // [START definePartial]
  ai.definePartial(
    'personality',
    'Talk like a {{#if style}}{{style}}{{else}}helpful assistant{{/if}}.'
  );
  // [END definePartial]

  // [START defineHelper]
  ai.defineHelper('shout', (text: string) => text.toUpperCase());
  // [END defineHelper]
}

function fn09() {
  // [START definePromptTempl]
  const myPrompt = ai.definePrompt({
    name: 'myPrompt',
    model: 'googleai/gemini-flash-latest',
    input: {
      schema: z.object({
        name: z.string(),
      }),
    },
    prompt: 'Hello, {{name}}. How are you today?',
  });
  // [END definePromptTempl]
}

function fn10() {
  // [START definePromptFn]
  const myPrompt = ai.definePrompt({
    name: 'myPrompt',
    model: 'googleai/gemini-flash-latest',
    input: {
      schema: z.object({
        name: z.string(),
      }),
    },
    messages: async (input) => {
      return [
        {
          role: 'user',
          content: [{ text: `Hello, ${input.name}. How are you today?` }],
        },
      ];
    },
  });
  // [END definePromptFn]
}
