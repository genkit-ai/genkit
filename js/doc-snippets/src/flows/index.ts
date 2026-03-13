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
import { genkit } from 'genkit/beta';

const ai = genkit({
  plugins: [googleAI()],
});

// [START ex01]
export const menuSuggestionFlow = ai.defineFlow(
  {
    name: 'menuSuggestionFlow',
  },
  async (restaurantTheme) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `Invent a menu item for a ${restaurantTheme} themed restaurant.`,
    });
    return text;
  }
);
// [END ex01]

// [START ex02]
const MenuItemSchema = z.object({
  dishname: z.string(),
  description: z.string(),
});

export const menuSuggestionFlowWithSchema = ai.defineFlow(
  {
    name: 'menuSuggestionFlow',
    inputSchema: z.string(),
    outputSchema: MenuItemSchema,
  },
  async (restaurantTheme) => {
    const { output } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `Invent a menu item for a ${restaurantTheme} themed restaurant.`,
      output: { schema: MenuItemSchema },
    });
    if (output == null) {
      throw new Error("Response doesn't satisfy schema.");
    }
    return output;
  }
);
// [END ex02]

// [START ex03]
export const menuSuggestionFlowMarkdown = ai.defineFlow(
  {
    name: 'menuSuggestionFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (restaurantTheme) => {
    const { output } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `Invent a menu item for a ${restaurantTheme} themed restaurant.`,
      output: { schema: MenuItemSchema },
    });
    if (output == null) {
      throw new Error("Response doesn't satisfy schema.");
    }
    return `**${output.dishname}**: ${output.description}`;
  }
);
// [END ex03]

// [START ex06]
export const menuSuggestionStreamingFlow = ai.defineFlow(
  {
    name: 'menuSuggestionFlow',
    inputSchema: z.string(),
    streamSchema: z.string(),
    outputSchema: z.object({ theme: z.string(), menuItem: z.string() }),
  },
  async (restaurantTheme, { sendChunk }) => {
    const response = await ai.generateStream({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `Invent a menu item for a ${restaurantTheme} themed restaurant.`,
    });

    for await (const chunk of response.stream) {
      // Here, you could process the chunk in some way before sending it to
      // the output stream via streamingCallback(). In this example, we output
      // the text of the chunk, unmodified.
      sendChunk(chunk.text);
    }

    return {
      theme: restaurantTheme,
      menuItem: (await response.response).text,
    };
  }
);
// [END ex06]

// [START ex10]
const PrixFixeMenuSchema = z.object({
  starter: z.string(),
  soup: z.string(),
  main: z.string(),
  dessert: z.string(),
});

export const complexMenuSuggestionFlow = ai.defineFlow(
  {
    name: 'complexMenuSuggestionFlow',
    inputSchema: z.string(),
    outputSchema: PrixFixeMenuSchema,
  },
  async (theme: string): Promise<z.infer<typeof PrixFixeMenuSchema>> => {
    const chat = ai.chat({ model: googleAI.model('gemini-2.5-flash') });
    await chat.send('What makes a good prix fixe menu?');
    await chat.send(
      'What are some ingredients, seasonings, and cooking techniques that ' +
        `would work for a ${theme} themed menu?`
    );
    const { output } = await chat.send({
      prompt:
        `Based on our discussion, invent a prix fixe menu for a ${theme} ` +
        'themed restaurant.',
      output: {
        schema: PrixFixeMenuSchema,
      },
    });
    if (!output) {
      throw new Error('No data generated.');
    }
    return output;
  }
);
// [END ex10]

// [START ex11]
export const menuQuestionFlow = ai.defineFlow(
  {
    name: 'menuQuestionFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (input: string): Promise<string> => {
    const menu = await ai.run(
      'retrieve-daily-menu',
      async (): Promise<string> => {
        // Retrieve today's menu. (This could be a database access or simply
        // fetching the menu from your website.)

        // [START_EXCLUDE]
        const menu = `
Today's menu

- Breakfast: spam and eggs
- Lunch: spam sandwich with a cup of spam soup
- Dinner: spam roast with a side of spammed potatoes
      `;
        // [END_EXCLUDE]

        return menu;
      }
    );
    const { text } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      system: "Help the user answer questions about today's menu.",
      prompt: input,
      docs: [{ content: [{ text: menu }] }],
    });
    return text;
  }
);
// [END ex11]

async function fn() {
  // [START ex04]
  const { text } = await menuSuggestionFlow('bistro');
  // [END ex04]

  // [START ex05]
  const { dishname, description } =
    await menuSuggestionFlowWithSchema('bistro');
  // [END ex05]

  // [START ex07]
  const response = menuSuggestionStreamingFlow.stream('Danube');
  // [END ex07]
  // [START ex08]
  for await (const chunk of response.stream) {
    console.log('chunk', chunk);
  }
  // [END ex08]
  // [START ex09]
  const output = await response.output;
  // [END ex09]
}
