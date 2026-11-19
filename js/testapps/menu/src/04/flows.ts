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
import { Document, z } from 'genkit';
import { ai } from '../genkit.js';
import {
  AnswerOutputSchema,
  MenuItemSchema,
  MenuQuestionInputSchema,
} from '../types.js';
import { vectraStore } from '../vector-store.js';
import { s04_ragDataMenuPrompt } from './prompts.js';

// A local vectra index (.vectra/menu-items, see ../vector-store.ts).
const menuStore = vectraStore({
  ai,
  indexName: 'menu-items',
  embedder: googleAI.embedder('gemini-embedding-001'),
  embedderOptions: { taskType: 'RETRIEVAL_DOCUMENT' },
});

// Define a flow which indexes items on the menu.

export const s04_indexMenuItemsFlow = ai.defineFlow(
  {
    name: 's04_indexMenuItems',
    inputSchema: z.array(MenuItemSchema),
    outputSchema: z.object({ rows: z.number() }),
  },
  async (menuItems) => {
    // Store each document with its text indexed,
    // and its original JSON data as its metadata.
    const documents = menuItems.map((menuItem) => {
      const text = `${menuItem.title} ${menuItem.price} \n ${menuItem.description}`;
      return Document.fromText(text, menuItem);
    });
    await menuStore.index(documents);
    return { rows: menuItems.length };
  }
);

// Define a flow which generates a response to the question,
// by retrieving relevant items from the menu.
// View this flow's trace to see the context that was retrieved,
// and how it was included in the prompt.

export const s04_ragMenuQuestionFlow = ai.defineFlow(
  {
    name: 's04_ragMenuQuestion',
    inputSchema: MenuQuestionInputSchema,
    outputSchema: AnswerOutputSchema,
  },
  async (input) => {
    // Retrieve the 3 most relevant menu items for the question
    const docs = await ai.run('retrieve', () =>
      menuStore.query(input.question, { k: 3 })
    );
    // Each document's metadata is the original menu item (see indexing above).
    const menuData = docs.map((doc) => MenuItemSchema.parse(doc.metadata));

    // Generate the response
    const response = await s04_ragDataMenuPrompt({
      menuData: menuData,
      question: input.question,
    });
    return { answer: response.text };
  }
);
