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

import {
  devLocalIndexerRef,
  devLocalRetrieverRef,
} from '@genkit-ai/dev-local-vectorstore';
import { z } from 'genkit';
import { Document } from 'genkit/retriever';
import { ai } from '../genkit.js';
import {
  AnswerOutputSchema,
  MenuItemSchema,
  MenuQuestionInputSchema,
  type MenuItem,
} from '../types';
import { s04_ragDataMenuPrompt } from './prompts';

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
    await ai.index({
      indexer: devLocalIndexerRef('menu-items'),
      documents,
    });
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
    const docs = await ai.retrieve({
      retriever: devLocalRetrieverRef('menu-items'),
      query: input.question,
      options: { k: 3 },
    });
    const menuData: Array<MenuItem> = docs.map(
      (doc) => (doc.metadata || {}) as MenuItem
    );

    // Generate the response
    const response = await s04_ragDataMenuPrompt({
      menuData: menuData,
      question: input.question,
    });
    return { answer: response.text };
  }
);
