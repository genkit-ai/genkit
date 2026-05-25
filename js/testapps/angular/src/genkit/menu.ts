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

import { z } from 'genkit';
import { ai } from './index';

const MenuItemSchema = z.object({
  category: z.enum(['Appetizer', 'Main Course', 'Dessert', 'Drink']),
  name: z.string().describe('Creative dish name fitting the theme'),
  description: z
    .string()
    .describe('A short, appetizing description (1-2 sentences)'),
  price: z.string().describe('Realistic price as a dollar amount, e.g. $12.99'),
});

const MenuSchema = z.object({
  restaurantName: z.string().describe('A creative themed restaurant name'),
  items: z.array(MenuItemSchema),
});

export type MenuItem = z.infer<typeof MenuItemSchema>;
export type Menu = z.infer<typeof MenuSchema>;

const categories = ['Appetizer', 'Main Course', 'Dessert', 'Drink'] as const;

export const suggestMenu = ai.defineFlow(
  {
    name: 'suggestMenu',
    inputSchema: z.string().nullable(),
    outputSchema: MenuSchema,
    streamSchema: MenuItemSchema,
  },
  async (theme: string | null, { sendChunk }) => {
    const resolvedTheme = theme || 'pirate';

    // First, generate a restaurant name
    const nameResult = await ai.generate({
      prompt: `Invent a short, creative restaurant name for a ${resolvedTheme}-themed restaurant. Return just the name, nothing else.`,
    });
    const restaurantName = nameResult.text.trim();

    // Generate one dish per category, streaming each as it completes
    const items: MenuItem[] = [];
    for (const category of categories) {
      const result = await ai.generate({
        prompt: `You are designing the menu for "${restaurantName}", a ${resolvedTheme}-themed restaurant.
Invent one creative ${category.toLowerCase()} that fits the theme perfectly.`,
        output: { schema: MenuItemSchema },
        config: {
          // Lock the category so the model can't override it
          stopSequences: [],
        },
      });
      const output = result.output;
      if (!output) {
        throw new Error(
          `Model failed to generate a valid ${category} menu item.`
        );
      }
      const item = { ...output, category };
      items.push(item);
      sendChunk(item);
    }

    return { restaurantName, items };
  }
);
