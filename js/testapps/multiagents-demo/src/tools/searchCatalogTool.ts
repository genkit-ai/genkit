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
import { ai } from '../config/genkit';
import { mockCatalog } from '../data/catalog';

export const searchCatalogTool = ai.defineTool(
  {
    name: 'searchCatalog',
    description: 'Search for products by name, category, or keyword',
    inputSchema: z.object({
      query: z
        .string()
        .describe('Search query (product name, category, or keyword)'),
    }),
  },
  async (input) => {
    const query = input.query.toLowerCase();
    const results = mockCatalog.filter(
      (item) =>
        item.name.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query)
    );
    return {
      success: true,
      results,
      count: results.length,
    };
  }
);
