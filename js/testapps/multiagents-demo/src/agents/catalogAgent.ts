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

import { ai } from '../config/genkit';

import {
  fetchCatalogTool,
  fetchMostPopularItemsTool,
  getProductDetailsTool,
  searchCatalogTool,
} from '../tools';

// Simple agent for browsing and searching products
export const agent = ai.definePrompt({
  name: 'catalogAgent',
  description: 'Catalog Agent can help customers browse and search products',
  model: 'googleai/gemini-2.5-pro',
  tools: [
    fetchCatalogTool,
    fetchMostPopularItemsTool,
    searchCatalogTool,
    getProductDetailsTool,
  ],
  system: `You are a helpful catalog agent for TechStore Computer Shop.
      Help customers browse products, search for items, and get product details.
      Be friendly and provide accurate information about products, prices, and availability.`,
});
