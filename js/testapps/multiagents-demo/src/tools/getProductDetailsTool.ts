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

export const getProductDetailsTool = ai.defineTool(
  {
    name: 'getProductDetails',
    description: 'Get detailed information about a specific product by ID',
    inputSchema: z.object({
      productId: z
        .string()
        .describe('The ID of the product to get details for'),
    }),
  },
  async (input) => {
    const product = mockCatalog.find((item) => item.id === input.productId);
    if (!product) {
      return {
        success: false,
        message: 'Product not found',
      };
    }
    return {
      success: true,
      product,
    };
  }
);
