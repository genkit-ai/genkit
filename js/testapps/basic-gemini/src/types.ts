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

const baseCategorySchema = z.object({
  name: z.string(),
});

type Category = z.infer<typeof baseCategorySchema> & {
  subcategories?: Category[];
};

const categorySchema: z.ZodType<Category> = baseCategorySchema.extend({
  subcategories: z.lazy(() =>
    categorySchema
      .array()
      .describe('make sure there are at least 2-3 levels of subcategories')
      .optional()
  ),
});

export const WeaponSchema = z.object({
  name: z.string(),
  damage: z.number(),
  category: categorySchema,
});

export const RpgCharacterSchema = z.object({
  name: z.string().describe('name of the character'),
  backstory: z.string().describe("character's backstory, about a paragraph"),
  weapons: z.array(WeaponSchema),
  class: z.enum(['RANGER', 'WIZZARD', 'TANK', 'HEALER', 'ENGINEER']),
  affiliation: z.string().optional(),
});
