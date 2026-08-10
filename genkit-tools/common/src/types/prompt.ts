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

//
// IMPORTANT: Please keep type definitions in sync with
//   genkit/js/dotprompt/src/metadata.ts
//

import { z } from 'zod';
import { MiddlewareRefSchema } from './middleware';
import { GenerationCommonConfigSchema } from './model';

/**
 * Formal schema for prompt YAML frontmatter.
 */
export const PromptFrontmatterSchema = z.object({
  name: z.string().optional(),
  variant: z.string().optional(),
  model: z.string().optional(),
  tools: z.array(z.string()).optional(),
  use: z.array(MiddlewareRefSchema).optional(),
  candidates: z.number().optional(),
  config: GenerationCommonConfigSchema.passthrough().optional(),
  input: z
    .object({
      schema: z.unknown(),
      default: z.any(),
    })
    .optional(),
  output: z
    .object({
      format: z.enum(['json', 'text', 'media']).optional(),
      schema: z.unknown().optional(),
    })
    .optional(),
  metadata: z.record(z.unknown()).optional(),
});

export type PromptFrontmatter = z.infer<typeof PromptFrontmatterSchema>;
