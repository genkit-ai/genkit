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
// IMPORTANT: Keep this file in sync with js/ai/src/retriever.ts!
//
import { z } from 'zod';
import { DocumentDataSchema } from './document.js';

/**
 * Zod schema for retriever request.
 */
export const RetrieverRequestSchema = z.object({
  query: DocumentDataSchema,
  options: z.any().optional(),
});
export type RetrieverRequest = z.infer<typeof RetrieverRequestSchema>;

/**
 * Zod schema for retriever response.
 */
export const RetrieverResponseSchema = z.object({
  documents: z.array(DocumentDataSchema),
});
export type RetrieverResponse = z.infer<typeof RetrieverResponseSchema>;

/**
 * Zod schema of common retriever options.
 */
export const CommonRetrieverOptionsSchema = z.object({
  k: z.number().describe('Number of documents to retrieve').optional(),
});
