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

/**
 * This file defines schema and types that are used by Genkit evaluators.
 *
 * NOTE: Keep this file in sync with genkit/ai/src/evaluator.ts!
 * Eventually tools will be source of truth for these types (by generating a
 * JSON schema) but until then this file must be manually kept in sync
 */
import { z } from 'zod';
import { DocumentDataSchema, DocumentPartSchema } from './document.js';

//
// IMPORTANT: Keep this file in sync with genkit/ai/src/reranker.ts!
//

/**
 * Zod schema for a reranked document metadata.
 */
export const RankedDocumentMetadataSchema = z
  .object({
    score: z.number(), // Enforces that 'score' must be a number
  })
  .passthrough(); // Allows other properties in 'metadata' with any type

/**
 * Zod schema for a reranked document.
 */
export const RankedDocumentDataSchema = z.object({
  content: z.array(DocumentPartSchema),
  metadata: RankedDocumentMetadataSchema,
});
export type RankedDocumentData = z.infer<typeof RankedDocumentDataSchema>;

/**
 * Zod schema for a reranker request.
 */
export const RerankerRequestSchema = z.object({
  query: DocumentDataSchema,
  documents: z.array(DocumentDataSchema),
  options: z.any().optional(),
});

/**
 * Zod schema for a reranker response.
 */
export const RerankerResponseSchema = z.object({
  documents: z.array(RankedDocumentDataSchema),
});
export type RerankerResponse = z.infer<typeof RerankerResponseSchema>;

/**
 * Zod schema for common reranker options.
 */
export const CommonRerankerOptionsSchema = z.object({
  k: z.number().describe('Number of documents to rerank').optional(),
});
