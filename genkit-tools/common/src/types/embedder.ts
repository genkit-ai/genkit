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

import { z } from 'zod';
import { DocumentDataSchema } from './document.js';

//
// IMPORTANT: Keep this file in sync with genkit/ai/src/embedder.ts!
//

/**
 * A batch (array) of embeddings.
 */
export type EmbeddingBatch = { embedding: number[] }[];

/**
 * EmbeddingSchema includes the embedding and also metadata so you know
 * which of multiple embeddings corresponds to which part of a document.
 */
export const EmbeddingSchema = z.object({
  embedding: z.array(z.number()),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type Embedding = z.infer<typeof EmbeddingSchema>;

/**
 * Zod schema of an embed request.
 */
export const EmbedRequestSchema = z.object({
  input: z.array(DocumentDataSchema),
  options: z.any().optional(),
});

/**
 * Zod schema of an embed response.
 */
export const EmbedResponseSchema = z.object({
  embeddings: z.array(EmbeddingSchema),
});
export type EmbedResponse = z.infer<typeof EmbedResponseSchema>;
