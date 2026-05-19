/**
 * @license
 *
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
 * Embedder types — reference helpers, schemas, and action types for building
 * embedding model plugins.
 *
 * ```ts
 * import { embedderRef, type EmbedderAction } from 'genkit/embedder';
 * ```
 *
 * @module embedder
 */

export {
  EmbedderInfoSchema,
  embedderRef,
  type EmbedRequest,
  type EmbedderAction,
  type EmbedderArgument,
  type EmbedderInfo,
  type EmbedderParams,
  type EmbedderReference,
  type Embedding,
  type EmbeddingBatch,
} from '@genkit-ai/ai/embedder';
