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
 * Retriever and indexer types — `Document`, reference helpers, and schemas for
 * building RAG (retrieval-augmented generation) plugins.
 *
 * ```ts
 * import { Document, retrieverRef, indexerRef } from 'genkit/retriever';
 * ```
 *
 * @module retriever
 */

export {
  CommonRetrieverOptionsSchema,
  Document,
  DocumentDataSchema,
  IndexerInfoSchema,
  RetrieverInfoSchema,
  indexerRef,
  retrieverRef,
  type DocumentData,
  type IndexerAction,
  type IndexerArgument,
  type IndexerFn,
  type IndexerInfo,
  type IndexerParams,
  type IndexerReference,
  type MediaPart,
  type Part,
  type RetrieverAction,
  type RetrieverArgument,
  type RetrieverFn,
  type RetrieverInfo,
  type RetrieverParams,
  type RetrieverReference,
  type SimpleRetrieverOptions,
  type TextPart,
} from '@genkit-ai/ai/retriever';
