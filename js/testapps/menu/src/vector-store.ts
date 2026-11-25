/**
 * Copyright 2026 Google LLC
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

// Glue between Genkit and vectra, a local file-based vector database. Any
// vector database plugs in the same way: embed with `ai.embed`, store the
// vector next to the serialized Document, and map query results back to
// Documents. Fine for local development, not for production.

import {
  Document,
  type DocumentData,
  type EmbedderArgument,
  type Genkit,
  type GenkitSchema,
} from 'genkit';
import type { InferOutput } from 'genkit/schema';
import { createHash } from 'node:crypto';
import * as path from 'node:path';
import { LocalIndex } from 'vectra';

// vectra only allows primitive metadata values, so the Document is stored as
// JSON.
type ItemMetadata = { doc: string };

export interface VectorStoreOptions<C extends GenkitSchema = GenkitSchema> {
  ai: Genkit;
  /** Index folder, created under `.vectra/` in the working directory. */
  indexName: string;
  embedder: EmbedderArgument<C>;
  embedderOptions?: InferOutput<C>;
}

export interface VectorStore {
  index(docs: Document[]): Promise<void>;
  query(
    query: string | DocumentData,
    options?: { k?: number }
  ): Promise<Document[]>;
}

export function vectraStore<C extends GenkitSchema = GenkitSchema>({
  ai,
  indexName,
  embedder,
  embedderOptions,
}: VectorStoreOptions<C>): VectorStore {
  const index = new LocalIndex<ItemMetadata>(
    path.resolve('.vectra', indexName)
  );

  return {
    async index(docs) {
      if (!(await index.isIndexCreated())) {
        await index.createIndex();
      }
      const items = await Promise.all(
        docs.map(async (doc) => {
          const embeddings = await ai.embed({
            embedder,
            content: doc,
            options: embedderOptions,
          });
          // Some embedders (e.g. multimodal video) return several embeddings
          // per document; each becomes its own item.
          return doc.getEmbeddingDocuments(embeddings).map((d, i) => {
            const json = JSON.stringify(d.toJSON());
            return {
              // Content-derived id, so re-indexing the same doc is a no-op.
              id: createHash('sha256').update(json).digest('hex'),
              vector: embeddings[i].embedding,
              metadata: { doc: json },
            };
          });
        })
      );
      await index.beginUpdate();
      for (const item of items.flat()) {
        await index.upsertItem(item);
      }
      await index.endUpdate();
    },

    async query(query, options) {
      if (!(await index.isIndexCreated())) {
        return [];
      }
      const [{ embedding }] = await ai.embed({
        embedder,
        content: query,
        options: embedderOptions,
      });
      const text = typeof query === 'string' ? query : '';
      const results = await index.queryItems(embedding, text, options?.k ?? 3);
      return results.map(
        (r) => new Document(JSON.parse(r.item.metadata.doc) as DocumentData)
      );
    },
  };
}
