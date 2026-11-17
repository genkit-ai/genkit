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

import similarity from 'compute-cosine-similarity';
import * as fs from 'fs';
import { z, type Embedding, type Genkit, type GenkitSchema } from 'genkit';
import type { EmbedderArgument } from 'genkit/embedder';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
  type DocumentData,
} from 'genkit/retriever';
import type { InferOutput } from 'genkit/schema';
import { Md5 } from 'ts-md5';

const _LOCAL_FILESTORE = '__db_{INDEX_NAME}.json';

interface DbValue {
  doc: DocumentData;
  embedding: Embedding;
}

function loadFilestore(indexName: string) {
  let existingData = {};
  const indexFileName = _LOCAL_FILESTORE.replace('{INDEX_NAME}', indexName);
  if (fs.existsSync(indexFileName)) {
    existingData = JSON.parse(fs.readFileSync(indexFileName).toString());
  }
  return existingData;
}

function addDocument(
  embedding: Embedding,
  doc: Document,
  contents: Record<string, DbValue>
) {
  const id = Md5.hashStr(JSON.stringify(doc));
  if (!(id in contents)) {
    contents[id] = { doc, embedding };
  } else {
    console.debug(`Skipping ${id} since it is already present`);
  }
}

interface Params<EmbedderCustomOptions extends GenkitSchema> {
  indexName: string;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: InferOutput<EmbedderCustomOptions>;
}

/**
 * Local file-based vectorstore plugin that provides retriever and indexer.
 *
 * NOT INTENDED FOR USE IN PRODUCTION
 */
export function devLocalVectorstore<EmbedderCustomOptions extends GenkitSchema>(
  params: Params<EmbedderCustomOptions>[]
): GenkitPlugin {
  return genkitPlugin('devLocalVectorstore', async (ai) => {
    params.map((p) => configureDevLocalRetriever(ai, p));
    params.map((p) => configureDevLocalIndexer(ai, p));
  });
}

export default devLocalVectorstore;

/**
 * Local file-based vectorstore retriever reference
 */
export function devLocalRetrieverRef(indexName: string) {
  return retrieverRef({
    name: `devLocalVectorstore/${indexName}`,
    info: {
      label: `Local file-based Retriever - ${indexName}`,
    },
    configSchema: CommonRetrieverOptionsSchema.optional(),
  });
}

/**
 * Local file-based indexer reference
 */
export function devLocalIndexerRef(indexName: string) {
  return indexerRef({
    name: `devLocalVectorstore/${indexName}`,
    info: {
      label: `Local file-based Indexer - ${indexName}`,
    },
    configSchema: z.null().optional(),
  });
}

async function importDocumentsToLocalVectorstore<
  EmbedderCustomOptions extends GenkitSchema,
>(
  ai: Genkit,
  params: {
    indexName: string;
    docs: Array<Document>;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: InferOutput<EmbedderCustomOptions>;
  }
) {
  const { docs, embedder, embedderOptions } = { ...params };
  const data = loadFilestore(params.indexName);

  await Promise.all(
    docs.map(async (doc) => {
      const embeddings = await ai.embed({
        embedder,
        content: doc,
        options: embedderOptions,
      });
      const embeddingDocs = doc.getEmbeddingDocuments(embeddings);
      for (const i in embeddingDocs) {
        addDocument(embeddings[i], embeddingDocs[i], data);
      }
    })
  );

  // Update the file
  fs.writeFileSync(
    _LOCAL_FILESTORE.replace('{INDEX_NAME}', params.indexName),
    JSON.stringify(data, null, 2)
  );
}

async function getClosestDocuments<
  I extends GenkitSchema,
  EmbedderCustomOptions extends GenkitSchema,
>(params: {
  queryEmbeddings: Array<number>;
  db: Record<string, DbValue>;
  k: number;
}): Promise<Document[]> {
  const scoredDocs: { score: number; doc: Document }[] = [];
  // Very dumb way to check for similar docs.
  for (const value of Object.values(params.db)) {
    const thisEmbedding = value.embedding.embedding;
    const score = similarity(params.queryEmbeddings, thisEmbedding) ?? 0;
    scoredDocs.push({
      score,
      doc: new Document(value.doc),
    });
  }

  scoredDocs.sort((a, b) => (a.score > b.score ? -1 : 1));
  return scoredDocs.slice(0, params.k).map((o) => o.doc);
}

/**
 * Configures a local vectorstore retriever
 */
function configureDevLocalRetriever<EmbedderCustomOptions extends GenkitSchema>(
  ai: Genkit,
  params: {
    indexName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: InferOutput<EmbedderCustomOptions>;
  }
) {
  const { embedder, embedderOptions } = params;
  const vectorstore = ai.defineRetriever(
    {
      name: `devLocalVectorstore/${params.indexName}`,
      configSchema: CommonRetrieverOptionsSchema,
    },
    async (content, options) => {
      const db = loadFilestore(params.indexName);
      const embeddings = await ai.embed({
        embedder,
        content,
        options: embedderOptions,
      });
      return {
        documents: await getClosestDocuments({
          k: options?.k ?? 3,
          queryEmbeddings: embeddings[0].embedding,
          db,
        }),
      };
    }
  );
  return vectorstore;
}

/**
 * Configures a local vectorstore indexer.
 */
function configureDevLocalIndexer<EmbedderCustomOptions extends GenkitSchema>(
  ai: Genkit,
  params: {
    indexName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: InferOutput<EmbedderCustomOptions>;
  }
) {
  const { embedder, embedderOptions } = params;
  const vectorstore = ai.defineIndexer(
    { name: `devLocalVectorstore/${params.indexName}` },
    async (docs) => {
      await importDocumentsToLocalVectorstore(ai, {
        indexName: params.indexName,
        docs,
        embedder,
        embedderOptions: embedderOptions,
      });
    }
  );
  return vectorstore;
}
