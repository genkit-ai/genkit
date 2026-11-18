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

import {
  Pinecone,
  type CreateIndexOptions,
  type PineconeConfiguration,
  type RecordMetadata,
} from '@pinecone-database/pinecone';
import { z, type Genkit } from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';

import type { EmbedderArgument, Embedding } from 'genkit/embedder';
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
} from 'genkit/retriever';
import { Md5 } from 'ts-md5';

const SparseVectorSchema = z
  .object({
    indices: z.number().array(),
    values: z.number().array(),
  })
  .refine(
    (input) => {
      return input.indices.length === input.values.length;
    },
    {
      message: 'Indices and values must be of the same length',
    }
  );

const PineconeRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().max(1000),
  namespace: z.string().optional(),
  filter: z.record(z.string(), z.any()).optional(),
  // includeValues is always false
  // includeMetadata is always true
  sparseVector: SparseVectorSchema.optional(),
});

const PineconeIndexerOptionsSchema = z.object({
  namespace: z.string().optional(),
});

const CONTENT_KEY = '_content';
const CONTENT_TYPE = '_contentType';

/**
 * pineconeRetrieverRef function creates a retriever for Pinecone.
 * @param params The params for the new Pinecone retriever
 * @param params.indexId The indexId for the Pinecone retriever
 * @param params.displayName  A display name for the retriever.
If not specified, the default label will be `Pinecone - <indexId>`
 * @returns A reference to a Pinecone retriever.
 */
export const pineconeRetrieverRef = (params: {
  indexId: string;
  displayName?: string;
}) => {
  return retrieverRef({
    name: `pinecone/${params.indexId}`,
    info: {
      label: params.displayName ?? `Pinecone - ${params.indexId}`,
    },
    configSchema: PineconeRetrieverOptionsSchema,
  });
};

/**
 * pineconeIndexerRef function creates an indexer for Pinecone.
 * @param params The params for the new Pinecone indexer.
 * @param params.indexId The indexId for the Pinecone indexer.
 * @param params.displayName  A display name for the indexer.
If not specified, the default label will be `Pinecone - <indexId>`
 * @returns A reference to a Pinecone indexer.
 */
export const pineconeIndexerRef = (params: {
  indexId: string;
  displayName?: string;
}) => {
  return indexerRef({
    name: `pinecone/${params.indexId}`,
    info: {
      label: params.displayName ?? `Pinecone - ${params.indexId}`,
    },
    configSchema: PineconeIndexerOptionsSchema.optional(),
  });
};

/**
 * Pinecone plugin that provides a Pinecone retriever and indexer
 * @param params An array of params to set up Pinecone retrievers and indexers
 * @param params.clientParams PineconeConfiguration containing the
PINECONE_API_KEY. If not set, the PINECONE_API_KEY environment variable will
be used instead.
 * @param params.indexId The name of the index
 * @param params.indexHostUrl The host url for the index. If not set, only indexId will be used to target the index
 * @param params.embedder The embedder to use for the indexer and retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @returns The Pinecone Genkit plugin
 */
export function pinecone<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: {
    clientParams?: PineconeConfiguration;
    indexId: string;
    indexHostUrl?: string;
    contentKey?: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }[]
): GenkitPlugin {
  return genkitPlugin('pinecone', async (ai: Genkit) => {
    params.map((i) => configurePineconeRetriever(ai, i));
    params.map((i) => configurePineconeIndexer(ai, i));
  });
}

export default pinecone;

/**
 * Configures a Pinecone retriever.
 * @param ai A Genkit instance
 * @param params The params for the retriever
 * @param params.indexId The name of the retriever
 * @param params.indexHostUrl The host url for the index. If not set, only the indexId will be used to target the index
 * @param params.clientParams PineconeConfiguration containing the
PINECONE_API_KEY. If not set, the PINECONE_API_KEY environment variable will
be used instead.
 * @param params.textKey Deprecated. Please use contentKey.
 * @param params.contentKey The metadata key that contains the
content. If not specified, the value '_content' is used by default.
 * @param params.embedder The embedder to use for the retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @returns A Pinecone retriever
 */
export function configurePineconeRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    indexId: string;
    indexHostUrl?: string;
    clientParams?: PineconeConfiguration;
    /**
     * @deprecated use contentKey instead.
     */
    textKey?: string;
    contentKey?: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {
  const { indexId, indexHostUrl, embedder, embedderOptions } = {
    ...params,
  };
  const pineconeConfig = params.clientParams ?? getDefaultConfig();
  const contentKey = params.contentKey ?? params.textKey ?? CONTENT_KEY;
  const pinecone = new Pinecone(pineconeConfig);
  const index = pinecone.index(indexId, indexHostUrl);

  return ai.defineRetriever(
    {
      name: `pinecone/${params.indexId}`,
      configSchema: PineconeRetrieverOptionsSchema,
    },
    async (content, options) => {
      const queryEmbeddings = await ai.embed({
        embedder,
        content,
        options: embedderOptions,
      });
      const scopedIndex = !!options.namespace
        ? index.namespace(options.namespace)
        : index;
      const response = await scopedIndex.query({
        topK: options.k,
        vector: queryEmbeddings[0].embedding,
        includeValues: false,
        includeMetadata: true,
      });
      return {
        documents: response.matches
          .map((m) => m.metadata)
          .filter((m): m is RecordMetadata => !!m)
          .map((m) => {
            const metadata = m;
            return Document.fromData(
              metadata[contentKey] as string,
              metadata[CONTENT_TYPE] as string,
              metadata.docMetadata
                ? (JSON.parse(metadata.docMetadata as string) as Record<
                    string,
                    unknown
                  >)
                : undefined
            );
          }),
      };
    }
  );
}

/**
 * Configures a Pinecone indexer.
 * @param ai A Genkit instance
 * @param params The params for the indexer
 * @param params.indexId The name of the indexer
 * @param params.indexHostUrl The host url for the index. If not set, only the indexId will be used to target the index
 * @param params.clientParams PineconeConfiguration containing the
PINECONE_API_KEY. If not set, the PINECONE_API_KEY environment variable will
be used instead.
 * @param params.textKey Deprecated. Please use contentKey.
 * @param params.contentKey The metadata key that contains the
content. If not specified, the value '_content' is used by default.
 * @param params.embedder The embedder to use for the retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @returns A Genkit indexer
 */
export function configurePineconeIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    indexId: string;
    indexHostUrl?: string;
    clientParams?: PineconeConfiguration;
    /**
     * @deprecated use contentKey instead.
     */
    textKey?: string;
    contentKey?: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {
  const { indexId, indexHostUrl, embedder, embedderOptions } = {
    ...params,
  };
  const pineconeConfig = params.clientParams ?? getDefaultConfig();
  const contentKey = params.contentKey ?? params.textKey ?? CONTENT_KEY;
  const pinecone = new Pinecone(pineconeConfig);
  const index = pinecone.index(indexId, indexHostUrl);

  return ai.defineIndexer(
    {
      name: `pinecone/${params.indexId}`,
      configSchema: PineconeIndexerOptionsSchema.optional(),
    },
    async (docs, options) => {
      const scopedIndex = !!options?.namespace
        ? index.namespace(options.namespace)
        : index;

      const embeddings = await Promise.all(
        docs.map((doc) =>
          ai.embed({
            embedder,
            content: doc,
            options: embedderOptions,
          })
        )
      );
      await scopedIndex.upsert(
        embeddings
          .map((value, i) => {
            const doc = docs[i];
            // The array of embeddings for this document
            const docEmbeddings: Embedding[] = value;

            // Create one doc per docEmbedding so we can store them 1:1.
            // They should be unique because the embedding metadata is
            // added to the new docs.
            const embeddingDocs = doc.getEmbeddingDocuments(docEmbeddings);

            return docEmbeddings.map((docEmbedding, j) => {
              const metadata: RecordMetadata = {
                docMetadata: JSON.stringify(embeddingDocs[j].metadata),
              };
              metadata[contentKey] = embeddingDocs[j].data;
              metadata[CONTENT_TYPE] = embeddingDocs[j].dataType || '';
              const id = Md5.hashStr(JSON.stringify(embeddingDocs[j]));
              return {
                id,
                values: docEmbedding.embedding,
                metadata,
              };
            });
          })
          .reduce((acc, val) => {
            return acc.concat(val);
          }, [])
      );
    }
  );
}

/**
 * Helper function for creating a Pinecone index.
 * @param params The params for creating a Pinecone index
 * @param params.clientParams The params to initialize Pinecone.
 * @param params.options The options for creating the index.
 * @returns A Pinecone index.
 */
export async function createPineconeIndex(params: {
  clientParams?: PineconeConfiguration;
  options: CreateIndexOptions;
}) {
  const pineconeConfig = params.clientParams ?? getDefaultConfig();
  const pinecone = new Pinecone(pineconeConfig);
  return await pinecone.createIndex(params.options);
}

/**
 * Helper function to describe a Pinecone index. Use it to check if a newly created index is ready for use.
 * @param params The params for describing a Pinecone index.
 * @param params.clientParams The params to initialize Pinecone.
 * @param params.name The name of the Pinecone index to describe.
 * @return A description of the Pinecone index.
 */
export async function describePineconeIndex(params: {
  clientParams?: PineconeConfiguration;
  name: string;
}) {
  const pineconeConfig = params.clientParams ?? getDefaultConfig();
  const pinecone = new Pinecone(pineconeConfig);
  return await pinecone.describeIndex(params.name);
}

/**
 * Helper function for deleting pinecone indices.
 * @param params The params for deleting a Pinecone index.
 * @param params.clientParams The params to initialize Pinecone.
 * @param params.name The name of the Pinecone index to delete.
 * @returns a void Promise that is fulfilled when the index has been deleted.
 */
export async function deletePineconeIndex(params: {
  clientParams?: PineconeConfiguration;
  name: string;
}) {
  const pineconeConfig = params.clientParams ?? getDefaultConfig();
  const pinecone = new Pinecone(pineconeConfig);
  return await pinecone.deleteIndex(params.name);
}

function getDefaultConfig() {
  const maybeApiKey = process.env.PINECONE_API_KEY;
  if (!maybeApiKey)
    throw new Error(
      'Please pass in the API key or set PINECONE_API_KEY environment variable.\n' +
        'For more details see https://genkit.dev/docs/plugins/pinecone'
    );
  return { apiKey: maybeApiKey } as PineconeConfiguration;
}
