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

import {
  GlideClient,
  GlideFt,
  type Field,
  type GlideClientConfiguration,
  type NumericField,
  type TagField,
  type TextField,
  type VectorField,
} from '@valkey/valkey-glide';
import {
  Document,
  indexerRef,
  retrieverRef,
  z,
  type EmbedderArgument,
  type Genkit,
} from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import { CommonRetrieverOptionsSchema } from 'genkit/retriever';
import { Md5 } from 'ts-md5';

/** Distance metric for the HNSW index. */
export type ValkeyDistanceMetric = 'COSINE' | 'L2' | 'IP';

/** Declares a metadata field to index for query-time filtering. */
export type ValkeyMetadataField = {
  name: string;
  type: 'TAG' | 'NUMERIC';
};

const ValkeyRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().max(1000).default(10),
  /** Raw FT.SEARCH pre-filter expression. Do not pass untrusted user input. */
  filter: z.string().optional(),
});

const ValkeyIndexerOptionsSchema = z.null().optional();

type ValkeyPluginParams<
  EmbedderCustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  indexName: string;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
  clientConfig: GlideClientConfiguration;
  dimension: number;
  prefix?: string;
  distanceMetric?: ValkeyDistanceMetric;
  metadataFields?: ValkeyMetadataField[];
}[];

/** Tracks active GlideClient instances for cleanup on shutdown. */
const activeClients: GlideClient[] = [];

/**
 * Close all GlideClient connections created by the Valkey plugin.
 * Call this during application shutdown to prevent connection leaks.
 */
export async function closeValkeyClients(): Promise<void> {
  const clients = activeClients.splice(0);
  await Promise.all(clients.map((c) => c.close()));
}

/**
 * Valkey plugin that provides a Valkey vector store retriever and indexer.
 * Requires a Valkey instance with the valkey-search module loaded.
 */
export function valkeyPlugin<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: ValkeyPluginParams<EmbedderCustomOptions>
): GenkitPlugin {
  return genkitPlugin('valkey', async (ai: Genkit) => {
    for (const config of params) {
      const client = await GlideClient.createClient(config.clientConfig);
      activeClients.push(client);
      const prefix = config.prefix ?? config.indexName;
      const distanceMetric = config.distanceMetric ?? 'COSINE';
      const metadataFields = config.metadataFields ?? [];

      await ensureIndex(client, config.indexName, config.dimension, prefix, distanceMetric, metadataFields);

      configureValkeyIndexer(ai, { ...config, prefix, client, metadataFields });
      configureValkeyRetriever(ai, { ...config, prefix, client });
    }
  });
}

/**
 * Creates a reference to a Valkey retriever.
 */
export const valkeyRetrieverRef = (params: {
  indexName: string;
  displayName?: string;
}) => {
  return retrieverRef({
    name: `valkey/${params.indexName}`,
    info: {
      label: params.displayName ?? `Valkey - ${params.indexName}`,
    },
    configSchema: ValkeyRetrieverOptionsSchema.optional(),
  });
};

/**
 * Creates a reference to a Valkey indexer.
 */
export const valkeyIndexerRef = (params: {
  indexName: string;
  displayName?: string;
}) => {
  return indexerRef({
    name: `valkey/${params.indexName}`,
    info: {
      label: params.displayName ?? `Valkey - ${params.indexName}`,
    },
    configSchema: ValkeyIndexerOptionsSchema,
  });
};

export default valkeyPlugin;

/** Alias exports matching the naming convention from the narrative. */
export const valkeyIndexer = valkeyIndexerRef;
export const valkeyRetriever = valkeyRetrieverRef;

/**
 * Ensures the FT index exists. If it already exists, the duplicate error is
 * swallowed silently.
 */
async function ensureIndex(
  client: GlideClient,
  indexName: string,
  dimension: number,
  prefix: string,
  distanceMetric: ValkeyDistanceMetric,
  metadataFields: ValkeyMetadataField[]
): Promise<void> {
  const vectorField: VectorField = {
    type: 'VECTOR',
    name: 'embedding',
    attributes: {
      algorithm: 'HNSW',
      dimensions: dimension,
      distanceMetric,
      type: 'FLOAT32',
    },
  };
  const contentField: TextField = { type: 'TEXT', name: '_content' };
  const metaField: TextField = { type: 'TEXT', name: '_metadata' };

  const schema: Field[] = [vectorField, contentField, metaField];

  for (const mf of metadataFields) {
    if (mf.type === 'NUMERIC') {
      const f: NumericField = { type: 'NUMERIC', name: mf.name };
      schema.push(f);
    } else {
      const f: TagField = { type: 'TAG', name: mf.name };
      schema.push(f);
    }
  }

  try {
    await GlideFt.create(
      client,
      indexName,
      schema,
      {
        dataType: 'HASH',
        prefixes: [`${prefix}:`],
      }
    );
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    if (!message.includes('already exists')) {
      throw err;
    }
  }
}

/**
 * Configures a Valkey indexer that embeds documents and stores them as Hashes.
 */
function configureValkeyIndexer<EmbedderCustomOptions extends z.ZodTypeAny>(
  ai: Genkit,
  params: {
    indexName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    prefix: string;
    client: GlideClient;
    metadataFields: ValkeyMetadataField[];
  }
) {
  const { indexName, embedder, embedderOptions, prefix, client, metadataFields } = params;

  return ai.defineIndexer(
    {
      name: `valkey/${indexName}`,
      configSchema: ValkeyIndexerOptionsSchema,
    },
    async (docs) => {
      // Batch-embed all documents in a single call for efficiency.
      // Fall back to per-document embedding if embedMany is unavailable.
      let embeddings: { embedding: number[] }[];
      if (ai.embedMany) {
        embeddings = await ai.embedMany({
          embedder,
          content: docs,
          options: embedderOptions,
        });
      } else {
        embeddings = await Promise.all(
          docs.map((doc) =>
            ai.embed({
              embedder,
              content: doc,
              options: embedderOptions,
            }).then((result) => result[0])
          )
        );
      }

      for (let i = 0; i < docs.length; i++) {
        const doc = docs[i];
        const docEmbeddings = [embeddings[i]];
        const embeddingDocs = doc.getEmbeddingDocuments(docEmbeddings);

        for (let j = 0; j < docEmbeddings.length; j++) {
          const embedding = docEmbeddings[j].embedding;
          const embeddingDoc = embeddingDocs[j];
          const id = Md5.hashStr(JSON.stringify(embeddingDoc));
          const embeddingBuffer = Buffer.from(
            new Float32Array(embedding).buffer
          );

          const fields: { field: string; value: string | Buffer }[] = [
            { field: 'embedding', value: embeddingBuffer },
            { field: '_content', value: embeddingDoc.data },
            {
              field: '_metadata',
              value: JSON.stringify(embeddingDoc.metadata ?? {}),
            },
            {
              field: '_dataType',
              value: embeddingDoc.dataType ?? '',
            },
          ];

          // Store declared metadata keys as top-level HASH fields for filtering.
          if (embeddingDoc.metadata) {
            for (const mf of metadataFields) {
              const val = (embeddingDoc.metadata as Record<string, unknown>)[mf.name];
              if (val !== undefined && val !== null) {
                fields.push({ field: mf.name, value: String(val) });
              }
            }
          }

          await client.hset(`${prefix}:${id}`, fields);
        }
      }
    }
  );
}

/**
 * Configures a Valkey retriever that performs KNN vector search.
 */
function configureValkeyRetriever<EmbedderCustomOptions extends z.ZodTypeAny>(
  ai: Genkit,
  params: {
    indexName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    prefix: string;
    client: GlideClient;
  }
) {
  const { indexName, embedder, embedderOptions, client } = params;

  return ai.defineRetriever(
    {
      name: `valkey/${indexName}`,
      configSchema: ValkeyRetrieverOptionsSchema.optional(),
    },
    async (content, options) => {
      const queryEmbeddings = await ai.embed({
        embedder,
        content,
        options: embedderOptions,
      });

      if (!queryEmbeddings.length) {
        throw new Error('valkey: embedder returned no embeddings for query');
      }
      const queryVector = queryEmbeddings[0].embedding;
      const queryBuffer = Buffer.from(new Float32Array(queryVector).buffer);
      const k = options?.k ?? 10;
      const filter = options?.filter;

      // Build KNN query with optional pre-filter expression.
      const knnQuery = filter
        ? `(${filter})=>[KNN $k @embedding $query_vec]`
        : '*=>[KNN $k @embedding $query_vec]';

      const result = await GlideFt.search(
        client,
        indexName,
        knnQuery,
        {
          params: [
            { key: 'k', value: k.toString() },
            { key: 'query_vec', value: queryBuffer },
          ],
          returnFields: [
            { fieldIdentifier: '_content' },
            { fieldIdentifier: '_metadata' },
            { fieldIdentifier: '_dataType' },
            { fieldIdentifier: '__embedding_score' },
          ],
        }
      );

      const [, records] = result;

      const documents = records.map((record) => {
        const fields = record.value;
        let contentValue = '';
        let metadataValue = '{}';
        let dataTypeValue = '';

        for (const field of fields) {
          const fieldKey =
            field.key instanceof Buffer
              ? field.key.toString()
              : String(field.key);
          const fieldVal =
            field.value instanceof Buffer
              ? field.value.toString()
              : String(field.value);

          switch (fieldKey) {
            case '_content':
              contentValue = fieldVal;
              break;
            case '_metadata':
              metadataValue = fieldVal;
              break;
            case '_dataType':
              dataTypeValue = fieldVal;
              break;
          }
        }

        const metadata = safeJsonParse(metadataValue);
        const effectiveDataType = dataTypeValue || 'text';
        return Document.fromData(contentValue, effectiveDataType, metadata);
      });

      return { documents };
    }
  );
}

function safeJsonParse(value: string): Record<string, unknown> | undefined {
  try {
    const parsed = JSON.parse(value);
    if (parsed && typeof parsed === 'object') {
      return parsed as Record<string, unknown>;
    }
    return undefined;
  } catch {
    return undefined;
  }
}
