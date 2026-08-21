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

import { z, type Genkit } from 'genkit';
import { type EmbedderArgument } from 'genkit/embedder';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
} from 'genkit/retriever';
import { randomUUID } from 'node:crypto';
import { PostgresEngine } from './engine.js';
import { DistanceStrategy, type QueryOptions } from './indexes.js';

export { Column, PostgresEngine } from './engine.js';
export {
  DistanceStrategy,
  ExactNearestNeighbor,
  HNSWIndex,
  HNSWQueryOptions,
  IVFFlatIndex,
  IVFFlatQueryOptions,
} from './indexes.js';

const PostgresRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().max(1000),
  filter: z.string().optional(),
});

const PostgresIndexerOptionsSchema = z.object({
  batchSize: z.number().default(100),
});

/**
 * postgresRetrieverRef function creates a retriever for Postgres.
 * @param params The params for the new Postgres retriever
 * @param params.tableName The table name for the postgres retriever
If not specified, the default label will be `Postgres - <tableName>`
 * @returns A reference to a Postgres retriever.
 */
export const postgresRetrieverRef = (params: { tableName: string }) => {
  return retrieverRef({
    name: `postgres/${params.tableName}`,
    info: {
      label: params.tableName ?? `Postgres - ${params.tableName}`,
    },
    configSchema: PostgresRetrieverOptionsSchema,
  });
};

/**
 * postgresIndexerRef function creates an indexer for Postgres.
 * @param params The params for the new Postgres indexer.
 * @param params.tableName The table name for the Postgres indexer.
If not specified, the default label will be `Postgres - <tableName>`
 * @returns A reference to a Postgres indexer.
 */
export const postgresIndexerRef = (params: { tableName: string }) => {
  return indexerRef({
    name: `postgres/${params.tableName}`,
    info: {
      label: params.tableName ?? `Postgres - ${params.tableName}`,
    },
    configSchema: PostgresIndexerOptionsSchema.optional(),
  });
};

/**
 * Postgres plugin that provides a Postgres retriever and indexer
 * @param params An array of params to set up Postgres retrievers and indexers
 * @param params.tableName The name of the table
 * @param params.embedder The embedder to use for the indexer and retriever
 * @param params.embedderOptions Options to customize the embedder
 * @returns The Postgres Genkit plugin
 */
export function postgres<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: {
    tableName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    engine: PostgresEngine;
    schemaName?: string;
    contentColumn?: string;
    embeddingColumn?: string;
    metadataColumns?: string[];
    ignoreMetadataColumns?: string[];
    idColumn?: string;
    metadataJsonColumn?: string;
    distanceStrategy?: DistanceStrategy;
    indexQueryOptions?: QueryOptions;
  }[]
): GenkitPlugin {
  return genkitPlugin('postgres', async (ai: Genkit) => {
    params.map((i) => configurePostgresRetriever(ai, i));
    params.map((i) => configurePostgresIndexer(ai, i));
  });
}

export default postgres;

/**
 * Configures a Postgres retriever.
 * @param ai A Genkit instance
 * @param params The params for the retriever
 * @param params.tableName The name of the table
 * @param params.embedder The embedder to use for the retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @returns A Postgres retriever
 */
export async function configurePostgresRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    tableName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    engine: PostgresEngine;
    schemaName?: string;
    contentColumn?: string;
    embeddingColumn?: string;
    metadataColumns?: string[];
    ignoreMetadataColumns?: string[];
    idColumn?: string;
    metadataJsonColumn?: string;
    distanceStrategy?: DistanceStrategy;
    indexQueryOptions?: QueryOptions;
  }
) {
  const schemaName = params.schemaName ?? 'public';
  const contentColumn = params.contentColumn ?? 'content';
  const embeddingColumn = params.embeddingColumn ?? 'embedding';
  const distanceStrategy =
    params.distanceStrategy ?? DistanceStrategy.COSINE_DISTANCE;
  if (!params.engine) {
    throw new Error('Engine is required');
  }

  async function checkColumns() {
    if (
      params.metadataColumns !== undefined &&
      params.ignoreMetadataColumns !== undefined
    ) {
      throw 'Can not use both metadata_columns and ignore_metadata_columns.';
    }

    const { rows } = await params.engine.pool.raw(
      `SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '${params.tableName}' AND table_schema = '${schemaName}'`
    );
    const columns: { [key: string]: any } = {};

    for (const index in rows) {
      const row = rows[index];
      columns[row['column_name']] = row['data_type'];
    }

    if (params.idColumn && !columns.hasOwnProperty(params.idColumn)) {
      throw `Id column: ${params.idColumn}, does not exist.`;
    }

    if (contentColumn && !columns.hasOwnProperty(contentColumn)) {
      throw `Content column: ${params.contentColumn}, does not exist.`;
    }

    const contentType = contentColumn ? columns[contentColumn] : '';

    if (contentType !== 'text' && !contentType.includes('char')) {
      throw `Content column: ${params.contentColumn}, is type: ${contentType}. It must be a type of character string.`;
    }

    if (embeddingColumn && !columns.hasOwnProperty(embeddingColumn)) {
      throw `Embedding column: ${embeddingColumn}, does not exist.`;
    }

    if (embeddingColumn && columns[embeddingColumn] !== 'USER-DEFINED') {
      throw `Embedding column: ${embeddingColumn} is not of type Vector.`;
    }

    const metadataJsonColumnToCheck = params.metadataJsonColumn ?? '';
    params.metadataJsonColumn = columns.hasOwnProperty(
      metadataJsonColumnToCheck
    )
      ? params.metadataJsonColumn
      : '';

    if (params.metadataColumns) {
      for (const column of params.metadataColumns) {
        if (column && !columns.hasOwnProperty(column)) {
          throw `Metadata column: ${column}, does not exist.`;
        }
      }
    }

    const allColumns = columns;
    if (
      params.ignoreMetadataColumns !== undefined &&
      params.ignoreMetadataColumns.length > 0
    ) {
      for (const column of params.ignoreMetadataColumns) {
        delete allColumns[column];
      }

      if (params.idColumn) {
        delete allColumns[params.idColumn];
      }
      if (contentColumn) {
        delete allColumns[contentColumn];
      }
      if (embeddingColumn) {
        delete allColumns[embeddingColumn];
      }
      params.metadataColumns = Object.keys(allColumns);
    }
  }

  async function queryCollection(
    embedding: number[],
    k?: number | undefined,
    filter?: string | undefined
  ) {
    k = k ?? 4;
    const operator = distanceStrategy.operator;
    const searchFunction = distanceStrategy.searchFunction;
    const _filter = filter !== undefined ? `WHERE ${filter}` : '';
    const metadataColNames =
      params.metadataColumns && params.metadataColumns.length > 0
        ? `"${params.metadataColumns.join('","')}"`
        : '';
    const metadataJsonColName = params.metadataJsonColumn
      ? `, "${params.metadataJsonColumn}"`
      : '';

    const query = `SELECT "${params.idColumn}", "${contentColumn}", "${embeddingColumn}", ${metadataColNames} ${metadataJsonColName}, ${searchFunction}("${embeddingColumn}", '[${embedding}]') as distance FROM "${schemaName}"."${params.tableName}" ${_filter} ORDER BY "${embeddingColumn}" ${operator} '[${embedding}]' LIMIT ${k};`;

    if (params.indexQueryOptions) {
      await params.engine.pool.raw(
        `SET LOCAL ${params.indexQueryOptions.to_string()}`
      );
    }

    const { rows } = await params.engine.pool.raw(query);

    return rows;
  }

  return ai.defineRetriever(
    {
      name: `postgres/${params.tableName}`,
      configSchema: PostgresRetrieverOptionsSchema,
    },
    async (content, options) => {
      console.log(`Retrieving data for table: ${params.tableName}`);
      checkColumns();
      const queryEmbeddings = await ai.embed({
        embedder: params.embedder,
        content,
        options: params.embedderOptions,
      });
      const embedding = queryEmbeddings[0].embedding;
      const results = await queryCollection(
        embedding,
        options.k,
        options.filter
      );
      const documents: Document[] = [];
      for (const row of results) {
        const metadata =
          params.metadataJsonColumn && row[params.metadataJsonColumn]
            ? row[params.metadataJsonColumn]
            : {};
        if (params.metadataColumns) {
          for (const col of params.metadataColumns) {
            metadata[col] = row[col];
          }
        }
        documents.push(
          new Document({
            content: row[contentColumn],
            metadata: metadata,
          })
        );
      }

      return { documents };
    }
  );
}

/**
 * Configures a Postgres indexer.
 * @param ai A Genkit instance
 * @param params The params for the indexer
 * @param params.tableName The name of the indexer
 * @param params.engine The engine to use for the indexer
 * @param params.embedder The embedder to use for the retriever
 * @param params.embedderOptions  Options to customize the embedder
 * @param params.metadataColumns The metadata columns to use for the indexer
 * @param params.idColumn The id column to use for the indexer
 * @param params.metadataJsonColumn The metadata json column to use for the indexer
 * @param params.contentColumn The content column to use for the indexer
 * @param params.embeddingColumn The embedding column to use for the indexer
 * @param params.schemaName The schema name to use for the indexer
 * @param params.chunkSize The chunk size to use for the indexer
 * @returns Add documents to vector store
 */
export function configurePostgresIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    tableName: string;
    engine: PostgresEngine;
    schemaName?: string;
    contentColumn?: string;
    embeddingColumn?: string;
    metadataColumns?: string[];
    ignoreMetadataColumns?: string[];
    idColumn?: string;
    metadataJsonColumn?: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }
) {
  const schemaName = params.schemaName ?? 'public';
  const contentColumn = params.contentColumn ?? 'content';
  const embeddingColumn = params.embeddingColumn ?? 'embedding';
  const idColumn = params.idColumn ?? 'id';
  const metadataJsonColumn = params.metadataJsonColumn ?? 'metadata';

  if (!params.engine) {
    throw new Error('Engine is required');
  }

  if (params.metadataColumns && params.ignoreMetadataColumns) {
    throw new Error(
      'Cannot use both metadataColumns and ignoreMetadataColumns'
    );
  }

  async function checkColumns() {
    const { rows } = await params.engine.pool.raw(
      `SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '${params.tableName}' AND table_schema = '${schemaName}'`
    );

    const columns: { [key: string]: any } = {};

    for (const index in rows) {
      const row = rows[index];
      columns[row['column_name']] = row['data_type'];
    }

    if (!columns.hasOwnProperty(idColumn)) {
      throw new Error(`Id column: ${idColumn}, does not exist.`);
    }

    if (!columns.hasOwnProperty(contentColumn)) {
      throw new Error(`Content column: ${contentColumn}, does not exist.`);
    }

    if (!columns.hasOwnProperty(embeddingColumn)) {
      throw new Error(`Embedding column: ${embeddingColumn}, does not exist.`);
    }

    if (columns[embeddingColumn] !== 'USER-DEFINED') {
      throw new Error(
        `Embedding column: ${embeddingColumn} is not of type Vector.`
      );
    }

    if (params.metadataColumns) {
      for (const column of params.metadataColumns) {
        if (column && !columns.hasOwnProperty(column)) {
          throw new Error(`Metadata column: ${column}, does not exist.`);
        }
      }
    }
  }

  return ai.defineIndexer(
    {
      name: `postgres/${params.tableName}`,
      configSchema: PostgresIndexerOptionsSchema.optional(),
    },
    async (
      docs: Document[] | { documents: Document[]; options?: any },
      options?: { batchSize?: number }
    ) => {
      try {
        await checkColumns();

        // Normalize input to always have documents array and merged options
        const documents = Array.isArray(docs) ? docs : docs.documents || [];
        const mergedOptions = Array.isArray(docs)
          ? options
          : docs.options || options || {};
        const batchSize = mergedOptions.batchSize || 100;

        console.log(
          `Indexing ${documents.length} documents in batches of ${batchSize}`
        );

        for (let i = 0; i < documents.length; i += batchSize) {
          const chunk = documents.slice(i, i + batchSize);
          const texts = chunk.map((doc) =>
            Array.isArray(doc.content)
              ? doc.content.map((c) => c.text).join(' ')
              : doc.content
          );

          let embeddings;
          try {
            if (ai.embedMany) {
              embeddings = await ai.embedMany({
                embedder: params.embedder,
                content: texts,
                options: params.embedderOptions,
              });
            } else {
              embeddings = await Promise.all(
                texts.map((text) =>
                  ai
                    .embed({
                      embedder: params.embedder,
                      content: text,
                      options: params.embedderOptions,
                    })
                    .then((res) => res[0])
                )
              );
            }
          } catch (error: unknown) {
            throw new Error('Embedding failed', { cause: error });
          }

          const insertData = chunk.map((doc, index) => ({
            [idColumn]: doc.metadata?.[idColumn] || randomUUID(),
            [contentColumn]: texts[index],
            [embeddingColumn]: JSON.stringify(embeddings[index].embedding),
            ...(metadataJsonColumn && {
              [metadataJsonColumn]: doc.metadata || {},
            }),
            ...Object.fromEntries(
              (params.metadataColumns || [])
                .filter((col) => doc.metadata?.[col] !== undefined)
                .map((col) => [col, doc.metadata?.[col]])
            ),
          }));

          const table = schemaName
            ? params.engine.pool.withSchema(schemaName).table(params.tableName)
            : params.engine.pool.table(params.tableName);

          await table.insert(insertData);
        }
      } catch (error: unknown) {
        console.error('Error in indexer:', error);
        throw error;
      }
    }
  );
}
