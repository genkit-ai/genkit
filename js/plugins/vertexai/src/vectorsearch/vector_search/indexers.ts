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

import type { Genkit, z } from 'genkit';
import { indexerRef, type IndexerAction } from 'genkit/retriever';
import {
  Datapoint,
  VertexAIVectorIndexerOptionsSchema,
  type VertexVectorSearchOptions,
} from './types.js';
import { upsertDatapoints } from './upsert_datapoints.js';

/**
 * Creates a reference to a Vertex AI indexer.
 *
 * @param {Object} params - The parameters for the indexer reference.
 * @param {string} params.indexId - The ID of the Vertex AI index.
 * @param {string} [params.displayName] - An optional display name for the indexer.
 * @returns {Object} - The indexer reference object.
 */
export const vertexAiIndexerRef = (params: {
  indexId: string;
  displayName?: string;
}) => {
  return indexerRef({
    name: `vertexai/${params.indexId}`,
    info: {
      label: params.displayName ?? `Vertex AI - ${params.indexId}`,
    },
    configSchema: VertexAIVectorIndexerOptionsSchema.optional(),
  });
};

/**
 * Creates Vertex AI indexers.
 *
 * This function returns a list of indexer actions for Vertex AI based on the provided
 * vector search options and embedder configurations.
 *
 * @param {VertexVectorSearchOptions<EmbedderCustomOptions>} params - The parameters for creating the indexers.
 * @returns {IndexerAction<z.ZodTypeAny>[]} - An array of indexer actions.
 */
export function vertexAiIndexers<EmbedderCustomOptions extends z.ZodTypeAny>(
  ai: Genkit,
  params: VertexVectorSearchOptions<EmbedderCustomOptions>
): IndexerAction<z.ZodTypeAny>[] {
  const vectorSearchOptions = params.pluginOptions.vectorSearchOptions;
  const indexers: IndexerAction<z.ZodTypeAny>[] = [];

  if (!vectorSearchOptions || vectorSearchOptions.length === 0) {
    return indexers;
  }

  for (const vectorSearchOption of vectorSearchOptions) {
    const { documentIndexer, indexId } = vectorSearchOption;
    const embedderReference =
      vectorSearchOption.embedder ?? params.defaultEmbedder;

    if (!embedderReference) {
      throw new Error(
        'Embedder reference is required to define Vertex AI retriever'
      );
    }
    const embedderOptions = vectorSearchOption.embedderOptions;

    const indexer = ai.defineIndexer(
      {
        name: `vertexai/${indexId}`,
        configSchema: VertexAIVectorIndexerOptionsSchema.optional(),
      },
      async (docs, options) => {
        let docIds: string[] = [];
        try {
          docIds = await documentIndexer(docs, options);
        } catch (error) {
          throw new Error(
            `Error storing your document content/metadata: ${error}`
          );
        }

        const embeddings = await ai.embedMany({
          embedder: embedderReference,
          content: docs,
          options: embedderOptions,
        });

        const datapoints = embeddings.map(({ embedding }, i) => {
          const dp = new Datapoint({
            datapointId: docIds[i],
            featureVector: embedding,
          });
          if (docs[i].metadata?.restricts) {
            dp.restricts = docs[i].metadata?.restricts;
          }
          if (docs[i].metadata?.numericRestricts) {
            dp.numericRestricts = docs[i].metadata?.numericRestricts;
          }
          if (docs[i].metadata?.crowdingTag) {
            dp.crowdingTag = docs[i].metadata?.crowdingTag;
          }
          return dp;
        });

        try {
          await upsertDatapoints({
            datapoints,
            authClient: params.authClient,
            projectId: params.pluginOptions.projectId!,
            location: params.pluginOptions.location!,
            indexId: indexId,
          });
        } catch (error) {
          throw error;
        }
      }
    );

    indexers.push(indexer);
  }
  return indexers;
}
