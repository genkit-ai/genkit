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
  retrieverRef,
  type Genkit,
  type RetrieverAction,
  type z,
} from 'genkit';
import { queryPublicEndpoint } from './query_public_endpoint.js';
import {
  VertexAIVectorRetrieverOptionsSchema,
  type VertexVectorSearchOptions,
} from './types.js';
import { getProjectNumber } from './utils.js';

const DEFAULT_K = 10;

/**
 * Creates Vertex AI retrievers.
 *
 * This function returns a list of retriever actions for Vertex AI based on the provided
 * vector search options and embedder configurations.
 *
 * @param {VertexVectorSearchOptions<EmbedderCustomOptions>} params - The parameters for creating the retrievers.
 * @returns {RetrieverAction<z.ZodTypeAny>[]} - An array of retriever actions.
 */
export function vertexAiRetrievers<EmbedderCustomOptions extends z.ZodTypeAny>(
  ai: Genkit,
  params: VertexVectorSearchOptions<EmbedderCustomOptions>
): RetrieverAction<z.ZodTypeAny>[] {
  const vectorSearchOptions = params.pluginOptions.vectorSearchOptions;
  const defaultEmbedder = params.defaultEmbedder;

  const retrievers: RetrieverAction<z.ZodTypeAny>[] = [];

  if (!vectorSearchOptions || vectorSearchOptions.length === 0) {
    return retrievers;
  }

  for (const vectorSearchOption of vectorSearchOptions) {
    const { documentRetriever, indexId, publicDomainName } = vectorSearchOption;
    const embedderOptions = vectorSearchOption.embedderOptions;

    const retriever = ai.defineRetriever(
      {
        name: `vertexai/${indexId}`,
        configSchema: VertexAIVectorRetrieverOptionsSchema.optional(),
      },
      async (content, options) => {
        const embedderReference =
          vectorSearchOption.embedder ?? defaultEmbedder;

        if (!embedderReference) {
          throw new Error(
            'Embedder reference is required to define Vertex AI retriever'
          );
        }

        const queryEmbedding = (
          await ai.embed({
            embedder: embedderReference,
            options: embedderOptions,
            content,
          })
        )[0].embedding; // Single embedding for text

        const accessToken = await params.authClient.getAccessToken();

        if (!accessToken) {
          throw new Error(
            'Error generating access token when defining Vertex AI retriever'
          );
        }

        const projectId = params.pluginOptions.projectId;
        if (!projectId) {
          throw new Error(
            'Project ID is required to define Vertex AI retriever'
          );
        }
        const projectNumber = await getProjectNumber(projectId);
        const location = params.pluginOptions.location;
        if (!location) {
          throw new Error('Location is required to define Vertex AI retriever');
        }

        const res = await queryPublicEndpoint({
          featureVector: queryEmbedding,
          neighborCount: options?.k || DEFAULT_K,
          accessToken,
          projectId,
          location,
          publicDomainName,
          projectNumber,
          indexEndpointId: vectorSearchOption.indexEndpointId,
          deployedIndexId: vectorSearchOption.deployedIndexId,
          restricts: content.metadata?.restricts,
          numericRestricts: content.metadata?.numericRestricts,
        });
        const nearestNeighbors = res.nearestNeighbors;

        const queryRes = nearestNeighbors ? nearestNeighbors[0] : null;
        const neighbors = queryRes ? queryRes.neighbors : null;
        if (!neighbors) {
          return { documents: [] };
        }

        const documents = await documentRetriever(neighbors, options);

        return { documents };
      }
    );

    retrievers.push(retriever);
  }

  return retrievers;
}

/**
 * Creates a reference to a Vertex AI retriever.
 *
 * @param {Object} params - The parameters for the retriever reference.
 * @param {string} params.indexId - The ID of the Vertex AI index.
 * @param {string} [params.displayName] - An optional display name for the retriever.
 * @returns {Object} - The retriever reference object.
 */
export const vertexAiRetrieverRef = (params: {
  indexId: string;
  displayName?: string;
}) => {
  return retrieverRef({
    name: `vertexai/${params.indexId}`,
    info: {
      label: params.displayName ?? `ertex AI - ${params.indexId}`,
    },
    configSchema: VertexAIVectorRetrieverOptionsSchema.optional(),
  });
};
