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

import * as aiplatform from '@google-cloud/aiplatform';
import { z, type GenkitSchema } from 'genkit';
import type { EmbedderReference } from 'genkit/embedder';
import { CommonRetrieverOptionsSchema, type Document } from 'genkit/retriever';
import type { InferOutput } from 'genkit/schema';
import type { GoogleAuth } from 'google-auth-library';
import type { PluginOptions } from '../types.js';

// This internal interface will be passed to the vertexIndexers and vertexRetrievers functions
export interface VertexVectorSearchOptions<
  EmbedderCustomOptions extends GenkitSchema,
> {
  pluginOptions: PluginOptions;
  authClient: GoogleAuth;
  defaultEmbedder?: EmbedderReference<EmbedderCustomOptions>;
}

export type IIndexDatapoint =
  aiplatform.protos.google.cloud.aiplatform.v1.IIndexDatapoint;

export class Datapoint extends aiplatform.protos.google.cloud.aiplatform.v1
  .IndexDatapoint {
  constructor(properties: IIndexDatapoint) {
    super(properties);
  }
}

export type IFindNeighborsRequest =
  aiplatform.protos.google.cloud.aiplatform.v1.IFindNeighborsRequest;
export type IFindNeighborsResponse =
  aiplatform.protos.google.cloud.aiplatform.v1.IFindNeighborsResponse;
export type ISparseEmbedding =
  aiplatform.protos.google.cloud.aiplatform.v1.IndexDatapoint.ISparseEmbedding;
export type IRestriction =
  aiplatform.protos.google.cloud.aiplatform.v1.IndexDatapoint.IRestriction;
export type INumericRestriction =
  aiplatform.protos.google.cloud.aiplatform.v1.IndexDatapoint.INumericRestriction;

// Define the Zod schema for ISparseEmbedding
export const SparseEmbeddingSchema = z.object({
  values: z.array(z.number()).optional(),
  dimensions: z.array(z.union([z.number(), z.string()])).optional(),
});

export type SparseEmbedding = z.infer<typeof SparseEmbeddingSchema>;

// Define the Zod schema for IRestriction
export const RestrictionSchema = z.object({
  namespace: z.string(),
  allowList: z.array(z.string()),
  denyList: z.array(z.string()),
});

export type Restriction = z.infer<typeof RestrictionSchema>;

export const NumericRestrictionOperatorSchema = z.enum([
  'OPERATOR_UNSPECIFIED',
  'LESS',
  'LESS_EQUAL',
  'EQUAL',
  'GREATER_EQUAL',
  'GREATER',
  'NOT_EQUAL',
]);

export type NumericRestrictionOperator = z.infer<
  typeof NumericRestrictionOperatorSchema
>;

// Define the Zod schema for INumericRestriction
export const NumericRestrictionSchema = z.object({
  valueInt: z.union([z.number(), z.string()]).optional(),
  valueFloat: z.number().optional(),
  valueDouble: z.number().optional(),
  namespace: z.string(),
  op: z.union([NumericRestrictionOperatorSchema, z.null()]).optional(),
});

export type NumericRestriction = z.infer<typeof NumericRestrictionSchema>;

// Define the Zod schema for ICrowdingTag
export const CrowdingTagSchema = z.object({
  crowdingAttribute: z.string().optional(),
});

export type CrowdingTag = z.infer<typeof CrowdingTagSchema>;

// Define the Zod schema for IIndexDatapoint
const IndexDatapointSchema = z.object({
  datapointId: z.string().optional(),
  featureVector: z.array(z.number()).optional(),
  sparseEmbedding: SparseEmbeddingSchema.optional(),
  restricts: z.array(RestrictionSchema).optional(),
  numericRestricts: z.array(NumericRestrictionSchema).optional(),
  crowdingTag: CrowdingTagSchema.optional(),
});

// Define the Zod schema for INeighbor
export const NeighborSchema = z.object({
  datapoint: IndexDatapointSchema.optional(),
  distance: z.number().optional(),
  sparseDistance: z.number().optional(),
});

export type Neighbor = z.infer<typeof NeighborSchema>;

// Define the Zod schema for INearestNeighbors
const NearestNeighborsSchema = z.object({
  id: z.string().optional(),
  neighbors: z.array(NeighborSchema).optional(),
});

// Define the Zod schema for IFindNeighborsResponse
export const FindNeighborsResponseSchema = z.object({
  nearestNeighbors: z.array(NearestNeighborsSchema).optional(),
});

export type FindNeighborsResponse = z.infer<typeof FindNeighborsResponseSchema>;

// TypeScript types for Zod schemas
type IndexDatapoint = z.infer<typeof IndexDatapointSchema>;

// Function to assert type equality
function assertTypeEquality<T>(value: T): void {}

// Asserting type equality
assertTypeEquality<IIndexDatapoint>({} as IndexDatapoint);
assertTypeEquality<IFindNeighborsResponse>({} as FindNeighborsResponse);

export const VertexAIVectorRetrieverOptionsSchema =
  CommonRetrieverOptionsSchema.extend({}).optional();

export type VertexAIVectorRetrieverOptions = z.infer<
  typeof VertexAIVectorRetrieverOptionsSchema
>;

export const VertexAIVectorIndexerOptionsSchema = z.any();

export type VertexAIVectorIndexerOptions = z.infer<
  typeof VertexAIVectorIndexerOptionsSchema
>;

/**
 * A document retriever function that takes an array of Neighbors from Vertex AI Vector Search query result, and resolves to a list of documents.
 * Also takes an options object that can be used to configure the retriever.
 */
export type DocumentRetriever<Options extends { k?: number } = { k?: number }> =
  (docIds: Neighbor[], options?: Options) => Promise<Document[]>;

/**
 * Indexer function that takes an array of documents, stores them in a database of the user's choice, and resolves to a list of document ids.
 * Also takes an options object that can be used to configure the indexer. Only Streaming Update Indexers are supported.
 */
export type DocumentIndexer<Options extends {} = {}> = (
  docs: Document[],
  options?: Options
) => Promise<string[]>;

export interface VectorSearchOptions<
  EmbedderCustomOptions extends GenkitSchema,
  IndexerOptions extends {},
  RetrieverOptions extends { k?: number },
> {
  // Specify the Vertex AI Index and IndexEndpoint to use for indexing and retrieval
  deployedIndexId: string;
  indexEndpointId: string;
  publicDomainName: string;
  indexId: string;
  // Document retriever and indexer functions to use for indexing and retrieval by the plugin's own indexers and retrievers
  documentRetriever: DocumentRetriever<RetrieverOptions>;
  documentIndexer: DocumentIndexer<IndexerOptions>;
  // Embedder and default options to use for indexing and retrieval
  embedder?: EmbedderReference<EmbedderCustomOptions>;
  embedderOptions?: InferOutput<EmbedderCustomOptions>;
}
