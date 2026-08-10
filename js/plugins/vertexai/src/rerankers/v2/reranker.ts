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
  RerankerInfo,
  RerankerReference,
  z,
  type RerankerAction,
} from 'genkit';
import { reranker as pluginReranker } from 'genkit/plugin';
import { RankedDocument, rerankerRef } from 'genkit/reranker';
import { checkModelName } from '../../common/utils.js';
import { rerankerRank } from './client.js';
import {
  RerankRequest,
  RerankRequestRecord,
  VertexRerankerClientOptions,
} from './types.js';

// Define the schema for the options used in the Vertex AI reranker
export const VertexRerankerConfigSchema = z
  .object({
    topN: z.number().optional().describe('Number of top documents to rerank'), // Optional: Number of documents to rerank
    ignoreRecordDetailsInResponse: z
      .boolean()
      .optional()
      .describe(
        'If true, the response will contain only record ID and score. By default, it is false, the response will contain record details.'
      ),
    location: z
      .string()
      .optional()
      .describe('Google Cloud location, e.g., "us-central1"'), // Optional: Location of the reranking model
  })
  .passthrough();
export type VertexRerankerConfigSchemaType = typeof VertexRerankerConfigSchema;
export type VertexRerankerConfig = z.infer<VertexRerankerConfigSchemaType>;

type ConfigSchemaType = VertexRerankerConfigSchemaType;

function commonRef(
  name: string,
  info?: RerankerInfo,
  configSchema: ConfigSchemaType = VertexRerankerConfigSchema
): RerankerReference<ConfigSchemaType> {
  return rerankerRef({
    name: `vertex-rerankers/${name}`,
    configSchema,
    info: info ?? {
      supports: {
        media: false,
      },
    },
  });
}

export const GENERIC_MODEL = commonRef('reranker');

export const DEFAULT_MODEL_NAME = 'semantic-ranker-default@latest';

export const KNOWN_MODELS = {
  'semantic-ranker-default@latest': commonRef('semantic-ranker-default@latest'),
  'semantic-ranker-default-004': commonRef('semantic-ranker-default-004'),
  'semantic-ranker-fast-004': commonRef('semantic-ranker-fast-004'),
  'semantic-ranker-default-003': commonRef('semantic-ranker-default-003'),
  'semantic-ranker-default-002': commonRef('semantic-ranker-default-002'),
} as const;
export type KnownModels = keyof typeof KNOWN_MODELS;
export type RerankerModelName = `semantic-ranker-${string}`;
export function isRerankerModelName(
  value?: string
): value is RerankerModelName {
  return !!value?.startsWith('semantic-ranker-');
}

export function reranker(
  version: string,
  options: VertexRerankerConfig = {}
): RerankerReference<VertexRerankerConfigSchemaType> {
  const name = checkModelName(version);
  return rerankerRef({
    name: `vertex-rerankers/${name}`,
    configSchema: VertexRerankerConfigSchema,
    info: {
      ...GENERIC_MODEL.info,
    },
  });
}

export function listKnownRerankers(clientOptions: VertexRerankerClientOptions) {
  return Object.keys(KNOWN_MODELS).map((name) =>
    defineReranker(name, clientOptions)
  );
}

export function defineReranker(
  name: string,
  clientOptions: VertexRerankerClientOptions
): RerankerAction {
  const ref = reranker(name);

  return pluginReranker(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
    },
    async (query, documents, options) => {
      const rerankRequest: RerankRequest = {
        // Note that model silently falls back to default if it's not recognized
        // This happens in the vertexai reranker service backend.
        model: checkModelName(ref.name),
        query: query.text,
        records: documents.map(toRerankerDoc),
        ...options,
      };
      const response = await rerankerRank(
        ref.name,
        rerankRequest,
        clientOptions
      );
      return { documents: fromRerankResponse(response, documents) };
    }
  );
}

function toRerankerDoc(doc, idx): RerankRequestRecord {
  return {
    id: `${idx}`,
    content: doc.text,
  };
}

function fromRerankResponse(response, documents): RankedDocument[] {
  return response.records.map((record) => {
    const doc = documents[record.id];
    return new RankedDocument({
      content: doc.content,
      metadata: {
        ...doc.metadata,
        score: record.score,
      },
    });
  });
}
