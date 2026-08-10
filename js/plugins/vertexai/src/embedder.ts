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

import { z, type Document, type Genkit } from 'genkit';
import {
  embedderRef,
  type EmbedderAction,
  type EmbedderReference,
} from 'genkit/embedder';
import type { GoogleAuth } from 'google-auth-library';
import type { PluginOptions } from './common/types.js';
import { predictModel, type PredictClient } from './predict.js';

/** @deprecated */
export const TaskTypeSchema = z.enum([
  'RETRIEVAL_DOCUMENT',
  'RETRIEVAL_QUERY',
  'SEMANTIC_SIMILARITY',
  'CLASSIFICATION',
  'CLUSTERING',
]);

/** @deprecated */
export type TaskType = z.infer<typeof TaskTypeSchema>;

/** @deprecated */
export const VertexEmbeddingConfigSchema = z.object({
  /**
   * The `task_type` parameter is defined as the intended downstream application
   * to help the model produce better quality embeddings.
   **/
  taskType: TaskTypeSchema.optional(),
  title: z.string().optional(),
  location: z.string().optional(),
  version: z.string().optional(),
  /**
   * The `outputDimensionality` parameter allows you to specify the dimensionality of the embedding output.
   * By default, the model generates embeddings with 768 dimensions. Models such as
   * `gemini-embedding-001`, `text-embedding-005`, and `text-multilingual-embedding-002`
   * allow the output dimensionality to be adjusted between 1 and 768.
   * By selecting a smaller output dimensionality, users can save memory and storage space, leading to more efficient computations.
   **/
  outputDimensionality: z.number().min(1).max(768).optional(),
});

/** @deprecated */
export type VertexEmbeddingConfig = z.infer<typeof VertexEmbeddingConfigSchema>;

type InputType = 'text' | 'image' | 'video';

function commonRef(
  name: string,
  input?: InputType[]
): EmbedderReference<typeof VertexEmbeddingConfigSchema> {
  return embedderRef({
    name: `vertexai/${name}`,
    configSchema: VertexEmbeddingConfigSchema,
    info: {
      dimensions: 768,
      label: `Vertex AI - ${name}`,
      supports: {
        input: input ?? ['text'],
      },
    },
  });
}

/** @deprecated */
export const textEmbeddingGecko003 = commonRef('textembedding-gecko@003');

/** @deprecated */
export const textEmbedding004 = commonRef('text-embedding-004');

/** @deprecated */
export const textEmbedding005 = commonRef('text-embedding-005');

/** @deprecated */
export const textEmbeddingGeckoMultilingual001 = commonRef(
  'textembedding-gecko-multilingual@001'
);

/** @deprecated */
export const textMultilingualEmbedding002 = commonRef(
  'text-multilingual-embedding-002'
);

/** @deprecated */
export const multimodalEmbedding001 = commonRef('multimodalembedding@001', [
  'text',
  'image',
  'video',
]);

/** @deprecated */
export const geminiEmbedding001 = embedderRef({
  name: 'vertexai/gemini-embedding-001',
  configSchema: VertexEmbeddingConfigSchema,
  info: {
    dimensions: 3072,
    label: 'Vertex AI - gemini-embedding-001',
    supports: {
      input: ['text'],
    },
  },
});

/** @deprecated */
export const SUPPORTED_EMBEDDER_MODELS: Record<string, EmbedderReference> = {
  'textembedding-gecko@003': textEmbeddingGecko003,
  'text-embedding-004': textEmbedding004,
  'text-embedding-005': textEmbedding005,
  'textembedding-gecko-multilingual@001': textEmbeddingGeckoMultilingual001,
  'text-multilingual-embedding-002': textMultilingualEmbedding002,
  'multimodalembedding@001': multimodalEmbedding001,
  'gemini-embedding-001': geminiEmbedding001,
};

// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api#request_body
interface MultimodalEmbeddingInstance {
  text?: string;
  image?: {
    // Union field can only be one of the following:
    bytesBase64Encoded?: string;
    gcsUri?: string;
    // End of list of possible types for union field.
    mimeType?: string;
  };
  video?: {
    // Union field can only be one of the following:
    bytesBase64Encoded?: string;
    gcsUri?: string;
    // End of list of possible types for union field.
    videoSegmentConfig?: {
      startOffsetSec: number;
      endOffsetSec: number;
      intervalSec: number;
    };
  };
  parameters?: {
    dimension: number;
  };
}

interface VideoEmbedding {
  startOffsetSec: number;
  endOffsetSec: number;
  embedding: number[];
}

interface MultimodalEmbeddingPrediction {
  textEmbedding?: number[];
  imageEmbedding?: number[];
  videoEmbeddings?: VideoEmbedding[];
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isMultimodalEmbeddingPrediction(
  value: unknown
): value is MultimodalEmbeddingPrediction {
  if (!isObject(value)) {
    return false;
  }
  if (!value.textEmbedding && !value.imageEmbedding && !value.videoEmbeddings) {
    return false;
  }
  if (value.textEmbedding && !Array.isArray(value.textEmbedding)) {
    return false;
  }
  if (value.imageEmbedding && !Array.isArray(value.imageEmbedding)) {
    return false;
  }
  if (value.videoEmbeddings && !Array.isArray(value.videoEmbeddings)) {
    return false;
  }
  if (value.videoEmbeddings) {
    for (const emb of value.videoEmbeddings as Array<unknown>) {
      if (!isObject(emb)) {
        return false;
      }
      if (!emb.embedding || !Array.isArray(emb.embedding)) {
        return false;
      }
    }
  }

  return true;
}

// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#request_body
interface TextEmbeddingInstance {
  task_type?: TaskType;
  content: string;
  title?: string;
}

interface TextEmbeddingPrediction {
  embeddings: {
    statistics: {
      truncated: boolean;
      token_count: number;
    };
    values: number[];
  };
}

type EmbeddingInstance = TextEmbeddingInstance | MultimodalEmbeddingInstance;
type EmbeddingPrediction =
  | TextEmbeddingPrediction
  | MultimodalEmbeddingPrediction;

function isMultiModal(embedder: EmbedderReference): boolean {
  const input = embedder.info?.supports?.input || '';
  return (input.includes('text') && input.includes('image')) || false;
}

/**
 * Determines if a document is valid for a particular embedder or not.
 * This is only used for multimodal embedders.
 * @param embedder the embedder name e.g. 'vertexai/multimodalembedding@001'
 * @param doc The document to check
 */
function checkValidDocument(
  embedder: EmbedderReference,
  doc: Document
): boolean {
  const isTextOnly = doc.text && doc.media.length == 0;
  const isSingleMediaOnly = !doc.text && doc.media.length == 1;
  if (isMultiModal(embedder)) {
    if (embedder.name == 'vertexai/multimodalembedding@001') {
      // We restrict which Document structure can be sent for this embedder because
      // while it could accept multiple text and image and video parts in a single
      // Document, it would return separate embeddings for each of those parts,
      // essentially just batching them. This is not consistent with our "one
      // embedding per Document" design. Since the same batching can be achieved by
      // sending multiple Documents with one part each, there seems to be no reason
      // to change the design.

      if (!isTextOnly && !isSingleMediaOnly) {
        throw new Error(
          'Documents for multimodalembedding@001 must be either only text or a single media part.'
        );
      }
      return true;
    }
    return false;
  } else {
    // Not multimodal - unexpected usage.
    // Currently text-only embedders just ignore media.
    throw new Error('Not implemented');
  }
}

type EmbeddingResult = {
  embedding: number[];
  metadata?: Record<string, unknown>;
};

/** @deprecated */
export function defineVertexAIEmbedder(
  ai: Genkit,
  name: string,
  client: GoogleAuth,
  options: PluginOptions
): EmbedderAction<any> {
  const embedder =
    SUPPORTED_EMBEDDER_MODELS[name] ??
    embedderRef({
      name: `vertexai/${name}`,
      configSchema: VertexEmbeddingConfigSchema,
      info: {
        dimensions: 768,
        label: `Vertex AI - ${name}`,
        supports: {
          input: ['text', 'image', 'video'],
        },
      },
    });
  const predictClients: Record<
    string,
    PredictClient<EmbeddingInstance, EmbeddingPrediction>
  > = {};
  const predictClientFactory = (
    config: VertexEmbeddingConfig
  ): PredictClient<EmbeddingInstance, EmbeddingPrediction> => {
    const requestLocation = config?.location || options.location;
    if (!predictClients[requestLocation]) {
      // TODO: Figure out how to allow different versions while still
      // sharing a single implementation.
      predictClients[requestLocation] = predictModel<
        EmbeddingInstance,
        EmbeddingPrediction
      >(
        client,
        {
          ...options,
          location: requestLocation,
        },
        name
      );
    }
    return predictClients[requestLocation];
  };

  return ai.defineEmbedder(
    {
      name: embedder.name,
      configSchema: embedder.configSchema,
      info: embedder.info!,
    },
    async (input, options) => {
      const predictClient = predictClientFactory(options);
      const response = await predictClient(
        input.map((doc: Document) => {
          let instance: EmbeddingInstance;
          if (isMultiModal(embedder) && checkValidDocument(embedder, doc)) {
            instance = {};
            if (doc.text) {
              instance.text = doc.text;
            }
            for (var media of doc.media) {
              if (
                isObject(media) &&
                typeof media.url === 'string' &&
                typeof media.contentType === 'string'
              ) {
                if (media.contentType?.startsWith('image/')) {
                  if (
                    media.url.startsWith('http') ||
                    media.url.startsWith('gs://')
                  ) {
                    instance.image = {
                      gcsUri: media.url,
                      mimeType: media.contentType,
                    };
                  } else {
                    instance.image = {
                      bytesBase64Encoded: media.url,
                      mimeType: media.contentType,
                    };
                  }
                } else if (media.contentType.startsWith('video/')) {
                  if (
                    media.url.startsWith('http') ||
                    media.url.startsWith('gs://')
                  ) {
                    instance.video = {
                      gcsUri: media.url,
                    };
                  } else {
                    instance.video = {
                      bytesBase64Encoded: media.url,
                    };
                  }
                  if (
                    instance.video &&
                    doc.metadata &&
                    doc.metadata.videoSegmentConfig
                  ) {
                    instance.video.videoSegmentConfig =
                      doc.metadata.videoSegmentConfig;
                  }
                } else {
                  throw new Error(
                    `Unsupported contentType: '${media.contentType}`
                  );
                }
              } else {
                // It needs to be a {url:string, contentType:string} object.
                throw new Error('Invalid media specified.');
              }
            }
          } else {
            // Text only embedder
            instance = {
              content: doc.text,
              task_type: options?.taskType,
              title: options?.title,
            };
          }
          return instance;
        }),
        { outputDimensionality: options?.outputDimensionality }
      );
      return {
        embeddings: response.predictions
          .map((p: EmbeddingPrediction) => {
            if (isMultimodalEmbeddingPrediction(p)) {
              const eArray: EmbeddingResult[] = [];
              if (p.imageEmbedding?.length) {
                const imageResult: EmbeddingResult = {
                  embedding: p.imageEmbedding,
                  metadata: { embedType: 'imageEmbedding' },
                };
                eArray.push(imageResult);
              }
              if (p.textEmbedding?.length) {
                const textResult: EmbeddingResult = {
                  embedding: p.textEmbedding,
                  metadata: { embedType: 'textEmbedding' },
                };
                eArray.push(textResult);
              }
              if (p.videoEmbeddings?.length) {
                for (const ve of p.videoEmbeddings) {
                  if (ve.embedding?.length) {
                    const { embedding, ...metadata } = ve;
                    (metadata as Record<string, unknown>).embedType =
                      'videoEmbedding';
                    const videoResult: EmbeddingResult = {
                      embedding,
                      metadata,
                    };
                    eArray.push(videoResult);
                  }
                }
              }
              return eArray;
            } else {
              return [
                {
                  embedding: p.embeddings.values,
                },
              ];
            }
          })
          .reduce((accumulator, value) => {
            return accumulator.concat(value);
          }, []),
      };
    }
  );
}
