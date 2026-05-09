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

import { z, type Document } from 'genkit';
import {
  EmbedderInfo,
  embedderRef,
  type EmbedderAction,
  type EmbedderReference,
} from 'genkit/embedder';
import { embedder as pluginEmbedder } from 'genkit/plugin';
import { isKnownKey } from '../common/utils.js';
import { embedContent } from './client.js';
import {
  ClientOptions,
  EmbedContentRequest,
  EmbeddingInstance,
  EmbeddingPrediction,
  EmbeddingResult,
  TaskTypeSchema,
  VertexPluginOptions,
  isMultimodalEmbeddingPrediction,
  isObject,
} from './types.js';
import { checkModelName, extractVersion } from './utils.js';

export const EmbeddingConfigSchema = z
  .object({
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
     * By default, the model generates embeddings with 768 dimensions.
     * By selecting a smaller output dimensionality, users can save memory and storage space, leading to more efficient computations.
     **/
    outputDimensionality: z.number().min(1).optional(),
    /**
     * For newly released embedders this parameter provides a hint for the proper
     * way to call the embedder. (Multimodal embedders have a different request
     * structure than non-multimodal embedders).
     * For well-known embedders, this value will be ignored since we will already
     * know if it's multimodal or not.
     */
    multimodal: z.boolean().optional(),
  })
  .passthrough();
export type EmbeddingConfigSchemaType = typeof EmbeddingConfigSchema;
export type EmbeddingConfig = z.infer<EmbeddingConfigSchemaType>;

// for commonRef
type ConfigSchemaType = EmbeddingConfigSchemaType;

function commonRef(
  name: string,
  info?: EmbedderInfo,
  configSchema: ConfigSchemaType = EmbeddingConfigSchema
): EmbedderReference<ConfigSchemaType> {
  return embedderRef({
    name: `vertexai/${name}`,
    configSchema,
    info: info ?? {
      dimensions: 768,
      supports: {
        input: ['text'],
      },
    },
  });
}

const GENERIC_TEXT_MODEL = commonRef('text', {
  dimensions: 3072,
  supports: { input: ['text'] },
});
const GENERIC_MULTIMODAL_MODEL = commonRef('multimodal', {
  dimensions: 768,
  supports: { input: ['text', 'image', 'video'] },
});

export const KNOWN_MODELS = {
  'text-embedding-005': commonRef('text-embedding-005'),
  'text-multilingual-embedding-002': commonRef(
    'text-multilingual-embedding-002'
  ),
  'multimodalembedding@001': commonRef('multimodalembedding@001', {
    dimensions: 768,
    supports: { input: ['text', 'image', 'video'] },
  }),
  'gemini-embedding-001': commonRef('gemini-embedding-001', {
    dimensions: 3072,
    supports: { input: ['text'] },
  }),
} as const;
export type KnownModels = keyof typeof KNOWN_MODELS; // For autocomplete

export type EmbedderModelName = `${string}embedding${string}`;
export function isEmbedderName(value: string): value is EmbedderModelName {
  return value.includes('embedding');
}

export function model(
  version: string,
  config: EmbeddingConfig = {}
): EmbedderReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, KNOWN_MODELS)) {
    const known = KNOWN_MODELS[name];
    return embedderRef({
      name: known.name,
      info: known.info,
      configSchema: known.configSchema,
      config,
    });
  }

  if (config.multimodal) {
    // Generic multimodal embedder format
    return embedderRef({
      name: `vertexai/${name}`,
      configSchema: EmbeddingConfigSchema,
      config,
      info: {
        ...GENERIC_MULTIMODAL_MODEL.info,
      },
    });
  }
  // Generic text-only embedder format
  return embedderRef({
    name: `vertexai/${name}`,
    configSchema: EmbeddingConfigSchema,
    config,
    info: {
      ...GENERIC_TEXT_MODEL.info,
    },
  });
}

export function listKnownModels(
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
) {
  return Object.keys(KNOWN_MODELS).map((name) =>
    defineEmbedder(name, clientOptions, pluginOptions)
  );
}

export function defineEmbedder(
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
): EmbedderAction<any> {
  const ref = model(name);

  return pluginEmbedder(
    {
      name: ref.name,
      configSchema: ref.configSchema,
      info: ref.info!,
    },
    async (request) => {
      const embedContentRequest: EmbedContentRequest = {
        instances: request.input.map((doc: Document) =>
          toEmbeddingInstance(ref, doc, request.options)
        ),
        parameters: {
          outputDimensionality: request.options?.outputDimensionality,
        },
      };

      const response = await embedContent(
        extractVersion(ref),
        embedContentRequest,
        clientOptions
      );

      return {
        embeddings: response.predictions
          .map(toEmbeddingResult)
          .reduce((accumulator, value) => {
            return accumulator.concat(value);
          }, []),
      };
    }
  );
}

function toEmbeddingInstance(
  embedder: EmbedderReference<ConfigSchemaType>,
  doc: Document,
  options?: EmbeddingConfig
): EmbeddingInstance {
  let instance: EmbeddingInstance;
  if (
    isMultiModalEmbedder(embedder) ||
    embedder.config?.multimodal ||
    options?.multimodal
  ) {
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
          if (media.url.startsWith('http') || media.url.startsWith('gs://')) {
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
          if (media.url.startsWith('http') || media.url.startsWith('gs://')) {
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
            instance.video.videoSegmentConfig = doc.metadata.videoSegmentConfig;
          }
        } else {
          throw new Error(`Unsupported contentType: '${media.contentType}`);
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
}

/**
 * Converts an `EmbeddingPrediction` object to an array of `EmbeddingResult` objects.
 *
 * There will only be multiple EmbeddingResult objects in the array if it is a
 * multimodal embedding prediction for a video.
 * A single video gets automatically broken into chunks and an embedding is
 * returned for each chunk. The metadata identifies which chunk of the video
 * it is for.
 *
 * @param prediction The input `EmbeddingPrediction` object.
 * @returns An array of `EmbeddingResult` objects, each representing a different embedding.
 */
function toEmbeddingResult(prediction: EmbeddingPrediction): EmbeddingResult[] {
  if (isMultimodalEmbeddingPrediction(prediction)) {
    const eArray: EmbeddingResult[] = [];
    if (prediction.imageEmbedding?.length) {
      const imageResult: EmbeddingResult = {
        embedding: prediction.imageEmbedding,
        metadata: { embedType: 'imageEmbedding' },
      };
      eArray.push(imageResult);
    }
    if (prediction.textEmbedding?.length) {
      const textResult: EmbeddingResult = {
        embedding: prediction.textEmbedding,
        metadata: { embedType: 'textEmbedding' },
      };
      eArray.push(textResult);
    }
    if (prediction.videoEmbeddings?.length) {
      for (const ve of prediction.videoEmbeddings) {
        if (ve.embedding?.length) {
          const { embedding, ...metadata } = ve;
          (metadata as Record<string, unknown>).embedType = 'videoEmbedding';
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
    // Text-only embedding
    return [
      {
        embedding: prediction.embeddings.values,
      },
    ];
  }
}

function isMultiModalEmbedder(
  embedder: EmbedderReference<ConfigSchemaType>
): boolean {
  if (embedder.config?.multimodal) {
    return true;
  }
  const input = embedder.info?.supports?.input || '';
  return (input.includes('text') && input.includes('image')) || false;
}

export const TEST_ONLY = { KNOWN_MODELS };
