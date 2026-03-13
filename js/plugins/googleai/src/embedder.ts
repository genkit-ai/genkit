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
  GoogleGenerativeAI,
  type EmbedContentRequest,
} from '@google/generative-ai';
import {
  GenkitError,
  z,
  type EmbedderAction,
  type EmbedderReference,
  type Genkit,
} from 'genkit';
import { embedderRef } from 'genkit/embedder';
import { getApiKeyFromEnvVar } from './common.js';
import type { PluginOptions } from './index.js';

/**
 * @deprecated
 */
export const TaskTypeSchema = z.enum([
  'RETRIEVAL_DOCUMENT',
  'RETRIEVAL_QUERY',
  'SEMANTIC_SIMILARITY',
  'CLASSIFICATION',
  'CLUSTERING',
]);
/**
 * @deprecated
 */
export type TaskType = z.infer<typeof TaskTypeSchema>;

/**
 * @deprecated
 */
export const GeminiEmbeddingConfigSchema = z.object({
  /** Override the API key provided at plugin initialization. */
  apiKey: z.string().optional(),
  /**
   * The `task_type` parameter is defined as the intended downstream application to help the model
   * produce better quality embeddings.
   **/
  taskType: TaskTypeSchema.optional(),
  title: z.string().optional(),
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

/**
 * @deprecated
 */
export type GeminiEmbeddingConfig = z.infer<typeof GeminiEmbeddingConfigSchema>;

/**
 * @deprecated
 */
export const textEmbeddingGecko001 = embedderRef({
  name: 'googleai/embedding-001',
  configSchema: GeminiEmbeddingConfigSchema,
  info: {
    dimensions: 768,
    label: 'Google Gen AI - Text Embedding Gecko (Legacy)',
    supports: {
      input: ['text'],
    },
  },
});

/**
 * @deprecated
 */
export const textEmbedding004 = embedderRef({
  name: 'googleai/text-embedding-004',
  configSchema: GeminiEmbeddingConfigSchema,
  info: {
    dimensions: 768,
    label: 'Google Gen AI - Text Embedding 001',
    supports: {
      input: ['text'],
    },
  },
});

/**
 * @deprecated
 */
export const geminiEmbedding001 = embedderRef({
  name: 'googleai/gemini-embedding-001',
  configSchema: GeminiEmbeddingConfigSchema,
  info: {
    dimensions: 768,
    label: 'Google Gen AI - Gemini Embedding 001',
    supports: {
      input: ['text'],
    },
  },
});

/**
 * @deprecated
 */
export const SUPPORTED_MODELS = {
  'embedding-001': textEmbeddingGecko001,
  'text-embedding-004': textEmbedding004,
  'gemini-embedding-001': geminiEmbedding001,
};

/**
 * @deprecated
 */
export function defineGoogleAIEmbedder(
  ai: Genkit,
  name: string,
  pluginOptions: PluginOptions
): EmbedderAction<any> {
  let apiKey: string | undefined;
  // DO NOT throw if {apiKey: false} was supplied to options.
  if (pluginOptions.apiKey !== false) {
    apiKey = pluginOptions?.apiKey || getApiKeyFromEnvVar();
    if (!apiKey)
      throw new Error(
        'Please pass in the API key or set either GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n' +
          'For more details see https://genkit.dev/docs/plugins/google-genai'
      );
  }
  const embedder: EmbedderReference =
    SUPPORTED_MODELS[name] ??
    embedderRef({
      name: name,
      configSchema: GeminiEmbeddingConfigSchema,
      info: {
        dimensions: 768,
        label: `Google AI - ${name}`,
        supports: {
          input: ['text', 'image', 'video'],
        },
      },
    });
  const apiModelName = embedder.name.startsWith('googleai/')
    ? embedder.name.substring('googleai/'.length)
    : embedder.name;
  return ai.defineEmbedder(
    {
      name: embedder.name,
      configSchema: GeminiEmbeddingConfigSchema,
      info: embedder.info!,
    },
    async (input, options) => {
      if (pluginOptions.apiKey === false && !options?.apiKey) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message:
            'GoogleAI plugin was initialized with {apiKey: false} but no apiKey configuration was passed at call time.',
        });
      }
      const client = new GoogleGenerativeAI(
        options?.apiKey || apiKey!
      ).getGenerativeModel({
        model:
          options?.version ||
          embedder.config?.version ||
          embedder.version ||
          apiModelName,
      });
      const embeddings = await Promise.all(
        input.map(async (doc) => {
          const response = await client.embedContent({
            taskType: options?.taskType,
            title: options?.title,
            content: {
              role: '',
              parts: [{ text: doc.text }],
            },
            outputDimensionality: options?.outputDimensionality,
          } as EmbedContentRequest);
          const values = response.embedding.values;
          return { embedding: values };
        })
      );
      return { embeddings };
    }
  );
}
