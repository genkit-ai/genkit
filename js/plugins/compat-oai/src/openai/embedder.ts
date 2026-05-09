/**
 * Copyright 2024 The Fire Company
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

// import { defineEmbedder, embedderRef } from '@genkit-ai/ai/embedder';

import { EmbedderInfo, embedderRef, EmbedderReference, z } from 'genkit';

export const TextEmbeddingConfigSchema = z.object({
  dimensions: z.number().optional(),
  encodingFormat: z.union([z.literal('float'), z.literal('base64')]).optional(),
});

const COMMON_EMBEDDER_INFO: EmbedderInfo = {
  dimensions: 1536,
  supports: {
    input: ['text'],
  },
};

function openAIEmbedderRef(
  name: string,
  info?: EmbedderInfo
): EmbedderReference<typeof TextEmbeddingConfigSchema> {
  return embedderRef({
    name,
    configSchema: TextEmbeddingConfigSchema,
    info: {
      ...COMMON_EMBEDDER_INFO,
      ...(info ?? {}),
    },
    namespace: 'openai',
  });
}

export const SUPPORTED_EMBEDDING_MODELS = {
  'text-embedding-3-small': openAIEmbedderRef('text-embedding-3-small'),
  'text-embedding-3-large': openAIEmbedderRef('text-embedding-3-large', {
    dimensions: 3072,
  }),
  'text-embedding-ada-002': openAIEmbedderRef('text-embedding-ada-002'),
};
