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

import { EmbedderInfo, embedderRef, EmbedderReference, z } from 'genkit';
import type { EmbedderRequestBuilder } from '../embedder.js';

/** Mistral embedder configuration schema. */
export const MistralEmbeddingConfigSchema = z.object({
  outputDimension: z
    .number()
    .int()
    .optional()
    .describe(
      'Dimension of the output embeddings, for models that support it. Sent as `output_dimension`.'
    ),
  encodingFormat: z.union([z.literal('float'), z.literal('base64')]).optional(),
  apiKey: z
    .string()
    .optional()
    .describe('Overrides the plugin API key for this request.'),
});

/**
 * Mistral uses `output_dimension` rather than `dimensions`, and rejects
 * unknown fields, so only the fields it accepts are set.
 */
export const mistralEmbedderRequestBuilder: EmbedderRequestBuilder = (
  req,
  params
) => {
  const { outputDimension } = req.options ?? {};
  // Request floats unless asked otherwise. Without an explicit format the
  // OpenAI SDK asks for base64 and decodes it.
  params.encoding_format ??= 'float';
  if (outputDimension !== undefined) {
    (params as unknown as Record<string, unknown>).output_dimension =
      outputDimension;
  }
};

/** Mistral EmbedderRef helper, with Mistral specific config. */
export function mistralEmbedderRef(params: {
  name: string;
  info?: EmbedderInfo;
  config?: z.infer<typeof MistralEmbeddingConfigSchema>;
}): EmbedderReference<typeof MistralEmbeddingConfigSchema> {
  return embedderRef({
    name: params.name,
    configSchema: MistralEmbeddingConfigSchema,
    info: params.info ?? {
      supports: {
        input: ['text'],
      },
    },
    config: params.config,
    namespace: 'mistral',
  });
}

export const SUPPORTED_MISTRAL_EMBEDDERS = {
  'mistral-embed': mistralEmbedderRef({
    name: 'mistral-embed',
    info: { dimensions: 1024, supports: { input: ['text'] } },
  }),
  'codestral-embed': mistralEmbedderRef({
    name: 'codestral-embed',
    info: { dimensions: 1536, supports: { input: ['text'] } },
  }),
};
