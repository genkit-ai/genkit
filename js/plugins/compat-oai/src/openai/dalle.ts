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
import { z } from 'genkit';
import { ModelInfo } from 'genkit/model';
import {
  ImageGenerationCommonConfigSchema,
  ImageRequestBuilder,
  compatOaiImageModelRef,
} from '../image.js';

/** OpenAI image generation ModelRef helper, same as the OpenAI-compatible spec.
 * */
export function openAIImageModelRef<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  info?: ModelInfo;
  configSchema?: CustomOptions;
  config?: any;
}) {
  return compatOaiImageModelRef({ ...params, namespace: 'openai' });
}

const gptImageConfigSchema = ImageGenerationCommonConfigSchema.omit({
  response_format: true,
}).extend({
  size: z
    .enum(['1024x1024', '1536x1024', '1024x1536', 'auto'])
    .optional(),
  background: z.enum(['transparent', 'opaque', 'auto']).optional(),
  moderation: z.enum(['low', 'auto']).optional(),
  output_compression: z.number().int().min(1).max(100).optional(),
  output_format: z.enum(['png', 'jpeg', 'webp']).optional(),
  quality: z.enum(['low', 'medium', 'high', 'auto']).optional(),
});

export const SUPPORTED_IMAGE_MODELS = {
  'dall-e-3': openAIImageModelRef({ name: 'dall-e-3' }),
  'gpt-image-1': openAIImageModelRef({
    name: 'gpt-image-1',
    configSchema: gptImageConfigSchema,
  }),
  'gpt-image-1.5': openAIImageModelRef({
    name: 'gpt-image-1.5',
    configSchema: gptImageConfigSchema,
  }),
  'gpt-image-1-mini': openAIImageModelRef({
    name: 'gpt-image-1-mini',
    configSchema: gptImageConfigSchema,
  }),
  'gpt-image-2': openAIImageModelRef({
    name: 'gpt-image-2',
    configSchema: gptImageConfigSchema,
  }),
  'gpt-image-2.5-sunburst': openAIImageModelRef({
    name: 'gpt-image-2.5-sunburst',
    configSchema: gptImageConfigSchema,
  }),
  'gpt-image-2.5-flare': openAIImageModelRef({
    name: 'gpt-image-2.5-flare',
    configSchema: gptImageConfigSchema,
  }),
};

export const gptImage1RequestBuilder: ImageRequestBuilder = (req, params) => {
  const {
    background,
    moderation,
    n,
    output_compression,
    output_format,
    quality,
    size,
    style,
    user,
  } = req.config ?? {};
  // GPT Image models do not support response format
  params.response_format = undefined;
  params.background = background;
  params.moderation = moderation;
  params.n = n;
  params.output_compression = output_compression;
  params.output_format = output_format;
  params.quality = quality;
  params.size = size;
  params.style = style;
  params.user = user;
};
