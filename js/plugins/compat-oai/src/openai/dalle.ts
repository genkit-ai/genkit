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

/** Returns whether an OpenAI model name belongs to the GPT image family. */
export function isGptImageModel(name: string): boolean {
  return /^(?:openai\/)?(?:gpt|chatgpt)-image-/.test(name);
}

/** GPT image config; supported sizes and quality levels depend on the model. */
export const GPTImageConfigSchema = ImageGenerationCommonConfigSchema.omit({
  response_format: true,
  style: true,
}).extend({
  size: z
    .string()
    .regex(/^(auto|[1-9]\d*x[1-9]\d*)$/)
    .optional(),
  background: z.enum(['transparent', 'opaque', 'auto']).optional(),
  moderation: z.enum(['low', 'auto']).optional(),
  output_compression: z.number().int().min(0).max(100).optional(),
  output_format: z.enum(['png', 'jpeg', 'webp']).optional(),
  quality: z.enum(['low', 'medium', 'high', 'auto', 'xhigh', 'max']).optional(),
  n: z.number().int().min(1).max(10).optional(),
});

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
  return compatOaiImageModelRef({
    ...params,
    configSchema:
      params.configSchema ??
      (isGptImageModel(params.name)
        ? GPTImageConfigSchema
        : ImageGenerationCommonConfigSchema),
    namespace: 'openai',
  });
}

export const SUPPORTED_IMAGE_MODELS = {
  'dall-e-3': openAIImageModelRef({ name: 'dall-e-3' }),
  'gpt-image-1': openAIImageModelRef({ name: 'gpt-image-1' }),
  'gpt-image-2': openAIImageModelRef({ name: 'gpt-image-2' }),
  'gpt-image-2.5-sunburst': openAIImageModelRef({
    name: 'gpt-image-2.5-sunburst',
  }),
  'gpt-image-2.5-flare': openAIImageModelRef({ name: 'gpt-image-2.5-flare' }),
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
    user,
  } = req.config ?? {};
  // GPT image models always return base64 and do not support response_format.
  params.response_format = undefined;
  params.background = background;
  params.moderation = moderation;
  params.n = n;
  params.output_compression = output_compression;
  params.output_format = output_format;
  params.quality = quality;
  params.size = size;
  params.user = user;
};
