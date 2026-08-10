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

import { z, type Genkit, type ModelReference } from 'genkit';
import { modelRef, type GenerateRequest, type ModelAction } from 'genkit/model';
import type { GoogleAuth } from 'google-auth-library';
import OpenAI from 'openai';
import { getGenkitClientHeader } from '../../common/index.js';
import {
  OpenAIConfigSchema,
  openaiCompatibleModel,
} from './openai_compatibility.js';

/** @deprecated */
export const ModelGardenModelConfigSchema = OpenAIConfigSchema.extend({
  location: z.string().optional(),
});

/** @deprecated */
export const llama31 = modelRef({
  name: 'vertexai/llama-3.1',
  info: {
    label: 'Llama 3.1',
    supports: {
      multiturn: true,
      tools: true,
      media: false,
      systemRole: true,
      output: ['text', 'json'],
    },
    versions: [
      'meta/llama3-405b-instruct-maas',
      // 8b and 70b versions are coming soon
    ],
  },
  configSchema: ModelGardenModelConfigSchema,
  version: 'meta/llama3-405b-instruct-maas',
}) as ModelReference<typeof ModelGardenModelConfigSchema>;

/** @deprecated */
export const llama32 = modelRef({
  name: 'vertexai/llama-3.2',
  info: {
    label: 'Llama 3.2',
    supports: {
      multiturn: true,
      tools: true,
      media: true,
      systemRole: true,
      output: ['text', 'json'],
    },
    versions: ['meta/llama-3.2-90b-vision-instruct-maas'],
  },
  configSchema: ModelGardenModelConfigSchema,
  version: 'meta/llama-3.2-90b-vision-instruct-maas',
}) as ModelReference<typeof ModelGardenModelConfigSchema>;

/**
 * @deprecated use `llama31` instead
 */
export const llama3 = modelRef({
  name: 'vertexai/llama3-405b',
  info: {
    label: 'Llama 3.1 405b',
    supports: {
      multiturn: true,
      tools: true,
      media: false,
      systemRole: true,
      output: ['text'],
    },
    versions: ['meta/llama3-405b-instruct-maas'],
  },
  configSchema: ModelGardenModelConfigSchema,
  version: 'meta/llama3-405b-instruct-maas',
}) as ModelReference<typeof ModelGardenModelConfigSchema>;

/** @deprecated */
export const SUPPORTED_OPENAI_FORMAT_MODELS = {
  'llama3-405b': llama3,
  'llama-3.1': llama31,
  'llama-3.2': llama32,
};

/** @deprecated */
export function modelGardenOpenaiCompatibleModel(
  ai: Genkit,
  name: string,
  projectId: string,
  location: string,
  googleAuth: GoogleAuth,
  baseUrlTemplate: string | undefined
): ModelAction<typeof ModelGardenModelConfigSchema> {
  const model = SUPPORTED_OPENAI_FORMAT_MODELS[name];
  if (!model) throw new Error(`Unsupported model: ${name}`);
  if (!baseUrlTemplate) {
    baseUrlTemplate =
      'https://{location}-aiplatform.googleapis.com/v1beta1/projects/{projectId}/locations/{location}/endpoints/openapi';
  }

  const clientFactory = async (
    request: GenerateRequest<typeof ModelGardenModelConfigSchema>
  ): Promise<OpenAI> => {
    const requestLocation = request.config?.location || location;
    return new OpenAI({
      baseURL: baseUrlTemplate!
        .replace(/{location}/g, requestLocation)
        .replace(/{projectId}/g, projectId),
      apiKey: (await googleAuth.getAccessToken())!,
      defaultHeaders: {
        'X-Goog-Api-Client': getGenkitClientHeader(),
      },
    });
  };
  return openaiCompatibleModel(ai, model, clientFactory);
}
