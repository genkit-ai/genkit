/**
 * Copyright 2025 Google LLC
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

import { ActionMetadata, GenkitError, z, type ModelReference } from 'genkit';
import {
  ModelInfo,
  modelRef,
  type GenerateRequest,
  type ModelAction,
} from 'genkit/model';
import type { GoogleAuth } from 'google-auth-library';
import OpenAI from 'openai';
import { getGenkitClientHeader } from '../../common/index.js';
import {
  OpenAIConfigSchema,
  defineOpenaiCompatibleModel,
} from './openai_compatibility.js';
import { PluginOptions } from './types.js';
import { checkModelName } from './utils.js';

export const LlamaConfigSchema = OpenAIConfigSchema.extend({
  location: z.string().optional(),
}).passthrough();
export type LlamaConfigSchemaType = typeof LlamaConfigSchema;
export type LlamaConfig = z.infer<LlamaConfigSchemaType>;

type ConfigSchemaType = LlamaConfigSchemaType;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = LlamaConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertex-model-garden/${name}`,
    configSchema,
    info: info ?? {
      supports: {
        multiturn: true,
        tools: true,
        media: true,
        systemRole: true,
        output: ['text', 'json'],
      },
    },
  });
}

export const GENERIC_MODEL = commonRef('llama');

export const KNOWN_MODELS = {
  'meta/llama-4-maverick-17b-128e-instruct-maas': commonRef(
    'meta/llama-4-maverick-17b-128e-instruct-maas'
  ),
  'meta/llama-4-scout-17b-16e-instruct-maas': commonRef(
    'meta/llama-4-scout-17b-16e-instruct-maas'
  ),
  'meta/llama-3.3-70b-instruct-maas': commonRef(
    'meta/llama-3.3-70b-instruct-maas'
  ),
};
export type KnownModels = keyof typeof KNOWN_MODELS;
export type LlamaModelName = `meta/llama-${string}`;
export function isLlamaModelName(value?: string): value is LlamaModelName {
  return !!value?.startsWith('meta/llama-');
}

export function model(
  version: string,
  options: LlamaConfig = {}
): ModelReference<LlamaConfigSchemaType> {
  const name = checkModelName(version);

  return modelRef({
    name: `vertex-model-garden/${name}`,
    config: options,
    configSchema: LlamaConfigSchema,
    info: {
      ...GENERIC_MODEL.info,
    },
  });
}

export interface ClientOptions {
  location: string; // e.g. 'us-central1' or 'us-east5'
  projectId: string;
  authClient: GoogleAuth;
  baseUrlTemplate?: string;
}

export function listActions(clientOptions: ClientOptions): ActionMetadata[] {
  // TODO: figure out where to get the list of models.
  return [];
}

export function listKnownModels(
  clientOptions: ClientOptions,
  pluginOptions?: PluginOptions
) {
  return Object.keys(KNOWN_MODELS).map((name) =>
    defineModel(name, clientOptions, pluginOptions)
  );
}

export function defineModel(
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: PluginOptions
): ModelAction<LlamaConfigSchemaType> {
  const ref = model(name);
  const clientFactory = async (
    request: GenerateRequest<LlamaConfigSchemaType>
  ): Promise<OpenAI> => {
    const options = await resolveOptions(clientOptions, request.config);
    return new OpenAI(options);
  };
  return defineOpenaiCompatibleModel(ref, clientFactory);
}

async function resolveOptions(
  clientOptions: ClientOptions,
  requestConfig?: LlamaConfig
) {
  const baseUrlTemplate =
    clientOptions.baseUrlTemplate ??
    'https://{location}-aiplatform.googleapis.com/v1/projects/{projectId}/locations/{location}/endpoints/openapi';
  const location = requestConfig?.location || clientOptions.location;
  const baseURL = baseUrlTemplate
    .replace(/{location}/g, location)
    .replace(/{projectId}/g, clientOptions.projectId);
  const apiKey = await clientOptions.authClient.getAccessToken();
  if (!apiKey) {
    throw new GenkitError({
      status: 'PERMISSION_DENIED',
      message: 'Unable to get accessToken',
    });
  }
  const defaultHeaders = {
    'X-Goog-Api-Client': getGenkitClientHeader(),
  };
  return { baseURL, apiKey, defaultHeaders };
}
