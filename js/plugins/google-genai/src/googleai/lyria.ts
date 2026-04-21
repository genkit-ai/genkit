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

import { ActionMetadata, modelActionMetadata, z } from 'genkit';
import { ModelAction, ModelInfo, ModelReference, modelRef } from 'genkit/model';
import { model as pluginModel } from 'genkit/plugin';
import {
  ensureToolIds,
  fromInteractionSync,
  toInteractionTurn,
} from '../common/interaction-converters.js';
import { ResponseModality } from '../common/interaction-types.js';
import { isKnownKey } from '../common/utils.js';
import { createInteraction } from './client.js';
import {
  ClientOptions,
  CreateInteractionRequest,
  GoogleAIPluginOptions,
  Model,
} from './types.js';
import {
  calculateApiKey,
  checkApiKey,
  checkModelName,
  extractVersion,
  modelName,
} from './utils.js';

export const LyriaConfigSchema = z
  .object({
    apiKey: z
      .string()
      .describe('Override the API key provided at plugin initialization.')
      .optional(),
    baseUrl: z
      .string()
      .describe(
        'Overrides the plugin-configured or default baseUrl, if specified.'
      )
      .optional(),
    apiVersion: z
      .string()
      .describe(
        'Overrides the plugin-configured or default apiVersion, if specified.'
      )
      .optional(),
    responseModalities: z
      .array(z.enum(['TEXT', 'IMAGE', 'AUDIO']))
      .describe(
        'The modalities to be used in response. Defaults to AUDIO and TEXT for Lyria.'
      )
      .optional(),
  })
  .passthrough();

export type LyriaConfigSchemaType = typeof LyriaConfigSchema;
export type LyriaConfig = z.infer<LyriaConfigSchemaType>;

type ConfigSchemaType = LyriaConfigSchemaType;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = LyriaConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `googleai/${name}`,
    configSchema,
    info:
      info ??
      ({
        supports: {
          multiturn: false,
          media: true, // Lyria supports images as multimodal input
          tools: false,
          toolChoice: false,
          systemRole: false,
          output: ['media', 'text'],
        },
      } as ModelInfo),
  });
}

const GENERIC_MODEL = commonRef('lyria-3');

const KNOWN_MODELS = {
  'lyria-3-clip-preview': commonRef('lyria-3-clip-preview'),
  'lyria-3-pro-preview': commonRef('lyria-3-pro-preview'),
} as const;

export type KnownModels = keyof typeof KNOWN_MODELS;

export type LyriaModelName = `lyria-${string}`;
export function isLyriaModelName(value?: string): value is LyriaModelName {
  return !!value?.startsWith('lyria-');
}

export function model(
  version: string,
  config: LyriaConfig = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, KNOWN_MODELS)) {
    return KNOWN_MODELS[name].withConfig(config);
  }

  return modelRef({
    name: `googleai/${name}`,
    config,
    configSchema: LyriaConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  return models
    .filter((m) => isLyriaModelName(modelName(m.name)))
    .filter((m) => !m.description || !m.description.includes('deprecated'))
    .map((m) => {
      const ref = model(m.name);
      return modelActionMetadata({
        name: ref.name,
        info: ref.info,
        configSchema: ref.configSchema,
      });
    });
}

export function listKnownModels(options?: GoogleAIPluginOptions) {
  return Object.keys(KNOWN_MODELS).map((name: string) =>
    defineModel(name, options)
  );
}

export function defineModel(
  name: string,
  pluginOptions?: GoogleAIPluginOptions
): ModelAction<LyriaConfigSchemaType> {
  checkApiKey(pluginOptions?.apiKey);
  const ref = model(name);
  const clientOptions: ClientOptions = {
    apiVersion: pluginOptions?.apiVersion,
    baseUrl: pluginOptions?.baseUrl,
    customHeaders: pluginOptions?.customHeaders,
  };

  return pluginModel(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
    },
    async (request) => {
      const {
        apiKey: apiKeyConfig,
        baseUrl,
        apiVersion,
        responseModalities,
        ...rest
      } = request.config || {};

      const apiKey = calculateApiKey(pluginOptions?.apiKey, apiKeyConfig);
      const newClientOptions: ClientOptions = {
        ...clientOptions,
        apiKey,
        baseUrl: baseUrl || clientOptions.baseUrl,
        apiVersion: apiVersion || clientOptions.apiVersion,
      };

      const messages = structuredClone(request.messages);

      let responseModalitiesConverted: ResponseModality[] = ['audio', 'text'];
      if (responseModalities) {
        responseModalitiesConverted = responseModalities.map(
          (m) => m.toLowerCase() as ResponseModality
        );
      }

      const req: CreateInteractionRequest = {
        model: extractVersion(ref),
        input: ensureToolIds(messages).map(toInteractionTurn),
        response_modalities: responseModalitiesConverted,
        ...rest,
      };

      const response = await createInteraction(apiKey, req, newClientOptions);

      return fromInteractionSync(response);
    }
  );
}

export const TEST_ONLY = { KNOWN_MODELS };
