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

import { ActionMetadata, modelActionMetadata, z } from 'genkit';
import { ModelAction, ModelInfo, ModelReference, modelRef } from 'genkit/model';
import { model as pluginModel } from 'genkit/plugin';
import { isKnownKey } from '../common/utils.js';
import { createInteraction } from './client.js';
import {
  ensureToolIds,
  fromInteractionSync,
  toInteractionSteps,
  toInteractionTool,
} from './interaction-converters.js';
import { InteractionTool, ResponseModality } from './interaction-types.js';
import {
  ClientOptions,
  CreateInteractionRequest,
  GoogleAIPluginOptions,
  Model,
} from './types.js';
import {
  calculateApiKey,
  calculateRequestOptions,
  checkApiKey,
  checkModelName,
  extractVersion,
  modelName,
  removeClientOptionOverrides,
} from './utils.js';

export const AntigravityConfigSchema = z
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
    previousInteractionId: z
      .string()
      .describe('The ID of the previous interaction, if any.')
      .optional(),
    store: z
      .boolean()
      .describe(
        'Whether to store the response and request for later retrieval.'
      )
      .optional(),
    environment: z
      .union([z.string(), z.record(z.unknown())])
      .describe('The environment configuration for the sandbox.')
      .optional(),
    responseModalities: z
      .array(z.enum(['TEXT', 'IMAGE']))
      .describe('The modalities to be used in response.')
      .optional(),
    tools: z
      .array(z.record(z.unknown()))
      .describe(
        'Raw Interactions tools, e.g. `{ type: "google_search" }`. Genkit ' +
          'tools passed to generate() are appended after these. When any ' +
          'tools are sent, the agent may be limited to exactly that list ' +
          'rather than its default tools (code_execution, google_search, ' +
          'url_context), so list the built-ins you want to keep.'
      )
      .optional(),
  })
  .passthrough();

export type AntigravityConfigSchemaType = typeof AntigravityConfigSchema;
export type AntigravityConfig = z.infer<AntigravityConfigSchemaType>;

type ConfigSchemaType = AntigravityConfigSchemaType;

function commonRef(
  name: string,
  functionCalling = false,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = AntigravityConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `googleai/${name}`,
    configSchema,
    info:
      info ??
      ({
        supports: {
          multiturn: true,
          media: true,
          tools: functionCalling,
          toolChoice: false,
          systemRole: false,
          output: ['text'],
        },
      } as ModelInfo),
  });
}

const GENERIC_MODEL = commonRef('antigravity');

// Function calling is documented for the 09-2026 preview only; it is
// unverified for 05-2026 and for unknown versions.
const KNOWN_MODELS = {
  'antigravity-preview-05-2026': commonRef('antigravity-preview-05-2026'),
  'antigravity-preview-09-2026': commonRef('antigravity-preview-09-2026', true),
} as const;

export type KnownModels = keyof typeof KNOWN_MODELS;

export type AntigravityModelName = `antigravity-${string}`;
export function isAntigravityModelName(
  value?: string
): value is AntigravityModelName {
  return !!value?.startsWith('antigravity-');
}

export function model(
  version: string,
  config: AntigravityConfig = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, KNOWN_MODELS)) {
    return KNOWN_MODELS[name].withConfig(config);
  }

  return modelRef({
    name: `googleai/${name}`,
    config,
    configSchema: AntigravityConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  return models
    .filter((m) => isAntigravityModelName(modelName(m.name)))
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
): ModelAction<AntigravityConfigSchemaType> {
  checkApiKey(pluginOptions?.apiKey);
  const ref = model(name);
  const clientOptions: ClientOptions = {
    apiVersion: pluginOptions?.apiVersion,
    baseUrl: pluginOptions?.baseUrl,
    customHeaders: pluginOptions?.customHeaders,
    experimental_debugTraces: pluginOptions?.experimental_debugTraces,
  };

  return pluginModel(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
    },
    async (request, { abortSignal }) => {
      const clientOpt = calculateRequestOptions(
        { ...clientOptions, signal: abortSignal },
        request.config
      );

      const requestOptions = removeClientOptionOverrides(request.config || {});
      const {
        previousInteractionId,
        store,
        environment,
        responseModalities,
        tools: configTools,
        ...rest
      } = requestOptions;

      const apiKey = calculateApiKey(
        pluginOptions?.apiKey,
        request.config?.apiKey
      );

      const messages = structuredClone(request.messages);

      // Antigravity does not support system instructions, so we map them to user messages.
      for (const message of messages) {
        if (message.role === 'system') {
          message.role = 'user';
        }
      }

      let responseModalitiesConverted: ResponseModality[] | undefined;
      if (responseModalities) {
        responseModalitiesConverted = responseModalities.map(
          (m) => m.toLowerCase() as ResponseModality
        );
      }

      // Genkit function tools are appended after any raw tools from config
      // (built-ins such as google_search). When neither is set, `tools` stays
      // undefined so the agent keeps its default tool set. An explicitly
      // provided list is always forwarded, even if empty. Note that an
      // explicit list likely replaces the agent's default tools.
      const mergedTools: InteractionTool[] = [
        ...((configTools ?? []) as InteractionTool[]),
        ...(request.tools ?? []).map(toInteractionTool),
      ];

      const req: CreateInteractionRequest = {
        agent: extractVersion(ref),
        input: toInteractionSteps(ensureToolIds(messages)),
        previous_interaction_id: previousInteractionId,
        store,
        environment,
        response_modalities: responseModalitiesConverted,
        ...rest,
        ...(configTools || mergedTools.length ? { tools: mergedTools } : {}),
      };

      const response = await createInteraction(apiKey, req, clientOpt);

      return fromInteractionSync(response);
    }
  );
}

export const TEST_ONLY = { KNOWN_MODELS };
