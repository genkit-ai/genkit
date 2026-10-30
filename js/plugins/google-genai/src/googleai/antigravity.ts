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
import {
  ensureToolIds,
  fromInteractionSync,
  toInteractionConfigTool,
  toInteractionResponseFormat,
  toInteractionSteps,
  toInteractionTool,
} from '../common/interaction-converters.js';
import { InteractionTool } from '../common/interaction-types.js';
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
    model: z
      .union([
        z.enum([
          'gemini-4-argon',
          'gemini-3.8-flash',
          'gemini-3.6-flash',
          'gemini-3.5-flash-lite',
        ]),
        // Allows newer models. `string & {}` (rather than z.string()) keeps
        // the listed names in TypeScript autocomplete: a plain `string` in the
        // union would absorb the literals.
        z.custom<string & {}>((val) => typeof val === 'string'),
      ])
      .describe(
        'The Gemini model the agent runs on. The server picks a default ' +
          'if not set.'
      )
      .optional(),
  })
  .passthrough();

export type AntigravityConfigSchemaType = typeof AntigravityConfigSchema;
export type AntigravityConfig = z.infer<AntigravityConfigSchemaType>;

type ConfigSchemaType = AntigravityConfigSchemaType;

function commonRef(
  name: string,
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
          tools: true,
          toolChoice: false,
          systemRole: false,
          output: ['text'],
        },
      } as ModelInfo),
  });
}

const GENERIC_MODEL = commonRef('antigravity');

const KNOWN_MODELS = {
  'antigravity-preview-09-2026': commonRef('antigravity-preview-09-2026'),
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
        // Sent as agent_config.model; a top-level `model` would conflict with
        // `agent`.
        model: agentModel,
        tools: toolsFromConfig,
        ...rest
      } = requestOptions;

      // Built-in tools from config.tools (e.g. { type: 'google_search' }),
      // followed by the request's function tools.
      const tools: InteractionTool[] = [
        ...(Array.isArray(toolsFromConfig)
          ? toolsFromConfig.map(toInteractionConfigTool)
          : []),
        ...(request.tools ?? []).map(toInteractionTool),
      ];

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

      // Unlike Gemini models, Antigravity doesn't default to `store: false`:
      // the agent doesn't work at all with `store: false` (the server rejects
      // it), so that default would break every existing Antigravity app.
      // `store` is only sent when set; an explicit `store: false` (request
      // config, or else plugin options) is passed through as requested.
      const storeSetting =
        store ?? (pluginOptions?.store === false ? false : undefined);

      // Continue from the last model message: send its interaction ID and only
      // the messages after it, in the same environment (a new 'remote' sandbox
      // wouldn't have the previous turn's files). The agent also rejects tool
      // calls in `input`, so this is how a tool result is sent.
      // The message metadata doesn't record which model or agent created the
      // interaction, so a history that mixes stored Gemini turns and
      // Antigravity turns would continue the wrong interaction here (and the
      // Gemini path has the same issue in reverse).
      // TODO: record the creating model/agent in the metadata and check it.
      let previousId = previousInteractionId;
      let newMessages = messages;
      let environmentToUse = environment;
      if (!previousId) {
        for (let i = messages.length - 1; i >= 0; i--) {
          const { role, metadata } = messages[i];
          const prevId = metadata?.interactionId;
          if (role === 'model' && typeof prevId === 'string' && prevId) {
            previousId = prevId;
            newMessages = messages.slice(i + 1);
            const prevEnvironment = metadata?.environmentId;
            if (
              (environment === undefined || environment === 'remote') &&
              typeof prevEnvironment === 'string' &&
              prevEnvironment
            ) {
              environmentToUse = prevEnvironment;
            }
            break;
          }
        }
      } else {
        // As for Gemini models: an explicit previousInteractionId wins, but
        // history up to the last model turn is already on the server.
        for (let i = messages.length - 1; i >= 0; i--) {
          if (messages[i].role === 'model') {
            newMessages = messages.slice(i + 1);
            break;
          }
        }
      }

      const req: CreateInteractionRequest = {
        agent: extractVersion(ref),
        input: toInteractionSteps(ensureToolIds(newMessages)),
        previous_interaction_id: previousId,
        store: storeSetting,
        environment: environmentToUse,
        response_format: toInteractionResponseFormat({
          modalities: responseModalities,
        }),
        ...(agentModel
          ? { agent_config: { type: 'antigravity', model: agentModel } }
          : {}),
        ...(tools.length > 0 ? { tools } : {}),
        ...rest,
      };

      const response = await createInteraction(apiKey, req, clientOpt);

      return fromInteractionSync(response);
    }
  );
}

export const TEST_ONLY = { KNOWN_MODELS };
