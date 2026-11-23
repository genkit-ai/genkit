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
import {
  BackgroundModelAction,
  GenerateRequest,
  ModelInfo,
  ModelReference,
  modelRef,
} from 'genkit/model';
import { backgroundModel as pluginBackgroundModel } from 'genkit/plugin';
import {
  ensureToolIds,
  fromInteraction,
  toInteractionConfigTool,
  toInteractionResponseFormat,
  toInteractionSteps,
  toInteractionTool,
} from '../common/interaction-converters.js';
import {
  CreateInteractionRequest,
  InteractionTool,
} from '../common/interaction-types.js';
import { cleanSchema, isKnownKey } from '../common/utils.js';
import {
  cancelInteraction,
  createInteraction,
  getInteraction,
} from './client.js';
import { ClientOptions, Model, VertexPluginOptions } from './types.js';
import {
  applyContextOverrides,
  calculateRequestOptions,
  checkModelName,
  extractVersion,
  toInteractionsClientOptions,
} from './utils.js';

export const DeepResearchConfigSchema = z
  .object({
    apiKey: z
      .string()
      .describe('Overrides the plugin-configured API key, if specified.')
      .optional(),
    apiVersion: z
      .enum(['v1', 'v1beta1'])
      .describe(
        'Overrides the plugin-configured API version to use, if specified.'
      )
      .optional(),
    location: z
      .string()
      .describe(
        'Google Cloud location. Deep Research is only available on global ' +
          '(used when no location is set).'
      )
      .optional(),
    thinkingSummaries: z
      .enum(['AUTO', 'NONE'])
      .describe('Whether to include thought summaries in the response.')
      .optional(),
    store: z
      .boolean()
      .describe(
        'Whether to store the response and request for later retrieval.'
      )
      .optional(),
    responseModalities: z
      .array(z.enum(['TEXT', 'IMAGE', 'AUDIO']))
      .describe('The modalities to be used in response.')
      .optional(),
    visualization: z
      .enum(['AUTO', 'OFF'])
      .describe('Whether to enable agent-generated charts and images.')
      .optional(),
  })
  .passthrough();
export type DeepResearchConfigSchemaType = typeof DeepResearchConfigSchema;
export type DeepResearchConfig = z.infer<DeepResearchConfigSchemaType>;

type ConfigSchemaType = DeepResearchConfigSchemaType;

// `longRunning` isn't in ModelInfo's `supports` type, but Genkit requires it
// at runtime for background models (generateOperation). Declaring `supports`
// separately keeps it without a cast (excess-property checks only apply to
// object literals written in place).
const DEEP_RESEARCH_SUPPORTS = {
  multiturn: true,
  media: true,
  tools: true,
  toolChoice: false,
  systemRole: false,
  output: ['text', 'media'],
  longRunning: true,
};
const DEEP_RESEARCH_INFO: ModelInfo = { supports: DEEP_RESEARCH_SUPPORTS };

function commonRef(
  name: string,
  info: ModelInfo = DEEP_RESEARCH_INFO,
  configSchema: ConfigSchemaType = DeepResearchConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertexai/${name}`,
    configSchema,
    info,
  });
}

const GENERIC_MODEL = commonRef('deep-research');

const KNOWN_MODELS = {
  'deep-research-preview-04-2026': commonRef('deep-research-preview-04-2026'),
} as const;
export type KnownModels = keyof typeof KNOWN_MODELS; // For autocomplete

// Models we no longer suggest, but that still work: they still resolve and
// are still registered as actions, but aren't in the KnownModels type, so
// they don't show up in autocomplete. Set `stage` on each one (keeping the
// default `supports`). Remove a model once the server returns 404 for it.
const HIDDEN_MODELS = {} as const;

const ALL_MODELS = {
  ...HIDDEN_MODELS,
  ...KNOWN_MODELS,
};

export type DeepResearchModelName = `deep-research-${string}`;
export function isDeepResearchModelName(
  value?: string
): value is DeepResearchModelName {
  return !!value?.startsWith('deep-research-');
}

export function model(
  version: string,
  config: DeepResearchConfig = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, ALL_MODELS)) {
    return ALL_MODELS[name].withConfig(config);
  }

  return modelRef({
    name: `vertexai/${name}`,
    config,
    configSchema: DeepResearchConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  return models
    .filter((m: Model) => isDeepResearchModelName(m.name))
    .map((m: Model) => {
      const ref = model(m.name);
      return modelActionMetadata({
        name: ref.name,
        info: ref.info,
        configSchema: ref.configSchema,
      });
    });
}

export function listKnownModels(
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
) {
  return Object.keys(ALL_MODELS).map((name: string) =>
    defineModel(name, clientOptions, pluginOptions)
  );
}

const THINKING_SUMMARIES = { AUTO: 'auto', NONE: 'none' } as const;
const VISUALIZATION = { AUTO: 'auto', OFF: 'off' } as const;

/**
 * The `response_format`: JSON output and the requested output types as
 * entries; undefined when neither is set (the agent's defaults apply).
 */
function toDeepResearchResponseFormat(
  output: GenerateRequest['output'],
  responseModalities?: string[]
): Record<string, unknown> | Record<string, unknown>[] | undefined {
  const json =
    output?.format === 'json' || output?.contentType === 'application/json';
  return toInteractionResponseFormat({
    modalities: responseModalities,
    json: json
      ? { schema: output?.schema ? cleanSchema(output.schema) : undefined }
      : undefined,
  });
}

export function defineModel(
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
): BackgroundModelAction<DeepResearchConfigSchemaType> {
  const ref = model(name);

  return pluginBackgroundModel({
    name: ref.name,
    ...ref.info,
    configSchema: ref.configSchema,
    async start(request, options) {
      // Deep Research is only on global: a defaulted location becomes global.
      const clientOpt = toInteractionsClientOptions(
        applyContextOverrides(
          calculateRequestOptions(clientOptions, request.config),
          options?.context
        )
      );

      const {
        // Applied to the client options above.
        apiKey: _apiKey,
        apiVersion: _apiVersion,
        location: _location,
        thinkingSummaries,
        visualization,
        store,
        responseModalities,
        // Built-in tools, e.g. { googleSearch: {} } or { mcpServer: {...} }.
        tools: toolsFromConfig,
        ...rest
      } = request.config || {};

      const messages = structuredClone(request.messages);
      // Deep Research does not support system instructions, so we map them to
      // user messages.
      for (const message of messages) {
        if (message.role === 'system') {
          message.role = 'user';
        }
      }

      // Without tools, the agent uses its defaults (Google Search and URL
      // context); any tool given replaces them.
      const tools: InteractionTool[] = [
        ...(Array.isArray(toolsFromConfig)
          ? toolsFromConfig.map(toInteractionConfigTool)
          : []),
        ...(request.tools ?? []).map(toInteractionTool),
      ];

      const req: CreateInteractionRequest = {
        agent: extractVersion(ref),
        input: toInteractionSteps(ensureToolIds(messages)),
        // Required: Deep Research only runs in the background.
        background: true,
        agent_config: {
          type: 'deep-research',
          thinking_summaries: thinkingSummaries
            ? THINKING_SUMMARIES[thinkingSummaries]
            : undefined,
          visualization: visualization
            ? VISUALIZATION[visualization]
            : undefined,
        },
        // A chosen `store` (request config, else plugin options) is sent as
        // is, even `false`: the server then rejects the request (background
        // interactions must be stored), rather than the plugin storing
        // anyway. When nothing is set, the server's default is used.
        store: store ?? pluginOptions?.store,
        tools: tools.length ? tools : undefined,
        response_format: toDeepResearchResponseFormat(
          request.output,
          responseModalities
        ),
        ...rest,
      };

      const response = await createInteraction(req, clientOpt);
      return fromInteraction(response);
    },
    async check(operation, options) {
      const response = await getInteraction(
        operation.id,
        toInteractionsClientOptions(
          applyContextOverrides(clientOptions, options?.context)
        )
      );
      return fromInteraction(response);
    },
    async cancel(operation, options) {
      const response = await cancelInteraction(
        operation.id,
        toInteractionsClientOptions(
          applyContextOverrides(clientOptions, options?.context)
        )
      );
      return fromInteraction(response);
    },
  });
}

export const TEST_ONLY = { KNOWN_MODELS, HIDDEN_MODELS, GENERIC_MODEL };
