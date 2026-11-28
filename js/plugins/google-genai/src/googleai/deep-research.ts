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
import {
  BackgroundModelAction,
  ModelInfo,
  ModelReference,
  modelRef,
} from 'genkit/model';
import { backgroundModel as pluginBackgroundModel } from 'genkit/plugin';
import type { InferOutput } from 'genkit/schema';
import {
  ensureToolIds,
  fromInteraction,
  toInteractionConfigTool,
  toInteractionResponseFormat,
  toInteractionSteps,
  toInteractionTool,
} from '../common/interaction-converters.js';
import { InteractionTool } from '../common/interaction-types.js';
import { cleanSchema, isKnownKey } from '../common/utils.js';
import {
  cancelInteraction,
  createInteraction,
  getInteraction,
} from './client.js';
import {
  ClientOptions,
  CreateInteractionRequest,
  GoogleAIPluginOptions,
  Model,
} from './types.js';
import {
  DEFAULT_NAMESPACE,
  applyContextOverrides,
  calculateApiKey,
  checkApiKey,
  checkModelName,
  extractVersion,
  modelName,
  type PluginNamespace,
} from './utils.js';

export const DeepResearchConfigSchema = z
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
    thinkingSummaries: z
      .enum(['AUTO', 'NONE'])
      .describe('Whether to include thought summaries in the response.')
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
    responseModalities: z
      .array(z.enum(['TEXT', 'IMAGE', 'AUDIO']))
      .describe('The modalities to be used in response.')
      .optional(),
    visualization: z
      .enum(['AUTO', 'OFF'])
      .describe('Whether to enable agent-generated charts and images.')
      .optional(),
    collaborativePlanning: z
      .boolean()
      .describe(
        'Whether to enable multi-turn plan review before research begins.'
      )
      .optional(),
    tools: z
      .array(z.record(z.string(), z.unknown()))
      .describe(
        'Built-in tools, e.g. [{ googleSearch: {} }, { urlContext: {} }, ' +
          '{ codeExecution: {} }, { fileSearch: { fileSearchStoreNames } }, ' +
          '{ mcpServer: { name, url, headers, allowedTools } }]. Without ' +
          'tools the agent uses Google Search, URL context and code ' +
          'execution; any tool given replaces them.'
      )
      .optional(),
  })
  .passthrough();
export type DeepResearchConfigSchemaType = typeof DeepResearchConfigSchema;
export type DeepResearchConfig = InferOutput<DeepResearchConfigSchemaType>;

// This contains all the DeepResearch config schema types
type ConfigSchemaType = DeepResearchConfigSchemaType;

// `longRunning` isn't in ModelInfo's `supports` type, but Genkit requires it
// at runtime for background models (generateOperation). Declaring `supports`
// separately keeps it without a cast (excess-property checks only apply to
// object literals written in place).
const DEEP_RESEARCH_SUPPORTS = {
  multiturn: true,
  media: false,
  tools: false,
  toolChoice: false,
  systemRole: false,
  output: ['text'],
  longRunning: true,
};
const DEEP_RESEARCH_INFO: ModelInfo = { supports: DEEP_RESEARCH_SUPPORTS };

function commonRef(
  name: string,
  info: ModelInfo = DEEP_RESEARCH_INFO,
  configSchema: ConfigSchemaType = DeepResearchConfigSchema
): ModelReference<ConfigSchemaType> {
  // `model()` re-creates known refs under the requested namespace.
  return modelRef({
    name: `${DEFAULT_NAMESPACE}/${name}`,
    configSchema,
    info,
  });
}

const GENERIC_MODEL = commonRef('deep-research');

const ADVANCED_DEEP_RESEARCH_SUPPORTS = {
  multiturn: true,
  media: true,
  tools: true,
  toolChoice: false,
  systemRole: false,
  output: ['text', 'media'],
  longRunning: true,
};
const ADVANCED_DEEP_RESEARCH_INFO: ModelInfo = {
  supports: ADVANCED_DEEP_RESEARCH_SUPPORTS,
};

const KNOWN_MODELS = {
  'deep-research-preview-04-2026': commonRef(
    'deep-research-preview-04-2026',
    ADVANCED_DEEP_RESEARCH_INFO
  ),
  'deep-research-max-preview-04-2026': commonRef(
    'deep-research-max-preview-04-2026',
    ADVANCED_DEEP_RESEARCH_INFO
  ),
} as const;
export type KnownModels = keyof typeof KNOWN_MODELS; // For autocomplete

// Models we no longer suggest, but that still work: they still resolve and
// are still registered as actions, but aren't in the KnownModels type, so
// they don't show up in autocomplete. Set `stage` on each one (keeping the
// default `supports`).
const HIDDEN_MODELS = {
  // Deprecated; the shutdown date was Oct 23 2026 (changelog, Oct 8 2026),
  // but it still accepted requests on Oct 27 2026. Remove it once the server
  // returns 404. Replaced by deep-research-preview-04-2026 and
  // deep-research-max-preview-04-2026.
  'deep-research-pro-preview-12-2025': commonRef(
    'deep-research-pro-preview-12-2025',
    { ...GENERIC_MODEL.info, stage: 'deprecated' }
  ),
} as const;

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
  config: DeepResearchConfig = {},
  namespace: PluginNamespace = DEFAULT_NAMESPACE
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, ALL_MODELS)) {
    const known = ALL_MODELS[name];
    return modelRef({
      name: `${namespace}/${name}`,
      config,
      configSchema: known.configSchema,
      info: known.info,
    });
  }

  return modelRef({
    name: `${namespace}/${name}`,
    config,
    configSchema: DeepResearchConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

export function listActions(
  models: Model[],
  namespace: PluginNamespace = DEFAULT_NAMESPACE
): ActionMetadata[] {
  return (
    models
      .filter((m) => isDeepResearchModelName(modelName(m.name)))
      // Filter out deprecated
      .filter((m) => !m.description || !m.description.includes('deprecated'))
      .map((m) => {
        const ref = model(m.name, {}, namespace);
        return modelActionMetadata({
          name: ref.name,
          info: ref.info,
          configSchema: ref.configSchema,
        });
      })
  );
}

export function listKnownModels(
  options?: GoogleAIPluginOptions,
  namespace: PluginNamespace = DEFAULT_NAMESPACE
) {
  return Object.keys(ALL_MODELS).map((name: string) =>
    defineModel(name, options, namespace)
  );
}

export function defineModel(
  name: string,
  pluginOptions?: GoogleAIPluginOptions,
  namespace: PluginNamespace = DEFAULT_NAMESPACE
): BackgroundModelAction<DeepResearchConfigSchemaType> {
  checkApiKey(pluginOptions?.apiKey);
  const ref = model(name, {}, namespace);
  const clientOptions: ClientOptions = {
    apiVersion: pluginOptions?.apiVersion,
    baseUrl: pluginOptions?.baseUrl,
    customHeaders: pluginOptions?.customHeaders,
    experimental_debugTraces: pluginOptions?.experimental_debugTraces,
  };

  return pluginBackgroundModel({
    name: ref.name,
    ...ref.info,
    configSchema: ref.configSchema,
    async start(request, options) {
      const {
        apiKey: apiKeyConfig,
        baseUrl,
        apiVersion,
        thinkingSummaries,
        visualization,
        collaborativePlanning,
        previousInteractionId,
        store,
        responseModalities,
        // Built-in tools, e.g. [{ googleSearch: {} }, { mcpServer: {...} }].
        tools: toolsFromConfig,
        ...rest
      } = request.config || {};
      const newClientOptions = applyContextOverrides(
        {
          ...clientOptions,
          apiKey: calculateApiKey(pluginOptions?.apiKey, apiKeyConfig),
          baseUrl: baseUrl || clientOptions.baseUrl,
          apiVersion: apiVersion || clientOptions.apiVersion,
        },
        options?.context
      );

      const messages = structuredClone(request.messages);
      // Deep Research does not support system instructions, so we map them to
      // user messages.
      for (const message of messages) {
        if (message.role === 'system') {
          message.role = 'user';
        }
      }

      // Without tools, the agent uses its defaults (Google Search, URL context
      // and code execution); any tool given replaces them.
      const tools: InteractionTool[] = [
        ...(Array.isArray(toolsFromConfig)
          ? toolsFromConfig.map(toInteractionConfigTool)
          : []),
        ...(request.tools ?? []).map(toInteractionTool),
      ];

      // JSON output and the requested output types, as `response_format`
      // entries.
      const jsonOutput =
        request.output?.format === 'json' ||
        request.output?.contentType === 'application/json';
      const responseFormat = toInteractionResponseFormat({
        modalities: responseModalities,
        json: jsonOutput
          ? {
              schema: request.output?.schema
                ? cleanSchema(request.output.schema)
                : undefined,
            }
          : undefined,
      });

      const req: CreateInteractionRequest = {
        agent: extractVersion(ref),
        input: toInteractionSteps(ensureToolIds(messages)),
        background: true,
        agent_config: {
          type: 'deep-research',
          thinking_summaries: thinkingSummaries?.toLowerCase() as
            'auto' | 'none' | undefined,
          visualization: visualization?.toLowerCase() as
            'auto' | 'off' | undefined,
          collaborative_planning: collaborativePlanning,
        },
        previous_interaction_id: previousInteractionId,
        store,
        tools: tools.length ? tools : undefined,
        response_format: responseFormat,
        ...rest,
      };

      const response = await createInteraction(
        newClientOptions.apiKey,
        req,
        newClientOptions
      );

      return fromInteraction(response);
    },
    async check(operation, options) {
      const checkOptions = applyContextOverrides(
        {
          ...clientOptions,
          apiKey: calculateApiKey(pluginOptions?.apiKey, undefined),
        },
        options?.context
      );

      const response = await getInteraction(
        checkOptions.apiKey,
        operation.id,
        checkOptions
      );

      return fromInteraction(response);
    },
    async cancel(operation, options) {
      const cancelOptions = applyContextOverrides(
        {
          ...clientOptions,
          apiKey: calculateApiKey(pluginOptions?.apiKey, undefined),
        },
        options?.context
      );

      const response = await cancelInteraction(
        cancelOptions.apiKey,
        operation.id,
        cancelOptions
      );

      return fromInteraction(response);
    },
  });
}

export const TEST_ONLY = { KNOWN_MODELS, HIDDEN_MODELS, GENERIC_MODEL };
