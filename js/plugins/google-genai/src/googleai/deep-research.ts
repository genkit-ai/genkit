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
import {
  ensureToolIds,
  fromInteraction,
  toInteractionTool,
  toInteractionTurn,
} from '../common/interaction-converters.js';
import {
  InteractionTool,
  ResponseModality,
} from '../common/interaction-types.js';
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
  calculateApiKey,
  checkApiKey,
  checkModelName,
  extractVersion,
  modelName,
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
    googleSearch: z
      .union([z.boolean(), z.object({}).passthrough()])
      .describe(
        'Search the public web. Enabled by default if no tools are provided.'
      )
      .optional(),
    urlContext: z
      .union([z.boolean(), z.object({}).passthrough()])
      .describe(
        'Read and summarize web page content. Enabled by default if no tools are provided.'
      )
      .optional(),
    codeExecution: z
      .union([z.boolean(), z.object({}).passthrough()])
      .describe(
        'Execute code to perform calculations and data analysis. Enabled by default if no tools are provided.'
      )
      .optional(),
    fileSearch: z
      .object({
        fileSearchStoreNames: z.array(z.string()),
      })
      .passthrough()
      .optional(),
    mcpServers: z
      .array(
        z
          .object({
            name: z.string().optional(),
            url: z.string().optional(),
            headers: z.record(z.string()).optional(),
            allowedTools: z.array(z.string()).optional(),
          })
          .passthrough()
      )
      .optional(),
  })
  .passthrough();
export type DeepResearchConfigSchemaType = typeof DeepResearchConfigSchema;
export type DeepResearchConfig = z.infer<DeepResearchConfigSchemaType>;

// This contains all the DeepResearch config schema types
type ConfigSchemaType = DeepResearchConfigSchemaType;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = DeepResearchConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `googleai/${name}`,
    configSchema,
    info:
      info ??
      ({
        supports: {
          multiturn: true,
          media: false,
          tools: false,
          toolChoice: false,
          systemRole: false,
          output: ['text'],
          longRunning: true,
        },
      } as ModelInfo),
  });
}

const GENERIC_MODEL = commonRef('deep-research');

const ADVANCED_DEEP_RESEARCH_INFO = {
  supports: {
    multiturn: true,
    media: true,
    tools: true,
    toolChoice: false,
    systemRole: false,
    output: ['text', 'media'],
    longRunning: true,
  },
} as ModelInfo;

const KNOWN_MODELS = {
  'deep-research-pro-preview-12-2025': commonRef(
    'deep-research-pro-preview-12-2025'
  ),
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

  if (isKnownKey(name, KNOWN_MODELS)) {
    return KNOWN_MODELS[name].withConfig(config);
  }

  return modelRef({
    name: `googleai/${name}`,
    config,
    configSchema: DeepResearchConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  return (
    models
      .filter((m) => isDeepResearchModelName(modelName(m.name)))
      // Filter out deprecated
      .filter((m) => !m.description || !m.description.includes('deprecated'))
      .map((m) => {
        const ref = model(m.name);
        return modelActionMetadata({
          name: ref.name,
          info: ref.info,
          configSchema: ref.configSchema,
        });
      })
  );
}

export function listKnownModels(options?: GoogleAIPluginOptions) {
  return Object.keys(KNOWN_MODELS).map((name: string) =>
    defineModel(name, options)
  );
}

export function defineModel(
  name: string,
  pluginOptions?: GoogleAIPluginOptions
): BackgroundModelAction<DeepResearchConfigSchemaType> {
  checkApiKey(pluginOptions?.apiKey);
  const ref = model(name);
  const clientOptions: ClientOptions = {
    apiVersion: pluginOptions?.apiVersion,
    baseUrl: pluginOptions?.baseUrl,
    customHeaders: pluginOptions?.customHeaders,
  };

  return pluginBackgroundModel({
    name: ref.name,
    ...ref.info,
    configSchema: ref.configSchema,
    async start(request) {
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
        googleSearch,
        urlContext,
        codeExecution,
        fileSearch,
        mcpServers,
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
      // Deep Research does not support system instructions, so we map them to
      // user messages.
      for (const message of messages) {
        if (message.role === 'system') {
          message.role = 'user';
        }
      }

      const tools: InteractionTool[] = [];
      if (request.tools?.length) {
        for (const tool of request.tools) {
          tools.push(toInteractionTool(tool));
        }
      }

      if (googleSearch) {
        tools.push({
          type: 'google_search',
          ...(googleSearch === true ? {} : googleSearch),
        } as InteractionTool);
      }
      if (urlContext) {
        tools.push({
          type: 'url_context',
          ...(urlContext === true ? {} : urlContext),
        } as InteractionTool);
      }
      if (codeExecution) {
        tools.push({
          type: 'code_execution',
          ...(codeExecution === true ? {} : codeExecution),
        } as InteractionTool);
      }
      if (fileSearch) {
        const { fileSearchStoreNames, ...restFileSearch } = fileSearch;
        tools.push({
          type: 'file_search',
          file_search_store_names: fileSearchStoreNames,
          ...restFileSearch,
        } as InteractionTool);
      }
      if (mcpServers) {
        for (const mcpServer of mcpServers) {
          const { allowedTools, ...restMcp } = mcpServer;
          tools.push({
            type: 'mcp_server',
            ...restMcp,
            ...(allowedTools ? { allowed_tools: allowedTools } : {}),
          } as InteractionTool);
        }
      }

      let responseMimeType = request.output?.contentType;
      let responseFormat: any = undefined;
      if (
        request.output?.format === 'json' ||
        request.output?.contentType === 'application/json'
      ) {
        responseMimeType = 'application/json';
        if (request.output.schema) {
          responseFormat = cleanSchema(request.output.schema);
        }
      }

      let responseModalitiesConverted: ResponseModality[] | undefined;
      if (responseModalities) {
        responseModalitiesConverted = responseModalities.map(
          (m) => m.toLowerCase() as ResponseModality
        );
      }

      const req: CreateInteractionRequest = {
        agent: extractVersion(ref),
        input: ensureToolIds(messages).map(toInteractionTurn),
        background: true,
        agent_config: {
          type: 'deep-research',
          thinking_summaries: thinkingSummaries?.toLowerCase() as
            | 'auto'
            | 'none'
            | undefined,
          visualization: visualization?.toLowerCase() as
            | 'auto'
            | 'off'
            | undefined,
          collaborative_planning: collaborativePlanning,
        },
        previous_interaction_id: previousInteractionId,
        store,
        tools: tools.length ? tools : undefined,
        response_mime_type: responseMimeType,
        response_format: responseFormat,
        response_modalities: responseModalitiesConverted,
        ...rest,
      };

      const response = await createInteraction(apiKey, req, newClientOptions);

      return fromInteraction<ClientOptions>(response, newClientOptions);
    },
    async check(operation) {
      const storedOptions = operation.metadata?.clientOptions as
        | ClientOptions
        | undefined;
      const apiKey =
        storedOptions?.apiKey ||
        calculateApiKey(pluginOptions?.apiKey, undefined);

      const checkOptions: ClientOptions = {
        ...clientOptions,
        ...storedOptions,
      };

      const response = await getInteraction(apiKey, operation.id, checkOptions);

      return fromInteraction<ClientOptions>(response, checkOptions);
    },
    async cancel(operation) {
      const storedOptions = operation.metadata?.clientOptions as
        | ClientOptions
        | undefined;
      const apiKey =
        storedOptions?.apiKey ||
        calculateApiKey(pluginOptions?.apiKey, undefined);

      const cancelOptions: ClientOptions = {
        ...clientOptions,
        ...storedOptions,
      };

      const response = await cancelInteraction(
        apiKey,
        operation.id,
        cancelOptions
      );

      return fromInteraction<ClientOptions>(response, cancelOptions);
    },
  });
}

export const TEST_ONLY = { KNOWN_MODELS };
