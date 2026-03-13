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

import { ActionMetadata, GenkitError, modelActionMetadata, z } from 'genkit';
import {
  CandidateData,
  GenerationCommonConfigDescriptions,
  GenerationCommonConfigSchema,
  ModelAction,
  ModelInfo,
  ModelMiddleware,
  ModelReference,
  getBasicUsageStats,
  modelRef,
} from 'genkit/model';
import { downloadRequestMedia } from 'genkit/model/middleware';
import { model as pluginModel } from 'genkit/plugin';
import { runInNewSpan } from 'genkit/tracing';
import {
  fromGeminiCandidate,
  toGeminiFunctionModeEnum,
  toGeminiMessage,
  toGeminiSystemInstruction,
  toGeminiTool,
} from '../common/converters.js';
import {
  generateContent,
  generateContentStream,
  getVertexAIUrl,
} from './client.js';
import { toGeminiLabels, toGeminiSafetySettings } from './converters.js';
import {
  ClientOptions,
  Content,
  GenerateContentRequest,
  GenerateContentResponse,
  GoogleSearchRetrieval,
  GoogleSearchRetrievalTool,
  Model,
  Tool,
  ToolConfig,
  VertexPluginOptions,
} from './types.js';
import {
  calculateRequestOptions,
  checkModelName,
  cleanSchema,
  extractVersion,
  modelName,
} from './utils.js';

export const SafetySettingsSchema = z
  .object({
    category: z.enum([
      /** The harm category is unspecified. */
      'HARM_CATEGORY_UNSPECIFIED',
      /** The harm category is hate speech. */
      'HARM_CATEGORY_HATE_SPEECH',
      /** The harm category is dangerous content. */
      'HARM_CATEGORY_DANGEROUS_CONTENT',
      /** The harm category is harassment. */
      'HARM_CATEGORY_HARASSMENT',
      /** The harm category is sexually explicit content. */
      'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    ]),
    threshold: z.enum([
      'BLOCK_LOW_AND_ABOVE',
      'BLOCK_MEDIUM_AND_ABOVE',
      'BLOCK_ONLY_HIGH',
      'BLOCK_NONE',
    ]),
  })
  .passthrough();

const VertexRetrievalSchema = z
  .object({
    datastore: z
      .object({
        projectId: z.string().describe('Google Cloud Project ID.').optional(),
        location: z
          .string()
          .describe('Google Cloud region e.g. us-central1.')
          .optional(),
        dataStoreId: z
          .string()
          .describe(
            'The data store id, when project id and location are provided as ' +
              'separate options. Alternatively, the full path to the data ' +
              'store should be provided in the form: "projects/{project}/' +
              'locations/{location}/collections/default_collection/dataStores/{data_store}".'
          ),
      })
      .describe('Vertex AI Search data store details')
      .passthrough(),
    disableAttribution: z
      .boolean()
      .describe(
        'Disable using the search data in detecting grounding attribution. This ' +
          'does not affect how the result is given to the model for generation.'
      )
      .optional(),
  })
  .passthrough();

const GoogleSearchRetrievalSchema = z
  .object({
    disableAttribution: z
      .boolean()
      .describe(
        'Disable using the search data in detecting grounding attribution. This ' +
          'does not affect how the result is given to the model for generation.'
      )
      .optional(),
  })
  .passthrough();

/**
 * Zod schema of Gemini model options.
 * Please refer to: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig, for further information.
 */
export const GeminiConfigSchema = GenerationCommonConfigSchema.extend({
  apiKey: z
    .string()
    .describe('Overrides the plugin-configured API key, if specified.')
    .optional(),
  labels: z
    .record(z.string())
    .optional()
    .describe('Key-value labels to attach to the request for cost tracking.'),
  temperature: z
    .number()
    .min(0.0)
    .max(2.0)
    .describe(
      GenerationCommonConfigDescriptions.temperature +
        ' The default value is 1.0.'
    )
    .optional(),
  topP: z
    .number()
    .min(0)
    .max(1.0)
    .describe(
      GenerationCommonConfigDescriptions.topP + ' The default value is 0.95.'
    )
    .optional(),
  location: z
    .string()
    .describe('Google Cloud region e.g. us-central1.')
    .optional(),

  /**
   * Safety filter settings. See: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-filters#configurable-filters
   *
   * E.g.
   *
   * ```js
   * config: {
   *   safetySettings: [
   *     {
   *       category: 'HARM_CATEGORY_HATE_SPEECH',
   *       threshold: 'BLOCK_LOW_AND_ABOVE',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
   *       threshold: 'BLOCK_MEDIUM_AND_ABOVE',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_HARASSMENT',
   *       threshold: 'BLOCK_ONLY_HIGH',
   *     },
   *     {
   *       category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
   *       threshold: 'BLOCK_NONE',
   *     },
   *   ],
   * }
   * ```
   */
  safetySettings: z
    .array(SafetySettingsSchema)
    .describe(
      'Adjust how likely you are to see responses that could be harmful. ' +
        'Content is blocked based on the probability that it is harmful.'
    )
    .optional(),

  /**
   * Vertex retrieval options.
   *
   * E.g.
   *
   * ```js
   *   config: {
   *     vertexRetrieval: {
   *       datastore: {
   *         projectId: 'your-cloud-project',
   *         location: 'us-central1',
   *         collection: 'your-collection',
   *       },
   *       disableAttribution: true,
   *     }
   *   }
   * ```
   */
  vertexRetrieval: VertexRetrievalSchema.describe(
    'Retrieve from Vertex AI Search data store for grounding ' +
      'generative responses.'
  ).optional(),

  /**
   * Google Search retrieval options.
   *
   * ```js
   *   config: {
   *     googleSearchRetrieval: {
   *       disableAttribution: true,
   *     }
   *   }
   * ```
   */
  googleSearchRetrieval: GoogleSearchRetrievalSchema.describe(
    'Retrieve public web data for grounding, powered by Google Search.'
  ).optional(),

  /**
   * Function calling options.
   *
   * E.g. forced tool call:
   *
   * ```js
   *   config: {
   *     functionCallingConfig: {
   *       mode: 'ANY',
   *     }
   *   }
   * ```
   */
  functionCallingConfig: z
    .object({
      mode: z.enum(['MODE_UNSPECIFIED', 'AUTO', 'ANY', 'NONE']).optional(),
      allowedFunctionNames: z.array(z.string()).optional(),
      /**
       * When set to true, arguments of a single function call will be streamed out in
       * multiple parts/contents/responses. Partial parameter results will be returned in the
       * [FunctionCall.partial_args] field. This field is not supported in Gemini API.
       */
      streamFunctionCallArguments: z.boolean().optional(),
    })
    .describe(
      'Controls how the model uses the provided tools (function declarations). ' +
        'With AUTO (Default) mode, the model decides whether to generate a ' +
        'natural language response or suggest a function call based on the ' +
        'prompt and context. With ANY, the model is constrained to always ' +
        'predict a function call and guarantee function schema adherence. ' +
        'With NONE, the model is prohibited from making function calls.'
    )
    .passthrough()
    .optional(),
  /**
   * Retrieval config for search grounding and maps grounding
   */
  retrievalConfig: z
    .object({
      /**
       * User location for search grounding or
       * place location for maps grounding.
       */
      latLng: z
        .object({
          latitude: z.number().optional(),
          longitude: z.number().optional(),
        })
        .describe('User location for Google search or Google maps grounding.')
        .optional(),
      /**
       * Language code for the request. e.g. 'en-us'
       */
      languageCode: z.string().optional(),
    })
    .passthrough()
    .optional(),
  thinkingConfig: z
    .object({
      includeThoughts: z
        .boolean()
        .describe(
          'Indicates whether to include thoughts in the response.' +
            'If true, thoughts are returned only if the model supports ' +
            'thought and thoughts are available.'
        )
        .optional(),
      thinkingBudget: z
        .number()
        .min(0)
        .max(24576)
        .describe(
          'For Gemini 2.5 - Indicates the thinking budget in tokens. 0 is DISABLED. ' +
            '-1 is AUTOMATIC. The default values and allowed ranges are model ' +
            'dependent. The thinking budget parameter gives the model guidance ' +
            'on the number of thinking tokens it can use when generating a ' +
            'response. A greater number of tokens is typically associated with ' +
            'more detailed thinking, which is needed for solving more complex ' +
            'tasks. '
        )
        .optional(),
      thinkingLevel: z
        .enum(['MINIMAL', 'LOW', 'MEDIUM', 'HIGH'])
        .describe(
          'For Gemini 3.0 - Indicates the thinking level. A higher level ' +
            'is associated with more detailed thinking, which is needed for solving ' +
            'more complex tasks.'
        )
        .optional(),
    })
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiConfigSchemaType = typeof GeminiConfigSchema;
/**
 * Gemini model configuration options.
 *
 * E.g.
 * ```js
 *   config: {
 *     temperature: 0.9,
 *     maxOutputTokens: 300,
 *     safetySettings: [
 *       {
 *         category: 'HARM_CATEGORY_HATE_SPEECH',
 *         threshold: 'BLOCK_LOW_AND_ABOVE',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
 *         threshold: 'BLOCK_MEDIUM_AND_ABOVE',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_HARASSMENT',
 *         threshold: 'BLOCK_ONLY_HIGH',
 *       },
 *       {
 *         category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
 *         threshold: 'BLOCK_NONE',
 *       },
 *     ],
 *     functionCallingConfig: {
 *       mode: 'ANY',
 *     }
 *   }
 * ```
 */
export type GeminiConfig = z.infer<GeminiConfigSchemaType>;

export const GeminiImageConfigSchema = GeminiConfigSchema.extend({
  imageConfig: z
    .object({
      aspectRatio: z
        .enum([
          '1:1',
          '2:3',
          '3:2',
          '3:4',
          '4:3',
          '4:5',
          '5:4',
          '9:16',
          '16:9',
          '21:9',
        ])
        .optional(),
      imageSize: z.enum(['1K', '2K', '4K']).optional(),
    })
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiImageConfigSchemaType = typeof GeminiImageConfigSchema;
export type GeminiImageConfig = z.infer<GeminiImageConfigSchemaType>;

// This contains all the Gemini config schema types
type ConfigSchemaType = GeminiConfigSchemaType | GeminiImageConfigSchemaType;
type ConfigSchema = z.infer<ConfigSchemaType>;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = GeminiConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertexai/${name}`,
    configSchema,
    info: info ?? {
      supports: {
        multiturn: true,
        media: true,
        tools: true,
        toolChoice: true,
        systemRole: true,
        constrained: 'no-tools',
      },
    },
  });
}

const GENERIC_MODEL = commonRef('gemini');
const GENERIC_IMAGE_MODEL = commonRef(
  'gemini-image',
  undefined,
  GeminiImageConfigSchema
);

export const KNOWN_GEMINI_MODELS = {
  'gemini-3.1-pro-preview': commonRef('gemini-3.1-pro-preview'),
  'gemini-3-flash-preview': commonRef('gemini-3-flash-preview'),
  'gemini-3-pro-preview': commonRef('gemini-3-pro-preview'),
  'gemini-2.5-flash-lite': commonRef('gemini-2.5-flash-lite'),
  'gemini-2.5-pro': commonRef('gemini-2.5-pro'),
  'gemini-2.5-flash': commonRef('gemini-2.5-flash'),
} as const;
export type KnownGeminiModels = keyof typeof KNOWN_GEMINI_MODELS;
export type GeminiModelName = `gemini-${string}`;
export function isGeminiModelName(value?: string): value is GeminiModelName {
  return !!(
    value?.startsWith('gemini-') &&
    !value.includes('embedding') &&
    !value.includes('-image')
  );
}

export const KNOWN_IMAGE_MODELS = {
  'gemini-3-pro-image-preview': commonRef(
    'gemini-3-pro-image-preview',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema
  ),
  'gemini-2.5-flash-image': commonRef(
    'gemini-2.5-flash-image',
    undefined,
    GeminiImageConfigSchema
  ),
} as const;
export type KnownImageModels = keyof typeof KNOWN_IMAGE_MODELS;
export type ImageModelName = `gemini-${string}-image${string}`;
export function isImageModelName(value?: string): value is ImageModelName {
  return !!(value?.startsWith('gemini-') && value.includes('-image'));
}

const KNOWN_MODELS = {
  ...KNOWN_GEMINI_MODELS,
  ...KNOWN_IMAGE_MODELS,
};
export type KnownModels = keyof typeof KNOWN_MODELS;

export function model(
  version: string,
  config: ConfigSchema = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isImageModelName(name)) {
    return modelRef({
      name: `vertexai/${name}`,
      config,
      configSchema: GeminiImageConfigSchema,
      info: { ...GENERIC_IMAGE_MODEL.info },
    });
  }

  return modelRef({
    name: `vertexai/${name}`,
    config,
    configSchema: GeminiConfigSchema,
    info: {
      ...GENERIC_MODEL.info,
    },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  const KNOWN_DECOMISSIONED_MODELS = [
    'gemini-pro-vision',
    'gemini-pro',
    'gemini-ultra',
    'gemini-ultra-vision',
  ];

  return models
    .filter(
      (m) =>
        (isGeminiModelName(modelName(m.name)) ||
          isImageModelName(modelName(m.name))) &&
        !KNOWN_DECOMISSIONED_MODELS.includes(modelName(m.name) || '')
    )
    .map((m) => {
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
  return Object.keys(KNOWN_MODELS).map((name) =>
    defineModel(name, clientOptions, pluginOptions)
  );
}

/**
 * Define a Vertex AI Gemini model.
 */
export function defineModel(
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
): ModelAction {
  const ref = model(name);
  const middlewares: ModelMiddleware[] = [];
  if (ref.info?.supports?.media) {
    // the gemini api doesn't support downloading media from http(s)
    middlewares.push(
      downloadRequestMedia({
        maxBytes: 1024 * 1024 * 20,
        filter: (part) => {
          try {
            const url = new URL(part.media.url);
            if (
              // Gemini can handle these URLs
              ['www.youtube.com', 'youtube.com', 'youtu.be'].includes(
                url.hostname
              )
            )
              return false;
          } catch {}
          return true;
        },
      })
    );
  }

  return pluginModel(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
      use: middlewares,
    },
    async (request, { streamingRequested, sendChunk, abortSignal }) => {
      let clientOpt = { ...clientOptions, signal: abortSignal };

      // Make a copy of messages to avoid side-effects
      const messages = structuredClone(request.messages);
      if (messages.length === 0) throw new Error('No messages provided.');

      // Handle system instructions separately
      let systemInstruction: Content | undefined = undefined;
      const systemMessage = messages.find((m) => m.role === 'system');
      if (systemMessage) {
        messages.splice(messages.indexOf(systemMessage), 1);
        systemInstruction = toGeminiSystemInstruction(systemMessage);
      }

      const requestConfig: ConfigSchema = { ...request.config };

      const {
        apiKey: apiKeyFromConfig,
        functionCallingConfig,
        retrievalConfig,
        version: versionFromConfig,
        googleSearchRetrieval,
        tools: toolsFromConfig,
        vertexRetrieval,
        location,
        safetySettings,
        labels: labelsFromConfig,
        ...restOfConfig
      } = requestConfig;

      clientOpt = calculateRequestOptions(clientOpt, {
        location,
        apiKey: apiKeyFromConfig,
      });

      const labels = toGeminiLabels(labelsFromConfig);

      const tools: Tool[] = [];
      if (request.tools?.length) {
        tools.push({
          functionDeclarations: request.tools.map(toGeminiTool),
        });
      }

      let toolConfig: ToolConfig | undefined;
      if (functionCallingConfig) {
        toolConfig = {
          functionCallingConfig: {
            ...functionCallingConfig,
            allowedFunctionNames: functionCallingConfig.allowedFunctionNames,
            mode: toGeminiFunctionModeEnum(functionCallingConfig.mode),
          },
        };
      } else if (request.toolChoice) {
        toolConfig = {
          functionCallingConfig: {
            mode: toGeminiFunctionModeEnum(request.toolChoice),
          },
        };
      }

      if (retrievalConfig) {
        if (!toolConfig) {
          toolConfig = {};
        }
        toolConfig.retrievalConfig = structuredClone(retrievalConfig);
      }

      // Cannot use tools and function calling at the same time
      const jsonMode =
        (request.output?.format === 'json' || !!request.output?.schema) &&
        tools.length === 0;

      if (toolsFromConfig) {
        tools.push(...(toolsFromConfig as any[]));
      }

      if (googleSearchRetrieval) {
        // Gemini 1.5 models use googleSearchRetrieval, newer models use googleSearch.
        if (ref.name.startsWith('vertexai/gemini-1.5')) {
          tools.push({
            googleSearchRetrieval:
              googleSearchRetrieval as GoogleSearchRetrieval,
          } as GoogleSearchRetrievalTool);
        } else {
          tools.push({
            googleSearch: googleSearchRetrieval as GoogleSearchRetrieval,
          } as GoogleSearchRetrievalTool);
        }
      }

      if (vertexRetrieval) {
        const _projectId =
          vertexRetrieval.datastore.projectId ||
          (clientOptions.kind != 'express'
            ? clientOptions.projectId
            : undefined);
        const _location =
          vertexRetrieval.datastore.location ||
          (clientOptions.kind == 'regional'
            ? clientOptions.location
            : undefined);
        const _dataStoreId = vertexRetrieval.datastore.dataStoreId;
        if (!_projectId || !_location || !_dataStoreId) {
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message:
              'projectId, location and datastoreId are required for vertexRetrieval and could not be determined from configuration',
          });
        }
        const datastore = `projects/${_projectId}/locations/${_location}/collections/default_collection/dataStores/${_dataStoreId}`;
        tools.push({
          retrieval: {
            vertexAiSearch: {
              datastore,
            },
            disableAttribution: vertexRetrieval.disableAttribution,
          },
        });
      }

      const generateContentRequest: GenerateContentRequest = {
        systemInstruction,
        generationConfig: {
          ...restOfConfig,
          candidateCount: request.candidates || undefined,
          responseMimeType: jsonMode ? 'application/json' : undefined,
        },
        tools,
        toolConfig,
        safetySettings: toGeminiSafetySettings(safetySettings),
        contents: messages.map((message) => toGeminiMessage(message, ref)),
        labels,
      };

      const modelVersion = versionFromConfig || extractVersion(ref);

      if (jsonMode && request.output?.constrained) {
        if (pluginOptions?.legacyResponseSchema) {
          generateContentRequest.generationConfig!.responseSchema = cleanSchema(
            request.output.schema
          );
        } else {
          generateContentRequest.generationConfig!.responseJsonSchema =
            request.output.schema;
        }
      }

      const callGemini = async () => {
        let response: GenerateContentResponse;

        // Handle streaming and non-streaming responses
        if (streamingRequested) {
          const result = await generateContentStream(
            modelVersion,
            generateContentRequest,
            clientOpt
          );

          const chunks: CandidateData[] = [];
          for await (const item of result.stream) {
            (item as GenerateContentResponse).candidates?.forEach(
              (candidate) => {
                const c = fromGeminiCandidate(candidate, chunks);
                chunks.push(c);
                sendChunk({
                  index: c.index,
                  content: c.message.content,
                });
              }
            );
          }
          response = await result.response;
        } else {
          response = await generateContent(
            modelVersion,
            generateContentRequest,
            clientOpt
          );
        }

        if (!response.candidates?.length) {
          throw new GenkitError({
            status: 'FAILED_PRECONDITION',
            message: 'No valid candidates returned.',
          });
        }

        const candidateData = response.candidates.map((c) =>
          fromGeminiCandidate(c)
        );

        return {
          candidates: candidateData,
          custom: response,
          usage: {
            ...getBasicUsageStats(request.messages, candidateData),
            inputTokens: response.usageMetadata?.promptTokenCount,
            outputTokens: response.usageMetadata?.candidatesTokenCount,
            thoughtsTokens: response.usageMetadata?.thoughtsTokenCount,
            totalTokens: response.usageMetadata?.totalTokenCount,
            cachedContentTokens:
              response.usageMetadata?.cachedContentTokenCount,
          },
        };
      };

      // If debugTraces is enabled, we wrap the actual model call with a span,
      // add raw API params as for input.
      const msg = toGeminiMessage(messages[messages.length - 1], ref);
      return pluginOptions?.experimental_debugTraces
        ? await runInNewSpan(
            {
              metadata: {
                name: streamingRequested ? 'sendMessageStream' : 'sendMessage',
              },
            },
            async (metadata) => {
              metadata.input = {
                apiEndpoint: getVertexAIUrl({
                  includeProjectAndLocation: false,
                  resourcePath: '',
                  clientOptions: clientOpt,
                }),
                cache: {},
                model: modelVersion,
                generateContentOptions: generateContentRequest,
                parts: msg.parts,
                options: clientOpt,
              };
              const response = await callGemini();
              metadata.output = response.custom;
              return response;
            }
          )
        : await callGemini();
    }
  );
}

export const TEST_ONLY = {
  KNOWN_GEMINI_MODELS,
  KNOWN_IMAGE_MODELS,
  KNOWN_MODELS,
};
