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
  getGoogleAIUrl,
} from './client.js';
import {
  ClientOptions,
  Content as GeminiMessage,
  GenerateContentRequest,
  GenerateContentResponse,
  GenerationConfig,
  GoogleAIPluginOptions,
  GoogleSearchRetrievalTool,
  Model,
  SafetySetting,
  Tool,
  ToolConfig,
  UrlContextTool,
} from './types.js';
import {
  calculateApiKey,
  calculateRequestOptions,
  checkApiKey,
  checkModelName,
  cleanSchema,
  extractVersion,
  removeClientOptionOverrides,
} from './utils.js';

const MAX_INLINE_MEDIA_BYTES = 1024 * 1024 * 100; // 100 MB

/**
 * See https://ai.google.dev/gemini-api/docs/safety-settings#safety-filters.
 */
const SafetySettingsSchema = z
  .object({
    category: z.enum([
      'HARM_CATEGORY_UNSPECIFIED',
      'HARM_CATEGORY_HATE_SPEECH',
      'HARM_CATEGORY_SEXUALLY_EXPLICIT',
      'HARM_CATEGORY_HARASSMENT',
      'HARM_CATEGORY_DANGEROUS_CONTENT',
    ]),
    threshold: z.enum([
      'BLOCK_LOW_AND_ABOVE',
      'BLOCK_MEDIUM_AND_ABOVE',
      'BLOCK_ONLY_HIGH',
      'BLOCK_NONE',
    ]),
  })
  .passthrough();

const VoiceConfigSchema = z
  .object({
    prebuiltVoiceConfig: z
      .object({
        // TODO: Make this an array of objects so we can also specify the description
        // for each voiceName.
        voiceName: z
          .union([
            z.enum([
              'Zephyr',
              'Puck',
              'Charon',
              'Kore',
              'Fenrir',
              'Leda',
              'Orus',
              'Aoede',
              'Callirrhoe',
              'Autonoe',
              'Enceladus',
              'Iapetus',
              'Umbriel',
              'Algieba',
              'Despina',
              'Erinome',
              'Algenib',
              'Rasalgethi',
              'Laomedeia',
              'Achernar',
              'Alnilam',
              'Schedar',
              'Gacrux',
              'Pulcherrima',
              'Achird',
              'Zubenelgenubi',
              'Vindemiatrix',
              'Sadachbia',
              'Sadaltager',
              'Sulafat',
            ]),
            // To allow any new string values
            z.string(),
          ])
          .describe('Name of the preset voice to use')
          .optional(),
      })
      .describe('Configuration for the prebuilt speaker to use')
      .passthrough()
      .optional(),
  })
  .describe('Configuration for the voice to use')
  .passthrough();

export const GeminiConfigSchema = GenerationCommonConfigSchema.extend({
  apiKey: z
    .string()
    .describe('Overrides the plugin-configured API key, if specified.')
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
  safetySettings: z
    .array(SafetySettingsSchema)
    .describe(
      'Adjust how likely you are to see responses that could be harmful. ' +
        'Content is blocked based on the probability that it is harmful.'
    )
    .optional(),
  codeExecution: z
    .union([z.boolean(), z.object({}).strict()])
    .describe('Enables the model to generate and run code.')
    .optional(),
  contextCache: z
    .boolean()
    .describe(
      'Context caching allows you to save and reuse precomputed input ' +
        'tokens that you wish to use repeatedly.'
    )
    .optional(),
  functionCallingConfig: z
    .object({
      mode: z.enum(['MODE_UNSPECIFIED', 'AUTO', 'ANY', 'NONE']).optional(),
      allowedFunctionNames: z.array(z.string()).optional(),
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
  responseModalities: z
    .array(z.enum(['TEXT', 'IMAGE', 'AUDIO']))
    .describe('The modalities to be used in response.')
    .optional(),
  googleSearchRetrieval: z
    .union([z.boolean(), z.object({}).passthrough()])
    .describe(
      'Retrieve public web data for grounding, powered by Google Search.'
    )
    .optional(),
  fileSearch: z
    .object({
      fileSearchStoreNames: z
        .array(z.string())
        .describe(
          'The names of the fileSearchStores to retrieve from. ' +
            'Example: fileSearchStores/my-file-search-store-123'
        ),
      metadataFilter: z
        .string()
        .optional()
        .describe(
          'Metadata filter to apply to the semantic retrieval documents and chunks.'
        ),
      topK: z
        .number()
        .optional()
        .describe('The number of semantic retrieval chunks to retrieve.'),
    })
    .passthrough()
    .optional(),
  urlContext: z
    .union([z.boolean(), z.object({}).passthrough()])
    .describe('Return grounding metadata from links included in the query')
    .optional(),
  temperature: z
    .number()
    .min(0)
    .max(2)
    .describe(
      GenerationCommonConfigDescriptions.temperature +
        ' The default value is 1.0.'
    )
    .optional(),
  topP: z
    .number()
    .min(0)
    .max(1)
    .describe(
      GenerationCommonConfigDescriptions.topP + ' The default value is 0.95.'
    )
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
export type GeminiConfig = z.infer<GeminiConfigSchemaType>;

export const GeminiTtsConfigSchema = GeminiConfigSchema.extend({
  speechConfig: z
    .object({
      voiceConfig: VoiceConfigSchema.optional(),
      multiSpeakerVoiceConfig: z
        .object({
          speakerVoiceConfigs: z
            .array(
              z
                .object({
                  speaker: z.string().describe('Name of the speaker to use'),
                  voiceConfig: VoiceConfigSchema,
                })
                .describe(
                  'Configuration for a single speaker in a multi speaker setup'
                )
                .passthrough()
            )
            .describe('Configuration for all the enabled speaker voices'),
        })
        .describe('Configuration for multi-speaker setup')
        .passthrough()
        .optional(),
    })
    .describe('Speech generation config')
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiTtsConfigSchemaType = typeof GeminiTtsConfigSchema;
export type GeminiTtsConfig = z.infer<GeminiTtsConfigSchemaType>;

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

export const GemmaConfigSchema = GeminiConfigSchema.extend({
  temperature: z
    .number()
    .min(0.0)
    .max(1.0)
    .describe(
      GenerationCommonConfigDescriptions.temperature +
        ' The default value is 1.0.'
    )
    .optional(),
}).passthrough();
export type GemmaConfigSchemaType = typeof GemmaConfigSchema;
export type GemmaConfig = z.infer<GemmaConfigSchemaType>;

// This contains all the Gemini config schema types
type ConfigSchemaType =
  | GeminiConfigSchemaType
  | GeminiTtsConfigSchemaType
  | GeminiImageConfigSchemaType
  | GemmaConfigSchemaType;
type ConfigSchema = z.infer<ConfigSchemaType>;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = GeminiConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `googleai/${name}`,
    configSchema,
    info: info ?? {
      supports: {
        multiturn: true,
        media: true,
        tools: true,
        toolChoice: true,
        systemRole: true,
        constrained: 'no-tools',
        output: ['text', 'json'],
      },
    },
  });
}

const GENERIC_MODEL = commonRef('gemini');
const GENERIC_TTS_MODEL = commonRef(
  'gemini-tts',
  {
    supports: {
      multiturn: false,
      media: false,
      tools: false,
      toolChoice: false,
      systemRole: false,
      constrained: 'no-tools',
    },
  },
  GeminiTtsConfigSchema
);
const GENERIC_IMAGE_MODEL = commonRef(
  'gemini-image',
  {
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      toolChoice: true,
      systemRole: true,
      constrained: 'no-tools',
    },
  },
  GeminiImageConfigSchema
);
const GENERIC_GEMMA_MODEL = commonRef(
  'gemma-generic',
  undefined,
  GemmaConfigSchema
);

const KNOWN_GEMINI_MODELS = {
  'gemini-3.1-pro-preview-customtools': commonRef(
    'gemini-3.1-pro-preview-customtools'
  ),
  'gemini-3.1-pro-preview': commonRef('gemini-3.1-pro-preview'),
  'gemini-3-flash-preview': commonRef('gemini-3-flash-preview'),
  'gemini-3-pro-preview': commonRef('gemini-3-pro-preview'),
  'gemini-2.5-pro': commonRef('gemini-2.5-pro'),
  'gemini-2.5-flash': commonRef('gemini-2.5-flash'),
  'gemini-2.5-flash-lite': commonRef('gemini-2.5-flash-lite'),
};
export type KnownGeminiModels = keyof typeof KNOWN_GEMINI_MODELS;
export type GeminiModelName = `gemini-${string}`;
export function isGeminiModelName(value: string): value is GeminiModelName {
  return (
    value.startsWith('gemini-') &&
    !value.endsWith('-tts') &&
    !value.includes('-image')
  );
}

const KNOWN_TTS_MODELS = {
  'gemini-2.5-flash-preview-tts': commonRef(
    'gemini-2.5-flash-preview-tts',
    { ...GENERIC_TTS_MODEL.info },
    GeminiTtsConfigSchema
  ),
  'gemini-2.5-pro-preview-tts': commonRef(
    'gemini-2.5-pro-preview-tts',
    { ...GENERIC_TTS_MODEL.info },
    GeminiTtsConfigSchema
  ),
};
export type KnownTtsModels = keyof typeof KNOWN_TTS_MODELS;
export type TTSModelName = `gemini-${string}-tts`;
export function isTTSModelName(value: string): value is TTSModelName {
  return value.startsWith('gemini-') && value.endsWith('-tts');
}

const KNOWN_IMAGE_MODELS = {
  'gemini-3-pro-image-preview': commonRef(
    'gemini-3-pro-image-preview',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema
  ),
  'gemini-2.5-flash-image': commonRef(
    'gemini-2.5-flash-image',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema
  ),
} as const;
export type KnownImageModels = keyof typeof KNOWN_IMAGE_MODELS;
export type ImageModelName = `gemini-${string}-image${string}`;
export function isImageModelName(value: string): value is ImageModelName {
  return value.startsWith('gemini-') && value.includes('-image');
}

const KNOWN_GEMMA_MODELS = {
  'gemma-3-12b-it': commonRef('gemma-3-12b-it', undefined, GemmaConfigSchema),
  'gemma-3-1b-it': commonRef('gemma-3-1b-it', undefined, GemmaConfigSchema),
  'gemma-3-27b-it': commonRef('gemma-3-27b-it', undefined, GemmaConfigSchema),
  'gemma-3-4b-it': commonRef('gemma-3-4b-it', undefined, GemmaConfigSchema),
  'gemma-3n-e4b-it': commonRef('gemma-3n-e4b-it', undefined, GemmaConfigSchema),
} as const;
export type KnownGemmaModels = keyof typeof KNOWN_GEMMA_MODELS;
export type GemmaModelName = `gemma-${string}`;
export function isGemmaModelName(value: string): value is GemmaModelName {
  return value.startsWith('gemma-');
}

const KNOWN_MODELS = {
  ...KNOWN_GEMINI_MODELS,
  ...KNOWN_TTS_MODELS,
  ...KNOWN_IMAGE_MODELS,
  ...KNOWN_GEMMA_MODELS,
};

export function model(
  version: string,
  config: ConfigSchema = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isTTSModelName(name)) {
    return modelRef({
      name: `googleai/${name}`,
      config,
      configSchema: GeminiTtsConfigSchema,
      info: { ...GENERIC_TTS_MODEL.info },
    });
  }

  if (isImageModelName(name)) {
    return modelRef({
      name: `googleai/${name}`,
      config,
      configSchema: GeminiImageConfigSchema,
      info: { ...GENERIC_IMAGE_MODEL.info },
    });
  }

  if (isGemmaModelName(name)) {
    return modelRef({
      name: `googleai/${name}`,
      config,
      configSchema: GemmaConfigSchema,
      info: { ...GENERIC_GEMMA_MODEL.info },
    });
  }

  return modelRef({
    name: `googleai/${name}`,
    config,
    configSchema: GeminiConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

// Takes a full list of models, filters for current Gemini models only
// and returns a modelActionMetadata for each.
export function listActions(models: Model[]): ActionMetadata[] {
  return (
    models
      .filter((m) => m.supportedGenerationMethods.includes('generateContent'))
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

/**
 * Defines a new GoogleAI Gemini model.
 */
export function defineModel(
  name: string,
  pluginOptions?: GoogleAIPluginOptions
): ModelAction {
  checkApiKey(pluginOptions?.apiKey);
  const ref = model(name);
  const clientOptions: ClientOptions = {
    apiVersion: pluginOptions?.apiVersion,
    baseUrl: pluginOptions?.baseUrl,
    customHeaders: pluginOptions?.customHeaders,
  };

  const middleware: ModelMiddleware[] = [];
  if (ref.info?.supports?.media) {
    // For Gemini 2.0, external URLs are not supported, so we must download.
    // For newer models, we can pass the URL directly.
    const supportsExternalUrls = !name.startsWith('gemini-2.0');

    middleware.push(
      downloadRequestMedia({
        maxBytes: MAX_INLINE_MEDIA_BYTES,
        // don't download files that have been uploaded using the Files API
        // or external URLs supported by the model
        filter: (part) => {
          try {
            const url = new URL(part.media.url);
            if (
              // Gemini can handle these URLs
              [
                'generativelanguage.googleapis.com',
                'www.youtube.com',
                'youtube.com',
                'youtu.be',
              ].includes(url.hostname)
            )
              return false;

            // If model supports external URLs, allow http/https URLs to pass through
            if (
              supportsExternalUrls &&
              (url.protocol === 'https:' || url.protocol === 'http:')
            ) {
              return false;
            }
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
      use: middleware,
    },
    async (request, { streamingRequested, sendChunk, abortSignal }) => {
      const clientOpt = calculateRequestOptions(
        { ...clientOptions, signal: abortSignal },
        request.config
      );

      // Make a copy so that modifying the request will not produce side-effects
      const messages = [...request.messages];
      if (messages.length === 0) throw new Error('No messages provided.');

      // Gemini does not support messages with role system and instead expects
      // systemInstructions to be provided as a separate input. The first
      // message detected with role=system will be used for systemInstructions.
      let systemInstruction: GeminiMessage | undefined = undefined;
      const systemMessage = messages.find((m) => m.role === 'system');
      if (systemMessage) {
        messages.splice(messages.indexOf(systemMessage), 1);
        systemInstruction = toGeminiSystemInstruction(systemMessage);
      }

      const tools: Tool[] = [];
      if (request.tools?.length) {
        tools.push({
          functionDeclarations: request.tools.map(toGeminiTool),
        });
      }

      const requestOptions: ConfigSchema = {
        ...request.config,
      };
      const {
        apiKey: apiKeyFromConfig,
        safetySettings: safetySettingsFromConfig,
        codeExecution: codeExecutionFromConfig,
        version: versionFromConfig,
        functionCallingConfig,
        googleSearchRetrieval,
        fileSearch,
        urlContext,
        tools: toolsFromConfig,
        ...restOfConfigOptions
      } = requestOptions;

      if (codeExecutionFromConfig) {
        tools.push({
          codeExecution:
            codeExecutionFromConfig === true ? {} : codeExecutionFromConfig,
        });
      }

      if (toolsFromConfig) {
        tools.push(...(toolsFromConfig as any[]));
      }

      if (googleSearchRetrieval) {
        tools.push({
          googleSearch:
            googleSearchRetrieval === true ? {} : googleSearchRetrieval,
        } as GoogleSearchRetrievalTool);
      }

      if (fileSearch) {
        tools.push({
          fileSearch,
        });
      }

      if (urlContext) {
        tools.push({
          urlContext: urlContext === true ? {} : urlContext,
        } as UrlContextTool);
      }

      let toolConfig: ToolConfig | undefined;
      if (functionCallingConfig) {
        toolConfig = {
          functionCallingConfig: {
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

      // Cannot use tools with JSON mode
      const jsonMode =
        request.output?.format === 'json' ||
        (request.output?.contentType === 'application/json' &&
          tools.length === 0);

      const generationConfig: GenerationConfig = {
        ...removeClientOptionOverrides(restOfConfigOptions),
        candidateCount: request.candidates || undefined,
        responseMimeType: jsonMode ? 'application/json' : undefined,
      };

      if (request.output?.constrained && jsonMode) {
        if (pluginOptions?.legacyResponseSchema) {
          generationConfig.responseSchema = cleanSchema(request.output.schema);
        } else {
          generationConfig.responseJsonSchema = request.output.schema;
        }
      }

      const msg = toGeminiMessage(messages[messages.length - 1], ref);

      let generateContentRequest: GenerateContentRequest = {
        systemInstruction,
        generationConfig,
        tools: tools.length ? tools : undefined,
        toolConfig,
        safetySettings: safetySettingsFromConfig?.filter(
          (setting) => setting.category !== 'HARM_CATEGORY_UNSPECIFIED'
        ) as SafetySetting[],
        contents: messages.map((message) => toGeminiMessage(message, ref)),
      };

      const modelVersion = versionFromConfig || extractVersion(ref);

      const generateApiKey = calculateApiKey(
        pluginOptions?.apiKey,
        requestOptions.apiKey
      );

      const callGemini = async () => {
        let response: GenerateContentResponse;

        if (streamingRequested) {
          const result = await generateContentStream(
            generateApiKey,
            modelVersion,
            generateContentRequest,
            clientOpt
          );
          const chunks: CandidateData[] = [];
          for await (const item of result.stream) {
            item.candidates?.forEach((candidate) => {
              const c = fromGeminiCandidate(candidate, chunks);
              chunks.push(c);
              sendChunk({
                index: c.index,
                content: c.message.content,
              });
            });
          }
          response = await result.response;
        } else {
          response = await generateContent(
            generateApiKey,
            modelVersion,
            generateContentRequest,
            clientOpt
          );
        }

        const candidates = response.candidates || [];
        if (response.candidates?.['undefined']) {
          candidates.push(response.candidates['undefined']);
        }
        if (!candidates.length) {
          throw new GenkitError({
            status: 'FAILED_PRECONDITION',
            message: 'No valid candidates returned.',
          });
        }

        const candidateData =
          candidates.map((c) => fromGeminiCandidate(c)) || [];

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

      // If debugTraces is enabled, we wrap the actual model call with a span, add raw
      // API params as for input.
      return pluginOptions?.experimental_debugTraces
        ? await runInNewSpan(
            {
              metadata: {
                name: streamingRequested ? 'sendMessageStream' : 'sendMessage',
              },
            },
            async (metadata) => {
              metadata.input = {
                apiEndpoint: getGoogleAIUrl({
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

export const TEST_ONLY = { KNOWN_MODELS };
