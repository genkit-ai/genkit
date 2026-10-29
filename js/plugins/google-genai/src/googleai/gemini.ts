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

import {
  ActionMetadata,
  GENKIT_UI_METADATA,
  GENKIT_UI_WIDGETS,
  GenkitError,
  annotateSchema,
  modelActionMetadata,
  z,
} from 'genkit';
import { logger } from 'genkit/logging';
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
import {
  fromGeminiCandidate,
  toGeminiFunctionModeEnum,
  toGeminiMessage,
  toGeminiSystemInstruction,
  toGeminiTool,
} from '../common/converters.js';
import {
  applyLatLngToGoogleMapsTools,
  createInteractionChunkConverter,
  fromInteractionSync,
  mergeStreamedAudio,
  removeId3FromAudio,
  selectInteractionInput,
  toInteractionConfigTool,
  toInteractionGenerationConfig,
  toInteractionResponseFormat,
  toInteractionSteps,
  toInteractionTool,
  toInteractionToolChoice,
} from '../common/interaction-converters.js';
import {
  InteractionTool,
  ModelGenerationConfig,
} from '../common/interaction-types.js';
import { isKnownKey, warnDroppedConfigField } from '../common/utils.js';
import {
  createInteraction,
  createInteractionStream,
  generateContent,
  generateContentStream,
} from './client.js';
import {
  ClientOptions,
  CreateInteractionRequest,
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
  isObject,
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

/**
 * Reports whether a safety setting is equivalent to the model default (no
 * additional filtering), i.e. it can be omitted without changing behavior.
 *
 * Per the Gemini API docs, the adjustable safety filters are Off by default for
 * current Gemini models, so `BLOCK_NONE` requests the same behavior as sending
 * no safety settings. `HARM_CATEGORY_UNSPECIFIED` entries are also dropped on
 * the generateContent path.
 * See https://ai.google.dev/gemini-api/docs/safety-settings
 */
function isDefaultSafetySetting(setting: {
  category?: string;
  threshold?: string;
}): boolean {
  return (
    setting.category === 'HARM_CATEGORY_UNSPECIFIED' ||
    setting.threshold === 'BLOCK_NONE'
  );
}

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
  safetySettings: annotateSchema(
    z
      .array(SafetySettingsSchema)
      .describe(
        'Adjust how likely you are to see responses that could be harmful. ' +
          'Content is blocked based on the probability that it is harmful.'
      )
      .optional(),
    { [GENKIT_UI_METADATA.WIDGET]: GENKIT_UI_WIDGETS.SAFETY_SETTINGS }
  ),
  codeExecution: z
    .union([z.boolean(), z.object({}).strict()])
    .describe('Enables the model to generate and run code.')
    .optional(),
  // TODO(v2): Remove. This plugin does not implement context caching, so this
  // field has no effect (it is passed through on generateContent models and
  // dropped with a warning on Interactions models, which reject it). Kept for
  // now because removing it is a breaking change.
  /** @deprecated Not implemented by this plugin. Will be removed in v2. */
  contextCache: z
    .boolean()
    .describe(
      'Context caching allows you to save and reuse precomputed input ' +
        'tokens that you wish to use repeatedly. Not currently implemented ' +
        'by this plugin.'
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
  googleSearchRetrieval: z // some models use this, some use just googleSearch
    .union([z.boolean(), z.object({}).passthrough()])
    .describe(
      'Retrieve public web data for grounding, powered by Google Search.'
    )
    .optional(),
  googleSearch: z
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
  retrievalConfig: z
    .object({
      latLng: z
        .object({
          latitude: z.number().optional(),
          longitude: z.number().optional(),
        })
        .optional(),
      languageCode: z.string().optional(),
    })
    .passthrough()
    .describe('Configuration for retrieval tools.')
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
  serviceTier: z
    .union([z.enum(['standard', 'flex', 'priority']), z.string()])
    .describe('Service tier for the Gemini API.')
    .optional(),
  previousInteractionId: z
    .string()
    .describe('The ID of the previous interaction, if any.')
    .optional(),
  store: z
    .boolean()
    .describe(
      'Interactions API models: whether to store the interaction for later ' +
        'retrieval (defaults to false). generateContent models: whether ' +
        "the server logs the request for the API key's project; sent only " +
        "when set (otherwise the project's logging setting applies)."
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
  audioFormat: z
    .object({
      mimeType: z
        .union([
          z.enum(['audio/wav', 'audio/l16', 'audio/mulaw', 'audio/alaw']),
          // Allows other formats. `string & {}` (rather than z.string()) keeps
          // the listed ones in TypeScript autocomplete.
          z.custom<string & {}>((val) => typeof val === 'string'),
        ])
        .describe(
          'Audio encoding. Defaults to audio/wav (with a header) for ' +
            'non-streamed requests and audio/l16 (raw PCM, no header) for ' +
            'streamed requests.'
        )
        .optional(),
      sampleRate: z
        .number()
        .int()
        .describe('Sample rate in Hz, for example 24000, 16000 or 8000.')
        .optional(),
    })
    .describe(
      'Format of the generated audio. Supported by models that use the ' +
        'Interactions API (for example gemini-3.8-flash-tts).'
    )
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
          '1:4',
          '1:8',
          '2:3',
          '3:2',
          '3:4',
          '4:1',
          '4:3',
          '4:5',
          '5:4',
          '8:1',
          '9:16',
          '16:9',
          '21:9',
        ])
        .optional(),
      imageSize: z
        .enum([
          '256',
          '256P',
          '256PX',
          '512',
          '512P',
          '512PX',
          '1K',
          '2K',
          '4K',
        ])
        .optional(),
    })
    .passthrough()
    .optional(),
  google_search: z
    .object({
      searchTypes: z
        .object({
          webSearch: z.object({}).optional(),
          imageSearch: z.object({}).optional(),
        })
        .optional(),
    })
    .describe(
      'Retrieve public web data for grounding, powered by Google Search.'
    )
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiImageConfigSchemaType = typeof GeminiImageConfigSchema;
export type GeminiImageConfig = z.infer<GeminiImageConfigSchemaType>;

/**
 * Config for video generation models such as gemini-omni-1.1-flash.
 * See https://ai.google.dev/gemini-api/docs/omni
 */
export const GeminiVideoConfigSchema = GeminiConfigSchema.extend({
  videoFormat: z
    .object({
      aspectRatio: z
        .union([
          z.enum(['16:9', '9:16']),
          // Allows other values. `string & {}` (rather than z.string()) keeps
          // the listed ones in TypeScript autocomplete.
          z.custom<string & {}>((val) => typeof val === 'string'),
        ])
        .describe(
          'Aspect ratio of the generated video. Defaults to 16:9 (landscape).'
        )
        .optional(),
      resolution: z
        .union([
          z.enum(['360p', '720p', '1080p', '4k']),
          z.custom<string & {}>((val) => typeof val === 'string'),
        ])
        .describe(
          'Resolution of the generated video. Defaults to 720p. 1080p and ' +
            '4k are upscaled.'
        )
        .optional(),
      delivery: z
        .union([
          z.enum(['uri']),
          z.custom<string & {}>((val) => typeof val === 'string'),
        ])
        .describe(
          'How the video is returned. By default it is returned inline. ' +
            '"uri" returns a Files API URI instead, which is recommended ' +
            'for large videos (requires store: true).'
        )
        .optional(),
    })
    .describe('Format of the generated video.')
    .passthrough()
    .optional(),
  videoConfig: z
    .object({
      task: z
        .union([
          z.enum([
            'text_to_video',
            'image_to_video',
            'reference_to_video',
            'edit',
            'extend',
          ]),
          z.custom<string & {}>((val) => typeof val === 'string'),
        ])
        .describe(
          'The kind of generation to perform. If not set, the model infers ' +
            'it from the prompt and inputs. Setting it adds constraints, so ' +
            'only use it when prompting alone does not give the intended mode.'
        )
        .optional(),
    })
    .describe('Video generation settings.')
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiVideoConfigSchemaType = typeof GeminiVideoConfigSchema;
export type GeminiVideoConfig = z.infer<GeminiVideoConfigSchemaType>;

export const GemmaConfigSchema = GeminiConfigSchema.extend({
  // No range check: the accepted range depends on the model, so the value is
  // sent as is and the server validates it.
  temperature: z
    .number()
    .describe(GenerationCommonConfigDescriptions.temperature)
    .optional(),
}).passthrough();
export type GemmaConfigSchemaType = typeof GemmaConfigSchema;
export type GemmaConfig = z.infer<GemmaConfigSchemaType>;

// This contains all the Gemini config schema types
type ConfigSchemaType =
  | GeminiConfigSchemaType
  | GeminiTtsConfigSchemaType
  | GeminiImageConfigSchemaType
  | GeminiVideoConfigSchemaType
  | GemmaConfigSchemaType;
type ConfigSchema = z.infer<ConfigSchemaType>;

const modelInteractionsMap = new Map<string, boolean>();

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = GeminiConfigSchema,
  useInteractions: boolean = true
): ModelReference<ConfigSchemaType> {
  modelInteractionsMap.set(name, useInteractions);

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
        constrained: 'all',
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
      constrained: 'none',
      output: ['media'],
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
      constrained: 'all',
    },
  },
  GeminiImageConfigSchema
);
// Per the Omni guide: text, image and video in, video (with audio) out; edits
// continue a stored interaction. Tools and structured output aren't
// documented for video generation. The guide says system instructions aren't
// supported, but they are applied (checked live, Oct 2026).
const GENERIC_VIDEO_MODEL = commonRef(
  'gemini-video',
  {
    supports: {
      multiturn: true,
      media: true,
      tools: false,
      toolChoice: false,
      systemRole: true,
      constrained: 'none',
      output: ['media'],
    },
  },
  GeminiVideoConfigSchema
);
const GENERIC_GEMMA_MODEL = commonRef(
  'gemma-generic',
  undefined,
  GemmaConfigSchema
);

const KNOWN_GEMINI_MODELS = {
  'gemini-pro-latest': commonRef('gemini-pro-latest'),
  'gemini-flash-latest': commonRef('gemini-flash-latest'),
  'gemini-flash-lite-latest': commonRef('gemini-flash-lite-latest'),
  'gemini-4-argon': commonRef('gemini-4-argon'),
  'gemini-3.8-flash': commonRef('gemini-3.8-flash'),
  'gemini-3.6-flash': commonRef('gemini-3.6-flash'),
  'gemini-3.5-flash-lite': commonRef('gemini-3.5-flash-lite'),

  'gemini-3.1-flash-lite': commonRef(
    'gemini-3.1-flash-lite',
    undefined,
    GeminiConfigSchema,
    false
  ),
  'gemini-3.1-pro-preview-customtools': commonRef(
    'gemini-3.1-pro-preview-customtools',
    undefined,
    GeminiConfigSchema,
    false
  ),
  'gemini-3.1-pro-preview': commonRef(
    'gemini-3.1-pro-preview',
    undefined,
    GeminiConfigSchema,
    false
  ),
  'gemini-3-flash-preview': commonRef(
    'gemini-3-flash-preview',
    undefined,
    GeminiConfigSchema,
    false
  ),
};
export type KnownGeminiModels = keyof typeof KNOWN_GEMINI_MODELS;
export type GeminiModelName = `gemini-${string}`;
export function isGeminiModelName(value: string): value is GeminiModelName {
  return (
    value.startsWith('gemini-') &&
    !isTTSModelName(value) &&
    !isImageModelName(value) &&
    !isVideoModelName(value)
  );
}

const KNOWN_TTS_MODELS = {
  'gemini-3.8-flash-tts': commonRef(
    'gemini-3.8-flash-tts',
    { ...GENERIC_TTS_MODEL.info },
    GeminiTtsConfigSchema
  ),
  'gemini-3.8-flash-lite-tts': commonRef(
    'gemini-3.8-flash-lite-tts',
    { ...GENERIC_TTS_MODEL.info },
    GeminiTtsConfigSchema
  ),
} as const;
export type KnownTtsModels = keyof typeof KNOWN_TTS_MODELS;
export type TTSModelName = `gemini-${string}-tts${string}`;
export function isTTSModelName(value: string): value is TTSModelName {
  return value.startsWith('gemini-') && value.includes('-tts');
}

const KNOWN_IMAGE_MODELS = {
  // Function calling works, but structured output doesn't: with a schema the
  // model still returns only an image (checked live, Oct 2026). 'none' makes
  // Genkit ask for JSON in the prompt instead of sending a schema.
  'gemini-nano-banana-2.1': commonRef(
    'gemini-nano-banana-2.1',
    {
      ...GENERIC_IMAGE_MODEL.info,
      supports: { ...GENERIC_IMAGE_MODEL.info?.supports, constrained: 'none' },
    },
    GeminiImageConfigSchema
  ),
  'gemini-3.1-flash-lite-image': commonRef(
    'gemini-3.1-flash-lite-image',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema
  ),
  'gemini-3-pro-image': commonRef(
    'gemini-3-pro-image',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema,
    false
  ),
} as const;
export type KnownImageModels = keyof typeof KNOWN_IMAGE_MODELS;
export type ImageModelName =
  | `gemini-${string}-image${string}`
  | `gemini-nano-banana-${string}`;
/** Image models: `-image` names and the Nano Banana names (as in vertexAI). */
export function isImageModelName(value: string): value is ImageModelName {
  return (
    value.startsWith('gemini-') &&
    (value.includes('-image') || value.startsWith('gemini-nano-banana-'))
  );
}

const KNOWN_VIDEO_MODELS = {
  'gemini-omni-1.1-flash': commonRef(
    'gemini-omni-1.1-flash',
    { ...GENERIC_VIDEO_MODEL.info },
    GeminiVideoConfigSchema
  ),
} as const;
export type KnownVideoModels = keyof typeof KNOWN_VIDEO_MODELS;
export type VideoModelName = `gemini-omni-${string}`;
export function isVideoModelName(value: string): value is VideoModelName {
  return value.startsWith('gemini-omni-');
}

const KNOWN_GEMMA_MODELS = {
  'gemma-4-26b-a4b-it': commonRef(
    'gemma-4-26b-a4b-it',
    undefined,
    GemmaConfigSchema,
    false
  ),
  'gemma-4-31b-it': commonRef(
    'gemma-4-31b-it',
    undefined,
    GemmaConfigSchema,
    false
  ),
} as const;
export type KnownGemmaModels = keyof typeof KNOWN_GEMMA_MODELS;
export type GemmaModelName = `gemma-${string}`;
export function isGemmaModelName(value: string): value is GemmaModelName {
  return value.startsWith('gemma-');
}

// Models we no longer suggest, but that still work: they still resolve and
// are still registered as actions, but aren't in the Known*Models types, so
// they don't show up in autocomplete. Set `stage` on each one (keeping the
// default `supports`):
// - 'deprecated': deprecated by Google, or < 1 month from shutdown.
// - 'legacy': not deprecated, but no longer offered to new projects.
const HIDDEN_MODELS = {
  // Gemini 2.5: Google limits access to accounts that used it before; new
  // projects should use newer models.
  'gemini-2.5-pro': commonRef(
    'gemini-2.5-pro',
    { ...GENERIC_MODEL.info, stage: 'legacy' },
    GeminiConfigSchema,
    false
  ),
  'gemini-2.5-flash': commonRef(
    'gemini-2.5-flash',
    { ...GENERIC_MODEL.info, stage: 'legacy' },
    GeminiConfigSchema,
    false
  ),
  'gemini-2.5-flash-lite': commonRef(
    'gemini-2.5-flash-lite',
    { ...GENERIC_MODEL.info, stage: 'legacy' },
    GeminiConfigSchema,
    false
  ),
  'gemini-2.5-flash-image': commonRef(
    'gemini-2.5-flash-image',
    { ...GENERIC_IMAGE_MODEL.info, stage: 'legacy' },
    GeminiImageConfigSchema,
    false
  ),

  // Shut down on Nov 17 2026.
  'gemini-2.5-flash-preview-tts': commonRef(
    'gemini-2.5-flash-preview-tts',
    { ...GENERIC_TTS_MODEL.info, stage: 'deprecated' },
    GeminiTtsConfigSchema,
    false
  ),
  'gemini-2.5-pro-preview-tts': commonRef(
    'gemini-2.5-pro-preview-tts',
    { ...GENERIC_TTS_MODEL.info, stage: 'deprecated' },
    GeminiTtsConfigSchema,
    false
  ),
  'gemini-3.1-flash-tts-preview': commonRef(
    'gemini-3.1-flash-tts-preview',
    { ...GENERIC_TTS_MODEL.info, stage: 'deprecated' },
    GeminiTtsConfigSchema,
    false
  ),

  // Deprecated, not shutting down: the server routes requests for it to
  // gemini-3.6-flash (per the deprecations page, Oct 2026). Still sent through
  // generateContent, as before: the routing happens on the server.
  'gemini-3.5-flash': commonRef(
    'gemini-3.5-flash',
    { ...GENERIC_MODEL.info, stage: 'deprecated' },
    GeminiConfigSchema,
    false
  ),

  // Deprecated (no shutdown date yet); replaced by gemini-nano-banana-2.1.
  'gemini-3.1-flash-image': commonRef(
    'gemini-3.1-flash-image',
    { ...GENERIC_IMAGE_MODEL.info, stage: 'deprecated' },
    GeminiImageConfigSchema,
    false
  ),
} as const;

const KNOWN_MODELS = {
  ...KNOWN_GEMINI_MODELS,
  ...KNOWN_TTS_MODELS,
  ...KNOWN_IMAGE_MODELS,
  ...KNOWN_VIDEO_MODELS,
  ...KNOWN_GEMMA_MODELS,
};

const ALL_MODELS = {
  ...HIDDEN_MODELS,
  ...KNOWN_MODELS,
};

export function model(
  version: string,
  config: ConfigSchema = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, ALL_MODELS)) {
    return ALL_MODELS[name].withConfig(config);
  }

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

  if (isVideoModelName(name)) {
    return modelRef({
      name: `googleai/${name}`,
      config,
      configSchema: GeminiVideoConfigSchema,
      info: { ...GENERIC_VIDEO_MODEL.info },
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
  return Object.keys(ALL_MODELS).map((name: string) =>
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
    experimental_debugTraces: pluginOptions?.experimental_debugTraces,
  };

  const middleware: ModelMiddleware[] = [];
  if (ref.info?.supports?.media) {
    middleware.push(
      downloadRequestMedia({
        maxBytes: MAX_INLINE_MEDIA_BYTES,
        // don't download files that have been uploaded using the Files API
        // or external URLs supported by the model
        filter: (part) => {
          try {
            const url = new URL(part.media.url);
            // Allow http/https URLs to pass through
            if (url.protocol === 'https:' || url.protocol === 'http:') {
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

      const modelVersion = request.config?.version || extractVersion(ref);
      const isGemma = isGemmaModelName(modelVersion);
      const useInteractions = modelInteractionsMap.get(modelVersion) ?? true;

      // Make a copy so that modifying the request will not produce side-effects
      const messages = request.messages.map((m) => ({ ...m }));
      if (messages.length === 0) throw new Error('No messages provided.');

      if (isGemma) {
        // Gemma does not allow previous thoughts
        messages.forEach((m) => {
          m.content = m.content.filter(
            (p) => !p.reasoning && !p.metadata?.thoughtSignature
          );
        });
      }

      // Gemini does not support messages with role system and instead expects
      // systemInstructions to be provided as a separate input. The first
      // message detected with role=system will be used for systemInstructions.
      let systemInstruction: GeminiMessage | undefined = undefined;
      let interactionsSystemInstruction: string | undefined = undefined;
      const systemMessage = messages.find((m) => m.role === 'system');
      if (systemMessage) {
        messages.splice(messages.indexOf(systemMessage), 1);
        systemInstruction = toGeminiSystemInstruction(systemMessage);
        if (
          useInteractions &&
          systemMessage.content.some((c) => c.text === undefined)
        ) {
          // Technically it's not the model itself, but 'useInteractions' or not,
          // however, useInteractions is determined by which model... so
          // this makes the most sense without dragging the user into
          // the nitty gritty of how their stuff is going through the backend.
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message:
              'System message contains non-text content which is not supported for this model.',
          });
        }
        interactionsSystemInstruction = systemMessage.content
          .map((c) => c.text)
          .join('\n');
      }

      const tools: Tool[] = [];
      const interactionsTools: InteractionTool[] = [];

      if (request.tools?.length) {
        if (useInteractions) {
          interactionsTools.push(...request.tools.map(toInteractionTool));
        } else {
          tools.push({
            functionDeclarations: request.tools.map(toGeminiTool),
          });
        }
      }

      const requestOptions: ConfigSchema = {
        ...request.config,
      };

      const {
        apiKey: apiKeyFromConfig,
        safetySettings: safetySettingsFromConfig,
        codeExecution: codeExecutionFromConfig,
        version: versionFromConfig,
        toolConfig: toolConfigConfig,
        functionCallingConfig,
        googleSearchRetrieval,
        google_search,
        googleSearch,
        fileSearch,
        urlContext,
        tools: toolsFromConfig,
        retrievalConfig,
        serviceTier,
        previousInteractionId: previousInteractionIdFromConfig,
        store: storeFromConfig,
        responseModalities: responseModalitiesFromConfig,
        // Interactions-only: sent as `response_format` below.
        audioFormat: audioFormatFromConfig,
        videoFormat: videoFormatFromConfig,
        ...restOfConfigOptions
      } = requestOptions;

      if (!useInteractions) {
        // generateContent rejects unknown generation_config fields with a
        // 400, and has no equivalent for these (e.g. the older preview TTS
        // models still use this schema).
        if (audioFormatFromConfig !== undefined) {
          warnDroppedConfigField(modelVersion, 'audioFormat');
        }
        if (videoFormatFromConfig !== undefined) {
          warnDroppedConfigField(modelVersion, 'videoFormat');
        }
      }

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

      if (googleSearch || google_search) {
        tools.push({
          google_search: google_search || googleSearch,
        }) as GoogleSearchRetrievalTool;
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

      if (useInteractions) {
        // The Gemini API's Interactions endpoint does not accept
        // `safety_settings` (it returns 400), even though the published
        // Interactions spec lists the field. The Gemini Enterprise Agent
        // Platform accepts it, so the vertexAI plugin should map safety
        // settings rather than copy this guard. Permissive settings can be
        // dropped safely: per the Gemini API docs, the adjustable safety
        // filters are Off by default for current Gemini models, so BLOCK_NONE
        // (and HARM_CATEGORY_UNSPECIFIED entries) never block anything beyond
        // the default. Built-in protections against core harms always apply.
        // See https://ai.google.dev/gemini-api/docs/safety-settings
        //
        // Blocking thresholds (BLOCK_ONLY_HIGH and stricter) cannot be honored,
        // so throw rather than silently weaken the requested filtering.
        const blockingSafetySettings = (safetySettingsFromConfig ?? []).filter(
          (setting) => !isDefaultSafetySetting(setting)
        );
        if (blockingSafetySettings.length > 0) {
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message:
              `safetySettings with blocking thresholds are not supported for model '${modelVersion}'. ` +
              'This model applies no additional safety filters by default ' +
              '(built-in protections against core harms still apply). ' +
              'Remove the safetySettings or set their thresholds to BLOCK_NONE.',
            detail: { safetySettings: blockingSafetySettings },
          });
        }
        if (toolConfigConfig) {
          logger.warn(
            'toolConfig is not supported for this model with the Interactions API and will be ignored.'
          );
        }
        if (request.candidates && request.candidates > 1) {
          logger.warn(
            'Multiple candidates are not supported for this model with the Interactions API; only one candidate will be returned.'
          );
        }
        if (Array.isArray(toolsFromConfig)) {
          interactionsTools.push(
            ...toolsFromConfig.map(toInteractionConfigTool)
          );
        }
        if (codeExecutionFromConfig) {
          interactionsTools.push(
            toInteractionConfigTool({ codeExecution: codeExecutionFromConfig })
          );
        }
        if (googleSearchRetrieval) {
          interactionsTools.push(
            toInteractionConfigTool({ googleSearch: googleSearchRetrieval })
          );
        }
        if (googleSearch || google_search) {
          const gs = google_search || googleSearch;
          interactionsTools.push(
            toInteractionConfigTool({ google_search: gs })
          );
        }
        if (fileSearch) {
          interactionsTools.push(toInteractionConfigTool({ fileSearch }));
        }
        if (urlContext) {
          interactionsTools.push(toInteractionConfigTool({ urlContext }));
        }
        if (retrievalConfig) {
          // There is no retrieval config on the Interactions path; the user
          // location is set on the google_maps tool instead.
          const { latLng, ...unsupportedRetrievalConfig } = retrievalConfig;
          if (
            latLng &&
            !applyLatLngToGoogleMapsTools(interactionsTools, latLng)
          ) {
            logger.warn(
              'retrievalConfig.latLng is only used with the googleMaps tool for this model and will be ignored.'
            );
          }
          const unsupportedKeys = Object.keys(unsupportedRetrievalConfig);
          if (unsupportedKeys.length > 0) {
            logger.warn(
              `retrievalConfig.${unsupportedKeys.join(', retrievalConfig.')} ` +
                'is not supported for this model with the Interactions API and will be ignored.'
            );
          }
        }
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

      if (toolConfigConfig) {
        // We need it in snake case or it doesn't work
        if (
          Object.hasOwnProperty.call(
            toolConfigConfig,
            'includeServerSideToolInvocations'
          )
        ) {
          toolConfigConfig['include_server_side_tool_invocations'] =
            toolConfigConfig['includeServerSideToolInvocations'];
          delete toolConfigConfig['includeServerSideToolInvocations'];
        }
        toolConfig = {
          ...toolConfig,
          ...toolConfigConfig,
        };
      }

      if (retrievalConfig) {
        toolConfig = {
          ...toolConfig,
          retrievalConfig,
        };
      }

      const jsonMode =
        request.output?.format === 'json' ||
        request.output?.contentType === 'application/json';

      const sanitizedConfigOptions = {
        ...removeClientOptionOverrides(restOfConfigOptions),
      };

      const interactionGenerationConfig: ModelGenerationConfig =
        toInteractionGenerationConfig(
          sanitizedConfigOptions,
          useInteractions
            ? (field) => warnDroppedConfigField(modelVersion, field)
            : undefined
        );

      if (useInteractions) {
        const toolChoice = toInteractionToolChoice(
          functionCallingConfig,
          request.toolChoice
        );
        if (toolChoice) {
          interactionGenerationConfig.tool_choice = toolChoice;
        }
      }

      const generationConfig: GenerationConfig = {
        ...sanitizedConfigOptions,
        ...(responseModalitiesFromConfig
          ? { responseModalities: responseModalitiesFromConfig }
          : {}),
        candidateCount: request.candidates || undefined,
        responseMimeType: jsonMode ? 'application/json' : undefined,
      };

      if (isTTSModelName(modelVersion)) {
        if (!generationConfig.responseModalities) {
          generationConfig.responseModalities = ['AUDIO'];
        }
      } else if (isImageModelName(modelVersion)) {
        if (!generationConfig.responseModalities) {
          generationConfig.responseModalities = ['TEXT', 'IMAGE'];
        }
      }

      if (request.output?.constrained && jsonMode) {
        if (pluginOptions?.legacyResponseSchema) {
          generationConfig.responseSchema = cleanSchema(request.output.schema);
        } else {
          generationConfig.responseJsonSchema = request.output.schema;
        }
      }

      const requestApiKey = calculateApiKey(
        pluginOptions?.apiKey,
        requestOptions.apiKey
      );

      if (useInteractions) {
        const store = storeFromConfig === true || pluginOptions?.store === true;
        const { messages: newMessages, previousInteractionId } =
          selectInteractionInput(
            messages,
            store,
            previousInteractionIdFromConfig
          );

        // Turn-level speech metadata (speaker/style) is only accepted together
        // with a speech_config; without one the server returns a generic 400
        // ("Request contains an invalid argument"), so explain what to set.
        const speechConfig = interactionGenerationConfig.speech_config;
        if (!speechConfig) {
          const hasSpeechMetadata = newMessages
            .filter((m) => m.role === 'user')
            .flatMap((m) => m.content)
            .some(
              (p) =>
                p.text !== undefined && isObject(p.metadata?.speechMetadata)
            );
          if (hasSpeechMetadata) {
            throw new GenkitError({
              status: 'INVALID_ARGUMENT',
              message:
                `speechMetadata (speaker/style) requires a voice for model ` +
                `'${modelVersion}'. Set ` +
                'config.speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName ' +
                '(or multiSpeakerVoiceConfig for multiple speakers).',
            });
          }
        }

        // Multi-speaker TTS requires every non-empty text part to name a
        // speaker that matches one of the configured speakers. The server's
        // own error talks about "text turns", so explain what to set instead.
        if (speechConfig && !Array.isArray(speechConfig)) {
          const speakerNames = speechConfig.speakers
            .map((s) => s.speaker)
            .filter((name): name is string => !!name);
          const textParts = newMessages
            .filter((m) => m.role === 'user')
            .flatMap((m) => m.content)
            .filter((p) => !!p.text?.trim());
          const invalidTurn = textParts.some((p) => {
            const speechMetadata = p.metadata?.speechMetadata;
            return (
              !isObject(speechMetadata) ||
              typeof speechMetadata.speaker !== 'string' ||
              !speakerNames.includes(speechMetadata.speaker)
            );
          });
          if (invalidTurn) {
            throw new GenkitError({
              status: 'INVALID_ARGUMENT',
              message:
                `Multi-speaker speech for model '${modelVersion}' requires ` +
                'each turn to be a separate text part with ' +
                '`metadata: { speechMetadata: { speaker } }` set to one of ' +
                `the configured speakers (${speakerNames.join(', ')}).`,
            });
          }
        }

        const req: CreateInteractionRequest = {
          system_instruction: interactionsSystemInstruction,
          model: modelVersion,
          tools: interactionsTools.length ? interactionsTools : undefined,
          generation_config: interactionGenerationConfig,
          stream: streamingRequested,
          input: toInteractionSteps(newMessages),
          service_tier: serviceTier,
          store,
        };

        // One `response_format` with an entry per output type (the
        // deprecated `response_modalities` and `generation_config
        // .image_config` aren't sent). TTS and image models ask for the same
        // outputs as before by default.
        const responseFormat = toInteractionResponseFormat({
          modalities:
            responseModalitiesFromConfig ??
            (isTTSModelName(modelVersion)
              ? ['AUDIO']
              : isImageModelName(modelVersion)
                ? ['TEXT', 'IMAGE']
                : undefined),
          imageConfig:
            'imageConfig' in requestOptions
              ? requestOptions.imageConfig
              : undefined,
          audioFormat: audioFormatFromConfig,
          videoFormat: videoFormatFromConfig,
          json: jsonMode
            ? {
                schema: request.output?.constrained
                  ? pluginOptions?.legacyResponseSchema
                    ? cleanSchema(request.output.schema)
                    : request.output.schema
                  : undefined,
              }
            : undefined,
        });
        if (responseFormat) {
          req.response_format = responseFormat;
        }

        if (
          previousInteractionId &&
          typeof previousInteractionId === 'string'
        ) {
          req.previous_interaction_id = previousInteractionId;
        }
        if (!streamingRequested) {
          const response = await createInteraction(
            requestApiKey,
            req,
            clientOpt
          );
          const out = fromInteractionSync(response);
          return out;
        } else {
          const result = await createInteractionStream(
            requestApiKey,
            req,
            clientOpt
          );
          const toChunkParts = createInteractionChunkConverter();
          for await (const event of result.stream) {
            const chunkParts = toChunkParts(event).map(removeId3FromAudio);
            if (chunkParts.length > 0) {
              sendChunk({
                index: 0,
                content: chunkParts,
              });
            }
          }
          const response = await result.response;
          const out = fromInteractionSync(response);
          // Streamed audio arrives as many headerless chunks (raw PCM by
          // default); the final response joins them into one part of the
          // same format.
          if (out.message?.content) {
            out.message.content = mergeStreamedAudio(
              out.message.content.map(removeId3FromAudio)
            );
          }
          return out;
        }
      }

      // There's no stored conversation to continue on generateContent.
      if (previousInteractionIdFromConfig) {
        logger.warn(
          'previousInteractionId is not supported for this model and will be ignored.'
        );
      }

      // On generateContent, `store` only controls whether the server logs
      // this request for the API key's project (viewable in AI Studio's Logs
      // and Datasets); there's nothing to continue. Sent only when set;
      // otherwise the project's logging setting applies.
      const generateContentStore = storeFromConfig ?? pluginOptions?.store;

      let generateContentRequest: GenerateContentRequest = {
        systemInstruction,
        generationConfig,
        tools: tools.length ? tools : undefined,
        toolConfig,
        safetySettings: safetySettingsFromConfig?.filter(
          (setting) => setting.category !== 'HARM_CATEGORY_UNSPECIFIED'
        ) as SafetySetting[],
        contents: messages.map((message) => toGeminiMessage(message, ref)),
        serviceTier,
        store: generateContentStore,
      };

      let response: GenerateContentResponse;

      if (streamingRequested) {
        const result = await generateContentStream(
          requestApiKey,
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
          requestApiKey,
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

      const candidateData = candidates.map((c) => fromGeminiCandidate(c)) || [];

      return {
        candidates: candidateData,
        custom: response,
        usage: {
          ...getBasicUsageStats(request.messages, candidateData),
          inputTokens: response.usageMetadata?.promptTokenCount,
          outputTokens: response.usageMetadata?.candidatesTokenCount,
          thoughtsTokens: response.usageMetadata?.thoughtsTokenCount,
          totalTokens: response.usageMetadata?.totalTokenCount,
          cachedContentTokens: response.usageMetadata?.cachedContentTokenCount,
        },
      };
    }
  );
}

export const TEST_ONLY = { KNOWN_MODELS, HIDDEN_MODELS };
