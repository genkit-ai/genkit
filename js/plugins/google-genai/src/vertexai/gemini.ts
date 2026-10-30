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
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
  GenerationCommonConfigDescriptions,
  GenerationCommonConfigSchema,
  MessageData,
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
  selectInteractionInput,
  toInteractionConfigTool,
  toInteractionGenerationConfig,
  toInteractionGoogleSearch,
  toInteractionResponseFormat,
  toInteractionSafetySettings,
  toInteractionSteps,
  toInteractionTool,
  toInteractionToolChoice,
} from '../common/interaction-converters.js';
import {
  CreateInteractionRequest,
  InteractionTool,
  Step,
} from '../common/interaction-types.js';
import { isObject } from '../common/types.js';
import {
  isKnownKey,
  isStringArray,
  warnDroppedConfigField,
} from '../common/utils.js';
import {
  createInteraction,
  createInteractionStream,
  generateContent,
  generateContentStream,
} from './client.js';
import { toGeminiLabels, toGeminiSafetySettings } from './converters.js';
import {
  ClientOptions,
  Content,
  GenerateContentRequest,
  GenerateContentResponse,
  GoogleSearchRetrieval,
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
  toInteractionsClientOptions,
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
  apiVersion: z
    .enum(['v1', 'v1beta1'])
    .describe(
      'Overrides the plugin-configured API version to use, if specified.'
    )
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
  store: z
    .boolean()
    .describe(
      'Whether to store the interaction for later retrieval. Defaults to ' +
        'false. Only used by models on the Interactions API (on ' +
        'generateContent models, store: true is ignored with a warning).'
    )
    .optional(),
  previousInteractionId: z
    .string()
    .describe(
      'The ID of the previous interaction to continue. Requires store. ' +
        'Only used by models on the Interactions API.'
    )
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
  payGo: z
    .enum(['priority', 'priority-only', 'flex', 'flex-only'])
    .describe(
      'PayGo consumption options. Priority provides more consistent performance than Standard PayGo. ' +
        'Flex is a cost-effective option for non-critical workloads. ' +
        'The "-only" options use only PayGo and no Provisioned Throughput.'
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
      imageSize: z.enum(['512', '1K', '2K', '4K']).optional(),
    })
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiImageConfigSchemaType = typeof GeminiImageConfigSchema;
export type GeminiImageConfig = z.infer<GeminiImageConfigSchemaType>;

const VoiceConfigSchema = z
  .object({
    voice: z
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
        // Voice Library and custom voice IDs (e.g. `voice_...`). `string & {}`
        // keeps the prebuilt names in TypeScript autocomplete.
        z.custom<string & {}>((val) => typeof val === 'string'),
      ])
      .describe(
        'A prebuilt voice name, an Extended Voice Library voice ID, or a ' +
          'custom voice ID.'
      )
      .optional(),
    // The googleAI shape, also accepted by the platform, so TTS code can move
    // between the two plugins.
    prebuiltVoiceConfig: z
      .object({
        voiceName: z.string().describe('Name of the preset voice to use'),
      })
      .passthrough()
      .optional(),
  })
  .describe(
    'Configuration for the voice to use: `voice`, or `prebuiltVoiceConfig` ' +
      '(the googleAI shape)'
  )
  .passthrough();

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
            .describe(
              'Configuration for the two speaker voices. Each text part must ' +
                'name its speaker in `metadata.speechMetadata.speaker`.'
            ),
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
          z.custom<string & {}>((val) => typeof val === 'string'),
        ])
        .describe(
          'Audio encoding. Defaults to audio/wav (with a header) for ' +
            'non-streamed requests and audio/l16 (raw PCM, no header) for ' +
            'streamed requests.'
        )
        .optional(),
    })
    .describe('Format of the generated audio.')
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiTtsConfigSchemaType = typeof GeminiTtsConfigSchema;
export type GeminiTtsConfig = z.infer<GeminiTtsConfigSchemaType>;

// Gemini Omni (video generation). The fields follow the platform's
// Interactions `VideoResponseFormat` and the Omni guide's `video_config`.
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
        .describe('Resolution of the generated video. Defaults to 720p.')
        .optional(),
      delivery: z
        .union([
          z.enum(['inline', 'uri']),
          z.custom<string & {}>((val) => typeof val === 'string'),
        ])
        .describe(
          'How the video is returned. By default it is returned inline. ' +
            '"uri" stores it in Cloud Storage instead (set gcsUri).'
        )
        .optional(),
      gcsUri: z
        .string()
        .describe(
          'The Cloud Storage URI to store the video in. Required when ' +
            'delivery is "uri".'
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
            'it from the prompt and inputs.'
        )
        .optional(),
    })
    .describe('Video generation settings.')
    .passthrough()
    .optional(),
}).passthrough();
export type GeminiVideoConfigSchemaType = typeof GeminiVideoConfigSchema;
export type GeminiVideoConfig = z.infer<GeminiVideoConfigSchemaType>;

// This contains all the Gemini config schema types
type ConfigSchemaType =
  | GeminiConfigSchemaType
  | GeminiImageConfigSchemaType
  | GeminiTtsConfigSchemaType
  | GeminiVideoConfigSchemaType;
type ConfigSchema = z.infer<ConfigSchemaType>;

/** Whether each known model uses the Interactions API (see `usesInteractions`). */
const modelInteractionsMap = new Map<string, boolean>();

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = GeminiConfigSchema,
  useInteractions: boolean = true
): ModelReference<ConfigSchemaType> {
  modelInteractionsMap.set(name, useInteractions);

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
        constrained: 'all',
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
  GeminiTtsConfigSchema,
  false
);
// Per the Omni guide: text, image and video in, video (with audio) out; edits
// continue a stored interaction. Tools and structured output aren't
// documented for video generation.
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

// Models with `false` stay on generateContent: older models (like in the
// googleAI plugin) and models whose docs page says the Interactions API isn't
// supported.
export const KNOWN_GEMINI_MODELS = {
  'gemini-4-argon': commonRef('gemini-4-argon'),
  // Allowlist only.
  'gemini-4-argon-cyber-permissive': commonRef(
    'gemini-4-argon-cyber-permissive'
  ),
  'gemini-3.8-flash': commonRef('gemini-3.8-flash'),
  // Allowlist only.
  'gemini-3.8-flash-cyber': commonRef(
    'gemini-3.8-flash-cyber',
    undefined,
    GeminiConfigSchema,
    false
  ),
  'gemini-3.7-flash': commonRef('gemini-3.7-flash'),
  'gemini-3.5-flash-lite': commonRef('gemini-3.5-flash-lite'),
  'gemini-3.5-flash': commonRef(
    'gemini-3.5-flash',
    undefined,
    GeminiConfigSchema,
    false
  ),
  'gemini-3.1-flash-lite': commonRef(
    'gemini-3.1-flash-lite',
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
  // Optimized for custom tools (an extra endpoint of 3.1 Pro).
  'gemini-3.1-pro-preview-customtools': commonRef(
    'gemini-3.1-pro-preview-customtools',
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
} as const;
export type KnownGeminiModels = keyof typeof KNOWN_GEMINI_MODELS;
export type GeminiModelName = `gemini-${string}`;
export function isGeminiModelName(value?: string): value is GeminiModelName {
  return !!(
    value?.startsWith('gemini-') &&
    !value.includes('embedding') &&
    !isImageModelName(value) &&
    !isTTSModelName(value) &&
    !isVideoModelName(value)
  );
}

export const KNOWN_IMAGE_MODELS = {
  // seed, topK, logprobs, temperature and topP aren't supported (the server
  // returns an error).
  'gemini-nano-banana-2.1': commonRef(
    'gemini-nano-banana-2.1',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema,
    false
  ),
  'gemini-3.1-flash-lite-image': commonRef(
    'gemini-3.1-flash-lite-image',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema,
    false
  ),
  'gemini-3.1-flash-image': commonRef(
    'gemini-3.1-flash-image',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema,
    false
  ),
  'gemini-3-pro-image': commonRef(
    'gemini-3-pro-image',
    { ...GENERIC_IMAGE_MODEL.info },
    GeminiImageConfigSchema,
    false
  ),
  'gemini-2.5-flash-image': commonRef(
    'gemini-2.5-flash-image',
    undefined,
    GeminiImageConfigSchema,
    false
  ),
} as const;
export type KnownImageModels = keyof typeof KNOWN_IMAGE_MODELS;
export type ImageModelName =
  | `gemini-${string}-image${string}`
  | `gemini-nano-banana-${string}`;
export function isImageModelName(value?: string): value is ImageModelName {
  return !!(
    value?.startsWith('gemini-') &&
    (value.includes('-image') || value.startsWith('gemini-nano-banana-'))
  );
}

// TTS models only work on generateContent (the Interactions API rejects them,
// checked live Oct 2026).
export const KNOWN_TTS_MODELS = {
  'gemini-3.8-flash-tts': commonRef(
    'gemini-3.8-flash-tts',
    { ...GENERIC_TTS_MODEL.info },
    GeminiTtsConfigSchema,
    false
  ),
  'gemini-3.8-flash-lite-tts': commonRef(
    'gemini-3.8-flash-lite-tts',
    { ...GENERIC_TTS_MODEL.info },
    GeminiTtsConfigSchema,
    false
  ),
} as const;
export type KnownTtsModels = keyof typeof KNOWN_TTS_MODELS;
export type TTSModelName = `gemini-${string}-tts${string}`;
export function isTTSModelName(value?: string): value is TTSModelName {
  return !!(value?.startsWith('gemini-') && value.includes('-tts'));
}

// Omni video models use the Interactions API.
export const KNOWN_VIDEO_MODELS = {
  'gemini-omni-1.1-flash-preview': commonRef(
    'gemini-omni-1.1-flash-preview',
    { ...GENERIC_VIDEO_MODEL.info },
    GeminiVideoConfigSchema
  ),
  'gemini-omni-flash-preview': commonRef(
    'gemini-omni-flash-preview',
    { ...GENERIC_VIDEO_MODEL.info },
    GeminiVideoConfigSchema
  ),
} as const;
export type KnownVideoModels = keyof typeof KNOWN_VIDEO_MODELS;
export type VideoModelName = `gemini-omni-${string}`;
export function isVideoModelName(value?: string): value is VideoModelName {
  return !!value?.startsWith('gemini-omni-');
}

const KNOWN_MODELS = {
  ...KNOWN_GEMINI_MODELS,
  ...KNOWN_IMAGE_MODELS,
  ...KNOWN_TTS_MODELS,
  ...KNOWN_VIDEO_MODELS,
};
export type KnownModels = keyof typeof KNOWN_MODELS;

// Models we no longer suggest, but that still work: they still resolve and
// are still registered as actions, but aren't in the Known*Models types, so
// they don't show up in autocomplete. Set `stage` on each one (keeping the
// default `supports`):
// - 'deprecated': deprecated by Google, or < 1 month from shutdown.
// - 'legacy': not deprecated, but no longer offered to new projects.
// Remove a model once the server returns 404 for it.
const HIDDEN_MODELS = {
  // Retirement date Oct 20 2026; kept until the server stops serving them.
  // They aren't on the Interactions API, so they must stay listed (`false`):
  // unlisted names would be sent to the Interactions API.
  'gemini-2.5-pro': commonRef(
    'gemini-2.5-pro',
    { ...GENERIC_MODEL.info, stage: 'deprecated' },
    GeminiConfigSchema,
    false
  ),
  'gemini-2.5-flash': commonRef(
    'gemini-2.5-flash',
    { ...GENERIC_MODEL.info, stage: 'deprecated' },
    GeminiConfigSchema,
    false
  ),
  'gemini-2.5-flash-lite': commonRef(
    'gemini-2.5-flash-lite',
    { ...GENERIC_MODEL.info, stage: 'deprecated' },
    GeminiConfigSchema,
    false
  ),
} as const;

const ALL_MODELS = {
  ...HIDDEN_MODELS,
  ...KNOWN_MODELS,
};

/**
 * Whether a model is called through the Interactions API. Known older models
 * stay on generateContent (`false` in their `commonRef`); newer and unlisted
 * models use the Interactions API. Tuned model endpoints stay on
 * generateContent.
 */
function usesInteractions(modelVersion: string): boolean {
  if (modelVersion.includes('endpoints/')) {
    return false;
  }
  // The platform's Interactions API doesn't support image or TTS models (it
  // returns 400 "Model 'gemini-nano-banana-2.1' is not supported by the
  // Interactions API"; the 3.8 TTS models fail too; Oct 2026), so they stay
  // on generateContent.
  return (
    modelInteractionsMap.get(modelVersion) ??
    !(isImageModelName(modelVersion) || isTTSModelName(modelVersion))
  );
}

export function model(
  version: string,
  config: ConfigSchema = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, ALL_MODELS)) {
    return ALL_MODELS[name].withConfig(config);
  }

  if (isImageModelName(name)) {
    return modelRef({
      name: `vertexai/${name}`,
      config,
      configSchema: GeminiImageConfigSchema,
      info: { ...GENERIC_IMAGE_MODEL.info },
    });
  }

  if (isVideoModelName(name)) {
    return modelRef({
      name: `vertexai/${name}`,
      config,
      configSchema: GeminiVideoConfigSchema,
      info: { ...GENERIC_VIDEO_MODEL.info },
    });
  }

  if (isTTSModelName(name)) {
    return modelRef({
      name: `vertexai/${name}`,
      config,
      configSchema: GeminiTtsConfigSchema,
      info: { ...GENERIC_TTS_MODEL.info },
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
          isImageModelName(modelName(m.name)) ||
          isTTSModelName(modelName(m.name)) ||
          isVideoModelName(modelName(m.name))) &&
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
  return Object.keys(ALL_MODELS).map((name) =>
    defineModel(name, clientOptions, pluginOptions)
  );
}

/**
 * Converts the TTS `audioFormat` config (`{ mimeType: 'audio/mulaw' }`) to
 * the generateContent `responseFormat` (`[{ audio: { mimeType:
 * 'AUDIO_MULAW' } }]`). Returns `undefined` if it isn't set, so the server
 * defaults apply (audio/wav, or audio/l16 when streaming).
 */
function toGeminiAudioResponseFormat(
  audioFormat: unknown
): Record<string, unknown>[] | undefined {
  if (!isObject(audioFormat) || typeof audioFormat.mimeType !== 'string') {
    return undefined;
  }
  const { mimeType, ...rest } = audioFormat;
  return [
    {
      audio: {
        ...rest,
        // 'audio/mulaw' -> 'AUDIO_MULAW'; enum values pass through.
        mimeType: mimeType.toUpperCase().replace('/', '_'),
      },
    },
  ];
}

/**
 * Builds the Agent Search data store path for `vertexRetrieval`, filling in
 * the project and location from the plugin options when not set.
 */
function toVertexAISearchDatastore(
  vertexRetrieval: NonNullable<ConfigSchema['vertexRetrieval']>,
  clientOptions: ClientOptions
): string {
  const projectId =
    vertexRetrieval.datastore.projectId ||
    (clientOptions.kind != 'express' ? clientOptions.projectId : undefined);
  // Data stores are in `global`, `us` or `eu`, so global and multi-regional
  // client locations are valid fallbacks.
  // TODO: Regional locations (e.g. us-central1) aren't data store locations;
  // falling back to one probably fails on the server. Kept as before until
  // checked live.
  const location =
    vertexRetrieval.datastore.location ||
    (clientOptions.kind != 'express' ? clientOptions.location : undefined);
  const dataStoreId = vertexRetrieval.datastore.dataStoreId;
  if (!projectId || !location || !dataStoreId) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'projectId, location and datastoreId are required for vertexRetrieval and could not be determined from configuration',
    });
  }
  return `projects/${projectId}/locations/${location}/collections/default_collection/dataStores/${dataStoreId}`;
}

/**
 * Server-side tool step types that are intentionally NOT sent back in a
 * stateless history (`input`) on this platform.
 *
 * Resending them makes the platform return 501 "Operation is not
 * implemented, or supported, or enabled" (seen live, Oct 2026, whatever the
 * step order, arguments or signature), while the Gemini API (googleAI)
 * accepts the same history. `store: true` conversations work, because the
 * steps stay on the server. Without them the model loses the earlier tool
 * results (it may call the tool again). Only step types seen working are
 * resent; code execution steps are accepted and are not listed here.
 * Remove an entry once the platform accepts it.
 */
const UNSUPPORTED_RESENT_STEP_TYPES: ReadonlySet<string> = new Set([
  // Rejected with 501 (seen live).
  'google_search_call',
  'google_search_result',
  'url_context_call',
  'url_context_result',
  'google_maps_call',
  'google_maps_result',
  // Not seen working here (untestable or not available for models).
  'file_search_call',
  'file_search_result',
  'mcp_server_tool_call',
  'mcp_server_tool_result',
]);

/** Whether the dropped-steps warning has been logged (once per process). */
let warnedDroppedResentSteps = false;

/**
 * Removes the steps listed in `UNSUPPORTED_RESENT_STEP_TYPES`, warning once
 * when any were removed.
 */
function withoutUnsupportedResentSteps(steps: Step[]): Step[] {
  const kept = steps.filter(
    (step) => !UNSUPPORTED_RESENT_STEP_TYPES.has(step.type)
  );
  if (kept.length < steps.length && !warnedDroppedResentSteps) {
    warnedDroppedResentSteps = true;
    logger.warn(
      'Earlier server-side tool results (e.g. Google Search, URL context, ' +
        'Google Maps) are not sent back in this conversation history, ' +
        'because the server rejects them. The model may call those tools ' +
        'again. Set store: true to keep the full tool history on the server.'
    );
  }
  return kept;
}

/**
 * Converts the request's tools and tool config fields to Interactions tools.
 */
function toVertexInteractionTools(
  request: GenerateRequest,
  config: ConfigSchema,
  clientOptions: ClientOptions,
  warnDropped: (field: string) => void
): InteractionTool[] {
  const tools: InteractionTool[] = [];
  if (request.tools?.length) {
    tools.push(...request.tools.map(toInteractionTool));
  }
  if (Array.isArray(config.tools)) {
    tools.push(...config.tools.map(toInteractionConfigTool));
  }
  if (config.googleSearchRetrieval) {
    const { disableAttribution, ...searchConfig } =
      config.googleSearchRetrieval;
    if (disableAttribution !== undefined) {
      warnDropped('googleSearchRetrieval.disableAttribution');
    }
    tools.push(toInteractionGoogleSearch(searchConfig));
  }
  const vertexRetrieval = config.vertexRetrieval;
  if (vertexRetrieval) {
    if (vertexRetrieval.disableAttribution !== undefined) {
      warnDropped('vertexRetrieval.disableAttribution');
    }
    tools.push({
      type: 'retrieval',
      retrieval_types: ['vertex_ai_search'],
      vertex_ai_search_config: {
        datastores: [toVertexAISearchDatastore(vertexRetrieval, clientOptions)],
      },
    });
  }
  if (config.retrievalConfig) {
    // There is no retrieval config on the Interactions path; the user
    // location is set on the google_maps tool instead.
    const { latLng, ...unsupportedRetrievalConfig } = config.retrievalConfig;
    if (latLng && !applyLatLngToGoogleMapsTools(tools, latLng)) {
      logger.warn(
        'retrievalConfig.latLng is only used with the googleMaps tool for this model and will be ignored.'
      );
    }
    for (const key of Object.keys(unsupportedRetrievalConfig)) {
      warnDropped(`retrievalConfig.${key}`);
    }
  }
  return tools;
}

/**
 * Generates a response with the Interactions API (see `usesInteractions`).
 */
async function generateWithInteractions(params: {
  request: GenerateRequest;
  /** The request messages, without the system message. */
  messages: MessageData[];
  systemMessage?: MessageData;
  modelVersion: string;
  config: ConfigSchema;
  /** The plugin's client options (used for the vertexRetrieval fallback). */
  pluginClientOptions: ClientOptions;
  /** The client options for this request (location, API key, headers). */
  requestClientOptions: ClientOptions;
  pluginOptions?: VertexPluginOptions;
  streamingRequested: boolean;
  sendChunk: (chunk: GenerateResponseChunkData) => void;
}): Promise<GenerateResponseData> {
  const { request, modelVersion, pluginOptions, config } = params;
  const {
    // Applied to the client options and headers by the caller.
    apiKey: _apiKey,
    apiVersion: _apiVersion,
    location: _location,
    payGo: _payGo,
    version: _version,
    // Converted to tools (toVertexInteractionTools).
    tools: _tools,
    googleSearchRetrieval: _googleSearchRetrieval,
    vertexRetrieval: _vertexRetrieval,
    retrievalConfig: _retrievalConfig,
    functionCallingConfig,
    safetySettings,
    labels,
    store: storeFromConfig,
    previousInteractionId: previousInteractionIdFromConfig,
    responseModalities,
    // Sent as `response_format` entries below.
    videoFormat,
    imageConfig,
    // TTS models don't use the Interactions API.
    audioFormat,
    ...generationFields
  } = config;
  const warnDropped = (field: string) =>
    warnDroppedConfigField(modelVersion, field);
  if (audioFormat !== undefined) {
    warnDropped('audioFormat');
  }

  const systemMessage = params.systemMessage;
  if (systemMessage?.content.some((c) => c.text === undefined)) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'System message contains non-text content which is not supported for this model.',
    });
  }

  if (request.candidates && request.candidates > 1) {
    logger.warn(
      'Multiple candidates are not supported for this model with the Interactions API; only one candidate will be returned.'
    );
  }

  // The vertexRetrieval data store fallback uses the plugin's project and
  // location, not `config.location` (same as on generateContent).
  const tools = toVertexInteractionTools(
    request,
    config,
    params.pluginClientOptions,
    warnDropped
  );

  if (functionCallingConfig?.streamFunctionCallArguments !== undefined) {
    warnDropped('functionCallingConfig.streamFunctionCallArguments');
  }
  const generationConfig = toInteractionGenerationConfig(
    generationFields,
    warnDropped
  );
  const toolChoice = toInteractionToolChoice(
    functionCallingConfig,
    request.toolChoice
  );
  if (toolChoice) {
    generationConfig.tool_choice = toolChoice;
  }

  const store = storeFromConfig === true || pluginOptions?.store === true;
  const { messages, previousInteractionId } = selectInteractionInput(
    params.messages,
    store,
    previousInteractionIdFromConfig
  );

  const req: CreateInteractionRequest = {
    system_instruction: systemMessage?.content.map((c) => c.text).join('\n'),
    model: modelVersion,
    tools: tools.length ? tools : undefined,
    generation_config: generationConfig,
    stream: params.streamingRequested,
    input: withoutUnsupportedResentSteps(toInteractionSteps(messages)),
    store,
    safety_settings: toInteractionSafetySettings(safetySettings),
    labels: toGeminiLabels(labels),
  };
  if (previousInteractionId) {
    req.previous_interaction_id = previousInteractionId;
  }

  // One `response_format` with an entry per output type (the deprecated
  // `response_modalities` and `generation_config.image_config` aren't sent).
  const jsonMode =
    request.output?.format === 'json' || !!request.output?.schema;
  const responseFormat = toInteractionResponseFormat({
    modalities:
      Array.isArray(responseModalities) && isStringArray(responseModalities)
        ? responseModalities
        : isImageModelName(modelVersion)
          ? ['TEXT', 'IMAGE']
          : undefined,
    imageConfig,
    videoFormat,
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

  if (!params.streamingRequested) {
    return fromInteractionSync(
      await createInteraction(req, params.requestClientOptions)
    );
  }
  const result = await createInteractionStream(
    req,
    params.requestClientOptions
  );
  const toChunkParts = createInteractionChunkConverter();
  for await (const event of result.stream) {
    const content = toChunkParts(event);
    if (content.length > 0) {
      params.sendChunk({ index: 0, content });
    }
  }
  return fromInteractionSync(await result.response);
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
        apiVersion: apiVersionFromConfig,
        functionCallingConfig,
        retrievalConfig,
        version: versionFromConfig,
        googleSearchRetrieval,
        tools: toolsFromConfig,
        vertexRetrieval,
        location,
        safetySettings,
        labels: labelsFromConfig,
        payGo,
        store: storeFromConfig,
        previousInteractionId: previousInteractionIdFromConfig,
        // TTS models: sent as `responseFormat` below.
        audioFormat: audioFormatFromConfig,
        // Video (Omni) models use the Interactions API.
        videoFormat: videoFormatFromConfig,
        ...restOfConfig
      } = requestConfig;

      clientOpt = calculateRequestOptions(clientOpt, {
        location,
        apiKey: apiKeyFromConfig,
        apiVersion: apiVersionFromConfig,
      });

      if (payGo) {
        const payGoHeaders: Record<string, string> = {};
        if (payGo === 'priority') {
          payGoHeaders['X-Vertex-AI-LLM-Shared-Request-Type'] = 'priority';
        } else if (payGo === 'priority-only') {
          payGoHeaders['X-Vertex-AI-LLM-Request-Type'] = 'shared';
          payGoHeaders['X-Vertex-AI-LLM-Shared-Request-Type'] = 'priority';
        } else if (payGo === 'flex') {
          payGoHeaders['X-Vertex-AI-LLM-Shared-Request-Type'] = 'flex';
        } else if (payGo === 'flex-only') {
          payGoHeaders['X-Vertex-AI-LLM-Request-Type'] = 'shared';
          payGoHeaders['X-Vertex-AI-LLM-Shared-Request-Type'] = 'flex';
        }
        clientOpt.customHeaders = {
          ...clientOpt.customHeaders,
          ...payGoHeaders,
        };
      }

      const modelVersion = versionFromConfig || extractVersion(ref);
      if (usesInteractions(modelVersion)) {
        return generateWithInteractions({
          request,
          messages,
          systemMessage,
          modelVersion,
          config: requestConfig,
          pluginClientOptions: clientOptions,
          // A defaulted location (us-central1) becomes global here.
          requestClientOptions: toInteractionsClientOptions(clientOpt),
          pluginOptions,
          streamingRequested,
          sendChunk,
        });
      }

      // The platform's generateContent has no `store` field and never stores
      // anything for later retrieval, so only `store: true` (request config
      // or plugin options) asks for something that can't happen.
      if ((storeFromConfig ?? pluginOptions?.store) === true) {
        warnDroppedConfigField(modelVersion, 'store');
      }
      // There's no stored conversation to continue on generateContent.
      if (previousInteractionIdFromConfig) {
        logger.warn(
          'previousInteractionId is not supported for this model and will be ignored.'
        );
      }
      if (videoFormatFromConfig !== undefined) {
        warnDroppedConfigField(modelVersion, 'videoFormat');
      }

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

      const jsonMode =
        request.output?.format === 'json' || !!request.output?.schema;

      if (toolsFromConfig) {
        tools.push(...(toolsFromConfig as any[]));
      }

      if (googleSearchRetrieval) {
        tools.push({
          googleSearch: googleSearchRetrieval as GoogleSearchRetrieval,
        });
      }

      if (vertexRetrieval) {
        tools.push({
          retrieval: {
            vertexAiSearch: {
              datastore: toVertexAISearchDatastore(
                vertexRetrieval,
                clientOptions
              ),
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

      if (isImageModelName(modelVersion)) {
        if (!generateContentRequest.generationConfig!.responseModalities) {
          generateContentRequest.generationConfig!.responseModalities = [
            'TEXT',
            'IMAGE',
          ];
        }
      } else if (isTTSModelName(modelVersion)) {
        if (!generateContentRequest.generationConfig!.responseModalities) {
          generateContentRequest.generationConfig!.responseModalities = [
            'AUDIO',
          ];
        }
      }

      const responseFormat = toGeminiAudioResponseFormat(audioFormatFromConfig);
      if (responseFormat) {
        generateContentRequest.generationConfig!.responseFormat =
          responseFormat;
      }

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
          (item as GenerateContentResponse).candidates?.forEach((candidate) => {
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
      if (streamingRequested && isTTSModelName(modelVersion)) {
        // Streamed audio arrives as many headerless chunks (audio/l16 by
        // default); join them into one part, like a non-streamed response.
        for (const candidate of candidateData) {
          candidate.message.content = mergeStreamedAudio(
            candidate.message.content
          );
        }
      }

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

export const TEST_ONLY = {
  KNOWN_GEMINI_MODELS,
  KNOWN_IMAGE_MODELS,
  KNOWN_TTS_MODELS,
  KNOWN_VIDEO_MODELS,
  HIDDEN_MODELS,
  ALL_MODELS,
  KNOWN_MODELS,
  resetDroppedResentStepsWarning: () => {
    warnedDroppedResentSteps = false;
  },
};
