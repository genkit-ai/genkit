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

import { modelActionMetadata, z, type ActionMetadata } from 'genkit';
import {
  modelRef,
  type ModelAction,
  type ModelInfo,
  type ModelReference,
} from 'genkit/model';
import { model as pluginModel } from 'genkit/plugin';
import { generateMusic } from './client.js';
import {
  fromMusicGenerationResponse,
  toMusicGenerationRequest,
} from './converters.js';
import {
  AUDIO_URL_TTL_HOURS,
  DEFAULT_OUTPUT_FORMAT,
  DEFAULT_REGION,
  MINIMAX_REGIONS,
  MUSIC_AUDIO_FORMATS,
  MUSIC_BITRATES,
  MUSIC_OUTPUT_FORMATS,
  MUSIC_SAMPLE_RATES,
  type ClientOptions,
  type MiniMaxPluginOptions,
} from './types.js';
import {
  PLUGIN_NAME,
  calculateApiKey,
  checkApiKey,
  checkModelName,
} from './utils.js';

/**
 * Builds a zod schema accepting only the given numbers.
 */
function numberEnum(
  values: readonly [number, number, ...number[]]
): z.ZodUnion<[z.ZodLiteral<number>, z.ZodLiteral<number>]> {
  return z.union(
    values.map((value) => z.literal(value)) as unknown as [
      z.ZodLiteral<number>,
      z.ZodLiteral<number>,
    ]
  );
}

/**
 * Config accepted by the MiniMax music generation models.
 *
 * The generation fields mirror the `POST /v1/music_generation` request body.
 */
export const MiniMaxMusicConfigSchema = z
  .object({
    apiKey: z
      .string()
      .describe('Override the API key provided at plugin initialization.')
      .optional(),
    region: z
      .enum(MINIMAX_REGIONS)
      .describe(
        'Overrides the plugin-configured or default region, which selects the API endpoint.'
      )
      .optional(),
    baseUrl: z
      .string()
      .describe('Overrides the base URL derived from the region.')
      .optional(),
    prompt: z
      .string()
      .max(2000)
      .describe(
        'Style, mood or scenario description. Only used when the request messages carry no text.'
      )
      .optional(),
    lyrics: z
      .string()
      .min(1)
      .max(3500)
      .describe(
        'Lyrics to sing, with lines separated by newlines and optional section tags such as [Verse] or [Chorus].'
      )
      .optional(),
    stream: z
      .boolean()
      .describe(
        'Not supported by this model, which always returns a single response.'
      )
      .optional(),
    output_format: z
      .enum(MUSIC_OUTPUT_FORMATS)
      .describe(
        `Encoding of the generated audio. \`url\` returns a URL that expires after ${AUDIO_URL_TTL_HOURS} hours, \`hex\` inlines the audio bytes. Defaults to \`${DEFAULT_OUTPUT_FORMAT}\`.`
      )
      .optional(),
    audio_setting: z
      .object({
        sample_rate: numberEnum(MUSIC_SAMPLE_RATES)
          .describe('Sample rate of the generated audio.')
          .optional(),
        bitrate: numberEnum(MUSIC_BITRATES)
          .describe('Bitrate of the generated audio.')
          .optional(),
        format: z
          .enum(MUSIC_AUDIO_FORMATS)
          .describe('Container format of the generated audio.')
          .optional(),
      })
      .describe('Audio output configuration.')
      .optional(),
    lyrics_optimizer: z
      .boolean()
      .describe('Let the service derive the lyrics from the prompt.')
      .optional(),
    is_instrumental: z
      .boolean()
      .describe('Generate vocal-free audio, which makes lyrics unnecessary.')
      .optional(),
    aigc_watermark: z
      .boolean()
      .describe(
        'Embed an AIGC watermark. Only accepted by the `cn` region endpoint.'
      )
      .optional(),
  })
  .passthrough();

/** Type of {@link MiniMaxMusicConfigSchema}. */
export type MiniMaxMusicConfigSchemaType = typeof MiniMaxMusicConfigSchema;

/** Config accepted by the MiniMax music generation models. */
export type MiniMaxMusicConfig = z.infer<MiniMaxMusicConfigSchemaType>;

type ConfigSchemaType = MiniMaxMusicConfigSchemaType;

/**
 * Model used when only the generic `music` name is requested.
 */
export const DEFAULT_MUSIC_MODEL = 'music-3.0';

/**
 * Prefix of the music cover models, which take reference audio instead of a
 * prompt and are therefore not served by this model.
 */
const COVER_MODEL_PREFIX = 'music-cover';

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = MiniMaxMusicConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `${PLUGIN_NAME}/${name}`,
    configSchema,
    info:
      info ??
      ({
        label: `MiniMax - ${name}`,
        supports: {
          multiturn: false,
          media: false,
          tools: false,
          toolChoice: false,
          systemRole: false,
          output: ['media'],
        },
      } as ModelInfo),
  });
}

const GENERIC_MODEL = commonRef(DEFAULT_MUSIC_MODEL);

const KNOWN_MODELS = {
  'music-3.0': commonRef('music-3.0'),
  'music-2.6': commonRef('music-2.6'),
  'music-3.0-free': commonRef('music-3.0-free'),
  'music-2.6-free': commonRef('music-2.6-free'),
} as const;

/** Music generation models known to this plugin. */
export type KnownMusicModels = keyof typeof KNOWN_MODELS;

/** Any music generation model name. */
export type MiniMaxMusicModelName = `music-${string}`;

/**
 * Reports whether a name refers to a music generation model.
 *
 * Cover models share the `music-` prefix but take reference audio rather than a
 * prompt, so they are not claimed here.
 */
export function isMiniMaxMusicModelName(
  value?: string
): value is MiniMaxMusicModelName {
  return !!value?.startsWith('music-') && !value.startsWith(COVER_MODEL_PREFIX);
}

/**
 * Returns a reference to a music generation model.
 */
export function model(
  version: string,
  config: MiniMaxMusicConfig = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (Object.prototype.hasOwnProperty.call(KNOWN_MODELS, name)) {
    return KNOWN_MODELS[name as KnownMusicModels].withConfig(config);
  }

  return modelRef({
    name: `${PLUGIN_NAME}/${name}`,
    config,
    configSchema: MiniMaxMusicConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

/**
 * Lists the music generation models known to this plugin.
 *
 * The API does not expose a model discovery endpoint, so the known models are
 * reported instead.
 */
export function listActions(): ActionMetadata[] {
  return Object.keys(KNOWN_MODELS).map((name) => {
    const ref = model(name);
    return modelActionMetadata({
      name: ref.name,
      info: ref.info,
      configSchema: ref.configSchema,
    });
  });
}

/**
 * Defines a model action for every known music generation model.
 */
export function listKnownModels(
  pluginOptions?: MiniMaxPluginOptions
): ModelAction<ConfigSchemaType>[] {
  return Object.keys(KNOWN_MODELS).map((name) =>
    defineModel(name, pluginOptions)
  );
}

/**
 * Defines the model action that calls the music generation endpoint.
 */
export function defineModel(
  name: string,
  pluginOptions?: MiniMaxPluginOptions
): ModelAction<ConfigSchemaType> {
  checkApiKey(pluginOptions?.apiKey);
  const ref = model(name);
  const version = checkModelName(ref.name);

  return pluginModel(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
    },
    async (request, { abortSignal }) => {
      const config = request.config ?? {};
      const region = config.region ?? pluginOptions?.region ?? DEFAULT_REGION;

      const clientOptions: ClientOptions = {
        apiKey: calculateApiKey(pluginOptions?.apiKey, config.apiKey),
        region,
        baseUrl: config.baseUrl ?? pluginOptions?.baseUrl,
        customHeaders: pluginOptions?.customHeaders,
        signal: abortSignal,
      };

      const musicRequest = toMusicGenerationRequest(version, request, region);
      const response = await generateMusic(musicRequest, clientOptions);

      return fromMusicGenerationResponse(
        request,
        response,
        musicRequest.output_format,
        musicRequest.audio_setting?.format
      );
    }
  );
}

export const TEST_ONLY = { GENERIC_MODEL, KNOWN_MODELS };
