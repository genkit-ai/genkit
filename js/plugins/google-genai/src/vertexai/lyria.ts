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
  GenerateRequest,
  modelActionMetadata,
  modelRef,
  ModelReference,
  z,
} from 'genkit';
import { ModelAction, ModelInfo } from 'genkit/model';
import { model as pluginModel } from 'genkit/plugin';
import {
  ensureToolIds,
  fromInteractionSync,
  toInteractionTurn,
} from '../common/interaction-converters.js';
import {
  CreateInteractionRequest,
  ResponseModality,
} from '../common/interaction-types.js';
import { isKnownKey } from '../common/utils.js';
import { createInteraction, lyriaPredict } from './client.js';
import { fromLyriaResponse, toLyriaPredictRequest } from './converters.js';
import { ClientOptions, Model, VertexPluginOptions } from './types.js';
import {
  calculateRequestOptions,
  checkModelName,
  extractVersion,
  shouldUseLegacyEndpoint,
} from './utils.js';

export const LyriaConfigSchema = z
  .object({
    negativePrompt: z
      .string()
      .optional()
      .describe(
        'Optional. A description of what to exclude from the generated audio.'
      ),
    seed: z
      .number()
      .optional()
      .describe(
        'Optional. A seed for deterministic generation. If provided, the model will attempt to produce the same audio given the same prompt and other parameters. Cannot be used with sample_count in the same request.'
      ),
    sampleCount: z
      .number()
      .optional()
      .describe(
        'Optional. The number of audio samples to generate. Default is 1 if not specified and seed is not used. Cannot be used with seed in the same request.'
      ),
    location: z
      .string()
      .describe(
        'Lyria is only available in global. If you initialize your plugin with a different region, you must set this to global.'
      )
      .optional(),
  })
  .passthrough();
export type LyriaConfigSchemaType = typeof LyriaConfigSchema;
export type LyriaConfig = z.infer<LyriaConfigSchemaType>;

export const Lyria3ConfigSchema = z
  .object({
    location: z
      .string()
      .describe(
        'Lyria is only available in global. If you initialize your plugin with a different region, you must set this to global.'
      )
      .optional(),
    responseModalities: z
      .array(z.enum(['TEXT', 'IMAGE', 'AUDIO']))
      .describe(
        'The modalities to be used in response. Defaults to AUDIO and TEXT for Lyria.'
      )
      .optional(),
  })
  .passthrough();
export type Lyria3ConfigSchemaType = typeof Lyria3ConfigSchema;
export type Lyria3Config = z.infer<Lyria3ConfigSchemaType>;

type ConfigSchemaType = LyriaConfigSchemaType | Lyria3ConfigSchemaType;
type ConfigSchema = z.infer<ConfigSchemaType>;

function isLegacyLyriaRequest(
  request: GenerateRequest<ConfigSchemaType>,
  name: string
): request is GenerateRequest<LyriaConfigSchemaType> {
  return shouldUseLegacyEndpoint(name);
}

function isLyria3Request(
  request: GenerateRequest<ConfigSchemaType>,
  name: string
): request is GenerateRequest<Lyria3ConfigSchemaType> {
  return !shouldUseLegacyEndpoint(name);
}

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = Lyria3ConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertexai/${name}`,
    configSchema,
    info: info ?? {
      supports: {
        media: true,
        multiturn: false,
        tools: false,
        systemRole: false,
        output: ['text', 'media'],
      },
    },
  });
}

const GENERIC_MODEL = commonRef('lyria');
const GENERIC_LEGACY_MODEL = commonRef(
  'lyria-002',
  {
    supports: {
      media: true,
      multiturn: false,
      tools: false,
      systemRole: false,
      output: ['media'],
    },
  },
  LyriaConfigSchema
);

const KNOWN_LYRIA_LEGACY_MODELS = {
  'lyria-002': commonRef(
    'lyria-002',
    { ...GENERIC_LEGACY_MODEL.info },
    LyriaConfigSchema
  ),
} as const;
export type KnownLyriaLegacyModels = keyof typeof KNOWN_LYRIA_LEGACY_MODELS; // For autocorrect
export type LyriaLegacyModelName = `lyria-002`;

const KNOWN_LYRIA_INTERACTIONS_MODELS = {
  'lyria-3-clip-preview': commonRef('lyria-3-clip-preview'),
  'lyria-3-pro-preview': commonRef('lyria-3-pro-preview'),
} as const;
export type KnownLyriaInteractionsModels =
  keyof typeof KNOWN_LYRIA_INTERACTIONS_MODELS;

const KNOWN_MODELS = {
  ...KNOWN_LYRIA_LEGACY_MODELS,
  ...KNOWN_LYRIA_INTERACTIONS_MODELS,
};
export type KnownModels = keyof typeof KNOWN_MODELS;

export type LyriaModelName = `lyria-${string}`;
export function isLyriaModelName(value?: string): value is LyriaModelName {
  return !!value?.startsWith('lyria-');
}

export function model(
  version: string,
  config: ConfigSchema = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);

  if (isKnownKey(name, KNOWN_MODELS)) {
    return KNOWN_MODELS[name].withConfig(config);
  }

  return modelRef({
    name: `vertexai/${name}`,
    config,
    configSchema: Lyria3ConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  return models
    .filter((m: Model) => isLyriaModelName(m.name))
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
  return Object.keys(KNOWN_MODELS).map((name: string) =>
    defineModel(name, clientOptions, pluginOptions)
  );
}

export function defineModel(
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
): ModelAction<ConfigSchemaType> {
  const ref = model(name);

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
      if (isLegacyLyriaRequest(request, name)) {
        const lyriaPredictRequest = toLyriaPredictRequest(request);

        const response = await lyriaPredict(
          extractVersion(ref),
          lyriaPredictRequest,
          clientOpt
        );

        if (!response.predictions || response.predictions.length == 0) {
          throw new Error(
            'Model returned no predictions. Possibly due to content filters.'
          );
        }

        return fromLyriaResponse(response, request);
      } else if (isLyria3Request(request, name)) {
        const messages = [...request.messages];
        if (messages.length === 0) throw new Error('No messages provided');

        let responseModalitiesConverted: ResponseModality[] = ['audio', 'text'];
        if (request.config?.responseModalities) {
          responseModalitiesConverted = request.config.responseModalities.map(
            (m) => m.toLowerCase() as ResponseModality
          );
        }

        const req: CreateInteractionRequest = {
          model: extractVersion(ref),
          input: ensureToolIds(messages).map(toInteractionTurn),
          response_modalities: responseModalitiesConverted,
        };

        const response = await createInteraction(req, clientOpt);

        return fromInteractionSync(response);
      } else {
        throw new Error(`Unsupported model config schema for ${name}`);
      }
    }
  );
}

export const TEST_ONLY = {
  GENERIC_MODEL,
  KNOWN_MODELS,
  KNOWN_LYRIA_LEGACY_MODELS,
  KNOWN_LYRIA_INTERACTIONS_MODELS,
};
