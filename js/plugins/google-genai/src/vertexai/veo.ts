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
  ModelReference,
  modelActionMetadata,
  modelRef,
  z,
} from 'genkit';
import { BackgroundModelAction, ModelInfo } from 'genkit/model';
import { backgroundModel as pluginBackgroundModel } from 'genkit/plugin';
import { veoCheckOperation, veoPredict } from './client.js';
import {
  fromVeoOperation,
  toVeoClientOptions,
  toVeoModel,
  toVeoOperationRequest,
  toVeoPredictRequest,
} from './converters.js';
import { ClientOptions, Model, VertexPluginOptions } from './types.js';
import {
  calculateRequestOptions,
  checkModelName,
  extractVersion,
} from './utils.js';

export const VeoConfigSchema = z
  .object({
    sampleCount: z.number().optional().describe('Number of output videos'),
    storageUri: z
      .string()
      .optional()
      .describe('The gcs bucket where to save the generated videos'),
    fps: z
      .number()
      .optional()
      .describe('Frames per second for video generation'),
    durationSeconds: z
      .number()
      .optional()
      .describe('Duration of the clip for video generation in seconds'),
    seed: z
      .number()
      .optional()
      .describe(
        'The RNG seed. If RNG seed is exactly same for each request with unchanged ' +
          'inputs, the prediction results will be consistent. Otherwise, a random RNG ' +
          'seed will be used each time to produce a different result. If the sample ' +
          'count is greater than 1, random seeds will be used for each sample.'
      ),
    aspectRatio: z
      .enum(['9:16', '16:9'])
      .optional()
      .describe('The aspect ratio for the generated video'),
    resolution: z
      .enum(['720p', '1080p'])
      .optional()
      .describe('The resolution for the generated video'),
    personGeneration: z
      .enum(['dont_allow', 'allow_adult', 'allow_all'])
      .optional()
      .describe(
        'Specifies the policy for generating persons in videos, including age restrictions'
      ),
    pubsubTopic: z
      .string()
      .optional()
      .describe('The pubsub topic to publish the video generation progress to'),
    negativePrompt: z
      .string()
      .optional()
      .describe(
        'In addition to the text context, negative prompts can be explicitly stated here to help generate the video'
      ),
    enhancePrompt: z
      .boolean()
      .optional()
      .describe(
        'If true, the prompt will be improved before it is used to generate videos. ' +
          'The RNG seed, if provided, will not result in consistent results if prompts are enhanced.'
      ),
    generateAudio: z
      .boolean()
      .optional()
      .describe('If true, audio will be generated along with the video'),
    compressionQuality: z
      .enum(['optimized', 'lossless'])
      .default('optimized')
      .optional()
      .describe('Compression quality of the generated video'),
    resizeMode: z
      .enum(['pad', 'crop'])
      .default('pad')
      .optional()
      .describe(
        'Veo 3 only. The resize mode that the model uses to resize the video'
      ),
    location: z
      .string()
      .describe('Google Cloud region e.g. us-central1. or global')
      .optional(),
  })
  .passthrough();
export type VeoConfigSchemaType = typeof VeoConfigSchema;
export type VeoConfig = z.infer<VeoConfigSchemaType>;

// This contains all the Veo config schema types
type ConfigSchemaType = VeoConfigSchemaType;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = VeoConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertexai/${name}`,
    configSchema,
    info:
      info ??
      ({
        supports: {
          media: true,
          multiturn: false,
          tools: false,
          systemRole: false,
          output: ['media'],
          longRunning: true,
        },
      } as ModelInfo), // TODO(ifielker): Remove this cast if we fix longRunning
  });
}

const GENERIC_MODEL = commonRef('veo');

const KNOWN_MODELS = {
  'veo-2.0-generate-001': commonRef('veo-2.0-generate-001'),
  'veo-3.0-generate-001': commonRef('veo-3.0-generate-001'),
  'veo-3.0-fast-generate-001': commonRef('veo-3.0-fast-generate-001'),
  'veo-3.1-fast-generate-001': commonRef('veo-3.1-fast-generate-001'),
  'veo-3.1-generate-001': commonRef('veo-3.1-generate-001'),
} as const;
export type KnownModels = keyof typeof KNOWN_MODELS; // For autocomplete
export type VeoModelName = `veo-${string}`;
export function isVeoModelName(value?: string): value is VeoModelName {
  return !!value?.startsWith('veo-');
}

export function model(
  version: string,
  config: VeoConfig = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);
  return modelRef({
    name: `vertexai/${name}`,
    config,
    configSchema: VeoConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

// Takes a full list of models, filters for current Veo models only
// and returns a modelActionMetadata for each.
export function listActions(models: Model[]): ActionMetadata[] {
  return models
    .filter((m: Model) => isVeoModelName(m.name))
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
): BackgroundModelAction<VeoConfigSchemaType> {
  const ref = model(name);

  return pluginBackgroundModel({
    name: ref.name,
    ...ref.info,
    configSchema: ref.configSchema,
    async start(request) {
      const clientOpt = calculateRequestOptions(clientOptions, request.config);
      const veoPredictRequest = toVeoPredictRequest(request);

      const response = await veoPredict(
        extractVersion(ref),
        veoPredictRequest,
        clientOpt
      );

      return fromVeoOperation(response);
    },
    async check(operation) {
      const response = await veoCheckOperation(
        toVeoModel(operation),
        toVeoOperationRequest(operation),
        toVeoClientOptions(operation, clientOptions)
      );
      return fromVeoOperation(response);
    },
  });
}

export const TEST_ONLY = {
  GENERIC_MODEL,
  KNOWN_MODELS,
};
