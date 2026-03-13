/**
 * Copyright 2024 The Fire Company
 * Copyright 2024 Google LLC
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
import type {
  GenerateRequest,
  GenerateResponseData,
  ModelReference,
} from 'genkit';
import { GenerationCommonConfigSchema, Message, modelRef, z } from 'genkit';
import type { ModelAction, ModelInfo } from 'genkit/model';
import { model } from 'genkit/plugin';
import OpenAI from 'openai';
import { Response } from 'openai/core.mjs';
import type {
  SpeechCreateParams,
  Transcription,
  TranscriptionCreateParams,
} from 'openai/resources/audio/index.mjs';
import { PluginOptions } from './index.js';
import { maybeCreateRequestScopedOpenAIClient, toModelName } from './utils.js';

export type SpeechRequestBuilder = (
  req: GenerateRequest,
  params: SpeechCreateParams
) => void;
export type TranscriptionRequestBuilder = (
  req: GenerateRequest,
  params: TranscriptionCreateParams
) => void;

export const TRANSCRIPTION_MODEL_INFO = {
  supports: {
    media: true,
    output: ['text', 'json'],
    multiturn: false,
    systemRole: false,
    tools: false,
  },
};

export const SPEECH_MODEL_INFO: ModelInfo = {
  supports: {
    media: false,
    output: ['media'],
    multiturn: false,
    systemRole: false,
    tools: false,
  },
};

const ChunkingStrategySchema = z.object({
  type: z.string(),
  prefix_padding_ms: z.number().int().optional(),
  silence_duration_ms: z.number().int().optional(),
  threshold: z.number().min(0).max(1.0).optional(),
});
export const TranscriptionConfigSchema = GenerationCommonConfigSchema.pick({
  temperature: true,
}).extend({
  chunking_strategy: z
    .union([z.literal('auto'), ChunkingStrategySchema])
    .optional(),
  include: z.array(z.any()).optional(),
  language: z.string().optional(),
  timestamp_granularities: z.array(z.enum(['word', 'segment'])).optional(),
  response_format: z
    .enum(['json', 'text', 'srt', 'verbose_json', 'vtt'])
    .optional(),
  // TODO stream support
});

export const SpeechConfigSchema = z.object({
  voice: z
    .enum(['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'])
    .default('alloy'),
  speed: z.number().min(0.25).max(4.0).optional(),
  response_format: z
    .enum(['mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'])
    .optional(),
});

/**
 * Supported media formats for Audio generation
 */
export const RESPONSE_FORMAT_MEDIA_TYPES = {
  mp3: 'audio/mpeg',
  opus: 'audio/opus',
  aac: 'audio/aac',
  flac: 'audio/flac',
  wav: 'audio/wav',
  pcm: 'audio/L16',
};

export function toTTSRequest(
  modelName: string,
  request: GenerateRequest,
  requestBuilder?: SpeechRequestBuilder
): SpeechCreateParams {
  const {
    voice,
    version: modelVersion,
    temperature,
    maxOutputTokens,
    stopSequences,
    topK,
    topP,
    ...restOfConfig
  } = request.config ?? {};

  let options: SpeechCreateParams = {
    model: modelVersion ?? modelName,
    input: new Message(request.messages[0]).text,
    voice: voice ?? 'alloy',
  };
  if (requestBuilder) {
    requestBuilder(request, options);
  } else {
    options = {
      ...options,
      ...restOfConfig, // passthorugh rest of the config
    };
  }
  for (const k in options) {
    if (options[k] === undefined) {
      delete options[k];
    }
  }
  return options;
}

export async function speechToGenerateResponse(
  response: Response,
  responseFormat: 'mp3' | 'opus' | 'aac' | 'flac' | 'wav' | 'pcm' = 'mp3'
): Promise<GenerateResponseData> {
  const resultArrayBuffer = await response.arrayBuffer();
  const resultBuffer = Buffer.from(new Uint8Array(resultArrayBuffer));
  const mediaType = RESPONSE_FORMAT_MEDIA_TYPES[responseFormat];
  return {
    message: {
      role: 'model',
      content: [
        {
          media: {
            contentType: mediaType,
            url: `data:${mediaType};base64,${resultBuffer.toString('base64')}`,
          },
        },
      ],
    },
    finishReason: 'stop',
    raw: response,
  };
}

/**
 * Method to define a new Genkit Model that is compatible with the Open AI Audio
 * API. 
 *
 * These models are to be used to create audio speech from a given request.
 * @param params An object containing parameters for defining the OpenAI speech
 * model.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.

 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAISpeechModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  client: OpenAI;
  modelRef?: ModelReference<CustomOptions>;
  requestBuilder?: SpeechRequestBuilder;
  pluginOptions: PluginOptions;
}): ModelAction {
  const {
    name,
    client: defaultClient,
    pluginOptions,
    modelRef,
    requestBuilder,
  } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName = `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    async (request, { abortSignal }) => {
      const ttsRequest = toTTSRequest(modelName, request, requestBuilder);
      const client = maybeCreateRequestScopedOpenAIClient(
        pluginOptions,
        request,
        defaultClient
      );
      const result = await client.audio.speech.create(ttsRequest, {
        signal: abortSignal,
      });
      return await speechToGenerateResponse(result, ttsRequest.response_format);
    }
  );
}

/** Speech generation ModelRef helper, with reasonable defaults for
 * OpenAI-compatible providers */
export function compatOaiSpeechModelRef<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  info?: ModelInfo;
  configSchema?: CustomOptions;
  config?: any;
  namespace?: string;
}) {
  const {
    name,
    info = SPEECH_MODEL_INFO,
    configSchema,
    config = undefined,
    namespace,
  } = params;
  return modelRef({
    name,
    configSchema: configSchema || (SpeechConfigSchema as any),
    info,
    config,
    namespace,
  });
}

export function toSttRequest(
  modelName: string,
  request: GenerateRequest,
  requestBuilder?: TranscriptionRequestBuilder
): TranscriptionCreateParams {
  const message = new Message(request.messages[0]);
  const media = message.media;
  if (!media?.url) {
    throw new Error('No media found in the request');
  }
  const mediaBuffer = Buffer.from(
    media.url.slice(media.url.indexOf(',') + 1),
    'base64'
  );
  const mediaFile = new File([mediaBuffer], 'input', {
    type:
      media.contentType ??
      media.url.slice('data:'.length, media.url.indexOf(';')),
  });
  const {
    temperature,
    version: modelVersion,
    maxOutputTokens,
    stopSequences,
    topK,
    topP,
    ...restOfConfig
  } = request.config ?? {};

  let options: TranscriptionCreateParams = {
    model: modelVersion ?? modelName,
    file: mediaFile,
    prompt: message.text,
    temperature,
  };
  if (requestBuilder) {
    requestBuilder(request, options);
  } else {
    options = {
      ...options,
      ...restOfConfig, // passthrough rest of the config
    };
  }
  const outputFormat = request.output?.format as 'json' | 'text' | 'media';
  const customFormat = request.config?.response_format;
  if (outputFormat && customFormat) {
    if (
      outputFormat === 'json' &&
      customFormat !== 'json' &&
      customFormat !== 'verbose_json'
    ) {
      throw new Error(
        `Custom response format ${customFormat} is not compatible with output format ${outputFormat}`
      );
    }
  }
  if (outputFormat === 'media') {
    throw new Error(`Output format ${outputFormat} is not supported.`);
  }
  options.response_format = customFormat || outputFormat || 'text';
  for (const k in options) {
    if (options[k] === undefined) {
      delete options[k];
    }
  }
  return options;
}

export function transcriptionToGenerateResponse(
  result: Transcription | string
): GenerateResponseData {
  return {
    message: {
      role: 'model',
      content: [
        {
          text: typeof result === 'string' ? result : result.text,
        },
      ],
    },
    finishReason: 'stop',
    raw: result,
  };
}

/**
 * Method to define a new Genkit Model that is compatible with Open AI
 * Transcriptions API. 
 *
 * These models are to be used to transcribe audio to text.
 *
 * @param params An object containing parameters for defining the OpenAI
 * transcription model.
 * @param params.ai The Genkit AI instance.
 * @param params.name The name of the model.
 * @param params.client The OpenAI client instance.
 * @param params.modelRef Optional reference to the model's configuration and
 * custom options.

 * @returns the created {@link ModelAction}
 */
export function defineCompatOpenAITranscriptionModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  client: OpenAI;
  pluginOptions?: PluginOptions;
  modelRef?: ModelReference<CustomOptions>;
  requestBuilder?: TranscriptionRequestBuilder;
}): ModelAction {
  const {
    name,
    pluginOptions,
    client: defaultClient,
    modelRef,
    requestBuilder,
  } = params;
  const modelName = toModelName(name, pluginOptions?.name);
  const actionName =
    modelRef?.name ?? `${pluginOptions?.name ?? 'compat-oai'}/${modelName}`;

  return model(
    {
      name: actionName,
      ...modelRef?.info,
      configSchema: modelRef?.configSchema,
    },
    async (request, { abortSignal }) => {
      const params = toSttRequest(modelName, request, requestBuilder);
      const client = maybeCreateRequestScopedOpenAIClient(
        pluginOptions,
        request,
        defaultClient
      );
      // Explicitly setting stream to false ensures we use the non-streaming overload
      const result = await client.audio.transcriptions.create(
        {
          ...params,
          stream: false,
        },
        { signal: abortSignal }
      );
      return transcriptionToGenerateResponse(result);
    }
  );
}

/** Transcription ModelRef helper, with reasonable defaults for
 * OpenAI-compatible providers */
export function compatOaiTranscriptionModelRef<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  info?: ModelInfo;
  configSchema?: CustomOptions;
  config?: any;
  namespace?: string;
}) {
  const {
    name,
    info = TRANSCRIPTION_MODEL_INFO,
    configSchema,
    config = undefined,
    namespace,
  } = params;
  return modelRef({
    name,
    configSchema: configSchema || (TranscriptionConfigSchema as any),
    info,
    config,
    namespace,
  });
}
