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

import {
  ActionMetadata,
  embedderActionMetadata,
  embedderRef,
  EmbedderReference,
  modelActionMetadata,
  ModelReference,
  z,
} from 'genkit';
import { type GenkitPluginV2, type ResolvableAction } from 'genkit/plugin';
import { type ActionType } from 'genkit/registry';
import OpenAI from 'openai';
import {
  defineCompatOpenAISpeechModel,
  defineCompatOpenAITranscriptionModel,
  SpeechConfigSchema,
  TranscriptionConfigSchema,
} from '../audio.js';
import { defineCompatOpenAIEmbedder } from '../embedder.js';
import {
  defineCompatOpenAIImageModel,
  ImageGenerationCommonConfigSchema,
} from '../image.js';
import { openAICompatible, PluginOptions } from '../index.js';
import { defineCompatOpenAIModel } from '../model.js';
import {
  gptImage1RequestBuilder,
  openAIImageModelRef,
  SUPPORTED_IMAGE_MODELS,
} from './dalle.js';
import {
  SUPPORTED_EMBEDDING_MODELS,
  TextEmbeddingConfigSchema,
} from './embedder.js';
import {
  OpenAIChatCompletionConfigSchema,
  openAIModelRef,
  SUPPORTED_GPT_MODELS,
} from './gpt.js';
import { openAITranscriptionModelRef, SUPPORTED_STT_MODELS } from './stt.js';
import { openAISpeechModelRef, SUPPORTED_TTS_MODELS } from './tts.js';
import {
  defineOpenAIWhisperModel,
  openAIWhisperModelRef,
  SUPPORTED_WHISPER_MODELS,
  WhisperConfigSchema,
} from './whisper.js';

export type OpenAIPluginOptions = Omit<PluginOptions, 'name' | 'baseURL'>;

const UNSUPPORTED_MODEL_MATCHERS = ['babbage', 'davinci', 'codex'];

function createResolver(pluginOptions: PluginOptions) {
  return async (client: OpenAI, actionType: ActionType, actionName: string) => {
    if (actionType === 'embedder') {
      return defineCompatOpenAIEmbedder({
        name: actionName,
        client,
        pluginOptions,
      });
    } else if (
      actionName.includes('gpt-image-1') ||
      actionName.includes('dall-e')
    ) {
      const modelRef = openAIImageModelRef({ name: actionName });
      return defineCompatOpenAIImageModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    } else if (actionName.includes('tts')) {
      const modelRef = openAISpeechModelRef({ name: actionName });
      return defineCompatOpenAISpeechModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    } else if (actionName.includes('whisper')) {
      const modelRef = openAIWhisperModelRef({ name: actionName });
      return defineOpenAIWhisperModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    } else if (actionName.includes('transcribe')) {
      const modelRef = openAITranscriptionModelRef({
        name: actionName,
      });
      return defineCompatOpenAITranscriptionModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    } else {
      const modelRef = openAIModelRef({ name: actionName });
      return defineCompatOpenAIModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    }
  };
}

function filterOpenAiModels(model: OpenAI.Model): boolean {
  return !UNSUPPORTED_MODEL_MATCHERS.some((m) => model.id.includes(m));
}

const listActions = async (client: OpenAI): Promise<ActionMetadata[]> => {
  return await client.models.list().then((response) =>
    response.data.filter(filterOpenAiModels).map((model: OpenAI.Model) => {
      if (model.id.includes('embedding')) {
        return embedderActionMetadata({
          name: model.id,
          configSchema: TextEmbeddingConfigSchema,
          info: SUPPORTED_EMBEDDING_MODELS[model.id]?.info,
        });
      } else if (
        model.id.includes('gpt-image-1') ||
        model.id.includes('dall-e')
      ) {
        const modelRef =
          SUPPORTED_IMAGE_MODELS[model.id] ??
          openAIImageModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else if (model.id.includes('tts')) {
        const modelRef =
          SUPPORTED_TTS_MODELS[model.id] ??
          openAISpeechModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else if (model.id.includes('whisper')) {
        const modelRef =
          SUPPORTED_WHISPER_MODELS[model.id] ??
          openAIWhisperModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else if (model.id.includes('transcribe')) {
        const modelRef =
          SUPPORTED_STT_MODELS[model.id] ??
          openAITranscriptionModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else {
        const modelRef =
          SUPPORTED_GPT_MODELS[model.id] ?? openAIModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      }
    })
  );
};

export function openAIPlugin(options?: OpenAIPluginOptions): GenkitPluginV2 {
  const pluginOptions = { name: 'openai', ...options };
  return openAICompatible({
    name: 'openai',
    ...options,
    initializer: async (client) => {
      const models = [] as ResolvableAction[];
      models.push(
        ...Object.values(SUPPORTED_GPT_MODELS).map((modelRef) =>
          defineCompatOpenAIModel({
            name: modelRef.name,
            client,
            pluginOptions,
            modelRef,
          })
        )
      );
      models.push(
        ...Object.values(SUPPORTED_EMBEDDING_MODELS).map((embedderRef) =>
          defineCompatOpenAIEmbedder({
            name: embedderRef.name,
            client,
            pluginOptions,
            embedderRef,
          })
        )
      );
      models.push(
        ...Object.values(SUPPORTED_TTS_MODELS).map((modelRef) =>
          defineCompatOpenAISpeechModel({
            name: modelRef.name,
            client,
            pluginOptions,
            modelRef,
          })
        )
      );
      models.push(
        ...Object.values(SUPPORTED_WHISPER_MODELS).map((modelRef) =>
          defineOpenAIWhisperModel({
            name: modelRef.name,
            client,
            pluginOptions,
            modelRef,
          })
        )
      );
      models.push(
        ...Object.values(SUPPORTED_STT_MODELS).map((modelRef) =>
          defineCompatOpenAITranscriptionModel({
            name: modelRef.name,
            client,
            pluginOptions,
            modelRef,
          })
        )
      );
      models.push(
        ...Object.values(SUPPORTED_IMAGE_MODELS).map((modelRef) =>
          defineCompatOpenAIImageModel({
            name: modelRef.name,
            client,
            pluginOptions,
            modelRef,
            requestBuilder: modelRef.name.includes('gpt-image-1')
              ? gptImage1RequestBuilder
              : undefined,
          })
        )
      );
      return models;
    },
    resolver: createResolver(pluginOptions),
    listActions,
  });
}

export type OpenAIPlugin = {
  (params?: OpenAIPluginOptions): GenkitPluginV2;
  model(
    name:
      | keyof typeof SUPPORTED_IMAGE_MODELS
      | (`dall-e${string}` & {})
      | (`gpt-image-${string}` & {}),
    config?: z.infer<typeof ImageGenerationCommonConfigSchema>
  ): ModelReference<typeof ImageGenerationCommonConfigSchema>;
  model(
    name:
      | keyof typeof SUPPORTED_TTS_MODELS
      | (`tts-${string}` & {})
      | (`${string}-tts` & {}),
    config?: z.infer<typeof SpeechConfigSchema>
  ): ModelReference<typeof SpeechConfigSchema>;
  model(
    name: keyof typeof SUPPORTED_WHISPER_MODELS | (`whisper-${string}` & {}),
    config?: z.infer<typeof WhisperConfigSchema>
  ): ModelReference<typeof WhisperConfigSchema>;
  model(
    name: keyof typeof SUPPORTED_STT_MODELS | (`${string}-transcribe` & {}),
    config?: z.infer<typeof TranscriptionConfigSchema>
  ): ModelReference<typeof TranscriptionConfigSchema>;
  model(
    name:
      | keyof typeof SUPPORTED_GPT_MODELS
      | (`gpt-${string}` & {})
      | (`o${number}` & {}),
    config?: z.infer<typeof OpenAIChatCompletionConfigSchema>
  ): ModelReference<typeof OpenAIChatCompletionConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
  embedder(
    name:
      | keyof typeof SUPPORTED_EMBEDDING_MODELS
      | (`${string}-embedding-${string}` & {}),
    config?: z.infer<typeof TextEmbeddingConfigSchema>
  ): EmbedderReference<typeof TextEmbeddingConfigSchema>;
  embedder(name: string, config?: any): EmbedderReference<z.ZodTypeAny>;
};

const model = ((name: string, config?: any): ModelReference<z.ZodTypeAny> => {
  if (name.includes('gpt-image-1') || name.includes('dall-e')) {
    return openAIImageModelRef({
      name,
      config,
    });
  }
  if (name.includes('tts')) {
    return openAISpeechModelRef({
      name,
      config,
    });
  }
  if (name.includes('whisper')) {
    return openAIWhisperModelRef({
      name,
      config,
    });
  }
  if (name.includes('transcribe')) {
    return openAITranscriptionModelRef({
      name,
      config,
    });
  }
  return openAIModelRef({
    name,
    config,
  });
}) as OpenAIPlugin['model'];

const embedder = ((
  name: string,
  config?: any
): EmbedderReference<z.ZodTypeAny> => {
  return embedderRef({
    name,
    config,
    configSchema: TextEmbeddingConfigSchema,
    namespace: 'openai',
  });
}) as OpenAIPlugin['embedder'];

/**
 * This module provides an interface to the OpenAI models through the Genkit
 * plugin system. It allows users to interact with various models by providing
 * an API key and optional configuration.
 *
 * The main export is the `openai` plugin, which can be configured with an API
 * key either directly or through environment variables. It initializes the
 * OpenAI client and makes available the models for use.
 *
 * Exports:
 * - openai: The main plugin function to interact with OpenAI.
 *
 * Usage:
 * To use the models, initialize the openai plugin inside `configureGenkit` and
 * pass the configuration options. If no API key is provided in the options, the
 * environment variable `OPENAI_API_KEY` must be set.
 *
 * Example:
 * ```
 * import { openAI } from '@genkit-ai/compat-oai/openai';
 *
 * export default configureGenkit({
 *  plugins: [
 *    openai()
 *    ... // other plugins
 *  ]
 * });
 * ```
 */
export const openAI: OpenAIPlugin = Object.assign(openAIPlugin, {
  model,
  embedder,
});

export default openAI;
