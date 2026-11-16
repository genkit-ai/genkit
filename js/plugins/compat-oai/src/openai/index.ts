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
import { lookup } from '../utils.js';
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
  gptModelRef,
  OpenAIChatCompletionConfigSchema,
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

// gpt-6-cyber supports Chat Completions, so only gpt-5.6-cyber is matched.
const UNSUPPORTED_MODEL_MATCHERS: RegExp[] = [
  /moderation/,
  /realtime/,
  /^gpt-live/,
  /(^|-)audio(-|$)/,
  /-pro(-|$)/,
  /codex/,
  /^gpt-5\.6-cyber/,
  /rosalind/,
  /^gpt-daybreak-/,
  /^((text|code)-)?(ada|babbage|curie|davinci)(-\d{3}$|:)/,
  /^computer-use/,
  /deep-research/,
  /^sora/,
];

/** Model an `ft:` id was tuned from, so its suffix can't change its kind. */
function baseId(id: string): string {
  return id.startsWith('ft:') ? id.split(':')[1] : id;
}

/** Image ids this plugin serves through its image model. */
function isImageModelId(id: string): boolean {
  return id.includes('gpt-image-1') || id.includes('dall-e');
}

function createResolver(pluginOptions: PluginOptions) {
  return async (client: OpenAI, actionType: ActionType, actionName: string) => {
    const base = baseId(actionName);
    if (actionType === 'embedder') {
      return defineCompatOpenAIEmbedder({
        name: actionName,
        client,
        pluginOptions,
      });
    } else if (isImageModelId(base)) {
      const modelRef = openAIImageModelRef({ name: actionName });
      return defineCompatOpenAIImageModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    } else if (base.includes('tts')) {
      const modelRef = openAISpeechModelRef({ name: actionName });
      return defineCompatOpenAISpeechModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    } else if (base.includes('whisper')) {
      const modelRef = openAIWhisperModelRef({ name: actionName });
      return defineOpenAIWhisperModel({
        name: modelRef.name,
        client,
        pluginOptions,
        modelRef,
      });
    } else if (base.includes('transcribe')) {
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
      const modelRef = gptModelRef(actionName);
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
  const base = baseId(model.id);
  if (UNSUPPORTED_MODEL_MATCHERS.some((m) => m.test(base))) {
    return false;
  }
  // Other image ids (gpt-image-2*, chatgpt-image-latest) aren't served yet and
  // would otherwise be listed as chat models.
  return isImageModelId(base) || !base.includes('image');
}

const listActions = async (client: OpenAI): Promise<ActionMetadata[]> => {
  return await client.models.list().then((response) =>
    response.data.filter(filterOpenAiModels).map((model: OpenAI.Model) => {
      const base = baseId(model.id);
      if (base.includes('embedding')) {
        return embedderActionMetadata({
          name: model.id,
          configSchema: TextEmbeddingConfigSchema,
          info: lookup(SUPPORTED_EMBEDDING_MODELS, model.id)?.info,
        });
      } else if (isImageModelId(base)) {
        const modelRef =
          lookup(SUPPORTED_IMAGE_MODELS, model.id) ??
          openAIImageModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else if (base.includes('tts')) {
        const modelRef =
          lookup(SUPPORTED_TTS_MODELS, model.id) ??
          openAISpeechModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else if (base.includes('whisper')) {
        const modelRef =
          lookup(SUPPORTED_WHISPER_MODELS, model.id) ??
          openAIWhisperModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else if (base.includes('transcribe')) {
        const modelRef =
          lookup(SUPPORTED_STT_MODELS, model.id) ??
          openAITranscriptionModelRef({ name: model.id });
        return modelActionMetadata({
          name: modelRef.name,
          info: modelRef.info,
          configSchema: modelRef.configSchema,
        });
      } else {
        const modelRef = gptModelRef(model.id);
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
  const base = baseId(name);
  if (isImageModelId(base)) {
    return openAIImageModelRef({
      name,
      config,
    });
  }
  if (base.includes('tts')) {
    return openAISpeechModelRef({
      name,
      config,
    });
  }
  if (base.includes('whisper')) {
    return openAIWhisperModelRef({
      name,
      config,
    });
  }
  if (base.includes('transcribe')) {
    return openAITranscriptionModelRef({
      name,
      config,
    });
  }
  return gptModelRef(name, config);
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
