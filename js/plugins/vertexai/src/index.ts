/**
 * @license
 *
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

/**
 * @module /
 */

import {
  embedderRef,
  modelActionMetadata,
  modelRef,
  type EmbedderReference,
  type Genkit,
  type ModelReference,
  type z,
} from 'genkit';
import { genkitPlugin, type GenkitPlugin } from 'genkit/plugin';
import type { ActionType } from 'genkit/registry';
import { getDerivedParams } from './common/index.js';
import type { PluginOptions } from './common/types.js';
import {
  SUPPORTED_EMBEDDER_MODELS,
  VertexEmbeddingConfigSchema,
  defineVertexAIEmbedder,
  geminiEmbedding001,
  multimodalEmbedding001,
  textEmbedding004,
  textEmbedding005,
  textEmbeddingGecko003,
  textEmbeddingGeckoMultilingual001,
  textMultilingualEmbedding002,
  type VertexEmbeddingConfig,
} from './embedder.js';
import {
  GeminiConfigSchema,
  SUPPORTED_GEMINI_MODELS,
  SafetySettingsSchema,
  defineGeminiKnownModel,
  defineGeminiModel,
  gemini,
  gemini10Pro,
  gemini15Flash,
  gemini15Pro,
  gemini20Flash,
  gemini20Flash001,
  gemini20FlashLite,
  gemini20FlashLitePreview0205,
  gemini20ProExp0205,
  gemini25FlashLite,
  gemini25FlashPreview0417,
  gemini25ProExp0325,
  gemini25ProPreview0325,
  type GeminiConfig,
  type GeminiVersionString,
} from './gemini.js';
import {
  GENERIC_IMAGEN_INFO,
  ImagenConfigSchema,
  SUPPORTED_IMAGEN_MODELS,
  defineImagenModel,
  imagen2,
  imagen3,
  imagen3Fast,
  type ACTUAL_IMAGEN_MODELS,
} from './imagen.js';
import { listModels } from './list-models.js';
export type { PluginOptions } from './common/types.js';
export {
  GeminiConfigSchema,
  ImagenConfigSchema,
  SafetySettingsSchema,
  gemini,
  gemini10Pro,
  gemini15Flash,
  gemini15Pro,
  gemini20Flash,
  gemini20Flash001,
  gemini20FlashLite,
  gemini20FlashLitePreview0205,
  gemini20ProExp0205,
  gemini25FlashLite,
  gemini25FlashPreview0417,
  gemini25ProExp0325,
  gemini25ProPreview0325,
  geminiEmbedding001,
  imagen2,
  imagen3,
  imagen3Fast,
  multimodalEmbedding001,
  textEmbedding004,
  textEmbedding005,
  textEmbeddingGecko003,
  textEmbeddingGeckoMultilingual001,
  textMultilingualEmbedding002,
  type GeminiConfig,
  type GeminiVersionString,
};

function warnDeprecated() {
  console.error(
    `\n\x1b[41m\x1b[37m\x1b[1m  DEPRECATION WARNING  \x1b[0m\n` +
      `\x1b[31mThe \x1b[1m@genkit-ai/vertexai\x1b[0m\x1b[31m plugin is deprecated and will be REMOVED in a future release.\x1b[0m\n` +
      `\x1b[33m👉 Please migrate to \x1b[1m@genkit-ai/google-genai\x1b[0m\x1b[33m to avoid breaking changes.\x1b[0m\n` +
      `\x1b[90mDocs: https://genkit.dev/docs/js/integrations/vertex-ai/ \x1b[0m\n`
  );
}

async function initializer(ai: Genkit, options?: PluginOptions) {
  warnDeprecated();
  if (options?.location === 'global') {
    // Not a breaking change, it already doesn't work.
    throw new Error(
      'The vertexAI plugin in the @genkit-ai/vertexai package does not support "global" location.\n' +
        'Support for "global" location is available in the new vertexAI plugin in the @genkit-ai/google-genai package.\n\n' +
        'Please switch your import to use the latest features:\n' +
        "  import { vertexAI } from '@genkit-ai/google-genai';\n\n"
    );
  }
  const { projectId, location, vertexClientFactory, authClient } =
    await getDerivedParams(options);

  Object.keys(SUPPORTED_IMAGEN_MODELS).map((name) =>
    defineImagenModel(ai, name, authClient, { projectId, location })
  );
  Object.keys(SUPPORTED_GEMINI_MODELS).map((name) =>
    defineGeminiKnownModel(
      ai,
      name,
      vertexClientFactory,
      {
        projectId,
        location,
      },
      options?.experimental_debugTraces
    )
  );
  if (options?.models) {
    for (const modelOrRef of options?.models) {
      const modelName =
        typeof modelOrRef === 'string'
          ? modelOrRef
          : // strip out the `vertexai/` prefix
            modelOrRef.name.split('/')[1];
      const modelRef =
        typeof modelOrRef === 'string' ? gemini(modelOrRef) : modelOrRef;
      defineGeminiModel({
        ai,
        modelName: modelRef.name,
        version: modelName,
        modelInfo: modelRef.info,
        vertexClientFactory,
        options: {
          projectId,
          location,
        },
        debugTraces: options.experimental_debugTraces,
      });
    }
  }

  Object.keys(SUPPORTED_EMBEDDER_MODELS).map((name) =>
    defineVertexAIEmbedder(ai, name, authClient, { projectId, location })
  );
}

async function resolver(
  ai: Genkit,
  actionType: ActionType,
  actionName: string,
  options?: PluginOptions
) {
  warnDeprecated();
  switch (actionType) {
    case 'model':
      await resolveModel(ai, actionName, options);
      break;
    case 'embedder':
      await resolveEmbedder(ai, actionName, options);
      break;
    default:
    // no-op
  }
}

async function resolveModel(
  ai: Genkit,
  actionName: string,
  options?: PluginOptions
) {
  const { projectId, location, vertexClientFactory, authClient } =
    await getDerivedParams(options);

  if (actionName.startsWith('imagen')) {
    defineImagenModel(ai, actionName, authClient, { projectId, location });
    return;
  }

  const modelRef = gemini(actionName);
  defineGeminiModel({
    ai,
    modelName: modelRef.name,
    version: actionName,
    modelInfo: modelRef.info,
    vertexClientFactory,
    options: {
      projectId,
      location,
    },
    debugTraces: options?.experimental_debugTraces,
  });
}

async function resolveEmbedder(
  ai: Genkit,
  actionName: string,
  options?: PluginOptions
) {
  const { projectId, location, authClient } = await getDerivedParams(options);

  defineVertexAIEmbedder(ai, actionName, authClient, { projectId, location });
}

// Vertex AI list models still returns these and the API does not indicate in any way
// that those models are not served anymore.
const KNOWN_DECOMISSIONED_MODELS = [
  'gemini-pro-vision',
  'gemini-pro',
  'gemini-ultra',
  'gemini-ultra-vision',
];

async function listActions(options?: PluginOptions) {
  const { location, projectId, authClient } = await getDerivedParams(options);
  const models = await listModels(authClient, location, projectId);
  // Vertex has a lot of models, and no way to figure out the "type" of the model...
  // so, for list actions we only fetch known model "families".
  return [
    // Gemini
    ...models
      .filter(
        (m) =>
          m.name.includes('gemini') &&
          !KNOWN_DECOMISSIONED_MODELS.includes(m.name.split('/').at(-1)!)
      )
      .map((m) => {
        const ref = gemini(m.name.split('/').at(-1)!);

        return modelActionMetadata({
          name: ref.name,
          info: ref.info,
          configSchema: GeminiConfigSchema,
        });
      }),
    // Imagen
    ...models
      .filter((m) => m.name.includes('imagen'))
      .map((m) => {
        const name = m.name.split('/').at(-1)!;

        return modelActionMetadata({
          name: 'vertexai/' + name,
          info: {
            ...GENERIC_IMAGEN_INFO,
            label: `Vertex AI - ${name}`,
          },
          configSchema: ImagenConfigSchema,
        });
      }),
  ];
}

/**
 * Add Google Cloud Vertex AI to Genkit. Includes Gemini and Imagen models and text embedder.
 */
function vertexAIPlugin(options?: PluginOptions): GenkitPlugin {
  let listActionsCache;
  return genkitPlugin(
    'vertexai',
    async (ai: Genkit) => await initializer(ai, options),
    async (ai: Genkit, actionType: ActionType, actionName: string) =>
      await resolver(ai, actionType, actionName, options),
    async () => {
      if (listActionsCache) return listActionsCache;
      listActionsCache = await listActions(options);
      return listActionsCache;
    }
  );
}

/**
 * @deprecated
 */
export type VertexAIPlugin = {
  (params?: PluginOptions): GenkitPlugin;
  model(
    name: keyof typeof SUPPORTED_GEMINI_MODELS | (`gemini-${string}` & {}),
    config?: z.infer<typeof GeminiConfigSchema>
  ): ModelReference<typeof GeminiConfigSchema>;
  model(
    name: keyof typeof ACTUAL_IMAGEN_MODELS | (`imagen${string}` & {}),
    config?: z.infer<typeof ImagenConfigSchema>
  ): ModelReference<typeof ImagenConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
  embedder(
    name: string,
    config?: VertexEmbeddingConfig
  ): EmbedderReference<typeof VertexEmbeddingConfigSchema>;
};

/**
 * Google Cloud Vertex AI plugin for Genkit.
 * Includes Gemini and Imagen models and text embedder.
 * @deprecated Please use `import { vertexAI } from '@genkit-ai/google-genai';` instead. Replace model constants with e.g. vertexAI.model('gemini-2.5-pro')
 */
export const vertexAI = vertexAIPlugin as VertexAIPlugin;
// provide generic implementation for the model function overloads.
(vertexAI as any).model = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  warnDeprecated();
  if (name.startsWith('imagen')) {
    return modelRef({
      name: `vertexai/${name}`,
      config,
      configSchema: ImagenConfigSchema,
    });
  }
  return modelRef({
    name: `vertexai/${name}`,
    config,
    configSchema: GeminiConfigSchema,
  });
};
vertexAI.embedder = (
  name: string,
  config?: VertexEmbeddingConfig
): EmbedderReference<typeof VertexEmbeddingConfigSchema> => {
  warnDeprecated();
  return embedderRef({
    name: `vertexai/${name}`,
    config,
    configSchema: VertexEmbeddingConfigSchema,
  });
};

export default vertexAI;
