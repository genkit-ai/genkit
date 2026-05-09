/**
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
  ActionFnArg,
  BackgroundAction,
  GenkitError,
  MiddlewareWithOptions,
  Operation,
  OperationSchema,
  action,
  backgroundAction,
  defineAction,
  registerBackgroundAction,
  z,
  type Action,
  type ActionMetadata,
  type ActionParams,
  type SimpleMiddleware,
  type StreamingCallback,
} from '@genkit-ai/core';
import { logger } from '@genkit-ai/core/logging';
import type { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import { performance } from 'node:perf_hooks';
import {
  CandidateData,
  GenerateRequest,
  GenerateRequestSchema,
  GenerateResponseChunkData,
  GenerateResponseChunkSchema,
  GenerateResponseData,
  GenerateResponseSchema,
  GenerationUsage,
  MessageData,
  ModelInfo,
  Part,
} from './model-types.js';
import {
  augmentWithContext,
  simulateConstrainedGeneration,
} from './model/middleware.js';
import {
  CustomPartSchema,
  DataPartSchema,
  MediaPartSchema,
  MultipartToolResponseSchema,
  ReasoningPartSchema,
  ResourcePartSchema,
  TextPartSchema,
  ToolRequestPartSchema,
  ToolResponsePartSchema,
  type CustomPart,
  type DataPart,
  type MediaPart,
  type MultipartToolResponse,
  type ReasoningPart,
  type ResourcePart,
  type TextPart,
  type ToolRequestPart,
  type ToolResponsePart,
} from './parts.js';
export { defineGenerateAction } from './generate/action.js';
export * from './model-types.js';
export {
  CustomPartSchema,
  DataPartSchema,
  MediaPartSchema,
  MultipartToolResponseSchema,
  ReasoningPartSchema,
  ResourcePartSchema,
  TextPartSchema,
  ToolRequestPartSchema,
  ToolResponsePartSchema,
  simulateConstrainedGeneration,
  type CustomPart,
  type DataPart,
  type MediaPart,
  type MultipartToolResponse,
  type ReasoningPart,
  type ResourcePart,
  type TextPart,
  type ToolRequestPart,
  type ToolResponsePart,
};

export type ModelAction<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
> = Action<
  typeof GenerateRequestSchema,
  typeof GenerateResponseSchema,
  typeof GenerateResponseChunkSchema
> & {
  __configSchema: CustomOptionsSchema;
};

export type BackgroundModelAction<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
> = BackgroundAction<
  typeof GenerateRequestSchema,
  typeof GenerateResponseSchema
> & {
  __configSchema: CustomOptionsSchema;
};

export type ModelMiddleware = SimpleMiddleware<
  z.infer<typeof GenerateRequestSchema>,
  z.infer<typeof GenerateResponseSchema>
>;

export type ModelMiddlewareWithOptions = MiddlewareWithOptions<
  z.infer<typeof GenerateRequestSchema>,
  z.infer<typeof GenerateResponseSchema>,
  z.infer<typeof GenerateResponseChunkSchema>
>;

export type ModelMiddlewareArgument =
  | ModelMiddleware
  | ModelMiddlewareWithOptions;

export type DefineModelOptions<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
> = {
  name: string;
  /** Known version names for this model, e.g. `gemini-1.0-pro-001`. */
  versions?: string[];
  /** Capabilities this model supports. */
  supports?: ModelInfo['supports'];
  /** Custom options schema for this model. */
  configSchema?: CustomOptionsSchema;
  /** Descriptive name for this model e.g. 'Google AI - Gemini Pro'. */
  label?: string;
  /** Middleware to be used with this model. */
  use?: ModelMiddlewareArgument[];
};

export function model<CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny>(
  options: DefineModelOptions<CustomOptionsSchema>,
  runner: (
    request: GenerateRequest<CustomOptionsSchema>,
    options: ActionFnArg<GenerateResponseChunkData>
  ) => Promise<GenerateResponseData>
): ModelAction<CustomOptionsSchema> {
  const act = action(modelActionOptions(options), (input, ctx) => {
    const startTimeMs = performance.now();
    return runner(input, ctx).then((response) => {
      const timedResponse = {
        ...response,
        latencyMs: performance.now() - startTimeMs,
      };
      return timedResponse;
    });
  });
  Object.assign(act, {
    __configSchema: options.configSchema || z.unknown(),
  });
  return act as ModelAction<CustomOptionsSchema>;
}

function modelActionOptions<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: DefineModelOptions<CustomOptionsSchema>
): ActionParams<typeof GenerateRequestSchema, typeof GenerateResponseSchema> {
  const label = options.label || options.name;
  const middleware = getModelMiddleware(options);
  return {
    actionType: 'model',
    name: options.name,
    description: label,
    inputSchema: GenerateRequestSchema,
    outputSchema: GenerateResponseSchema,
    metadata: {
      model: {
        label,
        customOptions: options.configSchema
          ? toJsonSchema({ schema: options.configSchema })
          : undefined,
        versions: options.versions,
        supports: options.supports,
      },
    },
    use: middleware,
  };
}

/**
 * Defines a new model and adds it to the registry.
 */
export function defineModel<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: {
    apiVersion: 'v2';
  } & DefineModelOptions<CustomOptionsSchema>,
  runner: (
    request: GenerateRequest<CustomOptionsSchema>,
    options: ActionFnArg<GenerateResponseChunkData>
  ) => Promise<GenerateResponseData>
): ModelAction<CustomOptionsSchema>;

/**
 * Defines a new model and adds it to the registry.
 */
export function defineModel<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: DefineModelOptions<CustomOptionsSchema>,
  runner: (
    request: GenerateRequest<CustomOptionsSchema>,
    streamingCallback?: StreamingCallback<GenerateResponseChunkData>
  ) => Promise<GenerateResponseData>
): ModelAction<CustomOptionsSchema>;

export function defineModel<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: any,
  runner: (
    request: GenerateRequest<CustomOptionsSchema>,
    options: any
  ) => Promise<GenerateResponseData>
): ModelAction<CustomOptionsSchema> {
  const act = defineAction(
    registry,
    modelActionOptions(options),
    (input, ctx) => {
      const startTimeMs = performance.now();
      const secondParam =
        options.apiVersion === 'v2'
          ? ctx
          : ctx.streamingRequested
            ? ctx.sendChunk
            : undefined;
      return runner(input, secondParam).then((response) => {
        const timedResponse = {
          ...response,
          latencyMs: performance.now() - startTimeMs,
        };
        return timedResponse;
      });
    }
  );
  Object.assign(act, {
    __configSchema: options.configSchema || z.unknown(),
  });
  return act as ModelAction<CustomOptionsSchema>;
}

export type DefineBackgroundModelOptions<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
> = DefineModelOptions<CustomOptionsSchema> & {
  start: (
    request: GenerateRequest<CustomOptionsSchema>
  ) => Promise<Operation<GenerateResponseData>>;
  check: (
    operation: Operation<GenerateResponseData>
  ) => Promise<Operation<GenerateResponseData>>;
  cancel?: (
    operation: Operation<GenerateResponseData>
  ) => Promise<Operation<GenerateResponseData>>;
};

/**
 * Defines a new model that runs in the background.
 */
export function defineBackgroundModel<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: DefineBackgroundModelOptions<CustomOptionsSchema>
): BackgroundModelAction<CustomOptionsSchema> {
  const act = backgroundModel(options);
  registerBackgroundAction(registry, act);
  return act;
}
/**
 * Defines a new model that runs in the background.
 */
export function backgroundModel<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: DefineBackgroundModelOptions<CustomOptionsSchema>
): BackgroundModelAction<CustomOptionsSchema> {
  const label = options.label || options.name;
  const middleware = getModelMiddleware(options);
  const act = backgroundAction({
    actionType: 'background-model',
    name: options.name,
    description: label,
    inputSchema: GenerateRequestSchema,
    outputSchema: GenerateResponseSchema,
    metadata: {
      model: {
        label,
        customOptions: options.configSchema
          ? toJsonSchema({ schema: options.configSchema })
          : undefined,
        versions: options.versions,
        supports: options.supports,
      },
    },
    use: middleware,
    async start(request) {
      const startTimeMs = performance.now();
      const response = await options.start(request);
      Object.assign(response, {
        latencyMs: performance.now() - startTimeMs,
      });
      return response;
    },
    async check(op) {
      return options.check(op);
    },
    cancel: options.cancel
      ? async (op) => {
          if (!options.cancel) {
            throw new GenkitError({
              status: 'UNIMPLEMENTED',
              message: 'cancel not implemented',
            });
          }
          return options.cancel(op);
        }
      : undefined,
  }) as BackgroundModelAction<CustomOptionsSchema>;
  Object.assign(act, {
    __configSchema: options.configSchema || z.unknown(),
  });
  return act;
}

function getModelMiddleware(options: {
  use?: ModelMiddlewareArgument[];
  name: string;
  supports?: ModelInfo['supports'];
}) {
  const middleware: ModelMiddlewareArgument[] = options.use || [];
  if (!options?.supports?.context) middleware.push(augmentWithContext());
  const constratedSimulator = simulateConstrainedGeneration();
  middleware.push((req, next) => {
    if (
      !options?.supports?.constrained ||
      options?.supports?.constrained === 'none' ||
      (options?.supports?.constrained === 'no-tools' &&
        (req.tools?.length ?? 0) > 0)
    ) {
      return constratedSimulator(req, next);
    }
    return next(req);
  });

  return middleware;
}

export const ModelReferenceSchema = z.object({
  name: z.string(),
  config: z.any().optional(),
});

export interface ModelReference<CustomOptions extends z.ZodTypeAny>
  extends z.infer<typeof ModelReferenceSchema> {
  name: string;
  configSchema?: CustomOptions;
  info?: ModelInfo;
  version?: string;
  config?: z.infer<CustomOptions>;

  withConfig(cfg: z.infer<CustomOptions>): ModelReference<CustomOptions>;
  withVersion(version: string): ModelReference<CustomOptions>;
}

/**
 * Packages model information into ActionMetadata object.
 */
export function modelActionMetadata({
  name,
  info,
  configSchema,
  background,
}: {
  name: string;
  info?: ModelInfo;
  configSchema?: z.ZodTypeAny;
  background?: boolean;
}): ActionMetadata {
  return {
    actionType: background ? 'background-model' : 'model',
    name: name,
    inputJsonSchema: toJsonSchema({ schema: GenerateRequestSchema }),
    outputJsonSchema: background
      ? toJsonSchema({ schema: OperationSchema })
      : toJsonSchema({ schema: GenerateResponseSchema }),
    metadata: {
      model: {
        ...info,
        customOptions: configSchema
          ? toJsonSchema({ schema: configSchema })
          : undefined,
      },
    },
  } as ActionMetadata;
}

/** Cretes a model reference. */
export function modelRef<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: Omit<
    ModelReference<CustomOptionsSchema>,
    'withConfig' | 'withVersion'
  > & {
    namespace?: string;
  }
): ModelReference<CustomOptionsSchema> {
  let name = options.name;
  if (options.namespace && !name.startsWith(options.namespace + '/')) {
    name = `${options.namespace}/${name}`;
  }
  const ref: Partial<ModelReference<CustomOptionsSchema>> = {
    ...options,
    name,
  };
  ref.withConfig = (
    cfg: z.infer<CustomOptionsSchema>
  ): ModelReference<CustomOptionsSchema> => {
    return modelRef({
      ...options,
      name,
      config: cfg,
    });
  };
  ref.withVersion = (version: string): ModelReference<CustomOptionsSchema> => {
    return modelRef({
      ...options,
      name,
      version,
    });
  };
  return ref as ModelReference<CustomOptionsSchema>;
}

/** Container for counting usage stats for a single input/output {Part} */
type PartCounts = {
  characters: number;
  images: number;
  videos: number;
  audio: number;
};

/**
 * Calculates basic usage statistics from the given model inputs and outputs.
 */
export function getBasicUsageStats(
  input: MessageData[],
  response: MessageData | CandidateData[]
): GenerationUsage {
  const inputCounts = getPartCounts(input.flatMap((md) => md.content));
  const outputCounts = getPartCounts(
    Array.isArray(response)
      ? response.flatMap((c) => c.message.content)
      : response.content
  );
  return {
    inputCharacters: inputCounts.characters,
    inputImages: inputCounts.images,
    inputVideos: inputCounts.videos,
    inputAudioFiles: inputCounts.audio,
    outputCharacters: outputCounts.characters,
    outputImages: outputCounts.images,
    outputVideos: outputCounts.videos,
    outputAudioFiles: outputCounts.audio,
  };
}

function getPartCounts(parts: Part[]): PartCounts {
  return parts.reduce(
    (counts, part) => {
      const isImage =
        part.media?.contentType?.startsWith('image') ||
        part.media?.url?.startsWith('data:image');
      const isVideo =
        part.media?.contentType?.startsWith('video') ||
        part.media?.url?.startsWith('data:video');
      const isAudio =
        part.media?.contentType?.startsWith('audio') ||
        part.media?.url?.startsWith('data:audio');
      return {
        characters: counts.characters + (part.text?.length || 0),
        images: counts.images + (isImage ? 1 : 0),
        videos: counts.videos + (isVideo ? 1 : 0),
        audio: counts.audio + (isAudio ? 1 : 0),
      };
    },
    { characters: 0, images: 0, videos: 0, audio: 0 }
  );
}

export type ModelArgument<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> =
  | ModelAction<CustomOptions>
  | ModelReference<CustomOptions>
  | string;

export interface ResolvedModel<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  modelAction: ModelAction;
  config?: z.infer<CustomOptions>;
  version?: string;
}

export async function resolveModel<C extends z.ZodTypeAny = z.ZodTypeAny>(
  registry: Registry,
  model: ModelArgument<C> | undefined,
  options?: { warnDeprecated?: boolean }
): Promise<ResolvedModel<C>> {
  let out: ResolvedModel<C>;
  let modelId: string;

  if (!model) {
    model = await registry.lookupValue('defaultModel', 'defaultModel');
  }
  if (!model) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: 'Must supply a `model` to `generate()` calls.',
    });
  }
  if (typeof model === 'string') {
    modelId = model;
    out = { modelAction: await lookupModel(registry, model) };
  } else if (model.hasOwnProperty('__action')) {
    modelId = (model as ModelAction).__action.name;
    out = { modelAction: model as ModelAction };
  } else {
    const ref = model as ModelReference<any>;
    modelId = ref.name;
    out = {
      modelAction: await lookupModel(registry, ref.name),
      config: {
        ...ref.config,
      },
      version: ref.version,
    };
  }

  if (!out.modelAction) {
    throw new GenkitError({
      status: 'NOT_FOUND',
      message: `Model '${modelId}' not found`,
    });
  }

  if (
    options?.warnDeprecated &&
    out.modelAction.__action.metadata?.model?.stage === 'deprecated'
  ) {
    logger.warn(
      `Model '${out.modelAction.__action.name}' is deprecated and may be removed in a future release.`
    );
  }

  return out;
}

async function lookupModel(
  registry: Registry,
  model: string
): Promise<ModelAction> {
  return (
    (await registry.lookupAction(`/model/${model}`)) ||
    (await registry.lookupAction(`/background-model/${model}`))
  );
}
