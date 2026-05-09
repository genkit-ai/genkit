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
  action,
  defineAction,
  z,
  type Action,
  type ActionFnArg,
  type ActionMetadata,
  type ActionParams,
} from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import { toJsonSchema } from '@genkit-ai/core/schema';
import { Document, DocumentDataSchema, type DocumentData } from './document.js';

/**
 * A batch (array) of embeddings.
 */
export type EmbeddingBatch = { embedding: number[] }[];

/**
 * EmbeddingSchema includes the embedding and also metadata so you know
 * which of multiple embeddings corresponds to which part of a document.
 */
export const EmbeddingSchema = z.object({
  embedding: z.array(z.number()),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type Embedding = z.infer<typeof EmbeddingSchema>;

/**
 * A function used for embedder definition, encapsulates embedder implementation.
 */
export type EmbedderFn<EmbedderOptions extends z.ZodTypeAny> = (
  input: Document[],
  embedderOpts?: z.infer<EmbedderOptions>
) => Promise<EmbedResponse>;

/**
 * Zod schema of an embed request.
 */
const EmbedRequestSchema = z.object({
  input: z.array(DocumentDataSchema),
  options: z.any().optional(),
});

export interface EmbedRequest<O = any> {
  input: Document[];
  options?: O;
}

/**
 * Zod schema of an embed response.
 */
const EmbedResponseSchema = z.object({
  embeddings: z.array(EmbeddingSchema),
  // TODO: stats, etc.
});
type EmbedResponse = z.infer<typeof EmbedResponseSchema>;

/**
 * Embedder action -- a subtype of {@link Action} with input/output types for embedders.
 */
export type EmbedderAction<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> =
  Action<typeof EmbedRequestSchema, typeof EmbedResponseSchema> & {
    __configSchema?: CustomOptions;
  };

/**
 * Options of an `embed` function.
 */
export interface EmbedderParams<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  embedder: EmbedderArgument<CustomOptions>;
  content: string | DocumentData;
  metadata?: Record<string, unknown>;
  options?: z.infer<CustomOptions>;
}

function withMetadata<CustomOptions extends z.ZodTypeAny>(
  embedder: Action<typeof EmbedRequestSchema, typeof EmbedResponseSchema>,
  configSchema?: CustomOptions
): EmbedderAction<CustomOptions> {
  const withMeta = embedder as EmbedderAction<CustomOptions>;
  withMeta.__configSchema = configSchema;
  return withMeta;
}

export interface EmbedderOptions<ConfigSchema extends z.ZodTypeAny> {
  name: string;
  configSchema?: ConfigSchema;
  info?: EmbedderInfo;
}

/**
 * Creates embedder model for the provided {@link EmbedderFn} model implementation.
 *
 * Unlike `defineEmbedder` this function does not register the embedder in the reigistry.
 */
export function embedder<ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny>(
  options: EmbedderOptions<ConfigSchema>,
  runner: (
    input: EmbedRequest<z.infer<ConfigSchema>>,
    opts: ActionFnArg<any>
  ) => Promise<EmbedResponse>
) {
  const embedder = action(embedderActionOptions(options), (i, opts) =>
    runner(
      {
        input: i.input.map((dd) => new Document(dd)),
        options: i.options,
      },
      opts
    )
  );
  const ewm = withMetadata(
    embedder as Action<typeof EmbedRequestSchema, typeof EmbedResponseSchema>,
    options.configSchema
  );
  return ewm;
}

function embedderActionOptions<
  ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: EmbedderOptions<ConfigSchema>
): ActionParams<typeof EmbedRequestSchema, typeof EmbedResponseSchema> {
  return {
    actionType: 'embedder',
    name: options.name,
    inputSchema: EmbedRequestSchema,
    outputSchema: EmbedResponseSchema,
    metadata: {
      type: 'embedder',
      info: options.info,
      embedder: {
        customOptions: options.configSchema
          ? toJsonSchema({ schema: options.configSchema })
          : undefined,
      },
    },
  };
}

/**
 * Creates embedder model for the provided {@link EmbedderFn} model implementation.
 */
export function defineEmbedder<
  ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  options: EmbedderOptions<ConfigSchema>,
  runner: EmbedderFn<ConfigSchema>
) {
  const embedder = defineAction(registry, embedderActionOptions(options), (i) =>
    runner(
      i.input.map((dd) => new Document(dd)),
      i.options
    )
  );
  const ewm = withMetadata(
    embedder as Action<typeof EmbedRequestSchema, typeof EmbedResponseSchema>,
    options.configSchema
  );
  return ewm;
}

/**
 * A union type representing all the types that can refer to an embedder.
 */
export type EmbedderArgument<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> = string | EmbedderAction<CustomOptions> | EmbedderReference<CustomOptions>;

/**
 * A veneer for interacting with embedder models.
 */
export async function embed<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny>(
  registry: Registry,
  params: EmbedderParams<CustomOptions>
): Promise<Embedding[]> {
  const embedder = await resolveEmbedder(registry, params);
  if (!embedder.embedderAction) {
    let embedderId: string;
    if (typeof params.embedder === 'string') {
      embedderId = params.embedder;
    } else if ((params.embedder as EmbedderAction)?.__action?.name) {
      embedderId = (params.embedder as EmbedderAction).__action.name;
    } else {
      embedderId = (params.embedder as EmbedderReference<any>).name;
    }
    throw new Error(`Unable to resolve embedder ${embedderId}`);
  }
  const response = await embedder.embedderAction({
    input:
      typeof params.content === 'string'
        ? [Document.fromText(params.content, params.metadata)]
        : [params.content],
    options: {
      version: embedder.version,
      ...embedder.config,
      ...params.options,
    },
  });
  return response.embeddings;
}

interface ResolvedEmbedder<CustomOptions extends z.ZodTypeAny = z.ZodTypeAny> {
  embedderAction: EmbedderAction<CustomOptions>;
  config?: z.infer<CustomOptions>;
  version?: string;
}

async function resolveEmbedder<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  params: EmbedderParams<CustomOptions>
): Promise<ResolvedEmbedder<CustomOptions>> {
  if (typeof params.embedder === 'string') {
    return {
      embedderAction: await registry.lookupAction(
        `/embedder/${params.embedder}`
      ),
    };
  } else if (Object.hasOwnProperty.call(params.embedder, '__action')) {
    return {
      embedderAction: params.embedder as EmbedderAction<CustomOptions>,
    };
  } else if (Object.hasOwnProperty.call(params.embedder, 'name')) {
    const ref = params.embedder as EmbedderReference<any>;
    return {
      embedderAction: await registry.lookupAction(
        `/embedder/${(params.embedder as EmbedderReference).name}`
      ),
      config: {
        ...ref.config,
      },
      version: ref.version,
    };
  }
  throw new Error(`failed to resolve embedder ${params.embedder}`);
}

/**
 * A veneer for interacting with embedder models in bulk.
 */
export async function embedMany<
  ConfigSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  registry: Registry,
  params: {
    embedder: EmbedderArgument<ConfigSchema>;
    content: string[] | DocumentData[];
    metadata?: Record<string, unknown>;
    options?: z.infer<ConfigSchema>;
  }
): Promise<EmbeddingBatch> {
  let embedder: EmbedderAction<ConfigSchema>;
  if (typeof params.embedder === 'string') {
    embedder = await registry.lookupAction(`/embedder/${params.embedder}`);
  } else if (Object.hasOwnProperty.call(params.embedder, 'info')) {
    embedder = await registry.lookupAction(
      `/embedder/${(params.embedder as EmbedderReference).name}`
    );
  } else {
    embedder = params.embedder as EmbedderAction<ConfigSchema>;
  }
  if (!embedder) {
    throw new Error('Unable to utilize the provided embedder');
  }
  const response = await embedder({
    input: params.content.map((i) =>
      typeof i === 'string' ? Document.fromText(i, params.metadata) : i
    ),
    options: params.options,
  });
  return response.embeddings;
}

/**
 * Zod schema of embedder info object.
 */
export const EmbedderInfoSchema = z.object({
  /** Friendly label for this model (e.g. "Google AI - Gemini Pro") */
  label: z.string().optional(),
  /** Supported model capabilities. */
  supports: z
    .object({
      /** Model can input this type of data. */
      input: z.array(z.enum(['text', 'image', 'video'])).optional(),
      /** Model can support multiple languages */
      multilingual: z.boolean().optional(),
    })
    .optional(),
  /** Embedding dimension */
  dimensions: z.number().optional(),
});
export type EmbedderInfo = z.infer<typeof EmbedderInfoSchema>;

/**
 * A reference object that can used to resolve an embedder instance. Include additional type information
 * about the specific embedder, e.g. custom config options schema.
 */
export interface EmbedderReference<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
> {
  name: string;
  configSchema?: CustomOptions;
  info?: EmbedderInfo;
  config?: z.infer<CustomOptions>;
  version?: string;
}

/**
 * Helper method to configure a {@link EmbedderReference} to a plugin.
 */
export function embedderRef<
  CustomOptionsSchema extends z.ZodTypeAny = z.ZodTypeAny,
>(
  options: EmbedderReference<CustomOptionsSchema> & {
    namespace?: string;
  }
): EmbedderReference<CustomOptionsSchema> {
  let name = options.name;
  if (options.namespace && !name.startsWith(options.namespace + '/')) {
    name = `${options.namespace}/${name}`;
  }
  return { ...options, name };
}

/**
 * Packages embedder information into ActionMetadata object.
 */
export function embedderActionMetadata({
  name,
  info,
  configSchema,
}: {
  name: string;
  info?: EmbedderInfo;
  configSchema?: z.ZodTypeAny;
}): ActionMetadata {
  return {
    actionType: 'embedder',
    name: name,
    inputJsonSchema: toJsonSchema({ schema: EmbedRequestSchema }),
    outputJsonSchema: toJsonSchema({ schema: EmbedResponseSchema }),
    metadata: {
      embedder: {
        ...info,
        customOptions: configSchema
          ? toJsonSchema({ schema: configSchema })
          : undefined,
      },
    },
  } as ActionMetadata;
}
